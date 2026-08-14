# backend/api/v1/interview.py
# ═══════════════════════════════════════════════════════
# 模拟面试 Agent 的 API 层（FastAPI 路由）
# 和考试 API 的区别：
#   考试：提交后后台异步跑，学员轮询结果
#   面试：实时对话，每轮同步等 LLM 回复
#
# 4个接口：
#   POST /sessions              创建面试（拿开场白）
#   POST /sessions/{id}/chat    学员发消息（拿回复）
#   GET  /sessions/{id}/report  查面试报告
#   GET  /sessions              查面试列表
#   POST /sessions/{id}/chat/stream  SSE流式（语音用）
# ═══════════════════════════════════════════════════════

import json
import uuid

from fastapi import (
    APIRouter, Depends, HTTPException)
# HumanMessage：学员说的话
# AIMessage：面试官说的话
from langchain_core.messages import (
    HumanMessage, AIMessage)
from pydantic import BaseModel
# EventSourceResponse：SSE 流式响应
# 用于 chat/stream 接口，边生成边推 token
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import text

from backend.agents.interview.graph import (
    build_interview_graph)
from backend.agents.interview.state import (
    InterviewStage)
from backend.core.logger import get_logger
# build_thread_id：拼存档编号
# build_config：构建图运行配置（含 thread_id）
from backend.core.memory import (
    build_thread_id, build_config)
from backend.dependencies import (
    AsyncSessionLocal, get_current_user)

router = APIRouter()
logger = get_logger(__name__)
# 模块级单例：导入时编译一次图，所有请求共用
# 和考试 API 一样，避免每次请求重新编译
_graph = build_interview_graph()

# ── 请求体模型 ──

class StartSessionRequest(BaseModel):
    """创建面试会话的请求体"""
    # 目标岗位（必填，如"Python后端"）
    # 影响出题方向和难度
    target_position: str
    # 简历审查记录 ID（可选）
    # 有值 → 从简历提取项目/技能做联动
    # 无值 → 面试时让学员自述项目
    resume_review_id: str | None = None

class ChatRequest(BaseModel):
    """学员聊天消息的请求体"""
    # 学员这一轮发的消息
    message: str


# 辅助函数：提取消息列表中最后一条面试官（AI）消息的文本
def _get_last_ai_message(messages: list) -> str:
    """提取消息列表中最后一条面试官（AI）消息的文本。"""
    for msg in reversed(messages): # 从后往前找
        if isinstance(msg, AIMessage): # 找到最后一条 AI 消息
            content = msg.content
            if isinstance(content, list):
                # 拼接各片段文本
                return "".join(
                    p.get("text", "")
                    if isinstance(p, dict) else str(p)
                    for p in content
                )
            return str(content) # 普通字符串直接返回
    # 没有 AI 消息时的兜底文案
    return "面试已开始，请等待面试官回应..."


# status_code=201：HTTP 状态码「已创建」
# 表示成功创建了一个新资源（面试会话）
# 区别于 200（OK）和 202（已接受，异步处理）
# 这里用 201 因为开场白是同步等出来的
@router.post("/sessions", status_code=201)
async def start_session(
        # 请求体：岗位 + 可选简历 ID
        req: StartSessionRequest,
        # 依赖注入：自动从请求头解析当前登录用户
        current_user: dict = Depends(get_current_user),
    ):
    """创建新面试会话，触发首轮开场白。
    同步等待 LLM 返回开场白后一起响应。

    请求：{target_position, resume_review_id?}
    响应：201 +
        {session_id, target_position,
         status, message(开场白)}
    """
    # 从登录信息取学员 ID
    student_id = current_user["user_id"]
    # 从登录信息取租户 ID（多租户隔离）
    tenant_id  = current_user["tenant_id"]
    # 生成全局唯一的会话 ID（每次不同）
    session_id = str(uuid.uuid4())
    # 存档编号：MemorySaver 靠它找图的存档
    # 格式：student_{student_id}_session_{session_id}
    thread_id  = build_thread_id(
        student_id, session_id)

    # ── 第1步：写入初始会话记录 ──
    # 往 interview_sessions 表插一条记录
    # status='in_progress' 表示面试进行中
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO interview_sessions
                (id, tenant_id, student_id,
                session_id, thread_id,
                target_position,
                resume_review_id, status)
                VALUES
                (:id, :tenant_id, :student_id,
                :session_id, :thread_id,
                :target_position,
                :resume_review_id, 'in_progress')
            """),
            {
                # 主键 ID（和 session_id 不同）
                "id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "student_id": student_id,
                # 会话 ID（前端后续请求带这个）
                "session_id": session_id,
                # 存档编号（图恢复时靠它找存档）
                "thread_id": thread_id,
                # 目标岗位（如"Python后端"）
                "target_position": req.target_position,
                # 简历审查记录 ID（可选）
                # 有值 → 从简历提取项目/技能
                # 无值 → 面试时让学员自述项目
                "resume_review_id": req.resume_review_id,
            },
        )
        # 提交事务，数据才真正写入数据库
        await session.commit()

    # ── 第2步：构造图的初始 State ──
    # 首轮 MemorySaver 里还没存档
    # 必须传全量 22 个字段（后续轮次只传增量）
    initial_state = {
        # 触发图执行的特殊消息
        # load_context 看到 [开始面试] 就知道
        # 这是首轮，走初始化流程
        "messages": [
            HumanMessage(content="[开始面试]")],
        "student_id": student_id,
        "tenant_id": tenant_id,
        "session_id": session_id,
        # 目标岗位（影响出题方向和难度）
        "target_position": req.target_position,
        # 简历审查记录 ID（load_context 用它
        # 去查简历项目和技术栈）
        "resume_review_id": req.resume_review_id,
        # 简历项目列表（load_context 填充）
        "resume_projects": [],
        # 简历技能列表（load_context 填充）
        "resume_skills": [],
        # 当前阶段：从热身开始
        "current_stage": InterviewStage.WARMUP.value,
        # 本阶段轮数（0=还没开始）
        "stage_turn_count": 0,
        # 总轮数（0 → load_context 走首轮）
        "total_turn_count": 0,
        # 最大轮数上限（到了就强制结束）
        "max_turns": 40,
        # 题库（load_context 从 DB 加载）
        "question_bank": [],
        # 当前题目（None=还没出题）
        "current_question": None,
        # 已深挖的项目名列表
        "projects_asked": [],
        # 上轮回答质量（初始给 adequate）
        "last_answer_quality": "adequate",
        # 当前题追问次数（0=没追问过）
        "followup_count": 0,
        # 历史对话摘要（首轮没有）
        "existing_summary": None,
        # 是否触发记忆压缩（首轮不需要）
        "should_summarize": False,
        # 面试报告（结束时生成）
        "report": None,
        # 降级标记（正常流程=False）
        "fallback_used": False,
        # 结构化输出（结束时填充）
        "structured_output": None,
    }

    # ── 第3步：首轮图执行 ──
    # 和考试不同：面试首轮是同步等待的
    # 因为前端需要立刻拿到开场白显示给学员
    # （考试是后台异步跑，面试是实时对话）

    # 构建运行配置（内含 thread_id）
    config = build_config(student_id, session_id)
    # 跑一遍图：load_context → check_stage
    # → evaluate_answer → generate_response
    # → save_memory
    # 首轮 evaluate_answer 会跳过评估
    # generate_response 走 warmup 分支
    # 生成面试官开场白
    result = await _graph.ainvoke(initial_state, config=config)
    # 从最终 State 取所有消息
    messages = result.get("messages", [])
    # 提取最后一条 AI 消息（面试官开场白）
    opening_msg = _get_last_ai_message(messages)

    # 记录日志
    logger.info("interview.session_started",
                session_id=session_id,
                target_position=req.target_position)

    # 返回给前端：会话信息 + 开场白
    # 前端拿到 session_id 后续聊天都带它
    # 拿到 message 直接显示面试官的第一句话
    return {
        "session_id": session_id,
        "target_position": req.target_position, # 目标岗位
        "status": "in_progress", # 面试进行中
        "message": opening_msg, # AI面试官开场白
    }



# ───────────────────────────────────────────────────────
# POST /sessions/{session_id}/chat  学员发消息
# ───────────────────────────────────────────────────────
# 面试的核心交互接口：学员说一句，面试官回一句
#
# 和 start_session 的区别：
#   start_session：首轮，传全量 22 字段
#   chat：后续轮，只传增量（新消息）
#   完整 State 由 MemorySaver 按 thread_id 自动恢复
#
# 流程：
#   1. 验证会话归属
#   2. 只传新消息给图
#   3. 图跑完返回面试官回复 + 当前状态
#   4. 如果面试结束，附带报告摘要
# ───────────────────────────────────────────────────────
@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    学员发送消息，面试官生成回应。
    MemorySaver 通过 thread_id 自动恢复
    历史 State，只需传入新消息。

    请求：{message}
    响应：{session_id, reply, current_stage,
           total_turns, is_finished,
           report_summary?}
    """
    # 从登录信息取学员和租户 ID
    student_id = current_user["user_id"]
    tenant_id = current_user["tenant_id"]

    # ── 第1步：验证会话归属 ──
    # 三个条件同时匹配：
    # session_id + tenant_id + student_id
    # 防止学员A访问学员B的面试
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT session_id
                FROM interview_sessions
                WHERE session_id = :session_id
                  AND tenant_id = :tenant_id
                  AND student_id = :student_id
            """),
            {"session_id": session_id,
             "tenant_id": tenant_id,
             "student_id": student_id},
        )
        # 查不到 → 会话不存在或不属于该学员
        if not result.fetchone():
            raise HTTPException(
                status_code=404,
                detail="面试会话不存在")

    # 构建运行配置（内含 thread_id）
    config = build_config(student_id, session_id)

    # ── 第2步：只传增量 ──
    # 完整 State 由 MemorySaver 按 thread_id
    # 自动恢复，这里只需要传学员的新消息
    # add_messages 注解会把新消息追加到历史里
    state_update = {
        # 学员这轮说的话（追加到 messages）
        "messages": [
            HumanMessage(content=req.message)],
        # 显式带上 ID 字段
        # 防 MemorySaver 丢失时图崩掉
        "student_id": student_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
    }

    # ── 第3步：跑一遍图 ──
    # 图会从 MemorySaver 恢复上次的 State
    # 然后跑：load_context → check_stage
    # → evaluate_answer → generate_response
    # → save_memory
    result = await _graph.ainvoke(
        state_update, config=config)

    # ── 第4步：从最终 State 提取返回数据 ──
    # 当前阶段（warmup/tech_base/project/
    # closing/finished）
    current_stage = result.get(
        "current_stage",
        InterviewStage.WARMUP.value)
    # 总轮数
    total_turns = result.get("total_turn_count", 0)
    # 是否已结束（阶段 == finished）
    is_finished = (
        current_stage== InterviewStage.FINISHED.value)
    # 面试官这轮的回复
    reply = _get_last_ai_message(result.get("messages", []))

    response = {
        "session_id": session_id,
        "reply": reply, # 面试官这轮的回复
        "current_stage": current_stage, # 当前阶段
        "total_turns": total_turns, # 总轮数
        "is_finished": is_finished, # 是否结束
    }

    # 面试结束时附带报告摘要
    # 前端可以直接展示，不用再调 /report
    if is_finished:
        # 从最终 State 取面试报告
        report = result.get("report") or {}
        # 面试报告摘要
        response["report_summary"] = {
            # 综合得分
            "overall_score": report.get("overall_score", 0),
            # 核心优势
            "strengths": report.get("strengths", []),
            # 提升方向
            "improvements": report.get("improvements", []),
        }
    # 返回结果
    return response



# ───────────────────────────────────────────────────────
# GET /sessions/{session_id}/report  查面试报告
# ───────────────────────────────────────────────────────
# 面试结束后，学员查看完整的五维度评估报告
# 数据从数据库读（save_report 节点写入的）
#
# 和 chat 接口返回的 report_summary 的区别：
#   chat 的 report_summary：只有3个字段（简版）
#   这个接口：返回完整报告（7个字段）
# ───────────────────────────────────────────────────────
@router.get("/sessions/{session_id}/report")
async def get_report(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """学员查看面试完整报告"""
    # 从 interview_sessions 表查报告数据
    # WHERE 里匹配 student_id（防越权）
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT session_id,
                       target_position,
                       status,
                       overall_score,
                       report,
                       finished_at
                FROM interview_sessions
                WHERE session_id = :session_id
                  AND student_id = :student_id
            """),
            {"session_id": session_id,
             "student_id": current_user["user_id"]},
        )
        # .mappings()：把行变成字典
        row = result.mappings().fetchone()

    # 查不到 → 不存在或不属于该学员
    if not row:
        raise HTTPException(
            status_code=404,
            detail="面试记录不存在")

    # 还没结束 → 报告没生成，让学员等
    if row["status"] != "finished":
        raise HTTPException(
            status_code=400,
            detail="面试尚未结束，报告生成中")

    # report 列是 JSONB 类型
    # ORM 返回的类型不确定，三种情况都要处理
    raw_report = row["report"]
    if isinstance(raw_report, dict):
        # 情况1：ORM 已经帮你解析好了，直接用
        report = raw_report
    elif raw_report:
        # 情况2：是个 JSON 字符串，手动解析
        report = json.loads(raw_report)
    else:
        # 情况3：None 或空，给个空字典兜底
        report = {}

    # 摊平成前端需要的完整报告格式
    # 对应 InterviewReport 模型的 7 个字段
    return {
        "session_id": session_id,
        "target_position": row["target_position"], # 目标职位
        # 综合得分（五维度加权平均）
        "overall_score": row["overall_score"],
        # 五个维度的详细评估
        # 每个含 dimension/score/comment/
        # highlights/weaknesses
        "dimensions": report.get("dimensions", []),
        "strengths": report.get("strengths", []), # 2-3条核心优势
        # 2-3条提升方向
        "improvements": report.get("improvements", []),
        # 综合评语
        "overall_comment": report.get("overall_comment", ""),
        "recommended_topics": # 建议复习的知识点
            report.get("recommended_topics", []),
        # 下一步备考建议
        "next_step_advice": report.get("next_step_advice", ""),
    }


# ───────────────────────────────────────────────────────
# GET /sessions  查面试列表
# ───────────────────────────────────────────────────────
# 学员看自己所有的面试记录（列表页）
# 只返回摘要，不返回报告详情
# 按创建时间倒序，最多50条
# ───────────────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(
    current_user: dict = Depends(get_current_user),
):
    """学员查询自己的面试记录列表"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT session_id,
                       target_position,
                       overall_score,
                       status,
                       finished_at,
                       created_at
                FROM interview_sessions
                WHERE student_id = :student_id
                ORDER BY created_at DESC
                LIMIT 50
            """),
            {"student_id": current_user["user_id"]},
        )
        rows = result.mappings().all()

    # 组装列表项
    items = [
        {
            "session_id": row["session_id"],
            "target_position": row["target_position"],
            # 综合得分（未结束时为 None）
            "overall_score": row["overall_score"],
            "status": row["status"],
            # 结束时间（ISO 格式字符串）
            "finished_at":
                row["finished_at"].isoformat()
                if row["finished_at"] else None,
            # 创建时间
            "created_at":
                row["created_at"].isoformat()
                if row["created_at"] else None,
        }
        for row in rows
    ]

    return {
        "items": items, # 面试记录列表
        "total": len(items), # 总数
    }



# ───────────────────────────────────────────────────────
# POST /sessions/{id}/chat/stream  SSE 流式接口
# ───────────────────────────────────────────────────────
# 和 /chat 功能一样，但回复是「一个字一个字推」的
# 用于语音交互模式：学员说完话，面试官边想边说
#
# SSE（Server-Sent Events）：
#   服务器单向推流，客户端用 EventSource 接收
#   每条消息格式：data: {"type":"token","content":"你"}
#
# 三种事件类型：
#   token → 面试官回复的一个片段（逐字推）
#   done  → 流结束，带最终状态（阶段/轮数/报告）
#   error → 出错了
#
# 和 /chat 的区别：
#   /chat：等 LLM 全生成完，一次性返回
#   /stream：LLM 每生成一个 token 就推一个
# ───────────────────────────────────────────────────────
@router.post("/sessions/{session_id}/chat/stream")
async def chat_stream(
    session_id: str,
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    SSE 流式接口，适配语音交互模式。
    边生成边推 token。

    请求：{message}
    响应：text/event-stream
          data: {"type":"token","content":"..."}
          data: {"type":"done",...}
          data: {"type":"error",...}
    """
    student_id = current_user["user_id"]
    tenant_id  = current_user["tenant_id"]

    # 验证会话归属（同 /chat）
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT session_id "
                 "FROM interview_sessions "
                 "WHERE session_id = :sid "
                 "AND tenant_id = :tid"),
            {"sid": session_id,
             "tid": tenant_id},
        )
        if not result.fetchone():
            raise HTTPException(
                status_code=404,
                detail="面试会话不存在")

    # 构建运行配置（同 /chat）
    config = build_config(student_id, session_id)
    # 只传增量（同 /chat）
    state_update = {
        # 用户消息
        "messages": [HumanMessage(content=req.message)],
        "student_id": student_id,
        "session_id": session_id,
        "tenant_id":  tenant_id,
    }

    # ── SSE 异步生成器 ──
    # yield 一条 = 推一条 SSE 事件给前端
    async def event_generator():
        try:
            # astream_events：流式跑图
            # 和 ainvoke 的区别：
            #   ainvoke：跑完才返回
            #   astream_events：每产生一个token/事件就回调一次
            # version="v2"：事件格式版本
            async for event in _graph.astream_events(
                state_update,
                config=config,
                version="v2"
            ):
                # 事件类型（如 on_chat_model_stream）
                evt = event["event"]
                # 事件来自哪个节点
                node = event.get(
                    "metadata", {}).get("langgraph_node", "")

                # 只转发 generate_response 节点
                # 的 token 流（面试官回复）
                # 其他节点（check_stage 等）的
                # LLM 输出不暴露给前端
                if (evt == "on_chat_model_stream"
                    and node == "generate_response"):
                    # 取本次 token 片段
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        # 推一个 token 事件
                        yield {
                        "data":
                            json.dumps(
                                {
                                    "type": "token",
                                    "content":chunk.content
                                },
                                ensure_ascii=False, # 中文不转义
                            )
                        }

            # ── 流结束后：组装 done 事件 ──
            # 问题：astream_events 只推了 token
            # 前端拿到了面试官的回复（一个字一个字）
            # 但前端还不知道：现在哪个阶段？
            # 面试结束了没？总分多少？
            # 所以流跑完后，要再读一次最终 State
            # 把这些信息打包成 done 事件推给前端

            # aget_state：从 MemorySaver 读
            # 图跑完后的最终 State（检查点）
            final = await _graph.aget_state(config)
            # final.values 就是最终的 State 字典
            # final 为 None 时给空字典兜底
            state = final.values if final else {}
            # 从 State 取当前阶段
            current_stage = state.get(
                "current_stage",
                InterviewStage.WARMUP.value)
            # 从 State 取总轮数
            total_turns = state.get("total_turn_count", 0)
            # 判断面试是否结束
            is_finished = (
                current_stage== InterviewStage.FINISHED.value)
            # 完整回复文本
            # 前端已经通过 token 事件逐字收到了
            # 这里再给一次完整版，是兜底用的：
            # 万一前端漏了几个 token，
            # 可以用这个完整版覆盖
            reply = _get_last_ai_message(
                state.get("messages", []))

            # done 事件：告诉前端「流结束了」
            # 同时带上最终状态
            done_payload: dict = {
                "type": "done", # 事件类型标识（前端靠它区分）
                "reply": reply, # 完整回复（兜底用）
                "current_stage": current_stage, # 当前阶段
                "total_turns": total_turns, # 总轮数
                "is_finished": is_finished, # 是否结束
            }
            # 面试结束 → 额外附带报告摘要
            # 前端可以直接展示，不用再调接口
            if is_finished:
                # 从 State 取报告
                # or {} 防止 report 是 None
                report = state.get("report", {})
                # 往 done_payload 里加一个report_summary字段
                done_payload["report_summary"] = {
                    "overall_score": # 综合得分
                        report.get("overall_score", 0),
                    "strengths": # 核心优势
                        report.get("strengths", []),
                    "improvements": # 提升方向
                        report.get("improvements", []),
                }

            # 推 done 事件给前端
            # 前端收到 type=done 就知道：
            # 不会再有 token 了，可以收尾了
            yield {
                "data": json.dumps(
                    done_payload,
                    ensure_ascii=False
                )
            }

        except Exception as e:
            # 任何异常 → 推一个 error 事件
            # 前端收到 type=error 就显示错误提示
            # 不能让前端一直等着（流已经断了）
            logger.error("interview.chat_stream_error",
                         error=str(e),
                         exc_info=True)
            yield {
                "data": json.dumps(
                    {"type": "error",
                     "message":"流式输出异常，请重试"},
                    ensure_ascii=False,
                )
            }

    # 用 SSE 响应包装异步生成器
    # EventSourceResponse 会：
    #   1. 设置 Content-Type: text/event-stream
    #   2. 每次 yield 就推一条 data: 给前端
    #   3. 生成器结束 → 关闭连接
    # 前端用 EventSource API 接收
    return EventSourceResponse(event_generator())



