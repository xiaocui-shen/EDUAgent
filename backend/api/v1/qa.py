# backend/api/v1/qa.py

import json
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from backend.agents.qa.graph import build_qa_graph
from backend.core.memory import build_thread_id
from backend.dependencies import get_current_user
from backend.core.logger import get_logger
from backend.langfuse_handler import langfuse_handler

router = APIRouter()
logger = get_logger(__name__)


# ── 请求 / 响应模型 ───────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    course_id: str | None = Field(
        None, description="课程 ID（可选，限定检索范围）")
    message: str = Field(..., min_length=1, max_length=2000) # 消息
    enable_web_search: bool = Field(
        False, description="低置信度时是否先走 Web Search 再给 LLM")

class ChatResponse(BaseModel):
    session_id: str # 会话 ID
    answer: str # 回答
    answer_mode: str # "rag" / "web_augmented" / "llm_direct" / "general"
    confidence: float # 置信度
    sources: list[str] # 回答的来源（文档来源或联网搜索来源）
    fallback_used: bool # 是否用了降级兜底策略

# 单条历史消息，是HistoryResponse.messages列表中的元素
class SessionMessage(BaseModel):
    role: str # "user" / "assistant"
    content: str # 消息内容
    created_at: str # 消息创建时间

# GET /history 接口返回这个结构，前端用来展示历史对话记录。
class HistoryResponse(BaseModel):
    session_id: str # 会话 ID
    messages: list[SessionMessage] # 对话历史列表
    summary: str | None # 对话摘要（压缩后的）
    total_turns: int # 总对话轮数

# -------------------------------------------------------
# 智能问答 同步接口
# -------------------------------------------------------
# 这个接口是「一问一答」模式：前端发一个问题，后端跑完整个 QA Agent 流程，
# 把完整答案一次性返回。适合不需要「打字机效果」的场景。
# -------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,                              # 前端发来的请求体（session_id、message 等）
    current_user: dict = Depends(get_current_user), # 从 JWT Token 里解析出的当前用户信息
):
    """智能问答：发送消息，获取 RAG 或 LLM 直答（非流式）"""

    # ── 第1步：构建 LangGraph 图对象 ──
    # build_qa_graph() 会把 10 个节点（classify_query、retrieve、generate_rag 等）
    # 按路由规则连成一张完整的流程图。每次调用都新建，保证状态干净。
    graph = build_qa_graph()

    # ── 第2步：生成 thread_id（LangGraph 的记忆检索键）──
    # 格式：student_{用户ID}_session_{会话ID}
    # LangGraph 靠这个键找到对应的对话历史，实现多轮上下文感知。
    thread_id = build_thread_id(current_user["user_id"], req.session_id)

    # ── 第3步：组装初始状态（initial_state）──
    # 这是 QA Agent 的「起跑线」，所有节点都从这里开始读数据。
    initial_state = {
        "messages": [HumanMessage(content=req.message)],  # 用户当前这条消息
        "student_id": current_user["user_id"], # 学生 ID（查数据库用）
        "tenant_id": current_user["tenant_id"], # 租户 ID（多租户隔离用）
        "session_id": req.session_id, # 前端会话 ID
        "course_id": req.course_id, # 课程 ID（可选，限定检索范围）
        "query_type": "PRECISE", # 占位值！classify_query 节点会覆盖它
        "enable_web_search": req.enable_web_search, # 用户是否勾选了「联网搜索」
        "web_search_results": [], # 每轮重置，防止上轮搜索结果污染本轮
    }

    # ── 第4步：配置 LangGraph 的 thread_id ──
    # LangGraph 的 config 是固定格式，thread_id 必须放在 configurable 字典里。
    config: dict = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": current_user["user_id"],
            "langfuse_session_id": req.session_id,
        },
    }

    # ── 第5步：运行图（ainvoke = 异步一次性执行）──
    # 图会从 START 开始，按路由规则逐个节点执行，直到 END。
    # 返回值 result 就是最终的 QAState 字典，
    #   包含 answer、answer_mode、confidence 等。
    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        # 图执行出错时，记日志 + 返回 500 错误给前端
        logger.error("chat.invoke_error",
                     error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AGENT_ERROR", "message": str(e)},
        )

    # ── 第6步：把结果包装成 ChatResponse 返回 ──
    # result 是 QAState 字典，用 .get() 安全取值（字段不存在时给默认值）。
    return ChatResponse(
        session_id=req.session_id,
        answer=result.get("answer", ""), # AI 的完整回答
        # 回答模式：rag/web_augmented/llm_direct/general
        answer_mode=result.get("answer_mode", "llm_direct"),
        confidence=(result.get("structured_output") or {}).get(
            "confidence", 0.0), # 检索置信度（0~1）
        sources=result.get("sources", []), # 参考来源列表（URL 或文档名）
        fallback_used=result.get("fallback_used", False), # 是否用了兜底策略
    )


# -------------------------------------------------------
# 智能问答 流式接口
# -------------------------------------------------------
# 这个接口是「打字机」模式：前端发一个问题，后端边跑边推 token，
# 前端可以实时显示「AI 正在打字」的效果。用的是 SSE（Server-Sent Events）。
# -------------------------------------------------------
@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,                              # 和同步接口入参完全一样
    current_user: dict = Depends(get_current_user),
):
    """智能问答流式接口（SSE），与 /chat 入参相同，响应改为流式推送"""

    # ── 构建图和 thread_id（和同步接口一样）──
    graph = build_qa_graph()
    thread_id = build_thread_id(current_user["user_id"], req.session_id)

    # ── 初始状态（和同步接口一样）──
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "student_id": current_user["user_id"],
        "tenant_id": current_user["tenant_id"],
        "session_id": req.session_id,
        "course_id": req.course_id,
        "query_type": "PRECISE",
        "enable_web_search": req.enable_web_search,
        "web_search_results": [],
    }
    config: dict = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": current_user["user_id"],
            "langfuse_session_id": req.session_id,
        },
    }

    # ─ 只对这三个生成节点做 token 级流式推送 ──
    # 其他节点（classify_query、retrieve 等）不推 token，只推进度提示。
    # 因为只有这三个节点会调用 LLM 生成文本，才有 token 可推。
    _GENERATE_NODES = {
        "generate_rag", "generate_direct", "generate_general"}

    # ── 节点开始时推送的进度文案 ──
    # 前端收到 "progress" 事件后，会显示对应的加载提示（如「理解问题中...」）。
    # 注意：hyde_generate 和 classify_query 共用「理解问题中...」，
    # 因为对前端来说，这两个阶段用户感知都是「AI 在理解问题」。
    _PROGRESS_LABELS = {
        "classify_query": "理解问题中...",
        "hyde_generate": "理解问题中...",
        "multi_query_rewrite": "改写查询中...",
        "retrieve": "召回相关文档...",
        "web_search": "搜索互联网...",
        "generate_general": "思考中...",
    }

    # ── 事件生成器（核心）──
    # 这是一个异步生成器函数，用 yield 逐个推送 SSE 事件给前端。
    # EventSourceResponse 会把每个 yield 的字典转成 SSE 格式发送。
    async def event_generator():
        # 这三个变量用来收集最终元数据，在流结束后一次性推送（meta 事件）。
        answer_mode = "llm_direct"   # 默认值，节点执行时会被覆盖
        confidence = 0.0             # 置信度 默认值
        sources: list[str] = []      # 回答的来源 默认值

        try:
            # ── 用 astream_events 替代 ainvoke ──
            # ainvoke 是「跑完再返回」，astream_events 是「边跑边推事件」。
            # version="v2" 是 LangGraph 的事件格式版本，必须指定。
            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                evt = event["event"] # 事件类型
                # 当前执行的节点名
                node = event.get("metadata", {}).get("langgraph_node", "")

                # ── 节点开始：推送进度提示 ─
                # on_chain_start = 节点刚开始执行
                # 如果这个节点在 _PROGRESS_LABELS 里，就推一个progress事件给前端
                if evt == "on_chain_start" and node in _PROGRESS_LABELS:
                    yield {
                        "data": json.dumps(
                            {
                                "type": "progress",
                                # 如「理解问题中...」
                                "stage": _PROGRESS_LABELS[node]
                            },
                            ensure_ascii=False, # 保证中文不被转成 \uXXXX
                        )
                    }

                # ── LLM 实时 token ─
                # on_chat_model_stream = LLM 正在吐 token
                # 只在三个生成节点里推 token
                # 其他节点的 LLM 调用如 classify_query 里的策略判断）不推
                elif (evt == "on_chat_model_stream"
                      and node in _GENERATE_NODES):
                    chunk = event["data"].get("chunk")
                    # chunk.content 就是当前这个 token 的文本
                    if chunk and chunk.content:
                        yield {
                            "data": json.dumps(
                                {
                                    "type": "token",
                                    "content": chunk.content
                                },
                                ensure_ascii=False,
                            )
                        }

                # ── 节点结束：捕获元数据 ──
                # on_chain_end = 节点执行完毕
                # 从输出里提取 answer_mode、sources、confidence，
                # 留给最后的 meta 事件用
                elif evt == "on_chain_end" and node in _GENERATE_NODES:
                    output = event["data"].get("output", {})
                    if isinstance(output, dict):
                        _mode = output.get("answer_mode")
                        if _mode:
                            answer_mode = _mode  # 覆盖默认值
                        _srcs = output.get("sources")
                        if _srcs is not None:
                            sources = _srcs # 覆盖默认值
                        # confidence藏在structured_output里（生成节点的返回格式）
                        _conf = (output.get("structured_output") or {}).get("confidence")
                        if _conf is not None:
                            confidence = _conf   # 覆盖默认值

        except Exception as e:
            # 流式执行出错时，推一个 error 事件给前端，前端可以显示错误提示
            logger.error("chat_stream.error",
                         error=str(e), exc_info=True)
            yield {
                "data": json.dumps(
                    {
                        "type": "error",
                        "message": "流式输出异常，请使用普通接口重试"
                    },
                    ensure_ascii=False,
                )
            }
            return  # 出错后直接结束，不再推 meta 和 done

        # ── 元数据帧（流结束后一次性推送）──
        # 前端收到 meta 事件后，可以更新 UI 上的置信度、来源列表等信息。
        # 这些信息不适合在 token 流里逐字推，所以等生成完了再一次性给。
        yield {
            "data": json.dumps(
                {
                    "type": "meta",
                    "session_id": req.session_id,
                    "answer_mode": answer_mode,
                    "confidence": confidence,
                    "sources": sources,
                },
                ensure_ascii=False,
            )
        }

        # ── 结束信号 ──
        # 前端收到 done 事件后，知道这次流式推送彻底结束了，可以隐藏加载动画。
        yield {"data": json.dumps({"type": "done"})}

    # ── 返回 SSE 响应 ──
    # EventSourceResponse 会把 event_generator() 的每个 yield 转成 SSE 格式：
    #   data: {"type": "progress", "stage": "理解问题中..."}\n\n
    #   data: {"type": "token", "content": "Spring"}\n\n
    #   ...
    return EventSourceResponse(event_generator())


# -------------------------------------------------------
# 获取会话历史接口
# -------------------------------------------------------
# 前端用这个接口拉取某个会话的完整对话记录 + 摘要，用于展示历史消息列表。
# 数据来源有两处：
#   ① PostgreSQL 的 qa_sessions 表 → 拿摘要（summary）
#   ② LangGraph 的 MemorySaver（checkpoint）→ 拿消息历史（messages）
# -------------------------------------------------------
@router.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse)
async def get_session_history(
        session_id: str, # 路径参数：前端指定的会话 ID
        current_user: dict = Depends(get_current_user), # 从JWT解析当前用户
    ):
    """获取会话历史消息与摘要"""

    # ── 导入依赖（放在函数内，避免模块加载时循环依赖）──
    from sqlalchemy import text as sa_text
    from langchain_core.messages import (
        HumanMessage as LCHuman, # 重命名避免和 SessionMessage 混淆
        AIMessage as LCAi)
    from backend.dependencies import AsyncSessionLocal

    # ─ 生成 thread_id（和 chat 接口一样的逻辑）──
    # 格式：student_{用户ID}_session_{会话ID}
    # 这个键同时用于查数据库（qa_sessions 表）和查 MemorySaver（checkpoint）
    student_id = current_user["user_id"]
    thread_id = build_thread_id(student_id, session_id)

    # ═══════════════════════════════════════════════════════════
    # 第①步：从 PostgreSQL 读对话摘要
    # ═══════════════════════════════════════════════════════════
    # 摘要是什么？当对话超过 10 轮时，系统会把历史压缩成一段文字摘要，
    # 存在 qa_sessions 表的 summary 字段里。这样后续对话不用带全部历史，
    # 只带摘要 + 最近几轮，防止撑爆 LLM 上下文。
    summary = None
    try:
        async with AsyncSessionLocal() as db_session:
            result = await db_session.execute(
                sa_text(
                    "SELECT summary FROM qa_sessions "
                    "WHERE thread_id = :tid AND student_id = :sid"
                ),
                {"tid": thread_id, "sid": student_id},
            )
            row = result.fetchone()
            if row:
                summary = row[0] # row[0] 就是 summary 字段的值
    except Exception as e:
        # 查摘要失败不阻断接口，summary 保持 None，前端不显示摘要即可
        logger.warning("get_history.db_error", error=str(e))

    # ═══════════════════════════════════════════════════════════
    # 第②步：从 MemorySaver 读消息历史
    # ═══════════════════════════════════════════════════════════
    # MemorySaver 是LangGraph的「记忆存储器」，每次图执行完会自动保存状态。
    # 用aget_state()可以读取某个thread_id对应的历史状态，里面包含messages列表。
    messages: list[SessionMessage] = []
    total_turns = 0
    try:
        # 建图（和 chat 接口一样，每次新建保证状态干净）
        graph = build_qa_graph()
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_user_id": student_id,
                "langfuse_session_id": session_id,
            },
        }

        # aget_state() 读取该 thread_id 的最新保存状态
        # 返回的 state.values 就是当时保存的 QAState 字典
        state = await graph.aget_state(config)

        if state and state.values:
            # 从 state.values 里取出 messages 列表
            # 这个列表包含所有轮次的 HumanMessage 和 AIMessage
            for msg in state.values.get("messages", []):
                # ── 提取消息文本 ──
                # LangChain的消息对象有两种取值方式：
                #   - 旧版本用 .text 属性
                #   - 新版本用 .content 属性
                # 这里兼容两种写法
                content = (
                    msg.text
                    if hasattr(msg, "text") and not callable(msg.text)
                    else str(msg.content)
                )

                # ── 按消息类型封装成 SessionMessage ──
                # 只保留user和assistant两种角色，过滤掉SystemMessage等
                if isinstance(msg, LCHuman):
                    messages.append(
                        SessionMessage(role="user", content=content, created_at="")
                    )
                elif isinstance(msg, LCAi):
                    messages.append(
                        SessionMessage(role="assistant", content=content, created_at="")
                    )

            # 统计对话轮数：一条 user 消息 = 一轮
            total_turns = sum(1 for m in messages if m.role == "user")

    except Exception as e:
        # 读 checkpoint 失败不阻断接口，返回空列表，前端显示「暂无历史」
        logger.warning("get_history.checkpoint_error",
                       error=str(e))

    # ── 返回历史响应 ──
    return HistoryResponse(
        session_id=session_id,
        messages=messages,      # 对话历史列表（按时间顺序）
        summary=summary,        # 对话摘要（可能为 None）
        total_turns=total_turns, # 总轮数（方便前端显示「共 X 轮对话」）
    )
