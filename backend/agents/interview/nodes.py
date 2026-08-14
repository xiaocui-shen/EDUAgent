# backend/agents/interview/nodes.py
# -------------------------------------------------------
# 7-4
# -------------------------------------------------------
import asyncio
import json
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import text

from backend.agents.interview.state import (
    InterviewState, InterviewStage, AnswerQuality, InterviewReport,
)
from backend.agents.interview.prompts import (
    SYSTEM_PROMPT, WARMUP_PROMPT, INTRO_EVAL_TECH_FIRST_PROMPT,
    TECH_BASE_PROMPT, PROJECT_PROMPT, CLOSING_PROMPT,
    STAGE_TRANSITION_PROMPTS, EVALUATE_ANSWER_PROMPT,
    EVALUATE_THINK_PROMPT, REPORT_SYSTEM_PROMPT,
    GENERATE_REPORT_PROMPT, TECH_BASE_GENERATE_PROMPT,
)
from langchain_core.messages import BaseMessage

from backend.core.llm_factory import (
    get_llm, get_structured_llm)
from backend.core.memory import (
    build_thread_id, should_trigger_summary,
    compress_to_summary, trim_messages_to_window,
)
from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal

logger = get_logger(__name__)            # 本模块日志器

DEFAULT_MAX_TURNS = 40 # 一场面试的总轮数上限（兜底防止无限进行）
MAX_FOLLOWUP_PER_QUESTION = 2 # 同一道题最多追问几次

# 各阶段「最少 / 最多」轮数（check_stage 用，详见 7.5）
STAGE_MIN_TURNS = {
    InterviewStage.WARMUP.value: 1, # 热身至少1轮 自我介绍完即推进
    InterviewStage.TECH_BASE.value: 6,   # 技术环节至少 6 轮
    InterviewStage.PROJECT.value:   2,   # 项目环节至少 2 轮
    InterviewStage.CLOSING.value:   2,   # 反问环节至少 2 轮
}
STAGE_MAX_TURNS = {
    InterviewStage.WARMUP.value: 4,   # 热身最多 4 轮（避免卡住）
    InterviewStage.TECH_BASE.value: 12,  # 技术环节最多 12 轮
    InterviewStage.PROJECT.value:   14,  # 项目环节最多 14 轮
    InterviewStage.CLOSING.value:   4,   # 反问环节最多 4 轮
}


def _msg_text(msg: BaseMessage) -> str:
    """安全提取消息文本，兼容 str 和 list content"""
    content = msg.content # 取出消息内容（可能是 str 或 list）
    if isinstance(content, list):  # 多模态消息：content 是片段列表
        return "".join( # 把各片段的文本拼起来
            p.get("text", "")
            # dict 取 text 字段，否则 str()
            if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content) # 普通字符串直接返回


async def _generate_questions_by_llm(
        target_position: str, count: int = 8) -> list:
    """根据目标岗位用 LLM 动态生成技术面试题，作为 DB 题库的替代/补充。
    参数：
        target_position（岗位，决定出题方向）
        count（题数，默认 8）
    返回：
        题目字典列表，每项={id, content, difficulty, tags, asked}
        失败返回 []
    """
    try: # 出题失败不应让整个加载崩，包try
        # 取模型（温度 0.3，出题略有多样性）
        llm = get_llm("interview", temperature=0.3)
        # 拼出题 Prompt
        prompt = TECH_BASE_GENERATE_PROMPT.format(
            position=target_position, # 填岗位
            count=count, # 填题数
        )
        response = await llm.ainvoke([
            HumanMessage(content=prompt)]) # 调大模型
        text = _msg_text(response).strip() # 安全取出回复文本

        # 兼容 LLM 有时用 ```json ... ``` 包裹输出
        #   把里面以 [ 开头的那段抠出来
        if "```" in text:
            parts = text.split("```") # 按围栏切开
            for part in parts:
                stripped = part.strip()
                # 去掉可能的 "json" 语言标记
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                # 找到以 [ 开头的题目数组
                if stripped.startswith("["):
                    text = stripped
                    break

        questions_raw = json.loads(text) # 解析成 Python 列表
        result = [] # 收集规整后的题目
        for i, q in enumerate(questions_raw): # 逐题处理
            # 跳过格式不对/无题干的
            if (not isinstance(q, dict)
                    or not q.get("content")):
                continue
            result.append({ # 规整成统一结构
                "id": f"llm_{i}", # LLM 题用 llm_0/llm_1...当id
                "content": q["content"], # 题干
                # 难度（缺省 medium）
                "difficulty": q.get("difficulty", "medium"),
                # 标签（缺省用岗位名）
                "tags": q.get("tags", [target_position]),
                "asked": False, # 初始都没问过
            })
        logger.info("generate_questions_llm.done",
                    count=len(result),
                    position=target_position)
        return result # 返回题库
    except Exception as e: # 出题失败（LLM 报错/JSON 解析失败等）
        logger.warning("generate_questions_llm.failed",
                       error=str(e))
        return [] # 返回空，让 load_context 降级到纯 DB 题库


# -------------------------------------------------------
# 7-4 load_context节点
# -------------------------------------------------------
async def load_context_node(state: InterviewState) -> dict:
    """上下文加载节点（每轮第一个执行）。
    读取 state：
        student_id / session_id 定位会话（拼 thread_id）
        total_turn_count 判断是否首轮（==0 即首轮）
        resume_review_id 首轮用它读简历联动数据（可为空）
        target_position 首轮用它生成/查询题库
        resume_projects / resume_skills
            首轮的兜底初始值
    返回（合并进 state）：
        首轮：current_stage / 各计数 / question_bank / resume_projects / resume_skills
              / existing_summary 等一整套初始化
        非首轮：只刷新 existing_summary
    每轮都执行：加载历史摘要（from DB）。
    首轮额外执行：初始化 State、加载简历联动数据、从题库拉题。
    """
    student_id = state.get("student_id") or "" # 取学员 ID
    session_id = state.get("session_id") or "" # 取会话 ID
    # 缺 ID 无法定位会话 → 返回空
    if not student_id or not session_id:
        logger.error("load_context.missing_ids",
                     state_keys=list(state.keys()))
        return {}
    # 拼 thread_id（摘要表/检查点的 key）
    thread_id = build_thread_id(student_id, session_id)
    # 总轮数为 0 = 首轮
    is_first_turn = state.get("total_turn_count", 0) == 0
    updates: dict = {} # 收集要写回 state 的更新

    # ── 内部查询函数（各自独立 session，可并行执行）──────────
    async def _fetch_resume(resume_review_id: str) -> dict:
        # 按 resume_review_id 去 resume_reviews 表
        #   读简历结构化数据（第4章简历Agent写的）
        try:
            async with AsyncSessionLocal() as s: # 独立 DB 会话
                result = await s.execute(
                    text("SELECT structured_data "
                         "FROM resume_reviews "
                         "WHERE id = :id"),
                    {"id": resume_review_id},
                )
                row = result.mappings().fetchone()
            if row and row["structured_data"]: # 读到数据
                # structured_data 可能已是 dict 或 JSON 字符串
                structured = (
                    row["structured_data"]
                    if isinstance(row["structured_data"], dict)
                    else json.loads(row["structured_data"])
                )
                return { # 只取项目和技能两项
                    "resume_projects":
                        structured.get("projects", []),
                    "resume_skills":
                        structured.get("skills_list", []),
                }
        except Exception as e: # 读失败不影响面试，记日志返回空
            logger.warning(
                "load_context.resume_load_failed",
                error=str(e)
            )
        return {}

    async def _fetch_questions(target: str) -> list:
        # 从interview_questions表查岗位题库
        # 匹配该岗位或 general 通用题
        try:
            async with AsyncSessionLocal() as s:
                result = await s.execute(
                    text("""
                        SELECT id, content, difficulty, tags
                        FROM interview_questions
                        WHERE (target_position = :target 
                        OR target_position = 'general')
                        AND is_active = TRUE
                        ORDER BY
                            CASE difficulty
                                WHEN 'medium' THEN 1
                                WHEN 'easy'   THEN 2
                                ELSE 3
                            END,
                            RANDOM()
                        LIMIT 20
                    """),
                    {"target": target},
                )
                rows = result.mappings().all()
            # 规整成统一题目结构，并返回
            return [
                {
                    "id": str(r["id"]),
                    "content": r["content"],
                    "difficulty": r["difficulty"],
                    "tags":
                        r["tags"]
                        if isinstance(r["tags"], list)
                        else (r["tags"] or []),
                    "asked": False, # 初始都没问过
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(
                "load_context.question_bank_failed",
                error=str(e)
            )
            return []

    async def _fetch_summary() -> str | None:
        # 读历史对话摘要
        # 长对话压缩后存在 interview_sessions.summary
        try:
            async with AsyncSessionLocal() as s:
                result = await s.execute(
                    text("SELECT summary "
                         "FROM interview_sessions "
                         "WHERE thread_id = :tid"),
                    {"tid": thread_id},
                )
                row = result.mappings().fetchone()
            return row["summary"] if row else None
        except Exception as e:
            logger.warning(
                "load_context.summary_load_failed",
                error=str(e)
            )
            return None

    # ── 并行执行所有查询 ─────────────────────────────────────
    if is_first_turn: # 首轮：完整初始化
        updates.update({ # 一次性写入所有起始字段
            # 起始阶段 = 热身
            "current_stage":InterviewStage.WARMUP.value,
            "stage_turn_count": 0, # 当前阶段轮数清零
            "total_turn_count": 0, # 总轮数清零
            "followup_count": 0, # 追问次数清零
            "projects_asked": [], # 已深挖项目清空
            "last_answer_quality": # 上轮质量给中性初值
                AnswerQuality.ADEQUATE.value,
            "max_turns": DEFAULT_MAX_TURNS, # 轮数上限 40
            "report": None, # 报告未生成
            "should_summarize": False, # 暂不压缩
            "resume_projects": # 简历项目兜底空
                state.get("resume_projects") or [],
            "resume_skills":  # 简历技能兜底空
                state.get("resume_skills") or [],
        })
        # 简历联动 ID（可能为空）
        resume_review_id = state.get("resume_review_id")
        target = state.get("target_position", "") # 目标岗位

        # 没有简历 ID 时的占位协程（返回空）
        async def _noop_resume():
            return {}

        # 四路I/O各起一个协程：简历/DB题库/LLM出题/历史摘要
        resume_coro = _fetch_resume(resume_review_id) \
            if resume_review_id else _noop_resume()
        questions_coro = _fetch_questions(target)
        llm_q_coro = _generate_questions_by_llm(target)
        summary_coro = _fetch_summary()

        import asyncio
        # gather 同时发起四路、等全部完成（总耗时≈最慢的）
        (resume_data, db_questions,
         llm_questions, existing_summary) = await (
            asyncio.gather(
                resume_coro,
                questions_coro,
                llm_q_coro,
                summary_coro
            )
        )

        # LLM 生成题优先（岗位精准），DB 题兜底/补充
        if llm_questions:
            # LLM 题在前，DB 题补后
            question_bank = llm_questions + (
                db_questions if db_questions else [])
        else: # LLM 出题失败 → 纯用 DB 题
            question_bank = db_questions

        if resume_data: # 读到了简历联动数据
            updates["resume_projects"] = resume_data.get(
                "resume_projects", [])
            updates["resume_skills"]   = resume_data.get(
                "resume_skills", [])
            logger.info(
                "load_context.resume_loaded", projects=len(updates["resume_projects"])
            )

        updates["question_bank"] = question_bank # 写入题库
        # 写入历史摘要
        updates["existing_summary"] = existing_summary
        logger.info(
            "load_context.question_bank_loaded",
            count=len(question_bank)
        )
    else: # 非首轮：只刷新摘要
        updates["existing_summary"] = await _fetch_summary()

    return updates # 返回更新，写回 state


# ------------------------------------------------------
# 7-5 节点: check_stage 面试状态控制器
# ------------------------------------------------------
async def check_stage_node(state: InterviewState) -> dict:
    """
    阶段推进节点（纯逻辑，不调 LLM）：判断当前阶段是否要推进 / 强制结束。

    读取 state：
        current_stage / stage_turn_count / total_turn_count / max_turns / messages
        （TECH_BASE 还看 question_bank，PROJECT 还看 resume_projects/projects_asked）
    返回（写回 state）：
        推进时：{current_stage: 新阶段, stage_turn_count:0, followup_count:0, current_question:None}
        不推进：{}（空 dict，什么都不改）

    阶段推进条件：
        WARMUP → TECH_BASE: turns >= 1（学员完成自我介绍）
        TECH_BASE → PROJECT: stage_turns >= 6 或题库问完
        PROJECT → CLOSING:   stage_turns >= 2 或项目全覆盖
        任意 → FINISHED(强制): total_turns >= max-2 或发送结束关键词
        CLOSING → FINISHED:  stage_turns >= 2
    """
    current_stage = state.get(
        "current_stage", InterviewStage.WARMUP.value) # 当前阶段
    stage_turns = state.get("stage_turn_count", 0) # 当前阶段已进行轮数
    total_turns = state.get("total_turn_count", 0) # 总轮数
    max_turns = state.get("max_turns", DEFAULT_MAX_TURNS) # 轮数上限
    messages = state.get("messages", []) # 对话消息列表

    # 已结束则不再处理（直接交给后面的条件路由去生成报告）
    if current_stage == InterviewStage.FINISHED.value:
        return {}

    # 从后往前找学员最后一条消息（用于检测"结束面试"关键词）
    latest_student_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage): # 找到第一条 Human 消息
            latest_student_msg = _msg_text(msg)
            break

    # 强制结束判断：
    # 总轮数快到上限，或学员主动说要结束 → 直接跳 FINISHED（跳过 CLOSING）
    force_end_keywords = ["结束面试", "不想继续", "面试结束",
                          "结束吧", "到此为止"]
    is_forced_end = (
        total_turns >= max_turns - 2 # 轮数 ≥ 上限-2
        or any(
            kw in latest_student_msg for kw in force_end_keywords
        )  # 或命中关键词
    )

    if is_forced_end: # 触发强制结束
        logger.info("check_stage.forced_to_finished",
                    reason="keyword_or_max_turns")
        return {"current_stage": InterviewStage.FINISHED.value}

    # CLOSING → FINISHED（反问环节满 2 轮就结束）
    if current_stage == InterviewStage.CLOSING.value:
        if stage_turns >= STAGE_MIN_TURNS[
            InterviewStage.CLOSING.value]:
            logger.info("check_stage.advance",
                        from_stage="closing",
                        to_stage="finished")
            return {"current_stage": InterviewStage.FINISHED.value}

        return {} # 还没满 2 轮，继续 closing

    # 常规阶段推进：取当前阶段的最小/最大轮数
    # 至少问几轮才允许推进
    stage_min = STAGE_MIN_TURNS.get(current_stage, 2)
    # 问满几轮就强制推进
    stage_max = STAGE_MAX_TURNS.get(current_stage, 10)

    should_advance = False
    if stage_turns >= stage_max: # 到上限 → 强制推进
        should_advance = True
    elif stage_turns >= stage_min: # 到下限 → 看是否满足内容条件
        should_advance = _check_advance_condition(
            current_stage, state, latest_student_msg)

    if should_advance: # 决定推进
        next_stage = _next_stage(current_stage) # 算出下一阶段
        logger.info("check_stage.advance",
                    from_stage=current_stage,
                    to_stage=next_stage)
        return {
            "current_stage": next_stage, # 切到新阶段
            "stage_turn_count": 0, # 新阶段轮数从 0 开始
            "followup_count": 0, # 追问计数清零
            "current_question": None, # 清掉当前题，让新阶段重新选题
        }

    return {} # 不推进，保持当前阶段

# ------------------------------------------------------
# 7-5 三个辅助函数
# ------------------------------------------------------
def _check_advance_condition(
        stage: str, state: InterviewState, latest_msg: str) -> bool:
    """判断当前阶段「内容上」是否满足推进条件（轮数已达下限后才调用）。"""
    # WARMUP 阶段
    if stage == InterviewStage.WARMUP.value:
        # WARMUP min=1，学员发任意消息即视为完成自我介绍，立即推进
        return True

    # TECH_BASE 阶段
    if stage == InterviewStage.TECH_BASE.value:
        # 取题库
        bank = state.get("question_bank", [])
        # 数已问过的题
        asked_count = sum(1 for q in bank if q.get("asked", False))
        # 至少问过 8 道题（或题库全问完）
        return asked_count >= min(8, len(bank))

    # PROJECT 阶段
    if stage == InterviewStage.PROJECT.value:
        # 简历项目
        projects = state.get("resume_projects", [])
        # 已深挖的项目名
        projects_asked = state.get("projects_asked", [])
        if not projects:
            return True # 无简历项目，直接推进
        # 所有项目都深挖过
        return len(projects_asked) >= len(projects)

    return False # 其它阶段不在这里判断


def _next_stage(current: str) -> str:
    """返回下一个阶段的枚举值字符串（按固定顺序往后走一格）。"""
    # 阶段固定顺序
    order = [
        InterviewStage.WARMUP.value,
        InterviewStage.TECH_BASE.value,
        InterviewStage.PROJECT.value,
        InterviewStage.CLOSING.value,
        InterviewStage.FINISHED.value,
    ]
    try:
        idx = order.index(current) # 找当前阶段位置
        # 下一个阶段的枚举值字符串 到尾就 FINISHED
        next_order = order[idx + 1] \
            if idx + 1 < len(order) \
            else InterviewStage.FINISHED.value
        return next_order
    except ValueError: # current 不在表里（异常）
        return InterviewStage.CLOSING.value # 兜底切到 closing


def _get_last_student_answer(messages: list) -> str:
    """提取消息列表中最后一条学员（Human）消息的文本。"""
    for msg in reversed(messages): # 从后往前找
        if isinstance(msg, HumanMessage):
            return _msg_text(msg)
    return "" # 没有学员消息就返回空串


# ----------------------------------------------------------
# 7-6 节点: evaluate_answer 回答评估器
# ----------------------------------------------------------
async def evaluate_answer_node(state: InterviewState) -> dict:
    """
    评估学员上一轮回答质量（EXCELLENT / ADEQUATE / WEAK / NO_ANSWER）。

    读取 state：
        total_turn_count / current_stage  判断是否跳过评估
        messages                          取学员最后一条回答
        current_question                  取当前题干（评估时作上下文）
        stage_turn_count                  用于 +1
    返回（写回 state）：
        last_answer_quality  本轮质量标签（给 generate_response 决定追问/换题）
        total_turn_count / stage_turn_count  各 +1（轮次真实发生）

    首轮和热身阶段不评估，直接返回 ADEQUATE。
    """
    total_turns = state.get("total_turn_count", 0) # 当前总轮数
    current_stage = state.get(
        "current_stage", InterviewStage.WARMUP.value) # 当前阶段
    messages = state.get("messages", []) # 对话消息

    # 首轮或热身阶段跳过评估：直接给中性标签 ADEQUATE，但计数照常 +1
    if (total_turns == 0
            or current_stage == InterviewStage.WARMUP.value):
        return {
            # 默认质量
            "last_answer_quality": AnswerQuality.ADEQUATE.value,
            "total_turn_count": total_turns + 1, # 总轮数 +1
            "stage_turn_count": # 阶段轮数 +1
                state.get("stage_turn_count", 0) + 1,
        }

    # 取学员最后一条回答文本
    student_answer = _get_last_student_answer(messages)

    # 明确未作答：空回答，或直说"不知道/不清楚/没学过"
    #   → 直接 NO_ANSWER，省一次 LLM
    if (not student_answer.strip()
            or student_answer.strip()
            in ["不知道", "不清楚", "没学过"]):
        return {
            # 标记未作答
            "last_answer_quality": AnswerQuality.NO_ANSWER.value,
            # 计数照常 +1
            "total_turn_count": total_turns + 1,
            "stage_turn_count": state.get("stage_turn_count", 0) + 1,
        }

    # 当前题（可能为空 dict）
    current_q = state.get("current_question") or {}
    # 题干（缺省占位）
    question_text = current_q.get("content", "上一个问题")
    # 回答截断到 800 字，控制 token
    answer_text = student_answer[:800]

    # ── 第一步：Think Tool 推理分析（不约束格式，让 LLM 写判断依据）──
    reasoning_trace = "" # 推理结论文本，初始为空
    try: # 拼推理 Prompt
        think_prompt = EVALUATE_THINK_PROMPT.format(
            question=question_text,
            answer=answer_text,
        )
        # 取模型（温度0，评估要稳定）
        think_llm  = get_llm("qa", temperature=0)
        think_resp = await think_llm.ainvoke([
            HumanMessage(content=think_prompt)]) # 调 LLM 推理
        # 取出推理文本（"内心独白"）
        reasoning_trace = _msg_text(think_resp).strip()
        logger.info("evaluate_think.done",
                    stage=current_stage)
    except Exception as e: # 推理失败不致命，继续走主评估
        logger.warning("evaluate_think.failed",
                       error=str(e))

    # ── 第二步：主评估，把第一步的推理结论注入 prompt 末尾 ───────
    think_context = ( # 构造推理文本，没有就空字符串
        f"\n\n【评估前分析】\n{reasoning_trace}"
        if reasoning_trace else ""
    )
    # 拼主评估 Prompt
    prompt = EVALUATE_ANSWER_PROMPT.format(
        question=question_text,
        answer=answer_text,
    ) + think_context # 末尾接上推理依据

    try:
        llm = get_llm("qa", temperature=0) # 同样温度0
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        # 取文本转小写，便于匹配
        quality_str = _msg_text(response).strip().lower()

        # 用 in 匹配（LLM 可能带空格/换行），命中哪个就用哪个标签
        if "excellent" in quality_str or "优秀" in quality_str:
            quality = AnswerQuality.EXCELLENT.value
        elif "weak" in quality_str or "较弱" in quality_str:
            quality = AnswerQuality.WEAK.value
        elif "no_answer" in quality_str or "未作答" in quality_str:
            quality = AnswerQuality.NO_ANSWER.value
        else: # 兜底：模糊情况按基本及格
            quality = AnswerQuality.ADEQUATE.value
    except Exception as e: # 评估失败 → 兜底 ADEQUATE
        logger.warning("evaluate_answer.failed",
                       error=str(e))
        quality = AnswerQuality.ADEQUATE.value

    logger.info(
        "evaluate_answer.done",
        stage=current_stage,
        quality=quality,
        total_turns=total_turns + 1,
    )

    return {
        "last_answer_quality": quality,  # 本轮质量标签
        "total_turn_count": total_turns + 1, # 总轮数 +1
        # 阶段轮数 +1
        "stage_turn_count": state.get("stage_turn_count", 0) + 1,
    }


# ------------------------------------------------------------
# 7-7 节点: generate_response 回应生成器节点
# ------------------------------------------------------------
async def generate_response_node(state: InterviewState) -> dict:
    """回应生成器节点 分派主函数：
        根据当前阶段调用对应子函数生成面试官回应。
    读取 state：
        current_stage    决定分派到哪个 _respond_*
        stage_turn_count 判断是否刚切换阶段（==1 → 插过渡语）
        messages         传给子函数 + 判断是否触发摘要
    返回（写回 state）：
        messages          面试官这一轮的回复（AIMessage）
        should_summarize  本轮是否要触发摘要压缩
            以及各 _respond_* 子函数返回的字段
            如 current_question/projects_asked/followup_count

    PROJECT/CLOSING 阶段刚切换时（stage_turns==1）插入静态过渡语；
    TECH_BASE 的过渡（介绍评价+第一题）由 _respond_tech_base 内部处理。
    """
    current_stage = state.get(
        "current_stage", InterviewStage.WARMUP.value) # 当前阶段
    stage_turns = state.get("stage_turn_count", 0) # 当前阶段轮数
    messages = state.get("messages", []) # 对话消息

    # 阶段刚切换时（stage_turns==1）插入静态过渡语；
    #   只 PROJECT项目深挖开始 和 CLOSING最后一个阶段 的时候用
    transition_prefix = "" # 过渡语前缀，初始为空
    # 判断是否刚切换阶段（==1）且不是 WARMUP、TECH_BASE
    if stage_turns == 1 and current_stage not in (
        InterviewStage.WARMUP.value, InterviewStage.TECH_BASE.value
    ):
        transition_prefix = STAGE_TRANSITION_PROMPTS.get(
            current_stage, "")  # 取该阶段过渡语

    # 按阶段分派到对应子函数，各自返回 (回复文本, 要写回的 updates)
    # 破冰暖场阶段：WARMUP
    if current_stage == InterviewStage.WARMUP.value:
        response_text, updates = await _respond_warmup(state)
    # 技术基础阶段：TECH_BASE
    elif current_stage == InterviewStage.TECH_BASE.value:
        response_text, updates = await _respond_tech_base(state)
    # 项目深挖阶段：PROJECT
    elif current_stage == InterviewStage.PROJECT.value:
        response_text, updates = await _respond_tech_base(state)
    elif current_stage == InterviewStage.PROJECT.value:
        response_text, updates = await _respond_project(state)
    # 结束阶段：CLOSING
    elif current_stage == InterviewStage.CLOSING.value:
        response_text, updates = await _respond_closing(state)
    else: # FINISHED 等：兜底文案
        response_text = "感谢您参加本次模拟面试，请稍等，正在生成评估报告..."
        updates = {}

    if transition_prefix: # 有过渡语就拼到回复开头
        response_text = transition_prefix + "\n\n" + response_text
    # 面试官回复（追加进 messages）
    updates["messages"] = [AIMessage(content=response_text)]
    # 判断是否该压缩摘要
    updates["should_summarize"] = should_trigger_summary(messages)

    logger.info(
        "generate_response.done",
        stage=current_stage,
        response_length=len(response_text),
    )

    return updates # 返回更新，写回 state

# ------------------------------------------------------------
# 7-7 两个工具函数
# ------------------------------------------------------------
def _build_system_with_summary(state: InterviewState) -> str:
    """构建包含历史摘要的 System Prompt
    有摘要就追加，让面试官"记得"早前对话"""
    position = state.get("target_position", "软件工程师")  # 岗位
    system = SYSTEM_PROMPT.format(position=position) # 系统提示词
    summary = state.get("existing_summary") # 历史摘要（可能为空）
    if summary: # 有摘要才追加
        system += f"\n\n【学员面试历史摘要】\n{summary}"
    return system


def _build_messages(
        system: str, # System Prompt 文本
        history: list, # 完整对话历史
        current_prompt: str, # 本轮要让 LLM 做的"任务描述"
        window: int = 10, # 滑动窗口大小（保留最近几条历史）
    ) -> list:
    """构建发送给 LLM 的消息列表：
        [SystemMessage]
            + 滑动窗口历史（排除最后一条学员消息）
            + current_prompt
    """
    result = [SystemMessage(content=system)] # 第一条永远是 System

    # 滑动窗口裁剪：保留最近 window_size 轮对话
    # history[:-1] 排除最后一条学员消息
    #   它已在 evaluate_answer 处理过，且 current_prompt 已涵盖
    windowed = trim_messages_to_window(
        history[:-1],
        window_size=window # 截取最近 window 条
    )
    for msg in windowed:
        # 跳过历史里的 System（避免重复）
        if not isinstance(msg, SystemMessage):
            result.append(msg)
    # 末尾追加本轮任务描述
    result.append(HumanMessage(content=current_prompt))
    return result


# ------------------------------------------------------------
# 7-7 开场破冰阶段
# ------------------------------------------------------------
async def _respond_warmup(state: InterviewState) -> tuple[str, dict]:
    """破冰热身阶段：只生成欢迎开场白，邀请自我介绍（WARMUP 仅 1 轮）"""
    messages = state.get("messages", []) # 对话历史
    position = state.get("target_position", "软件工程师") # 岗位
    system = _build_system_with_summary(state) # 带摘要的系统提示词
    # 开场白 Prompt
    prompt = WARMUP_PROMPT["opening"].format(position=position)
    # 温度0.7，开场白更自然
    llm = get_llm("interview", temperature=0.7)
    response = await llm.ainvoke( # 调 LLM
        # 构建发送给 LLM 的消息列表
        _build_messages(system, messages, prompt, window=4))
    return _msg_text(response).strip(), {} # 返回(回复文本, 空updates)


# ------------------------------------------------------------
# 7-7 tech_base技术基础阶段
# ------------------------------------------------------------
async def _respond_tech_base(
        state: InterviewState) -> tuple[str, dict]:
    """技术基础阶段：出题或追问。

    读取 state：
        last_answer_quality  上轮质量（EXCELLENT 才追问）
        followup_count       当前题已追问次数（上限 2）
        current_question     当前题（None 表示要出新题）
        question_bank        题库（按 asked 标记选下一题）
        stage_turn_count     ==1 表示刚从 WARMUP 切来
        target_position      岗位
    返回 (回复文本, updates)，updates 可能含：
        question_bank（标记 asked）
        current_question
        followup_count
    """
    messages = state.get("messages", []) # 对话历史

    # 从 State 里读上一个节点（evaluate_answer）写的评分结果
    # 默认值给 ADEQUATE:
    # 万一上一轮没评估（比如首轮）,就当一般处理,不追问也不出错,保守策略
    # 拿到 quality 之后，下面的分支就用它决定追问还是换题
    quality = state.get(
        "last_answer_quality",
        AnswerQuality.ADEQUATE.value
    )
    # 当前题已追问次数
    followup_count = state.get("followup_count", 0)
    current_q = state.get("current_question") # 当前题
    bank = state.get("question_bank", []) # 题库
    stage_turns = state.get("stage_turn_count", 0) # 本阶段轮数
    position = state.get("target_position", "软件工程师")  # 岗位
    system = _build_system_with_summary(state) # 带摘要的系统提示词
    updates: dict = {} # 收集要写回的字段

    # ── 路径1·特殊：刚从 WARMUP 推进而来
    #   stage_turns==1，尚无 current_question──
    if stage_turns == 1 and current_q is None:
        unasked = [q for q in bank if not q.get("asked", False)]  # 未问过的题
        if not unasked: # 题库空：给一句兜底
            return ("感谢你的自我介绍！接下来我们开始技术基础环节。", {})

        first_q  = unasked[0] # 取第一道题
        new_bank = [ # 把这道题标记成 asked=True（生成新list）
            {**q, "asked": True} # {**q}字典kv解包展开
            if q["id"] == first_q["id"] else q
            for q in bank
        ]
        updates["question_bank"] = new_bank # 写回更新后的题库
        updates["current_question"] = first_q # 设为当前题
        updates["followup_count"] = 0 # 追问计数清零

        # 学员的自我介绍文本
        intro_text = _get_last_student_answer(messages)
        # 过渡 Prompt：评价介绍 + 问第一题
        prompt = INTRO_EVAL_TECH_FIRST_PROMPT.format(
            intro=intro_text[:500],
            position=position,
            first_question=first_q["content"],
        )
        llm = get_llm("interview", temperature=0.5)
        # 评价一下自我介绍 + 从题库拿第一道题
        # 用的是专门的 INTRO_EVAL_TECH_FIRST_PROMPT
        # 一次 LLM 调用搞定两件事
        response = await llm.ainvoke(
            _build_messages(system, messages, prompt, window=4))
        return _msg_text(response).strip(), updates

    # ── 判断是否追问当前题：只有回答 EXCELLENT 且没追问满 2 次才追问 ──
    # 三个条件都满足 才追问
    should_followup = (
        current_q is not None # 当前题存在
        # followup_count < 2 这道题已经追问了几次？不到2次才能继续追。
        and followup_count < MAX_FOLLOWUP_PER_QUESTION
        # 答得好(EXCELLENT)才值得追。答得一般或差，追问他更答不上来。
        and quality == AnswerQuality.EXCELLENT.value
    ) # True or False

    # 路径2·追问当前题
    if should_followup:
        # 把标签拼成技术栈字符串
        tech_stack_str = ", ".join(current_q.get("tags", []))
        # 追问 Prompt：上一轮问题 + 回答 + 评价 + 技术栈
        prompt = TECH_BASE_PROMPT["followup"].format(
            question=current_q["content"], # 上一轮问题
            answer=_get_last_student_answer(messages)[:600], # 回答
            quality=quality, # 回答质量
            tech_stack=tech_stack_str, # 技术栈
        )
        # 追问次数 +1
        updates["followup_count"] = followup_count + 1
    # 路径3·出下一道题
    else:
        # 未问过的题
        unasked = [q for q in bank if not q.get("asked", False)]
        if not unasked: # 题问完了 → 提示进入项目环节
            return ("好的，基础技术问题我们就聊到这里，"
                    "接下来我想了解一下你的项目经历。",
                    # 清空当前题
                    {"current_question": None},)

        next_q = unasked[0] # 取下一道题
        new_bank = [
            # 把 q 的所有键值复制一份，再把 asked 改成 True
            {**q, "asked": True} # 标记成 asked=True
            if q["id"] == next_q["id"] else q
            for q in bank
        ]
        updates["question_bank"] = new_bank # 写回题库
        updates["current_question"] = next_q # 设为当前题
        updates["followup_count"] = 0 # 追问计数清零

        # 上一题题干
        prev_question = current_q["content"] if current_q else ""
        # 上一题回答
        prev_answer = _get_last_student_answer(messages)[:600]
        # 反馈上题 + 问下题
        prompt = TECH_BASE_PROMPT["ask_with_feedback"].format(
            question=prev_question, # 上一题题干
            answer=prev_answer, # 上一题回答
            quality=quality, # 上一题回答的质量
            next_question=next_q["content"], # 下一题题干
        )
    # 大模型生成：路径2·追问当前题 and 路径3·出下一道题
    llm = get_llm("interview", temperature=0.4) # 温度0.4
    response = await llm.ainvoke( # window=8
        _build_messages(system, messages, prompt, window=8))
    # 返回(回复文本, updates)
    return _msg_text(response).strip(), updates

# ------------------------------------------------------------
# 7-8 回应生成节点 项目深挖阶段
# ------------------------------------------------------------
async def _respond_project(
        state: InterviewState) -> tuple[str, dict]:
    """项目深挖阶段：基于简历项目针对性追问。
    读取 state：
        last_answer_quality  上轮质量（EXCELLENT 或 ADEQUATE 都可追问，门槛比技术环节低）
        followup_count       当前项目已追问次数（上限 2）
        resume_projects      简历项目列表
        projects_asked       已深挖过的项目名
        current_question     当前正在深挖的项目（动态构造的 dict）
        target_position      岗位
    返回 (回复文本, updates)，updates 可能含：
        followup_count / current_question / projects_asked
    """
    messages = state.get("messages", []) # 对话历史
    quality = state.get( # 上轮回答质量
        "last_answer_quality", AnswerQuality.ADEQUATE.value)
    followup_count = state.get("followup_count", 0) # 当前项目已追问次数
    projects = state.get("resume_projects", []) # 简历中的项目
    projects_asked = state.get("projects_asked", []) # 已深挖的项目名
    # current_q 是当前正在深挖的项目（动态构造的 dict）
    current_q = state.get("current_question") # 当前项目（dict）
    position = state.get("target_position", "工程师")  # 岗位
    system = _build_system_with_summary(state) # 带摘要的系统提示词
    updates: dict = {} # 收集要写回的字段

    if projects: # 有简历联动
        # 找下一个还没深挖过的项目
        next_project = None
        for p in projects:
            # 如果项目名不在已深挖的项目名列表中
            if p.get("name") not in projects_asked:
                # 一旦找到还没深挖的项目 就终止循环
                next_project = p
                break

        # 项目环节追问门槛更低：EXCELLENT优秀 或 ADEQUATE及格 均可追问
        should_followup = (
            current_q is not None # 当前项目存在
            # 追问次数未达上限
            and followup_count < MAX_FOLLOWUP_PER_QUESTION
            and quality in ( # 回答质量达EXCELLENT优秀 或 ADEQUATE及格
                AnswerQuality.EXCELLENT.value,
                AnswerQuality.ADEQUATE.value)
        )

        # 路径1·追问当前项目
        if should_followup:
            # 技术栈字符串
            tech_stack_str = ", ".join(
                current_q.get("tech_stack", []))
            # 先点评上题回答，再追问一个更深的问题。
            prompt = PROJECT_PROMPT["followup_with_feedback"].format(
                # 当前项目名
                project_name=current_q.get("project_name", ""),
                question=current_q["content"], # 上一个问题
                # 上一个问题学员的回答
                answer=_get_last_student_answer(messages)[:600],
                quality=quality, # 上一个问题的评分等级
                tech_stack=tech_stack_str, # 技术栈字符串
                question_num=followup_count + 1, # 第几个追问
            )
            # 追问次数 +1
            updates["followup_count"] = followup_count + 1

        # 路径2·切到新项目，问第1问
        elif next_project:
            # 新项目的技术栈字符串
            tech_stack_str = ", ".join(
                next_project.get("tech_stack", []))
            # 新项目的亮点字符串
            highlights_str = "\n".join( # 把亮点拼成多行
                f"- {h}" for h in next_project.get("highlights", [])
            ) or "（无量化数据）"
            # prompt 告诉 LLM：
            #   这是简历上的项目信息（内部参考，别直接念出来）
            #   生成第1问，问项目背景和职责。
            prompt = PROJECT_PROMPT["new_project"].format(
                project_name=next_project["name"], # 新项目名
                # 学员在项目中的角色
                project_role=next_project.get("role", "开发者"),
                tech_stack=tech_stack_str, # 技术栈字符串
                highlights=highlights_str, # 项目亮点字符串
                description=next_project.get( # 项目描述摘要
                    "description", "")[:200],
            )
            # 动态构造当前项目 dict（非题库题）
            updates["current_question"] = {
                # 项目深挖：{当前项目名}
                "content": f"项目深挖：{next_project['name']}",
                "project_name": next_project["name"], # 当前项目名
                # 当前项目技术栈
                "tech_stack": next_project.get("tech_stack", []),
            }
            # 当前项目加入 已深挖项目列表
            updates["projects_asked"] = projects_asked + [
                next_project["name"]]
            updates["followup_count"] = 0 # 新项目追问计数清零

        # 路径3·所有项目都深挖完了 → 构造综合题提示词
        else:
            prompt = PROJECT_PROMPT["synthesis"].format(
                position=position)

    # 路径4·无简历联动
    else:
        # 基于学员自述出项目题（或架构设计题）
        prompt = PROJECT_PROMPT["no_resume"].format(
            position=position, # 目标岗位
            # 学员最后一条消息 上次的回答
            answer=_get_last_student_answer(messages)[:400],
        )

    llm = get_llm("interview", temperature=0.5) # 温度0.5
    response = await llm.ainvoke( # window=10
        _build_messages(system, messages, prompt, window=10))
    # 返回(回复文本, updates)
    return _msg_text(response).strip(), updates


# ------------------------------------------------------------
# 7-8 回应生成节点 反问收尾阶段
# ------------------------------------------------------------
async def _respond_closing(
        state: InterviewState) -> tuple[str, dict]:
    """反问收尾阶段：开放学员提问。
    读取 state：
        stage_turn_count  ≤1 发提问邀请
                          >1 回应学员的具体问题
        messages          取学员的提问
        target_position   回应时带上岗位上下文
    返回 (回复文本, {})——本阶段不更新状态字段。
    """
    messages = state.get("messages", []) # 对话历史
    stage_turns = state.get("stage_turn_count", 0) # 本阶段轮数
    position = state.get("target_position", "工程师") # 目标岗位
    system = _build_system_with_summary(state) # 带摘要的提示词

    # 刚进 CLOSING反问收尾阶段：邀请学员提问
    if stage_turns <= 1:
        prompt = CLOSING_PROMPT["opening"]
    # 学员已提问：回应其问题
    else:
        # 学员的提问
        last_answer = _get_last_student_answer(messages)
        prompt = CLOSING_PROMPT["respond_question"].format(
            question=last_answer[:300],
            position=position, # 目标岗位
        )
    # 温度0.6，回应更灵活
    llm = get_llm("interview", temperature=0.6)
    response = await llm.ainvoke( # window=4
        _build_messages(system, messages, prompt, window=4))
    return _msg_text(response).strip(), {} # 返回(回复文本, 空updates)


# -----------------------------------------------------------
# 7-9 报告生成节点 generate_report_node
# -----------------------------------------------------------
async def generate_report_node(state: InterviewState) -> dict:
    """
    基于完整对话记录和历史摘要，生成五维度结构化评估报告。

    读取 state：
        messages           完整对话（取最近 40 条格式化）
        existing_summary   历史摘要（早期对话压缩，作全局视角）
        target_position    岗位
        total_turn_count   总轮数（写进 Prompt 和 structured_output）
        session_id         写进 structured_output
    返回（写回 state）：
        report             五维度报告 dict（给 save_report 存库）
        messages           面试结束通知（AIMessage）
        structured_output  给 API 层直接读的摘要数据
    """
    messages = state.get("messages", []) # 完整对话
    summary = state.get("existing_summary", "") # 历史摘要
    position = state.get("target_position", "软件工程师") # 岗位

    # 取最近 40 条对话，只保留学员/面试官消息，格式化成文本
    recent_messages = [
        msg for msg in messages[-40:] # 最近 40 条
        # 只要人/AI 消息
        if isinstance(msg, (HumanMessage, AIMessage))
    ]
    conversation_text = "\n".join([ # 拼成"角色：内容"多行文本
        f"{'面试官' if isinstance(m, AIMessage) else '学员'}："
        f"{_msg_text(m)}"
        for m in recent_messages
    ])
    # 拼报告 Prompt（双层输入）
    prompt = GENERATE_REPORT_PROMPT.format(
        position=position, # 目标岗位
        history_summary=summary or "（无历史摘要）", # 早期对话用摘要
        conversation=conversation_text[:4000], # 近期对话截到4000字
        total_turns=state.get("total_turn_count", 0), # 对话总轮数
    )
    # 获取结构化 LLM，输出自动解析为 InterviewReport 对象
    structured_llm = get_structured_llm(
        "interview", InterviewReport)

    report_dict = None # 初始化报告字典为空
    # 最多重试3次（InterviewReport schema复杂，DeepSeek偶发返回无效JSON导致None）
    for attempt in range(3):
        try:
            result: InterviewReport = await structured_llm.ainvoke([
                SystemMessage(
                    content=REPORT_SYSTEM_PROMPT.format(
                        position=position)),
                HumanMessage(content=prompt),
            ])
            if result is None:
                raise ValueError("structured output returned None")
            report_dict = result.model_dump()
            break # 成功拿到结果，跳出重试循环
        except Exception as e:
            logger.warning(
                "generate_report.failed",
                attempt=attempt, error=str(e))
    # 三次均失败 → 硬编码保守报告
    if report_dict is None:
        report_dict = {
            "dimensions": [],
            "overall_score": 60, # 保守默认分，提示报告生成有误
            "strengths": ["参与了完整的模拟面试流程"],
            "improvements": ["建议深化技术基础知识", "加强项目经历的量化描述"],
            "overall_comment": "感谢参加本次模拟面试，参考各维度建议持续提升。",
            "recommended_topics": [],
            "next_step_advice": "建议针对薄弱点系统复习，1-2周后再次模拟面试。",
        }

    overall_score = report_dict.get("overall_score", 0) # 取综合分
    closing_message = ( # 拼面试结束通知（展示给前端）
        f"好的，本次模拟面试到这里就结束了，感谢你的参与！\n\n"
        f"下面是你的整体表现总结：\n\n"
        f"**综合评分：{overall_score} / 100**\n\n"
        f"详细的各维度分析和改进建议可以在面试记录中查看。"
    )

    logger.info(
        "generate_report.done",
        overall_score=overall_score,
        dimensions=len(report_dict.get("dimensions", [])),
    )

    return { # 给 save_report节点 存库
        "report": report_dict, # 面试报告字典
        # 结束通知（追加进 messages）
        "messages": [AIMessage(content=closing_message)],
        "structured_output": { # 给 API 层直接读的摘要
            "session_id": state["session_id"],
            "report": report_dict, # 面试报告字典
            "total_turns": state.get("total_turn_count", 0),
        },
    }


# -----------------------------------------------------------
# 7-10 保存面试结果节点 save_report_node
# -----------------------------------------------------------
async def save_report_node(state: InterviewState) -> dict:
    """
    将 InterviewReport 写入 interview_sessions表
    更新 status=finished。

    读取 state：
        student_id / session_id  拼 thread_id 定位会话行
        report                   7.9 生成的报告 dict
    返回：{}（不改 state，只落库）
    """

    thread_id = build_thread_id( # langgraph存档id
        state["student_id"], state["session_id"])
    report = state.get("report", {}) # 取面试报告 dict

    async with AsyncSessionLocal() as session: # 异步DB会话
        try:
            # UPDATE（行已在 POST /sessions 插过）
            await session.execute(
                text("""
                    UPDATE interview_sessions
                    SET report = :report,
                        overall_score = :overall_score,
                        status = 'finished',
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE thread_id = :thread_id
                """),
                {
                    "report": json.dumps( # 面试报告
                        report, ensure_ascii=False),
                    "overall_score": report.get(
                        "overall_score", 0), # 综合分单独存一列
                    "thread_id": thread_id, # 会话id
                },
            )
            await session.commit() # 提交
            logger.info("save_report.done",
                        thread_id=thread_id)
        except Exception as e:
            # 失败回滚 + 记日志（不抛，不阻断流程）
            await session.rollback()
            logger.error("save_report.db_failed",
                         error=str(e))
    # 返回：{}（不改 state，只落库）
    return {}


# -----------------------------------------------------------
# 7-10 保存记忆节点 save_memory_node
# -----------------------------------------------------------
async def save_memory_node(state: InterviewState) -> dict:
    """
    按需触发摘要压缩，将最新摘要 UPSERT 到 interview_sessions表。

    读取 state：
        messages / student_id / session_id / tenant_id / target_position / resume_review_id
        existing_summary   现有摘要（压缩时与之合并）
        should_summarize   是否触发压缩
    返回：{}（不改 state，只落库）

    ON CONFLICT (thread_id) 需要表上有 UNIQUE(thread_id) 约束。
    """
    messages = state.get("messages", []) # 对话历史
    student_id = state["student_id"] # 学员 ID
    session_id = state["session_id"] # 会话 ID
    tenant_id = state["tenant_id"] # 租户 ID
    thread_id = build_thread_id(student_id, session_id) # graph存档id
    summary = state.get("existing_summary") # 现有摘要

    # 按需压缩：should_summarize 为 True 时调 LLM 把对话压成新摘要
    if state.get("should_summarize", False):
        try: # 与旧摘要合并压缩
            summary = await compress_to_summary(messages, summary)
        except Exception as e:
            logger.warning("save_memory.compress_failed", error=str(e))

    async with AsyncSessionLocal() as session: # 开异步 DB 会话
        try: # UPSERT：存在则更新摘要，不存在则插入
            await session.execute(
                text("""
                    INSERT INTO interview_sessions
                        (id, tenant_id, student_id, 
                        session_id, thread_id,
                        target_position, resume_review_id, 
                        summary, status)
                    VALUES
                        (:id, :tenant_id, :student_id, 
                        :session_id, :thread_id,
                        :target_position, :resume_review_id, 
                        :summary, 'in_progress')
                    ON CONFLICT (thread_id) DO UPDATE
                        SET summary = EXCLUDED.summary,
                            updated_at = NOW()
                """),
                {
                    "id": str(uuid.uuid4()), # 插入时的新主键
                    "tenant_id": tenant_id,
                    "student_id": student_id,
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "target_position": state.get(
                        "target_position", ""), # 目标岗位
                    "resume_review_id": state.get("resume_review_id"),
                    "summary": summary, # 最新摘要
                },
            )
            await session.commit() # 提交
        except Exception as e: # 失败回滚 + 记日志
            await session.rollback()
            logger.warning("save_memory.db_failed",
                           error=str(e))
    # 返回：{}（不改 state，只落库）
    return {}