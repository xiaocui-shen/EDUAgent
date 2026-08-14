# backend/api/v1/unified_chat.py
# 统一 AI 助手入口：意图识别 → 路由决策可视化 → Agent 执行

# SSE 事件类型：
#   routing_decision:
#       路由决策结果（agent_type/confidence/reason/execution_mode）
#   progress: QA 流水线进度提示（检索知识库...等）
#   token: QA Agent 流式 token
#   guidance: 非 QA 意图的引导消息（含跳转链接）
#   pipeline_plan:多Agent协同计划steps数组，含各步label/desc/action_url
#   meta: QA 回答完毕后的元数据（answer_mode / sources 等）
#   done: 流结束信号
#   error: 异常

import json
import re
from dataclasses import dataclass
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

# 8.2 定义的枚举
from backend.core.orchestrator import AgentType, ExecutionMode
# 5.10的thread_id 工具
from backend.core.memory import build_thread_id
from backend.core.llm_factory import get_llm # 3.4 的 LLM 工厂
from backend.dependencies import get_current_user # 3.6 的认证依赖
from backend.core.logger import get_logger
from backend.langfuse_handler import langfuse_handler

router = APIRouter()
logger = get_logger(__name__)


# ── Agent 中文名映射（路由卡片展示用）──────────────────────────
_AGENT_DISPLAY = {
    AgentType.QA:        "智能问答",
    AgentType.EXAM:      "试卷批改",
    AgentType.RESUME:    "简历审查",
    AgentType.INTERVIEW: "模拟面试",
}

# ── 非 QA 意图的引导消息和跳转路径 ─────────────────────────────
# exam/resume/interview需要专门的页面（上传文件/选岗位）
# 统一入口不直接执行，而是返回引导卡片，让前端把用户导到对应功能页。
_GUIDANCE = {
    AgentType.EXAM: {
        "message": "检测到您需要进行试卷批改。请前往「试卷批改」页面上传 Word 格式答卷，AI 将自动完成三轨批改并分析知识薄弱点。",
        "action_label": "前往试卷批改",
        "action_url": "/exam",
    },
    AgentType.RESUME: {
        "message": "检测到您需要进行简历审查。请前往「简历审查」页面上传 PDF 格式简历，AI 将从六个维度进行评分并给出改进建议。",
        "action_label": "前往简历审查",
        "action_url": "/resume",
    },
    AgentType.INTERVIEW: {
        "message": "检测到您需要进行模拟面试。请前往「模拟面试」页面选择目标岗位和技术方向，AI 将进行技术原理与项目深挖双轨考察。",
        "action_label": "前往模拟面试",
        "action_url": "/interview",
    },
}

# ── QA Agent 节点进度提示（与 qa.py 保持一致，8.5 用到）────────
# 节点名 -> 给前端用于显示的进度提示
_PROGRESS_LABELS = {
    "classify_query":        "理解问题中...",
    "retrieve":              "召回相关文档...",
    "rerank":                "精排中...",
    "hyde_generate":         "理解问题中...",
    "multi_query_rewrite":   "改写查询中...",
    "generate_general":      "思考中...",
}
# QA三个生成节点 的 名字集合
_GENERATE_NODES = {
    "generate_rag", "generate_direct", "generate_general"}


# -----------------------------------------------------------
# 规则拦截：五类零 Token 回复
# -----------------------------------------------------------
# ── 去除末尾标点空白，用于关键词精确比对 ──────────────────────────
_STRIP_TAIL_RE = re.compile(r"[\s!！?？。~～,.，。]+$")

# ── 类别一：问候（你好 / 在吗）────────────────────────────────────
_HELLO_KEYWORDS = frozenset([
    "你好", "您好", "hi", "hello", "hey", "哈喽", "嗨",
    "在吗", "在不在", "在线吗", "有人吗",
])

# ── 类别二：感谢（谢谢 / 辛苦了 / 太棒了）────────────────────────
_THANKS_KEYWORDS = frozenset([
    "谢谢", "感谢", "多谢", "谢了", "非常感谢", "万分感谢",
    "辛苦了", "辛苦", "麻烦了", "不好意思",
    "太棒了", "太好了", "厉害", "厉害了", "牛", "牛啊", "牛逼",
    "好的好的", "明白了", "懂了", "知道了", "收到",
])

# ── 类别三：道别（再见 / bye）──────────────────────────────────────
_BYE_KEYWORDS = frozenset([
    "再见", "拜拜", "拜", "88", "886", "bye", "goodbye", "byebye",
    "下次见", "下次再聊", "先走了", "先撤了", "溜了", "闪了",
])

# ── 类别四：身份询问（你是谁 / 介绍你自己）── 正则匹配 ──────────────
_IDENTITY_RE = re.compile(
    r"(你|您)(是谁|叫什么|的名字|是什么|是.*AI|是.*机器人|是.*助手)"
    r"|介绍.{0,4}(你自己|自己|一下)"
    r"|你是谁"
    r"|你叫(啥|什么名)",
    re.IGNORECASE,
)

# ── 类别五：功能询问（你能做什么 / 怎么用）── 正则匹配 ──────────────
_CAPABILITY_RE = re.compile(
    r"(你|您)(能|可以|会).{0,6}(做|帮|干)"
    r"|(你|您).{0,4}(功能|用途|能力|特点)"
    r"|怎么(用|使用)(你|您|这个)?"
    r"|(使用说明|帮助菜单|help|usage)"
    r"|你能帮(我|忙)吗",
    re.IGNORECASE,
)


# ── 多 Agent 能力说明（HELLO / IDENTITY / CAPABILITY 共享结尾）────
_MULTI_AGENT_TIP = (
    "- **多 Agent 协同**：描述综合需求（如「帮我准备求职」），"
    "AI 将自动串联简历审查 + 模拟面试两个 Agent 协同为您服务\n"
)


# ── 五类回复模板（节选问候，其余结构相同）──────────────────────────
_REPLY_HELLO = (
    "您好！我是 EduAgent AI 助手。\n\n"
    "我可以帮您：\n"
    "- **技术问答**：直接输入问题，我会从知识库中检索并解答\n"
    "- **试卷批改**：告诉我「我要提交试卷」，AI 完成三轨批改和知识薄弱点分析\n"
    "- **简历审查**：告诉我「帮我看看简历」，AI 给出六维度评分与改进建议\n"
    "- **模拟面试**：告诉我「我要模拟面试」，开启技术双轨考察\n"
    + _MULTI_AGENT_TIP
    + "\n请问有什么可以帮到您？"
)

_REPLY_THANKS = (
    "不客气，很高兴能帮到您！\n\n"
    "如果还有其他问题，随时告诉我，我随时在线。"
)

_REPLY_BYE = (
    "再见！希望今天的学习对您有所帮助，期待下次与您交流。\n\n"
    "祝您学习顺利！"
)

_REPLY_IDENTITY = (
    "我是 **EduAgent AI 助手**，一个面向 IT 教育培训的智能学习辅助系统。\n\n"
    "我由多个专业 Agent 协同构成：\n"
    "- **智能问答 Agent**：基于 RAG 知识库，7x24 即时解答技术问题\n"
    "- **试卷批改 Agent**：AI 三轨并行批改 + 知识薄弱点分析\n"
    "- **简历审查 Agent**：六维度质量评审，提供原文定位的修改建议\n"
    "- **模拟面试 Agent**：技术原理 + 项目深挖双轨考察，生成结构化报告\n"
    + _MULTI_AGENT_TIP
    + "\n有什么可以帮到您吗？"
)

_REPLY_CAPABILITY = (
    "我能为您提供以下功能：\n\n"
    "**单 Agent 直达**\n"
    "- 直接输入技术问题 → 智能问答（RAG 知识库检索）\n"
    "- 「提交试卷批改」 → 上传 Word 答卷，AI 完成批改并分析薄弱点\n"
    "- 「审查我的简历」 → 上传 PDF 简历，六维度评分 + 改进建议\n"
    "- 「开始模拟面试」 → 选择岗位方向，进入双轨面试考察\n\n"
    "**多 Agent 协同**\n"
    + _MULTI_AGENT_TIP
    + "\n直接告诉我您的需求，我会自动路由到最合适的 Agent。"
)


# ── 规则前置拦截函数 ──────────────────────────
def _pre_filter(text: str) -> str | None:
    """
    规则前置拦截，返回模板回复字符串；无命中返回 None（继续正常路由）。
    五类社交/元场景均为零 Token 消耗（不调 LLM）。
    """
    t = text.strip() # 去首尾空白
    # 转小写 + 去末尾标点，得到比对用文本
    t_lower = _STRIP_TAIL_RE.sub("", t.lower())

    # 类别一：问候（精确匹配整句）
    if t_lower in _HELLO_KEYWORDS:
        return _REPLY_HELLO

    # 类别二：感谢（精确匹配整句）
    if t_lower in _THANKS_KEYWORDS:
        return _REPLY_THANKS

    # 类别三：道别（精确匹配整句）
    if t_lower in _BYE_KEYWORDS:
        return _REPLY_BYE

    # 类别四：身份询问（正则，用原文 t 而非去标点的 t_lower）
    if _IDENTITY_RE.search(t):
        return _REPLY_IDENTITY

    # 类别五：功能询问（正则）
    if _CAPABILITY_RE.search(t):
        return _REPLY_CAPABILITY

    return None # 五类都没命中 → 交给 LLM 路由


# ----------------------------------------------------------
# LLM 意图路由
# ----------------------------------------------------------
# ── LLM 路由 Prompt：让模型把用户输入归到 6 类之一，返回 JSON ──
_ROUTE_PROMPT = """判断用户需求应路由到哪个功能。

可选功能：
- qa          : 技术知识问答（用户直接提问，不涉及文件上传）
- exam        : 试卷/作业批改（需上传 Word 答卷，用户提到"批改""作业""提交试卷"等）
- resume      : 简历审查（需上传 PDF 简历，用户提到"简历""帮我看看简历"等）
- interview   : 模拟面试（用户提到"面试""练习面试""模拟面试"等）
- multi_agent : 综合求职准备（用户同时提到简历 + 面试，或"全套""一条龙""求职准备"等）
- clarify     : 意图不明确，无法判断，需要追问

严格按以下 JSON 格式返回，不要有其他内容：
{{"label": "功能名", "reason": "一句话说明判断依据"}}

用户输入：{message}"""

# label → AgentType 映射（multi_agent / clarify 用 QA 占位）
_LABEL_TO_AGENT: dict[str, AgentType] = {
    "qa":          AgentType.QA,
    "exam":        AgentType.EXAM,
    "resume":      AgentType.RESUME,
    "interview":   AgentType.INTERVIEW,
    "multi_agent": AgentType.QA,   # pipeline 入口，以 QA 作占位
    "clarify":     AgentType.QA,
}
# 合法 label 集合（校验 LLM 输出用）
_VALID_LABELS = frozenset(_LABEL_TO_AGENT.keys())


# label → ExecutionMode 映射
_LABEL_TO_MODE: dict[str, ExecutionMode] = {
    "qa":          ExecutionMode.SINGLE,
    "exam":        ExecutionMode.SINGLE,
    "resume":      ExecutionMode.SINGLE,
    "interview":   ExecutionMode.SINGLE,
    "multi_agent": ExecutionMode.PIPELINE,
    "clarify":     ExecutionMode.CLARIFY,
}
'''在 backend/core/orchestrator.py中
class ExecutionMode(str, Enum):
    """执行模式：决定一个请求怎么跑。"""
    SINGLE   = "single"     # 单 Agent 直达（最常见）
    PIPELINE = "pipeline"   # 多 Agent 串联（如求职全链路）
    CLARIFY  = "clarify"    # 澄清对话（意图不明，需追问）
'''

@dataclass
class _RouteResult:
    """LLM 路由结果，对齐前端展示需要的字段。"""
    # "qa"|"exam"|"resume"|"interview"|"multi_agent"|"clarify"
    label: str # ⬆️
    agent_type: AgentType # 由 label 映射
    execution_mode: ExecutionMode # 由 label 映射
    confidence: float # LLM路由固定返回0.85 无置信度概念，仅供前端展示
    reason: str # 一句话判断依据


async def _llm_route(message: str) -> _RouteResult:
    """调用 LLM 对用户输入做跨 Agent 路由判断。
    无需本地分类模型（这个路径调用频率低，LLM 开销可接受）。
    异常时降级返回 qa 路由，不阻断 SSE 流。

    参数：message（用户原始输入）
    返回：_RouteResult
        label+agent_type+execution_mode+confidence+reason
    """
    try:
        # 取"intent"专用模型（3.4 配置），温度0求稳定
        llm = get_llm("intent", temperature=0)
        resp = await llm.ainvoke([ # 调 LLM
            HumanMessage(
                content=_ROUTE_PROMPT.format(message=message)
            )
        ])
        raw = resp.text.strip()
        parsed = json.loads(raw) # 解析 LLM 返回的 JSON
        # 取 label（缺省 qa）
        label = parsed.get("label", "qa").strip().lower()
        # 取判断依据
        reason = parsed.get("reason", "LLM 路由判断")

        # LLM 返回了非法 label → 降级 qa
        if label not in _VALID_LABELS:
            logger.warning(
                "unified_chat.llm_route_unknown_label", label=label,
                fallback="qa"
            )
            label = "qa"

        logger.info("unified_chat.llm_route_result",
                    label=label,
                    reason=reason)

    except Exception as e: # 调用/解析失败 → 降级 qa，不阻断流
        logger.warning("unified_chat.llm_route_failed",
                       error=str(e),
                       fallback="qa")
        label = "qa"
        reason = "路由判断异常，默认转入智能问答"
    # 组装路由结果
    return _RouteResult(
        label=label, # 路由标签 最终指向具体Agent
        agent_type=_LABEL_TO_AGENT[label], # label→AgentType
        # label→ExecutionMode
        execution_mode=_LABEL_TO_MODE[label],
        confidence=0.85, # 固定值，仅供前端展示
        reason=reason,
    )


# --------------------------------------------------------
# 8-5 SSE分发
# --------------------------------------------------------
class UnifiedChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    message: str = Field( # 非空、限长
        ..., min_length=1, max_length=2000, description="用户输入")


def _sse(data: dict) -> dict:
    """把dict包成sse_starlette要的格式：
    {"data": "<JSON字符串>"}。ensure_ascii=False 保留中文。"""
    return {"data": json.dumps(data, ensure_ascii=False)}


@router.post("/stream")
async def unified_chat_stream(
    req: UnifiedChatRequest, # 请求体：session_id + message
    current_user: dict = Depends(get_current_user), # 认证依赖
):
    """ 统一 AI 助手流式接口（SSE）。

    请求：{session_id, message}
    响应：text/event-stream，事件序列随路由结果不同而不同

    流程：
        1. 规则前置拦截（五类社交/元场景，零 Token 直接返回）
        2. LLM 路由判断（_llm_route，DeepSeek 直接分类，无本地模型）
        3. 推送 routing_decision 事件（前端显示路由卡片）
        4a. qa 路由 → 流式执行 QA Agent
        4b. exam/resume/interview → 推送 guidance 引导跳转
        4c. multi_agent  → 推送求职全链路 pipeline 计划
        4d. clarify → 推送追问提示
    """

    async def event_generator(): # 异步生成器：逐个 yield SSE 事件
        # ── Step 0：规则前置拦截（零Token，五类模板回复）──────────
        # 命中则返回模板字符串，否则 None
        pre_reply = _pre_filter(req.message)
        if pre_reply is not None:
            # 模板内容当作一个 token 事件推出
            yield _sse({"type": "token", "content": pre_reply})
            yield _sse({"type": "done"}) # 直接结束
            return # 直接结束函数 不再走路由（省 LLM）

        # ── Step 1：LLM 路由判断 ────────────────────────────
        # 调用 LLM 对用户输入做跨 Agent 路由判断。
        # 得到 _RouteResult（label/agent_type/mode/...）
        decision = await _llm_route(req.message)

        # ── Step 2：推送路由决策卡片（前端显示"已转接到 XX"）──────
        yield _sse({
            "type": "routing_decision",
            # 路由到的 Agent
            "agent_type": decision.agent_type.value,
            "agent_display": _AGENT_DISPLAY.get(
                decision.agent_type, ""), # Agent中文名
            "confidence": round(decision.confidence, 4), # 置信度
            "reason": decision.reason, # 判断依据
            # 执行模式
            "execution_mode": decision.execution_mode.value,
        })

        # ── Step 3：按路由结果分发处理 ──────────────────────
        label = decision.label # 用 label 决定走哪条分支

        # ── 3a：qa → 流式执行 QA Agent ──────────────────────
        if label == "qa":
            # 把 QA 的流式事件透传出去
            async for event in _stream_qa_agent(
                    req, current_user):
                yield event
            '''_stream_qa_agent函数是个异步生成器 
                    里面用yield一个一个往外吐事件
                    async for 就是"它吐一个我接一个"
                _stream_qa_agent 吐event事件
                    ▼
                event_generator 接住event事件
                    ▼
                yield 再把event事件吐出去
                    ▼
                SSE 推给前端
            '''

        # ── 3b：exam / resume / interview → 引导跳转 ────────
        elif label in ("exam", "resume", "interview"):
            # 取对应引导文案
            guidance = _GUIDANCE.get(decision.agent_type, {})
            yield _sse({
                "type": "guidance", # 事件类型
                "message": guidance.get( # 引导文案
                    "message", "请前往对应功能页面操作"),
                # 引导动作说明 “前往试卷批改”等...
                "action_label": guidance.get("action_label", ""),
                # URL跳转路径 /dashboard是兜底的前端的"主页/控制台"页面
                "action_url": guidance.get(
                    "action_url", "/dashboard"),
            })

        # ── 3c：multi_agent → 求职全链路 pipeline_plan ──────
        elif label == "multi_agent":
            yield _sse({
                "type": "pipeline_plan",
                "title": "求职全链路",
                "intro": "已为您规划「求职全链路」，"
                         "建议按顺序完成以下两个步骤，"
                         "简历审查结果将作为模拟面试的参考依据。",
                "steps": [
                    {
                    "step": 1,
                    "agent_type": "resume",
                    "label": "简历审查",
                    "desc":
                        "上传 PDF 简历，"
                        "AI从六个维度评分并给出带原文定位的改进建议",
                    "action_label": "开始简历审查",
                    "action_url": "/resume",
                    "tip": "建议先完成此步骤，"
                           "审查结果将作为面试的参考依据",
                    },
                    {
                    "step": 2,
                    "agent_type": "interview",
                    "label": "模拟面试",
                    "desc":
                        "基于简历背景，"
                        "进行技术原理 + 项目深挖双轨考察，"
                        "生成结构化雷达图报告",
                    "action_label": "开始模拟面试",
                    "action_url": "/interview",
                    "tip": "完成简历审查后进行效果更佳",
                    },
                ],
            })

        # ── 3d：clarify → 追问提示 ───────────────────────────
        else:
            yield _sse({
                "type": "guidance",
                "message":
                    "您的问题我还不太确定应该用哪个功能来帮您，"
                    "能否描述得更具体一些？"
                    "例如：您是想提问技术知识、提交试卷批改、审查简历，"
                    "还是进行模拟面试？",
                "action_label": "",
                "action_url": "",
            })

        yield _sse({"type": "done"}) # 所有分支最终都推一个done收尾

    # 用 SSE 响应包装生成器 并返回
    return EventSourceResponse(event_generator())


async def _stream_qa_agent(req: UnifiedChatRequest, current_user: dict):
    """在统一入口中流式执行 QA Agent，复用qa.py的astream_events逻辑。

    读取：req.message / req.session_id、current_user
    yield：progress / token / meta / error 四类 SSE 事件
    """
    # 懒加载 获取 编排器单例
    from backend.core.orchestrator import get_orchestrator
    orchestrator = get_orchestrator()
    # 复用编排器缓存的 QA 图对象（不重复 build）
    graph = orchestrator._get_agent_graph(AgentType.QA)

    # Graph MemorySaver的存档编号
    thread_id = build_thread_id(
        current_user["user_id"], req.session_id)
    # 构造 QA 图输入 State
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "student_id": current_user["user_id"],
        "tenant_id": current_user["tenant_id"],
        "session_id": req.session_id,
        "course_id": None,
        "query_type": "PRECISE", # QA 的查询类型（见第 5 章）
    }
    # 构造 QA 图配置
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": current_user["user_id"],
            "langfuse_session_id": req.session_id,
        },
    }

    answer_mode = None # 回答模式
    confidence  = 0.0 # 置信度
    sources: list[str] = [] # 答案的引用来源

    try:
        # 流式运行QA图，逐事件处理
        # astream_events函数 langchain官方做过一次升级 v2是固定写法
        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2" # 用新版的astream_events函数方法
        ):
            evt = event["event"] # 事件类型
            # 获取节点名称
            node = event.get(
                "metadata", {}).get("langgraph_node", "")

            # ① 节点开始且在进度表里 → 推progress（"检索知识库..."等）
            # on_chain_start表示事件类型是节点开始（langchain官方的标记）
            if (evt == "on_chain_start"
                    and node in _PROGRESS_LABELS):
                yield _sse({
                    "type": "progress", # 事件类型 QA流程开始
                    # 根据节点名 获取给前端显示的进度文案。
                    "stage": _PROGRESS_LABELS[node]
                })

            # ② 生成节点的 token 流 → 推 token（答案逐字）
            elif (evt == "on_chat_model_stream"
                  and node in _GENERATE_NODES):
                # 生成的回答片段
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    yield _sse({
                        "type": "token",
                        "content": chunk.content
                    })

            # ③ 生成节点结束
            elif (evt == "on_chain_end"
                  and node in _GENERATE_NODES):
                # 获取节点输出
                output = event["data"].get("output", {})
                # 仅在值非空时更新 答案模式、来源、置信度
                if output.get("answer_mode"): # 答案模式
                    answer_mode = output["answer_mode"]
                if output.get("sources"): # 来源
                    sources = output["sources"]
                # 置信度
                confidence = (
                    output.get("structured_output") or {}
                ).get("confidence", confidence)

    except Exception as e: # QA 执行异常 → 推 error 并结束
        logger.error("unified_chat.qa_stream_error",
                     error=str(e), exc_info=True)
        yield _sse({
            "type": "error",
            "message": "问答服务异常，请稍后重试"
        })
        return

    # ④ QA答完 → 推 meta事件，返回 答案模式 / 来源 / 置信度
    # 前端据此显示"引用了 N 篇"，N=len(sources)
    if answer_mode:
        yield _sse({
            "type": "meta",
            "answer_mode": answer_mode,
            "confidence": confidence,
            "sources": sources
        })
