
# backend/agents/qa/graph.py

from langgraph.graph import StateGraph, START, END

from backend.agents.qa.state import QAState
from backend.agents.qa.nodes import (
    classify_query_node,
    hyde_generate_node,
    multi_query_rewrite_node,
    retrieve_node,
    generate_rag_node,
    web_search_node,
    generate_direct_node,
    generate_general_node,
    enqueue_pending_node,
    save_memory_node,
)
from backend.core.memory import get_memory_saver


def _route_by_query_type(state: QAState) -> str:
    """classify_query 之后的路由：
        根据 query_type 和 enable_web_search 分流。

    GENERAL + enable_web_search=True  → "GENERAL_WEB"：先联网再回答
    GENERAL + enable_web_search=False → "GENERAL"：直接 LLM
    PRECISE / VAGUE / BROAD           → 原样返回：直接进对应检索分支
    """
    qt = state.get("query_type", "PRECISE").upper()
    if qt == "GENERAL" and state.get("enable_web_search", False):
        return "GENERAL_WEB"
    return qt


def _route_by_confidence(state: QAState) -> str:
    """retrieve 之后的路由：根据置信度和联网开关分流。

    is_high_confidence=True              → "high"：RAG 高质量回答
    is_high_confidence=False
      + enable_web_search=True           → "low_web"：先联网补充再直答
      + enable_web_search=False          → "low_direct"：直接 LLM 兜底
    """
    if state.get("is_high_confidence", False):
        return "high"
    if state.get("enable_web_search", False):
        return "low_web"
    return "low_direct"


def _route_after_web_search(state: QAState) -> str:
    """web_search 节点被两条路径共用，走完搜索后需要区分去向：
    - 来自 GENERAL_WEB 路径（query_type=GENERAL）→ generate_general
    - 来自低置信度路径                            → generate_direct
    """
    if state.get("query_type", "").upper() == "GENERAL":
        return "generate_general"
    return "generate_direct"


def build_qa_graph():
    """构建并编译 QA Agent 的 LangGraph 状态图。
    Returns:
        编译后的 CompiledGraph，供 API 层和 Orchestrator 调用。
    """
    builder = StateGraph(QAState)

    # ── 注册节点 ──────────────────────────────────────────────
    # 意图分类 判断问题类型，返回 PRECISE精准问题 / VAGUE / BROAD / GENERAL通用问题
    builder.add_node("classify_query",       classify_query_node)
    # 生成假设文档（生成假设问题、参考答案文档） VAGUE 问题太模糊
    builder.add_node("hyde_generate",        hyde_generate_node)
    # 拆成多个子问题  BROAD 问题太宽泛
    builder.add_node("multi_query_rewrite",  multi_query_rewrite_node)
    # 混合检索并重排，返回检索结果
    builder.add_node("retrieve",             retrieve_node)
    # 对于 high 高置信度问题，生成 rag 回答
    builder.add_node("generate_rag",         generate_rag_node)
    # 联网搜索，返回搜索结果
    builder.add_node("web_search",           web_search_node)
    # 对于专业问题 LLM回答
    builder.add_node("generate_direct",      generate_direct_node)
    # 对于通用问题 LLM回答
    builder.add_node("generate_general",     generate_general_node)
    # 将低分问题保存到数据库
    builder.add_node("enqueue_pending",      enqueue_pending_node)
    # 保存记忆
    builder.add_node("save_memory",          save_memory_node)

    # ── 入口固定边 ────────────────────────────────────────────
    builder.add_edge(START, "classify_query")

    # ── 条件边①：Query 类型路由 ──────────────────────────────
    builder.add_conditional_edges(
        "classify_query", # 来源节点
        _route_by_query_type, # 路由函数（条件函数）
        # 路由函数返回的路标 → 目标节点
        {
            "GENERAL":     "generate_general",
            "GENERAL_WEB": "web_search",
            "PRECISE":     "retrieve",
            "VAGUE":       "hyde_generate",
            "BROAD":       "multi_query_rewrite",
        },
    )

    # VAGUE / BROAD 预处理完成后汇入 retrieve
    builder.add_edge("hyde_generate",       "retrieve")
    builder.add_edge("multi_query_rewrite", "retrieve")

    # ── 条件边②：置信度路由 ──────────────────────────────────
    builder.add_conditional_edges(
        "retrieve", # 来源节点
        _route_by_confidence, # 路由函数（条件函数）
        # 路标 → 目标节点
        {
            "high":       "generate_rag",
            "low_web":    "web_search",
            "low_direct": "generate_direct",
        },
    )

    # ── 条件边③：web_search 出口路由 ─────────────────────────
    builder.add_conditional_edges(
        "web_search", # 来源节点
        _route_after_web_search, # 路由函数（条件函数）
        # 路标 → 目标节点
        {
            "generate_general": "generate_general",
            "generate_direct":  "generate_direct",
        },
    )

    # ── 固定边：各生成节点 → 收尾节点 ────────────────────────
    builder.add_edge("generate_rag",     "save_memory")
    builder.add_edge("generate_general", "save_memory")
    builder.add_edge("generate_direct",  "enqueue_pending")
    builder.add_edge("enqueue_pending",  "save_memory")
    builder.add_edge("save_memory",      END)

    # ── 编译（绑定 MemorySaver 实现多轮记忆）────────────────
    checkpointer = get_memory_saver("qa")
    return builder.compile(checkpointer=checkpointer)


if __name__ == '__main__':
    graph = build_qa_graph()
    print('图编译成功')
    print('节点列表：', list(graph.nodes.keys()))
