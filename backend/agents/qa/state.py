# backend/agents/qa/state.py

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class QAState(TypedDict):
    """
    智能问答 Agent 的完整状态定义。
    所有节点通过读写此 State 进行数据传递。
    """

    # ── ① 消息历史（LangGraph 核心，add_messages reducer）────────
    # 节点每次返回 messages 时自动追加，不覆盖历史
    messages: Annotated[list[BaseMessage], add_messages]

    # ── ② 请求上下文（Orchestrator 注入，节点只读）───────────────
    student_id:  str            # 学员 ID
    tenant_id:   str            # 租户 ID（Milvus / DB 多租户隔离）
    session_id:  str            # 会话 ID（用于构造 thread_id）
    course_id:   Optional[str]  # 课程 ID，限制检索范围；None = 全库检索

    # ── ③ Query 处理中间结果──────────────────────────────────────
    original_query:    str         # 用户原始输入，全程不变
    query_type:        str         # GENERAL / PRECISE / VAGUE / BROAD
    rewritten_queries: list[str]   # BROAD 分支：Multi-Query 改写后的子 Query 列表
    hyde_document:     Optional[str]  # VAGUE 分支：HyDE 生成的假设文档文本

    # ── ④ 检索与精排结果─────────────────────────────────────────
    # retrieve() 内部已处理 BGE-M3 编码，State 无需存储中间向量
    ranked_chunks:      list[dict]   # BGEReranker 精排后的 Top-K chunk
    confidence:         float        # 精排置信度 [0, 1]
    is_high_confidence: bool         # confidence >= 0.75
    web_search_results: list[dict]   # Web Search MCP 返回结果（低置信度时填充）

    # ── ⑤ 生成结果 & 控制标记──────────────────────────────────────
    answer:            str           # 最终回答文本
    sources:           list[str]     # 来源标注列表（高置信度 RAG 时填充）
    answer_mode:       str           # "rag" / "llm_direct"
    existing_summary:  Optional[str] # 当前会话的历史摘要（从 DB 读取）
    should_summarize:  bool          # 是否触发摘要压缩
    enable_web_search: bool          # True = 低置信度时先走 Web Search 再兜底
    fallback_used:     bool          # 是否触发了降级处理
    structured_output: Optional[dict]  # 传给 Orchestrator 的结构化数据
