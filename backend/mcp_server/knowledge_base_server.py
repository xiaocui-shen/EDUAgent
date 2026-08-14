# backend/mcp/knowledge_base_server.py

import asyncio
import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="EduAgent-KnowledgeBase",
    stateless_http=True, # 无状态模式
    json_response=True, # JSON 响应
)

# EduAgent 知识库搜索工具
# 调用backend.core.reranker.retrieve RAG检索流水线函数
@mcp.tool()
async def search_knowledge_base(
    query: str,
    tenant_id: str,
    course_id: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Hybrid semantic + keyword search over the EduAgent Milvus knowledge base.

    Uses BGE-M3 dual-vector encoding combined with Milvus hybrid_search
    and WeightedRanker fusion, followed by BGE-Reranker cross-encoder reranking.

    Args:
        query:     Search query in Chinese or English
        tenant_id: Tenant identifier for data isolation (e.g. "tenant_default")
        course_id: Optional course UUID to restrict search scope
        top_k:     Number of documents to return after reranking (default: 3)

    Returns:
        List of ranked documents, each with:
        - content:            chunk text
        - source_name:        document name and chapter
        - score:              reranker confidence score [0, 1]
        - confidence:         top document confidence score
        - is_high_confidence: whether confidence >= 0.75
    """
    from backend.core.reranker import retrieve
    from backend.core.logger import get_logger
    logger = get_logger(__name__)
    # retrieve() 内部包含BGE-M3编码和CrossEncoder推理
    # 两者都是同步阻塞操作，用run_in_executor避免阻塞async事件循环
    loop = asyncio.get_running_loop() # 获取当前事件循环
    # 阻塞型同步函数交给独立线程执行，从而不阻塞当前事件循环
    ranked_docs, confidence = await loop.run_in_executor(
        None,
        lambda: retrieve(
            query=query,
            tenant_id=tenant_id,
            course_id=course_id,
            rerank_top_k=top_k,
        ),
    )

    logger.info(
        "kb_mcp.search_done",
        query_preview=query[:50],
        hits=len(ranked_docs),
        confidence=round(confidence, 4),
    )

    return [
        {
            "content": doc.content, # 搜索结果内容
            "source_name": doc.metadata.get("source_name", ""), # 结果的来源
            "score": round(doc.score, 6), # 搜索结果分数
            "confidence": round(confidence, 4), # 搜索结果置信度
            "is_high_confidence": confidence >= 0.75, # 搜索结果是否高置信度
        }
        for doc in ranked_docs
    ] # 返回3条结果


# ── 独立运行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    sys.path.insert(0, str(__file__).split("/backend/")[0])
    load_dotenv(".env.local")

    port = int(os.getenv("KB_MCP_SERVER_PORT", "8001"))
    print(f"Knowledge Base MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)
