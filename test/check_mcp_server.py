# test/check_mcp_server.py
# 前提：已运行 或启动了
# C:\......\uv_eduagent\.venv\Scripts\python.exe
# C:\......\06-EduAgent项目\backend\main.py


import asyncio
from backend.mcp_server.client import call_mcp_tool # mcp客户端

async def main():
    # ── 知识库 MCP（路径前缀 /mcp/kb，实际端点 /mcp/kb/mcp）────────
    kb_results = await call_mcp_tool(
        server_url="http://localhost:8000/mcp/kb", # 集成模式：带路径前缀
        tool_name="search_knowledge_base",
        arguments={"query": "milvus索引都有哪些？",
                   "tenant_id": "tenant_default"},
    )
    print(f"[KB] 命中 {len(kb_results)} 条")
    for i, r in enumerate(kb_results, 1):
        print(f"  [{i}] score={r['score']:.4f}  "
              f"{r['source_name']}\n\t{r['content'][:20]}")

    print()

    # ── Web 搜索 MCP（路径前缀 /mcp/search，实际端点 /mcp/search/mcp）─
    search_results = await call_mcp_tool(
        server_url="http://localhost:8000/mcp/search",  # 集成模式：带路径前缀
        tool_name="web_search",
        arguments={"query": "BGE-M3 向量模型", "max_results": 3},
    )
    print(f"[Search] 搜索结果 {len(search_results)} 条")
    for i, r in enumerate(search_results, 1):
        print(f"  [{i}] {r['title']}\n\t{r['content']}")

asyncio.run(main())
