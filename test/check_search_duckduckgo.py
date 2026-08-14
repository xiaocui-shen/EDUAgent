


from backend.mcp_server.web_search_server import _search_duckduckgo
import asyncio

ret = asyncio.run(_search_duckduckgo(
    "如何使用 Python 创建一个 HTTP 服务器？", 5
))
print(ret)