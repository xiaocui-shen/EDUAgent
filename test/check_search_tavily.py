

from backend.mcp_server.web_search_server import _search_tavily
import asyncio, os

# tavily的APIkey直接放到了系统环境变量
# print(os.getenv('tavilyApiKey'))
ret = asyncio.run(_search_tavily(
    "如何使用 Python 创建一个 HTTP 服务器？",
    5,
    os.getenv("tavilyApiKey")
))

print(ret)

