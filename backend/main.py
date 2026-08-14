# backend/main.py

# Windows conda坑 专用：把base的Library/bin加入DLL搜索路径，
# 让 _lzma.pyd 能找到 liblzma.dll（非 Windows conda环境跳过）
'''
import os
import sys
if sys.platform == "win32":
    _conda_base_lib_bin = os.path.normpath(
        os.path.join(os.path.dirname(sys.executable), "..", "..", "Library", "bin")
    )
    if os.path.isdir(_conda_base_lib_bin):
        os.add_dll_directory(_conda_base_lib_bin)
'''

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
settings = get_settings() # 配置（第 3 章）
# 日志（第 3 章）
from backend.core.logger import configure_logging, get_logger
from backend.api.router import api_router # 8.6.1 聚合的总路由

# MCP Sub-Apps（第 5.11 章实现）在 lifespan 定义前创建
# 这样 lifespan 与下面的 mount 共享同一实例
# 知识库 MCP（5.11）
from backend.mcp_server.knowledge_base_server import (
    mcp as kb_mcp)
# 联网搜索 MCP（5.11）
from backend.mcp_server.web_search_server import (
    mcp as ws_mcp)

# 把 MCP 变成 ASGI 子应用对象
_kb_app = kb_mcp.streamable_http_app()
_ws_app = ws_mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging() # 初始化结构化日志，让日志生效（第3章）
    logger = get_logger(__name__)
    logger.info("app.starting | env=%s port=%s",
                settings.app_env, settings.app_port)

    # ① DB Schema 自动迁移（幂等，每次启动执行）—— 第 3 章的建表/迁移
    try:
        from backend.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning( # 迁移失败只告警，不拦启动
            "app.migrations_failed | error=%s", e)

    # ② 并行预热三个本地模型（首次加载慢，提前热好，避免首个请求卡顿）
    import asyncio
    try:
        from backend.core.reranker import BGEReranker # 重排序模型
        from backend.core.query_classifier import (
            QueryClassifier) # 意图分类器（5.9）
        from backend.core.knowledge_base import (
            BGEMEmbedder) # BGE-M3 嵌入（5.4）
        # 获取当前正在运行的事件循环对象
        loop = asyncio.get_running_loop()
        await asyncio.gather(                                    # 三个模型并行加载（各跑在线程池里）
            loop.run_in_executor(
                None, BGEReranker.get_instance),
            loop.run_in_executor(
                None, QueryClassifier.get_instance),
            loop.run_in_executor(
                None, BGEMEmbedder.get_instance),
        )
        logger.info("app.local_models_warmed_up")
    except Exception as e:
        logger.warning(
            "app.local_models_warmup_failed | error=%s",
            e)  # 预热失败也不拦启动

    # ③ 驱动 MCP Server 的 lifespan
    # mount 不会自动调用子应用的 lifespan，需手动嵌套进入
    async with _kb_app.router.lifespan_context(_kb_app):
        async with _ws_app.router.lifespan_context(_ws_app):
            logger.info("app.started")
            yield # TODO ← 应用运行期间停在这里

            # ── fastapi app 关闭时执行 ──
            logger.info("app.shutting_down")
            # LLM 工厂 清缓存
            from backend.core.llm_factory import LLMFactory
            LLMFactory.clear_cache()
            logger.info("app.shutdown_complete")



app = FastAPI(
    title="EduAgent API",
    description="AI 助教智能体系统 API",
    version="1.0.0",
    docs_url="/docs", # Swagger 文档
    redoc_url="/redoc", # ReDoc 文档
    lifespan=lifespan, # 挂上面的生命周期钩子 加入mcp
)

# CORS：允许前端开发端口跨域访问（3000/5173/8080）
app.add_middleware(
    CORSMiddleware, # CORS跨域中间件
    # 允许的跨域的域名和端口
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True, # 允许携带 Cookie
    allow_methods=["*"], # 允许所有 HTTP 方法
    allow_headers=["*"], # 允许所有 HTTP 头
)
'''【关于CORS跨域】
浏览器有个安全规矩——A 网站的 JS 不许偷偷调 B 网站的接口。
CORS 就是 B 网站说"我允许 A 来调"的白名单。
1. 为啥要有这规矩？
    没有的话，你打开一个恶意网页，它的 JS 就能偷偷调你银行接口
    利用你访问银行网站存留在浏览器的Cookie
    浏览器为了防这事，默认禁止跨域调用。
2. 什么算"跨域"？
    协议、域名、端口，任何一个不一样就算跨域。
    localhost:3000（前端）调 localhost:8000（后端），端口不同，跨域。
3. 怎么解决？
    后端加 CORS 中间件，白名单里写上前端地址。
    浏览器一看"哦，后端同意了"，放行。  
4. 前后端分离就一定会跨域，加中间件配白名单是固定操作，照抄就行。
    不用深究浏览器预检请求那些细节。
5. 正式上线咋办？白名单写死你的前端域名！
    allow_origins=[
        "https://eduagent.cn",          # 用户直接输域名访问 80端口
        "https://www.eduagent.cn",      # 用户带 www 访问 80端口
        "https://app.eduagent.cn",      # 前端部署在子域名 80端口
        "http://47.96.123.45:3000",     # 前端跑在服务器3000端口
        "http://47.96.123.45",          # 前端用nginx代理到80端口
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],

6. 本地开发时就是之前那些 localhost:3000、localhost:5173。
    上线后换成交付给用户时他们实际访问的地址。
'''


# 挂总路由（8.6.1），所有业务接口在 /api/v1 下
app.include_router(api_router, prefix="/api/v1")

# 挂知识库 MCP 子应用（5.11）
app.mount("/mcp/kb", _kb_app)
# 挂联网搜索 MCP 子应用（5.11）
app.mount("/mcp/search", _ws_app)

@app.get("/health", tags=["系统"]) # 健康检查（运维探活用）
async def health_check():
    '''
    "env": settings.app_env 来源于配置
        local — 本地开发
        development — 本地开发
        staging — 测试/预发环境
        production — 正式生产
    上线时 .env配置文件里写 APP_ENV=production 就行。
    健康检查返回它，运维一看就知道"这台机器跑的是生产环境"
    不会跟测试机搞混。
    '''
    return {"status": "ok", "env": settings.app_env}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # 如果在终端启动，进入项目路径下，输入命令：
    # uvicorn backend.main:app --host 0.0.0.0 --port 8000