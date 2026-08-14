# backend/langfuse_handler.py
# Langfuse v4 可观测性回调处理器（OTEL 协议）
# 配套 Docker 服务端 langfuse/langfuse:3
# 用法：在 LangGraph 的 config["callbacks"] 中传入 langfuse_handler

import os
from dotenv import load_dotenv

# pydantic-settings 只把 .env.local 读到 Settings 对象里，不会注入 os.environ。
# 必须手动 load_dotenv，否则 langfuse get_client() 读不到环境变量。
env_file = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
    ".env.local"
)
load_dotenv(env_file)

from langfuse import get_client
from langfuse.langchain import CallbackHandler

# v4 写法：get_client() 初始化单例，自动读取 LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL
get_client()
langfuse_handler = CallbackHandler()