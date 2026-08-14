# backend/api/router.py
# API 路由总入口

from fastapi import APIRouter
from backend.api.v1 import ( # 六个子模块
    auth, qa, exam, resume, interview, unified_chat)

api_router = APIRouter() # 总路由对象

# 把每个子 router的路由前缀 和 标签 都聚合进来
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["认证"]
) # 第 3 章
api_router.include_router(
    unified_chat.router,
    prefix="/chat",
    tags=["AI助手"]
)   # 第 8 章 统一入口
api_router.include_router(
    qa.router,
    prefix="/qa",
    tags=["智能问答"]
)  # 第 5 章
api_router.include_router(
    exam.router,
    prefix="/exam",
    tags=["试卷批改"]
)  # 第 6 章
api_router.include_router(
    resume.router,
    prefix="/resume",
    tags=["简历审查"]
)  # 第 4 章
api_router.include_router(
    interview.router,
    prefix="/interview",
    tags=["模拟面试"]
)  # 第 7 章
