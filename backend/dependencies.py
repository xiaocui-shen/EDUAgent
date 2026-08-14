# backend/dependencies.py（数据库部分）
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
# 配置项
from backend.config import get_settings
settings = get_settings()

# ── 创建异步引擎（连接池）──
engine = create_async_engine(
    settings.database_url, # 来自 config.py，最终来自 .env.local
    pool_size=10, # 连接池基础大小
    max_overflow=20, # 高峰时最多再额外开 20 个连接
    pool_pre_ping=True,  # 取连接前先 ping 一下，死了自动重建
    echo=False, # True 会打印所有 SQL，调试时可临时打开
)

# ── 会话工厂 ──
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：获取异步数据库会话，自动提交 / 回滚 / 关闭"""
    async with AsyncSessionLocal() as session:
        try:
            yield session # 把会话交给接口使用
            await session.commit() # 接口正常结束 → 自动提交
        except Exception:
            await session.rollback() # 出错 → 自动回滚
            raise


# ── JWT 鉴权 ──────────────────────────────────────────────
# 解析 Authorization: Bearer 头
from fastapi.security import (HTTPBearer,
                              HTTPAuthorizationCredentials)
from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt # python-jose：JWT 的编解码库

# FastAPI 安全方案：
#   自动从请求头解析 "Authorization: Bearer <token>"
bearer_scheme = HTTPBearer()

async def get_current_user(
    # Depends(bearer_scheme)：
        # FastAPI 自动取出 Bearer Token；没带或格式错会直接 401
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme),
) -> dict:
    """FastAPI 依赖：验证 JWT Token，返回当前用户信息。
    返回 {"user_id": str, "role": str, "tenant_id": str}；
    Token 无效则抛 401。"""
    # 预先准备好「401 凭证无效」异常，多处复用
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 用密钥和算法解码 Token；若签名不对/过期，会抛 JWTError
        payload = jwt.decode(
            credentials.credentials, # 实际的 token 字符串
            settings.jwt_secret_key, # 验签密钥（和签发时同一个）
            algorithms=[settings.jwt_algorithm], # HS256加密算法
        )
        user_id: str = payload.get("sub") # 标准字段 sub=用户ID
        role: str = payload.get("role", "student") # 角色
        tenant_id: str = payload.get(
            "tenant_id", settings.default_tenant_id) # 租户

        if not user_id: # Token 里没有用户ID，视为无效
            raise credentials_exception

    except JWTError:  # 解码失败（签名错/过期等）
        raise credentials_exception

    return {"user_id": user_id,
            "role": role,
            "tenant_id": tenant_id}
# ── JWT 鉴权 内容结束 ───────────────────────────────────────

if __name__ == '__main__':
    # 运行测试
    # 原有数据库测试逻辑
    import asyncio
    from sqlalchemy import text
    async def main():
        # 模拟 FastAPI 调用依赖：只能迭代一次拿到 session，进行查询
        # 只能迭代一次的原因是 get_db 全程只执行了一次yield
        async for db in get_db():
            r = await db.execute(text(
                "SELECT current_database(), count(*) "
                "FROM information_schema.tables "
                "WHERE table_schema = :s"
                ),
                {"s": "public"}
            )
            row = r.fetchone()
            print("get_db 连到库:", row[0])
            print("在public中，表的数量:", row[1])


    asyncio.run(main())
    print("get_db 测试通过 ✅")

    # ========== 新增：get_current_user 鉴权测试 ==========
    import asyncio
    from datetime import datetime, timedelta
    from jose import jwt as jose_jwt
    print("\n=== 开始测试JWT鉴权依赖 ===")

    # 1. 手动生成测试Token（和登录签发逻辑一致）
    test_payload = {
        "sub": "user_001",
        "role": "admin",
        "tenant_id": "tenant_default",
        "exp": datetime.utcnow() + timedelta(minutes=120)
    }
    test_token = jose_jwt.encode(
        test_payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    print(f"生成测试Token:\n{test_token}\n")

    # 2. 模拟 HTTPBearer 传入token，调用鉴权函数
    class MockCredentials:
        credentials = test_token

    try:
        user_info = asyncio.run(get_current_user(
            MockCredentials()))
        print("✅ Token解析成功，当前用户信息：")
        print(user_info)
    except HTTPException as e:
        print("❌ Token校验失败：", e.detail)

    # 3. 测试非法Token（预期抛出401异常）
    print("\n=== 测试非法Token ===")
    class BadMockCredentials:
        credentials = "fake.token.123"
    try:
        asyncio.run(get_current_user(BadMockCredentials()))
    except HTTPException as e:
        print("预期捕获异常，detail：", e.detail)
