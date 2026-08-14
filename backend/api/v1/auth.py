# backend/api/v1/auth.py
# 登录认证接口：/login（签发 Token）与 /me（验证鉴权）

import asyncio
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from jose import jwt
from passlib.context import CryptContext

from backend.config import get_settings
from backend.dependencies import get_db, get_current_user
from backend.core.logger import get_logger
logger = get_logger(__name__)

# ── 兼容性补丁：
# passlib 1.7.4 要读 bcrypt.__about__.__version__
# 而 bcrypt>=4 删了它
import bcrypt as _bcrypt_mod, types as _types
if not hasattr(_bcrypt_mod, "__about__"):
    # 获取版本
    _about = _types.SimpleNamespace(
        __version__=getattr(
            _bcrypt_mod, "__version__", "4.x"
        )
    )
    # 注入假的 __about__，让 passlib 能探测到版本
    _bcrypt_mod.__about__ = _about

# 本模块的路由集合
router = APIRouter()

# 密码哈希对象
pwd_context = CryptContext(
    schemes=["bcrypt"], deprecated="auto")

class LoginRequest(BaseModel):
    """登录请求体。"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")
class TokenResponse(BaseModel):
    """登录成功的响应体。"""
    access_token: str                 # JWT 令牌
    token_type:   str = "bearer"      # 令牌类型，固定 bearer
    expires_in:   int                 # 有效期（秒）
    role:         str                 # 用户角色
    user_id:      str                 # 用户ID

def _create_access_token(
        data: dict, expires_minutes: int) -> str:
    """把身份信息 + 过期时间打包，用密钥签名成 JWT 字符串。"""
    settings = get_settings()
    payload = data.copy()  # 拷一份，避免改到原字典
    # 鉴权token的到期时间
    expire = (datetime.now(timezone.utc) +
              timedelta(minutes=expires_minutes))
    payload["exp"] = expire # exp 是 JWT 标准的过期字段
    # 用密钥签名成 JWT，并返回
    return jwt.encode(payload,
                      settings.jwt_secret_key,
                      algorithm=settings.jwt_algorithm)

# 登录接口
@router.post("/login", response_model=TokenResponse)
async def login(
    req: LoginRequest, # 请求体自动解析为 LoginRequest
    db: AsyncSession = Depends(get_db), # 注入数据库会话
):
    """用户登录，返回 JWT Access Token（支持用户名或邮箱登录）"""
    settings = get_settings()

    # 查用户：用户名或邮箱都行（:val 同时匹配两列），参数化查询防注入
    result = await db.execute(
        text("SELECT id, password_hash, "
             "role, tenant_id, is_active "
             "FROM users "
             "WHERE username = :val "
             "OR email = :val LIMIT 1"
        ),
        {"val": req.username},
    )
    row = result.fetchone()

    if not row: # 用户不存在
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误")

    if not row.is_active: # 账号被禁用
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用，请联系管理员")

    # 密码校验是 CPU 密集型（~100ms），用线程池避免阻塞事件循环
    loop = asyncio.get_running_loop()
    password_ok = await loop.run_in_executor(
        None,
        pwd_context.verify, # 密码校验函数
        req.password, # 参数1：明文密码
        row.password_hash # 参数2：数据获取的密文密码
    )

    if not password_ok: # 密码不对
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误")

    # 校验通过，签发 Token（把用户身份装进去）
    token = _create_access_token(
        data={"sub": str(row.id),
              "role": row.role,
              "tenant_id": row.tenant_id},
        expires_minutes=
            settings.jwt_access_token_expire_minutes,
    )
    logger.info("auth.login_success",
                user_id=str(row.id),
                role=row.role)

    return TokenResponse(
        access_token=token,
        expires_in= # 分钟转秒
            settings.jwt_access_token_expire_minutes * 60,
        role=row.role,
        user_id=str(row.id),
    )


@router.get("/me")
async def get_me(
    # 注入当前用户（顺带完成鉴权）
    current_user: dict = Depends(get_current_user),
):
    """获取当前登录用户信息（用于验证 Token 是否有效）。"""
    return current_user


# ── 模块自测：验证密码哈希与 Token 编解码──────────────
if __name__ == "__main__":
    # 测试密码哈希
    def check_pwd_hash():
        '''测试密码哈希'''
        # ① 密码哈希 + 校验
        h = pwd_context.hash("Student@123456")
        print("➡️ 密码哈希:", h)
        print("正确密码校验:",
              pwd_context.verify("Student@123456", h))
        print("错误密码校验:",
              pwd_context.verify("wrong", h))

        # ② Token 签发jwt token + 再从token中解码出用户信息
        s = get_settings()
        tk = _create_access_token(
            data ={"sub": "u-1",
             "role": "student",
             "tenant_id": "tenant_default"},
            expires_minutes=10)
        print("➡️ jwt token:", tk)
        decoded = jwt.decode(
            tk,
            s.jwt_secret_key,
            algorithms=[s.jwt_algorithm]
        )
        print("解码出 sub:", decoded["sub"],
              "| role:", decoded["role"])

    check_pwd_hash() # 运行测试

    # 测试运行/login、/me接口
    from fastapi import FastAPI
    import uvicorn
    app = FastAPI()
    app.include_router(router) # 添加路由
    # 启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)

    '''浏览器访问 http://127.0.0.1:8000/docs，测试四个场景
    student01
    Student@123456
    
    ① 访问 /me 没Token，说明没有登录，返回 403；
    ② 访问 /login，但使用错误账号密码，返回 401；
    ③ 访问 /login，使用正确账号密码，返回 200 和 JWT Token；
    ④ 访问 /me 带上 Token 访问受保护接口，返回 200；    
    '''

