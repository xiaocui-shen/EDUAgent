# scripts/seed_data.py
# 执行：python scripts/seed_data.py
# 用途：灌入本地开发测试账号

import asyncio
import uuid
import os
# python-dotenv：从 .env 文件加载环境变量
from dotenv import load_dotenv
import asyncpg # PostgreSQL异步驱动（脚本直接用它，简单直接）
from passlib.context import CryptContext

# 兼容性补丁：同 auth.py，让 passlib 能读到 bcrypt 版本
import bcrypt as _b, types as _t
if not hasattr(_b, "__about__"):
    _b.__about__ = _t.SimpleNamespace(__version__=getattr(_b, "__version__", "4.x"))

# 把.env.local配置项添加到环境变量中
def get_project_root() -> str:
    """获取工程根目录"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)
    return project_root  # 返回工程根目录
def get_abs_path(relative_path: str) -> str:
    """将工程内的相对路径转为绝对路径 :return: 绝对路径"""
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)
env_file = get_abs_path(".env.local")
load_dotenv(env_file) # TODO 配置加入临时环境变量

# 用环境变量拼出 asyncpg 的连接串
# （注意 asyncpg 用的是 postgresql:// 而非 +asyncpg）
DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5433)}"
    f"/{os.getenv('DB_NAME', 'eduagent')}"
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
TENANT_ID = "tenant_default"

async def seed_users():
    """灌入 4 个测试账号（已存在则跳过）。"""
    conn = await asyncpg.connect(DB_DSN) # 连接数据库
    print("✅ 数据库连接成功，开始灌入测试账号...")
    try:
        users = [
            {"username": "admin",
             "email": "admin@eduagent.local",
             "pwd": "Admin@123456",
             "role": "admin"},
            {"username": "teacher01",
             "email": "teacher01@eduagent.local",
             "pwd": "Teacher@123456",
             "role": "teacher"},
            {"username": "student01",
             "email": "student01@eduagent.local",
             "pwd": "Student@123456",
             "role": "student"},
            {"username": "student02",
             "email": "student02@eduagent.local",
             "pwd": "Student@123456",
             "role": "student"},
        ]
        for u in users:
            await conn.execute(
                """
                INSERT INTO users (id, tenant_id, username, email, password_hash, role)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (tenant_id, email) DO NOTHING
                """,
                str(uuid.uuid4()),
                TENANT_ID,
                u["username"],
                u["email"],
                pwd_context.hash(u["pwd"]), # 存哈希，绝不存明文
                u["role"],
            )
        print(f"✅测试账号灌入{len(users)} 个，已存在则跳过：")
        print("   admin@eduagent.local      / Admin@123456")
        print("   teacher01@eduagent.local  / Teacher@123456")
        print("   student01@eduagent.local  / Student@123456")
        print("   student02@eduagent.local  / Student@123456")
    finally:
        await conn.close() # 无论成败都关闭连接


if __name__ == "__main__":
    asyncio.run(seed_users())
