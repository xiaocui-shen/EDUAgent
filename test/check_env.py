import sys
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from pymilvus import MilvusClient



async def check_postgres():
    DB_URL = ("postgresql+asyncpg://"
              "root:123456"
              "@192.168.88.101:5432"
              "/eduagent")
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        r = await conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL 连通：", r.scalar())
    await engine.dispose()

def check_milvus():
    milvus = MilvusClient(
        f"http://192.168.88.101:19530",
        db_name='itcast',
    )
    print("✅ Milvus 连通，已有集合：", milvus.list_collections())

def check_python():
    print('✅ Python版本：', sys.version[:6])

async def main():
    await check_postgres()
    check_milvus()
    check_python()

asyncio.run(main())
