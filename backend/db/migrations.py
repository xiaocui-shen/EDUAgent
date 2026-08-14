# backend/db/migrations.py
from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 所有补丁，按时间顺序追加。SQL 必须幂等（带 IF NOT EXISTS）
_MIGRATIONS: list[tuple[str, str]] = [
    (
        "exam_submissions.weak_points",
        "ALTER TABLE exam_submissions ADD COLUMN "
        "IF NOT EXISTS weak_points JSONB",
    ),
    (
        "exam_reviews.knowledge_tag",
        "ALTER TABLE exam_reviews ADD COLUMN "
        "IF NOT EXISTS knowledge_tag VARCHAR(128)",
    ),
    (
        "idx_exam_submissions_student_created",
        "CREATE INDEX IF NOT EXISTS "
        "idx_exam_submissions_student_created "
        "ON exam_submissions (student_id, created_at DESC)",
    ),
    # ……
    # TODO 每次给 init_db.sql 加字段，就在这里同步追加一条
]


async def run_migrations() -> None:
    """应用启动时执行所有 Schema 补丁；单条失败只警告，不阻断启动。"""
    async with AsyncSessionLocal() as session:
        for desc, sql in _MIGRATIONS:
            try:
                await session.execute(text(sql))
                await session.commit()
            except Exception as e:
                await session.rollback()
                if "already exists" not in str(e):
                    logger.warning(
                        "db.migration_failed",
                        column=desc,
                        error=str(e)
                    )
    logger.info("db.migrations_done", count=len(_MIGRATIONS))


if __name__ == '__main__':
    # 运行测试
    from backend.core.logger import configure_logging
    configure_logging() # 测试运行临时让全局日志生效
    import asyncio
    asyncio.run(run_migrations())