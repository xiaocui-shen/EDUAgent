# backend/api/v1/resume.py

import asyncio
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text

from backend.agents.resume.graph import build_resume_graph
from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal, get_current_user
from backend.langfuse_handler import langfuse_handler

router = APIRouter()
logger = get_logger(__name__)

# ── 线程本地图：每个线程一份独立的图实例，避免并发竞争 ──
import threading
_graph_local = threading.local() # 创建本地线程

def _get_graph():
    """获取线程本地的图实例（当前线程没有就编译一个）。"""
    if not hasattr(_graph_local, "graph"):
        # 判断线程是否有 graph 这个属性，如果没有则创建
        _graph_local.graph = build_resume_graph()
    return _graph_local.graph

# ── GC 保护：模块级集合持有后台任务的强引用，防止被垃圾回收 ──
_background_tasks: set[asyncio.Task] = set()
RESUME_REVIEW_TIMEOUT_SECONDS = 15 * 60 # 审查超时阈值：15 分钟


async def _mark_review_failed(review_id: str, error_msg: str) -> None:
    """把仍处于当前事件循环的任务记录标记为 failed
    processing / done / failed 三种状态（还有一个 404）
    只有processing正在执行的任务才会被标记为failed
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                UPDATE resume_reviews
                SET status = 'failed', error_msg = :error_msg, updated_at = NOW()
                WHERE id = :review_id AND status = 'processing'
            """),
            {"review_id": review_id, "error_msg": error_msg[:1000]},
        )
        await session.commit()


@router.post("/upload", status_code=202) # 202 Accepted：已接受、正在处理
async def upload_resume(
    file: UploadFile = File(...),                # 上传的文件
    current_user: dict = Depends(get_current_user), # 鉴权：拿到当前用户
):
    """上传 PDF 简历，触发异步审查，立即返回 review_id。"""
    if not file.filename.lower().endswith(".pdf"): # 只收 PDF
        raise HTTPException(status_code=400, detail="仅支持 PDF 格式")

    review_id  = str(uuid.uuid4()) # 生成简历审查任务ID
    student_id = current_user["user_id"]
    tenant_id  = current_user["tenant_id"]

    # 1. 读取并校验文件（用 await 读，避免阻塞事件循环）
    MAX_PDF_SIZE = 20 * 1024 * 1024 # 20MB 上限
    file_bytes = await file.read()
    if len(file_bytes) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="文件过大，最大支持 20MB")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # 暂存到临时目录 项目根路径下的 tmp 目录下
    # import tempfile
    # tmp_path = os.path.join(tempfile.gettempdir(),
    #                         f"{review_id}_upload.pdf")
    from backend.config import get_abs_path
    tmp_path = get_abs_path(f"tmp/{review_id}_upload.pdf")

    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    logger.info("upload_resume.file_saved",
                review_id=review_id,
                tmp_path=tmp_path)

    # 2. 写入 resume_reviews 初始记录（status=processing）
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO resume_reviews (id, tenant_id, student_id, pdf_minio_path, status)
                VALUES (:id, :tenant_id, :student_id, :pdf_minio_path, 'processing')
            """),
            {"id": review_id, "tenant_id": tenant_id, "student_id": student_id,
             "pdf_minio_path": f"resumes/{student_id}/{review_id}.pdf"},
        )
        await session.commit()

    # 3. 准备初始 State，后台启动图执行
    initial_state = {
        "messages": [], "student_id": student_id, "tenant_id": tenant_id,
        "review_id": review_id, "pdf_minio_path": "", "pdf_local_path": tmp_path,
        "raw_text": "", "page_count": 0, "structured": None,
        "dimension_scores": [], "weighted_score": 0.0, "issues": [],
        "summary": None, "fallback_used": False, "structured_output": None,
    }

    def _on_task_done(t: asyncio.Task):
        """任务结束回调函数：移除引用、清理临时文件、失败则标记 failed。"""
        _background_tasks.discard(t) # 从集合移除（释放强引用）
        if os.path.exists(tmp_path): # 清理临时 PDF
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        mark_failed_msg = None
        if t.cancelled(): # 任务被取消（如服务重启）
            mark_failed_msg = "审查任务被服务重启中断，请重试。"
            logger.warning("resume.background_task_cancelled",
                           review_id=review_id)
        else:
            exc = t.exception() # 任务抛了异常
            if exc:
                mark_failed_msg = f"审查任务执行失败：{exc}"
                logger.error("resume.background_task_failed",
                             review_id=review_id,
                             error=str(exc),
                             exc_info=exc)
        if mark_failed_msg: # 异步把记录标记为 failed
            try: # 获取当前运行中的事件循环
                loop = asyncio.get_running_loop()
                # 创建异步任务，扔后台就不管了：把任务失败写入数据库
                loop.create_task(_mark_review_failed(
                    review_id, mark_failed_msg))
            except RuntimeError:
                pass # 循环已关闭：下次查询走超时兜底

    graph = _get_graph() # 获取图对象
    config = {
        "configurable": {
            "thread_id": f"student_{student_id}_session_{review_id}",
        },
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": student_id,
            "langfuse_session_id": review_id,
        },
    }
    task = asyncio.create_task(graph.ainvoke(initial_state, config=config)) # 后台执行
    _background_tasks.add(task) # GC 保护：持有强引用
    task.add_done_callback(_on_task_done) # 告诉后台任务：任务完成自动执行回调
    logger.info("upload_resume.task_started", review_id=review_id)

    return { # 立即返回（不等审查完成）
        "review_id": review_id, "status": "processing",
        "message": "简历已上传，正在审查中，预计 30-60 秒完成。",
    }


@router.get("/reviews/{review_id}")
async def get_review(review_id: str, current_user: dict = Depends(get_current_user)):
    """查询审查状态/结果。processing / done / failed / 404。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, scores, issues, summary, error_msg, created_at, updated_at
                FROM resume_reviews
                WHERE id = :review_id AND student_id = :student_id
            """),
            {"review_id": review_id, "student_id": current_user["user_id"]},
        )
        row = result.mappings().fetchone()

    if not row: # 不存在
        raise HTTPException(status_code=404, detail="审查记录不存在")

    # 还在处理中
    if row["status"] == "processing":
        # 超时兜底：防止后台任务中断后长期卡 processing 状态
        last_ts = row["updated_at"] or row["created_at"] # 更新事件获创建时间
        if isinstance(last_ts, datetime): # 时间戳是datetime类型
            # 如果距离创建时间已超过 60 秒，则标记为超时状态
            elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if elapsed >= RESUME_REVIEW_TIMEOUT_SECONDS:
                timeout_msg = "审查任务超时或被中断，请重新上传后重试。"
                # 标记失败写入数据库
                await _mark_review_failed(review_id, timeout_msg)
                return {"review_id": review_id,
                        "status": "failed",
                        "error_msg": timeout_msg}
        return {"review_id": review_id, "status": "processing"}

    # 已经标记为失败
    if row["status"] == "failed":
        return {"review_id": review_id, "status": "failed",
                "error_msg": row.get("error_msg")
                             or "审查任务失败，请重新上传。"}

    # TODO 视图函数的逻辑至此就剩下 done 状态了
    # status == done：JSONB 列已被 asyncpg 自动反序列化为 dict/list，无需 json.loads
    def _to_dict(v):
        # 转成字典或列表
        if v is None:
            return {}
        if isinstance(v, (dict, list)):
            return v
        import json as _json
        return _json.loads(v)
    # 转换成字典
    scores_data = _to_dict(row["scores"])
    return {
        "review_id": review_id, "status": "done",
        "weighted_score": scores_data.get("weighted_score", 0),
        "dimension_scores": scores_data.get("dimension_scores", []),
        "issues":
            _to_dict(row["issues"])
            if not isinstance(row["issues"], list)
            else row["issues"],
        "summary": _to_dict(row["summary"]),
    }


@router.delete("/reviews/{review_id}", status_code=204)
async def delete_review(
        review_id: str, current_user: dict = Depends(get_current_user)):
    """删除简历审查记录（WHERE 带 student_id，只能删自己的）。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                DELETE FROM resume_reviews
                WHERE id = :review_id AND student_id = :student_id
            """),
            {"review_id": review_id, "student_id": current_user["user_id"]},
        )
        await session.commit()
    if result.rowcount == 0: # 没删到任何行 = 不存在/无权限
        raise HTTPException(status_code=404, detail="记录不存在")


@router.get("/reviews")
async def list_reviews(current_user: dict = Depends(get_current_user)):
    """列出本人历史审查记录（摘要，按时间倒序）。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, created_at,
                       (scores::jsonb ->> 'weighted_score')::float AS weighted_score
                FROM resume_reviews
                WHERE student_id = :student_id
                ORDER BY created_at DESC
                LIMIT 50
            """),
            {"student_id": current_user["user_id"]},
        )
        rows = result.mappings().all() # 查询结果转换成列表

    items = [
        {"review_id": row["id"], "status": row["status"],
         "created_at":
             row["created_at"].isoformat()
             if row["created_at"] else None,
         "weighted_score": row["weighted_score"]}
        for row in rows
    ]
    return {"items": items, "total": len(items)}


if __name__ == '__main__':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    from fastapi import FastAPI
    from backend.api.v1.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router)  # 注册auth.py中的路由
    # 注册当前文件中的路由，所有接口都注册到/resume下
    # /upload ➡ /resume/upload
    app.include_router(router, prefix="/resume")

    async def check_resume():
        import httpx, fitz
        from httpx import ASGITransport

        # 造一个测试 PDF
        _d = fitz.open()
        _p = _d.new_page()
        _p.insert_text(
            (50, 60),
            "张三 后端开发\n技能 Java Spring Boot",
            fontsize=11,
            fontname="china-s"
        )
        from backend.config import get_abs_path
        test_pdf_path = get_abs_path("tmp/upload_test.pdf")
        _d.save(test_pdf_path)
        _d.close()

        async with httpx.AsyncClient(
                transport=ASGITransport(app=app), # 本地内存测试
                base_url="http://t") as c: # 所以url地址无所谓真假对错
            tok = (await c.post(
                "/login",
                json={"username": "student01",
                      "password": "Student@123456"}
            )).json()["access_token"] # 访问登录接口获取 Token
            H = {"Authorization": f"Bearer {tok}"} # 创建请求头

            with open(test_pdf_path, "rb") as f: # 打开测试 PDF
                # 上传测试 PDF
                r = await c.post(
                    "/resume/upload",
                    headers=H, # 带上了请求头
                    files={"file":
                        ("resume.pdf", f, "application/pdf")})
            print("① 上传:", r.status_code,
                  "| status:", r.json()["status"])
            rid = r.json()["review_id"] # 获取简历审查记录 ID

            # ② 轮询等待审查完成（最多 60 秒，每 10 秒打印一次进度）
            final_status = None
            for i in range(60):
                r2 = await c.get(f"/resume/reviews/{rid}", headers=H)
                final_status = r2.json()["status"]
                if final_status in ("done", "failed"):
                    break
                if i % 10 == 0:
                    print(f"② 轮询({i}s):", r2.status_code,
                          "| status:", final_status)
                await asyncio.sleep(1)
            else:
                print("② 超时：审查未在 60 秒内完成")
            print("② 审查结果:", r2.status_code,
                  "| status:", final_status,
                  "| 得分:", r2.json().get("weighted_score", "N/A"))

            # 等回调删文件
            await asyncio.sleep(1)

            # 列出自己的全部简历审查记录
            r3 = await c.get("/resume/reviews", headers=H)
            print("③ 列表 total>=1:", r3.json()["total"] >= 1)

            r4 = await c.post(
                "/resume/upload",
                headers=H,
                files={"file": ("x.txt", b"hello", "text/plain")})
            print("④ 非PDF:", r4.status_code, "|", r4.json()["detail"])

            r5 = await c.post("/resume/upload", files={"file": ("r.pdf", b"%PDF", "application/pdf")})
            print("⑤ 无token:", r5.status_code)

            r6 = await c.delete(f"/resume/reviews/{rid}", headers=H)
            print("⑥ 删除:", r6.status_code)

            r7 = await c.get(f"/resume/reviews/{rid}", headers=H)
            print("⑦ 删后查询:", r7.status_code)

    print('➡️ 内测resume的全部接口：')

    async def main():
        await check_resume()
        import uvicorn
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())

    # "username": "student01"
    # "password": "Student@123456"




