# backend/api/v1/exam.py
# ═══════════════════════════════════════════════════════
# 这个文件是「考试批改Agent」的 API 层（FastAPI 路由）
# 相当于整个批改流程的「前台接待」：
#   学员提交试卷 → 这里接
#   学员查成绩   → 这里接
#   教师看待审   → 这里接
#   教师确认发布 → 这里接
#
# 它本身不做批改，批改是 LangGraph 图干的活。
# 这个文件只负责：收请求 → 启动图 → 返回结果
# ═══════════════════════════════════════════════════════

import asyncio
import os
import tempfile
import uuid

from fastapi import (APIRouter, Depends, UploadFile,
                     File, Form, HTTPException, status)
from pydantic import BaseModel
# Command：LangGraph 提供的「恢复指令」
# 教师确认时用它告诉图：
# 「从 interrupt() 暂停处继续跑」
from langgraph.types import Command
from sqlalchemy import text

from backend.agents.exam.graph import build_exam_graph
from backend.dependencies import (
    get_current_user, AsyncSessionLocal)
from backend.core.memory import build_config
from backend.core.logger import get_logger
from backend.langfuse_handler import langfuse_handler

router = APIRouter()
logger = get_logger(__name__)

# ── 模块级编译图 ──
# build_exam_graph() 会编译整个 LangGraph 状态图
# 编译一次就够了，后面所有请求共用同一个图实例
# 如果放在函数里，每次请求都重新编译，浪费性能
_graph = build_exam_graph()

# ── 后台任务引用集合 ──
# asyncio.create_task() 创建的 Task 如果没有被引用，
# Python 的垃圾回收器(GC)可能把它回收掉，任务就没了
# 所以用一个 set 持有所有正在跑的 Task 引用
# 任务完成后从 set 移除（见 _on_task_done）
_background_tasks: set[asyncio.Task] = set()


# ───────────────────────────────────────────────────────
# POST /submit 学员提交试卷
# ───────────────────────────────────────────────────────
# 整体流程（8步）：
#   1. 校验文件格式（必须 .docx）
#   2. 保存文件到临时目录
#   3. 校验试卷ID是否存在
#   4. 检查是否重复提交
#   5. 写入提交记录（status=ai_processing）
#   6. 构建 LangGraph 运行配置（thread_id）
#   7. 构建图的初始 State
#   8. 启动后台任务跑图，立即返回 202
# ───────────────────────────────────────────────────────
@router.post("/submit", status_code=202)
async def submit_exam(
    exam_id: str      = Form(...),
    file:    UploadFile = File(...),
    # Depends：FastAPI 依赖注入
    # 自动从请求头解析当前登录用户
    current_user: dict = Depends(get_current_user),
):
    """
    学员提交作答 Word 试卷，触发 AI 三轨批改。
    立即返回 202 和 submission_id，
    批改在后台异步完成。
    """
    # ── 第1步：校验文件格式 ──
    # 只接受 .docx，其他格式直接拒绝
    if not (file.filename or "").endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持 .docx 格式",
        )

    # ── 第2步：生成唯一ID + 保存文件 ──
    # submission_id：这次提交的唯一标识
    # 贯穿整个流程（DB记录、图存档、临时文件名）
    submission_id = str(uuid.uuid4())
    student_id    = current_user["user_id"]
    # 临时文件路径：系统临时目录 + submission_id.docx
    tmp_path      = os.path.join(
        tempfile.gettempdir(),
        f"{submission_id}.docx")

    # 把上传的文件内容读出来，写到临时目录
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    # ── 第3步：验证试卷 ID 是否存在 ──
    # 去 exams 表查，确认这张试卷真的存在
    # 还要匹配 tenant_id（多租户隔离）
    # A机构看不到B机构的卷
    async with AsyncSessionLocal() as session:
        exam_row = (await session.execute(
            text("SELECT id FROM exams "
                 "WHERE id = :exam_id "
                 "AND tenant_id = :tenant_id"),
            {"exam_id": exam_id,
             "tenant_id": current_user["tenant_id"]},
        )).fetchone()

    # 查不到 → 试卷不存在，报错
    if not exam_row:
        # 顺手把刚保存的临时文件删了（别留垃圾）
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"试卷 ID 不存在（{exam_id}）。"
                "本地开发环境请先运行 "
                "python scripts/seed_data.py。"
            ),
        )

    # ── 第4步：检查是否重复提交 ──
    # 规则：同一学员 + 同一试卷，只能有一份有效提交
    existing_id: str | None = None
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("""
                SELECT id, status
                FROM exam_submissions
                WHERE exam_id = :exam_id
                  AND student_id = :student_id
            """),
            {"exam_id": exam_id,
             "student_id": student_id},
        )).fetchone()

    if row:
        _existing_id, _existing_status = (
            str(row[0]), row[1])
        # 旧记录已到「待确认/已审/已发布」阶段
        # 说明批改结果已经出来了，不允许重新提交
        if _existing_status in (
                "pending_review",
                "reviewed", "published"):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="该试卷已有待确认或已发布的"
                       "批改结果，无法重新提交。",
            )
        # 旧记录还在 ai_processing 或 submitted
        # 说明上次批改没跑完，允许重提（先删旧）
        existing_id = _existing_id

    # ── 第5步：写入提交记录 ──
    # 开事务：要么全成功，要么全回滚
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # 如果有旧记录，先删掉
            if existing_id:
                await session.execute(
                    text("DELETE FROM exam_submissions "
                         "WHERE id = :id"),
                    {"id": existing_id},
                )
            # 插入新记录，状态设为 ai_processing
            # 表示「AI 正在批改中」
            await session.execute(
                text("""
                    INSERT INTO exam_submissions
                        (id, tenant_id, exam_id,
                         student_id, status,
                         submitted_at)
                    VALUES
                        (:id, :tenant_id, :exam_id,
                         :student_id,
                         'ai_processing', NOW())
                """),
                {
                    "id":         submission_id,
                    "tenant_id":
                        current_user["tenant_id"],
                    "exam_id":    exam_id,
                    "student_id": student_id,
                },
            )

    # ── 第6步：构建 LangGraph 运行配置 ──
    # build_config 会生成 thread_id（图的「存档编号」）
    # interrupt() 暂停时，图的状态存在这个编号下
    # 教师确认时靠这个编号找回存档、恢复执行
    config = build_config(
        current_user["user_id"], submission_id)

    # ── 第7步：构建图的初始 State ──
    # 这就是 ExamState 的初始值
    # 图从第一个节点开始，逐步往里面填数据
    initial_state = {
        "messages":           [],
        "student_id":
            current_user["user_id"],
        "tenant_id":
            current_user["tenant_id"],
        "session_id":         submission_id,
        "exam_id":            exam_id,
        "submission_id":      submission_id,
        # Word 文件的临时路径
        # parse_word_node 会读它
        "word_file_path":     tmp_path,
        # 下面这些初始都是空的
        # 各节点跑完后逐步填充
        "parsed_questions":   [],
        "objective_results":  [],
        "subjective_results": [],
        "code_results":       [],
        "pre_review_summary": {},
        "weak_points":        [],
        "weak_points_summary": "",
        "teacher_decision":   None,
        "final_results":      [],
        "structured_output":  None,
        "fallback_used":      False,
        "teacher_notified":   False,
        "published":          False,
    }

    # ── 后台任务的「完成回调」──
    # 不管批改成功还是失败，这个函数都会被调用
    def _on_task_done(t: asyncio.Task):
        # 从 set 移除引用，允许 GC 回收
        _background_tasks.discard(t)
        # 判断是否失败：没被取消 且 有异常
        task_failed = (
            not t.cancelled()
            and t.exception() is not None
        )

        if task_failed:
            logger.error(
                "exam.background_task_failed",
                submission_id=submission_id,
                error=str(t.exception()),
                exc_info=t.exception(),
            )

        # 清理工作（异步的，所以再包一层 Task）
        async def _cleanup():
            if task_failed:
                # 批改失败 → 状态回滚为 submitted
                # 这样学员可以重新提交
                # WHERE 里加了 status='ai_processing'
                # 防止误改已被教师确认的记录
                try:
                    async with AsyncSessionLocal() \
                            as db_sess:
                        async with db_sess.begin():
                            await db_sess.execute(
                                text("""
                                    UPDATE
                                      exam_submissions
                                    SET status =
                                        'submitted',
                                        updated_at =
                                        NOW()
                                    WHERE id = :sid
                                      AND status =
                                      'ai_processing'
                                """),
                                {"sid": submission_id},
                            )
                except Exception as db_err:
                    logger.warning(
                        "exam.status_reset_failed",
                        submission_id=submission_id,
                        error=str(db_err),
                    )
            # 无论成败，都清理临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # 清理工作也做成 Task，同样持有引用防 GC
        cleanup_task = asyncio.ensure_future(
            _cleanup())
        _background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(
            _background_tasks.discard)

    # ── 第8步：启动后台批改任务 ──
    # asyncio.create_task：把图的执行丢到后台跑
    # 当前请求不用等它跑完，立即返回 202
    #
    # 关键：图跑到 teacher_review_node 时
    # 会 interrupt()，ainvoke 自动返回（不是报错）
    # 此时 Task 完成 → _on_task_done 被调用
    # 图的状态存在 MemorySaver 里
    # 等教师确认后通过 /confirm 接口恢复
    task = asyncio.create_task(
        _graph.ainvoke(initial_state, config=config))
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)

    logger.info(
        "exam.submitted",
        submission_id=submission_id,
        student_id=current_user["user_id"],
        exam_id=exam_id,
    )

    # 立即返回 202，告诉学员「收到了，正在批改」
    # 学员后续通过 GET /my-submissions/{id} 轮询
    return {
        "submission_id": submission_id,
        "status":        "ai_processing",
        "message":
            "试卷已提交，AI 正在批改中，"
            "完成后等待教师确认。",
    }


# ───────────────────────────────────────────────────────
# GET /my-submissions  学员查询接口
# ───────────────────────────────────────────────────────
# 学员看自己提交过哪些试卷、当前状态是什么
# 只返回摘要信息（不返回逐题详情）
# 前端列表页用这个接口
# ───────────────────────────────────────────────────────
@router.get("/my-submissions")
async def list_my_submissions(
    current_user: dict = Depends(get_current_user),
):
    """学员查询自己所有的试卷提交记录"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT es.id AS submission_id,
                       e.title AS exam_title,
                       es.exam_id, es.status,
                       es.submitted_at
                FROM exam_submissions es
                LEFT JOIN exams e
                  ON e.id = es.exam_id
                WHERE es.student_id = :student_id
                ORDER BY es.submitted_at DESC
                LIMIT 20
            """),
            {"student_id": current_user["user_id"]},
        )
        # .mappings()：把每行变成字典
        # （可以用列名取值，而不是下标）
        # .all()：取所有行
        rows = result.mappings().all()

    return {
        "items": [
            {
                "submission_id":
                    str(r["submission_id"]),
                "exam_id": str(r["exam_id"]),
                "exam_title":
                    r["exam_title"] or "",
                "status": r["status"],
                "submitted_at":
                    r["submitted_at"].isoformat()
                    if r["submitted_at"] else None,
            }
            for r in rows
        ]
    }


# ───────────────────────────────────────────────────────
# GET /my-submissions/{submission_id}  学员查详情
# ───────────────────────────────────────────────────────
# 两种情况：
#   未发布（ai_processing / pending_review）
#     → 只返回状态，让学员继续等（前端轮询）
#   已发布（published）
#     → 返回完整批改结果（逐题得分 + 薄弱点）
#
# 为什么未发布不返回详情？
#   因为 AI 批改结果存在 MemorySaver 里
#   还没写入数据库，学员不应该看到未确认的分数
#   必须等教师确认发布后，学员才能看
# ───────────────────────────────────────────────────────
@router.get("/my-submissions/{submission_id}")
async def get_my_submission(
    submission_id: str,
    current_user:  dict = Depends(get_current_user),
):
    """
    学员查询批改状态与结果。
    - 未发布：只返回状态，等待
    - 已发布：返回完整批改结果
    """
    # 先查提交记录的基本信息
    # WHERE 里同时匹配 student_id
    # 防止学员A查学员B的成绩（权限隔离）
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, weak_points,
                       weak_points_summary
                FROM exam_submissions
                WHERE id = :sid
                  AND student_id = :student_id
            """),
            {"sid": submission_id,
             "student_id": current_user["user_id"]},
        )
        row = result.mappings().fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="提交记录不存在")

    # 未发布：只返回状态，让前端继续轮询
    if row["status"] != "published":
        return {
            "submission_id": submission_id,
            "status": row["status"],
        }

    # ── 已发布：从 exam_reviews 读逐题批改结果 ──
    # exam_reviews 是 publish_results_node 写入的
    # JOIN questions 表：拿题号、题目内容、满分等
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT er.question_id,
                       er.question_type,
                       er.knowledge_tag,
                       er.student_answer,
                       er.ai_score,
                       er.ai_feedback,
                       er.teacher_score,
                       er.teacher_comment,
                       er.final_score,
                       er.needs_review,
                       er.ai_raw_result,
                       q.question_no,
                       q.content,
                       q.correct_answer,
                       q.score AS full_score
                FROM exam_reviews er
                JOIN questions q
                  ON q.id = er.question_id
                WHERE er.submission_id = :sid
                ORDER BY q.question_no
            """),
            {"sid": submission_id},
        )
        reviews = result.mappings().all()

    # ── 组装返回数据 ──
    by_question      = []
    total_score      = 0
    full_score_total = 0

    for r in reviews:
        import json as _json
        # ai_raw_result 存的是 JSON 字符串
        # 需要解析成字典才能取里面的字段
        raw = r["ai_raw_result"] or {}
        if isinstance(raw, str):
            try:
                raw = _json.loads(raw)
            except Exception:
                raw = {}
        # final_score：教师改过用教师的，没改用AI的
        final = (
            r["final_score"]
            if r["final_score"] is not None
            else r["ai_score"]
        )
        by_question.append({
            "question_id":
                str(r["question_id"]),
            "question_no": r["question_no"],
            "question_type": r["question_type"],
            "full_score": r["full_score"],
            "score": r["ai_score"],
            "final_score": final,
            "student_answer":
                r["student_answer"] or "",
            "correct_answer":
                r["correct_answer"] or "",
            "ai_feedback":
                r["ai_feedback"] or "",
            "teacher_comment":
                r["teacher_comment"] or "",
            "needs_review": r["needs_review"],
            # 下面两个从 ai_raw_result JSON 里取
            # point_results：每个得分点的命中情况
            "point_results":
                raw.get("point_results", []),
            # quality_feedback：答案质量评价
            "quality_feedback":
                raw.get("quality_feedback", []),
        })
        total_score      += final or 0
        full_score_total += r["full_score"] or 0

    # 薄弱点：可能是 list 也可能是 None
    weak_points = (
        row["weak_points"]
        if isinstance(row["weak_points"], list)
        else []
    )
    weak_points_summary = (
        row["weak_points_summary"] or ""
    )

    return {
        "submission_id": submission_id,
        "status":        "published",
        "pre_review_summary": {
            "total_score": total_score,
            "full_score":  full_score_total,
            "score_rate":
                round(total_score
                      / full_score_total, 4)
                if full_score_total else 0,
            "by_question": by_question,
        },
        "weak_points":         weak_points,
        "weak_points_summary": weak_points_summary,
    }


# ───────────────────────────────────────────────────────
# GET /pending-reviews 教师接口
# ───────────────────────────────────────────────────────
# 教师看「有哪些试卷等我确认」
# 数据来源有两部分：
#   1. 数据库 exam_submissions
#      （status=pending_review 的记录）
#   2. MemorySaver（图暂停时保存的 State）
#      → AI 预批改的分数、薄弱点等
#
# 为什么要从 MemorySaver 读？
#   因为图在 teacher_review_node 处 interrupt() 了
#   此时批改结果还没写入数据库
#   （要等教师确认后才写）
#   所以只能从图的「存档」里读
# ───────────────────────────────────────────────────────
@router.get("/pending-reviews")
async def get_pending_reviews(
    current_user: dict = Depends(get_current_user),
):
    """教师获取所有待确认的提交"""
    # 第1部分：从数据库查待确认的提交列表
    # JOIN users：拿学员姓名
    # JOIN exams：拿试卷标题
    # tenant_id 过滤：教师只能看自己机构的
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT
                    es.id AS submission_id,
                    es.student_id,
                    u.username AS student_name,
                    e.title    AS exam_title,
                    es.submitted_at
                FROM exam_submissions es
                JOIN users u
                  ON u.id = es.student_id
                JOIN exams e
                  ON e.id = es.exam_id
                WHERE es.tenant_id = :tenant_id
                  AND es.status = 'pending_review'
                ORDER BY es.submitted_at DESC
            """),
            {"tenant_id":
                current_user["tenant_id"]},
        )
        rows = result.mappings().all()

    items = []
    for row in rows:
        submission_id = str(row["submission_id"])
        # 默认值：万一 MemorySaver 读不到
        # 至少不报错，列表还能显示
        pre_review = {
            "total_score": 0,
            "full_score": 0,
            "needs_review_count": 0,
        }
        weak_points = []

        # 第2部分：从 MemorySaver 读 AI 预批改结果
        # 图在 interrupt() 时把 State 存在了
        # MemorySaver 里，这里通过 thread_id
        # 找到存档，读出 State
        try:
            thread_id = await _get_thread_id(
                submission_id)
            config = {
                "configurable":
                    {"thread_id": thread_id},
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": current_user["user_id"],
                    "langfuse_session_id": submission_id,
                },
            }
            # aget_state：读取图在某 thread 的
            # 最新快照（就是 interrupt 时的 State）
            snapshot = await _graph.aget_state(
                config)
            if snapshot and snapshot.values:
                sv = snapshot.values
                summary = sv.get(
                    "pre_review_summary", {})
                pre_review = {
                    "total_score":
                        summary.get(
                            "total_score", 0),
                    "full_score":
                        summary.get(
                            "full_score", 0),
                    "needs_review_count":
                        summary.get(
                            "needs_review_count",
                            0),
                }
                weak_points = sv.get(
                    "weak_points", [])
        except Exception as _e:
            # 读不到也不影响列表展示
            # 只是没有分数预览而已
            logger.warning(
                "exam.pending_review_state_read"
                "_failed",
                submission_id=submission_id,
                error=str(_e))

        items.append({
            "submission_id": submission_id,
            "student_name":
                row["student_name"],
            "exam_title": row["exam_title"],
            "submitted_at":
                row["submitted_at"].isoformat()
                if row["submitted_at"] else None,
            "pre_review": pre_review,
            # 列表页只显示前3条薄弱点
            # 详情点进去看全部
            "weak_points": weak_points[:3],
        })

    return {
        "items": items,
        "total": len(items),
    }


# ───────────────────────────────────────────────────────
# GET /submissions/{id}/review 教师看详情
# ───────────────────────────────────────────────────────
# 教师点进某份试卷，看 AI 预批改的完整结果
# 数据全部从 MemorySaver 读（图暂停时的 State）
# 包含：逐题预批改 + 薄弱点 + 薄弱点总结
#
# 和学员查详情的区别：
#   学员 → 从数据库读（教师确认发布后才有）
#   教师 → 从 MemorySaver 读（AI批改完就有）
# ───────────────────────────────────────────────────────
@router.get(
    "/submissions/{submission_id}/review")
async def get_submission_review(
    submission_id: str,
    current_user:  dict = Depends(get_current_user),
):
    """教师获取 AI 预批改详情"""
    # 拼出 thread_id，从 MemorySaver 读图存档
    config = {
        "configurable": {
            "thread_id":
                await _get_thread_id(
                    submission_id)
        },
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": current_user["user_id"],
            "langfuse_session_id": submission_id,
        },
    }

    try:
        state_snapshot = await _graph.aget_state(
            config)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail="找不到该提交记录的批改状态："
                   f"{e}")

    # 快照为空 → 图还没跑到 interrupt()
    # 或者记录不存在
    if (not state_snapshot
            or not state_snapshot.values):
        raise HTTPException(
            status_code=404,
            detail="批改尚未完成或记录不存在")

    # snapshot.values 就是图暂停时的完整 State
    sv = state_snapshot.values
    return {
        "submission_id": submission_id,
        "student_id":
            sv.get("student_id", ""),
        "pre_review_summary":
            sv.get("pre_review_summary", {}),
        "weak_points":
            sv.get("weak_points", []),
        "weak_points_summary":
            sv.get("weak_points_summary", ""),
    }

# ── GET /submissions/{id}/review 教师看详情的辅助函数 ────
# ────────────── 从 DB 获取 thread_id ───────────────────
# thread_id 是 MemorySaver 的「存档编号」
# 格式由 build_thread_id() 生成，一般是：
#   student_{student_id}_session_{submission_id}
#
# 为什么需要这个函数？
#   MemorySaver 靠 thread_id 找存档
#   但 API 层只有 submission_id
#   所以要先查 DB 拿到 student_id
#   再拼出 thread_id
# ───────────────────────────────────────────────────────
async def _get_thread_id(
        submission_id: str) -> str:
    """从 exam_submissions 读 student_id，
    拼出 MemorySaver 的 thread_id"""
    from backend.core.memory import build_thread_id
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("SELECT student_id "
                     "FROM exam_submissions "
                     "WHERE id = :sid"),
                {"sid": submission_id},
            )
            row = result.fetchone()
            if row:
                return build_thread_id(
                    str(row[0]), submission_id)
    except Exception as e:
        logger.warning(
            "exam.get_thread_id_failed",
            submission_id=submission_id,
            error=str(e))
    # 兜底：查不到就拼一个默认值（至少不报错）
    return (
        f"student_unknown_session_"
        f"{submission_id}")



# ───────────────────────────────────────────────────────
# POST /confirm 教师确认，恢复 interrupt
# ───────────────────────────────────────────────────────
# 这是整个流程的「最后一棒」：
#   教师看完 AI 预批改结果后，点「确认」或「改分」
#   这个接口把教师的决策传给图
#   图从 interrupt() 处恢复，接着跑：
#     apply_teacher_decision → publish_results
#   批改结果写入数据库，学员就能看到成绩了
#
# 核心机制：Command(resume=decision)
#   interrupt() 暂停时，图的状态存在 MemorySaver
#   Command(resume=...) 告诉 LangGraph：
#     「用这个值作为 interrupt() 的返回值，继续跑」
#   于是 teacher_review_node 里
#   interrupt() 拿到 decision，图继续往下走
# ───────────────────────────────────────────────────────

# 请求体模型：教师确认时传什么
class ConfirmRequest(BaseModel):
    # approve：同意 AI 的批改，不改分
    # modify：要改某几道题的分/评语
    action:        str
    # 改分明细列表，格式：
    # [{"question_id": "xxx",
    #   "new_score": 8,
    #   "comment": "步骤不完整"}]
    modifications: list[dict] = []


@router.post(
    "/submissions/{submission_id}/confirm")
async def confirm_review(
    submission_id: str,
    req:           ConfirmRequest,
    current_user:  dict = Depends(get_current_user),
):
    """
    教师确认批改结果，恢复 interrupt，
    触发 apply_teacher_decision
    → publish_results。
    """
    # 校验 action 只能是 approve 或 modify
    if req.action not in ("approve", "modify"):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,
            detail="action 只能为 approve 或 modify",
        )

    # 拼出 thread_id，找到图的存档
    thread_id = await _get_thread_id(
        submission_id)
    config = {
        "configurable":
            {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": current_user["user_id"],
            "langfuse_session_id": submission_id,
        },
    }

    # 组装教师决策对象
    # 这个对象会作为 interrupt() 的返回值
    # 传给 teacher_review_node
    # → 再传给 apply_teacher_decision_node
    decision = {
        "action":        req.action,
        "modifications": req.modifications,
        "teacher_id":
            current_user["user_id"],
    }

    try:
        # 关键一行：Command(resume=decision)
        # 让图从 interrupt() 暂停处继续执行
        # ainvoke 会接着跑完剩下的节点：
        #   apply_teacher_decision
        #   → publish_results
        # 返回的 result 就是最终的 State
        result = await _graph.ainvoke(
            Command(resume=decision),
            config=config)
    except Exception as e:
        logger.error(
            "exam.confirm_failed",
            submission_id=submission_id,
            error=str(e))
        raise HTTPException(
            status_code=
                status
                .HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"发布失败：{e}",
        )

    # 从最终 State 里取 structured_output
    # 这是 publish_results_node 返回的汇总数据
    structured = (
        result.get("structured_output", {})
        or {}
    )

    logger.info(
        "exam.published",
        submission_id=submission_id,
        teacher_id=current_user["user_id"],
        final_score=structured.get(
            "final_score", 0),
    )

    # 返回发布结果给教师前端
    return {
        "submission_id": submission_id,
        "status": "published",
        "final_score":
            structured.get("final_score", 0),
        "full_score":
            structured.get("full_score", 0),
        "score_rate":
            structured.get("score_rate", 0),
        "weak_points":
            structured.get("weak_points", []),
        "weak_points_summary":
            structured.get(
                "weak_points_summary", ""),
    }

