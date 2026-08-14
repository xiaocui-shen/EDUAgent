"""
HitL 改分率统计（自包含）
=========================
完整流程：
  1. 准备临时提交记录
  2. 运行批改图到 interrupt()，拿到 AI 分数
  3. 将 AI 分数写入 exam_reviews
  4. 用 gold/teacher_scores.json 模拟教师复核
  5. 统计并输出改分率

运行：python test/evaluation/check_hitl_rate.py
前置：PostgreSQL 已启动、users 表有 student01、
      tmp/student_answer.docx 存在
"""
import sys
import os
import asyncio
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8",
    errors="replace")

from sqlalchemy import text

from test.evaluation.eval_config import (
    load_gold, print_banner, PROJECT_ROOT,
)

# ── 配置（与 run_exam_eval.py 一致）────────────────────
EXAM_ID = "e0000001-0000-0000-0000-000000000001"
WORD_FILE = str(
    PROJECT_ROOT / "tmp" / "student_answer.docx")
GOLD_FILE = "teacher_scores.json"


# ── Step 1：准备临时提交记录 ───────────────────────────
async def setup_submission() -> tuple:
    """查出 student01 UUID，清旧记录，插临时提交"""
    from backend.dependencies import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        row = await session.execute(
            text("SELECT id FROM users "
                 "WHERE username='student01' LIMIT 1"))
        student_uuid = str(row.scalar())

    submission_id = str(uuid.uuid4())

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # 清旧批改记录（外键引用 submission）
            await session.execute(text("""
                DELETE FROM exam_reviews
                WHERE submission_id IN (
                    SELECT id FROM exam_submissions
                    WHERE exam_id=:eid
                      AND student_id=:sid)
            """), {"eid": EXAM_ID,
                   "sid": student_uuid})
            # 清旧提交记录
            await session.execute(text("""
                DELETE FROM exam_submissions
                WHERE exam_id=:eid
                  AND student_id=:sid
            """), {"eid": EXAM_ID,
                   "sid": student_uuid})
            # 插入临时提交
            await session.execute(text("""
                INSERT INTO exam_submissions
                (id, exam_id, student_id,
                 tenant_id, status)
                VALUES (:id,:eid,:sid,
                        'tenant_default',
                        'ai_processing')
            """), {"id": submission_id,
                   "eid": EXAM_ID,
                   "sid": student_uuid})

    return student_uuid, submission_id


# ── Step 2：运行批改图拿 AI 分数 ───────────────────────
async def run_exam_graph(student_uuid: str,
                         submission_id: str) -> dict:
    """跑到 interrupt() 暂停，返回含 AI 分数的 state"""
    from backend.agents.exam.graph import build_exam_graph

    graph = build_exam_graph()
    thread_id = f"hitl_eval_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [],
        "word_file_path": WORD_FILE,
        "exam_id": EXAM_ID,
        "submission_id": submission_id,
        "student_id": student_uuid,
        "tenant_id": "tenant_default",
        "session_id": "hitl_eval_session",
        "teacher_decision": None,
        "fallback_used": False,
        "sandbox_skipped": False,
    }
    state = await graph.ainvoke(
        initial_state, config=config)
    return state


# ── Step 3：AI 分数写入 exam_reviews ───────────────────
async def insert_ai_reviews(
        submission_id: str, state: dict) -> list:
    """把 AI 批改结果插入 exam_reviews"""
    from backend.dependencies import AsyncSessionLocal

    summary = state.get("pre_review_summary", {})
    by_question = summary.get("by_question", [])

    records = []
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for q in by_question:
                rid = str(uuid.uuid4())
                qid = q.get("question_id", "")
                ai_score = q.get("score", 0)
                q_type = q.get("question_type", "")
                await session.execute(text("""
                    INSERT INTO exam_reviews
                    (id, submission_id, question_id,
                     question_type, ai_score,
                     student_answer, ai_feedback)
                    VALUES (:rid,:sid,:qid,
                            :qt,:as,'','')
                """), {"rid": rid,
                       "sid": submission_id,
                       "qid": qid, "qt": q_type,
                       "as": ai_score})
                records.append({
                    "id": rid,
                    "question_id": qid,
                    "ai_score": ai_score,
                })
    return records


# ── Step 4：模拟教师复核 + 统计 ────────────────────────
async def simulate_teacher_review(
        records: list) -> tuple:
    """用 gold 教师分数填入 teacher_score"""
    from backend.dependencies import AsyncSessionLocal

    gold_list = load_gold(GOLD_FILE)
    gold_map = {
        item["question_id"]: item["teacher_score"]
        for item in gold_list
    } if gold_list else {}

    modified = 0
    total = len(records)

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for rec in records:
                qid = rec["question_id"]
                ai_score = rec["ai_score"]
                # Gold 有 → 教师分；没有 → 同意 AI
                t_score = gold_map.get(qid, ai_score)
                if t_score != ai_score:
                    modified += 1
                await session.execute(text("""
                    UPDATE exam_reviews
                    SET teacher_score=:ts,
                        final_score=:ts,
                        reviewed_at=NOW()
                    WHERE id=:rid
                """), {"ts": t_score,
                       "rid": rec["id"]})

    return total, modified


# ── 主流程 ─────────────────────────────────────────────
async def main():
    print_banner("HitL 改分率统计（自包含）")

    # 1. 准备数据
    print("\n  [1/4] 准备临时提交记录...")
    student_uuid, submission_id = (
        await setup_submission())

    # 2. 跑批改图
    print("  [2/4] 运行批改图（AI 打分）...")
    state = await run_exam_graph(
        student_uuid, submission_id)
    summary = state.get("pre_review_summary", {})
    n_q = len(summary.get("by_question", []))
    print(f"        AI 批改完成，共 {n_q} 题")

    # 3. 写入 exam_reviews
    print("  [3/4] 写入 exam_reviews...")
    records = await insert_ai_reviews(
        submission_id, state)

    # 4. 模拟教师复核 + 统计
    print("  [4/4] 模拟教师复核...")
    total, modified = (
        await simulate_teacher_review(records))

    # ── 输出报告 ───────────────────────────────────────
    rate = modified / total if total > 0 else 0
    print(f"\n  {'─' * 40}")
    print(f"  已复核题目数 : {total}")
    print(f"  教师改分次数 : {modified}")
    print(f"  HitL 改分率  : {rate:.1%}")
    print(f"  {'─' * 40}")
    if rate <= 0.2:
        print("  结论：改分率 ≤ 20%，"
              "AI 批改可信度高 ✅")
    else:
        print("  结论：改分率偏高，"
              "需优化 AI 批改 Prompt ⚠️")


if __name__ == "__main__":
    asyncio.run(main())