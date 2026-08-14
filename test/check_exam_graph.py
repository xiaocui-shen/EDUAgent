# test/check_exam_graph.py

import asyncio, uuid
from sqlalchemy import text
from langgraph.types import Command

from backend.agents.exam.graph import build_exam_graph
from backend.dependencies import AsyncSessionLocal

graph = build_exam_graph()
EXAM_ID = "e0000001-0000-0000-0000-000000000001"


async def main():
    # ── 测试准备：获取真实 student UUID，写入临时提交记录 ────────
    # 必须从数据库获取真实 student和teacher的UUID
    # 原因：考试提交和教师评审都有外键约束
    async with AsyncSessionLocal() as session:
        row = await session.execute(
            text("SELECT id FROM users "
                 "WHERE username = 'student01' LIMIT 1")
        )
        # row.scalar() 取结果集第一行第一列的那个值
        student_uuid = str(row.scalar())
        row = await session.execute(
            text("SELECT id FROM users "
                 "WHERE username = 'teacher01' LIMIT 1")
        )
        # row.scalar() 取结果集第一行第一列的那个值
        teacher_uuid = str(row.scalar())
    # ── 测试准备：生成临时 submission UUID ─────────────────────
    submission_id = str(uuid.uuid4())

    async with AsyncSessionLocal() as session:
        async with session.begin(): # 事务
            # 清除旧测试记录，避免 UNIQUE(exam_id, student_id) 重跑冲突
            await session.execute(
                text("""
                    DELETE FROM exam_reviews
                    WHERE submission_id IN (
                        SELECT id FROM exam_submissions
                        WHERE exam_id = :exam_id 
                        AND student_id = :student_id)"""),
                {"exam_id": EXAM_ID, "student_id": student_uuid},
            )
            await session.execute(
                text("DELETE FROM exam_submissions "
                     "WHERE exam_id=:exam_id "
                     "AND student_id=:student_id"),
                {"exam_id": EXAM_ID, "student_id": student_uuid},
            )
            await session.execute(
                text("""
                    INSERT INTO exam_submissions (id, exam_id, student_id, tenant_id, status)
                    VALUES (:id, :exam_id, :student_id, 'tenant_default', 'ai_processing')
                """),
                {"id": submission_id, "exam_id": EXAM_ID, "student_id": student_uuid},
            )

    # ── thread_id：MemorySaver 用它区分不同的批改任务 ─────────────
    # 两次 ainvoke() 必须用同一个 thread_id，图才能在第二次找回冻结的 State
    thread_id = f"test-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    # ── 初始状态：包含 word_file_path 和其他必要字段 ─────────────
    initial_state = {
        "messages": [],
        "word_file_path": "../tmp/student_answer.docx",
        "exam_id": EXAM_ID,
        "submission_id": submission_id,
        "student_id": student_uuid,
        "tenant_id": "tenant_default",
        "session_id": "test-session",
        "teacher_decision": None,
        "fallback_used": False,
        "sandbox_skipped": False,
    }

    # ═=第一次ainvoke：跑到teacher_review_node的interrupt()自动暂停 ══
    print("=" * 60)
    print("第一次 ainvoke：启动图，等待 interrupt 暂停...")
    print("=" * 60)
    result1 = await graph.ainvoke(initial_state, config=config)

    # interrupt() 暂停后，ainvoke 提前返回，State 已存入 MemorySaver
    # result1 包含截止到暂停时刻的所有 state 字段
    summary = result1.get("pre_review_summary", {})
    print(f"\n图已暂停（interrupt 触发）")
    print(f"AI 预评总分：{summary.get('total_score')} / "
          f"{summary.get('full_score')}")
    print(f"需复核题数：{summary.get('needs_review_count')}")
    print(f"薄弱点数量：{len(result1.get('weak_points', []))}")
    print(f"\n>>> 此刻图冻结，等待教师决策 <<<\n")

    # ══ 模拟教师操作：构造 decision，第二次 ainvoke 恢复图 ══════
    # 模拟教师决策1： 直接通过
    # decision = {
    #     "action": "approve", # 批改通过
    #     "modifications": [],
    #     # reviewed_by列是UUID类型，所以teacher_id必须UUID
    #     "teacher_id": teacher_uuid,
    # }
    # 模拟教师决策2： 修改分数
    decision = {
        "action": "modify", # 讲师修改分数
        "modifications": [
            {"question_id": result1['pre_review_summary'][
                'by_question'][0]['question_id'],
             "new_score": 5,
             "comment": "答案思路正确但表述不完整",
            }
        ],
        "teacher_id": teacher_uuid, # 必须是合法 UUID
    }

    print("=" * 60)
    print("第二次 ainvoke：Command(resume=decision)，恢复图执行...")
    print("=" * 60)
    result2 = await graph.ainvoke(
        Command(resume=decision), config=config)

    final = result2.get("final_results", [])
    print(f"\n图执行完毕")
    print(f"最终批改题数：{len(final)}")
    for q in final:
        print(f"  题{q['question_no']} [{q['question_type']}] "
              f"最终得分 {q['final_score']}/{q['full_score']}")

asyncio.run(main())
