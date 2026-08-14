"""
11-3 试卷批改 Agent 评估 — 对标教师基准
=============================================
教学目的：
  - 对应第6章（试卷批改 Agent）
  - 直接调用 build_exam_graph()，运行到 interrupt() 暂停点
  - 从 pre_review_summary.by_question 提取 AI 预批改分数
  - 量化两个核心指标：MAE（平均绝对误差）+ Cohen's κ（等级一致性）

核心思想：
  试卷批改的「正确答案」是老师给的分数，不是公式算出来的。
  所以评估逻辑是：让 AI 批改 → 让老师独立批改（不看 AI 结果）
  → 对比两者差距。差距越小，AI 批改越可信。

运行方式：
  PyCharm 右键运行，或：python test/evaluation/run_exam_eval.py

前置条件：
  - PostgreSQL 已启动
  - 已运行 scripts/seed_standard_exam.py（写入标准试卷）
  - users 表存在 student01
  - tmp/student_answer.docx 存在（学员答卷）
  - gold/teacher_scores.json 已准备（教师独立评分，格式见课件 11.3.2）
"""

import sys
import os
import asyncio
import uuid

# 支持 PyCharm 右键直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text

from test.evaluation.eval_config import (
    load_gold, print_banner, print_verdict,
    MAE_TARGET, KAPPA_TARGET, PROJECT_ROOT,
)


# ── 配置 ────────────────────────────────────────────────────
# 标准试卷 ID（seed_standard_exam.py 写入，固定值）
EXAM_ID = "e0000001-0000-0000-0000-000000000001"
# 教师评分 Gold 文件
GOLD_FILE = "teacher_scores.json"
# 学员答卷 Word 文件（用绝对路径，避免工作目录不同导致找不到）
WORD_FILE = str(PROJECT_ROOT / "tmp" / "student_answer.docx")


# ── Step 1：加载教师评分 Gold 数据 ──────────────────────────
def load_teacher_scores() -> dict:
    """
    加载教师独立评分，返回 {question_id: 条目} 字典。
    每条格式（见 gold/teacher_scores.json）：
    {
        "question_id": "e0000001-...-a4",   # 与 DB 题目 ID 一致
        "question_content": "请解释 Spring IOC...",
        "full_score": 10,
        "teacher_score": 8,                 # 老师独立给的分数
        "teacher_comment": "概念正确，但..."
    }
    """
    data = load_gold(GOLD_FILE)
    if not data:
        print("\n  请先准备 gold/teacher_scores.json，格式示例：")
        print('  [{"question_id": "e0000001-0000-0000-0000-0000000000a4",')
        print('    "question_content": "请解释 Spring IOC...",')
        print('    "full_score": 10, "teacher_score": 8,')
        print('    "teacher_comment": "概念正确，举例基本准确"}]')
        sys.exit(0)
    # 以 question_id 为键，方便后面和 AI 结果匹配
    return {item["question_id"]: item for item in data}


# ── Step 2：准备临时提交记录（满足外键约束）────────────────
async def setup_submission() -> tuple:
    """
    批改结果要写回 exam_submissions 表，所以必须先有一条提交记录。
    本函数做三件事：
      1. 查出 student01 的真实 UUID（外键约束要求）
      2. 删除旧的测试记录（避免 UNIQUE(exam_id, student_id) 重跑冲突）
      3. 插入一条临时提交记录（status='ai_processing'）
    返回 (student_uuid, submission_id)。
    """
    from backend.dependencies import AsyncSessionLocal

    # ① 查学生真实 UUID
    async with AsyncSessionLocal() as session:
        row = await session.execute(
            text("SELECT id FROM users "
                 "WHERE username = 'student01' LIMIT 1")
        )
        # row.scalar() 取结果集第一行第一列的值
        student_uuid = str(row.scalar())

    # ② 生成临时提交 ID
    submission_id = str(uuid.uuid4())

    # ③ 清旧记录 + 插入临时记录（同一事务）
    async with AsyncSessionLocal() as session:
        async with session.begin():  # 开启事务
            # 先删批改结果（外键引用 submission）
            await session.execute(
                text("""
                    DELETE FROM exam_reviews
                    WHERE submission_id IN (
                        SELECT id FROM exam_submissions
                        WHERE exam_id = :exam_id
                        AND student_id = :student_id)"""),
                {"exam_id": EXAM_ID, "student_id": student_uuid},
            )
            # 再删旧提交记录
            await session.execute(
                text("DELETE FROM exam_submissions "
                     "WHERE exam_id=:exam_id "
                     "AND student_id=:student_id"),
                {"exam_id": EXAM_ID, "student_id": student_uuid},
            )
            # 插入临时提交记录
            await session.execute(
                text("""
                    INSERT INTO exam_submissions
                    (id, exam_id, student_id, tenant_id, status)
                    VALUES (:id, :exam_id, :student_id,
                            'tenant_default', 'ai_processing')
                """),
                {"id": submission_id, "exam_id": EXAM_ID,
                 "student_id": student_uuid},
            )

    return student_uuid, submission_id


# ── Step 3：运行批改图，跑到 interrupt() 暂停 ───────────────
async def run_exam_graph(student_uuid: str, submission_id: str) -> dict:
    """
    直接调用试卷批改 Agent 的 LangGraph 图。
    关键：图执行到 teacher_review 节点的 interrupt() 会自动暂停，
    ainvoke 提前返回。此时 AI 预批改已完成，
    state["pre_review_summary"] 就是 AI 的完整批改结果。
    评估只需要 AI 预批改分数，所以不需要恢复图（无需教师决策）。
    """
    from backend.agents.exam.graph import build_exam_graph

    graph = build_exam_graph()

    # thread_id：MemorySaver 用它区分不同批改任务
    thread_id = f"eval_exam_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    # 初始 State（从 HTTP 接口传入的字段，这里手工构造）
    initial_state = {
        "messages": [],
        "word_file_path": WORD_FILE,      # 学员答卷 Word
        "exam_id": EXAM_ID,
        "submission_id": submission_id,
        "student_id": student_uuid,
        "tenant_id": "tenant_default",
        "session_id": "eval_exam_session",
        "teacher_decision": None,         # 教师还没审核
        "fallback_used": False,
        "sandbox_skipped": False,
    }

    # 第一次 ainvoke：跑到 interrupt() 自动暂停并返回
    state = await graph.ainvoke(initial_state, config=config)
    return state


# ── Step 4：提取 AI 预批改分数 ──────────────────────────────
def extract_ai_scores(state: dict) -> dict:
    """
    从 pre_review_summary.by_question 提取每道题的 AI 给分。
    返回 {question_id: {"score": x, "full_score": y,
                        "question_no": n, "question_type": t}}
    """
    summary = state.get("pre_review_summary", {})
    by_question = summary.get("by_question", [])

    ai_scores = {}
    for q in by_question:
        qid = q.get("question_id", "")
        ai_scores[qid] = {
            "score": q.get("score", 0),
            "full_score": q.get("full_score", 0),
            "question_no": q.get("question_no", 0),
            "question_type": q.get("question_type", ""),
        }
    return ai_scores


# ── Step 5：分数转等级（Cohen's κ 用）──────────────────────
def score_to_grade(score: int, full_score: int) -> str:
    """
    把原始分转成等级，用于 Cohen's κ 计算。
    κ 衡量的是「AI 和人工是否把题目分到同一等级」，
    而不是要求分数完全一致（老师之间打分也有主观差异）。
    """
    ratio = score / full_score if full_score > 0 else 0
    if ratio >= 0.85:
        return "优"
    elif ratio >= 0.60:
        return "良"
    else:
        return "差"


# ── Step 6：计算 MAE + Cohen's κ ────────────────────────────
def compute_metrics(ai_scores: dict, teacher_data: dict) -> dict:
    """
    对 AI 和教师都评了的题（取交集），计算两个指标：
      MAE       = mean(|AI分 - 教师分|)   → 平均差几分
      Cohen's κ = 等级一致性               → 是否分对档位
    """
    from sklearn.metrics import cohen_kappa_score

    # 只取两边都有的题目
    matched_ids = [qid for qid in ai_scores if qid in teacher_data]
    if not matched_ids:
        return {"error": "没有匹配的题目 ID，请检查 "
                         "teacher_scores.json 里的 question_id "
                         "是否和 AI 批改结果一致"}

    ai_list = []        # AI 分数列表
    teacher_list = []   # 教师分数列表
    full_scores = []    # 各题满分
    per_question = []   # 每题明细

    for qid in matched_ids:
        ai_score = ai_scores[qid]["score"]
        full_score = ai_scores[qid]["full_score"]
        teacher_score = teacher_data[qid]["teacher_score"]

        ai_list.append(ai_score)
        teacher_list.append(teacher_score)
        full_scores.append(full_score)

        per_question.append({
            "question_id": qid,
            "question_no": ai_scores[qid]["question_no"],
            "question_type": ai_scores[qid]["question_type"],
            "ai_score": ai_score,
            "teacher_score": teacher_score,
            "diff": abs(ai_score - teacher_score),
            "full_score": full_score,
            "question": teacher_data[qid].get(
                "question_content", "")[:30],
        })

    # MAE：所有误差的绝对值求平均
    diffs = [abs(a - t) for a, t in zip(ai_list, teacher_list)]
    mae = sum(diffs) / len(diffs)

    # Cohen's κ：先把分数转成等级再算
    ai_grades = [
        score_to_grade(s, f) for s, f in zip(ai_list, full_scores)]
    teacher_grades = [
        score_to_grade(s, f) for s, f in zip(teacher_list, full_scores)]
    try:
        kappa = cohen_kappa_score(teacher_grades, ai_grades)
    except Exception:
        # 等级全部相同等边界情况，κ 无定义
        kappa = float("nan")

    return {
        "matched_count": len(matched_ids),
        "mae": round(mae, 2),
        "kappa": round(kappa, 3),
        "per_question": per_question,
    }


# ── Step 7：输出评估报告 ────────────────────────────────────
def print_report(metrics: dict):
    if "error" in metrics:
        print(f"\n  ❌ 评估失败：{metrics['error']}")
        return False

    print_banner("试卷批改 Agent 评估报告（对标教师基准）")

    mae = metrics["mae"]
    kappa = metrics["kappa"]
    mae_ok = mae <= MAE_TARGET
    # κ 可能为 nan（边界情况），nan 比较恒为 False
    kappa_ok = kappa == kappa and kappa >= KAPPA_TARGET

    print(f"\n  匹配题目数 : {metrics['matched_count']} 道")
    print(f"\n  平均绝对误差 MAE : {mae:.2f} 分  "
          f"（目标 ≤ {MAE_TARGET:.1f}）  {'✅' if mae_ok else '❌'}")
    print(f"  等级一致性 κ     : {kappa:.3f}      "
          f"（目标 ≥ {KAPPA_TARGET:.1f}）  "
          f"{'✅' if kappa_ok else '❌'}")

    # 每题明细表
    print(f"\n  {'题':>2} {'题型':<14} {'AI':>5} {'师':>5} {'差':>4}")
    print("  " + "-" * 50)
    for q in sorted(metrics["per_question"],
                    key=lambda x: x["question_no"]):
        flag = " ⚠️" if q["diff"] >= 3 else ""
        print(f"  {q['question_no']:>2} {q['question_type']:<14} "
              f"{q['ai_score']:>3}/{q['full_score']:<2} "
              f"{q['teacher_score']:>3}/{q['full_score']:<2} "
              f"{q['diff']:>3}{flag}")
    print("  " + "-" * 50)

    all_pass = mae_ok and kappa_ok
    print_verdict(
        all_pass,
        pass_msg="AI 批改质量达标",
        fail_msg="批改质量不达标，需检查评分 Prompt",
    )

    # 优化建议
    if not all_pass:
        print("  优化方向：")
        if not mae_ok:
            print("    - MAE 偏高 → 检查简答题评分 Prompt / 得分点拆分粒度")
        if not kappa_ok:
            print("    - κ 偏低 → AI 给分档位与人工不一致，"
                  "检查置信度阈值与 needs_review 逻辑")
    return all_pass


# ── 主流程 ──────────────────────────────────────────────────
async def run_evaluation():
    print_banner("11-3  试卷批改 Agent 评估（MAE + Cohen's κ）")

    # Step 1：加载教师评分
    teacher_data = load_teacher_scores()
    print(f"\n  教师评分加载成功，共 {len(teacher_data)} 道题")
    print(f"  MAE 目标：≤ {MAE_TARGET:.1f} 分    "
          f"Cohen's κ 目标：≥ {KAPPA_TARGET:.1f}")

    # Step 2：准备临时提交记录
    print("\n  准备临时提交记录（满足外键约束）...")
    student_uuid, submission_id = await setup_submission()
    print(f"    student_id    = {student_uuid}")
    print(f"    submission_id = {submission_id}")

    # Step 3：运行批改图（跑到 interrupt 暂停）
    print("\n  运行 Exam Agent 批改中（到 interrupt 自动暂停）...")
    state = await run_exam_graph(student_uuid, submission_id)

    # Step 4：提取 AI 分数
    ai_scores = extract_ai_scores(state)
    summary = state.get("pre_review_summary", {})
    print(f"\n  AI 预批改完成：总分 {summary.get('total_score', 0)} / "
          f"{summary.get('full_score', 0)}，"
          f"共 {len(ai_scores)} 道题")
    print(f"  需教师复核题数：{summary.get('needs_review_count', 0)}")

    # Step 5-6：计算指标
    metrics = compute_metrics(ai_scores, teacher_data)

    # Step 7：输出报告
    print_report(metrics)


def main():
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()