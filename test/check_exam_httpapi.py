# test/check_exam_httpapi.py
# ═══════════════════════════════════════════════════════
# 考试批改 HTTP API 端到端测试
# 前置条件：
#   1. 后端已启动：uvicorn backend.main:app --port 8000
#   2. 数据库已初始化：python scripts/seed_data.py
#   3. tmp/student_answer.docx 存在
#
# 测试流程（模拟完整业务链路）：
#   学员登录 → 提交试卷 → 轮询等待AI批改
#   → 教师登录 → 查看待审列表 → 查看批改详情
#   → 教师确认发布 → 学员查看最终成绩
# ═══════════════════════════════════════════════════════

import asyncio
import time
import httpx
from sqlalchemy import text
from backend.config import get_abs_path
from backend.dependencies import AsyncSessionLocal

# ── 配置 ──────────────────────────────────────────────
BASE_URL = "http://localhost:8000/api/v1"
EXAM_ID  = "e0000001-0000-0000-0000-000000000001"
# 学员作答的 Word 文件（相对项目根目录）
WORD_FILE = get_abs_path("tmp/student_answer.docx")

# 种子数据里的测试账号
STUDENT_USER = "student01"
STUDENT_PWD  = "Student@123456"
TEACHER_USER = "teacher01"
TEACHER_PWD  = "Teacher@123456"

# 轮询配置：每3秒查一次，最多等120秒
POLL_INTERVAL = 3
POLL_TIMEOUT  = 120


# ── 工具函数 ──────────────────────────────────────────
def login(username: str, password: str) -> str:
    """登录，返回 JWT token"""
    resp = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"username": username, "password": password},
        trust_env=False,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"  [login] {username} 登录成功")
    return token


def auth_header(token: str) -> dict:
    """拼 Authorization 请求头"""
    return {"Authorization": f"Bearer {token}"}


async def cleanup_old_submissions():
    """清理旧测试数据，确保可重复运行"""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # 先查该学员对该试卷的提交记录
            row = (await session.execute(
                text("SELECT id FROM exam_submissions "
                     "WHERE exam_id = :exam_id "
                     "AND student_id = ("
                     "  SELECT id FROM users "
                     "  WHERE username = :uname)"),
                {"exam_id": EXAM_ID,
                 "uname": STUDENT_USER},
            )).fetchone()
            if row:
                old_id = str(row[0])
                # 先删关联的批改明细
                await session.execute(
                    text("DELETE FROM exam_reviews "
                         "WHERE submission_id = :sid"),
                    {"sid": old_id},
                )
                # 再删提交记录
                await session.execute(
                    text("DELETE FROM exam_submissions "
                         "WHERE id = :sid"),
                    {"sid": old_id},
                )
                print(f"  已清理旧记录: {old_id}")
            else:
                print("  无旧记录，无需清理")


# ══════════════════════════════════════════════════════
# 主测试流程
# ══════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 第0步：清理旧数据（保证可重复跑） ───────
    print("\n" + "=" * 60)
    print("第0步：清理旧测试数据")
    print("=" * 60)
    asyncio.run(cleanup_old_submissions())
    print("✅ 清理完成")

    # ── 第1步：学员登录 ──────────────────────────────
    print("\n" + "=" * 60)
    print("第1步：学员登录")
    print("=" * 60)
    student_token = login(STUDENT_USER, STUDENT_PWD)

    # ── 第2步：学员提交试卷 ──────────────────────────
    print("\n" + "=" * 60)
    print("第2步：学员提交试卷（POST /exam/submit）")
    print("=" * 60)
    with open(WORD_FILE, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/exam/submit",
            headers=auth_header(student_token),
            data={"exam_id": EXAM_ID},
            files={"file": ("student_answer.docx", f,
                            "application/vnd.openxmlformats"
                            "-officedocument.wordprocessingml"
                            ".document")},
            timeout=30.0,
            trust_env=False,
        )
    print(f"  状态码: {resp.status_code}")
    if resp.status_code == 409:
        # 清理没干净，报错退出
        print(f"  ❌ {resp.json()['detail']}")
        print("  清理步骤未生效，请检查数据库")
        exit(1)
    resp.raise_for_status()
    submit_result = resp.json()
    submission_id = submit_result["submission_id"]
    print(f"  submission_id: {submission_id}")
    print(f"  status: {submit_result['status']}")
    assert resp.status_code == 202
    assert submit_result["status"] == "ai_processing"
    print("✅ 提交成功，AI 正在后台批改")

    # ── 第3步：学员轮询等待批改完成 ──────────────────
    print("\n" + "=" * 60)
    print("第3步：轮询等待 AI 批改完成")
    print(f"  （每{POLL_INTERVAL}秒查一次，最多等{POLL_TIMEOUT}秒）")
    print("=" * 60)
    start = time.time()
    final_status = None
    while time.time() - start < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        resp = httpx.get(
            f"{BASE_URL}/exam/my-submissions/{submission_id}",
            headers=auth_header(student_token),
            trust_env=False,
        )
        resp.raise_for_status()
        data = resp.json()
        current_status = data["status"]
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] status = {current_status}")
        # AI批改完 → pending_review（等教师确认）
        # 或者之前已发布 → published
        if current_status in ("pending_review", "published"):
            final_status = current_status
            break
    else:
        print(f"  ❌ 超时（{POLL_TIMEOUT}s），AI批改未完成")
        exit(1)

    assert final_status in ("pending_review", "published")
    print(f"✅ AI 批改完成，当前状态: {final_status}")

    # ── 第4步：教师登录 ──────────────────────────────
    print("\n" + "=" * 60)
    print("第4步：教师登录")
    print("=" * 60)
    teacher_token = login(TEACHER_USER, TEACHER_PWD)

    # ── 第5步：教师查看待审列表 ──────────────────────
    print("\n" + "=" * 60)
    print("第5步：教师查看待审列表（GET /exam/pending-reviews）")
    print("=" * 60)
    resp = httpx.get(
        f"{BASE_URL}/exam/pending-reviews",
        headers=auth_header(teacher_token),
        trust_env=False,
    )
    resp.raise_for_status()
    pending = resp.json()
    print(f"  待审总数: {pending['total']}")
    # 找到刚提交的那份
    found = False
    for item in pending["items"]:
        if item["submission_id"] == submission_id:
            found = True
            print(f"  找到目标提交:")
            print(f"    学员: {item['student_name']}")
            print(f"    试卷: {item['exam_title']}")
            pr = item["pre_review"]
            print(f"    AI预评: {pr['total_score']}/{pr['full_score']}")
            print(f"    需复核: {pr['needs_review_count']}题")
            break
    if not found:
        print("  ⚠️  待审列表中未找到该提交（可能已发布）")
    else:
        print("✅ 待审列表正确")

    # ── 第6步：教师查看批改详情 ──────────────────────
    print("\n" + "=" * 60)
    print("第6步：教师查看批改详情"
          "（GET /exam/submissions/{id}/review）")
    print("=" * 60)
    resp = httpx.get(
        f"{BASE_URL}/exam/submissions/{submission_id}/review",
        headers=auth_header(teacher_token),
        trust_env=False,
    )
    if resp.status_code == 404:
        print(f"  ⚠️  {resp.json()['detail']}")
        print("  （如果状态已是 published，详情接口可能无存档）")
    else:
        resp.raise_for_status()
        review = resp.json()
        summary = review.get("pre_review_summary", {})
        print(f"  AI预评总分: {summary.get('total_score')}"
              f" / {summary.get('full_score')}")
        print(f"  得分率: {summary.get('score_rate')}")
        print(f"  需复核: {summary.get('needs_review_count')}题")
        by_q = summary.get("by_question", [])
        print(f"  逐题明细（共{len(by_q)}题）:")
        for q in by_q[:5]:  # 最多打印前5题
            print(f"    题{q.get('question_no')} "
                  f"[{q.get('question_type')}] "
                  f"得分 {q.get('score')}/{q.get('full_score')} "
                  f"需复核={q.get('needs_review')}")
        weak = review.get("weak_points", [])
        print(f"  薄弱点数量: {len(weak)}")
        for wp in weak[:3]:
            print(f"    - {wp.get('tag')}: "
                  f"错{wp.get('wrong_count')}题")
        print("✅ 批改详情获取成功")

    # ── 第7步：教师确认发布 ──────────────────────────
    print("\n" + "=" * 60)
    print("第7步：教师确认发布"
          "（POST /exam/submissions/{id}/confirm）")
    print("=" * 60)
    # 测试 approve（直接通过）
    resp = httpx.post(
        f"{BASE_URL}/exam/submissions/{submission_id}/confirm",
        headers=auth_header(teacher_token),
        json={"action": "approve", "modifications": []},
        timeout=60.0,
        trust_env=False,
    )
    if resp.status_code == 500:
        print(f"  ❌ 发布失败: {resp.json().get('detail')}")
        exit(1)
    resp.raise_for_status()
    confirm_result = resp.json()
    print(f"  status: {confirm_result['status']}")
    print(f"  最终得分: {confirm_result['final_score']}"
          f" / {confirm_result['full_score']}")
    print(f"  得分率: {confirm_result['score_rate']}")
    assert confirm_result["status"] == "published"
    print("✅ 教师确认发布成功")

    # ── 第8步：学员查看最终成绩 ──────────────────────
    print("\n" + "=" * 60)
    print("第8步：学员查看最终成绩"
          "（GET /exam/my-submissions/{id}）")
    print("=" * 60)
    resp = httpx.get(
        f"{BASE_URL}/exam/my-submissions/{submission_id}",
        headers=auth_header(student_token),
        trust_env=False,
    )
    resp.raise_for_status()
    result = resp.json()
    assert result["status"] == "published"
    summary = result.get("pre_review_summary", {})
    print(f"  状态: {result['status']}")
    print(f"  总得分: {summary.get('total_score')}"
          f" / {summary.get('full_score')}")
    print(f"  得分率: {summary.get('score_rate')}")
    by_q = summary.get("by_question", [])
    print(f"  逐题成绩（共{len(by_q)}题）:")
    for q in by_q:
        print(f"    题{q.get('question_no')} "
              f"[{q.get('question_type')}] "
              f"最终 {q.get('final_score')}/{q.get('full_score')}")
    weak = result.get("weak_points", [])
    print(f"  薄弱点: {len(weak)}个")
    print(f"  总评: {result.get('weak_points_summary', '')}")
    print("✅ 学员查看成绩成功")

    # ── 第9步：学员查看提交列表 ──────────────────────
    print("\n" + "=" * 60)
    print("第9步：学员查看提交列表（GET /exam/my-submissions）")
    print("=" * 60)
    resp = httpx.get(
        f"{BASE_URL}/exam/my-submissions",
        headers=auth_header(student_token),
        trust_env=False,
    )
    resp.raise_for_status()
    submissions = resp.json()
    print(f"  提交记录数: {len(submissions['items'])}")
    for item in submissions["items"][:3]:
        print(f"    {item['exam_title']} | "
              f"{item['status']} | "
              f"{item['submitted_at']}")
    print("✅ 提交列表获取成功")

    # ── 全部通过 ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 考试批改 HTTP API 端到端测试全部通过！")
    print("=" * 60)
