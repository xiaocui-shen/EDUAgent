# scripts/manual_tests/itv_07_04_load_context.py
# 07-04 实测：load_context_node 首轮初始化——并行加载简历、题库、摘要 + LLM 动态出题。
#
# 前置条件：
#   1. PostgreSQL 已启动（docker-compose up -d postgres）
#   2. 已 seed：python scripts/seed_data.py（需存在 student01 账号）
#   3. .env.local 配好 DEEPSEEK_API_KEY
# 运行：python scripts/manual_tests/itv_07_04_load_context.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import load_env, section, base_state, setup_db_fixture

load_env()

from backend.agents.interview.nodes import load_context_node, _generate_questions_by_llm
from backend.agents.interview.state import InterviewStage


async def main():
    section("① 准备 DB fixture：写入李明简历 + 新建面试会话")
    ctx = await setup_db_fixture()
    print("student_id      :", ctx["student_id"])
    print("session_id      :", ctx["session_id"])
    print("resume_review_id:", ctx["resume_review_id"])
    #
    section("② 首轮 load_context_node（total_turn_count=0 → 走初始化路径）")
    state = base_state(
        student_id=ctx["student_id"],
        session_id=ctx["session_id"],
        resume_review_id=ctx["resume_review_id"],
        total_turn_count=0,
        messages=[HumanMessage(content="[开始面试]")],
    )
    updates = await load_context_node(state)

    print("初始化后的阶段:", updates.get("current_stage"),
          "(应为", InterviewStage.WARMUP.value, ")")
    print("\n简历联动数据（从 resume_reviews 读取）：")
    print("  resume_projects:", [p["name"] for p in updates.get("resume_projects", [])])
    print("  resume_skills  :", updates.get("resume_skills", [])[:5], "...")

    bank = updates.get("question_bank", [])
    print(f'updates: {updates}')
    print(f"\n题库 question_bank：共 {len(bank)} 题")
    print("  LLM 动态生成优先：")
    for q in bank:
        print(f"    [{q['id']:>6}] ({q['difficulty']}) {q['content'][:34]}")

    print("\n历史摘要 existing_summary:", updates.get("existing_summary"))

    section("③ 单独验证 LLM 动态出题（_generate_questions_by_llm）")
    qs = await _generate_questions_by_llm("AI大模型开发工程师", count=4)
    print(f"LLM 生成 {len(qs)} 题：")
    for q in qs:
        print(f"  - {q['content'][:40]}  tags={q['tags']}")

    section("✅ 07-04 load_context 首轮初始化验证完成")


if __name__ == "__main__":
    asyncio.run(main())
