# scripts/manual_tests/itv_07_07_resp_warmup_tech.py
# 07-07 实测：generate_response_node 在 WARMUP 和 TECH_BASE 阶段如何生成面试官回应。
# 演示：① WARMUP 开场白  ② TECH_BASE 首轮（评价自我介绍 + 出第一题）
#       ③ TECH_BASE 换题（上题反馈 + 下一题）
# 需 DeepSeek API。运行：python scripts/manual_tests/itv_07_07_resp_warmup_tech.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import ( section, base_state, make_state_tech_first, make_state_tech_next,
)

from backend.agents.interview.nodes import generate_response_node


async def run(title: str, state: dict):
    print(f"\n[{title}]")
    updates = await generate_response_node(state)
    reply = updates["messages"][0].content
    print("面试官输出：")
    print("  " + reply.replace("\n", "\n  "))
    if updates.get("current_question"):
        print(f"\n  （当前题已更新为: {updates['current_question']['content'][:40]}...）")
    return updates


async def main():
    section("① WARMUP：生成开场白，邀请自我介绍")
    await run("WARMUP opening", base_state(
        current_stage="warmup", stage_turn_count=0, total_turn_count=0,
        messages=[HumanMessage(content="[开始面试]")],
    ))

    section("② TECH_BASE 首轮：评价自我介绍 + 抛出第一道技术题")
    await run("INTRO_EVAL_TECH_FIRST", make_state_tech_first())

    section("③ TECH_BASE 换题：上题(adequate)简短反馈 + 出下一题")
    await run("ask_with_feedback", make_state_tech_next())

    section("✅ 07-07 WARMUP & TECH_BASE 回应生成完成")


if __name__ == "__main__":
    asyncio.run(main())
