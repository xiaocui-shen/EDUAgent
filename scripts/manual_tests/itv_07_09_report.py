# scripts/manual_tests/itv_07_09_report.py
# 07-09 实测：generate_report_node 基于完整对话生成五维度结构化报告。
# 喂入一份"已完成面试"的 State（李明的全程对话），看 LLM 输出的报告 JSON。
# 需 DeepSeek API。运行：python scripts/manual_tests/itv_07_09_report.py

import asyncio
import json

from itv_fixtures import load_env, section, make_state_for_report

load_env()

from backend.agents.interview.nodes import generate_report_node


async def main():
    section("① 输入：一份已完成面试的 State（李明全程对话）")
    state = make_state_for_report()
    print(f"对话消息数: {len(state['messages'])}  总轮数: {state['total_turn_count']}")
    print(f"岗位: {state['target_position']}")

    section("② 调用 generate_report_node，LLM 生成五维度结构化报告")
    updates = await generate_report_node(state)
    report  = updates["report"]

    print("综合评分:", report.get("overall_score"), "/ 100")
    print("\n五个维度：")
    for d in report.get("dimensions", []):
        print(f"  - {d['dimension']:6s} {d['score']:>3} 分  {d['comment'][:36]}")
    print("\n核心优势:", report.get("strengths"))
    print("提升方向:", report.get("improvements"))
    print("建议复习:", report.get("recommended_topics"))
    print("下一步  :", report.get("next_step_advice"))

    section("③ 节点同时写入的三件事")
    print("report           : dict，供 save_report 写入 DB")
    print("messages         :", updates["messages"][0].content[:40], "...")
    print("structured_output:", list(updates["structured_output"].keys()))

    section("✅ 07-09 generate_report 生成完成")


if __name__ == "__main__":
    asyncio.run(main())
