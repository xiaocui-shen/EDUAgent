# scripts/manual_tests/itv_07_06_evaluate.py
# 07-06 实测：evaluate_answer_node 给学员回答打质量标签。
# 演示三条路径：① 优秀回答（走 Think Tool + LLM 两步）② 明确不会（走快速路径，不调用 LLM）
#               ③ WARMUP 阶段（跳过评估，直接 ADEQUATE）
# 需 DeepSeek API（.env.local 里 DEEPSEEK_API_KEY）。运行：
#   python scripts/manual_tests/itv_07_06_evaluate.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import ( section, make_state_eval_excellent, make_state_eval_no_answer,
    base_state, ANSWER_SCRIPT,
)

from backend.agents.interview.nodes import evaluate_answer_node


async def run(title: str, state: dict):
    cur_q = (state.get("current_question") or {}).get("content", "（无）")
    answer = state["messages"][-1].content
    print(f"\n[{title}]")
    print(f"  问题: {cur_q}")
    print(f"  学员回答: {answer[:50]}{'...' if len(answer) > 50 else ''}")
    updates = await evaluate_answer_node(state)
    print(f"  → 质量标签: {updates['last_answer_quality']}")
    print(f"  → 计数更新: total_turns={updates['total_turn_count']} "
          f"stage_turns={updates['stage_turn_count']}")


async def main():
    section("① 优秀回答（Transformer）→ Think Tool 推理 + LLM 评估")
    await run("excellent 预期", make_state_eval_excellent())

    section("② 明确不会（'不知道'）→ 快速路径，不调用 LLM")
    await run("no_answer 预期", make_state_eval_no_answer())

    section("③ WARMUP 阶段 → 跳过评估，直接 ADEQUATE")
    warmup_state = base_state(
        current_stage="warmup", stage_turn_count=1, total_turn_count=1,
        messages=[HumanMessage(content=ANSWER_SCRIPT[0])],
    )
    await run("warmup 跳过", warmup_state)

    section("✅ 07-06 evaluate_answer 三条路径验证完成")


if __name__ == "__main__":
    asyncio.run(main())
