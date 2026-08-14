# scripts/manual_tests/itv_07_05_check_stage.py
# 07-05 实测：check_stage_node 是纯逻辑节点（不调用 LLM、不读 DB），
# 喂入不同的 State 快照，观察它如何决定「推进 / 停留 / 强制结束」。
# 纯离线，无需启动服务。运行：python scripts/manual_tests/itv_07_05_check_stage.py

import asyncio
from langchain_core.messages import HumanMessage, AIMessage

from itv_fixtures import section, base_state, sample_question_bank

from backend.agents.interview.nodes import (
    check_stage_node, _check_advance_condition, _next_stage,
)
from backend.agents.interview.state import InterviewStage


def dtest_next_stage():
    section("① _next_stage：阶段顺序推进（纯函数）")
    chain = ["warmup", "tech_base", "project", "closing", "finished"]
    for cur in chain:
        print(f"  {cur:10s} → {_next_stage(cur)}")
    assert _next_stage("warmup") == "tech_base"
    assert _next_stage("closing") == "finished"
    assert _next_stage("finished") == "finished"  # 终态保持


async def run_case(title: str, state: dict):
    result = await check_stage_node(state)
    decision = result.get("current_stage", "（停留，返回 {}）")
    print(f"\n[{title}]")
    print(f"  输入: stage={state['current_stage']} stage_turns={state['stage_turn_count']} "
          f"total_turns={state['total_turn_count']}")
    print(f"  决策: {decision}")
    return result


async def dtest_check_stage():
    section("② check_stage_node：各场景的推进决策")

    # 场景1：WARMUP 第 1 轮，学员介绍完 → 推进到 tech_base
    await run_case("WARMUP→? 学员已自我介绍", base_state(
        current_stage="warmup", stage_turn_count=1, total_turn_count=1,
        messages=[HumanMessage(content="我是李明，做过 RAG 项目...")],
    ))

    # 场景2：TECH_BASE 才 3 轮（< min 6）→ 停留
    await run_case("TECH_BASE 轮数不够（3<6）", base_state(
        current_stage="tech_base", stage_turn_count=3, total_turn_count=4,
        question_bank=sample_question_bank(),
        messages=[HumanMessage(content="过拟合是...")],
    ))
    #
    # 场景3：TECH_BASE 满 6 轮且已问 8 题 → 推进到 project
    bank_all_asked = [{**q, "asked": True} for q in sample_question_bank()]
    await run_case("TECH_BASE→? 已问满 8 题", base_state(
        current_stage="tech_base", stage_turn_count=6, total_turn_count=7,
        question_bank=bank_all_asked,
        messages=[HumanMessage(content="RAG 是检索增强生成...")],
    ))
    #
    # 场景4：PROJECT 所有项目都深挖过 → 推进到 closing
    await run_case("PROJECT→? 项目全覆盖", base_state(
        current_stage="project", stage_turn_count=2, total_turn_count=11,
        resume_projects=[{"name": "RAG系统"}, {"name": "LoRA微调"}],
        projects_asked=["RAG系统", "LoRA微调"],
        messages=[HumanMessage(content="这个项目我负责...")],
    ))
    #
    # 场景5：CLOSING 满 2 轮 → 进入 finished
    await run_case("CLOSING→? 反问满 2 轮", base_state(
        current_stage="closing", stage_turn_count=2, total_turn_count=14,
        messages=[HumanMessage(content="团队技术栈是什么？")],
    ))

    # 场景6：强制结束——学员发"结束面试吧"关键词
    await run_case("任意阶段 强制结束（关键词）", base_state(
        current_stage="tech_base", stage_turn_count=3, total_turn_count=5,
        messages=[HumanMessage(content="没有其他问题了，结束面试吧。")],
    ))

    # 场景7：强制结束——逼近轮数上限（total >= max-2）
    await run_case("任意阶段 强制结束（轮数上限）", base_state(
        current_stage="project", stage_turn_count=5, total_turn_count=39, max_turns=40,
        messages=[HumanMessage(content="继续聊...")],
    ))


def dtest_advance_condition():
    section("③ _check_advance_condition：各阶段的推进判据")
    # WARMUP 永远 True
    print("WARMUP →", _check_advance_condition("warmup", base_state(), ""))
    # TECH_BASE：问满 8 题才 True
    bank = [{**q, "asked": i < 8} for i, q in enumerate(sample_question_bank())]
    print("TECH_BASE（已问8题）→",
          _check_advance_condition("tech_base", base_state(question_bank=bank), ""))
    # PROJECT：项目全覆盖才 True
    st = base_state(resume_projects=[{"name": "A"}], projects_asked=["A"])
    print("PROJECT（项目全覆盖）→", _check_advance_condition("project", st, ""))


async def main():
    dtest_next_stage()
    await dtest_check_stage()
    dtest_advance_condition()
    section("✅ 07-05 check_stage 全部场景验证通过")


if __name__ == "__main__":
    asyncio.run(main())
