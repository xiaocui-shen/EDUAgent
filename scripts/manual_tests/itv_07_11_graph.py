# scripts/manual_tests/itv_07_11_graph.py
# 07-11 实测：build_interview_graph 编译整张图，并用 MemorySaver 跑多轮，
# 观察 current_stage 如何随对话自动推进（warmup → tech_base → ...）。
#
# 前置条件：PostgreSQL 已启动 + 已 seed + .env.local 配好 DEEPSEEK_API_KEY。
# 运行：python scripts/manual_tests/itv_07_11_graph.py

import asyncio
from langchain_core.messages import HumanMessage

from itv_fixtures import (
    load_env, section, setup_db_fixture, ANSWER_SCRIPT, TARGET_POSITION,
)

load_env()

from backend.agents.interview.graph import build_interview_graph
from backend.agents.interview.state import InterviewStage
from backend.core.memory import build_config


def last_ai(messages):
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage":
            return m.content
    return ""


async def main():
    section("① 编译图（不需要 DB，仅验证拓扑）")
    graph = build_interview_graph()
    print("图编译成功，节点数:", len(graph.nodes))
    print("节点列表:", [n for n in graph.nodes if n not in ("__start__", "__end__")])

    section("② 准备 DB fixture + 首轮启动（完整 initial_state）")
    ctx    = await setup_db_fixture()
    config = build_config(ctx["student_id"], ctx["session_id"])

    initial_state = {
        "messages":          [HumanMessage(content="[开始面试]")],
        "student_id":        ctx["student_id"],
        "tenant_id":         ctx["tenant_id"] if "tenant_id" in ctx else "tenant_default",
        "session_id":        ctx["session_id"],
        "target_position":   TARGET_POSITION,
        "resume_review_id":  ctx["resume_review_id"],
        "resume_projects":   [], "resume_skills": [],
        "current_stage":     InterviewStage.WARMUP.value,
        "stage_turn_count":  0, "total_turn_count": 0, "max_turns": 40,
        "question_bank":     [], "current_question": None, "projects_asked": [],
        "last_answer_quality": "adequate", "followup_count": 0,
        "existing_summary":  None, "should_summarize": False,
        "report":            None, "fallback_used": False, "structured_output": None,
    }
    result = await graph.ainvoke(initial_state, config=config)
    print(f"  [首轮] stage={result['current_stage']} "
          f"turns={result['total_turn_count']}")
    print("  开场白:", last_ai(result["messages"])[:50], "...")

    section("③ 连续发 4 轮（只传增量，MemorySaver 自动恢复 State）")
    for i, answer in enumerate(ANSWER_SCRIPT[:4], start=1):
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=answer)],
             "student_id": ctx["student_id"], "session_id": ctx["session_id"],
             "tenant_id": "tenant_default"},
            config=config,
        )
        print(f"  [第{i}轮] stage={result['current_stage']:9s} "
              f"turns={result['total_turn_count']}  "
              f"面试官: {last_ai(result['messages'])[:36]}...")

    section("✅ 07-11 图装配 + 多轮推进验证完成")


if __name__ == "__main__":
    asyncio.run(main())
