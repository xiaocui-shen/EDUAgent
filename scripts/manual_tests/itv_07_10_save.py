# scripts/manual_tests/itv_07_10_save.py
# 07-10 实测：save_report_node（UPDATE 报告）+ save_memory_node（UPSERT 摘要）。
# 写库后立即回查，确认数据真的落库。
#
# 前置条件：PostgreSQL 已启动 + 已 seed（存在 student01）。
# 运行：python scripts/manual_tests/itv_07_10_save.py

import asyncio
import json
from sqlalchemy import text

from itv_fixtures import load_env, section, base_state, setup_db_fixture

load_env()

from backend.agents.interview.nodes import save_report_node, save_memory_node
from backend.core.memory import build_thread_id
from backend.dependencies import AsyncSessionLocal


SAMPLE_REPORT = {
    "dimensions": [
        {"dimension": "技术深度", "score": 72, "comment": "Transformer 与 RAG 掌握较好",
         "highlights": ["自注意力准确"], "weaknesses": ["自回归未答出"]},
    ],
    "overall_score": 76,
    "strengths": ["RAG 项目经验扎实", "表达逻辑清晰"],
    "improvements": ["补全大模型基础原理"],
    "overall_comment": "整体达到初级 AI 开发岗要求。",
    "recommended_topics": ["自回归生成", "LoRA 原理"],
    "next_step_advice": "系统梳理基础原理后再次模拟面试。",
}


async def query_session(thread_id: str) -> dict:
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("""SELECT status, overall_score, summary, report
                    FROM interview_sessions WHERE thread_id = :tid"""),
            {"tid": thread_id},
        )).mappings().fetchone()
    return dict(row) if row else {}


async def main():
    section("① 准备 DB fixture（已有一条 in_progress 会话）")
    ctx       = await setup_db_fixture()
    thread_id = ctx["thread_id"]
    print("thread_id:", thread_id)
    before = await query_session(thread_id)
    print("写入前:", {"status": before["status"], "overall_score": before["overall_score"]})

    section("② save_report_node：UPDATE 报告，status → finished")
    state = base_state(
        student_id=ctx["student_id"], session_id=ctx["session_id"],
        report=SAMPLE_REPORT,
    )
    await save_report_node(state)
    after = await query_session(thread_id)
    print("写入后 status       :", after["status"])
    print("写入后 overall_score:", after["overall_score"])
    report_db = after["report"] if isinstance(after["report"], dict) else json.loads(after["report"])
    print("写入后 report 维度数 :", len(report_db.get("dimensions", [])))

    section("③ save_memory_node：触发摘要压缩 + UPSERT summary")
    from langchain_core.messages import HumanMessage, AIMessage
    long_msgs = []
    for i in range(11):
        long_msgs.append(AIMessage(content=f"面试官第{i}个问题"))
        long_msgs.append(HumanMessage(content=f"学员第{i}个回答，涉及 RAG 与微调细节"))
    mem_state = base_state(
        student_id=ctx["student_id"], session_id=ctx["session_id"],
        messages=long_msgs, should_summarize=True,   # 强制触发压缩
    )
    await save_memory_node(mem_state)
    final = await query_session(thread_id)
    print("UPSERT 后 summary（前 60 字）:")
    print("  ", (final["summary"] or "（空）")[:60])

    section("✅ 07-10 save_report / save_memory 落库验证完成")


if __name__ == "__main__":
    asyncio.run(main())
