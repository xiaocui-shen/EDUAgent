# scripts/manual_tests/orch_08_02_single.py
# 纯离线，无需启动服务。运行：python scripts/manual_tests/orch_08_02_single.py

import sys
sys.path.insert(0, ".")

from backend.core.orchestrator import (
    Orchestrator, AgentType, ExecutionMode,
    AgentRequest, AgentResponse, PipelineResult,
)


def section(t):
    print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)


def check_enums():
    section("① 两个枚举")
    print("ExecutionMode:", [m.value for m in ExecutionMode])
    print("AgentType:    ", [a.value for a in AgentType])


def check_request():
    section("② AgentRequest：李明的模拟面试请求 + thread_id 拼接")
    req = AgentRequest(
        student_id="liming-001",
        session_id="sess-7",
        agent_type=AgentType.INTERVIEW,
        user_message="我要模拟面试",
        context={"resume_review_id": "d0000001-0000-0000-0000-000000000001"},
    )
    print("agent_type   :", req.agent_type.value)
    print("tenant_id(默认):", req.tenant_id)
    print("thread_id    :", req.thread_id)   # 由 student_id + session_id 拼出
    print("context      :", req.context)


def check_response_models():
    section("③ AgentResponse / PipelineResult 构造")
    resp = AgentResponse(success=True, agent_type=AgentType.QA, content="RAG 是检索增强生成...")
    print("AgentResponse:", {"success": resp.success, "agent": resp.agent_type.value,
                             "content": resp.content[:12] + "...", "fallback": resp.fallback_used})
    pr = PipelineResult(steps=[resp], combined={"step_1": {"agent_type": "qa"}})
    print("PipelineResult.steps 数:", len(pr.steps), " combined 键:", list(pr.combined.keys()))


def check_orchestrator_init():
    section("④ Orchestrator 初始化：Pipeline 注册表 + 懒加载缓存")
    orch = Orchestrator()
    print("初始图缓存（懒加载，应为空）:", orch._agent_graphs)
    print("预置 Pipeline:")
    for key, seq in orch._pipelines.items():
        print(f"  {key:18s} → {[a.value for a in seq]}")


if __name__ == "__main__":
    check_enums()
    check_request()
    check_response_models()
    check_orchestrator_init()
    section("✅ 08-02 枚举 / Schema / 初始化 全部离线验证通过")
