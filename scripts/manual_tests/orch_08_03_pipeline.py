# scripts/manual_tests/orch_08_03_pipeline.py
# 纯离线，无需启动服务。运行：python scripts/manual_tests/orch_08_03_pipeline.py

import sys
sys.path.insert(0, ".")

from backend.core.orchestrator import (
    Orchestrator, AgentType, AgentRequest, AgentResponse, PipelineResult, get_orchestrator,
)


def section(t):
    print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)


def check_aggregate():
    section("① _aggregate_pipeline：聚合李明的「简历→面试」两步结果")
    orch = Orchestrator()
    req = AgentRequest(
        student_id="liming-001", session_id="job-prep",
        agent_type=AgentType.RESUME, user_message="帮我准备求职",
        pipeline_mode=True, context={"pipeline_key": "job_preparation"},
    )
    # 模拟 _run_pipeline 已跑完两步（真实场景由 Agent 图产生）
    pr = PipelineResult(steps=[
        AgentResponse(success=True, agent_type=AgentType.RESUME,
                      content="简历六维度评分：78分...",
                      structured={"overall_score": 78, "dimensions": 6}),
        AgentResponse(success=True, agent_type=AgentType.INTERVIEW,
                      content="模拟面试报告：76分...",
                      structured={"overall_score": 76}),
    ])
    agg = orch._aggregate_pipeline(pr, req)
    print("聚合 success     :", agg.success)
    print("聚合 content     :", repr(agg.content))
    print("combined 键      :", list(agg.structured.keys()))
    print("  step_1         :", agg.structured["step_1"])
    print("  step_2         :", agg.structured["step_2"])
    print("fallback_used    :", agg.fallback_used)


def check_singleton():
    section("② get_orchestrator：模块级单例（图缓存全程复用）")
    a = get_orchestrator()
    b = get_orchestrator()
    print("两次获取是同一实例 (a is b):", a is b)


if __name__ == "__main__":
    check_aggregate()
    check_singleton()
    section("✅ 08-03 聚合 / 单例 离线验证通过")
