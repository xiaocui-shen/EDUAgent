# scripts/manual_tests/itv_07_02_state.py
# 07-02 实测：InterviewState / 枚举 / 报告 Pydantic 模型
# 纯离线，无需启动任何服务。运行：python scripts/manual_tests/itv_07_02_state.py

from itv_fixtures import section, base_state, RESUME_PROJECTS

from backend.agents.interview.state import (
    InterviewState, InterviewStage, AnswerQuality,
    DimensionEval, InterviewReport,
)


def dtest_enums():
    section("① 两个枚举：状态机的取值空间")
    print("InterviewStage（面试阶段）:", [s.value for s in InterviewStage])
    print("AnswerQuality（回答质量）:", [q.value for q in AnswerQuality])
    # 继承 str，可直接当字符串比较
    print("\nInterviewStage.WARMUP == 'warmup' ?", InterviewStage.WARMUP == "warmup")
    print("终态判断：FINISHED.value =", InterviewStage.FINISHED.value)


def dtest_state():
    section("② InterviewState：构造一个完整的面试状态快照")
    state = base_state(
        current_stage=InterviewStage.TECH_BASE.value,
        stage_turn_count=3,
        total_turn_count=3,
        resume_projects=RESUME_PROJECTS,
        last_answer_quality=AnswerQuality.EXCELLENT.value,
    )
    print(f'state: {state}')
    print("State 字段总数（TypedDict 注解数）:", len(InterviewState.__annotations__))
    print("实际构造的字段数:", len(state))
    print("\n关键字段当前值：")
    for k in ("current_stage", "stage_turn_count", "total_turn_count",
              "last_answer_quality", "max_turns"):
        print(f"  {k:20s} = {state[k]}")
    print(f"  resume_projects      = {len(state['resume_projects'])} 个项目"
          f"（{state['resume_projects'][0]['name']} ...）")


def dtest_report_models():
    section("③ 两层报告模型：DimensionEval → InterviewReport")

    # 第一层：单维度
    dim = DimensionEval(
        dimension="技术深度",
        score=82,
        comment="Transformer 与 RAG 原理掌握扎实，能讲清注意力计算与混合检索。",
        highlights=["自注意力公式准确", "RAG 流程完整"],
        weaknesses=["自回归生成原理未答出"],
    )
    print("DimensionEval 实例：")
    print(" ", dim.model_dump())

    # 第二层：五维度汇总（直接作为 with_structured_output 的目标类型）
    report = InterviewReport(
        dimensions=[dim],
        overall_score=78,
        strengths=["RAG 流程理解清晰", "项目有量化数据支撑"],
        improvements=["补强自回归语言模型原理", "加强抗压临场应变"],
        overall_comment="整体达到初级 AI 开发岗要求，项目经验是亮点。",
        recommended_topics=["自回归生成", "LoRA 原理", "向量检索"],
        next_step_advice="针对薄弱知识点系统复习后，1-2 周内再次模拟面试。",
    )
    print("\nInterviewReport 校验通过，overall_score =", report.overall_score)

    # 模拟 generate_report_node 的取值方式
    report_dict = report.model_dump()
    print("\nreport.model_dump() 的顶层键：")
    print(" ", list(report_dict.keys()))


if __name__ == "__main__":
    dtest_enums()
    dtest_state()
    dtest_report_models()
    section("✅ 07-02 全部对象构造与校验通过")
