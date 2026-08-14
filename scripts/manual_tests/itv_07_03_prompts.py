# scripts/manual_tests/itv_07_03_prompts.py
# 07-03 实测：把每个 Prompt 用真实简历 / 题目 / 答案 .format() 渲染出来，
# 直观看到「最终送进 LLM 的文本长什么样」。纯离线，无需启动服务。
# 运行：python scripts/manual_tests/itv_07_03_prompts.py

from itv_fixtures import (section, TARGET_POSITION, RESUME_PROJECTS, ANSWER_SCRIPT)


from backend.agents.interview.prompts import (
    SYSTEM_PROMPT, TECH_BASE_GENERATE_PROMPT, WARMUP_PROMPT,
    INTRO_EVAL_TECH_FIRST_PROMPT, TECH_BASE_PROMPT, PROJECT_PROMPT,
    CLOSING_PROMPT, STAGE_TRANSITION_PROMPTS,
    EVALUATE_THINK_PROMPT, EVALUATE_ANSWER_PROMPT, GENERATE_REPORT_PROMPT,
)

Q_TRANSFORMER = "请解释 Transformer 中自注意力机制的计算过程。"


def show(name: str, rendered: str):
    print(f"\n----- {name} -----")
    print(rendered.strip())


def main():
    section("① SYSTEM_PROMPT（注入每次 LLM 调用，固定面试官角色）")
    show("SYSTEM_PROMPT", SYSTEM_PROMPT.format(position=TARGET_POSITION))

    section("② TECH_BASE_GENERATE_PROMPT（首轮让 LLM 动态出 8 道岗位题）")
    show("动态出题", TECH_BASE_GENERATE_PROMPT.format(position=TARGET_POSITION, count=8))

    section("③ WARMUP opening（开场白邀请自我介绍）")
    show("WARMUP[opening]", WARMUP_PROMPT["opening"].format(position=TARGET_POSITION))

    section("④ INTRO_EVAL_TECH_FIRST（评价自我介绍 + 出第一题，仅过渡轮用一次）")
    show("过渡专用", INTRO_EVAL_TECH_FIRST_PROMPT.format(
        intro=ANSWER_SCRIPT[0],
        position=TARGET_POSITION,
        first_question=Q_TRANSFORMER,
    ))

    section("⑤ TECH_BASE ask_with_feedback（上题反馈 + 出下一题，技术环节主流程）")
    show("ask_with_feedback", TECH_BASE_PROMPT["ask_with_feedback"].format(
        question=Q_TRANSFORMER,
        answer=ANSWER_SCRIPT[1],
        quality="excellent",
        next_question="过拟合和欠拟合分别是什么？如何缓解过拟合？",
    ))

    section("⑥ PROJECT new_project（深挖简历第一个项目，项目数据注入为内部参考）")
    proj = RESUME_PROJECTS[0]
    show("new_project", PROJECT_PROMPT["new_project"].format(
        project_name=proj["name"],
        project_role=proj["role"],
        tech_stack=", ".join(proj["tech_stack"]),
        highlights="\n".join(f"- {h}" for h in proj["highlights"]),
        description=proj["description"][:200],
    ))

    section("⑦ STAGE_TRANSITION_PROMPTS（静态过渡语，不调用 LLM）")
    print("可用的过渡键：", list(STAGE_TRANSITION_PROMPTS.keys()))
    for k, v in STAGE_TRANSITION_PROMPTS.items():
        print(f"  [{k}] {v}")

    section("⑧ Think Tool 两步：EVALUATE_THINK（自由推理）→ EVALUATE_ANSWER（打标签）")
    show("第一步 EVALUATE_THINK", EVALUATE_THINK_PROMPT.format(
        question=Q_TRANSFORMER, answer=ANSWER_SCRIPT[1]))
    show("第二步 EVALUATE_ANSWER", EVALUATE_ANSWER_PROMPT.format(
        question=Q_TRANSFORMER, answer=ANSWER_SCRIPT[1]))

    section("⑨ GENERATE_REPORT（面试结束生成五维度报告）")
    show("GENERATE_REPORT（节选前 400 字）", GENERATE_REPORT_PROMPT.format(
        position=TARGET_POSITION,
        total_turns=13,
        history_summary="（无历史摘要）",
        conversation="面试官：... \n学员：...",
    )[:400])

    section("✅ 07-03 全部 Prompt 渲染成功")


if __name__ == "__main__":
    main()
