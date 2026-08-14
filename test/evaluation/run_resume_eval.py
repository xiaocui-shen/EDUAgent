"""
11-4 简历审查 Agent 评估 — 问题召回率 + 评分一致性
=============================================
教学目的：
  - 对应第4章（简历审查 Agent）
  - 直接调用 build_resume_graph()，读取内部 state
  - 量化两个核心指标：Issue Recall（问题召回率）+ Score MAE（评分一致性）

核心思想：
  简历审查的价值在于「发现问题」和「给出可信评分」。
  所以评估逻辑是：让专家先审查一份简历（独立标注问题和分数）
  → 让 AI 审查同一份简历 → 对比两者。
    Issue Recall = AI 发现的问题数 / 专家标注的问题数
    Score MAE    = |AI 综合分 - 专家综合分|

运行方式：
  PyCharm 右键运行，或：python test/evaluation/run_resume_eval.py

前置条件：
  - PostgreSQL 已启动（save_results 节点要写库）
  - tmp/简历示例.pdf 存在（待审查的简历）
  - gold/resume_gold.json 已准备（专家标注，格式见课件 11.4.2）

⚠️ 安全提示：
  save_results 节点执行完会删除 pdf_local_path 指向的文件。
  所以本脚本先把原始简历复制成临时副本，再把副本路径传给图，
  绝不直接传原始文件，避免误删用户简历。
"""

import sys
import os
import asyncio
import shutil
import uuid

# 支持 PyCharm 右键直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage

from test.evaluation.eval_config import (
    load_gold, print_banner, print_verdict,
    ISSUE_RECALL_TARGET, SCORE_MAE_TARGET, PROJECT_ROOT,
)


# ── 配置 ────────────────────────────────────────────────────
GOLD_FILE = "resume_gold.json"
# 简历存放目录（tmp/ 下）
RESUME_DIR = PROJECT_ROOT / "tmp"


# ── Step 1：加载专家标注 Gold 数据 ──────────────────────────
def load_resume_gold() -> list:
    """
    加载专家标注数据，返回简历条目列表。
    每条格式（见 gold/resume_gold.json）：
    {
        "resume_file": "简历示例.pdf",      # tmp/ 下的简历文件名
        "expert_score": 78,                 # 专家给的综合分（0-100）
        "expert_issues": ["项目缺少量化指标", ...]  # 专家发现的问题
    }
    """
    data = load_gold(GOLD_FILE)
    if not data:
        print("\n  请先准备 gold/resume_gold.json，格式示例：")
        print('  [{"resume_file": "简历示例.pdf",')
        print('    "expert_score": 78,')
        print('    "expert_issues": ["项目缺少量化指标", ...]}]')
        sys.exit(0)
    return data


# ── Step 2：运行简历审查图 ──────────────────────────────────
async def run_resume_graph(pdf_path: str) -> dict:
    """
    直接调用简历审查 Agent 的 LangGraph 图，返回完整 state。
    关键：
      - 传入的是临时副本路径（save_results 会删除该路径文件）
      - review_id 用随机 UUID：save_results 的 UPDATE 匹配不到记录，
        相当于空操作，不会污染数据库
    返回的 state 里：
      - weighted_score：六维度加权综合分（0-100）
      - dimension_scores：六个维度各自的得分
      - issues：逐条问题诊断列表
    """
    from backend.agents.resume.graph import build_resume_graph

    graph = build_resume_graph()

    initial_state = {
        "messages": [],
        "student_id": "eval_student",
        "tenant_id": "tenant_default",
        "review_id": str(uuid.uuid4()),   # 假 ID，UPDATE 空操作
        "pdf_minio_path": "",             # 本地模式，留空
        "pdf_local_path": pdf_path,       # 临时副本路径
    }

    # 简历图无 interrupt、无 checkpointer，一次跑完
    state = await graph.ainvoke(initial_state)
    return state


# ── Step 3：LLM 判定单个专家问题是否被 AI 发现 ──────────────
async def judge_issue_found(expert_issue: str,
                            agent_issues_text: str) -> bool:
    """
    用 LLM 判定：专家发现的这个问题，AI 的问题列表里有没有
    语义相同或相近的一条。返回 True = AI 发现了。
    """
    from backend.core.llm_factory import get_llm
    # 温度设为 0 ，让LLM 输出更稳定
    llm = get_llm("resume", temperature=0)
    prompt = f"""请判断以下专家发现的简历问题，是否被 AI 审查出来了。
（只要 AI 的问题列表中有一条与专家问题语义相同或相近，就算发现）

专家发现的问题：{expert_issue}

AI 发现的问题列表：
{agent_issues_text}

请只输出一个词：Yes 或 No"""

    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = resp.content.strip().lower()
        return "yes" in text
    except Exception as e:
        print(f"    [WARN] LLM Judge 失败：{e}")
        return False


# ── Step 4：计算单份简历的 Issue Recall ─────────────────────
async def compute_issue_recall(expert_issues: list,
                               agent_issues: list) -> tuple:
    """
    对每个专家问题，判定 AI 是否发现。
    返回 (recall, matched_count, total)。
    """
    # 把 AI 的问题描述拼成文本（供 Judge 阅读）
    agent_issues_text = "\n".join(
        f"- {item.get('description', '')}" for item in agent_issues
    ) or "（AI 未发现问题）"

    matched = 0
    for expert_issue in expert_issues:
        found = await judge_issue_found(expert_issue, agent_issues_text)
        flag = "✅" if found else "❌"
        print(f"      {flag} 专家问题：{expert_issue[:30]}")
        if found:
            matched += 1

    total = len(expert_issues)
    recall = matched / total if total > 0 else 0.0
    return recall, matched, total


# ── Step 5：主评估流程 ──────────────────────────────────────
async def run_evaluation():
    print_banner("11-4  简历审查 Agent 评估（Issue Recall + Score MAE）")

    # 加载专家标注
    gold_data = load_resume_gold()
    total = len(gold_data)
    print(f"\n  专家标注简历数：{total} 份")
    print(f"  Issue Recall 目标：≥ {ISSUE_RECALL_TARGET*100:.0f}%")
    print(f"  Score MAE 目标：≤ {SCORE_MAE_TARGET:.1f} 分")

    recalls = []       # 每份简历的 Issue Recall
    score_errors = []  # 每份简历的评分误差
    results = []

    for i, item in enumerate(gold_data):
        resume_file = item["resume_file"]
        expert_score = item["expert_score"]
        expert_issues = item.get("expert_issues", [])

        print(f"\n  [{i+1}/{total}] 审查简历：{resume_file}")

        src_pdf = RESUME_DIR / resume_file
        if not src_pdf.exists():
            print(f"    ❌ 简历文件不存在：{src_pdf}")
            results.append({"resume": resume_file, "error": "文件不存在"})
            continue

        # 复制成临时副本（保护原始文件，save_results 只删副本）
        tmp_pdf = RESUME_DIR / f"_eval_resume_{uuid.uuid4().hex}.pdf"
        shutil.copy2(src_pdf, tmp_pdf)

        try:
            state = await run_resume_graph(str(tmp_pdf))
        except Exception as e:
            print(f"    ❌ 图运行失败：{e}")
            results.append({"resume": resume_file, "error": str(e)})
            # 图若中途失败，副本可能没被清理，这里补删
            if tmp_pdf.exists():
                os.remove(tmp_pdf)
            continue

        # 提取 AI 结果
        weighted_score = state.get("weighted_score", 0)
        agent_issues = state.get("issues", [])
        dimension_scores = state.get("dimension_scores", [])

        print(f"    AI 综合分 : {weighted_score:.1f} / 100"
              f"（专家 {expert_score}）")
        print(f"    AI 问题数 : {len(agent_issues)} 条"
              f"（专家 {len(expert_issues)} 条）")

        # 六维度得分明细（供参考）
        print(f"    六维度得分:")
        for d in dimension_scores:
            print(f"      {d.get('dimension', ''):<8} "
                  f"{d.get('score', 0):>3}  "
                  f"(权重 {d.get('weight', 0):.2f})")

        # 指标1：Issue Recall
        print(f"    Issue Recall 判定:")
        recall, matched, n = await compute_issue_recall(
            expert_issues, agent_issues)
        print(f"    → 召回 {matched}/{n} = {recall*100:.1f}%")

        # 指标2：评分误差
        score_err = abs(weighted_score - expert_score)
        print(f"    → 评分误差 = |{weighted_score:.1f} - "
              f"{expert_score}| = {score_err:.1f} 分")

        recalls.append(recall)
        score_errors.append(score_err)
        results.append({
            "resume": resume_file,
            "recall": recall,
            "score_error": score_err,
            "weighted_score": weighted_score,
            "expert_score": expert_score,
        })

    # ── 输出报告 ────────────────────────────────────────────
    if not recalls:
        print("\n  ❌ 没有成功评估的简历，请检查前置条件。")
        return

    avg_recall = sum(recalls) / len(recalls)
    score_mae = sum(score_errors) / len(score_errors)

    print_banner("评估报告")
    print(f"\n  评估简历数 : {len(recalls)}")
    print(f"\n  Issue Recall : {avg_recall*100:.1f}%  "
          f"目标 ≥ {ISSUE_RECALL_TARGET*100:.0f}%  "
          f"{'✅' if avg_recall >= ISSUE_RECALL_TARGET else '❌'}")
    print(f"  Score MAE    : {score_mae:.2f} 分  "
          f"目标 ≤ {SCORE_MAE_TARGET:.1f}  "
          f"{'✅' if score_mae <= SCORE_MAE_TARGET else '❌'}")

    all_pass = (avg_recall >= ISSUE_RECALL_TARGET
                and score_mae <= SCORE_MAE_TARGET)
    print_verdict(
        all_pass,
        pass_msg="简历审查 Agent 质量达标",
        fail_msg="简历审查 Agent 需优化",
    )

    # 优化建议
    if not all_pass:
        print("  优化方向：")
        if avg_recall < ISSUE_RECALL_TARGET:
            print("    - Recall 不足 → 检查 diagnose_issues Prompt / "
                  "六维度评审的问题挖掘深度")
        if score_mae > SCORE_MAE_TARGET:
            print("    - MAE 偏高 → 检查六维度评分 Prompt / 维度权重设置")


def main():
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
