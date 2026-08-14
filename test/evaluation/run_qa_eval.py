"""
11-2 QA Agent 评估 — 检索召回率 + 回答正确性
=============================================
教学目的：
  - 对应第5章（RAG QA 系统）
  - 直接调用 build_qa_graph()，读取内部 state 的 ranked_chunks 字段
  - 量化两个核心指标：Retrieval Recall@K 和 Answer Correctness

运行方式：
  PyCharm 右键运行，或：python test/evaluation/run_qa_eval.py

前置条件：
  - PostgreSQL + Milvus 已启动
  - 知识库已构建（scripts/build_knowledge_base.py）
  - gold/qa_gold.json 已准备（格式见课件 11.2.2）
"""

import sys
import os
import asyncio
import json
import time
import multiprocessing

# 支持 PyCharm 右键直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Windows 控制台默认 GBK，无法输出 emoji，强制 UTF-8
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from test.evaluation.eval_config import (
    load_gold, print_banner, print_verdict,
    RECALL_TARGET, CORRECTNESS_TARGET, PROJECT_ROOT,
)


# ── 配置 ────────────────────────────────────────────────────
GOLD_FILE = "qa_gold.json"
TOP_K = 3  # Recall@K 的 K 值


# ── Step 1：加载 Gold 数据集 ────────────────────────────────
def load_qa_gold() -> list[dict]:
    """
    加载评测数据集。每条格式：
    {
        "question": "什么是多态？",
        "gold_keywords": ["多态", "重写", "继承"],   # 检索命中判定关键词
        "gold_answer": "多态是面向对象的核心特性..."  # 标准答案（用于 Judge）
    }
    """
    data = load_gold(GOLD_FILE)
    if not data:
        print("\n  请先准备 gold/qa_gold.json，格式示例：")
        print('  [{"question": "什么是多态？",')
        print('    "gold_keywords": ["多态", "重写", "继承"],')
        print('    "gold_answer": "多态是指..."}]')
        sys.exit(0)
    return data


# ── Step 1.5：查询 student01 真实 UUID ─────────────────────
async def get_student_uuid() -> str:
    """
    从 users 表查出 student01 的真实 UUID。
    qa_sessions.student_id 是 UUID 类型，
    不能传自定义字符串（否则 asyncpg 报 DataError）。
    """
    from sqlalchemy import text
    from backend.dependencies import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        row = await session.execute(
            text("SELECT id FROM users "
                 "WHERE username = 'student01' LIMIT 1")
        )
        uid = row.scalar()
        if uid is None:
            print("  [ERROR] users 表无 student01，请先初始化数据")
            sys.exit(1)
        return str(uid)


# ── Step 2：运行 QA Graph 获取完整 state ────────────────────
async def run_qa_graph(question: str, session_id: str,
                       student_id: str) -> dict:
    """
    直接调用 QA Agent 的 LangGraph 图，返回完整 state。
    关键：state["ranked_chunks"] 包含检索+重排后的 Top-K 文档块。
    """
    from backend.agents.qa.graph import build_qa_graph

    graph = build_qa_graph()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "student_id": student_id,
        "tenant_id": "tenant_default",
        "session_id": session_id,
        "course_id": None,
        "query_type": "PRECISE",
        "enable_web_search": False,
        "web_search_results": [],
    }
    config = {"configurable": {"thread_id": f"eval_qa_{session_id}"}}

    state = await graph.ainvoke(initial_state, config=config)
    return state


# ── Step 3：计算 Recall@K ───────────────────────────────────
def check_recall(ranked_chunks: list[dict], gold_keywords: list[str], k: int = TOP_K) -> bool:
    """
    判定检索是否命中：Top-K 个 chunk 的 content 中是否包含 gold_keywords 的任意一个。

    ranked_chunks 每个元素是 dict，典型字段：
      - content / text / page_content：文档块文本
      - score / relevance_score：重排分数
      - metadata：来源信息
    """
    top_chunks = ranked_chunks[:k]

    # 提取所有 chunk 的文本内容（兼容不同字段名）
    chunk_texts = []
    for chunk in top_chunks:
        text = chunk.get("content") or chunk.get("text") or chunk.get("page_content") or ""
        chunk_texts.append(text.lower())

    # 检查 gold_keywords 是否出现在任意 chunk 中
    for kw in gold_keywords:
        kw_lower = kw.lower()
        for text in chunk_texts:
            if kw_lower in text:
                return True
    return False


# ── Step 4：LLM-as-Judge 判定回答正确性 ─────────────────────
async def judge_answer(question: str, gold_answer: str, agent_answer: str) -> bool:
    """
    用 LLM 判定 Agent 回答是否与标准答案语义一致。
    返回 True = Correct，False = Incorrect。
    """
    from backend.core.llm_factory import get_llm
    # 温度设为 0 ，让LLM 输出更稳定
    llm = get_llm("qa", temperature=0)
    prompt = f"""请判断以下 Agent 回答是否与标准答案语义一致（核心含义相同即可，不要求措辞完全一样）。

问题：{question}

标准答案：{gold_answer}

Agent 回答：{agent_answer}

请只输出一个词：Correct 或 Incorrect"""

    try:
        from langchain_core.messages import HumanMessage as HM
        resp = await llm.ainvoke([HM(content=prompt)])
        text = resp.content.strip().lower() if hasattr(resp, "content") else str(resp).strip().lower()
        return "correct" in text
    except Exception as e:
        print(f"    [WARN] LLM Judge 失败：{e}")
        return False


# ── Step 5：主评估流程 ──────────────────────────────────────
async def run_evaluation():
    print_banner("11-2  QA Agent 评估（Recall@K + Answer Correctness）")

    # 加载数据
    gold_data = load_qa_gold()
    total = len(gold_data)

    # 查出 student01 真实 UUID（qa_sessions 外键要求）
    student_id = await get_student_uuid()

    print(f"\n  评测数据集：{total} 条问题")
    print(f"  Recall@{TOP_K} 目标：≥ {RECALL_TARGET*100:.0f}%")
    print(f"  Correctness 目标：≥ {CORRECTNESS_TARGET*100:.0f}%")
    print("-" * 62)

    recall_hits = 0
    correct_hits = 0
    results = []

    for i, item in enumerate(gold_data):
        question = item["question"]
        gold_keywords = item.get("gold_keywords", [])
        gold_answer = item.get("gold_answer", "")

        print(f"\n  [{i+1}/{total}] {question[:40]}...")

        # 运行 QA Graph
        start = time.time()
        try:
            state = await run_qa_graph(
                question, session_id=f"eval_{i}",
                student_id=student_id)
        except Exception as e:
            print(f"    ❌ Graph 运行失败：{e}")
            results.append({"question": question, "recall": False, "correct": False, "error": str(e)})
            continue
        latency = round((time.time() - start) * 1000)

        # 指标1：Recall@K
        ranked_chunks = state.get("ranked_chunks", [])
        recall_ok = check_recall(ranked_chunks, gold_keywords, k=TOP_K)
        if recall_ok:
            recall_hits += 1
        print(f"    检索召回 : {'✅' if recall_ok else '❌'}  (Top-{TOP_K} 共 {len(ranked_chunks)} chunks, {latency}ms)")

        # 指标2：Answer Correctness
        agent_answer = state.get("answer", "")
        if gold_answer and agent_answer:
            correct_ok = await judge_answer(question, gold_answer, agent_answer)
        else:
            correct_ok = False
            print(f"    [WARN] 缺少 gold_answer 或 agent_answer，跳过 Judge")
        if correct_ok:
            correct_hits += 1
        print(f"    回答正确 : {'✅' if correct_ok else '❌'}")

        results.append({
            "question": question,
            "recall": recall_ok,
            "correct": correct_ok,
            "latency_ms": latency,
            "confidence": state.get("confidence", 0),
            "answer_mode": state.get("answer_mode", ""),
        })

    # ── 输出报告 ────────────────────────────────────────────
    recall_rate = recall_hits / total if total > 0 else 0
    correct_rate = correct_hits / total if total > 0 else 0

    print_banner("评估报告")
    print(f"\n  评测样本数 : {total}")
    print(f"\n  Retrieval Recall@{TOP_K} : {recall_rate*100:.1f}%  ({recall_hits}/{total})  目标 ≥ {RECALL_TARGET*100:.0f}%  {'✅' if recall_rate >= RECALL_TARGET else '❌'}")
    print(f"  Answer Correctness : {correct_rate*100:.1f}%  ({correct_hits}/{total})  目标 ≥ {CORRECTNESS_TARGET*100:.0f}%  {'✅' if correct_rate >= CORRECTNESS_TARGET else '❌'}")

    # 置信度分布
    confidences = [r["confidence"] for r in results if "confidence" in r]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"\n  平均检索置信度 : {avg_conf:.3f}")

    # 失败明细
    failed = [r for r in results if not r["recall"] or not r["correct"]]
    if failed:
        print(f"\n  未通过明细（{len(failed)} 条）：")
        for r in failed:
            flags = f"{'R❌' if not r['recall'] else 'R✅'} {'C❌' if not r['correct'] else 'C✅'}"
            print(f"    {flags}  {r['question'][:50]}")

    # 综合结论
    all_pass = recall_rate >= RECALL_TARGET and correct_rate >= CORRECTNESS_TARGET
    print_verdict(all_pass, pass_msg="QA Agent 质量达标", fail_msg="QA Agent 需优化")

    # 优化建议
    if not all_pass:
        print("  优化方向：")
        if recall_rate < RECALL_TARGET:
            print("    - Recall 不足 → 检查 Chunk 切分粒度 / Embedding 模型 / Hybrid Search 权重")
        if correct_rate < CORRECTNESS_TARGET:
            print("    - Correctness 不足 → 检查生成 Prompt / Rerank 阈值 / Context 构造")


def main():
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    # Windows 下 BGE-M3 的 DataLoader 用 spawn 多进程，
    # 必须 freeze_support 防止子进程重新导入主模块报错
    multiprocessing.freeze_support()
    main()
