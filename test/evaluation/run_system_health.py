"""
11-6 系统级健康度评估
=====================
教学目的：
  - 从「系统整体」视角评估，不再针对单个 Agent
  - 指标1：P95 延迟（95% 请求在多少 ms 内返回）
  - 指标2：兜底率（触发降级回答的比例）
  - 指标3（可选）：Langfuse token 用量统计

运行方式：
  PyCharm 右键运行，或：
  python test/evaluation/run_system_health.py

前置条件：
  - 后端服务已启动（uvicorn backend.main:app）
  - PostgreSQL + Milvus 已启动
  - gold/qa_gold.json 已准备（复用 11-2 的问题）
"""

import sys
import os
import asyncio
import time
import statistics

# 支持 PyCharm 右键直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Windows 控制台默认 GBK，强制 UTF-8
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8",
    errors="replace")
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding="utf-8",
    errors="replace")

from test.evaluation.eval_config import (
    load_gold, print_banner, print_verdict,
    login, auth_headers, API_PREFIX,
    P95_TARGET_MS, FALLBACK_RATE_TARGET,
)

# ── 配置 ───────────────────────────────────────────────
GOLD_FILE = "qa_gold.json"
# 每个问题重复测几轮（凑够样本量算 P95）
ROUNDS = 2


# ── Step 1：发送单次 QA 请求 ───────────────────────────
async def send_qa_request(
        client, headers: dict,
        question: str,
        session_id: str) -> tuple:
    """
    调用 POST /api/v1/qa/chat（非流式），
    返回 (latency_ms, fallback_used, answer_mode)。
    """
    payload = {
        "session_id": session_id,
        "message": question,
        "course_id": None,
        "enable_web_search": False,
    }

    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{API_PREFIX}/qa/chat",
            json=payload,
            headers=headers,
            timeout=60.0,
        )
        latency_ms = round(
            (time.perf_counter() - start) * 1000)

        if resp.status_code != 200:
            print(f"\n    [WARN] HTTP "
                  f"{resp.status_code}")
            return latency_ms, True, "error"

        data = resp.json()
        # answer_mode 为 llm_direct 或
        # fallback_used=True → 没走 RAG，触发兜底
        mode = data.get("answer_mode", "")
        fallback = (
            data.get("fallback_used", False)
            or mode == "llm_direct"
        )
        return latency_ms, fallback, mode

    except Exception as e:
        latency_ms = round(
            (time.perf_counter() - start) * 1000)
        print(f"\n    [ERROR] 请求失败：{e}")
        return latency_ms, True, "error"


# ── Step 2：批量压测，收集延迟数据 ─────────────────────
async def run_latency_test(
        questions: list) -> list:
    """
    对每个问题发 ROUNDS 轮请求，
    收集所有 (latency_ms, fallback, mode)。
    """
    import httpx

    # 登录拿 token
    print("  登录获取 Token...")
    token = login()
    headers = auth_headers(token)

    results = []
    total = len(questions) * ROUNDS
    idx = 0

    async with httpx.AsyncClient() as client:
        for r in range(ROUNDS):
            for q_item in questions:
                idx += 1
                question = q_item["question"]
                sid = f"health_{r}_{idx}"
                print(f"  [{idx}/{total}] "
                      f"{question[:25]}...",
                      end=" ", flush=True)

                lat, fb, mode = (
                    await send_qa_request(
                        client, headers,
                        question, sid))
                results.append({
                    "latency_ms": lat,
                    "fallback": fb,
                    "mode": mode,
                })
                flag = "✅" if not fb else "⚠️"
                print(f"{lat}ms {flag} ({mode})")

    return results


# ── Step 3：计算 P95 + 兜底率 ──────────────────────────
def compute_health_metrics(
        results: list) -> dict:
    """从结果列表计算 P95 延迟和兜底率"""
    latencies = sorted(
        r["latency_ms"] for r in results)
    n = len(latencies)

    # P95：第 95 百分位的延迟值
    p95_idx = int(n * 0.95)
    p95_idx = min(p95_idx, n - 1)
    p95 = latencies[p95_idx]

    # 辅助分位数
    p50 = latencies[n // 2]
    avg = round(statistics.mean(latencies))

    # 兜底率
    fallback_count = sum(
        1 for r in results if r["fallback"])
    fallback_rate = fallback_count / n

    return {
        "total": n,
        "p50_ms": p50,
        "p95_ms": p95,
        "avg_ms": avg,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
    }


# ── Step 4（可选）：Langfuse 用量统计 ──────────────────
def fetch_langfuse_stats() -> dict | None:
    """
    从 Langfuse 拉取全部 GENERATION 观测的 token 用量。
    Langfuse v4：client.api.legacy.observations_v1.get_many()
    服务端 type='GENERATION' 过滤 + 分页（每页最多100）。
    Langfuse 未启动或无数据时返回 None。
    """
    try:
        from langfuse import get_client
        client = get_client()

        total_input = 0
        total_output = 0
        gen_count = 0
        page = 1

        # 分页拉取所有 GENERATION
        while True:
            resp = (
                client.api.legacy.observations_v1.get_many(
                    limit=100,
                    type="GENERATION",
                    page=page))
            if not resp.data:
                break
            for obs in resp.data:
                if obs.usage:
                    total_input += (
                        obs.usage.input or 0)
                    total_output += (
                        obs.usage.output or 0)
                    gen_count += 1
            page += 1

        if gen_count == 0:
            return None

        return {
            "trace_count": gen_count,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }
    except Exception as e:
        print(f"  [INFO] Langfuse 不可用：{e}")
        return None


# ── Step 5：输出报告 ───────────────────────────────────
def print_report(metrics: dict,
                 langfuse_stats: dict | None):
    """打印系统健康度报告"""
    print_banner("系统健康度报告")

    # 延迟指标
    p95 = metrics["p95_ms"]
    p95_pass = p95 <= P95_TARGET_MS
    print(f"\n  请求样本数 : {metrics['total']}")
    print(f"  平均延迟   : {metrics['avg_ms']} ms")
    print(f"  P50 延迟   : {metrics['p50_ms']} ms")
    print(f"  P95 延迟   : {p95} ms"
          f"  目标 ≤ {P95_TARGET_MS} ms"
          f"  {'✅' if p95_pass else '❌'}")

    # 兜底率
    fb_rate = metrics["fallback_rate"]
    fb_pass = fb_rate <= FALLBACK_RATE_TARGET
    print(f"\n  兜底次数   : "
          f"{metrics['fallback_count']}"
          f"/{metrics['total']}")
    print(f"  兜底率     : {fb_rate:.1%}"
          f"  目标 ≤ {FALLBACK_RATE_TARGET:.0%}"
          f"  {'✅' if fb_pass else '❌'}")

    # Langfuse 用量统计
    if langfuse_stats:
        print(f"\n  ── Langfuse Token 用量"
              f"（最近 "
              f"{langfuse_stats['trace_count']}"
              f" 次 LLM 调用）──")
        print(f"  输入 tokens: "
              f"{langfuse_stats['total_input_tokens']:,}")
        print(f"  输出 tokens: "
              f"{langfuse_stats['total_output_tokens']:,}")
        print(f"  合计 tokens: "
              f"{langfuse_stats['total_tokens']:,}")
    else:
        print("\n  Langfuse 统计：不可用"
              "（未启动或无数据）")

    # 综合结论
    all_pass = p95_pass and fb_pass
    print_verdict(
        all_pass,
        pass_msg="系统健康度达标",
        fail_msg="系统健康度不达标")

    if not all_pass:
        print("  优化方向：")
        if not p95_pass:
            print("    - P95 过高 → 检查 BGE-M3 推理"
                  " / Milvus 查询 / LLM 响应")
        if not fb_pass:
            print("    - 兜底率过高 → 检查知识库覆盖"
                  " / Rerank 阈值 / 检索召回")


# ── 主流程 ─────────────────────────────────────────────
async def main(mode: str = "all"):
    """
    mode:
      "latency"  — 仅压测（P95 + 兜底率）
      "langfuse" — 仅 Langfuse token 统计
      "all"      — 压测 + Langfuse（默认）
    """
    print_banner(
        "11-6  系统级健康度评估"
        "（P95 + 兜底率 + Token）")
    print(f"\n  运行模式：{mode}")

    metrics = None
    langfuse_stats = None

    # ── 压测部分 ──────────────────────────────────────
    if mode in ("latency", "all"):
        gold_data = load_gold(GOLD_FILE)
        if not gold_data:
            print("  请先准备 gold/qa_gold.json")
            sys.exit(1)

        print(f"\n  测试问题数：{len(gold_data)}")
        print(f"  重复轮次  ：{ROUNDS}")
        print(f"  总请求数  ："
              f"{len(gold_data) * ROUNDS}")
        print(f"  P95 目标  ：≤ {P95_TARGET_MS} ms")
        print(f"  兜底率目标：≤ "
              f"{FALLBACK_RATE_TARGET:.0%}")
        print("-" * 62)

        print("\n  开始压测...")
        results = await run_latency_test(gold_data)
        metrics = compute_health_metrics(results)

    # ── Langfuse 部分 ─────────────────────────────────
    if mode in ("langfuse", "all"):
        print("\n  拉取 Langfuse 统计...")
        langfuse_stats = fetch_langfuse_stats()

    # ── 输出报告 ──────────────────────────────────────
    if metrics:
        print_report(metrics, langfuse_stats)
    elif langfuse_stats:
        # langfuse-only 模式：单独输出 token 统计
        print_banner("Langfuse Token 用量报告")
        print(f"\n  LLM 调用次数: "
              f"{langfuse_stats['trace_count']}")
        print(f"  输入 tokens  : "
              f"{langfuse_stats['total_input_tokens']:,}")
        print(f"  输出 tokens  : "
              f"{langfuse_stats['total_output_tokens']:,}")
        print(f"  合计 tokens  : "
              f"{langfuse_stats['total_tokens']:,}")
        print()
    else:
        print("\n  [WARN] 无数据可输出。")


if __name__ == "__main__":
    # asyncio.run(main(mode="langfuse")) # 仅Token统计
    # asyncio.run(main(mode="latency")) # 仅压测
    asyncio.run(main(mode="all")) # 全部
