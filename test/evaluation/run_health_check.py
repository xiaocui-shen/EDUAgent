"""
11-1 四 Agent 连通性检查（健康检查）
=====================================
教学目的：
  - 对应第3章（认证鉴权）和第8章（系统集成）
  - 验证后端已启动、JWT 鉴权正常、四个 Agent 路由可达
  - 作为后续 11-2 ~ 11-6 深入评估的前置门槛

运行方式：
  1. 启动后端：uvicorn backend.main:app --reload --port 8000
  2. PyCharm 右键本文件 → Run，或命令行：
     python test/evaluation/run_health_check.py
"""

import sys
import os
import time

# 支持 PyCharm 右键直接运行：将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx
from test.evaluation.eval_config import (
    API_PREFIX, STUDENT_USERNAME, STUDENT_PASSWORD,
    print_banner, print_verdict,
)


def check_login() -> tuple[str, int]:
    """
    Step 1：登录获取 JWT。
    返回 (access_token, latency_ms)。登录失败则 token 为空字符串。
    """
    start = time.time()
    try:
        resp = httpx.post(
            f"{API_PREFIX}/auth/login",
            json={"username": STUDENT_USERNAME, "password": STUDENT_PASSWORD},
            timeout=10.0,
        )
        latency = round((time.time() - start) * 1000)
        if resp.status_code == 200:
            token = resp.json().get("access_token", "")
            return token, latency
        else:
            print(f"    登录返回 HTTP {resp.status_code}：{resp.text[:100]}")
            return "", latency
    except Exception as e:
        latency = round((time.time() - start) * 1000)
        print(f"    登录异常：{e}")
        return "", latency


def check_endpoint(token: str, name: str, method: str, path: str) -> dict:
    """
    Step 2：对单个 Agent 端点发请求，记录状态码和延迟。
    判定逻辑：
      - HTTP < 500 且无连接异常 → 路由可达（\u2705）
      - HTTP >= 500 或连接失败  → 服务异常（\u274c）
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{API_PREFIX}{path}"
    start = time.time()

    try:
        resp = httpx.request(method, url, headers=headers, timeout=10.0)
        latency = round((time.time() - start) * 1000)
        ok = resp.status_code < 500
        return {
            "name": name,
            "ok": ok,
            "status": resp.status_code,
            "latency_ms": latency,
            "note": "" if ok else resp.text[:80],
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000)
        return {
            "name": name,
            "ok": False,
            "status": 0,
            "latency_ms": latency,
            "note": str(e)[:80],
        }


def main():
    print_banner("11-1  EduAgent 四 Agent 连通性检查")

    # ── Step 1：登录 ──────────────────────────────────────────
    print("\n  [Step 1] 登录鉴权...")
    token, login_ms = check_login()

    if not token:
        print(f"  登录鉴权   : \u274c  ({login_ms} ms)")
        print("\n  \u274c 登录失败，后续检查无法继续。")
        print("  请确认：")
        print("    1. 后端已启动：uvicorn backend.main:app --reload --port 8000")
        print("    2. 数据库已初始化：python scripts/seed_data.py")
        print(f"    3. 账号 {STUDENT_USERNAME} 存在且密码正确")
        print_verdict(False, fail_msg="系统不可用")
        return

    print(f"  登录鉴权   : \u2705  ({login_ms} ms)")

    # ── Step 2：逐端点检查 ────────────────────────────────────
    print("\n  [Step 2] 检查四个 Agent 端点...")

    # 端点列表：(显示名, HTTP方法, 路径)
    # QA 没有裸列表接口，用 /sessions/{id}/history 探活（404=路由存在）
    endpoints = [
        ("QA Agent",    "GET", "/qa/sessions/_health_check/history"),
        ("Resume Agent", "GET", "/resume/reviews"),
        ("Exam Agent",  "GET", "/exam/my-submissions"),
        ("Interview",   "GET", "/interview/sessions"),
    ]

    results = []
    for name, method, path in endpoints:
        r = check_endpoint(token, name, method, path)
        results.append(r)
        flag = "\u2705" if r["ok"] else "\u274c"
        status_info = f"HTTP {r['status']}" if r["status"] > 0 else "连接失败"
        print(f"  {r['name']:<14}: {flag}  {status_info}  ({r['latency_ms']} ms)")
        if r["note"]:
            print(f"               {r['note']}")

    # ── Step 3：综合结论 ──────────────────────────────────────
    all_ok = all(r["ok"] for r in results)
    print_verdict(all_ok, pass_msg="系统就绪，可进入 11-2 ~ 11-6 深入评估")


if __name__ == "__main__":
    main()
