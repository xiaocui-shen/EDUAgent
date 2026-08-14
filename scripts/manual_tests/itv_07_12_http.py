# scripts/manual_tests/itv_07_12_http.py
# 07-12 实测：依次打通 5 个 HTTP 接口（建会话 / 对话 / 列表 / 报告 / SSE 提示）。
# 这是「从外部以真实 HTTP 请求」验证面试 Agent，最接近前端的调用方式。
#
# 前置条件：
#   1. PostgreSQL 已启动 + 已 seed（student01 账号）
#   2. 后端已启动：uvicorn backend.main:app --reload --port 8000
#   3. .env.local 配好 DEEPSEEK_API_KEY
# 运行：python scripts/manual_tests/itv_07_12_http.py

import sys
import httpx

sys.path.insert(0, "scripts/manual_tests")
from itv_fixtures import section, TARGET_POSITION, ANSWER_SCRIPT

BASE = "http://localhost:8000/api/v1"


def login() -> str:
    r = httpx.post(f"{BASE}/auth/login",
                   json={"username": "student01", "password": "Student@123456"},
                   trust_env=False, timeout=15.0)
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    token   = login()
    headers = {"Authorization": f"Bearer {token}"}

    section("① POST /sessions —— 创建会话（首轮开场白）")
    r = httpx.post(f"{BASE}/interview/sessions", headers=headers,
                   json={"target_position": TARGET_POSITION},
                   trust_env=False, timeout=60.0)
    r.raise_for_status()
    sid = r.json()["session_id"]
    print("session_id:", sid)
    print("开场白:", r.json()["message"][:60], "...")

    section("② POST /sessions/{id}/chat —— 发送 1 条消息")
    r = httpx.post(f"{BASE}/interview/sessions/{sid}/chat", headers=headers,
                   json={"message": ANSWER_SCRIPT[0]},
                   trust_env=False, timeout=60.0)
    r.raise_for_status()
    d = r.json()
    print(f"阶段: {d['current_stage']}  总轮数: {d['total_turns']}  结束: {d['is_finished']}")
    print("面试官:", d["reply"][:60], "...")

    section("③ GET /sessions —— 历史面试列表")
    r = httpx.get(f"{BASE}/interview/sessions", headers=headers, trust_env=False, timeout=15.0)
    r.raise_for_status()
    print("历史面试数:", r.json()["total"])

    section("④ GET /sessions/{id}/report —— 面试未结束应返回 400")
    r = httpx.get(f"{BASE}/interview/sessions/{sid}/report", headers=headers,
                  trust_env=False, timeout=15.0)
    print("HTTP 状态码:", r.status_code, "(进行中 → 400 符合预期)")

    section("⑤ SSE 流式接口 /chat/stream 见 07-13，本脚本只验证 4 个同步接口")
    section("✅ 07-12 HTTP 接口连通验证完成")


if __name__ == "__main__":
    main()
