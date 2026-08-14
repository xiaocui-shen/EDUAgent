# scripts/manual_tests/uc_08_07_e2e.py
# 运行：python scripts/manual_tests/uc_08_07_e2e.py

import sys
import json
import httpx

sys.path.insert(0, "scripts/manual_tests")
from itv_fixtures import load_env, seed_resume   # 复用第 7 章李明的简历 fixture

load_env()

BASE_URL   = "http://localhost:8000/api/v1"
STUDENT_UN = "student01"
STUDENT_PW = "Student@123456"

# 李明的五句话，依次触发：前置拦截 / qa / multi_agent / interview / 前置拦截
MESSAGES = [
    "你好",                      # → 前置拦截（HELLO 模板，零 Token）
    "什么是 RAG 的核心原理？",    # → LLM 路由 qa → QA Agent 流式作答
    "帮我准备求职",              # → LLM 路由 multi_agent → 求职全链路计划
    "我要模拟面试",              # → LLM 路由 interview → 引导跳转 /interview
    "谢谢",                      # → 前置拦截（THANKS 模板，零 Token）
]


def login(username: str, password: str) -> str:
    resp = httpx.post(f"{BASE_URL}/auth/login",
                      json={"username": username, "password": password}, trust_env=False)
    resp.raise_for_status()
    return resp.json()["access_token"]


def print_section(title: str):
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def stream_one(headers: dict, session_id: str, message: str):
    """对一句话发起 SSE 流式请求，按事件类型分别处理。"""
    print(f"\n👤 李明：{message}")
    print("🤖 助手：", end="", flush=True)

    with httpx.stream(                                   # 流式 POST，边收边处理
        "POST", f"{BASE_URL}/chat/stream",
        headers=headers,
        json={"session_id": session_id, "message": message},
        timeout=60.0,
    ) as resp:
        for line in resp.iter_lines():                  # 逐行读 SSE
            if not line.startswith("data:"):            # 只处理 data: 行
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            event = json.loads(data_str)                # 解析事件
            etype = event.get("type")

            if etype == "routing_decision":             # 路由决策卡片
                print(f"\n  〔路由→{event['agent_display']}"
                      f"（{event['execution_mode']}）：{event['reason']}〕")
                print("  ", end="", flush=True)
            elif etype == "progress":                   # QA 进度提示
                print(f"\n  …{event['stage']}", flush=True)
                print("  ", end="", flush=True)
            elif etype == "token":                      # 文本片段（模板回复 / QA 答案）
                print(event["content"], end="", flush=True)
            elif etype == "guidance":                   # 引导跳转卡片
                print(f"\n  〔引导：{event['message']}〕")
                if event.get("action_url"):
                    print(f"  〔按钮：{event['action_label']} → {event['action_url']}〕")
            elif etype == "pipeline_plan":              # 求职全链路计划
                print(f"\n  〔{event['title']}〕{event['intro']}")
                for step in event["steps"]:
                    print(f"    {step['step']}. {step['label']} → {step['action_url']}")
            elif etype == "meta":                       # QA 元数据（答案模式/来源）
                print(f"\n  〔来源：{event.get('sources', [])}〕")
            elif etype == "error":                      # 异常
                print(f"\n  ⚠ {event['message']}")
            # done 事件无需特殊处理，循环自然结束
    print()


if __name__ == "__main__":
    print_section("① 登录 + 写入李明简历（供求职链路使用）")
    token = login(STUDENT_UN, STUDENT_PW)
    headers = {"Authorization": f"Bearer {token}"}
    seeded = __import__("asyncio").run(seed_resume())   # 写入李明简历，resume_review_id 备用
    print(f"token: {token[:18]}...  resume_review_id: {seeded['resume_review_id']}")

    print_section("② 统一入口五句话，覆盖全部路由分支")
    session_id = "liming-unified-001"
    for msg in MESSAGES:
        stream_one(headers, session_id, msg)

    print_section("✅ 08-07 统一入口端到端：全分支验证完成")
