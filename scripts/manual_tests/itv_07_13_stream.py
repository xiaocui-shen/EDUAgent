# scripts/manual_tests/itv_07_13_stream.py
# 单独验证 SSE 流式接口

import sys
import json
import httpx

sys.path.insert(0, ".")

BASE_URL = "http://localhost:8000/api/v1"


def login(username, password):
    resp = httpx.post(f"{BASE_URL}/auth/login",
                      json={"username": username, "password": password}, trust_env=False)
    resp.raise_for_status()
    return resp.json()["access_token"]


if __name__ == "__main__":
    token      = login("student01", "Student@123456")
    headers    = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # 先创建一个会话（与正常流程一致，AI 大模型岗）
    resp = httpx.post(
        f"{BASE_URL}/interview/sessions",
        headers=headers,
        json={"target_position": "AI大模型开发工程师"},
        timeout=30.0,
        trust_env=False
    )
    session_id = resp.json()["session_id"]
    print(f"session_id: {session_id}")
    print(f"开场白：{resp.json()['message'][:80]}")

    # 流式发送第一条消息（李明的自我介绍）
    print("\n--- 流式对话 ---")
    with httpx.stream(                                # 用流式方式发 POST，边收边处理
        "POST",
        f"{BASE_URL}/interview/sessions/{session_id}/chat/stream",
        headers=headers,
        json={"message": "你好，我是李明，Python/PyTorch/LangChain 方向，做过 RAG 问答和 LoRA 微调项目。"},
        timeout=30.0,
    ) as resp:
        for line in resp.iter_lines():               # 逐行读取 SSE 流
            if not line.startswith("data:"):         # SSE 每个事件以 "data:" 开头，其它行跳过
                continue
            data_str = line[len("data:"):].strip()   # 去掉 "data:" 前缀，取出 JSON 文本
            if not data_str:                         # 空行（事件分隔）跳过
                continue
            event = json.loads(data_str)             # 解析成事件 dict
            if event["type"] == "token":             # token 事件：面试官回复的一个片段
                print(event["content"], end="", flush=True)  # 实时打印（不换行）
            elif event["type"] == "done":            # done 事件：本轮结束 + 最终状态
                print(f"\n\n[done] 阶段={event['current_stage']} 总轮数={event['total_turns']}")
            elif event["type"] == "error":           # error 事件：流式出错
                print(f"\n[error] {event['message']}")
