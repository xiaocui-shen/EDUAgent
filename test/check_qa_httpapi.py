# test/check_qa_httpapi.py

import json
import httpx


BASE_URL  = "http://localhost:8000/api/v1"
USERNAME  = "student01"          # 或 "student01@eduagent.local"
PASSWORD  = "Student@123456"
SESSION_A = "qa-test-session-001"   # 普通测试会话
SESSION_B = "qa-test-session-002"   # 多轮记忆测试会话


def login() -> str:
    resp = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        trust_env=False,   # 禁止读取系统代理，避免 macOS 代理导致 502
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"[login] token 获取成功（前20字符）: {token[:20]}...")
    return token


def chat(token: str, message: str, session_id: str = SESSION_A,
         enable_web_search: bool = False) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/qa/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_id":        session_id,
            "message":           message,
            "enable_web_search": enable_web_search,
        },
        timeout=90.0,
        trust_env=False,
    )
    resp.raise_for_status()
    return resp.json()


def stream_chat(token: str, message: str, session_id: str = SESSION_A) -> None:
    """打印 SSE 流式事件，每行一条"""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/qa/chat/stream",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "text/event-stream",
        },
        json={"session_id": session_id, "message": message},
        timeout=60.0,
        trust_env=False,
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                data = line[5:].strip()
                try:
                    print(json.loads(data))
                except json.JSONDecodeError:
                    print(data)


def print_result(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  answer_mode : {result['answer_mode']}")
    print(f"  confidence  : {result['confidence']:.4f}")
    print(f"  sources     : {result['sources']}")
    print(f"  answer      : {result['answer'][:200]}...")
    print(f"{'='*60}")


# scripts/manual_tests/test_qa.py（续）

if __name__ == "__main__":
    token = login()

    # ──────────────────────────────────────────────────────────
    # 场景①  GENERAL 路径：打招呼
    # 预期：answer_mode="general"，无📚来源，不走检索
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景① GENERAL 路径")
    r = chat(token, "你好，请介绍一下你自己")
    print_result("GENERAL", r)
    assert r["answer_mode"] == "general", f"预期 general，实际 {r['answer_mode']}"
    assert r["confidence"] == 1.0
    assert "📚" not in r["answer"]
    print("✅ GENERAL 路径通过")

    # ──────────────────────────────────────────────────────────
    # 场景②  PRECISE 路径：具体技术问题
    # 预期：answer_mode="rag"，confidence≥0.75，sources 非空
    # 日志关注：classify_query.rag_strategy=PRECISE → retrieve.done
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景② PRECISE 路径")
    r = chat(token, "课程中关于HNSW的剪枝怎么实现？")
    print_result("PRECISE", r)
    assert r["answer_mode"] == "rag", f"预期 rag，实际 {r['answer_mode']}"
    assert r["confidence"] >= 0.75, f"置信度过低：{r['confidence']}"
    assert len(r["sources"]) > 0, "RAG 回答应有来源"
    assert "📚" in r["answer"]
    print("✅ PRECISE 路径通过")

    # ──────────────────────────────────────────────────────────
    # 场景③  VAGUE 路径：模糊追问（需在 PRECISE 之后同一会话）
    # 预期：日志出现 hyde_generate.done，比模糊 Query 直接检索效果更好
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景③ VAGUE 路径（模糊追问）")
    r = chat(token, "课程没懂")
    print_result("VAGUE", r)
    # VAGUE 可能高置信度（rag）也可能低置信度（llm_direct），关键看日志
    print("  ⚠️  请在后端日志确认：hyde_generate.done 被触发")
    print("✅ VAGUE 路径已发送，请核对日志")

    # ──────────────────────────────────────────────────────────
    # 场景④  BROAD 路径：宽泛问题
    # 预期：日志出现 multi_query_rewrite.done，rewritten_queries 有多条子 Query
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景④ BROAD 路径（宽泛问题）")
    r = chat(token, "课程里HNSW全面总结一下核心概念")
    print_result("BROAD", r)
    print("  ⚠️  请在后端日志确认：multi_query_rewrite.done，queries 有 2-3 条")
    print("✅ BROAD 路径已发送，请核对日志")


    # ──────────────────────────────────────────────────────────
    # 场景⑤  低置信度：知识库没有的内容
    # 预期：answer_mode="llm_direct"，回答末尾含 ⚠️ 说明
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景⑤ 低置信度（知识库无相关内容）")
    r = chat(token, "向量数据库的高可用架构怎么设计")
    print_result("低置信度", r)
    assert r["answer_mode"] in ("llm_direct", "rag"), f"answer_mode={r['answer_mode']}"
    if r["answer_mode"] == "llm_direct":
        assert "⚠️" in r["answer"], "llm_direct 回答应含 ⚠️ 说明"
        print("✅ 低置信度路径通过（llm_direct + ⚠️ 说明）")
    else:
        print(f"  知识库中有相关内容（confidence={r['confidence']:.4f}），场景未触发，换一个冷门问题重试")

    # ──────────────────────────────────────────────────────────
    # 场景⑥  Web 兜底：同一问题开启联网
    # 预期：answer_mode="web_augmented"，sources 含 URL
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景⑥ Web 兜底（enable_web_search=True）")
    r = chat(token,
             "向量数据库的高可用架构怎么设计",
             enable_web_search=True)
    print_result("Web兜底", r)
    if r["answer_mode"] == "web_augmented":
        assert len(r["sources"]) > 0, "web_augmented 应有 URL 来源"
        print("✅ Web 兜底路径通过（web_augmented + URL 来源）")
    else:
        print(f"  answer_mode={r['answer_mode']}，Web Search MCP 可能未启动，检查 .env.local 的 WEB_SEARCH_MCP_URL")


    # ──────────────────────────────────────────────────────────
    # 场景⑦  多轮记忆：第2轮问题引用第1轮答案
    # 使用独立会话 SESSION_B，避免和前面的测试互相干扰
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景⑦ 多轮记忆")

    print("  第1轮：问HNSW剪枝")
    r1 = chat(token,
              "课程中HNSW的剪枝怎么实现",
              session_id=SESSION_B)
    print_result("第1轮", r1)

    print("\n  第2轮：用代词引用第1轮内容（'其中'指代课程中HNSW剪枝）")
    r2 = chat(token,
              "其中有哪些核心概念",
              session_id=SESSION_B)
    print_result("第2轮", r2)

    # 第2轮回答里应该提到HNSW或剪枝，说明历史被续接
    keywords = ["HNSW", "剪枝"]
    found = any(kw in r2["answer"] for kw in keywords)
    assert found, f"第2轮回答未提到相关词，记忆续接可能失败：{r2['answer'][:200]}"
    print("✅ 多轮记忆通过（第2轮正确理解代词引用）")

    # 验证历史接口
    import httpx as _httpx
    hist = _httpx.get(
        f"{BASE_URL}/qa/sessions/{SESSION_B}/history",
        headers={"Authorization": f"Bearer {token}"},
        trust_env=False,
    ).json()
    assert hist["total_turns"] == 2, f"预期2轮，实际 {hist['total_turns']}"
    print(f"✅ 历史接口通过（total_turns={hist['total_turns']}）")


    # ──────────────────────────────────────────────────────────
    # 场景⑧  SSE 流式
    # ──────────────────────────────────────────────────────────
    print("\n>>> 场景⑧ SSE 流式接口")
    print("  （观察事件顺序：progress → token × N → meta → done）")
    stream_chat(token,
                "简短介绍一下python，10字以内",
                session_id="stream-test-001")
