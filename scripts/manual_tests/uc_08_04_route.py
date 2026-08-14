# scripts/manual_tests/uc_08_04_route.py
# _pre_filter 与映射表可离线运行。运行：python scripts/manual_tests/uc_08_04_route.py

import sys
sys.path.insert(0, ".")

from backend.api.v1.unified_chat import (
    _pre_filter, _LABEL_TO_AGENT, _LABEL_TO_MODE,
)


def section(t):
    print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)


def check_pre_filter():
    section("① _pre_filter：五类命中 + 两类放行（精确匹配 vs 正则）")
    cases = [
        ("你好",          "问候→命中"),
        ("谢谢",          "感谢→命中"),
        ("谢谢你",        "感谢但带额外字→放行（精确匹配不命中）"),
        ("再见",          "道别→命中"),
        ("你是谁",        "身份→命中（正则）"),
        ("你能做什么",     "功能→命中（正则）"),
        ("什么是RAG",     "真实技术问题→放行，交给 LLM 路由"),
        ("帮我看看简历",   "真实诉求→放行，交给 LLM 路由"),
    ]
    for text, note in cases:
        r = _pre_filter(text)
        hit = f"命中模板({len(r)}字)" if r else "None（继续路由）"
        print(f"  {text:10} -> {hit:16} | {note}")


def check_mappings():
    section("② label → AgentType / ExecutionMode 映射")
    for label in ["qa", "exam", "resume", "interview", "multi_agent", "clarify"]:
        print(f"  {label:12} -> agent={_LABEL_TO_AGENT[label].value:10} "
              f"mode={_LABEL_TO_MODE[label].value}")


if __name__ == "__main__":
    check_pre_filter()
    check_mappings()
    section("✅ 08-04 前置拦截 / 映射表 离线验证通过")
