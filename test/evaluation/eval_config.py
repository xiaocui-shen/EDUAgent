# test/evaluation/eval_config.py
"""
第11章 评估体系 - 共享配置与工具函数
所有评估脚本通过本模块获取统一配置，避免各文件重复定义。
"""

import os
import sys
import json
from pathlib import Path

# 项目根目录（test/evaluation/ 的上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env.local（DB / LLM / Langfuse 等配置）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env.local")

# 后端地址
BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000")
API_PREFIX = f"{BASE_URL}/api/v1"

# 测试账号（seed_data.py 写入）
STUDENT_USERNAME = "student01"
STUDENT_PASSWORD = "Student@123456"

# Gold 数据目录
GOLD_DIR = Path(__file__).resolve().parent / "gold"

# ── 评估阈值 ─────────────────────────────────────────────
# 11-2 QA Agent
# 检索召回率：Top-K 命中 gold 关键词的比例 ≥ 90%
RECALL_TARGET = 0.90
# 回答正确率：LLM Judge 判定语义一致的比例 ≥ 85%
CORRECTNESS_TARGET = 0.85
# 11-3 试卷批改 Agent
# 平均绝对误差：AI 与教师打分平均差 ≤ 5 分
MAE_TARGET = 5.0
# Cohen's κ：AI 与教师等级一致性 ≥ 0.6（中等以上）
KAPPA_TARGET = 0.6
# 11-4 简历审查 Agent
# 问题召回率：专家标注的简历问题被 AI 发现的比例 ≥ 60%
ISSUE_RECALL_TARGET = 0.60
# 评分 MAE：AI 加权总分与专家分平均差 ≤ 5 分
SCORE_MAE_TARGET = 5.0
# 11-5 模拟面试 Agent
# 报告质量分：LLM Judge 对面试报告的评分 ≥ 80
REPORT_QUALITY_TARGET = 80
# 11-6 系统级健康度
# P95 延迟：95% 请求响应时间 ≤ 8 秒
P95_TARGET_MS = 8000
# 兜底率：触发降级/兜底回答的比例 ≤ 5%
FALLBACK_RATE_TARGET = 0.05


def load_gold(filename: str):
    """加载 gold/ 目录下的 JSON 评测数据。"""
    filepath = GOLD_DIR / filename
    if not filepath.exists():
        print(f"  [WARN] Gold 数据文件不存在：{filepath}")
        print(f"         请先按课件格式准备该文件。")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def login(username: str = STUDENT_USERNAME, password: str = STUDENT_PASSWORD) -> str:
    """登录获取 JWT access_token。"""
    import httpx
    resp = httpx.post(
        f"{API_PREFIX}/auth/login",
        json={"username": username, "password": password},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def auth_headers(token: str) -> dict:
    """构造带 Bearer Token 的请求头。"""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def print_banner(title: str):
    """打印评估报告标题横幅。"""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def print_verdict(passed: bool, pass_msg: str = "达标", fail_msg: str = "不达标"):
    """打印最终结论。"""
    sep = "=" * 62
    print(f"\n{sep}")
    if passed:
        print(f"  综合结论：{pass_msg} \u2705")
    else:
        print(f"  综合结论：{fail_msg} \u274c")
    print(f"{sep}\n")
