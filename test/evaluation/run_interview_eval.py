"""
11-5 模拟面试 Agent 评估 — 流程完整性 + 报告质量
=============================================
教学目的：
  - 对应第7章（模拟面试 Agent）
  - 直接调用 build_interview_graph()，用 MemorySaver 跑多轮对话
  - 用固定答题脚本（scripted student）驱动一场完整面试
  - 量化两个核心指标：流程完整性（五阶段状态机）+ 报告质量（LLM-as-Judge）

核心思想：
  模拟面试是多轮对话 Agent，它的价值在于：
    1. 面试流程是否完整（热身→技术基础→项目深挖→反问收尾→结束）
    2. 最终的五维度评估报告质量如何
  所以评估逻辑是：用固定答题脚本扮演「学员」，跑完一场面试，
  追踪阶段推进，面试结束后评估报告质量。

运行方式：
  PyCharm 右键运行，或：python test/evaluation/run_interview_eval.py

前置条件：
  - PostgreSQL 已启动
  - 已运行 scripts/seed_data.py（student01 账号存在）
  - 已运行 scripts/seed_interview_questions.py（题库存在）
  - gold/interview_gold.json 已准备（简历 + 答题脚本，格式见课件 11.5.2）
"""

import sys
import os
import asyncio
import json
import re
import uuid

# 支持 PyCharm 右键直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage

from test.evaluation.eval_config import (
    load_gold, print_banner, print_verdict,
    REPORT_QUALITY_TARGET,
)

# Windows 控制台 GBK 编码兜底：报告含特殊符号时不崩
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── 配置 ────────────────────────────────────────────────────
GOLD_FILE = "interview_gold.json"
# 面试应经历的五个阶段（状态机取值）
EXPECTED_STAGES = ["warmup", "tech_base", "project",
                   "closing", "finished"]
STAGE_NAMES = {
    "warmup": "热身",
    "tech_base": "技术基础",
    "project": "项目深挖",
    "closing": "反问收尾",
    "finished": "面试结束",
}


# ── Step 1：加载 Gold 数据 ──────────────────────────────────
def load_interview_gold() -> dict:
    """
    加载评测数据，格式（见 gold/interview_gold.json）：
    {
        "target_position": "AI大模型开发工程师",
        "resume": {...完整简历结构化数据...},
        "answer_script": ["自我介绍", "技术题回答", ...]  # 学员答题脚本
    }
    """
    data = load_gold(GOLD_FILE)
    if not data:
        print("\n  请先准备 gold/interview_gold.json，格式见课件 11.5.2")
        sys.exit(0)
    return data


# ── Step 2：准备 DB 数据（简历 + 面试会话）──────────────────
async def setup_interview_db(resume_structured: dict,
                             target_position: str) -> dict:
    """
    准备面试所需的 DB 数据（复刻第7章测试 fixture 的做法）：
      1. 查出 student01 的真实 UUID
      2. 写入一份简历（resume_reviews.structured_data，
         供 PROJECT 阶段项目深挖联动）
      3. 创建一条面试会话（interview_sessions, in_progress）
    返回 {"student_id", "session_id", "resume_review_id"}。
    """
    from sqlalchemy import text
    from backend.dependencies import AsyncSessionLocal
    from backend.core.memory import build_thread_id

    # ① 查 student01 真实 UUID
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("SELECT id FROM users "
                 "WHERE username = 'student01' LIMIT 1")
        )).scalar()
        if not row:
            raise RuntimeError(
                "student01 不存在，请先运行 scripts/seed_data.py")
        student_id = str(row)

    resume_review_id = str(uuid.uuid4())
    session_id = f"eval_itv_{uuid.uuid4().hex[:8]}"
    thread_id = build_thread_id(student_id, session_id)

    async with AsyncSessionLocal() as s:
        async with s.begin():
            # ② 写入简历（structured_data 存完整简历结构）
            await s.execute(
                text("""
                    INSERT INTO resume_reviews
                        (id, tenant_id, student_id, pdf_minio_path,
                         structured_data, status)
                    VALUES
                        (:id, 'tenant_default', :sid, '',
                         CAST(:data AS JSONB), 'done')
                """),
                {"id": resume_review_id, "sid": student_id,
                 "data": json.dumps(resume_structured,
                                    ensure_ascii=False)},
            )
            # ③ 创建面试会话
            await s.execute(
                text("""
                    INSERT INTO interview_sessions
                        (id, tenant_id, student_id, session_id,
                         thread_id, target_position,
                         resume_review_id, status)
                    VALUES
                        (:id, 'tenant_default', :sid, :sessid,
                         :tid, :target, :rrid, 'in_progress')
                    ON CONFLICT (thread_id) DO NOTHING
                """),
                {"id": str(uuid.uuid4()), "sid": student_id,
                 "sessid": session_id, "tid": thread_id,
                 "target": target_position,
                 "rrid": resume_review_id},
            )

    return {
        "student_id": student_id,
        "session_id": session_id,
        "resume_review_id": resume_review_id,
    }


# ── Step 3：驱动面试对话 ────────────────────────────────────
def last_ai_text(messages) -> str:
    """取最后一条 AIMessage 的内容（面试官说的话）。"""
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage":
            return m.content
    return ""


async def start_interview(graph, config, ctx,
                          target_position) -> dict:
    """首轮：传入完整 initial_state，启动面试。"""
    initial_state = {
        "messages": [HumanMessage(content="[开始面试]")],
        "student_id": ctx["student_id"],
        "tenant_id": "tenant_default",
        "session_id": ctx["session_id"],
        "target_position": target_position,
        "resume_review_id": ctx["resume_review_id"],
        "resume_projects": [], "resume_skills": [],
        "current_stage": "warmup",
        "stage_turn_count": 0, "total_turn_count": 0,
        "max_turns": 40,
        "question_bank": [], "current_question": None,
        "projects_asked": [],
        "last_answer_quality": "adequate", "followup_count": 0,
        "existing_summary": None, "should_summarize": False,
        "report": None, "fallback_used": False,
        "structured_output": None,
    }
    return await graph.ainvoke(initial_state, config=config)


async def send_answer(graph, config, ctx, answer) -> dict:
    """后续轮：只传增量消息，其余字段由 MemorySaver 自动恢复。"""
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=answer)],
         "student_id": ctx["student_id"],
         "session_id": ctx["session_id"],
         "tenant_id": "tenant_default"},
        config=config,
    )


# ── Step 4：报告结构检查 + LLM 质量评分 ─────────────────────
def check_report_structure(report: dict) -> tuple:
    """检查报告结构是否完整，返回 (是否通过, 说明)。"""
    if not report:
        return False, "报告为空"
    problems = []
    dims = report.get("dimensions", [])
    if len(dims) < 4:
        problems.append(f"仅 {len(dims)} 个维度")
    score = report.get("overall_score", -1)
    if not (0 <= score <= 100):
        problems.append("综合分越界")
    if not report.get("strengths"):
        problems.append("缺优势")
    if not report.get("improvements"):
        problems.append("缺改进方向")
    if not report.get("recommended_topics"):
        problems.append("缺推荐复习")
    if problems:
        return False, "；".join(problems)
    return True, f"{len(dims)} 个维度齐全"


def build_transcript(messages) -> str:
    """把对话消息列表转成可读文本（供 Judge 阅读）。"""
    lines = []
    for m in messages:
        role = m.__class__.__name__
        if role == "AIMessage":
            lines.append(f"面试官：{m.content}")
        elif role == "HumanMessage" and m.content != "[开始面试]":
            lines.append(f"学员：{m.content}")
    return "\n".join(lines)


async def judge_report_quality(transcript: str,
                               report: dict) -> int:
    """
    LLM-as-Judge：评估面试报告质量，返回 0-100 分。
    评分标准：是否基于真实对话、评语是否具体有据、
    评分是否合理、建议是否可操作。
    """
    from backend.core.llm_factory import get_llm

    llm = get_llm("interview", temperature=0)
    dims_text = json.dumps(
        [{"dimension": d.get("dimension"),
          "score": d.get("score"),
          "comment": d.get("comment")}
         for d in report.get("dimensions", [])],
        ensure_ascii=False)
    prompt = f"""你是「模拟面试报告的质量审核员」。你的唯一任务是评估下面这份 AI 生成的面试评估报告本身的写作质量。

⚠️⚠️⚠️ 核心原则（必须遵守）：
你评的是「报告写得好不好」，绝对不是「候选人面试表现好不好」。
- 候选人答得烂 + 报告准确指出薄弱点 = 高质量报告（应给高分）
- 候选人答得好 + 报告空洞无物 = 低质量报告（应给低分）
请把自己当作论文审稿人：你审的是论文的论证质量，不是实验结果好不好看。

【面试对话记录（供你核实报告是否有据可依）】
{transcript}

【AI 生成的评估报告（这才是你的评审对象）】
综合分：{report.get("overall_score")}
维度评估：{dims_text}
优势：{report.get("strengths")}
改进方向：{report.get("improvements")}
推荐复习：{report.get("recommended_topics")}

评估标准（全部针对报告文本本身）：
1. 有据性：报告的每条评语是否能在对话记录中找到对应依据（不凭空编造）
2. 具体性：维度评语是否引用了候选人的具体回答，而非泛泛而谈
3. 合理性：分数档位是否与对话中实际表现匹配（差→低分=合理=好报告）
4. 可操作性：改进建议是否具体到知识点/行动项，而非"加强学习"这类空话

再次强调：低分报告 ≠ 低质量报告。准确反映问题的报告就是好报告。

请只输出一个 0-100 的整数分数（评估报告写作质量，越高越好）："""

    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = (resp.content.strip()
                if hasattr(resp, "content") else str(resp))
        nums = re.findall(r"\d+", text)  # 提取回复中的数字
        if nums:
            return min(int(nums[0]), 100)
        return 0
    except Exception as e:
        print(f"    [WARN] LLM Judge 失败：{e}")
        return 0


# ── Step 5：主评估流程 ──────────────────────────────────────
async def run_evaluation():
    print_banner("11-5  模拟面试 Agent 评估（流程完整性 + 报告质量）")

    # 加载 Gold
    gold = load_interview_gold()
    target_position = gold["target_position"]
    resume = gold["resume"]
    answer_script = list(gold["answer_script"])
    script_len = len(answer_script)
    # 兜底：脚本用完仍未结束，用中性回答撑到 MAX_TURNS 强制推进
    # ⚠️ 不能写"请结束面试"——会触发 agent 终止信号，跳过后续阶段
    # 最坏情况总轮数 = MAX之和(4+12+14+4=34) - 真实答案14条 = 至少20条兜底
    answer_script += ["这个问题我不太了解，能换一个方向吗？"] * 20

    print(f"\n  目标岗位：{target_position}")
    print(f"  答题脚本：{script_len} 轮（+20 轮兜底）")
    print(f"  期望阶段：{' → '.join(EXPECTED_STAGES)}")

    # 准备 DB
    print("\n  准备 DB 数据（简历 + 面试会话）...")
    ctx = await setup_interview_db(resume, target_position)
    print(f"    student_id       = {ctx['student_id']}")
    print(f"    session_id       = {ctx['session_id']}")
    print(f"    resume_review_id = {ctx['resume_review_id']}")

    # 构建图 + config（thread_id 与 load_context 内部一致）
    from backend.agents.interview.graph import build_interview_graph
    from backend.core.memory import build_thread_id
    graph = build_interview_graph()
    config = {"configurable": {"thread_id": build_thread_id(
        ctx["student_id"], ctx["session_id"])}}

    # 首轮
    print("\n  开始面试...")
    result = await start_interview(graph, config, ctx,
                                   target_position)
    visited = [result.get("current_stage")]
    print(f"    [开场] 阶段={result.get('current_stage')}  "
          f"面试官：{last_ai_text(result['messages'])[:40]}...")

    # 多轮对话
    finished = result.get("current_stage") == "finished"
    for i, answer in enumerate(answer_script, start=1):
        if finished:
            break
        result = await send_answer(graph, config, ctx, answer)
        stage = result.get("current_stage")
        if stage not in visited:
            visited.append(stage)
        turns = result.get("total_turn_count", 0)
        print(f"    [第{i}轮] 阶段={stage:<10} 总轮数={turns}  "
              f"面试官：{last_ai_text(result['messages'])[:28]}...")
        if stage == "finished":
            finished = True

    # ── 指标1：流程完整性 ───────────────────────────────────
    reached = [s for s in EXPECTED_STAGES if s in visited]
    completeness = len(reached) / len(EXPECTED_STAGES)
    report = result.get("report")
    flow_ok = (completeness == 1.0 and finished
               and report is not None)

    # ── 指标2：报告质量 ─────────────────────────────────────
    struct_ok, struct_msg = check_report_structure(report)
    if report:
        transcript = build_transcript(result["messages"])
        quality = await judge_report_quality(transcript, report)
    else:
        quality = 0
    report_ok = struct_ok and quality >= REPORT_QUALITY_TARGET

    # ── 输出报告 ────────────────────────────────────────────
    print_banner("模拟面试 Agent 评估报告")

    print(f"\n  【流程完整性】")
    print(f"    经历阶段：{' → '.join(STAGE_NAMES.get(s, s) for s in reached)}")
    missing = [STAGE_NAMES.get(s, s)
               for s in EXPECTED_STAGES if s not in visited]
    if missing:
        print(f"    缺失阶段：{' / '.join(missing)}")
    print(f"    完整度：{len(reached)}/5 = {completeness*100:.0f}%  "
          f"{'✅' if completeness == 1.0 else '❌'}")
    print(f"    正常结束：{'✅' if finished else '❌'}    "
          f"报告生成：{'✅' if report else '❌'}")

    print(f"\n  【报告质量】")
    print(f"    结构检查：{'✅' if struct_ok else '❌'}（{struct_msg}）")
    if report:
        print(f"    模拟面试综合评分：{report.get('overall_score')} / 100")
        print(f"\n    ── 报告全文 ──")
        for d in report.get("dimensions", []):
            print(f"    [{d.get('dimension')}] {d.get('score')} 分")
            print(f"      评语：{d.get('comment', '')}")
            hl = d.get("highlights", [])
            wk = d.get("weaknesses", [])
            if hl:
                print(f"      亮点：{'；'.join(hl)}")
            if wk:
                print(f"      薄弱：{'；'.join(wk)}")
        print(f"\n    优势：{report.get('strengths', [])}")
        print(f"    改进方向：{report.get('improvements', [])}")
        print(f"    推荐复习：{report.get('recommended_topics', [])}")
        print(f"    综合评语：{report.get('overall_comment', '')}")
        print(f"    备考建议：{report.get('next_step_advice', '')}")
        print(f"    ── 报告全文结束 ──\n")
    print(f"    LLM 生成报告质量评分：{quality} / 100  "
          f"（目标 ≥ {REPORT_QUALITY_TARGET}）  "
          f"{'✅' if quality >= REPORT_QUALITY_TARGET else '❌'}")

    all_pass = flow_ok and report_ok
    print_verdict(
        all_pass,
        pass_msg="模拟面试 Agent 质量达标",
        fail_msg="模拟面试 Agent 需优化",
    )

    if not all_pass:
        print("  优化方向：")
        if not flow_ok:
            print("    - 流程不完整 → 检查 check_stage 阶段推进逻辑 / "
                  "提示词中的阶段转换条件")
        if not report_ok:
            print("    - 报告质量不足 → 检查 generate_report 评估 Prompt / "
                  "五维度评估结构")


def main():
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
