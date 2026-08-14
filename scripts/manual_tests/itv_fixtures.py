# scripts/manual_tests/itv_fixtures.py
# ─────────────────────────────────────────────────────────────────────────
# 模拟面试 Agent —— 全章节共享测试 fixture（支架）

# 第 7 章 07-02 ~ 07-13 的每个测试脚本都从这里导入，使用同一份「AI 大模型开发岗」
# 简历和同一套学员答题脚本。这样从 07-02 读到 07-13，就像在看同一场面试逐步展开：
#   07-02 看 State 长什么样 → 07-05 看阶段如何推进 → 07-06 看回答如何被评分
#   → 07-07/08 看面试官如何提问 → 07-09 看报告如何生成 → 07-11/13 看完整跑通

# 设计要点：
#   - RESUME_STRUCTURED：模拟「简历审查 Agent」产出的结构化结果。面试 Agent 不读 PDF，
#     只读 resume_reviews.structured_data 里的 projects 和 skills_list 两个键。
#   - ANSWER_SCRIPT：一名学员从自我介绍到结束面试的固定回答序列。
#   - base_state / make_state_*：构造各阶段的 InterviewState 快照，喂给单个节点测试。
#   - setup_db_fixture / seed_resume：把简历和会话写进 DB，供需要数据库的节点测试。

# 运行测试脚本时，本文件位于 scripts/manual_tests/ 下，与各测试脚本同目录，
# 直接 `from itv_fixtures import ...` 即可（Python 会把脚本所在目录加入 sys.path）。
# ────────────────────────────────────────────────────────────────────────

import sys
import uuid

# 项目根目录加入 sys.path，供 `import backend.*` 使用
sys.path.insert(0, ".")

from langchain_core.messages import HumanMessage, AIMessage


# ──────────────────────────────────────────────────────────────
# 一、统一测试身份
# ──────────────────────────────────────────────────────────────
TENANT_ID       = "tenant_default"
# 离线测试（不写 DB）直接用这个合成 ID；写 DB 的测试会改用 student01 的真实 UUID
STUDENT_ID      = "stu-itv-test-0001"
TARGET_POSITION = "AI大模型开发工程师"


def new_session_id() -> str:
    """每次测试用新的 session_id，避免 MemorySaver / DB 旧数据干扰。"""
    return f"itvtest-{uuid.uuid4().hex[:8]}"


def load_env():
    """加载 .env.local（DeepSeek key / DB 连接等）。"""
    from dotenv import load_dotenv
    load_dotenv(".env.local")


def section(title: str):
    """打印分节标题，让测试输出更易读。"""
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


# ──────────────────────────────────────────────────────────────
# 二、模拟简历（AI 大模型开发岗）
#     结构 == 简历审查 Agent 的 ResumeStructured.model_dump()
# ──────────────────────────────────────────────────────────────

RESUME_STRUCTURED = {
    "name":            "李明",
    "phone":           "138****6677",
    "email":           "liming@example.com",
    "target_position": "AI大模型开发工程师",
    "education": [
        {
            "school":   "华东师范大学",
            "major":    "计算机科学与技术",
            "degree":   "本科",
            "duration": "2021.09 - 2025.06",
            "gpa":      "3.7/4.0",
        }
    ],
    "skills_raw": "Python、PyTorch、LangChain、Transformer、RAG、LoRA 微调、Milvus、FastAPI、Prompt Engineering",
    "skills_list": [
        "Python", "PyTorch", "LangChain", "Transformer",
        "RAG", "LoRA微调", "Milvus", "FastAPI", "Prompt Engineering",
    ],
    "projects": [
        {
            "name":       "企业知识库 RAG 问答系统",
            "role":       "后端开发（核心）",
            "duration":   "2024.03 - 2024.09",
            "tech_stack": ["Python", "LangChain", "Milvus", "BGE-M3", "FastAPI"],
            "description": "面向企业内部文档的检索增强问答系统，支持 PDF/Word 解析、"
                           "稠密+稀疏混合检索与 BGE-Reranker 精排，最终接入大模型生成答案。",
            "highlights": [
                "知识库覆盖 5 万+ 文档分块（chunk）",
                "引入混合检索 + Reranker 后，检索准确率从 78% 提升到 92%",
                "P95 响应时间控制在 1.2s 以内",
            ],
        },
        {
            "name":       "客服领域大模型 LoRA 微调",
            "role":       "算法开发",
            "duration":   "2024.10 - 2025.01",
            "tech_stack": ["PyTorch", "PEFT", "LoRA", "DeepSpeed"],
            "description": "在 7B 开源基座模型上做 LoRA 微调以适配客服问答领域，"
                           "构建了 1.2 万条领域指令数据集。",
            "highlights": [
                "可训练参数仅占全参数的 0.3%",
                "领域问答准确率相比基座模型提升约 15%",
            ],
        },
    ],
    "work_experience": [],
    "certificates":    ["CET-6", "PyTorch 开发者认证"],
    "self_intro":      "对大模型应用开发有浓厚兴趣，熟悉 RAG 与微调全流程。",
}


# 面试 State 直接使用的两个派生字段
RESUME_PROJECTS = RESUME_STRUCTURED["projects"]
RESUME_SKILLS   = RESUME_STRUCTURED["skills_list"]


# ──────────────────────────────────────────────────────────────
# 三、学员答题脚本（一场完整面试的全部回答）
# ──────────────────────────────────────────────────────────────
ANSWER_SCRIPT = [
    # [0] WARMUP：自我介绍
    "你好，我是李明，华东师范大学计算机专业应届生。主要技术栈是 Python、PyTorch 和 "
    "LangChain，做过一个企业知识库 RAG 问答系统，还有一个客服领域的大模型 LoRA 微调项目，"
    "对大模型应用开发很感兴趣。",

    # [1] TECH：优秀回答（Transformer，应判 excellent）
    "Transformer 的核心是自注意力机制。Encoder 由多头自注意力和前馈网络堆叠而成，"
    "每个子层都有残差连接和 LayerNorm；Decoder 多了一个带 mask 的自注意力，防止看到未来 token，"
    "还有一个对 Encoder 输出做交叉注意力的子层。注意力计算是 softmax(QK^T/√d_k)·V。",

    # [2] TECH：基本回答（过拟合，缺深度，应判 adequate）
    "过拟合就是模型在训练集上表现很好，但在测试集上效果差。可以用正则化、dropout、"
    "增加数据来缓解。",

    # [3] TECH：明确不会（应判 no_answer，走快速路径）
    "不知道",

    # [4] TECH：优秀回答（RAG，应判 excellent）
    "RAG 是检索增强生成。先把文档切块、向量化存进向量库，用户提问时把问题也向量化，"
    "做相似度检索召回 top-k 文档，再把文档和问题拼进 prompt 交给大模型生成答案。"
    "好处是能基于最新的、私有的知识回答，显著减少幻觉。",

    # [5] PROJECT：项目背景与职责
    "这个 RAG 系统是我主导后端开发的。背景是公司内部文档太多、检索困难。我负责文档解析、"
    "切块策略、混合检索和 Reranker 精排这部分，知识库大概有 5 万多个 chunk。",

    # [6] PROJECT：深挖（混合检索细节）
    "我用 BGE-M3 同时生成稠密向量和稀疏向量，稠密向量走 Milvus 的 HNSW 索引，"
    "稀疏向量做关键词匹配，两路召回后再用 BGE-Reranker 精排取 top-5。"
    "加了精排之后准确率从 78% 提升到了 92%。",

    # [7] CLOSING：学员反问
    "想了解一下团队目前在大模型应用上主要的技术方向，以及新人会接触到哪些项目？",

    # [8] 强制结束
    "好的，没有其他问题了，结束面试吧。",
]


# ──────────────────────────────────────────────────────────────
# 四、固定题库（模拟 load_context 出题结果，供离线测试使用）
# ──────────────────────────────────────────────────────────────
def sample_question_bank() -> list:
    """一份固定的技术题库，每项结构与 load_context 产出的 question_bank 一致。"""
    qs = [
        ("请解释 Transformer 中自注意力机制的计算过程。",      ["Transformer", "注意力机制"]),
        ("过拟合和欠拟合分别是什么？如何缓解过拟合？",          ["机器学习基础"]),
        ("请描述自回归语言模型的文本生成原理。",                ["语言模型"]),
        ("RAG（检索增强生成）的基本流程是怎样的？",             ["RAG"]),
        ("向量相似度检索中，余弦相似度的原理是什么？",          ["向量检索"]),
        ("LoRA 微调的核心思想是什么？",                         ["LoRA", "微调"]),
        ("Encoder 和 Decoder 在结构上有什么区别？",            ["Transformer"]),
        ("什么是 Prompt Engineering？常见技巧有哪些？",         ["Prompt"]),
    ]
    return [
        {"id": f"q_{i}", "content": c, "difficulty": "medium", "tags": t, "asked": False}
        for i, (c, t) in enumerate(qs)
    ]


# ──────────────────────────────────────────────────────────────
# 五、InterviewState 快照构造器
# ──────────────────────────────────────────────────────────────
def base_state(**overrides) -> dict:
    """构造一个字段完整的 InterviewState（dict），可用关键字参数覆盖任意字段。"""
    state = {
        "messages":            [],
        "student_id":          STUDENT_ID,
        "tenant_id":           TENANT_ID,
        "session_id":          new_session_id(),
        "target_position":     TARGET_POSITION,
        "resume_review_id":    None,
        "resume_projects":     [],
        "resume_skills":       [],
        "current_stage":       "warmup",
        "stage_turn_count":    0,
        "total_turn_count":    0,
        "max_turns":           40,
        "question_bank":       [],
        "current_question":    None,
        "projects_asked":      [],
        "last_answer_quality": "adequate",
        "followup_count":      0,
        "existing_summary":    None,
        "should_summarize":    False,
        "report":              None,
        "fallback_used":       False,
        "structured_output":   None,
    }
    state.update(overrides)
    return state


def make_state_tech_first() -> dict:
    """刚从 WARMUP 推进到 TECH_BASE 的第 1 轮（current_question 还是 None）。
    用于触发 _respond_tech_base 的「评价自我介绍 + 出第一题」路径。"""
    msgs = [
        HumanMessage(content="[开始面试]"),
        AIMessage(content=f"欢迎来到 {TARGET_POSITION} 岗位的模拟面试！请先做个简单的自我介绍。"),
        HumanMessage(content=ANSWER_SCRIPT[0]),
    ]
    return base_state(
        current_stage="tech_base",
        stage_turn_count=1,
        total_turn_count=1,
        current_question=None,
        question_bank=sample_question_bank(),
        resume_projects=RESUME_PROJECTS,
        resume_skills=RESUME_SKILLS,
        messages=msgs,
    )


def make_state_tech_followup() -> dict:
    """技术环节：当前题已问、学员给出 excellent 回答，触发追问路径。"""
    bank = sample_question_bank()
    bank[0]["asked"] = True
    cur = bank[0]
    msgs = [
        AIMessage(content=cur["content"]),
        HumanMessage(content=ANSWER_SCRIPT[1]),
    ]
    return base_state(
        current_stage="tech_base",
        stage_turn_count=3,
        total_turn_count=3,
        current_question=cur,
        question_bank=bank,
        last_answer_quality="excellent",
        followup_count=0,
        messages=msgs,
    )


def make_state_tech_next() -> dict:
    """技术环节：上一题回答 adequate，触发「上题反馈 + 出下一题」换题路径。"""
    bank = sample_question_bank()
    bank[1]["asked"] = True
    cur = bank[1]
    msgs = [
        AIMessage(content=cur["content"]),
        HumanMessage(content=ANSWER_SCRIPT[2]),
    ]
    return base_state(
        current_stage="tech_base",
        stage_turn_count=5,
        total_turn_count=5,
        current_question=cur,
        question_bank=bank,
        last_answer_quality="adequate",
        followup_count=0,
        messages=msgs,
    )


def make_state_project_new() -> dict:
    """刚进入 PROJECT 阶段第 1 轮，current_question 为 None，触发「新项目第1问」路径。"""
    msgs = [
        AIMessage(content="好的，基础技术问题我们就聊到这里。"),
        HumanMessage(content=ANSWER_SCRIPT[4]),
    ]
    return base_state(
        current_stage="project",
        stage_turn_count=1,
        total_turn_count=9,
        current_question=None,
        resume_projects=RESUME_PROJECTS,
        resume_skills=RESUME_SKILLS,
        projects_asked=[],
        last_answer_quality="adequate",
        messages=msgs,
    )


def make_state_project_followup() -> dict:
    """PROJECT 阶段：正在深挖第一个项目，学员回答后触发追问路径。"""
    proj = RESUME_PROJECTS[0]
    cur = {
        "content":      f"项目深挖：{proj['name']}",
        "project_name": proj["name"],
        "tech_stack":   proj["tech_stack"],
    }
    msgs = [
        AIMessage(content=f"我们聊聊你的「{proj['name']}」项目，背景和你的核心职责是什么？"),
        HumanMessage(content=ANSWER_SCRIPT[5]),
    ]
    return base_state(
        current_stage="project",
        stage_turn_count=2,
        total_turn_count=10,
        current_question=cur,
        resume_projects=RESUME_PROJECTS,
        resume_skills=RESUME_SKILLS,
        projects_asked=[proj["name"]],
        last_answer_quality="adequate",
        followup_count=0,
        messages=msgs,
    )


def make_state_closing_opening() -> dict:
    """刚进入 CLOSING 阶段第 1 轮：面试官发出反问邀请。"""
    msgs = [
        AIMessage(content="项目我们了解得差不多了。"),
        HumanMessage(content=ANSWER_SCRIPT[6]),
    ]
    return base_state(
        current_stage="closing",
        stage_turn_count=1,
        total_turn_count=12,
        resume_projects=RESUME_PROJECTS,
        messages=msgs,
    )


def make_state_closing_respond() -> dict:
    """CLOSING 阶段：学员提了问题，面试官回应。"""
    msgs = [
        AIMessage(content="最后这个环节，你有什么想了解的吗？"),
        HumanMessage(content=ANSWER_SCRIPT[7]),
    ]
    return base_state(
        current_stage="closing",
        stage_turn_count=2,
        total_turn_count=13,
        messages=msgs,
    )


def make_state_eval_excellent() -> dict:
    """evaluate_answer 测试：优秀回答（走 Think Tool + LLM 评估）。"""
    cur = sample_question_bank()[0]
    msgs = [
        AIMessage(content=cur["content"]),
        HumanMessage(content=ANSWER_SCRIPT[1]),
    ]
    return base_state(
        current_stage="tech_base",
        stage_turn_count=2,
        total_turn_count=2,
        current_question=cur,
        messages=msgs,
    )


def make_state_eval_no_answer() -> dict:
    """evaluate_answer 测试：明确不会（走快速路径，不调用 LLM）。"""
    cur = sample_question_bank()[2]
    msgs = [
        AIMessage(content=cur["content"]),
        HumanMessage(content="不知道"),
    ]
    return base_state(
        current_stage="tech_base",
        stage_turn_count=4,
        total_turn_count=4,
        current_question=cur,
        messages=msgs,
    )


def make_state_for_report() -> dict:
    """构造一份「已完成面试」的 State，messages 为完整对话，供 generate_report 测试。"""
    pairs = [
        (f"欢迎来到 {TARGET_POSITION} 岗位的模拟面试！请先自我介绍一下。", ANSWER_SCRIPT[0]),
        ("好的，你的介绍逻辑清晰。第一题：请解释 Transformer 自注意力机制的计算过程。", ANSWER_SCRIPT[1]),
        ("回答得很到位。下一题：过拟合和欠拟合分别是什么？如何缓解过拟合？", ANSWER_SCRIPT[2]),
        ("基本说清了。再来一题：请描述自回归语言模型的生成原理。", ANSWER_SCRIPT[3]),
        ("没关系，我们换一个。RAG 的基本流程是怎样的？", ANSWER_SCRIPT[4]),
        ("很好。我们进入项目环节，聊聊你的 RAG 系统，背景和你的职责是什么？", ANSWER_SCRIPT[5]),
        ("不错。能再具体说说你的混合检索是怎么做的吗？", ANSWER_SCRIPT[6]),
        ("最后，你有什么想了解的吗？", ANSWER_SCRIPT[7]),
    ]
    msgs = [HumanMessage(content="[开始面试]")]
    for ai_text, human_text in pairs:
        msgs.append(AIMessage(content=ai_text))
        msgs.append(HumanMessage(content=human_text))
    return base_state(
        current_stage="finished",
        stage_turn_count=2,
        total_turn_count=len(pairs),
        resume_projects=RESUME_PROJECTS,
        resume_skills=RESUME_SKILLS,
        messages=msgs,
    )


# ──────────────────────────────────────────────────────────────
# 六、数据库 fixture（供 load_context / save_report / save_memory 测试）
# ──────────────────────────────────────────────────────────────
async def seed_resume(student_email: str = "student01@eduagent.local") -> dict:
    """
    向 resume_reviews 写入一条测试简历（structured_data = RESUME_STRUCTURED），
    返回 {"student_id": <真实UUID>, "resume_review_id": <UUID>}。
    需 PostgreSQL 已启动且已 seed（存在 student01 账号）。
    """
    import json
    from sqlalchemy import text
    from backend.dependencies import AsyncSessionLocal

    resume_review_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("SELECT id FROM users WHERE email = :e"),
            {"e": student_email},
        )).mappings().fetchone()
        if not row:
            raise RuntimeError(
                f"未找到用户 {student_email}，请先运行：python scripts/seed_data.py"
            )
        student_id = str(row["id"])

        await s.execute(
            text("""
                INSERT INTO resume_reviews
                    (id, tenant_id, student_id, pdf_minio_path, structured_data, status)
                VALUES
                    (:id, :tenant_id, :student_id, :path, CAST(:structured AS JSONB), 'done')
                ON CONFLICT (id) DO UPDATE
                    SET structured_data = EXCLUDED.structured_data
            """),
            {
                "id":         resume_review_id,
                "tenant_id":  TENANT_ID,
                "student_id": student_id,
                "path":       "test/fixtures/resume_ai_dev.pdf",
                "structured": json.dumps(RESUME_STRUCTURED, ensure_ascii=False),
            },
        )
        await s.commit()

    return {"student_id": student_id, "resume_review_id": resume_review_id}


async def setup_db_fixture(student_email: str = "student01@eduagent.local") -> dict:
    """
    完整 DB fixture：写入简历 + 新建一个 in_progress 面试会话。
    返回 {"student_id", "session_id", "resume_review_id", "thread_id"}。

    用于 load_context（读简历/会话摘要）、save_report（UPDATE 已有会话）、
    save_memory（UPSERT 会话）等需要数据库的节点测试。
    """
    from sqlalchemy import text
    from backend.dependencies import AsyncSessionLocal
    from backend.core.memory import build_thread_id

    seeded     = await seed_resume(student_email)
    student_id = seeded["student_id"]
    session_id = new_session_id()
    thread_id  = build_thread_id(student_id, session_id)

    async with AsyncSessionLocal() as s:
        await s.execute(
            text("""
                INSERT INTO interview_sessions
                    (id, tenant_id, student_id, session_id, thread_id,
                     target_position, resume_review_id, status)
                VALUES
                    (:id, :tenant_id, :student_id, :session_id, :thread_id,
                     :target, :rrid, 'in_progress')
                ON CONFLICT (thread_id) DO NOTHING
            """),
            {
                "id":         str(uuid.uuid4()),
                "tenant_id":  TENANT_ID,
                "student_id": student_id,
                "session_id": session_id,
                "thread_id":  thread_id,
                "target":     TARGET_POSITION,
                "rrid":       seeded["resume_review_id"],
            },
        )
        await s.commit()

    return {
        "student_id":       student_id,
        "session_id":       session_id,
        "resume_review_id": seeded["resume_review_id"],
        "thread_id":        thread_id,
    }

