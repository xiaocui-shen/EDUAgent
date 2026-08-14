# backend/agents/exam/state.py
# ──────────────────────────────────────
# 试卷批改 Agent 的 State 定义
# + LLM 结构化输出的 Pydantic 模型。
# State 是 LangGraph 图的「数据总线」，
# 所有节点都从它读数据、往它写数据。
# ──────────────────────────────────────

# ── 标准库类型注解 ─
from typing import Annotated, Optional
# Annotated：给字段加「元数据」，
# 这里用来告诉 LangGraph 如何合并多轮消息
# Optional：表示字段可以为 None（可选字段）

# ── TypedDict ──
from typing_extensions import TypedDict
# TypedDict：定义「字典的结构」，
# 让 IDE 和类型检查器知道 dict 里
# 有哪些键、什么类型。
# 普通 dict 没有结构约束，TypedDict 有。

# ── LangGraph 消息合并工具 ──
from langgraph.graph.message import add_messages
# add_messages：LangGraph 提供的「消息追加器」
# 当节点返回 {"messages": [新消息]} 时，
# add_messages 会把新消息追加到已有列表末尾，
# 而不是覆盖。这样多轮对话的历史消息不会丢。

# ── LangChain 消息基类 ──
from langchain_core.messages import BaseMessage
# BaseMessage：LangChain 消息对象的基类
# （HumanMessage、AIMessage 都继承它）
# 用 BaseMessage 作为类型注解，
# 表示这个列表里可以放任何类型的消息。

# ── Pydantic 数据模型 ──
from pydantic import BaseModel
# BaseModel：Pydantic 的基类，
# 用于定义「结构化数据模型」。
# LLM 的结构化输出（with_structured_output）
# 要求传入一个 BaseModel 子类，
# LLM 会按这个模型的字段生成 JSON，
# 再自动转成 Python 对象。


# ═══════════════════════════════════════
# 以下 4 个 Pydantic 模型，
# 都是给 LLM 结构化输出用的 Schema。
# LLM 批改简答题、分析薄弱点时，
# 会按这些模型的字段返回结构化 JSON。
# ══════════════════════════════════════

class ScoringPointResult(BaseModel):
    """单个得分点的评分结果

    简答题会被拆成多个得分点
    （scoring_points 表），
    每个得分点独立评分。
    例如一道 10 分的简答题拆成 3 个得分点
    （4分+3分+3分），
    LLM 对每个点判断「得了没得」。
    """
    point_id: str       # 得分点 ID
                          # （对应 scoring_points 表的 id）
    point_desc: str     # 得分点描述
                          # （如「答出 IOC 概念得 4 分」）
    point_score: int    # 该得分点的满分值（如 4）
    earned: bool        # 是否得分
                          # True = 得了，False = 没得
    evidence: str       # 得分证据
                          # earned=True 时，
                          # 填学员答案中对应的原文
    missing: str        # 失分原因
                          # earned=False 时，
                          # 填为什么没得分


class SubjectiveReviewResult(BaseModel):
    """简答题批改结果（LLM 结构化输出）

    这是 LLM 批改一道简答题后的
    完整返回结构。
    用 with_structured_output(...) 绑定后，
    LLM 调用完直接返回这个对象，
    不需要手动解析 JSON。
    """
    question_id: str        # 题目 ID
                              # （对应 questions 表的 id）
    student_answer: str     # 学员的答案原文
    total_score: int        # 学员实际得分
                              # （各得分点 earned 的分值之和）
    full_score: int         # 该题满分
    confidence: float       # LLM 置信度 [0, 1]
                              # 低于 0.7 时标记
                              # needs_review=True，
                              # 提醒教师复核
    point_results: list[ScoringPointResult]
                              # 每个得分点的评分明细
    overall_comment: str    # LLM 对这道题的整体评语
                              # （如「概念理解基本正确，
                              #  但缺少关键术语」）


class WeakPoint(BaseModel):
    """单个知识薄弱点

    批改完所有题目后，LLM 会分析学员
    在哪些知识点上错得多，生成薄弱点报告。
    例如：学员在「Spring IOC」相关的
    3 道题里错了 2 道，这就是一个薄弱点。
    """
    tag: str            # 知识点标签
                          # （如 "Spring IOC"、
                          #  "Redis缓存穿透"，
                          #  对应 questions 表的
                          #  knowledge_tag）
    wrong_count: int    # 该知识点下做错的题数
    total_count: int    # 该知识点下总共的题数
    question_nos: list  # 涉及的题目序号
                          # （如 [1, 3, 5]，
                          #  方便学员定位）
    suggestion: str     # 针对该知识点的复习建议
                          # （LLM 生成，
                          #  如「建议重新学习
                          #  第3章 IOC 相关内容」）


class WeakPointsReport(BaseModel):
    """知识薄弱点分析报告
    （LLM 结构化输出）

    这是 LLM 分析完所有薄弱点后的
    完整返回结构。
    包含多个 WeakPoint + 一段整体评价。
    """
    weak_points: list[WeakPoint]
                              # 薄弱点列表
                              # （可能 0 个或多个）
    overall_summary: str    # 整体评价，不超过 50 字
                              # （如「IOC 和 AOP 概念
                              #  掌握薄弱，建议重点
                              #  复习第3章」）


class TeacherDecision(BaseModel):
    """教师确认决策
    （interrupt 恢复时传入）

    这是 HitL（人在环中）的关键模型。
    流程：AI 批改完 → 图暂停（interrupt）
    → 教师在前端查看 → 教师提交决策
    → 图恢复执行。
    教师提交的数据就按这个模型结构化。
    """
    action: str         # 教师动作
                          # "approve"（同意 AI 结果）
                          # "modify"（要修改分数）
    modifications: list # 修改明细
                          # （仅 action="modify" 时有值）
                          # 格式：
                          # [{"question_id": "xxx",
                          #   "new_score": 8,
                          #   "comment": "给分偏低"}, ...]
    teacher_id: str     # 操作教师的 ID
                          # （对应 users 表的 id）


# ═══════════════════════════════════════
# ExamState：试卷批改 Agent 的完整 State
# （数据总线）
# ───────────────────────────────────────
# TypedDict 定义的字典，每个键代表
# State 中的一个字段。
# 图的每个节点都从 state 里读自己需要
# 的字段，执行完把结果写回 state。
# 最终 state 会保存到 MemorySaver
# （checkpoint），也会写入数据库。
# ═══════════════════════════════════════

class ExamState(TypedDict):
    """试卷批改 Agent 完整 State"""

    # ── 请求上下文 ────────────────────
    # 从 HTTP 接口传入，贯穿全流程

    messages: Annotated[list, add_messages]
    # 对话消息列表。
    # Annotated[..., add_messages]
    # 告诉 LangGraph：节点返回新消息时，
    # 追加到列表末尾，不要覆盖旧消息。
    # 这样多轮交互的历史不会丢。

    student_id: str     # 学生 ID
                          # （对应 users 表的 id，
                          #  查提交记录、写批改结果都用）
    tenant_id: str      # 租户 ID
                          # （多租户隔离，
                          #  所有数据库查询都要带
                          #  这个过滤条件）
    session_id: str     # 前端会话 ID
                          # （前端轮询状态时用）
    exam_id: str        # 试卷 ID
                          # （对应 exams 表的 id，
                          #  用来查这张试卷有哪些题目）
    submission_id: str  # 提交记录 ID
                          # （对应 exam_submissions 表
                          #  的 id，批改结果要写回
                          #  这条记录）
    word_file_path: str # 学员作答的 Word 文件
                          # 本地临时路径
                          # （上传后存到临时目录，
                          #  解析完可删除）

    # ── 解析结果 ──────────────────────
    # Word 解析 + DB 题目合并后的
    # 完整题目列表

    parsed_questions: list
    # 格式示例：
    # [{"question_no": 1,
    #   "question_type": "single_choice",
    #   "content": "...",
    #   "correct_answer": "A",
    #   "score": 10,
    #   "scoring_points": [...]}, ...]
    # 这个列表是后续三轨批改的
    # 「输入数据」，每个节点都从
    # 这里读题目。

    # ── 三轨批改结果 ────────────────
    # 客观题 / 简答题 / 代码题，并行批改

    objective_results: list
    # 客观题批改结果（单选/多选/判断）
    # 格式示例：
    # [{"question_id": "xxx",
    #   "question_no": 1,
    #   "student_answer": "A",
    #   "correct_answer": "A",
    #   "is_correct": True,
    #   "score": 10,
    #   "full_score": 10}, ...]
    # 客观题批改是确定性逻辑（对/错），
    # 不需要 LLM，直接字符串比对。

    subjective_results: list
    # 简答题批改结果
    # 格式示例：
    # [{"question_id": "xxx",
    #   "question_no": 3,
    #   "student_answer": "...",
    #   "total_score": 7,
    #   "full_score": 10,
    #   "confidence": 0.82,
    #   "point_results": [...],
    #   "overall_comment": "...",
    #   "needs_review": False}, ...]
    # 简答题需要 LLM 评分
    # （SubjectiveReviewResult 模型）。

    code_results: list
    # 代码题批改结果
    # 格式示例：
    # [{"question_id": "xxx",
    #   "question_no": 5,
    #   "student_code": "...",
    #   "test_results": [
    #     {"case": "test_1",
    #      "passed": True}, ...],
    #   "score": 8,
    #   "full_score": 10}, ...]
    # 代码题用沙箱执行学员代码
    # + 跑测试用例，根据通过率给分。

    # ── 汇总预批改结果 ────────────────
    # 三轨合并后的完整预批改

    pre_review_summary: dict
    # 三轨批改全部完成后，汇总成一份
    # 完整的预批改报告。
    # 格式示例：
    # {"total_score": 72,
    #  "full_score": 100,
    #  "question_count": 10,
    #  "objective_correct": 8,
    #  "subjective_avg_confidence": 0.85,
    #  ...}
    # 这份汇总会展示给教师，
    # 供 HitL 审核。

    # ── 知识薄弱点分析 ────────────────
    # 基于批改结果，LLM 分析学员薄弱环节

    weak_points: list
    # 薄弱点列表
    # （WeakPoint 模型结构，
    #  转成 dict 存入 State）
    # 格式示例：
    # [{"tag": "Spring IOC",
    #   "wrong_count": 2,
    #   "total_count": 3,
    #   "question_nos": [1, 3],
    #   "suggestion": "..."}, ...]

    weak_points_summary: str
    # 薄弱点整体评价
    # （如「IOC 和 AOP 概念掌握薄弱，
    #  建议重点复习第3章」）

    # ── Human-in-the-Loop ────────────
    # 教师审核环节

    teacher_notified: bool
    # 是否已通知教师审核。
    # True = 已发送通知（邮件/站内信），
    # False = 还没通知。
    # 用于防止重复通知。

    teacher_decision: Optional[dict]
    # 教师的决策结果
    # （TeacherDecision 模型结构，
    #  转成 dict）。
    # None = 教师还没审核；
    # 有值 = 教师已提交
    # （approve 或 modify）。
    # 这个字段在 interrupt 恢复时
    # 由前端传入。

    # ── 最终结果 ──────────────────────
    # 教师确认后生成，写入数据库

    final_results: list
    # 最终批改结果
    # （教师确认或修改后的版本）
    # 格式和 subjective_results 类似，
    # 但分数以教师确认为准。
    # 这份结果会写入 exam_reviews 表，
    # 并发布给学生。

    published: bool
    # 是否已发布给学生。
    # True = 学生能在前端看到成绩，
    # False = 还没发布。
    # 教师点「发布」按钮后，
    # 这个字段变为 True，
    # 同时更新 exam_submissions 表
    # 的 status 为 "published"。

    # ── 降级标记 ──────────────────────
    # 出错时的兜底信息

    fallback_used: bool
    # 是否触发了降级策略。
    # True = 某个环节出错后走了兜底方案
    # （如 LLM 超时改用规则评分）。
    # 前端可以据此显示提示
    # （如「本次批改部分题目
    #  使用了兜底策略」）。

    structured_output: Optional[dict]
    # LLM 结构化输出的原始结果
    # （完整 JSON，用于调试和审计）。
    # 正常流程不需要这个字段，
    # 但出问题时可以用来排查
    # LLM 返回了什么。
