# backend/agents/interview/state.py

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from enum import Enum


class InterviewStage(str, Enum):
    """面试的五个阶段（状态机的状态取值）。"""
    WARMUP    = "warmup"      # 热身：邀请自我介绍
    TECH_BASE = "tech_base"   # 技术基础：题库问答
    PROJECT   = "project"     # 项目深挖：针对简历项目追问
    CLOSING   = "closing"     # 反问收尾：让学员提问
    FINISHED  = "finished"    # 终态：触发报告生成


class AnswerQuality(str, Enum):
    """学员单次回答的质量标签（驱动追问/换题决策）。"""
    EXCELLENT = "excellent"   # 优秀：有技术细节/量化/原理
    ADEQUATE  = "adequate"    # 基本及格：方向对但缺深度
    WEAK      = "weak"        # 较弱：方向偏或太表面
    NO_ANSWER = "no_answer"   # 未作答：明说不知道或为空


class DimensionEval(BaseModel):
    """单个维度的评估结果"""
    dimension: str = Field(description="评估维度名称")
    score: int = Field(description="得分 0-100")
    comment: str = Field(description="1-2句评语")
    highlights: list[str] = Field(
        default_factory=list, description="亮点列举")
    weaknesses: list[str] = Field(
        default_factory=list, description="薄弱点列举")


class InterviewReport(BaseModel):
    """面试五维度评估报告"""
    dimensions: list[DimensionEval] # 五个维度的评估结果
    overall_score: int = Field(
        description="综合得分 0-100（五维度加权平均）")
    strengths: list[str] = Field(
        description="2-3条核心优势（真实面试表现）")
    improvements: list[str] = Field(
        description="2-3条重点提升方向")
    overall_comment: str = Field(
        description="2-3句综合评语")
    recommended_topics: list[str] = Field(
        description="建议重点复习的3-5个知识点")
    next_step_advice: str = Field(
        description="下一步备考建议（1-2句）")


class ReportWrapper(BaseModel):
    """InterviewReport顶层包装（with_structured_output需要顶层对象）"""
    report: InterviewReport


class InterviewState(TypedDict):
    """模拟面试 Agent 完整 State（22 个字段，跨轮由 MemorySaver 持久化）"""
    # ── 请求上下文 ─────────────────────────────────────────────
    messages:         Annotated[list[BaseMessage], add_messages]  # 对话消息；add_messages 让新消息自动追加而非覆盖
    student_id:       str           # 学员 ID
    tenant_id:        str           # 租户 ID（多租户隔离）
    session_id:       str           # 会话 ID（API 层生成）
    target_position:  str           # 目标岗位，决定出题方向

    # ── 简历联动数据（从简历审查结果读取，可为空）──────────────
    resume_review_id: Optional[str]  # 关联的简历审查记录 ID；非空才做项目深挖
    resume_projects:  list[dict]    # 简历项目列表（每项 = 名称/角色/技术栈/亮点）
    resume_skills:    list[str]     # 简历技能列表

    # ── 面试阶段控制 ───────────────────────────────────────────
    current_stage:    str           # 当前阶段 InterviewStage
    stage_turn_count: int           # 当前阶段已进行的轮数
    total_turn_count: int           # 总轮数（跨阶段累计）
    max_turns:        int           # 轮数上限，默认 40

    # ── 题目管理 ───────────────────────────────────────────────
    question_bank:    list[dict]    # 题库；每项 {id, content, difficulty, tags, asked}
    current_question: Optional[dict] # 当前正在问/追问的题
    projects_asked:   list[str]     # 已深挖过的项目名称列表（防重复深挖）

    # ── 回答质量追踪 ───────────────────────────────────────────
    last_answer_quality: str # 上一条回答的质量标签，AnswerQuality枚举值
    followup_count:      int        # 当前问题已追问次数（上限 2）

    # ── 记忆管理 ───────────────────────────────────────────────
    existing_summary:    Optional[str]   # 历史对话摘要（DB 持久化，长对话压缩用）
    should_summarize:    bool            # 本轮是否需要触发摘要压缩

    # ── 评估结果（面试结束后填充）────────────────────────────
    report:              Optional[dict]  # 五维度报告，InterviewReport.model_dump() 的结果

    # ── 降级标记 ────────────────────────────────────────────────
    fallback_used:       bool            # 是否用过兜底逻辑（如报告生成失败走兜底）
    structured_output:   Optional[dict]  # 供 API 层直接读取的结构化结果（如报告摘要）
