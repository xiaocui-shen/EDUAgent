# backend/agents/interview/graph.py

from langgraph.graph import StateGraph, START, END

from backend.agents.interview.state import (
    InterviewState, InterviewStage)
from backend.agents.interview.nodes import (
    load_context_node, # 加载简历、岗位、历史面试记录等上下文信息
    check_stage_node, # 判断当前面试阶段
    evaluate_answer_node, # 评估回答给评分、评价
    generate_response_node, # 生成下一轮面试题
    generate_report_node, # 生成面试报告
    save_report_node, # 保存面试报告
    save_memory_node, # 保存面试过程记忆
)
from backend.core.memory import get_memory_saver


def _route_after_check_stage(state: InterviewState) -> str:
    """
    check_stage 后的条件路由：
        - 已进入 FINISHED → 生成评估报告路径
        - 其他阶段 → 正常对话路径
    """
    if state.get("current_stage") == InterviewStage.FINISHED.value:
        return "generate_report" # 生成评估报告路径
    return "evaluate_answer" # 评估回答质量节点


def build_interview_graph():
    """构建并编译模拟面试 Agent 的 LangGraph 状态图。
        模拟面试过程中，一次对话就会执行一次graph
    图拓扑：
    START
    → load_context
    → check_stage
        ├─(finished)→generate_report→save_report→save_memory→END
        └─(else)→evaluate_answer→generate_response→save_memory→END
    """
    builder = StateGraph(InterviewState)

    # 注册节点
    builder.add_node("load_context", load_context_node)
    builder.add_node("check_stage", check_stage_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("save_report", save_report_node)
    builder.add_node("save_memory", save_memory_node)

    # 固定边
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "check_stage")

    # 条件分支
    builder.add_conditional_edges(
        "check_stage",
        _route_after_check_stage,
        {
            "generate_report": "generate_report",
            "evaluate_answer": "evaluate_answer",
        },
    )

    # 正常对话路径
    builder.add_edge(
        "evaluate_answer", "generate_response")
    builder.add_edge(
        "generate_response", "save_memory")
    builder.add_edge("save_memory", END)

    # 报告生成路径
    builder.add_edge(
        "generate_report", "save_report")
    builder.add_edge(
        "save_report", "save_memory")

    # 有 checkpointer：多轮对话需要状态跨请求持久化
    checkpointer = get_memory_saver("interview")
    return builder.compile(checkpointer=checkpointer)