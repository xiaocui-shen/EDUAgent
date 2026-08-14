# backend/core/memory.py

from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage)


from backend.core.logger import get_logger
from backend.langfuse_handler import langfuse_handler
logger = get_logger(__name__)

# 每个 Agent 独立的 MemorySaver 实例
_memory_savers: dict[str, MemorySaver] = {}
'''不同 Agent 的 State schema 不同
共用同一个 MemorySaver 会导致
    msgpack 序列化时 schema 字段冲突，必须隔离'''

def get_memory_saver(
        agent_type: str = "default") -> MemorySaver:
    """获取指定 Agent 类型的 MemorySaver 单例。
    本地阶段使用内存存储（进程重启后历史丢失）。
    生产阶段替换为 AsyncPostgresSaver 即可持久化，业务代码无需修改。
    Args:
        agent_type: Agent 标识符
        如 "qa" / "exam" / "resume" / "interview"
    Returns:
        MemorySaver 实例
        传给 StateGraph.compile(checkpointer=...)
    """
    if agent_type not in _memory_savers:
        _memory_savers[agent_type] = MemorySaver()
        logger.info("memory.saver_initialized",
                    agent=agent_type)
    return _memory_savers[agent_type]


def build_thread_id(
        student_id: str, session_id: str) -> str:
    """构建 LangGraph Checkpointer 使用的 thread_id。
    格式：student_{student_id}_session_{session_id}
    同一学员的不同会话有独立的历史，互不干扰。
    Example:
        build_thread_id("abc123", "xyz789")
        → "student_abc123_session_xyz789"
    """
    return f"student_{student_id}_session_{session_id}"


def build_config(student_id: str, session_id: str) -> dict:
    """构建 LangGraph 调用所需的 config 字典。
    用法：
        config = build_config(student_id, session_id)
        result = await graph.ainvoke(state, config=config)
    Returns:
        {"configurable": {"thread_id": "student_xxx_session_yyy"},
         "callbacks": [langfuse_handler],
         "metadata": {"langfuse_user_id": ..., "langfuse_session_id": ...}}
    """
    return {
        "configurable": {
            "thread_id":
                build_thread_id(student_id, session_id),
        },
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": student_id,
            "langfuse_session_id": session_id,
        },
    }


def trim_messages_to_window(
        messages: list[BaseMessage],
        window_size: int = 10,
    ) -> list[BaseMessage]:
    """滑动窗口裁剪：保留最近 window_size 轮对话。
    SystemMessage 始终保留在最前，不受 window_size 限制。
    Args:
        messages: 当前完整消息列表
        window_size:
            保留的对话轮数（1轮 = 1 Human + 1 AI），默认 10 轮
    Returns:
        裁剪后的消息列表
    """
    system_messages  = [
        m for m in messages
        if isinstance(m, SystemMessage)
    ] # SystemMessage 始终保留在最前，不参与裁剪
    dialogue_messages = [
        m for m in messages
        if not isinstance(m, SystemMessage)
    ] # 提取除了SystemMessage以外的其他消息
    # 1轮 = 2条消息
    max_dialogue_messages = window_size * 2

    if len(dialogue_messages) <= max_dialogue_messages:
        return messages   # 未超出窗口，不裁剪
    # 裁剪对话消息
    trimmed = dialogue_messages[-max_dialogue_messages:]

    logger.info(
        "memory.window_trimmed",
        original=len(dialogue_messages),
        kept=len(trimmed),
        window_size=window_size,
    )
    # 系统消息+裁剪后的对话消息 列表拼接后返回
    return system_messages + trimmed


def should_trigger_summary(
        messages: list[BaseMessage],
        threshold: int = 10,
    ) -> bool:
    """判断对话轮数是否超过阈值，决定是否触发摘要压缩。
    Args:
        messages:  当前消息列表
        threshold: 触发压缩的轮数阈值，默认 10 轮
    Returns:
        True → 需要压缩
    """
    dialogue_count = sum(
        1 for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    ) # SystemMessage不参与统计
    # 每轮 2 条，dialogue_count / 2 = 实际轮数
    return dialogue_count >= threshold * 2


async def compress_to_summary(
        messages: list[BaseMessage],
        existing_summary: Optional[str] = None,
    ) -> str:
    """将历史对话压缩为学员画像摘要。
    增量压缩：传入 existing_summary 防止已记录内容被重复写入。
    Args:
        messages:         待压缩的历史消息列表
        existing_summary: 上次的摘要（可选）
    Returns:
        压缩后的摘要文本
    """
    from backend.core.llm_factory import get_llm

    SUMMARY_PROMPT = """请将以下学员对话压缩为结构化学员画像摘要。

【压缩规则】
必须保留：学员明确不理解的知识点 / 反复出现的薄弱点 / 项目背景新增信息
选择性保留：已掌握知识点（简短标注）/ 学习进度信息
可以丢弃：已在上次摘要记录的内容 / 闲聊 / 已解决且理解的问题

【上一次摘要】
{previous_summary}

【本次新增对话】
{new_conversations}

请直接输出摘要文本，不要加任何前缀。"""

    '''把消息列表转成可读的对话文本，格式：学员：xxx\nAI：xxx\n...
     1. 判断消息类型：HumanMessage → "学员"，否则 → "AI"
     2. 取消息文本：优先 .text 属性（且不是方法），否则用 .content 兜底
     2.1 hasattr(m, 'text') 有text 属性 
     2.2 not callable(m.text) 且不是方法
     2.3 都不行就用 .content 转字符串
     3. 只保留人类和AI的消息，过滤掉 SystemMessage 等
    '''
    conversation_text = "\n".join([f"{'学员' if isinstance(m, HumanMessage) else 'AI'}："
        f"{m.text if hasattr(m, 'text') and not callable(m.text) else str(m.content)}"
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    ])
    # 构造完整提示词
    prompt_text = SUMMARY_PROMPT.format(
        previous_summary=existing_summary or "（无上次摘要）",
        new_conversations=conversation_text,
    )
    # 获取无结构化输出、开启thinking的deepseek LLM
    llm = get_llm("summarize")
    response = await llm.ainvoke([HumanMessage(content=prompt_text)])
    # 转为字符串并去除首尾空格
    summary_str = (
        response.content if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    logger.info(
        "memory.summary_generated",
        input_messages=len(messages),
        summary_length=len(summary_str),
    )
    return summary_str # 返回 压缩后的摘要文本字符串


if __name__ == '__main__':

    def check_memory():

        # ── 测试 1：thread_id 格式 ────────────────────────────────────
        tid = build_thread_id("student-uuid-001",
                              "session-uuid-abc")
        cfg = build_config("student-uuid-001",
                           "session-uuid-abc")
        # 真实结果
        print(f"thread_id : {tid}")
        print(f"config    : {cfg}")
        # 断言结果
        assert tid=="student_student-uuid-001_session_session-uuid-abc"
        assert cfg["configurable"]["thread_id"] == tid

        # ── 测试 2：滑动窗口裁剪 ─────────────────────────────────────
        messages = [SystemMessage(content="你是教学助手")]
        for i in range(1, 7):  # 模拟 6 轮对话
            messages.append(HumanMessage(content=f"问题{i}"))
            messages.append(AIMessage(content=f"回答{i}"))
        # 裁剪后的消息列表
        trimmed = trim_messages_to_window(messages, window_size=3)
        # 提取对话消息
        dialogue_only = [m
                         for m in trimmed
                         if not isinstance(m, SystemMessage)]
        # 真实结果
        print(f"\n原始消息数   : {len(messages)}（含1条System + 6轮对话）")
        print(f"裁剪后消息数 : {len(trimmed)}（System始终保留 + 最近3轮）")
        # 断言结果
        assert len(dialogue_only) == 6  # 3轮 × 2条
        assert isinstance(trimmed[0], SystemMessage)  # System 在最前
        assert trimmed[1].content == "问题4"  # 最近3轮从第4轮开始

        # ── 测试 3：摘要压缩触发判断 ─────────────────────────────────
        msgs_9_rounds = ([HumanMessage(content="q")] * 9
                         + [AIMessage(content="a")] * 9)
        msgs_10_rounds = ([HumanMessage(content="q")] * 10
                          + [AIMessage(content="a")] * 10)
        # 真实结果
        print(f"\n9 轮是否触发摘要  : "
              f"{should_trigger_summary(msgs_9_rounds, threshold=10)}")
        print(f"10 轮是否触发摘要 : "
              f"{should_trigger_summary(msgs_10_rounds, threshold=10)}")
        # 断言结果
        assert not should_trigger_summary(msgs_9_rounds, threshold=10)
        assert should_trigger_summary(msgs_10_rounds, threshold=10)

        # ── 测试 4：MemorySaver 隔离 ──────────────────────────────────
        saver_qa = get_memory_saver("qa")
        saver_exam = get_memory_saver("exam")
        saver_qa_dup = get_memory_saver("qa")
        # 断言结果
        assert saver_qa is not saver_exam  # 不同 Agent 互相独立
        assert saver_qa is saver_qa_dup  # 同一 Agent 返回同一实例（单例）

        print("\n✅ 没有出现 AssertionError 所有assert断言都通过！")

    check_memory()
