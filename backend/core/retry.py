# backend/core/retry.py
# 三层兜底机制：自动重试 → Agent 级降级 → 系统级兜底

import asyncio # 异步：用于超时控制和等待
from functools import wraps # @wraps：装饰器里保留原函数的名字/文档
# 类型注解：可调用对象 / 任意 / 可选
from typing import Callable, Any, Optional

# 自定义的异常
from backend.core.exceptions import (
    LLMAPIError,
    MilvusConnectionError,
    InvalidInputError,
    AuthenticationError,
)
from backend.core.logger import get_logger
logger = get_logger(__name__) # 日志对象


# ── 异常分类 ───────────────────────────────────────────────
# 可重试：多半是短暂故障（网络抖动、超时），重试一下可能就好
RETRYABLE_ERRORS = (
    LLMAPIError, # LLM API 错误
    MilvusConnectionError, # Milvus 连接错误
    TimeoutError, # 超时
    ConnectionError, # 连接错误
)
# 不可重试：重试也没用（输入非法、认证失败），应立即抛出
NON_RETRYABLE_ERRORS = (
    InvalidInputError, # 用户输入非法
    AuthenticationError, # 认证失败
)

MAX_RETRIES = 2 # 最多重试 2 次（加上首次 = 共 3 次尝试）
RETRY_DELAYS = [1.0, 3.0] # 第1次重试前等1秒，第2次前等3秒
TIMEOUT_PER_ATTEMPT = 30.0 # 单次调用最多等 30 秒，超时算失败


def with_retry(agent_type: str = ""):
    """三层兜底装饰器函数。
    给异步函数套上「重试 → 降级 → 系统兜底」三层保护。
    用法：
        @with_retry(agent_type="qa")
        async def _invoke():
            return await graph.ainvoke(state, config=config)
    """

    # 中间层：接收被装饰函数
    def decorator(func: Callable) -> Callable:

        # 最内层：真正的执行逻辑
        @wraps(func)  # 保留原函数的元信息（名字、docstring）
        async def wrapper(*args, **kwargs) -> Any:

            # ── 第一层：自动重试 ──────────────────────────
            # 记录最后一次的错误，留给后面降级用
            last_error: Optional[Exception] = None
            # 循环 3 次：attempt = 0, 1, 2
            for attempt in range(MAX_RETRIES + 1):
                try:
                    # 给单次调用套一个超时；超过 30 秒就抛TimeoutError
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=TIMEOUT_PER_ATTEMPT,
                    )
                    if attempt > 0: # 如果是重试成功的，记一条日志
                        logger.info(
                            "retry.succeeded",
                            agent_type=agent_type,
                            attempt=attempt + 1
                        )
                    return result # 成功，直接返回，结束
                # 不可重试异常：立即抛出，不再重试
                except NON_RETRYABLE_ERRORS as e:
                    logger.warning(
                        "retry.non_retryable_error",
                        agent_type=agent_type,
                        error=str(e)
                    )
                    raise # 原样抛出，交给上层处理
                # 其它（可重试）异常
                except Exception as e:
                    last_error = e # 记下异常信息
                    if attempt < MAX_RETRIES: # 还没到上限：等待后重试
                        # RETRY_DELAYS = [1.0, 3.0] 等待秒数
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "retry.attempt_failed",
                            agent_type=agent_type,
                            attempt=attempt + 1,
                            max_retries=MAX_RETRIES,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)  # 等1s或3s再重试
                    else:  # 到上限了：记录失败，跳出循环去降级
                        logger.error(
                            "retry.all_attempts_failed",
                            agent_type=agent_type,
                            error=str(e)
                        )

            # ── 第二层：Agent 级降级 ──────────────────────
            try:
                fallback_result = await AgentFallbackHandler.handle(
                    agent_type=agent_type, # 按 agent_type 找降级策略
                    original_error=last_error,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                )
                logger.info(
                    "retry.fallback_succeeded",
                    agent_type=agent_type
                )
                return fallback_result # 降级成功，返回降级结果
            except Exception as fallback_error: # 连降级都失败
                logger.error("retry.fallback_failed",
                             agent_type=agent_type,
                             error=str(fallback_error)
                             )

            # ── 第三层：系统级兜底 ────────────────────────
            logger.error(
                "retry.system_fallback",
                agent_type=agent_type,
                original_error=str(last_error)
            )
            # 最后的保底，永远不会再失败
            return _system_fallback_response(agent_type)

        return wrapper
    return decorator


class AgentFallbackHandler:
    """第二层降级：各 Agent 的专项降级策略"""

    @classmethod
    async def handle(
        cls,
        agent_type, # agent 名字
        original_error, # 系统抛出的异常
        func, # 被装饰的函数
        args, # 被装饰的函数的参数
        kwargs # 被装饰的函数的参数
    ) -> Any:
        """根据 agent_type 选择对应的降级策略。"""
        # 类型 → 降级方法 的映射表
        fallback_map = {
            "qa":               cls._qa_fallback,
            "exam_code":        cls._exam_code_fallback,
            "exam_subjective":  cls._exam_subjective_fallback,
            "resume":           cls._resume_fallback,
            "interview":        cls._interview_fallback,
        }
        # 根据agent名字获取函数对象
        handler = fallback_map.get(agent_type)
        if handler:
            # 执行降级函数
            return await handler(original_error, func, args, kwargs)
        # 没有对应降级策略，原样抛出（交给第三层系统兜底）
        raise original_error

    @classmethod
    async def _qa_fallback(
            cls, error, func, args, kwargs) -> dict:
        """问答降级：
        跳过 RAG 检索，直接用大模型的参数知识回答，并加 ⚠️ 提示。"""
        # 局部 import，避免循环依赖
        from backend.core.llm_factory import get_llm
        from langchain_core.messages import AIMessage
        logger.info("fallback.qa_direct_llm")

        # 尝试从调用参数里取出 state
        state = args[0] if args else kwargs.get("state", {})
        messages = state.get("messages", [])
        if not messages: # 取不到对话消息，无法有效降级
            raise ValueError("qa_fallback: messages unavailable, "
                             "escalating to system fallback")

        # 直接问大模型（不查知识库）
        llm = get_llm("qa")
        response = await llm.ainvoke(messages)
        fallback_content = (
            "⚠️ 知识库检索暂时不可用，以下为 AI 直接生成的回答，"
            "仅供参考，建议与教师确认：\n\n"
            + (response.content
               if hasattr(response, "content")
               else str(response))
        )
        return {
            "messages":
                messages + [AIMessage(content=fallback_content)],
            "fallback_used": True, # 标记：本次走了降级
            "structured_output": None,
        }

    @classmethod
    async def _exam_code_fallback(
            cls, error, func, args, kwargs) -> dict:
        """代码批改降级：评分服务暂时不可用时，标记需教师人工复核。"""
        logger.info("fallback.exam_code_basic")
        state = args[0] if args else kwargs.get("state", {})
        return {
            **state, # ** 把原 state 展开合并进来
            "fallback_used": True,
            "needs_teacher_review": True,
            "fallback_note": "代码评分服务暂时不可用，"
                             "已标记为需教师人工复核。",
        }

    @classmethod
    async def _exam_subjective_fallback(
            cls, error, func, args, kwargs) -> dict:
        """简答题批改降级：标记需教师复核。"""
        logger.info("fallback.exam_subjective_basic")
        state = args[0] if args else kwargs.get("state", {})
        return {
            **state,
            "fallback_used": True,
            "needs_teacher_review": True,
            "fallback_note": "AI 评分服务暂时不可用，"
                             "已标记为需教师人工批改。",
        }

    @classmethod
    async def _resume_fallback(
            cls, error, func, args, kwargs) -> dict:
        """简历审查降级：提示服务不可用 / 检查文件。"""
        logger.info("fallback.resume_service_unavailable")
        return {
            "fallback_used": True,
            "content": "简历审查服务暂时不可用，请稍后重试。"
                       "如持续失败，请检查上传的 PDF 文件是否完整。",
            "structured_output": None,
        }

    @classmethod
    async def _interview_fallback(cls, error, func, args, kwargs) -> dict:
        """面试降级：跳过深度分析，返回基础反馈。"""
        logger.info("fallback.interview_basic_feedback")
        return {
            "fallback_used": True,
            "content": "面试评估服务暂时不可用，"
                       "已记录本次面试对话，请稍后查看报告。",
            "structured_output": None,
        }


def _system_fallback_response(agent_type: str) -> dict:
    """第三层：系统级兜底。所有降级都失败后返回它，保证用户始终能收到响应。"""
    # 按 agent_type 给不同的友好提示
    messages = {
        "qa": "非常抱歉，智能问答服务暂时不可用，"
              "请稍后再试，或直接联系教师提问。",
        "exam": "非常抱歉，试卷批改服务暂时不可用，"
                "您的提交已保存，待服务恢复后将自动处理。",
        "resume": "非常抱歉，简历审查服务暂时不可用，请稍后重新上传。",
        "interview": "非常抱歉，模拟面试服务暂时不可用，请稍后重新开始。",
    }
    content = messages.get(agent_type, "服务暂时不可用，请稍后再试。")  # 找不到就用通用提示
    return {
        "messages": [],
        "content": content,
        "fallback_used": True,
        "system_fallback": True, # 标记：走到了最后一层系统兜底
        "structured_output": None,
    }

if __name__ == '__main__':
    # test_retry.py（项目根目录）
    import asyncio

    RETRY_DELAYS = [0.0, 0.0] # 测试时把重试等待调成 0，免得真等 1s/3s

    @with_retry(agent_type="qa")
    async def ok():
        return "success"

    calls = {"n": 0} # 总调用次数
    @with_retry(agent_type="qa")
    async def fail_twice_then_ok():
        calls["n"] += 1
        if calls["n"] < 3:  # 前两次抛可重试异常
            raise LLMAPIError("网络抖动")
        return "recovered"  # 第三次成功

    @with_retry(agent_type="qa")
    async def non_retryable():
        raise InvalidInputError("输入非法")  # 不可重试

    @with_retry(agent_type="")  # 空类型→无降级策略→走系统兜底
    async def always_fail():
        raise LLMAPIError("一直失败")

    async def main():
        print("① 一次成功:", await ok())
        print("② 失败2次后第3次成功:",
              await fail_twice_then_ok(),
              "| 总调用次数:", calls["n"])
        try:
            await non_retryable()
        except InvalidInputError as e:
            print("③ 不可重试异常立即抛出:", e)

        r = await always_fail()
        print("④ 全失败→系统兜底: system_fallback =",
              r.get("system_fallback"),
              "| content =", r["content"][:18], "...")

    asyncio.run(main())
