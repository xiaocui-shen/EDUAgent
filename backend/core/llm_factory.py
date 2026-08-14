# backend/core/llm_factory.py
# LLM Factory：统一封装大模型调用，按 Agent 类型路由。
# 规矩：所有 Agent 必须通过此模块获取模型，禁止直接调用 init_chat_model。

# 类型注解用：Type 表示「某个类本身」，Any 表示任意类型
from typing import Type, Any
from pydantic import BaseModel # 结构化输出的 Schema 都是它的子类
import httpx  # HTTP 客户端库（用来自定义网络行为）
from langchain.chat_models import init_chat_model # 创建聊天模型
# 聊天模型的基类（类型注解用）
from langchain_core.language_models import BaseChatModel
# 「可运行对象」基类，结构化模型属于它
from langchain_core.runnables import Runnable

from backend.config import get_settings #配置
from backend.core.logger import get_logger # 结构化日志

logger = get_logger(__name__) # 本模块的日志器，name用当前模块名

# ── 自定义 httpx 客户端：绕过系统代理 ───────────────────────────
'''创建一对同步/异步 http 客户端:
1. 禁用系统代理避免SSL报错（init_chat_model请求云上模型时）
2. 设置长超时适配大模型慢响应
3. 全局复用这两个单例 向DeepSeek云上llmAPI发送请求

4. trust_env=False详细解读：无视所有http代理配置，直连目标接口
4.1. init_chat_model底层使用httpx发请求给DeepSeek API
4.2. httpx默认会自动读取系统/环境变量里的代理，比如：
    HTTP_PROXY/HTTPS_PROXY、Windows系统代理
4.3. 走代理可能会因为http代理的设置或网络导致SSL/TLS握手报错
4.4. 而且DeepSeek国内能直连，所以设置trust_env=False，强制关掉代理
'''
# 异步客户端（给 ainvoke/astream 用）
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    trust_env=False,
    # 总超时 120 秒，建立连接超时 15 秒
    timeout=httpx.Timeout(120.0, connect=15.0),
)
# 同步客户端（给 invoke 用）
_HTTP_SYNC_CLIENT = httpx.Client(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)

# ── Agent 类型 → 模型标识符 的路由表 ────────────────────────────
_AGENT_MODEL_ROUTING: dict[str, str] = {
    "qa":               "chat",   # 智能问答
    "exam_subjective":  "chat",   # 试卷-简答题批改
    "exam_code":        "code",   # 试卷-代码题批改
    "resume":           "chat",   # 简历审查
    "interview":        "chat",   # 模拟面试
    "intent":           "chat",   # 意图识别
    "summarize":        "chat",   # 对话摘要压缩
}

# 模型标识符 → DeepSeek API 实际接受的 model 名称
_MODEL_ID_MAP: dict[str, str] = {
    "chat": get_settings().deepseek_model_chat,
    "code": get_settings().deepseek_model_coder,
}

class LLMFactory:
    """大模型工厂（统一获取模型的唯一入口）。
    用 @classmethod 定义方法，不用创建对象
    直接用 LLMFactory.get_llm(...) 调用
    用法：
        llm = LLMFactory.get_llm("qa") # 普通模型
        structured = LLMFactory.get_structured_llm(
            "resume",
            某Schema
            )  # 结构化输出模型
        response = await llm.ainvoke(messages)
    """
    # 类变量：模型实例缓存（缓存键 → 模型），全类共享
    _instances: dict[str, BaseChatModel] = {}

    @classmethod
    def _get_settings(cls):
        """内部小工具：取配置对象。"""
        return get_settings()

    @classmethod
    def _build_model_kwargs(cls, model_key: str) -> dict[str, Any]:
        """内部方法：组装 init_chat_model 需要的所有参数
        DeepSeek 走 OpenAI 兼容接口"""
        settings = cls._get_settings() # 配置
        # 把模型标识符转成 API 实际的 model 名
        model_id = _MODEL_ID_MAP[model_key]

        return {
            "model": model_id, # 模型名，如 "deepseek-chat"
            # 强制走 langchain-openai（DeepSeek 兼容 OpenAI 接口）
            "model_provider": "openai",
            "temperature": 0, # 默认 0：评分/批改要稳定输出
            "api_key": settings.deepseek_api_key, # 来自 .env.local
            # DeepSeek 接口地址
            "base_url": settings.deepseek_base_url,
            "max_retries": 0, # 模型层不重试；重试统一由retry.py管
            # 用上面绕过代理的异步客户端
            "http_async_client": _HTTP_ASYNC_CLIENT,
            "http_client": _HTTP_SYNC_CLIENT, # 同步客户端
        }

    @classmethod
    def get_llm(
        cls,
        agent_type: str, # Agent 类型，必须在路由表里
        temperature: float = 0, # 温度：对话类可传 0.3~0.7，评分类保持0
        streaming: bool = False, # 是否流式输出（问答/面试对话用）
        thinking: str = "enabled", # 官方默认开启、这里也开启，可传"disabled"关闭
    ) -> BaseChatModel:
        """按Agent类型获取开启thinking模式的llm实例，会缓存到当前进程内存。
        相同 (模型, 温度, 是否流式) 的组合只会创建一次，之后复用。"""
        # 校验：不认识的类型直接报错（早暴露问题）
        if agent_type not in _AGENT_MODEL_ROUTING:
            raise ValueError(
                f"未知 agent_type: '{agent_type}'，"
                f"可用类型：{list(_AGENT_MODEL_ROUTING.keys())}"
            )

        # 查路由表，拿到模型标识符
        model_key = _AGENT_MODEL_ROUTING[agent_type]

        # 用「模型_温度_是否流式_是否开启thinking模式」拼一个缓存键：不同组合各缓存一份
        # 一定要加thinking！一个agent中可能会调用thinking和非thinking的，要区分开！
        cache_key = f"{model_key}_{temperature}_{streaming}_{thinking}"
        if cache_key not in cls._instances: # 缓存里没有才新建
            kwargs = cls._build_model_kwargs(model_key) # 组装基础参数
            kwargs["streaming"] = streaming # 设置是否流式
            # deepseek-V4 默认 thinking 模式开启、这里也开启
            # thinking模式不支持 function calling 结构化输出
            kwargs["extra_body"] = {"thinking": {"type": thinking}}
            # 开启thinking模式时 temperature温度参数无效（官方直接锁定）
            # 后边get_structured_llm（关闭thinking）调用本函数时会传入温度
            kwargs["temperature"] = temperature  # 覆盖温度

            # 真正创建模型（** 表示把字典展开成关键字参数）
            llm = init_chat_model(**kwargs)
            cls._instances[cache_key] = llm # 存进缓存

            logger.info( #记一条结构化日志，便于观察
                "llm_factory.model_initialized",
                agent_type=agent_type, model_key=model_key,
                temperature=temperature, streaming=streaming,
                thinking=thinking,
            )

        return cls._instances[cache_key] # 返回缓存中的实例

    @classmethod
    def get_structured_llm(
        cls,
        agent_type: str,
        # 期望的输出结构（一个 Pydantic 模型类）
        output_schema: Type[BaseModel],
        temperature: float = 0,
    ) -> Runnable:
        """获取「绑定了结构化输出 Schema」的llm实例，关闭了thinking模式。
        调用它的ainvoke后，直接返回一个output_schema类型的对象（不是文本"""
        # 先拿普通模型
        llm = cls.get_llm(
            agent_type,
            temperature=temperature,
            thinking="disabled", # deepseekV4结构化输出必须关闭thinking模式
        )
        # 绑定Pydantic结构；method="function_calling"是DeepSeek必须的
        return llm.with_structured_output(
            output_schema,
            method="function_calling"
        )

    @classmethod
    def get_configurable_llm(cls, temperature: float = 0) -> BaseChatModel:
        """获取「运行时可动态切换模型」的实例
        （少数需要在运行时决定用哪个模型的场景用）。"""
        settings = cls._get_settings()
        return init_chat_model(
            model=settings.deepseek_model_chat,
            model_provider="openai",
            temperature=temperature,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            http_async_client=_HTTP_ASYNC_CLIENT,
            http_client=_HTTP_SYNC_CLIENT,
            # extra_body={"thinking": {"type": "disabled"}} # 关闭thinking模式
        )

    @classmethod
    def clear_cache(cls) -> None:
        """清空模型实例缓存（测试时用）。"""
        cls._instances.clear()
        logger.info("llm_factory.cache_cleared")


# ── 模块级便捷函数────────────────────────
# 比写 LLMFactory.get_llm(...) 更简洁
# 直接 from llm_factory import get_llm 即可调用。

def get_llm(agent_type: str,
            temperature: float = 0,
            streaming: bool = False) -> BaseChatModel:
    """LLMFactory.get_llm 的便捷入口。"""
    return LLMFactory.get_llm(agent_type,
                              temperature=temperature,
                              streaming=streaming)

def get_structured_llm(
        agent_type: str,
        output_schema: Type[BaseModel]
    ) -> Runnable:
    """LLMFactory.get_structured_llm 的便捷入口。"""
    return LLMFactory.get_structured_llm(
        agent_type, output_schema)


if __name__ == '__main__':
    # 测试运行
    from pydantic import BaseModel, Field
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

    # ① 按类型拿到模型实例
    llm = LLMFactory.get_llm("qa")
    print("① 是 BaseChatModel:", isinstance(llm, BaseChatModel),
          "|", type(llm).__name__)

    # ② 相同参数有缓存，返回同一实例
    print("② 缓存命中(同一实例):", LLMFactory.get_llm("qa") is llm)

    # ③ 未知 agent_type 会报错
    try:
        LLMFactory.get_llm("not_exist")
    except ValueError as e:
        print("③ 未知类型报错:", str(e)[:38], "...")

    # ④ 结构化输出：绑定 Pydantic，返回 Runnable
    class Demo(BaseModel):
        name: str = Field(description="姓名")
    print(
        "④ 结构化模型是 Runnable:",
        isinstance(
            LLMFactory.get_structured_llm(
                "resume",
                Demo
            ),
            Runnable
        )
    )

    # ⑤ 清空缓存
    LLMFactory.clear_cache()
    print("⑤ clear_cache 后缓存为空:",
          len(LLMFactory._instances) == 0)

    # ⑥ 联网测试，异步调用
    import asyncio
    from langchain_core.messages import HumanMessage
    async def main():
        llm = get_llm("qa")  # 通过工厂拿模型
        resp = await llm.ainvoke([
            HumanMessage(content="用一句话介绍 Python")
        ])
        print("⑥ DeepSeek 回复:\n\t", resp.text)  # .text取文本

    asyncio.run(main())