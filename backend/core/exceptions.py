# backend/core/exceptions.py
# EduAgent 统一异常体系：所有自定义异常都继承同一个基类，
# 便于「一次捕获全部」以及按「可重试 / 不可重试」分类处理。


class EduAgentBaseError(Exception):
    """所有EduAgent自定义异常的基类,继承Python内置的Exception
    比普通异常多带两样上下文：是哪个Agent出的错、以及细节字典"""
    def __init__(self,
                 message: str,
                 agent_type: str = "",
                 details: dict = None):
        '''
        :param message: 错误消息 描述文字
        :param agent_type: 哪个 Agent（qa/exam/...）
        :param details: 额外细节
        '''
        # 调用父类Exception的初始化，把错误消息存好
        super().__init__(message)
        # 记录出错的 Agent 类型，方便日志与排查
        self.agent_type = agent_type
        # 记录细节字典；传 None 时用空字典 {} 兜底，避免后续访问报错
        self.details = details or {}


class LLMAPIError(EduAgentBaseError):
    """大模型 API 调用失败（超时 / 限流 / 网络错误）。
    属于【可重试】异常。"""
    pass # 无需额外逻辑，直接继承自定义基类的能力即可

class AgentExecutionError(EduAgentBaseError):
    """Agent 业务逻辑执行失败。"""
    pass

class PipelineError(EduAgentBaseError):
    """多 Agent Pipeline（流水线编排）失败。"""
    pass

class IntentRouteError(EduAgentBaseError):
    """意图识别路由失败（没判断出该交给哪个 Agent）。"""
    pass

class MilvusConnectionError(EduAgentBaseError):
    """Milvus 向量库连接失败。属于【可重试】异常。"""
    pass

class FileParseError(EduAgentBaseError):
    """文件解析失败（Word / PDF）。"""
    pass

class InvalidInputError(EduAgentBaseError):
    """用户输入不合法。属于【不可重试】异常（重试也不会变合法）。"""
    pass

class AuthenticationError(EduAgentBaseError):
    """认证失败。属于【不可重试】异常。"""
    pass

if __name__ == '__main__':
    # 测试运行
    # 抛出一个带上下文的异常，并用基类捕获
    try:
        raise LLMAPIError("DeepSeek 超时",
                          agent_type="qa",
                          details={"timeout": 30})
    except EduAgentBaseError as e:
        # 对自定义异常类的信息e进行使用
        print(
            "捕获:", type(e).__name__, # 异常名字
            "| msg:", e, # 异常信息
            "| agent:", e.agent_type, # 出错的Agent名字
            "| details:", e.details # 错误细节字典
        )

    print("LLMAPIError 是 EduAgentBaseError 子类:",
          issubclass(LLMAPIError, EduAgentBaseError))
