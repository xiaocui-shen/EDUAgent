# backend/core/logger.py
# 全项目的日志工具：在标准库 logging 之上做一层包装，
# 支持「事件名 + 键值对」的结构化日志写法。

import logging # Python 标准库的日志模块（你用过的那个）
from logging.handlers import RotatingFileHandler # 按大小轮转的日志文件处理器
import os # 用于创建日志目录
import sys # 用于把日志输出到标准输出 stdout
# 读取配置（需要里面的 log_level 日志级别）
from backend.config import get_settings


class _Logger:
    """日志包装类
    作用：让我们能用 logger.info("事件名", key=value) 这种「结构化」写法，
    而不必每次手动拼字符串。内部仍委托给标准库 logging 真正输出。
    前缀下划线 _ 表示它是「内部类」
    外部通过下面的 get_logger() 来获取实例"""

    def __init__(self, name: str):
        # 构造方法：name 通常传 __name__（当前模块名）
        # 这样日志里能看出是哪个模块打的
        # self._log 标准库的 logger实例，后续都用它输出
        self._log = logging.getLogger(name)

    def _fmt(self, event: str, **kw) -> str:
        """内部方法：把「事件名+关键字参数」格式化成一行可读字符串。
        例：_fmt("user.login", user_id="u1") →
            "user.login | user_id='u1'"
        **kw 不定长字典参数"""
        if kw: # 如果传了关键字参数（即 key=value）
            '''
            1. 把每个键值对拼成 "key=value"
            2. {v!r} 如果v是字符串就加引号
            3. 再用空格连接，最前面加上事件名和分隔符 " | "
            4. 举例：event = "send_message"
                    kw = {"user_id": 1001, "content": "你好"}
                ➡️ send_message | user_id=1001 content='你好'
            '''
            return event+" | "+" ".join(
                f"{k}={v!r}" for k, v in kw.items()
            )
        return event # 没有关键字参数时，只返回事件名本身

    def debug(self, event: str, **kw):
        """DEBUG 级别（最详细，调试用）"""
        self._log.debug(self._fmt(event, **kw))     # 先格式化，再交给标准库以 debug 级别输出

    def info(self, event: str, *args, **kw):
        """INFO 级别（常规信息） 兼容两种写法：
        ① logger.info("事件", key=value)
        ② logger.info("模板 %s", 值)。
        *args元组不定长「位置参数」，用于支持第二种写法"""
        if args: # 如果传了位置参数，说明是「模板 + 值」的老写法
            # 直接交给标准库的log做 % 占位符格式化
            self._log.info(event, *args)
        else: # 否则按结构化写法处理
            self._log.info(self._fmt(event, **kw))

    def warning(self, event: str, *args, **kw):
        """WARNING 级别（警告：不影响运行但需注意）"""
        if args:
            self._log.warning(event, *args)
        else:
            self._log.warning(self._fmt(event, **kw))

    def error(self, event: str, **kw):
        """ERROR 级别（错误）
        支持传 exc_info=True: 把异常堆栈一并打印"""
        # 从kw中取出exc_info参数（没有则默认 False），并移除它
        exc_info = kw.pop("exc_info", False)
        self._log.error(self._fmt(event, **kw), exc_info=exc_info)

    def critical(self, event: str, **kw):
        """CRITICAL 级别（严重错误）"""
        self._log.critical(self._fmt(event, **kw))


def configure_logging() -> None:
    """全局日志配置函数：整个应用只在启动时（main.py）调用一次。
    负责设定日志格式、级别，并压低第三方库的噪音日志"""
    settings = get_settings() # 读配置，拿到log_level（如 "INFO"）
    # getattr(对象, 属性名字符串, 兜底默认值)
    # settings.log_level.upper() 配置中获取日志级别转大写
    # 根据配置的log_level字段，获取logging对象对应的日志界别
    # 获取不到时（或配置写错了），就默认INFO
    level = getattr(
        logging, settings.log_level.upper(), logging.INFO
    )
    # 日志对象基础配置
    logging.basicConfig(
        # 格式：时间 [级别] 模块名: 内容
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout, # 把日志打到标准输出（控制台）
        level=level, # 全局生效的日志级别
        force=True, # 覆盖已有的日志配置
    )
    # 下面把几个第三方库的日志级别单独调到WARNING
    # 屏蔽它们大量的INFO级别的日志噪音，避免刷屏
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.WARNING) # SQLAlchemy 的 SQL 执行日志
    logging.getLogger("sqlalchemy.pool").setLevel(
        logging.WARNING) # SQLAlchemy 连接池日志
    logging.getLogger("httpx").setLevel(
        logging.WARNING) # httpx HTTP 客户端日志
    logging.getLogger("httpcore").setLevel(
        logging.WARNING) # httpcore 底层日志

    # ── 文件日志：追加写，10MB 轮转，保留1个备份 ──
    log_dir = settings.log_path # 来自 config.py，已是项目根目录下的绝对路径
    os.makedirs(log_dir, exist_ok=True) # 确保目录存在
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 单文件最大 10MB
        backupCount=1,              # 保留 1 个备份 (app.log.1)
        encoding="utf-8",           # 中文不乱码
    )
    file_handler.setLevel(level) # 跟控制台共用 LOG_LEVEL
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(file_handler) # 挂到 root logger

def get_logger(name: str) -> _Logger:
    """对外的工厂函数：
    每个模块用 get_logger(__name__) 拿到自己的日志器。"""
    return _Logger(name) # 返回一个_Logger实例（包装了标准库 logger）

if __name__ == '__main__':
    # TODO 调用执行才会生效，比如让日志正常记录到文件中
    configure_logging()
    log = get_logger("demo")
    log.info("user.login", user_id="u1", role="student")
    log.warning("cache.miss", key="abc")
