# backend/config.py
# 全项目唯一的「配置中心」：从 .env.local 读取所有配置项，供任何模块取用。
import os
def get_project_root() -> str:
    """获取工程根目录：基于当前文件的绝对路径，向上推导工程根目录"""
    # 当前文件（path_utils.py）的绝对路径
    current_file = os.path.abspath(__file__)
    # 当前文件所在目录（base/）
    current_dir = os.path.dirname(current_file)
    # 工程根目录（base/ 的上一级）
    project_root = os.path.dirname(current_dir)
    return project_root  # 返回工程根目录
def get_abs_path(relative_path: str) -> str:
    """将工程内的相对路径转为绝对路径（统一路径基准）
    :param relative_path: 相对于工程根目录的路径，如 "base/config.py"
    :return: 绝对路径"""
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

# Pydantic 的「配置基类」，能自动从环境变量/.env 读取并做类型校验
from pydantic_settings import BaseSettings
# 标准库装饰器：缓存函数结果，让函数实际只执行一次
from functools import lru_cache

class Settings(BaseSettings):
    """配置模型：每个类属性对应 .env.local 里的一项配置。
    继承 BaseSettings 自动把同名（大小写不敏感）的配置读进来并转成对应类型。
    先读配置文件，读不到就用默认值"""

    # ── 数据库（PostgreSQL）──
    db_host: str = "localhost"   # 主机；写了默认值 = 可选项
    db_port: int = 5432          # 端口；
    db_name: str = "eduagent"    # 库名
    db_user: str                 # 用户名；没有默认值 = 必填
    db_password: str             # 密码；同样必填

    @property
    def database_url(self) -> str:
        """把上面几个散件拼成 SQLAlchemy 需要的连接串。
        用 @property 装饰后，可像访问属性一样
        settings.database_url 取值，不用加括号调用。"""
        return (
            f"postgresql+asyncpg://"
            f"{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── Milvus 向量库 ──
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_db_name: str = "itcast"  # 数据库名
    milvus_collection_name: str = "knowledge_base"

    # ── 大模型（DeepSeek）──
    deepseek_api_key: str = os.getenv('deepseek')
    # deepseek_api_key：如果配置文件没有，就从环境变量获取
    deepseek_base_url: str = "https://api.deepseek.com/v1" # url
    deepseek_model_chat: str = "deepseek-v4-flash" # 对话模型名
    deepseek_model_coder: str = "deepseek-v4-flash" # 代码模型名

    # ── 本地模型权重路径 ──
    bge_m3_model_path: str = "../models/embedding/bge-m3" # 嵌入模型
    reranker_model_path: str = \
        "../models/reranker/bge-reranker-v2-m3" # 精排模型
    classifier_model_path: str = \
        "../models/classifier/all-MiniLM-L6-v2" # 意图分类模型基座
    # 微调训练后的意图分类模型
    classifier_finetuned_model_path: str = \
        "../models/classifier/query-classifier-finetuned"

    # ── JWT 认证 ──
    jwt_secret_key: str # 必填：签发登录令牌用的密钥
    jwt_algorithm: str = "HS256" # 签名算法
    jwt_access_token_expire_minutes: int = 10080 # 令牌有效期（分钟）

    # ── MCP Server 地址（第五章用）──
    kb_mcp_server_url: str = \
        "http://localhost:8000/mcp/kb"
    web_search_mcp_url: str = \
        "http://localhost:8000/mcp/search"

    # ── Web 搜索（Tavily 可选；留空则自动用免费的 DuckDuckGo）──
    tavily_api_key: str = ""

    # ── 应用基础配置 ──
    app_env: str = "local"                     # 运行环境标识
    app_debug: bool = False                    # 是否调试模式
    app_host: str = "0.0.0.0"                  # 监听地址
    app_port: int = 8000                       # 监听端口
    log_level: str = "INFO"                    # 日志级别
    '''.env.local文件中如不配置默认日志文件路径的话
    这里直接强制指定绝对路径'''
    log_path: str = get_abs_path("logs")       # 日志文件路径
    default_tenant_id: str = "tenant_default"  # 多租户默认值

    class Config:
        """Pydantic 的元配置：告诉 BaseSettings 该怎么读取配置。"""
        env_file = get_abs_path(".env.local")  # 从这个文件读取配置
        env_file_encoding = "utf-8"      # 文件编码
        case_sensitive = False           # 大小写不敏感
        extra = "ignore" # .env.local多的、没定义的字段忽略不报错

@lru_cache() # 缓存装饰器：让函数只执行一次，返回结果成为全局单例
def get_settings() -> Settings:
    """获取全局唯一配置对象。任何模块要用配置，都调用这个函数。"""
    # 首次调用时创建实例；之后每次都返回同一个缓存对象
    return Settings()

if __name__ == '__main__':
    # 直接运行 进行测试
    s = get_settings()
    print("db_host:", s.db_host)
    print("db_port:", s.db_port)
    print("database_url:", s.database_url)
    print("default_tenant_id:", s.default_tenant_id)
    print("两次 get_settings 是同一对象(lru_cache):",
          get_settings() is get_settings())
    print("环境变量获取deepseek api key:", s.deepseek_api_key)