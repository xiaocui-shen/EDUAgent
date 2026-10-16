# EduAgent — 教育场景的 AI 多智能体辅助平台

> 面向 IT 教育培训机构的 AI 原生教学辅助系统。  
> 基于 **LangChain 1.2.10 + LangGraph 1.0.9 + FastAPI** 构建，核心大模型为 **DeepSeek-V3 / DeepSeek-Coder-V2**。

---

## 项目背景

IT 教培机构每天面对四类重复性高、耗费师资的业务：

| 场景 | 痛点 |
|------|------|
| 学员答疑 | 同类问题反复出现，老师精力有限 |
| 试卷批改 | 量大、主观题评分容易不一致 |
| 简历审查 | 老师无暇逐份精读打分 |
| 技术面试 | 缺少随时可陪练并给出专业反馈的「面试官」 |

EduAgent 将这四类业务分别封装为独立的 AI Agent，每个 Agent 融合了企业私有知识、内置了完整业务流程、并配备了工程化容错机制。

---

## 四大核心 Agent

| Agent | 业务能力 | 核心技术范式 |
|-------|---------|------------|
| **智能问答（QA）** | 基于课程知识库实时答疑 | RAG 混合检索 + BGE-M3 + Reranker 精排 + SSE 流式 |
| **试卷批改（Exam）** | 自动批改选择题 / 简答题 / 编程题 | 三轨并行批改 + HitL 教师确认环 |
| **简历审查（Resume）** | 六维度评分并给出改进建议 | PDF 解析 + 结构化抽取 + 多维度并行评分 |
| **模拟面试（Interview）** | 五阶段全程技术面试陪练 | 状态机对话 + Think Tool 双轨质量评估 |

四个 Agent 之上由 **Orchestrator 编排层**统一做意图路由与多 Agent 串联 Pipeline。

---

## 技术栈

| 层面 | 选型 |
|------|------|
| 开发语言 | Python 3.11（严格锁定） |
| Web 框架 | FastAPI + SSE-Starlette |
| Agent 框架 | LangChain 1.2.10 + LangGraph 1.0.9 |
| 主力大模型 | DeepSeek-V3（问答 / 简历 / 面试）|
| 代码批改模型 | DeepSeek-Coder-V2 |
| 向量数据库 | Milvus |
| 关系数据库 | PostgreSQL + SQLAlchemy（全异步）|
| 缓存 | Redis |
| 对象存储 | MinIO |
| 嵌入模型 | BGE-M3（进程内，dense + sparse 双输出）|
| 精排模型 | BGE-Reranker-large（进程内）|
| 意图分类 | MiniLM-L6-v2（进程内）|
| 前端 | Vue3 + TypeScript + Element Plus |

> BGE-M3、BGE-Reranker、MiniLM **三个本地模型均为进程内调用**，无需单独起服务，启动时随后端进程并行预热。

---

## 系统架构

```
┌─────────────────────────────────────────────────┐
│              前端层   Vue3 SPA (:3000)            │
└────────────────────┬────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────▼────────────────────────────┐
│            API 层   FastAPI (:8000)              │
│          JWT 鉴权 / SSE 流式 / 文件上传            │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│          编排层   Orchestrator                   │
│    MiniLM 意图识别 → 单 Agent 路由 / Pipeline     │
└──────┬───────────┬───────────┬───────────┬──────┘
       │           │           │           │
  ┌────▼───┐  ┌───▼────┐  ┌───▼────┐  ┌───▼──────┐
  │  QA    │  │  Exam  │  │Resume  │  │Interview │  ← LangGraph StateGraph × 4
  │ Agent  │  │ Agent  │  │ Agent  │  │  Agent   │
  └────────┘  └────────┘  └────────┘  └──────────┘
                       公共层
       ├── LLM Factory（DeepSeek API 统一封装）
       ├── BGE-Reranker / MiniLM / MemorySaver
       └── MCP Server（知识库检索 + Web 搜索）
                     │
┌────────────────────▼────────────────────────────┐
│   数据层   PostgreSQL · Redis · Milvus · MinIO   │
└─────────────────────────────────────────────────┘
```

---

## 环境要求

- Python **3.11**（严格锁定，不兼容其他版本）
- uv（推荐用于环境隔离）
- PostgreSQL 15
- Milvus 2.6.6
- DeepSeek v4 及相应 API Key

---

## 快速开始

### 1. 克隆项目 & 创建环境

```bash
git clone git@gitee.com:yorkxiangyu/eduagent.git
cd 项目根目录

uv init --name myproject --python 3.11.9 --bare

uv python pin 3.11.9 

echo. >> pyproject.toml && echo [tool.uv] >> pyproject.toml && echo index-url = "https://mirrors.aliyun.com/pypi/simple/" >> pyproject.toml

uv add setuptools==79.0.1

uv add -r requirements.txt

uv pip install --prerelease=allow "langchain-community==0.4.2" pypdf==6.14.2
```

### 2. 配置环境变量

打开 `.env.local`，修改你的配置

### 3. 启动基础设施

使用 PostgreSQL Milvus

```bash
cd /root/data_docker
docker-compose up -d
```

### 4. 初始化数据（首次运行）

```bash
切换虚拟环境
cd 项目根目录
python scripts/init_minio.py        # 创建 MinIO Bucket
python scripts/init_milvus.py       # 创建 Milvus Collection
python scripts/seed_data.py         # 写入测试用户与试卷数据
```

### 5. 验证环境

```bash
python scripts/verify_env.py
```

### 6. 启动后端

```bash
# 在项目根目录下执行
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 7. 启动前端

```bash
cd frontend
npm install
npm run dev
```

---

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI 后端 | **8000** | REST + SSE 接口，`/docs` 查看 Swagger |
| Vue3 前端 | **3000** | 学员 / 教师 / 管理员界面 |
| PostgreSQL | **5432** | 数据库 |
| Milvus | **19530** | 向量库 |
| Langfuse | **3100** | LLM 调用追踪与可观测性 |

> 所有连接配置统一从 `.env.local` 读取，**禁止硬编码端口号**。

---

## 访问地址 & 测试账号

| 地址 | 说明 |
|------|------|
| http://localhost:3000 | 前端应用 |
| http://localhost:8000/docs | FastAPI 接口文档（Swagger） |
| http://192.168.88.101:3100 | Langfuse 可观测性平台 |
| http://192.168.88.101:3000 | Attu milvus的可视化界面 |

| 角色   | 用户名    | 邮箱                     | 密码           |
| :----- | :-------- | :----------------------- | :------------- |
| 管理员 | admin     | admin@eduagent.local     | Admin@123456   |
| 教师   | teacher01 | teacher01@eduagent.local | Teacher@123456 |
| 学生   | student01 | student01@eduagent.local | Student@123456 |
| 学生   | student02 | student02@eduagent.local | Student@123456 |

__________________

## 目录结构

```
EduAgent项目/
├── .env.local                       # 环境变量配置（DB/Milvus/DeepSeek/JWT/Langfuse）
├── .gitignore
├── main.py                          # 根目录启动入口（开发用）
├── requirements.txt                 # Python 依赖清单
│
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 入口 & lifespan（模型预热 + DB 迁移）
│   ├── main_bak.py                  # 入口备份
│   ├── config.py                    # pydantic-settings 配置中心（读 .env.local）
│   ├── dependencies.py              # JWT 鉴权依赖注入 + AsyncSessionLocal
│   ├── langfuse_handler.py          # Langfuse 可观测性回调处理器
│   │
│   ├── api/
│   │   ├── router.py                # 路由汇总注册
│   │   └── v1/
│   │       ├── auth.py              # 登录/注册/Token 刷新
│   │       ├── qa.py                # QA 问答（chat / stream / history）
│   │       ├── exam.py              # 试卷批改（提交/查询/教师确认）
│   │       ├── resume.py            # 简历审查（上传/查询报告）
│   │       ├── interview.py         # 模拟面试（setup/chat/report）
│   │       └── unified_chat.py      # 第8章统一助手入口（意图路由 + SSE）
│   │
│   ├── agents/                      # 四大 Agent（每个含 graph/nodes/state/prompts）
│   │   ├── qa/                      # RAG 智能问答（第5章）
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   ├── state.py
│   │   │   └── prompts.py
│   │   ├── exam/                    # 试卷批改 HitL（第6章）
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   ├── state.py
│   │   │   └── prompts.py
│   │   ├── resume/                  # 简历审查 fan-out（第4/6章）
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   ├── state.py
│   │   │   └── prompts.py
│   │   └── interview/               # 模拟面试状态机（第7章）
│   │       ├── graph.py
│   │       ├── nodes.py
│   │       ├── state.py
│   │       └── prompts.py
│   │
│   ├── core/                        # 公共核心模块
│   │   ├── llm_factory.py           # LLM 统一工厂（DeepSeek via OpenAI 兼容接口）
│   │   ├── orchestrator.py          # 编排器（意图路由 + 单Agent/Pipeline 分发）
│   │   ├── knowledge_base.py        # BGE-M3 向量检索（Milvus）
│   │   ├── reranker.py              # BGE-Reranker 精排
│   │   ├── query_classifier.py      # Query 二分类器（general/specialized）
│   │   ├── memory.py                # MemorySaver 对话记忆 + build_config
│   │   ├── retry.py                 # 三层兜底重试装饰器（第3章）
│   │   ├── exceptions.py            # 统一异常体系
│   │   └── logger.py                # structlog 日志封装
│   │
│   ├── db/
│   │   └── migrations.py            # 数据迁移
│   │
│   └── mcp_server/                  # MCP Server（第5章）
│       ├── client.py                # MCP 客户端封装
│       ├── knowledge_base_server.py # 知识库检索 MCP 工具
│       └── web_search_server.py     # 联网搜索 MCP 工具
│
├── frontend/                        # Vue3 + Vite 前端
│
├── models/                          # 本地模型权重（.gitignore）
│   ├── embedding/bge-m3/            # 向量嵌入模型
│   ├── reranker/bge-reranker-v2-m3/ # 精排模型
│   └── classifier/
│       ├── all-MiniLM-L6-v2/        # 分类器基座
│       └── query-classifier-finetuned/  # 微调后分类器（含 checkpoints）
│
├── scripts/                         # 运维 & 数据脚本
│   ├── init_db.sql                  # PostgreSQL 建表 DDL
│   ├── init_milvus.py               # 初始化 Milvus 向量集合
│   ├── seed_data.py                 # 填充测试账号 + 基础数据
│   ├── seed_interview_questions.py  # 填充面试题库
│   ├── seed_standard_exam.py        # 填充标准试卷
│   ├── build_knowledge_base.py      # 导入知识库文档到 Milvus
│   └── manual_tests/                # 各章手动测试脚本
│       ├── itv_07_*.py              # 第7章面试节点逐步验证
│       ├── orch_08_*.py             # 第8章编排器验证
│       └── uc_08_*.py               # 第8章统一入口验证
│
├── test/                            # 环境自检 & 接口测试
│   ├── check_env.py
│   ├── check_build_knowledge.py
│   ├── check_mcp_server.py
│   ├── check_qa_httpapi.py
│   ├── check_exam_httpapi.py
│   ├── check_exam_graph.py
│   ├── check_search_duckduckgo.py
│   └── check_search_tavily.py
│
├── logs/                            # 运行日志
│   └── app.log
│
└── tmp/                             # 临时文件（上传的简历/试卷/训练数据）
```

---

## 知识库导入

首次使用或知识库重建后，需手动导入课程文档：

```bash
# 切换python虚拟环境
cd 项目根目录

# 重建 Milvus Collection
python scripts/init_milvus.py

# 导入文档（支持 PDF / Word / Markdown）
python scripts/build_knowledge_base.py --file <文档路径> --course_id <课程UUID>
```

---

## 容错机制

系统内置三层兜底，任何情况下用户都能拿到响应：

| 层级 | 触发条件 | 处理方式 |
|------|---------|---------|
| 第一层：自动重试 | 网络抖动 / LLM 超时 | 间隔 1s / 3s 重试，最多 2 次 |
| 第二层：Agent 降级 | 重试后仍失败 | 问答直答 / 批改跳过 Judge0 / 简历给格式提示 / 面试给基础反馈 |
| 第三层：系统兜底 | 所有降级均失败 | 友好提示 + 已完成结果持久化 + 错误记录 |

---

## 常见问题

**Q: Milvus 连接超时**

确认 etcd 和 milvus-standalone 容器均已 healthy：

```bash
docker-compose ps | grep -E "milvus|etcd"
```

**Q: 前端 SSE 无流式效果**

QA 和面试的 SSE 接口直连 `http://localhost:8000`，不经过 Vite proxy，属于正常设计。

**Q: 模拟面试如何快速结束？**

点击输入框右侧的**"结束面试"**按钮，AI 将跳过剩余阶段直接生成报告（约 10-30 秒）。

**Q: 试卷批改一直显示"AI 批改中"**

批改为后台异步任务，刷新页面可获取最新状态。若超过 15 分钟未完成，记录将自动标记为 `failed`，可重新提交。

---