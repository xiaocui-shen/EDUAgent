# backend/agents/resume/nodes.py

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.resume.state import (
    ResumeState, # 简历审查 Agent 完整 State
    ResumeStructured, # 简历结构化提取结果
    DimensionScore # 单个评审维度结果
)
from backend.agents.resume.prompts import (
    SYSTEM_PROMPT, # 系统提示词
    EXTRACT_STRUCTURED_PROMPT, # 结构化提取提示词
    DIMENSION_REVIEW_PROMPTS # 六维度评审提示词
)
from backend.core.llm_factory import get_structured_llm

from backend.core.logger import get_logger
logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# 四个基础节点：对pdf简历进行 上传、下载、文本提取、结构化提取
# ──────────────────────────────────────────────────────────────
# ── 节点①：upload_to_minio —— 本地模式空跑 ─────────────────────
async def upload_to_minio_node(state: ResumeState) -> dict:
    """预留节点：上传文件到云上对象存储（OSS/S3）
    文件已在本地临时路径，无需上传对象存储。"""
    logger.info("upload_to_minio.skip",
                review_id=state["review_id"],
                reason="local_mode")
    return {} # 返回空字典：不更新任何 State 字段，流程照常往下走


# ── 节点②：download_pdf —— 本地模式空跑 ───────────────────────
async def download_pdf_node(state: ResumeState) -> dict:
    """预留节点：从云上对象存储（OSS/S3）下载到本地
    文件已在 pdf_local_path，无需从对象存储下载，直接跳过。"""
    logger.info("download_pdf.skip",
                review_id=state["review_id"],
                reason="local_file_exists")
    return {}


# ── 节点③：extract_text —— PDF 文本提取 ───────────────────────
def _sync_extract_text(pdf_path: str) -> dict:
    """同步方法 提取PDF文本，不考虑分栏"""
    import fitz # 局部导包
    all_text_parts = [] # 存储所有文本

    # 使用上下文管理器，确保文件句柄在异常时也能安全释放
    with fitz.open(pdf_path) as doc:
        page_count = len(doc) # 页码数
        for page in doc: # 遍历 PDF 每页
            blocks = page.get_text("blocks") # 获取当前页的所有块
            '''PDF blocks 元组七字段拆解
            page.get_text("blocks") 返回每个文本/图片块，是个7元素的元组
            每个块 = (x0, y0, x1, y1, text, block_no, block_type)：
                1. x0/y0/x1/y1 组成这个内容块在 PDF 页面上的矩形边界
                    x0：块左上角 X 坐标
                    y0：块左上角 Y 坐标
                    x1：块右下角 X 坐标
                    y1：块右下角 Y 坐标
                2. text：块内全部文本内容（图片块这里是空字符串）
                3. block_no：块序号，一页内从 0 开始递增
                4. block_type：块类型标记
                    0 = 文字块（你代码保留的有效内容）
                    1 = 图片块（直接过滤丢弃）
            '''
            # 只保留文字块（type==0），过滤图片（type==1）
            text_blocks = [b for b in blocks if b[6] == 0]
            if not text_blocks:
                continue # 跳过没有文本的页

            # 所有块进行排序（不考虑分栏、IT行业简历几乎都是纵向单栏）
            # 先按y0纵坐标升序，同高度(差值<1)再按x0横坐标升序排序
            # PDF同行文本，因字体等微小差异，导致y0纵坐标不完全相等
            # round(b[1], 1)的作用是消除微小差异影响、对齐同行
            sorted_blocks = sorted(
                text_blocks,
                key=lambda b: (round(b[1], 1), b[0])
            )
            # sorted_blocks = sorted(text_blocks, key=lambda b: b[1]) # 简化 不考虑行内文本块上下浮动的情况

            # 拼接当前页的文本，过滤空行
            page_text = "\n".join(
                b[4].strip() # 块的第5个元素是文本内容，去除首尾空格
                for b in sorted_blocks
                if b[4].strip() # 如果文本块去除首尾空格后，非空
            )
            # 当前页文本添加到结果列表
            all_text_parts.append(page_text)

    # 每页加分页标记，方便 LLM 识别
    raw_text = "\n\n---PAGE BREAK---\n\n".join(all_text_parts)
    # 返回全部文本字符串和页码数
    return {"raw_text": raw_text, "page_count": page_count}


async def extract_text_node(state: ResumeState) -> dict:
    """异步节点：用线程池跑同步的 PDF 解析，避免阻塞事件循环。"""
    pdf_path = state["pdf_local_path"]
    try:
        loop = asyncio.get_running_loop()
        # run_in_executor：把同步阻塞的解析丢到线程池
        result = await loop.run_in_executor(
            None, # 用asyncio自带默认线程池，简单省事
            _sync_extract_text, # 同步方法
            pdf_path # 参数 PDF文件路径
        )
        raw_text   = result["raw_text"] # PDF全文文本字符串
        page_count = result["page_count"] # 页码数

        # 文字过少多半是扫描件/图片 PDF 告警
        if len(raw_text.strip()) < 200:
            logger.warning("extract_text.text_too_short",
                           text_length=len(raw_text.strip()))
        # 输出日志并返回pdf文本字符串 和 页码数
        logger.info("extract_text.done",
                    page_count=page_count,
                    text_length=len(raw_text))
        return {"raw_text": raw_text, "page_count": page_count}
    except Exception as e:
        logger.error("extract_text.failed",
                     error=str(e))
        raise


# ── 节点④：extract_structured —— LLM 结构化提取 ───────────────
async def extract_structured_node(state: ResumeState) -> dict:
    """用 LLM Function Calling 把原始文本提取成结构化简历（ResumeStructured）。"""
    raw_text = state["raw_text"]
    # 超长截断：截前 4000 字，防止超出 context 并省 token
    text_for_llm = raw_text[:4000] \
        if len(raw_text) > 4000 else raw_text
    # 简历文本添加到提示词
    prompt = EXTRACT_STRUCTURED_PROMPT.format(
        resume_text=text_for_llm)

    # 获取绑定了 ResumeStructured 结构的简历处理的llm对象
    structured_llm = get_structured_llm(
        "resume", ResumeStructured)

    structured_dict = None # 最终结构化输出字典
    for attempt in range(2): # 最多尝试 2 次（结构化输出偶发返回 None）
        try: # 调用 LLM
            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            if result is None: # 如果llm返回 None
                raise ValueError("structured output returned None")
            # 结构化输出对象转成字典
            structured_dict = result.model_dump()
            break
        except Exception as e:
            if attempt == 0: # 第1次失败停1秒后重试
                logger.warning("extract_structured.retry",
                               error=str(e))
                await asyncio.sleep(1)
            else: # 第2次失败
                logger.warning("extract_structured.failed",
                               error=str(e))

    if structured_dict is None: # 两次都失败 → 降级为空结果字典
        structured_dict = ResumeStructured(
            name="未能提取").model_dump()

    # 结构化输出结束
    logger.info("extract_structured.done",
                name=structured_dict.get("name", ""),
                projects_count=
                    len(structured_dict.get("projects", [])))
    # 返回结构化输出字典
    return {"structured": structured_dict}


# ──────────────────────────────────────────────────────────────
# Part 2 六维评分、综合加权评分
# ──────────────────────────────────────────────────────────────
# 六维度定义（名称 / 权重 / 评分侧重）。权重之和 = 1.0
# ──────────────────────────────────────────────────────────────
SIX_DIMENSIONS = [
    {"key": "project_depth",  "name": "项目深度",   "weight": 0.30,
     "focus": "项目描述是否有量化数据、技术选型理由、个人贡献、难点解决"},
    {"key": "tech_match",     "name": "技术匹配度", "weight": 0.25,
     "focus": "技术栈是否与目标岗位匹配，技能描述是否有层次（熟练/了解/掌握）"},
    {"key": "expression",     "name": "表达规范性", "weight": 0.15,
     "focus": "动词开头、STAR 结构、无错别字、无主语省略歧义"},
    {"key": "structure",      "name": "简历结构",   "weight": 0.15,
     "focus": "模块完整性、排版逻辑、信息密度、重要内容是否放前面"},
    {"key": "quantification", "name": "量化程度",   "weight": 0.10,
     "focus": "性能指标、用户量、优化幅度等量化数据的使用情况"},
    {"key": "authenticity",   "name": "真实可信度", "weight": 0.05,
     "focus": "表述是否夸大、技术深度描述是否与经验年限匹配、时间线是否合理"},
]


# ──────────────────────────────────────────────────────────────
# 节点⑤：run_six_dimensions —— 六维度并行评审
# ──────────────────────────────────────────────────────────────
async def run_six_dimensions_node(state: ResumeState) -> dict:
    """六维度并行评审：同时发 6个LLM 请求，各自独立评分，最后算加权综合分。"""
    # todo 1. 精简结构化摘要，省 token
    structured = state.get("structured") or {} # 节点④更新的结构化输出字典
    structured_summary = _build_structured_summary(structured)

    # 单个维度评审逻辑
    async def review_one_dimension(dim: dict) -> dict:
        """评审单个维度，返回 DimensionScore 单维度结果dict（内含 2 次尝试）。
        dim: 当前维度描述，从 SIX_DIMENSIONS 获取"""
        # todo 2.1. 准备提示词
        # 按维度获取提示词模板
        prompt_template = DIMENSION_REVIEW_PROMPTS.get(dim["key"], "")
        if not prompt_template: # 找不到对应提示词，直接降级
            return _empty_dimension_score(dim)
        # 拼接提示词
        prompt = prompt_template.format(
            resume_text=state["raw_text"][:3000], # 原文前 3000 字
            structured_summary=structured_summary, # 结构化摘要
            focus=dim["focus"], # 评分侧重点
        )

        # todo 2.2. llm推理，最多2次尝试，不行就降级
        for attempt in range(2): # 最多尝试 2 次
            try: # 从llm工厂获取llm，执行推理，获取结构化的结果并转字典
                structured_llm = get_structured_llm(
                    "resume", DimensionScore)
                result: DimensionScore = await structured_llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                # None检查：DeepSeek偶发生成无效JSON导致返回None
                if result is None:
                    raise ValueError("structured output returned None")
                d = result.model_dump() # 结果转字典
                d["dimension"] = dim["name"] # 中文维度名
                d["weight"] = dim["weight"] # 维度权重
                d["key"] = dim["key"] # 维度key 英文名
                return d # 返回结果
            except Exception as e:
                if attempt == 0: # 第一次失败：等 1 秒后重试
                    logger.warning("six_dimensions.dimension_retry",
                                   dimension=dim["name"], error=str(e))
                    await asyncio.sleep(1)
                else: # 第二次仍失败，直接降级
                    logger.warning("six_dimensions.dimension_failed",
                                   dimension=dim["name"], error=str(e))
                    return _empty_dimension_score(dim)

    # todo 2. 六维度并行：先建6个评审单个维度的协程任务，再用gather一起跑、等全部完成
    tasks = [review_one_dimension(dim) for dim in SIX_DIMENSIONS]
    dimension_scores = await asyncio.gather(*tasks)

    # todo 3. 算加权综合分 = Σ(得分 × 权重)
    #  权重和为1.0，每个维度得分 × 各自权重，再求和，结果落在0-100
    weighted_score = sum(d["score"] * d["weight"] for d in dimension_scores)

    logger.info("six_dimensions.done",
                weighted_score=round(weighted_score, 2),
                scores={d["key"]: d["score"] for d in dimension_scores})

    return {
        "dimension_scores": list(dimension_scores),
        "weighted_score":   round(weighted_score, 2),
    }


# ── 辅助：从结构化数据构建简洁摘要（供评审 Prompt 使用，省 token）──
def _build_structured_summary(structured: dict) -> str:
    lines = []
    if structured.get("name"):
        lines.append(f"姓名：{structured['name']}")
    if structured.get("target_position"): # 字典.get(key) 取不到时返回 None
        lines.append(f"求职意向：{structured['target_position']}")
    if structured.get("education"):
        # 只要第一条教育经历，最高学历写后边就活该
        edu = structured["education"][0]
        lines.append(f"最高学历：{edu.get('school','')} "
                     f"{edu.get('major','')} {edu.get('degree','')}")
    if structured.get("skills_list"):
        # 技能列表只取前10个，符合人真实的看简历体验（着重看前边、太多后边不爱看了）
        lines.append(f"技术栈：{', '.join(structured['skills_list'][:10])}")
    if structured.get("projects"):
        proj_names = [p.get("name", "") for p in structured["projects"]]
        # 项目数 + 前 3 个项目名
        lines.append(f"项目数量：{len(structured['projects'])} 个"
                     f"（{', '.join(proj_names[:3])}）")
    if structured.get("work_experience"):
        # 工作经历只写段数，不展开——因为六维度评审主要看项目细节，工作经历帮不上忙
        lines.append(f"工作经历：{len(structured['work_experience'])} 段")
    return "\n".join(lines) if lines else "（结构化提取失败，请基于原文评审）"


# ── 辅助：维度评审失败时的降级结果（给中性 50 分 + 标记复核）──
def _empty_dimension_score(dim: dict) -> dict:
    return {
        "key":         dim["key"],
        "dimension":   dim["name"],
        "score":       50,
        "weight":      dim["weight"],
        "issues":      ["该维度评审失败，建议人工复核"],
        "suggestions": [],
    }

# ==============================================================
# ──────────────────────────────────────────────────────────────
# Part 3 逐条问题诊断、整体评价
# ──────────────────────────────────────────────────────────────

from backend.core.llm_factory import get_llm
from backend.agents.resume.prompts import (
    DIAGNOSE_THINK_PROMPT, # 简历问题诊断清单之前的，宏观分析提示词
    DIAGNOSE_ISSUES_PROMPT, # 简历的逐条问题诊断 提示词
    GENERATE_SUMMARY_PROMPT # 简历审查的整体评价报告 提示词
)
from backend.agents.resume.state import (
    IssueList, # [单条诊断问题, ...]state
    ResumeSummary # 简历整体评价state
)

# ──────────────────────────────────────────────────────────────
# 节点⑥：diagnose_issues —— 逐条问题诊断
# ──────────────────────────────────────────────────────────────
async def diagnose_issues_node(state: ResumeState) -> dict:
    """汇总六维度问题，Think 前置推理后生成带优先级的诊断清单，并排序。"""
    # 六维度评分节点产出的完整结果
    dimension_scores = state.get("dimension_scores", [])
    raw_text = state["raw_text"] # 简历原始文本
    structured = state.get("structured") or {} # 简历结构化数据

    # todo ① 汇总各维度已发现的问题文本
    all_raw_issues = [] # 所有维度问题列表
    for dim in dimension_scores: # 逐个维度
        for issue_text in dim.get("issues", []): # 逐个问题
            # '[维度名称] + 具体一个问题文本' 添加到所有维度问题列表
            all_raw_issues.append(f"[{dim['dimension']}] {issue_text}")
    # 拼接所有问题文本字符串 '- [维度名称] + 具体一个问题文本' + \n + '- [维度名称...
    raw_issues_text = "\n".join(f"- {i}" for i in all_raw_issues) or "（暂无）"

    # todo ② Think宏观分析：前置推理 失败不影响主流程
    reasoning_trace = ""
    try:
        dimension_scores_summary = "\n".join(
            f"- {d['dimension']}：{d['score']}分"
            f" — 问题：{', '.join(d.get('issues', [])[:2])}"
            for d in dimension_scores
        )
        '''dimension_scores_summary 大概长这样：
        - 项目深度：75分 — 问题：缺少量化指标, 个人贡献不突出
        - 技术匹配度：85分 — 问题：部分技能未分级
        - 表达规范性：90分 — 问题：
        - 简历结构：70分 — 问题：模块顺序不合理
        - 量化程度：60分 — 问题：大部分无量化数据, 缺少性能指标
        - 真实可信度：88分 — 问题：时间线有1处断层
        '''
        # 构造 宏观分析 提示词
        think_prompt = DIAGNOSE_THINK_PROMPT.format(
            dimension_scores_summary=dimension_scores_summary,
            raw_issues=raw_issues_text,
        )
        # 用普通非结构化 LLM 做"先想后答"，返回一段文本（非结构化输出）
        # 目的：让 LLM 先宏观分析各维度评分的关联性、找出共同短板，
        # 后续结构化诊断时带上这段分析作为附加上下文，诊断更准确。
        think_llm = get_llm("resume", temperature=0)
        think_resp = await think_llm.ainvoke(
            [HumanMessage(content=think_prompt)])
        # 兼容不同 LangChain 版本的返回对象：
        # - 新版 AIMessage 有 .text 属性（直接拿字符串）
        # - 旧版 .text 可能是个方法（callable）而不是属性，不能直接 .text
        # - .content 是兜底方案，所有版本都有，但可能是列表格式
        # 判断逻辑：有 .text 且它不是方法 → 用 .text；否则降级用 .content 转字符串
        reasoning_trace = (
            think_resp.text 
            if hasattr(think_resp, "text")     # 1️⃣ 对象有这个属性吗？
               and not callable(think_resp.text) # 2️⃣ 且它不是可调用的方法
            else str(think_resp.content)         # 否则用 .content 兜底
        ).strip()
        logger.debug("diagnose_think.done",
                     dimensions=len(dimension_scores))
    except Exception as e:
        logger.warning("diagnose_think.failed", error=str(e))

    # todo ③ 结构化生成问题清单（IssueList）
    # 把 Think 结果作为附加上下文拼到诊断提示后面
    think_context = f"\n\n【诊断前宏观分析】\n{reasoning_trace}" \
        if reasoning_trace else ""
    # 构造 逐条问题诊断 提示词
    prompt = DIAGNOSE_ISSUES_PROMPT.format(
        resume_text=raw_text[:3000], # 简历文本前3000字
        structured_summary=_build_structured_summary(structured), # 摘要
        raw_issues=raw_issues_text, # 所有问题文本字符串
    ) + think_context

    try: # 用结构化 LLM 生成结构化问题清单（IssueList）
        structured_llm = get_structured_llm("resume", IssueList)
        result= await structured_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]) # 返回 IssueList 类型
        # None检查：DeepSeek偶发生成无效JSON导致返回None
        if result is None:
            raise ValueError("structured output returned None")
        # 每条 IssueItem 转 dict，返回 [dict, ...]
        issues = [item.model_dump() for item in result.items]
    except Exception as e:
        logger.warning("diagnose_issues.failed", error=str(e))
        # 降级：直接用各维度问题，统一标 medium
        issues = [
            {"priority": "medium",
             "dimension": dim["dimension"],
             "description": issue, # 每个问题文本字符串
             "location": "简历全文",
             "suggestion": "请参考评审建议修改"}
            for dim in dimension_scores # 先遍历维度
            for issue in dim.get("issues", []) # 再遍历出每个问题的文本字符串
        ]

    # todo ④ 按优先级排序：high(0) → medium(1) → low(2)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    # x是issues中的元素，是个dict，x.get("priority", "low")返回 high/medium/low
    # priority_order.get(high/medium/low) 返回 0/1/2
    # issues.sort(key=0/1/2) 基于0/1/2进行排序，升序
    issues.sort(key=lambda x: priority_order.get(x.get("priority", "low")))

    logger.info("diagnose_issues.done",
                total=len(issues),
                high=sum(1 for i in issues if i.get("priority") == "high"))
    return {"issues": issues} # 返回已经排序后的结构化问题清单 {"issues": [dict]}


# ──────────────────────────────────────────────────────────────
# 节点⑦：generate_summary —— 整体评价生成
# ──────────────────────────────────────────────────────────────
async def generate_summary_node(state: ResumeState) -> dict:
    """综合结构化信息、六维度评分、问题清单，生成面向学员的整体评价。"""
    structured = state.get("structured") or {} # 简历文本结构化信息
    dimension_scores = state.get("dimension_scores", []) # 六维度评分及详情
    issues = state.get("issues", []) # 结构化问题清单
    weighted_score = state.get("weighted_score", 0.0) # 总得分

    # 挑出高优先级问题（最多 5 条）
    high_issues = [
        i["description"]
        for i in issues
        if i.get("priority") == "high"
    ][:5]
    # 5条高优先级问题 拼接成文本字符串
    high_issues_text = ("\n".join(f"- {i}" for i in high_issues)
                        or "（无高优先级问题）")
    # 六维度得分拼接文本字符串
    scores_text = "\n".join(
        f"- {d['dimension']}：{d['score']}分（权重{int(d['weight'] * 100)}%）"
        for d in dimension_scores
    )
    # 简历整体评价报告 提示词
    prompt = GENERATE_SUMMARY_PROMPT.format(
        structured_summary=_build_structured_summary(structured),
        scores_summary=scores_text, # 六维度得分拼接文本字符串
        weighted_score=round(weighted_score, 1), # 总得分
        high_issues=high_issues_text, # 5条高优先级问题文本字符串
        # 目标职位
        target_position=structured.get("target_position", "后端开发"),
    )
    # 结构化 LLM，用于生成结构化报告，输出目标：ResumeSummary
    structured_llm = get_structured_llm("resume", ResumeSummary)

    summary_dict = None
    for attempt in range(2): # 最多尝试 2 次（结构化输出偶发返回 None）
        try: # 推理：用结构化 LLM 生成结构化报告
            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            if result is None: # 模型没按结构化输出 → None
                raise ValueError("structured output returned None")
            # 结构化的推理结果对象转为 dict
            summary_dict = result.model_dump()
            break
        except Exception as e:
            if attempt == 0: # 第一次失败，1秒后重试
                logger.warning("generate_summary.retry", error=str(e))
                await asyncio.sleep(1)
            else: # 第两次失败
                logger.warning("generate_summary.failed", error=str(e))

    if summary_dict is None: # 两次都失败 → 用默认评价兜底
        summary_dict = {
            "highlights": ["简历内容已完整提交"],
            "core_improvements":
                high_issues[:2] if high_issues else ["请参考各维度建议修改"],
            "overall_comment":
                f"综合评分 {round(weighted_score, 1)} 分，请参考各维度详细反馈。",
            "fit_assessment": "与目标岗位匹配度评估暂不可用",
        }

    logger.info("generate_summary.done",
                highlights_count=len(summary_dict.get("highlights", [])))
    return {"summary": summary_dict} # 返回结构化报告


# ==============================================================
# ──────────────────────────────────────────────────────────────
# Part 3:  节点⑧：save_results  持久化结果
# ──────────────────────────────────────────────────────────────
import os, json
from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal

async def save_results_node(state: ResumeState) -> dict:
    """把完整结果写入 resume_reviews（JSONB 字段），并清理临时文件。"""
    review_id = state["review_id"] # 评价 ID
    # 留一份完整结果给上层（API 可直接用）
    structured_output = {
        "review_id": review_id, "student_id": state["student_id"],
        "structured": state.get("structured"),
        "weighted_score": state.get("weighted_score", 0),
        "dimension_scores": state.get("dimension_scores", []),
        "issues": state.get("issues", []),
        "summary": state.get("summary"),
    }

    # 用统一的 SQLAlchemy 异步会话
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("""UPDATE resume_reviews
                        SET structured_data = :structured_data,
                            scores          = :scores,
                            issues          = :issues,
                            summary         = :summary,
                            status          = 'done',
                            updated_at      = NOW()
                        WHERE id = :review_id """),
                {
                    # JSONB 列：先 json.dumps 转 JSON 字符串；ensure_ascii=False 保留中文原文
                    "structured_data": json.dumps(
                        state.get("structured"), ensure_ascii=False),
                    "scores": json.dumps(
                        obj={
                            "dimension_scores":
                                state.get("dimension_scores", []),
                            "weighted_score":
                                state.get("weighted_score", 0)
                        },
                        ensure_ascii=False),
                    "issues":
                        json.dumps(state.get("issues", []), ensure_ascii=False),
                    "summary":
                        json.dumps(state.get("summary"), ensure_ascii=False),
                    "review_id": review_id,
                },
            )
            await session.commit()
            logger.info("save_results.db_written", review_id=review_id)
        except Exception as e:
            await session.rollback()
            logger.error("save_results.db_failed", error=str(e))
            raise

    # 清理本地临时 PDF
    local_path = state.get("pdf_local_path", "")
    if local_path and os.path.exists(local_path):
        os.remove(local_path)
        logger.info("save_results.tmp_cleaned", path=local_path)

    return {"fallback_used": False, "structured_output": structured_output}


if __name__ == "__main__":

    # ── 模块自测 1：实测 PDF文本提取 及LLM取结构化提内容────
    import fitz
    from backend.config import get_abs_path
    state = {"pdf_local_path":
             get_abs_path("tmp/简历示例.pdf")}

    async def check_extract_structured(state):
        # 先提取文本
        # state.update(...) 字典更新
        state.update(await extract_text_node(state))
        print("\n自测1[离线！]➡️ 提取文本测试：")
        print("\tpage_count:", state["page_count"])
        print("\traw_text长度:", len(state["raw_text"]))
        # print("raw_text:", _result["raw_text"])
        print("\t含 卷积神经网络:", "卷积神经网络" in state["raw_text"])
        print("\t含 大模型微调:", "大模型微调" in state["raw_text"])
        # 再让LLM结构化输出
        state.update(await extract_structured_node(state))
        print("\n自测1➡️[在线！] LLM结构化提取测试：")
        print("\t姓名:", state["structured"]["name"])
        print("\t技能:", state["structured"]["skills_list"])
        print("\t项目数量:", len(state["structured"]["projects"]))
        print("\t工作经历数量:", len(state["structured"]["work_experience"]))
        # print("\t整个state:", state)
        return state
    # 执行测试1
    # state = asyncio.run(check_extract_structured(state))
    print('-' * 20)

    # ── 模块自测 2：对六维度并行评审进行离线测试：(可单独运行)
    # 验证 并行（计时）+ 加权
    # 验证 失败降级
    import time
    async def check_six_dimensions_fake():
        print('\n➡️ 自测2[离线！]六维度并行评审测试')
        # 验证：全部失败 → 验证降级 的逻辑
        class FakeLLM:
            """假模型：sleep 0.1s 模拟网络，返回固定分数（或抛错模拟失败）。"""
            def __init__(self, score, fail=False):
                self.score, self.fail = score, fail
            async def ainvoke(self, messages):
                await asyncio.sleep(0.1)
                if self.fail:
                    raise RuntimeError("模拟维度失败")
                return DimensionScore(score=self.score)
        # 替换覆盖模型工厂函数，使维度模型都返回固定分数 或抛错模拟失败
        global get_structured_llm

        # ① 全部返回 80 分 → 验证并行（计时）+ 加权
        state = {"raw_text": "简历原文 ",
                 "structured": {"name": "张三", "skills_list": ["Java", "Redis"], "projects": [{"name": "电商系统"}]}}
        get_structured_llm = lambda agent_type, schema: (
            FakeLLM(score=80, fail=False)) # 强制80分，
        t = time.perf_counter()
        state.update(await run_six_dimensions_node(state))
        print("① 维度数:", len(state["dimension_scores"]))
        print("   加权综合分:", state["weighted_score"])
        print("   耗时:", round(time.perf_counter() - t, 2), "秒")
        print('\n')

        # ② 全部失败 → 验证降级
        state = {"raw_text": "简历原文 ",
                 "structured": {"name": "张三", "skills_list": ["Java", "Redis"], "projects": [{"name": "电商系统"}]}}
        get_structured_llm = lambda agent_type, schema: (
            FakeLLM(score=0, fail=True)) # 强行0分模、拟维度失败
        t = time.perf_counter()
        state.update(await run_six_dimensions_node(state))
        print("\n② 全失败降级 → 加权分:", state["weighted_score"],
              "| issues:", state["dimension_scores"][0]["issues"])
        print("耗时:", round(time.perf_counter() - t, 2), "秒")
    # 执行测试2
    # asyncio.run(check_six_dimensions_fake())
    print('-' * 20)

    # ── 模块自测 3：离线测试 diagnose_issues + generate_summary(可单独运行)
    # 验证：Think 宏观分析
    # 验证：diagnose 优先级排序 + 失败降级
    # 验证：generate_summary 成功
    async def check_diagnose_and_summary_fake():
        print('\n➡️ 自测3[离线！]问题诊断 + 整体评价测试')
        from types import SimpleNamespace
        from backend.agents.resume.state import IssueItem

        class FakeThink:
            """假 Think 模型：返回固定宏观分析文本"""
            async def ainvoke(self, messages):
                await asyncio.sleep(0.05)
                return SimpleNamespace(
                    content="宏观分析：项目深度不足是核心短板",
                    text="宏观分析：项目深度不足是核心短板")

        class FakeStructured:
            """假结构化模型：返回预设对象或抛错模拟失败"""
            def __init__(self, obj=None, fail=False):
                self.obj, self.fail = obj, fail
            async def ainvoke(self, messages):
                await asyncio.sleep(0.05)
                if self.fail:
                    raise RuntimeError("模拟LLM失败")
                return self.obj

        global get_llm, get_structured_llm

        state = {
            "raw_text": "简历原文 " * 50,
            "structured": {"name": "张三", "target_position": "后端", "projects": [{"name": "电商"}]},
            "dimension_scores": [
                {"key": "project_depth", "dimension": "项目深度", "score": 60, "weight": 0.3, "issues": ["项目描述空洞"]},
                {"key": "tech_match", "dimension": "技术匹配度", "score": 70, "weight": 0.25, "issues": ["技能无层次"]},
            ],
            "weighted_score": 64.0,
        }

        # ① diagnose 成功：故意乱序优先级，验证排序 high→medium→low
        get_llm = lambda *a, **k: FakeThink()
        mixed = IssueList(items=[
            IssueItem(priority="low",    dimension="结构",     description="证书未列",   location="末尾",       suggestion="补充"),
            IssueItem(priority="high",   dimension="项目深度", description="无量化",     location="项目-电商",  suggestion="加QPS"),
            IssueItem(priority="medium", dimension="表达",     description="非动词开头", location="项目第1句",  suggestion="改写"),
        ])
        get_structured_llm = lambda agent_type, schema: FakeStructured(obj=mixed)
        r = await diagnose_issues_node(state)
        print("① 排序后优先级:", [i["priority"] for i in r["issues"]])

        # ② diagnose 降级：结构化 LLM 失败 → 用维度问题兜底
        get_structured_llm = lambda agent_type, schema: FakeStructured(fail=True)
        r2 = await diagnose_issues_node(state)
        print("② 降级后问题数:", len(r2["issues"]),
              "| 全 medium:", all(i["priority"] == "medium" for i in r2["issues"]))

        # ③ generate_summary 成功
        summary_obj = ResumeSummary(
            highlights=["有项目经历"],
            core_improvements=["补充量化"],
            overall_comment="中等水平",
            fit_assessment="较匹配"
        )
        get_structured_llm = lambda agent_type, schema: FakeStructured(obj=summary_obj)
        state["issues"] = r["issues"]
        r3 = await generate_summary_node(state)
        print("③ summary 字段:", list(r3["summary"].keys()),
              "| fit:", r3["summary"]["fit_assessment"])
    # 执行测试3
    # asyncio.run(check_diagnose_and_summary_fake())
    print('-'*20)

    # ── 模块自测 4：节点⑧：save_results  持久化结果
    import uuid
    rid = str(uuid.uuid4())
    async def main():
        print('\n➡️ 自测4[离线！需开启postgresql]节点8 持久化结果')
        async with AsyncSessionLocal() as s:
            # 先插一条 processing 记录
            sid = (await s.execute(
                text("SELECT id FROM users WHERE username='student01'"))
                   ).scalar()
            await s.execute(
                text(
            """INSERT INTO resume_reviews 
            (id,tenant_id,student_id,pdf_minio_path,status)
            VALUES 
            (:id,'tenant_default',:sid,'resumes/x.pdf','processing')"""
                     ),
                {"id": rid, "sid": sid}
            )
            await s.commit()

        # 运行节点
        await save_results_node({
            "review_id": rid,
            "student_id": str(sid),
            "pdf_local_path": "/tmp/none.pdf",
            "structured": {"name": "张三",
                           "skills_list": ["Java"]},
            "weighted_score": 78.5,
            "dimension_scores": [
                {"key": "project_depth", "dimension": "项目深度",
                 "score": 75, "weight": 0.3}
            ],
            "issues": [{"priority": "high", "description": "无量化"}],
            "summary": {"highlights": ["有项目"], "overall_comment": "中等"},
        })

        # 查回验证（用 JSONB 操作符）
        async with AsyncSessionLocal() as s:
            row = (
                await s.execute(
                    text(
                        """SELECT 
                        status, 
                        (scores->>'weighted_score') AS ws,
                        (structured_data->>'name') AS nm, 
                        jsonb_array_length(issues) AS cnt
                           FROM resume_reviews WHERE id=:id
                        """
                    ),
                    {"id": rid})
                ).mappings().fetchone()

            print("status:", row["status"],
                  "| 加权分:", row["ws"],
                  "| 姓名:", row["nm"],
                  "| 问题数:", row["cnt"])

    # 执行测试4
    asyncio.run(main())


