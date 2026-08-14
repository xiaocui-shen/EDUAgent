# backend/agents/exam/nodes.py

# ===========================================================
# 6-3 word试卷文件解析
# ===========================================================

# ── 标准库 ──
import asyncio   # 异步编程库，用来跑线程池
import json      # JSON 序列化（后续节点用）
import uuid      # 生成唯一 ID（后续节点用）
from typing import Any  # 类型注解工具

# ── 第三方库 ──
import httpx     # HTTP 客户端（后续节点用）
# SQLAlchemy：ORM 框架，用来查数据库
from sqlalchemy import text
# LangChain 消息类型
from langchain_core.messages import (
    HumanMessage,   # 用户消息
    SystemMessage,  # 系统消息
)
# LangGraph 的 interrupt：暂停图，等人工审核
from langgraph.types import interrupt

# ── 项目内部模块 ──
# State 和 Pydantic 模型
from backend.agents.exam.state import (
    ExamState,              # 图的数据总线
    SubjectiveReviewResult, # 简答题批改结果模型
    WeakPointsReport,       # 薄弱点报告模型
)
# LLM 提示词模板
from backend.agents.exam.prompts import (
    SYSTEM_PROMPT,              # 系统提示词
    SUBJECTIVE_REVIEW_PROMPT,   # 简答题批改提示
    SUBJECTIVE_THINK_PROMPT,    # 简答题思考链提示
    CODE_REVIEW_PROMPT,         # 代码题批改提示
    WEAK_POINTS_ANALYSIS_PROMPT,# 薄弱点分析提示
)
# LLM 工厂：创建 LLM 实例
from backend.core.llm_factory import (
    get_llm,             # 获取普通 LLM
    get_structured_llm,  # 获取结构化输出 LLM
)
from backend.core.logger import get_logger  # 日志
# 异步数据库会话（异步上下文管理器）
from backend.dependencies import AsyncSessionLocal

# 创建本模块的日志记录器
logger = get_logger(__name__)


# ──────────────────────────────────────────────
# 工具函数（被节点调用的辅助小工具）
# ──────────────────────────────────────────────

def _get_message_content(msg) -> str:
    """统一获取消息文本内容
    （兼容 text 属性和 content 属性）
    """
    # 优先尝试 .text 属性
    # hasattr 检查有没有这个属性
    # not callable 确保它不是方法
    if hasattr(msg, "text") and \
       not callable(getattr(msg, "text", None)):
        return msg.text  # 有 .text → 直接返回
    # 再试 .content 属性
    if isinstance(msg.content, str):
        return msg.content  # 是字符串 → 返回
    # 都不是 → 强转兜底
    return str(msg.content)


def _chinese_to_int(s: str) -> int:
    """中文数字转整数
    转换失败时直接 int()，仍失败时返回 1
    """
    # 中文 → 阿拉伯数字 映射表
    cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    try:
        return int(s) # 先尝试直接 int() 转
    except ValueError:
        # 失败了（如 "三"）→ 查字典
        # 字典也没有 → 返回 1 兜底
        return cn_map.get(s, 1)


# ──────────────────────────────────────────────
# 节点1的核心功能函数：parse_word — 解析学员作答 Word
# ──────────────────────────────────────────────
def _sync_parse_word(word_path: str) -> list:
    """同步解析 Word 文件
    （在线程池中运行，避免阻塞事件循环）

    返回 list[dict]，每个 dict 包含：
        question_no:    题号（int）
        header_text:    原始题目行文本
        student_answer: 学员作答文本
        is_code:   True 表示代码题
    """
    # 延迟导入：只在调用时才加载
    from docx import Document
    import re  # 正则表达式模块

    # 打开 Word（.docx 本质是 zip 包）
    doc = Document(word_path)
    # ── 5 个状态机变量 ──
    parsed_questions = []     # 最终结果列表
    current_question = None   # 当前正在解析的题目
    current_answer_lines = [] # 当前题答案行暂存
    in_code_block = False     # 代码块开关（默认关）
    code_buffer = []          # 代码块内容暂存

    # 逐段遍历 Word 文档
    for para in doc.paragraphs:
        # 取段落文本并去首尾空格
        para_text = para.text.strip()

        # ── 分支①：空行处理 ──
        if not para_text:
            # 代码块内的空行有意义（函数分隔）
            if in_code_block:
                code_buffer.append("")
            # 非代码块的空行 → 直接跳过
            continue

        # ── 判断是否是题目开头行 ──
        # 支持：第1题、第一题、Q.3、题目5
        is_question_header = re.match(
            r"^(第?\s*[一二三四五六七八九十\d]+"
            r"\s*[题、。.]|Q\.?\s*\d+|题目\s*\d+)",
            para_text,
            re.IGNORECASE,  # 忽略大小写
        )

        # ── 分支②：题目开头行 ──
        if is_question_header:
            # 2.1. 先保存上一题（如果有）
            """读到新题目行时
                current_question 里还装着上一题的数据
                但 student_answer 还是空的（初始化时设的 ""）。
                这里把暂存区里攒的答案行拼成字符串，填进去。"""
            if current_question is not None:
                # 非代码题 → 拼接答案行
                if not current_question.get("is_code"):
                    # 代码块未关闭 → 用 code_buffer
                    # 否则 → 用 answer_lines
                    if in_code_block:
                        ans = "\n".join(code_buffer)
                    else:
                        ans = "\n".join(current_answer_lines)
                    current_question["student_answer"] \
                        = ans.strip()
                # 把上一题加入结果列表
                parsed_questions.append(current_question)

            # 2.2. 处理当前题
            # 从题目行提取题号数字
            match = re.search(
                r"[一二三四五六七八九十\d]+", para_text)
            if match: # 中文转 int
                q_no = _chinese_to_int(match.group())
            else: # 没匹配到就按序号递增
                q_no = len(parsed_questions) + 1

            # 初始化新题目对象
            current_question = {
                "question_no": q_no,      # 题号
                "header_text": para_text,  # 原题文本
                "student_answer": ""       # 答案（待填）
            }
            # 重置暂存区，准备读新题
            current_answer_lines = []
            code_buffer = []
            in_code_block = False  # 关代码开关

        # ── 分支③：代码块开关 ──
        elif para_text.startswith("```"):
            # 翻转开关：开→关 或 关→开
            in_code_block = not in_code_block
            # 当代码开关关闭时 说明代码区结束了，所以保存
            if not in_code_block and current_question:
                current_question["student_answer"] = \
                    "\n".join(code_buffer).strip()
                current_question["is_code"] = True

        # ── 分支④：代码块内部 ──
        # 分支③ 只处理 ```这一行本身
        # 分支④ 处理的是代码块里面的内容行
        elif in_code_block:
            # 保留原始缩进！用 para.text
            # 不是 para_text（那个被 strip 了）
            code_buffer.append(para.text)

        # ── 分支⑤：处理普通答案文本 ──
        elif current_question is not None:
            # 跳过模板提示行（不是学员答案）
            skip = ["作答区", "请在此处"]
            if any(para_text.startswith(p) for p in skip):
                pass  # 模板行 → 跳过
            else:
                # 答案前缀列表
                pfx = ["答：", "答:", "Answer:"]
                extracted = None
                # 检查是否以答案前缀开头
                for prefix in pfx: # 挨个试"答："、"答:"、"Answer:"
                    if para_text.startswith(prefix):
                        # 取冒号后面的内容
                        rest = para_text[len(prefix):].strip()
                        if rest: # 有内容 → 存下来
                            extracted = rest
                        break  # 有前缀就跳出循环
                # 冒号后面有内容 → 存内容
                if extracted is not None:
                    current_answer_lines.append(extracted)
                # 不是「答：」开头的行 → 整行存入
                elif not any(
                    para_text.startswith(p) for p in pfx):
                    # 不是前缀行 → 整行存入
                    current_answer_lines.append(para_text)
                # 如果是「答：」开头但后面没内容 → 什么都不存

    # ── 循环结束：保存最后一题 ──
    # 最后一题后面没有「下一题」触发保存
    # 所以必须手动处理
    if current_question is not None:
        # 非代码题 → 拼接答案
        if not current_question.get("is_code"):
            if in_code_block: # 非代码题为啥还要判断是代码块呢？
                # 特殊情况：学员写了代码但忘了关代码块
                ans = "\n".join(code_buffer)
            else:
                ans = "\n".join(current_answer_lines)
            # 答案存起来
            current_question["student_answer"] = ans.strip()
        # 把最后一题加入结果列表
        parsed_questions.append(current_question)

    return parsed_questions  # 返回所有题目列表


# ──────────────────────────────────────────────────────────────
# 节点1：parse_word_node — 解析 Word 试卷
# ──────────────────────────────────────────────────────────────
async def parse_word_node(state: ExamState) -> dict:
    """LangGraph 节点入口：解析 Word 试卷

    python-docx 的文件 I/O 和 XML 解析
    都是同步阻塞操作，不能直接 await。
    用 run_in_executor 放入线程池，
    事件循环继续处理其他请求。
    """
    # 从 state 取 Word 文件路径
    word_path = state["word_file_path"]

    try:
        # 获取当前事件循环
        loop = asyncio.get_running_loop()
        # 把同步函数丢进线程池执行
        # None = 用默认线程池
        # _sync_parse_word = 要执行的函数
        # word_path = 传给函数的参数
        parsed_questions = await loop.run_in_executor(
            None, _sync_parse_word, word_path)

        # 记录成功日志
        logger.info(
            "parse_word.done",
            file=word_path,
            questions_found=len(parsed_questions),
        )

        # 返回结果 → LangGraph 写入 state
        return {"parsed_questions": parsed_questions}

    except Exception as e:
        # 解析失败 → 记录错误日志
        logger.error(
            "parse_word.failed",
            error=str(e),
            file=word_path)
        # 优雅降级：返回空列表
        # 后续节点从 DB 补全题目信息
        # 学员答案为空 → 教师人工补批
        return {"parsed_questions": []}


# ──────────────────────────────────────────────────────────────
# 节点2：load_questions_meta — 加载试卷题目元数据
# ──────────────────────────────────────────────────────────────
async def load_questions_meta_node(state: ExamState) -> dict:
    """
    从数据库加载试卷的完整题目元数据（含标准答案、得分点、知识点标签），
    与解析出的学员答案合并，覆盖写入 parsed_questions。
    """
    exam_id = state["exam_id"] # 考试ID
    parsed = state["parsed_questions"] # parse_word_node的输出

    async with AsyncSessionLocal() as session:
        # ── ① 加载题目列表 ─────────────────────────────
        result = await session.execute(
            text("""
                SELECT id, question_no, question_type, content,
                       correct_answer, score, knowledge_tag
                FROM questions
                WHERE exam_id = :exam_id
                ORDER BY question_no
            """),
            {"exam_id": exam_id},
        )
        questions = result.mappings().all()

        # ── ② 加载得分点（仅简答题有）──────────────────────────
        question_ids = [str(q["id"]) for q in questions]
        scoring_points_rows = [] # 得分点列表
        if question_ids:
            # 生成占位符名字列表 [":qid_0", ":qid_1", ":qid_2"]
            param_names = [
                f":qid_{i}" for i in range(len(question_ids))
            ]
            # 生成「占位符 → 真实值」的字典
            # {"qid_0": "uuid-aaa", "qid_1": "uuid-bbb", ...}
            qid_params  = {
                f"qid_{i}":
                    qid for i, qid in enumerate(question_ids)
            }
            # 动态构造IN子句 把占位符拼进SQL
            # IN ({", ".join(param_names)})
            #   ➡️ IN (":qid_0, :qid_1, :qid_2")
            sp_result = await session.execute(
                text(f"""
                SELECT 
                    id, question_id, point_desc, point_score
                FROM scoring_points
                WHERE question_id 
                    IN ({", ".join(param_names)})
                    AND is_active = TRUE
                ORDER BY question_id, id
                """),
                qid_params,
            )
            # 得分点列表 [{}, ...]
            scoring_points_rows = sp_result.mappings().all()

    # ── ③ 按 question_id 聚合得分点 ──────────────────────
    sp_by_question: dict[str, list] = {}
    for sp in scoring_points_rows:
        qid = str(sp["question_id"]) # 题目ID
        # sp_by_question中不存在qid这个key就创建，设value为[]
        #   如果存在则直接向value（列表）中append得分点字典
        sp_by_question.setdefault(qid, []).append({
            "id": str(sp["id"]), # 得分点ID
            "desc": sp["point_desc"], # 得分点描述
            "score": sp["point_score"], # 得分点分数
        })

    # ── ④ 以 DB 题目为主，合并解析节点的结果 ───────────────────
    # parsed 是 parse_word_node的输出，学员答卷word的解析
    parsed_by_no = {p["question_no"]: p for p in parsed}
    merged_questions = []
    # 遍历数据库获取试卷的每个题目
    for q in questions:
        q_no = q["question_no"]
        # 组装一道题的完整信息（DB+解析+得分点）
        merged_questions.append({
            "question_id": str(q["id"]), # 题目UUID（数据库主键）
            "question_no": q_no, # 题目编号（第几题，如1、2、3）
            "question_type": q["question_type"], # 题型
            "content": q["content"], # 题目正文内容
            # 从Word解析结果中按题号取学员答案 取不到就给空串
            "student_answer": parsed_by_no.get(
                q_no, {}).get("student_answer", ""),
            # 标准答案（None转空串）
            "correct_answer": q["correct_answer"] or "",
            # 按题目ID取该题的得分点
            "scoring_points": sp_by_question.get(
                str(q["id"]), []), # 没有得分点就给空列表
            "full_score": q["score"], # 该题满分分值
            # 知识点标签（None转空串）
            "knowledge_tag": q["knowledge_tag"] or "",
        })

    logger.info(
        "load_questions_meta.done",
        exam_id=exam_id,
        total_questions=len(merged_questions),
    )

    return {"parsed_questions": merged_questions}

# ──────────────────────────────────────────────────────────────
# 6-5 客观题规则引擎
# ──────────────────────────────────────────────────────────────
# 客观题答案标准化
def _normalize_answer(answer: str) -> str:
    """
    标准化答案字符串，消除大小写/空格/标点差异，使多选题选项顺序无关。
    处理步骤：
        1. 统一大写（A/a → A）
        2. 去除所有空格、中文逗号、英文逗号
           "A, B, C" → "ABC"，"A，B，C" → "ABC"
        3. 字符排序（多选题 "BA" 和 "AB" 视为等价）
           sorted("ABC") → ['A','B','C'] → "ABC"
    """
    cleaned = answer.upper().replace(
        " ", "").replace(
        "，", "").replace(
        ",", "")
    return "".join(sorted(cleaned))

# ──────────────────────────────────────────────────────────────
# 第一轨：规则引擎（客观题）
# ──────────────────────────────────────────────────────────────
async def _run_objective_track(
        questions: list[dict]) -> list[dict]:
    """客观题规则批改。
    虽然声明为 async，但内部没有 await。
    保持async统一接口方便在asyncio.gather中与其他两轨并行。
    """
    results = []
    for q in questions:
        # 标准化学员答案
        student_ans = _normalize_answer(q["student_answer"])
        # 标准化标准答案
        correct_ans = _normalize_answer(q["correct_answer"])
        # 判断是否正确
        is_correct  = (student_ans == correct_ans)

        results.append({
            "question_id": q["question_id"], # 题目UUID（数据库主键）
            # 题目编号（第几题，如1、2、3）
            "question_no": q["question_no"],
            "question_type": q["question_type"], # 题型
            # 知识点标签
            "knowledge_tag": q.get("knowledge_tag", ""),
            "content": q.get("content", ""), # 题目正文内容
            "student_answer": q["student_answer"], # 学员答案
            "correct_answer": q["correct_answer"], # 标准答案
            "is_correct": is_correct, # 是否正确
            "score": q["full_score"] if is_correct else 0, # 得分
            "full_score": q["full_score"], # 该题满分分值
            "needs_review": False, # 客观题不需要教师复核
            "ai_feedback": "正确" if is_correct
                # 如果错误就填入正确答案
                else f"正确答案：{q['correct_answer']}",
        })

    return results


# ──────────────────────────────────────────────────────────────
# 6-6 第二轨：LLM 语义评分（简答题）
# ──────────────────────────────────────────────────────────────
async def _review_one_subjective(q: dict) -> dict:
    """批改单道简答题，两步流程：先 Think Tool 推理，再结构化评分。"""
    # 把得分点列表拼成人类可读文本，如：
    #   1. [4分] 答出IOC概念
    #   2. [3分] 举出应用场景
    # 用换行拼接每个得分点
    scoring_points_text = "\n".join([
        # # 遍历该题所有得分点，拼接 序号+分值+描述
        f"  {i + 1}. [{sp['score']}分] {sp['desc']}"
        for i, sp in enumerate(q["scoring_points"])
    ]) or "  （无预设得分点，请综合评分）" # 列表为空时的兜底文案

    # 拼接学员答案文本 学员答案为空时给提示 避免LLM收到空字符串
    student_answer_text = q["student_answer"] or "（学员未作答）"

    # ── 第一步：Think Tool 推理分析 ───────────────────────────
    # 先让LLM自由思考（不要求格式），分析学员答得对不对
    reasoning_trace = "" # 推理结论文本，初始为空
    try:
        # 用思考链提示词模板，填入题目、得分点、学员答案
        think_prompt = SUBJECTIVE_THINK_PROMPT.format(
            question_content=q["content"], # 题目正文
            scoring_points=scoring_points_text, # 得分点文本
            student_answer=student_answer_text, # 学员答案
        )
        # 获取LLM实例，temperature=0 保证评分稳定不随机
        think_llm  = get_llm( # 简答题专用模型
            "exam_subjective", temperature=0)
        # 异步调用LLM，只传一条用户消息（思考提示）
        think_resp = await think_llm.ainvoke([
            # 思考提示作为用户消息
            HumanMessage(content=think_prompt)])
        # 提取LLM返回的文本内容，去掉首尾空白
        reasoning_trace = _get_message_content(think_resp).strip()
        # 记录调试日志：思考完成
        logger.debug("subjective_think.done",
                     question_no=q.get("question_no")) # 哪道题
    except Exception as e: # 报错就跳过思考
        # 推理失败不影响主评分，降级为直接评分
        logger.warning("subjective_think.failed",
                       error=str(e)) # 记录失败原因

    # ── 第二步：结构化评分（附带推理结论）────────────────────
    # 如果第一步有推理结论，就拼到评分提示词后面当参考
    think_context = (
        f"\n\n【批改前分析】\n{reasoning_trace}" # 有推理→拼上
        if reasoning_trace else "" # 没推理→空串，不影响评分
    )
    # 用评分提示词模板，填入题目、得分点、满分、学员答案+思考结论
    review_prompt = SUBJECTIVE_REVIEW_PROMPT.format(
        question_content=q["content"], # 题目正文
        scoring_points=scoring_points_text, # 得分点文本
        full_score=q["full_score"], # 该题满分
        student_answer=student_answer_text, # 学员答案
    ) + think_context # 末尾追加思考结论（可能为空）

    # 获取结构化LLM：输出自动解析为SubjectiveReviewResult对象
    structured_llm = get_structured_llm( # 绑定输出结构
        "exam_subjective", SubjectiveReviewResult)
    # 异步调用：最多重试2次（DeepSeek偶发返回无效JSON导致None）
    result = None
    for attempt in range(2): # 最多尝试 2 次
        result = await structured_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT), # 系统提示词
            HumanMessage(content=review_prompt), # 评分提示+思考结论
        ])
        if result is not None: # 成功拿到结构化结果
            break
        # 返回None：DeepSeek生成的tool_call参数JSON无效
        logger.warning("subjective_review.none_retry",
                       attempt=attempt,
                       question_no=q.get("question_no"))
    if result is None: # 2次都失败→抛异常，由上层gather降级处理
        raise ValueError(
            f"structured output returned None after 2 attempts "
            f"(question_no={q.get('question_no')})")

    # 组装单题批改结果字典，返回给上层汇总
    return {
        "question_id":    q["question_id"], # 题目UUID
        "question_no":    q["question_no"], # 题号
        "question_type":  "short_answer", # 固定标记为简答题
        "knowledge_tag":  q.get("knowledge_tag", ""), # 知识点标签
        "content":        q.get("content", ""), # 题目正文
        "student_answer": q["student_answer"], # 学员原始答案
        "score":          result.total_score, # LLM给出的实际得分
        "full_score":     result.full_score, # 该题满分
        # 置信度低于0.7→标记需教师人工复核
        "needs_review":   result.confidence < 0.7, # True=需复核
        "confidence":     result.confidence, # LLM自评把握度[0,1]
        "ai_feedback":    result.overall_comment, # LLM整体评语
        "point_results":  [
            p.model_dump()
            for p in result.point_results
        ], # 每个得分点明细转dict
    }

async def _run_subjective_track(
        questions: list[dict]) -> list[dict]:
    """简答题批改，每 3 题一组并行处理。
    并行策略：
        把 N 道简答题切成 ⌈N/3⌉ 个组，组内 asyncio.gather 并行，
        组间顺序执行。目的是避免同时发起几十个 LLM 请求导致 API 限速。
    """
    if not questions: # 没有简答题→直接返回空列表
        return []

    GROUP_SIZE = 3 # 每组最多3道题并行（防止同时请求太多LLM被限速）
    # 切片分组：[0:3], [3:6], [6:9]... 每GROUP_SIZE个切一刀
    # 假设 questions 有7道题，groups 长这样：
    # [
    #   [题1, 题2, 题3],   ← 第1组
    #   [题4, 题5, 题6],   ← 第2组
    #   [题7],            ← 第3组（不足3个也单独一组）
    # ]
    groups = [questions[i: i+GROUP_SIZE]
              # 步长=GROUP_SIZE
              for i in range(0, len(questions), GROUP_SIZE)]

    all_results = [] # 存放所有题的批改结果
    for group in groups: # 逐组处理（组间顺序执行）
        # 组内并行：同时批改这一组的每道题
        # 假设当前 group = [题1, 题2, 题3]，展开后等价于：
        # asyncio.gather(
        #     _review_one_subjective(题1),  ← 协程1
        #     _review_one_subjective(题2),  ← 协程2
        #     _review_one_subjective(题3),  ← 协程3
        # )
        # 三个协程同时跑，谁先完成谁先返回，不用排队等
        group_results = await asyncio.gather(
            # 列表推导生成协程列表，* 解包成 gather 的多个参数
            *[_review_one_subjective(q) for q in group],
            return_exceptions=True, # 某题报错 异常作为结果返回
        )
        # 逐题检查结果：正常→收录，异常→降级处理
        # zip(group, group_results) 题目和结果一一对应
        for q, result in zip(group, group_results):
            # 结果是异常对象→批改失败
            if isinstance(result, Exception):
                logger.warning(
                    "subjective_track.question_failed",
                    question_id=q["question_id"], # 哪道题失败
                    error=str(result), # 失败原因
                )
                # 构造一个"失败占位"结果，保证题目不丢失
                all_results.append({
                    "question_id": q["question_id"], # 题目UUID
                    "question_no": q["question_no"], # 题号
                    "question_type": "short_answer", # 简答题
                    # 知识点标签
                    "knowledge_tag": q.get("knowledge_tag", ""),
                    "content": q.get("content", ""), # 题目正文
                    # 学员答案
                    "student_answer": q["student_answer"],
                    "score": 0, # 给0分（不是真的0分，是没批出来）
                    "full_score": q["full_score"], # 该题的满分
                    # 单题失败降级：标记 needs_review=True，不阻断整批
                    "needs_review": True, # 强制标记教师复核
                    "confidence": 0.0, # 置信度0（完全没把握）
                    "ai_feedback": "AI评分失败，已标记需教师人工批改",
                    "point_results": [], # 无得分点明细
                })
            else: # 正常结果→直接收录
                all_results.append(result)

    return all_results # 返回所有简答题的批改结果列表


# ──────────────────────────────────────────────────────────────
# 6-7 第三轨：LLM（代码题）
# ──────────────────────────────────────────────────────────────
async def _run_code_track(questions: list[dict]) -> list[dict]:
    """代码题批改入口：顺序逐题批改（代码题一般数量少，不必并行）。
    参数 questions：所有「代码题」的合并题目字典列表
        来自 6.4 的合并结果，已按题型筛选
        这里只会收到 question_type == "code" 的题。
    返回：每道代码题的批改结果字典组成的列表。
    """
    if not questions: # 没有代码题（空列表）
        return [] # 直接返回空，省去后续循环
    results = [] # 收集每道题的批改结果
    for q in questions: # 逐道代码题处理
        # 调用单题批改，await 等它出结果
        results.append(await _review_one_code(q))
    return results # 返回全部代码题的批改结果


async def _review_one_code(q: dict) -> dict:
    """批改「单道代码题」：交给大模型综合评估功能正确性 + 代码质量。
    参数 q：一道代码题的「合并题目字典」
    （就是 6.4 load_questions_meta_node 产出的那种），
        本函数会读取它的下列键：
        q["question_id"] 题目 ID（str），原样写进批改结果，方便回填数据库
        q["question_no"] 题号（int）
        q["content"] 题目内容（str），喂给 LLM 当评分依据
        q["student_answer"] 学员提交的代码（str）
        q["correct_answer"] 标准答案（str）：
            这道题的「满分参考实现」，给 LLM 当对照标杆
        q["full_score"] 该题满分（int）
        q["knowledge_tag"] 知识点标签（str，可能缺省，用 .get兜底）
    返回：一道代码题的批改结果字典
        含score/confidence/needs_review/quality_feedback等
    """
    student_code = q["student_answer"] # 取出学员代码
    # 取出参考答案（缺省给空串）
    reference_solution = q.get("correct_answer", "") or ""

    # 交给大模型评分：返回（逐条评语, 得分, 把握度）三元组
    feedback, score, confidence = await _llm_code_review(
        question_content=q["content"], # 题目内容
        student_code=student_code, # 学员代码
        full_score=q["full_score"], # 该题满分（评分上限）
        reference_solution=reference_solution,  # 参考答案用于对照
    )
    # ── 组装批改结果字典返回 ───────────────────────────────────
    return {
        "question_id": q["question_id"], # 题目数据库中的ID
        "question_no": q["question_no"], # 题号
        "question_type": "code", # 题型固定 code
        "knowledge_tag": q.get("knowledge_tag", ""), # 知识点
        "content": q.get("content", ""), # 题目内容（缺省空串）
        "student_answer": student_code, # 学员代码
        "score": score, # 本题最终得分（0~full_score）
        "full_score": q["full_score"], # 本题满分
        "confidence": confidence, # LLM 评分把握度（0~1）
        # 把握度低于 0.7 → 标记教师复核
        "needs_review": confidence < 0.7,
        # LLM 逐条评语（list[str]，下游会用到）
        "quality_feedback": feedback,
        "ai_feedback": "\n".join(feedback), # 评语拼成字符串 方便展示
    }

# 大模型综合评估代码：功能正确性 + 代码质量
async def _llm_code_review(
    question_content:   str,        # 题目内容
    student_code:       str,        # 学员代码
    full_score:         int,        # 该题满分（评分上限）
    reference_solution: str = "",   # 参考实现（默认空串）
) -> tuple[list[str], int, float]:
    """大模型综合评估代码：功能正确性 + 代码质量。
    返回一个三元组 (feedback, score, confidence)：
        feedback：list[str] 逐条评语（功能正确性 + 四个质量维度）
        score：int 得分，范围 0 ~ full_score
        confidence：float LLM自报的评分把握度，范围0~1 越不确定越低
    """
    # 用模板拼出完整 Prompt（把题目/参考答案/学员代码/满分 填进去）
    prompt = CODE_REVIEW_PROMPT.format(
        question=question_content, # {question}
        reference_solution= # {reference_solution}，没有就填兜底语
            reference_solution or "（无参考实现，仅按代码本身评估）",
        # {code} 没提交就填兜底语
        code=student_code or "（未提交代码）",
        full_score=full_score, # {full_score}：满分制
    )
    # 从 llm_factory 取「代码评估」用的模型实例
    llm = get_llm("exam_code")
    response = await llm.ainvoke([ # 异步调用大模型
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    # 取出回复纯文本
    # 并去掉模型可能多带的 ```json 代码围栏，方便 json.loads
    raw = _get_message_content(response).strip().replace(
        "```json", "").replace(
        "```", "").strip()
    try:
        data = json.loads(raw) # 把回复解析成字典
        # 取逐条评语列表（缺省空列表）
        feedback = data.get("feedback", [])
        # 取分数，满分封顶，防止大模型给负分或过高分数
        score = min(int(data.get("score", 0)), full_score)
        # LLM把握度（缺省 0）
        confidence = float(data.get("confidence", 0))
    except Exception: # 模型没返回合法 JSON
        # JSON解析失败时的降级：不崩，给0分+把握度 0（→ 必复核）+提示人工
        feedback = ["代码评估结果解析失败，请教师人工复核"]
        score = 0
        confidence = 0.0
    # 返回（评语列表, 得分, 把握度）
    return feedback, score, confidence


# ──────────────────────────────────────────────────────────────
# 6-8 节点: 三轨并行 汇总 薄弱点分析
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 节点3：run_three_tracks — 三轨并行批改
# ──────────────────────────────────────────────────────────────
async def run_three_tracks_node(state: ExamState) -> dict:
    """三轨并行批改：
        第一轨：规则引擎（客观题：单选/多选/判断）
        第二轨：LLM 语义评分（简答题，按3题一组并行）
        第三轨：LLM 代码质量评估（代码题，顺序执行）

    asyncio.gather(return_exceptions=True)：
        某一轨抛异常不会中断其他轨，异常作为返回值处理。
        失败的轨结果置为空列表，其余轨正常写入 State。
    """
    # 从 state 取出节点2合并好的全部题目
    questions = state["parsed_questions"]

    # ── 按题型分成三堆 ──────────────────────────────
    # 客观题：单选 / 多选 / 判断
    objective_qs  = [
        q for q in questions
        if q["question_type"] in
        ("single_choice", "multi_choice", "judge")
    ]
    # 简答题
    subjective_qs = [
        q for q in questions
        if q["question_type"] == "short_answer"
    ]
    # 代码题
    code_qs = [
        q for q in questions
        if q["question_type"] == "code"
    ]

    # 记录日志：三轨各分到几道题
    logger.info(
        "three_tracks.start",
        objective=len(objective_qs),   # 客观题数
        subjective=len(subjective_qs), # 简答题数
        code=len(code_qs),             # 代码题数
    )

    # ── 三轨并行启动 ────────────────────────────────
    # gather 同时跑三个协程，谁先完成谁先返回
    # return_exceptions=True：某轨炸了不炸其他轨
    raw = await asyncio.gather(
        _run_objective_track(objective_qs),   # 第一轨
        _run_subjective_track(subjective_qs), # 第二轨
        _run_code_track(code_qs),             # 第三轨
        return_exceptions=True, # 异常当返回值
    )

    # 逐轨检查结果
    # 第一轨（客观题）：没炸就用结果，炸了给空[]
    objective_results = (
        raw[0]  # gather返回的第1个结果
        if not isinstance(raw[0], Exception)  # 不是异常
        else []  # 是异常→给空列表，不拖后腿
    )
    # 第二轨（简答题）
    subjective_results = (
        raw[1]  # gather返回的第2个结果
        if not isinstance(raw[1], Exception)  # 不是异常
        else []  # 是异常→给空列表
    )
    # 第三轨（代码题）
    code_results = (
        raw[2]  # gather返回的第3个结果
        if not isinstance(raw[2], Exception)  # 不是异常
        else []  # 是异常→给空列表
    )

    # 记录哪一轨失败了（不影响正常流程）
    for name, exc in zip(  # 把名字和结果配对
        ["objective", "subjective", "code"],  # 三轨名
        raw  # 三轨返回值（可能是异常对象）
    ):
        # 如果这个返回值是异常对象
        if isinstance(exc, Exception):
            # 记个错误日志，但不抛出不中断
            logger.error(
                f"three_tracks.{name}_failed",  # 哪轨
                error=str(exc),  # 异常信息
            )

    # 记录日志：三轨各批改完几道
    logger.info(
        "three_tracks.done",  # 日志事件名
        objective_done=len(objective_results),  # 客观
        subjective_done=len(subjective_results),  # 简答
        code_done=len(code_results),  # 代码
    )

    # 三轨结果分别写入 state，供节点4汇总
    return {
        "objective_results": objective_results,  # 客观
        "subjective_results": subjective_results,  # 简答
        "code_results": code_results,  # 代码
    }


# ──────────────────────────────────────────────────────────────
# 节点4：aggregate_results — 汇总预批改结果
# ──────────────────────────────────────────────────────────────
async def aggregate_results_node(
        state: ExamState
    ) -> dict:
    """合并三轨结果，按题号排序，计算总分和需复核题数
    pre_review_summary 是后续HitL展示给教师的核心数据结构
    """
    # 把三轨结果拼成一个大列表
    all_results = (
        state.get("objective_results", [])  # 客观题
        + state.get("subjective_results", [])  # 简答题
        + state.get("code_results", [])  # 代码题
    )
    # 按题号升序排列（第1题、第2题...）
    all_results.sort(key=lambda x: x.get("question_no", 0))

    # 学员总得分（每题 score 加一起）
    total_score = sum(
        r.get("score", 0) for r in all_results
    )
    # 试卷满分（每题 full_score 加一起）
    full_score = sum(
        r.get("full_score", 0) for r in all_results
    )
    # 得分率（保留四位小数，满分0时给0）
    score_rate = (
        round(total_score / full_score, 4)
        if full_score > 0 else 0.0
    )
    # 统计需要教师复核的题数
    needs_review_count = sum(
        1 for r in all_results
        if r.get("needs_review", False)
    )

    # 组装汇总字典（教师看到的就是这个）
    summary = {
        "total_score": total_score,  # 总得分
        "full_score": full_score,  # 满分
        "score_rate": score_rate,  # 得分率
        # 需复核题数
        "needs_review_count": needs_review_count,
        "by_question": all_results,  # 每题明细
    }

    # 记录日志：汇总完毕
    logger.info(
        "aggregate_results.done",  # 事件名
        total_score=total_score,  # 总得分
        full_score=full_score,  # 满分
        score_rate=score_rate,  # 得分率
        needs_review=needs_review_count,  # 复核数
    )

    # 写入 state，后续节点5和教师审核都用
    return {"pre_review_summary": summary}


# ──────────────────────────────────────────────────────────────
# 节点5：analyze_weak_points — 知识薄弱点分析
# ──────────────────────────────────────────────────────────────
async def analyze_weak_points_node(
        state: ExamState
    ) -> dict:
    """分析学员知识薄弱点。
    路径1：有 knowledge_tag 的失分题
        → 按标签直接聚合
    路径2：全量失分题交 LLM
        → 推断知识点 + 生成 suggestion
    两路合并，去重，按 wrong_count 降序。
    """
    # 从节点4的汇总中取每题明细
    all_results = state.get(
        "pre_review_summary", {}
    ).get("by_question", [])

    # 收集失分题（得分 < 满分）
    wrong_questions = [
        r for r in all_results  # 遍历每题
        if r.get("score", 0)  # 实际得分
        < r.get("full_score", 1)  # 该题满分
    ]

    # 全对了→不用分析，直接返回
    if not wrong_questions:  # 失分题为空
        # 记个日志：这学员全对
        logger.info(
            "analyze_weak_points.no_wrong",  # 事件
            submission_id=state["submission_id"],  # 哪份
        )
        # 返回空薄弱点 + 表扬语
        return {
            "weak_points": [],  # 无薄弱点
            "weak_points_summary":  # 总体评价
                "本次试卷全部答对，表现优秀！",
        }

    # ── 路径1：有标签 → 直接聚合 ───────────────
    # 有知识点标签的失分题
    tagged = [
        r for r in wrong_questions  # 遍历失分题
        if r.get("knowledge_tag")  # 有标签的
    ]
    # 没有知识点标签的失分题
    untagged = [
        r for r in wrong_questions  # 遍历失分题
        if not r.get("knowledge_tag")  # 没标签的
    ]

    # 按标签聚合：{"知识点标签": {统计信息}}
    tagged_weak: dict[str, dict] = {}
    for r in tagged:  # 遍历有标签的失分题
        tag = r["knowledge_tag"]  # 取标签
        if tag not in tagged_weak:  # 第一次见
            tagged_weak[tag] = {
                "tag": tag,  # 知识点名
                "wrong_count": 0,  # 错题数
                "total_count": 0,  # 总题数
                "question_nos": [],  # 题号列表
                "suggestion": "",  # 复习建议
            }
        # 累加统计
        tagged_weak[tag]["wrong_count"] += 1
        tagged_weak[tag]["total_count"] += 1
        tagged_weak[tag]["question_nos"].append(
            r["question_no"]
        )
        """tagged_weak 就是一个按知识点标签分组的统计字典
            key 是标签名，value 是这个标签的错题统计
        假设学员错了3道题：
            第2题（标签"Spring IOC"）
            第5题（标签"Spring IOC"）
            第8题（标签"多线程"）
        那循环跑完后 tagged_weak 长这样：
        {
            "Spring IOC": {
                "tag": "Spring IOC",
                "wrong_count": 2, # 错了2道
                "total_count": 2, # 一共考了2道（这里只统计失分的）
                "question_nos": [2, 5], # 第2题、第5题
                "suggestion": "", # 先空着，后面LLM补
            },
            "多线程": {
                "tag": "多线程",
                "wrong_count": 1, # 错了1道
                "total_count": 1,
                "question_nos": [8], # 第8题
                "suggestion": "", # 先空着，后面LLM补
            },
        }
        """

    # ── 路径2：全量失分题交 LLM ───────────────
    llm_weak_points: list[dict] = []  # LLM结果
    llm_summary = ""  # LLM总体评价

    # 有标签+无标签的失分题都给LLM看
    questions_for_llm = tagged + untagged  # 合并
    if questions_for_llm:  # 有失分题才调LLM
        # 拼出错题描述文本（喂给LLM）
        wrong_desc = "\n".join([  # 多行拼一段
            f"第{r['question_no']}题"  # 题号
            f"（{r['question_type']}，"  # 题型
            f"{r['score']}/{r['full_score']}分）："  # 得分
            f"\n  题目："  # 题目内容前缀
            f"{r.get('content', '')[:200]}"  # 截200字
            f"\n  AI反馈："  # 反馈前缀
            f"{r.get('ai_feedback', '')[:150]}"  # 截150
            for r in questions_for_llm  # 遍历失分题
        ])

        # 用模板填入错题描述
        prompt = (
            WEAK_POINTS_ANALYSIS_PROMPT.format(
                wrong_questions=wrong_desc
            )
        )
        # 结构化LLM：输出自动解析为 WeakPointsReport 对象
        structured_llm = get_structured_llm(
            "exam_subjective", WeakPointsReport
        )

        # 先初始化变量，防止 except 分支未赋值导致 NameError
        llm_weak_points = []
        try:
            # 异步调用LLM，拿结构化报告
            report: WeakPointsReport = (
                await structured_llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
            )
            # None检查：DeepSeek偶发返回无效JSON导致None
            if report is None:
                raise ValueError("structured output returned None")
            # 取出薄弱点列表（Pydantic→dict）
            llm_weak_points = [
                wp.model_dump()  # 单个转字典
                for wp in report.weak_points  # 遍历
            ]
            # 取出总体评价文字
            llm_summary = report.overall_summary
        except Exception as e:  # LLM挂了或返回None
            # 降级处理，不崩整个流程
            logger.warning(
                "analyze_weak_points.llm_failed",
                error=str(e),
            )
            # 兖底文案（llm_weak_points 已在try前初始化为[]，此处只赋值summary）
            llm_summary = "薄弱点分析失败，请教师根据情况人工判断。"

    # ── 合并两路结果，去重 ───────────────────
    # 所有知识点标签：先把第一路结果知识点标签拿出来去重
    merged_tags = set(tagged_weak.keys())
    final_weak_points = []  # 最终结果列表

    # 遍历第二路LLM推断出的每个薄弱点
    for llm_wp in llm_weak_points:
        tag = llm_wp.get("tag", "")  # 取标签名
        if tag in tagged_weak:  # 规则已有此标签
            # 用LLM的suggestion学习建议补充进去
            tagged_weak[tag]["suggestion"] = (
                llm_wp.get("suggestion", "")  # 取建议
            )
        elif tag not in merged_tags:  # 知识点标签不在标签集合中
            # 纯LLM推断的→收录
            final_weak_points.append(llm_wp)  # 加入最终结果列表
            merged_tags.add(tag)  # 知识点标签进集合

    # 知识点标签对应的聚合结果也加入最终列表
    for wp in tagged_weak.values():  # 遍历每个薄弱知识点的聚合信息
        if not wp["suggestion"]:  # 没学习建议→给默认
            wp["suggestion"] = (
                f"建议重点复习 {wp['tag']} "  # 知识点标签名
                f"相关知识点。"  # 固定后缀
            )
        final_weak_points.append(wp)  # 加入最终列表

    # 按错题数降序（最薄弱的排前面）
    final_weak_points.sort(
        key=lambda x: x.get("wrong_count", 0),  # 排序键
        reverse=True,  # 降序
    )

    # 记录日志：分析完毕
    logger.info(
        "analyze_weak_points.done",  # 事件名
        submission_id=state["submission_id"],  # 哪份卷
        weak_count=len(final_weak_points),  # 几个薄弱点
    )

    # 写入 state，供后续报告生成用
    return {
        "weak_points": final_weak_points,  # 薄弱点列表
        "weak_points_summary": (  # 总体评价
            llm_summary  # LLM给的评价
            or "已完成薄弱点分析，请查看详情。"  # 兜底
        ),
    }

# ─────────────────────────────────────────────────────────────
# 6-9 HitL 人在图中
# ─────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
# 节点6：notify_teacher — 更新提交状态（通知教师）
# ────────────────────────────────────────────────────────────
async def notify_teacher_node(state: ExamState) -> dict:
    """将提交状态推进到 pending_review，等待教师确认。
    职责：
        数据库更新 exam_submissions.status = 'pending_review'
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                text("""
                    UPDATE exam_submissions
                    SET status = 'pending_review', 
                        updated_at = NOW()
                    WHERE id = :submission_id
                """),
                {"submission_id": state["submission_id"]},
            )

    logger.info("notify_teacher.done",
                submission_id=state["submission_id"])
    return {"teacher_notified": True}


# ────────────────────────────────────────────────────────────
# 节点7：teacher_review — Human-in-the-Loop 暂停点
# ────────────────────────────────────────────────────────────
async def teacher_review_node(state: ExamState) -> dict:
    """Human-in-the-Loop 核心节点。
    interrupt(display_data) 做两件事：
        1. 把 display_data 暴露给外部（教师可通过 GET 接口读取）
        2. 冻结图执行，State 自动保存到 MemorySaver
    图在此暂停，直到教师通过POST /confirm 传入Command(resume=decision)才恢复。
    恢复后，interrupt()的返回值就是 decision（teacher_decision字段）。
    """
    display_data = {
        "submission_id": state["submission_id"], # 提交ID
        "student_id": state["student_id"], # 学员ID
        # 预批改总结
        "pre_review_summary": state.get("pre_review_summary", {}),
        "weak_points": state.get("weak_points", []), # 薄弱点列表
        # 薄弱点总结
        "weak_points_summary": state.get("weak_points_summary", ""),
        "message": "请检查AI预批改结果和知识薄弱点分析，确认无误后点击发布。",
    }

    # TODO interrupt() 类似input() 图在此暂停执行，等待教师确认
    teacher_decision = interrupt(display_data)
    ''' display_data的作用：代码可读性
    图冻住（interrupt）
        ↓ state 存进 checkpoint
    前端轮询 /exam/status
        ↓ 发现 waiting_review
    前端调 /exam/review/{session_id}
        ↓ 后端从 checkpoint 读 state
    返回给教师看的数据
    '''

    logger.info(
        "teacher_review.resumed",
        submission_id=state["submission_id"],
        action=teacher_decision.get("action", "unknown"),
    )

    return {"teacher_decision": teacher_decision}

# ──────────────────────────────────────────────────────────────
# 6-10
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 节点8：apply_teacher_decision
# 作用：把教师的审核决策合并到批改结果里
# ──────────────────────────────────────────────────────────────
# 教师审核后只有两种动作：
#   approve → AI 的分就是最终分，不改
#   modify  → 教师指定某几道题改分/改评语
# 这个节点就是干这个「合并」的活。
# ──────────────────────────────────────────────────────────────
async def apply_teacher_decision_node(state: ExamState) -> dict:
    """将教师决策合并到批改结果中。
    approve：直接把 AI 分数作为最终分数
    modify：按 modifications 覆盖指定题目的得分和评语
    """
    # 从 state 取教师的决策对象
    decision = state.get("teacher_decision", {})
    # 教师动作action：approve 或 modify
    action = decision.get("action", "approve")
    # 改分明细列表（approve 时为空）
    modifications = decision.get("modifications", [])

    # 取 AI 预批改的完整题目列表
    # 这是「底子」，教师的修改是在这上面改
    all_results = list(
        state.get("pre_review_summary", {}).get("by_question", []))

    # 每道题先加一个 final_score 字段，初始值 = AI 给的分
    # 后面如果教师改了，再覆盖
    for r in all_results:
        r["final_score"] = r.get("score", 0)

    # ── 教师选了modify 且有改分明细列表modifications → 逐条覆盖 ──
    if action == "modify" and modifications:
        # 建一个「题目ID → 列表下标」的索引
        # 这样按 question_id 找题，不用每次遍历整个列表
        id_to_idx = {r["question_id"]: idx
                     for idx, r in enumerate(all_results)}
        # 遍历教师的每一条修改
        for mod in modifications:
            qid = mod.get("question_id") # 题目ID
            # 确认这道题在列表里
            if qid in id_to_idx:
                idx = id_to_idx[qid] # 列表下标
                # 教师给了新分数 → 覆盖
                if "new_score" in mod:
                    all_results[idx]["teacher_score"] = mod[
                        "new_score"]
                    # final_score 以教师为准
                    all_results[idx]["final_score"] = mod[
                        "new_score"]
                # 教师给了新评语 → 覆盖
                if "comment" in mod:
                    all_results[idx]["teacher_comment"] = mod[
                        "comment"]
                # 记录是谁审的
                all_results[idx]["reviewed_by"] = decision.get(
                    "teacher_id", "")

    # 记录日志
    logger.info(
        "apply_teacher_decision.done",
        action=action,
        modifications_count=len(modifications),
    )

    # 返回合并后的最终结果 → 写入 state
    return {"final_results": all_results}


# ──────────────────────────────────────────────────────────────
# 节点9：publish_results
# 作用：把最终批改结果写入数据库，发布给学员
# ──────────────────────────────────────────────────────────────
# 写两张表：
#   exam_reviews     → 每道题的批改明细（先删后插）
#   exam_submissions → 更新状态为 published
# 「先删后插」是为了幂等：
#   万一教师重复点发布，不会产生重复记录
# ──────────────────────────────────────────────────────────────
async def publish_results_node(state: ExamState) -> dict:
    """将最终批改结果写入数据库，发布给学员。
    写入逻辑：
        exam_reviews表──先删后插（幂等），每道题一行
        exam_submissions表──更新status='published' + weak_points JSON
    """
    # 从 state 取最终批改结果（教师确认后的）
    final_results = state.get("final_results", [])
    # 提交记录 ID（写回哪条提交记录）
    submission_id = state["submission_id"]
    # 审核教师 ID（没有就 None）
    teacher_id = state.get("teacher_decision", {}).get(
        "teacher_id", "") or None
    # 知识薄弱点列表
    weak_points = state.get("weak_points", [])

    async with AsyncSessionLocal() as session: # 异步数据库会话
        # session.begin()：开启事务 要么全成功，要么全回滚
        async with session.begin():
            # ── 逐题写入 exam_reviews表 ──
            for r in final_results:
                # 这道题如果之前写过，先删掉 防止重复发布时产生重复记录
                await session.execute(
                    text("""DELETE FROM exam_reviews
                        WHERE submission_id = :submission_id
                        AND question_id = :question_id
                    """),
                    {"submission_id": submission_id,
                     "question_id": r["question_id"]},
                )
                # 后插：写入这道题的完整批改记录
                await session.execute(
                    text("""INSERT INTO exam_reviews
                        (id, submission_id, question_id,
                        question_type, knowledge_tag, student_answer,
                        ai_score, ai_feedback, ai_raw_result,
                        teacher_score, teacher_comment, final_score,
                        needs_review, reviewed_by, reviewed_at)
                        VALUES (
                        :id, 
                        :submission_id, 
                        :question_id,
                        :question_type, 
                        :knowledge_tag, 
                        :student_answer,
                        :ai_score, 
                        :ai_feedback, 
                        :ai_raw_result,
                        :teacher_score, 
                        :teacher_comment, 
                        :final_score,
                        :needs_review, 
                        :reviewed_by, NOW())"""
                    ),
                    {
                        "id": str(uuid.uuid4()),  # 生成唯一 ID
                        "submission_id": submission_id, # 提交记录 ID
                        "question_id": r["question_id"], # 题目 ID
                        "question_type": r["question_type"], # 题型
                        # 知识点标签
                        "knowledge_tag": r.get("knowledge_tag", ""),
                        "student_answer": # 学生答案
                            r.get("student_answer", ""),
                        "ai_score": r.get("score", 0), # AI给的分数
                        # AI 评语
                        "ai_feedback": r.get("ai_feedback", ""),
                        # 完整原始结果存 JSON
                        "ai_raw_result": json.dumps(r),
                        # 教师改的分（没改就是 None）
                        "teacher_score": r.get("teacher_score"),
                        # 教师改的评语（没改就是 None）
                        "teacher_comment": r.get("teacher_comment"),
                        # 最终分：教师改过用教师的，没改用 AI 的
                        "final_score":
                            r.get("final_score", r.get("score", 0)),
                        # 是否需要审核
                        "needs_review": r.get("needs_review", False),
                        "reviewed_by": teacher_id, # 审核教师 ID
                    },
                )

            # ── 更新提交状态为已发布 ──
            # 学员前端看到 status=published 就知道成绩出来了
            await session.execute(
                text("""
                    UPDATE exam_submissions
                    SET status = 'published',
                        published_at = NOW(),
                        updated_at = NOW(),
                        weak_points = :weak_points,
                        weak_points_summary = :weak_points_summary
                    WHERE id = :submission_id
                """),
                {
                    "submission_id": submission_id, # 提交记录 ID
                    # 薄弱点列表转 JSON 字符串存入
                    "weak_points": json.dumps(weak_points),
                    "weak_points_summary": # 知识薄弱点总结
                        state.get("weak_points_summary", ""),
                },
            )

    # 算总分和满分（用于返回给前端）
    total_final = sum(r.get("final_score", 0) for r in final_results)
    full_score = sum(r.get("full_score", 0) for r in final_results)

    # 记录日志
    logger.info(
        "publish_results.done",
        submission_id=submission_id,
        final_score=total_final,
        weak_points_count=len(weak_points),
    )

    # 返回给 API 层，API 直接返给教师确认接口
    return {
        "published": True, # 是否发布成功
        "structured_output": {
            "submission_id": submission_id, # 提交记录 ID
            "final_score": total_final, # 最终分
            "full_score": full_score, # 满分
            "score_rate": # 得分率
                round(total_final / full_score, 4)
                if full_score else 0,
            "weak_points": weak_points, # 知识薄弱点列表
            "weak_points_summary": # 知识薄弱点总结
                state.get("weak_points_summary", ""),
            "published": True, # 是否发布成功
        },
    }


if __name__ == '__main__':
    parsed = _sync_parse_word(
        '../../../tmp/student_answer.docx')
    for q in parsed:
        print(f'题{q["question_no"]}: '
              f'{repr(q["student_answer"])}')

    # ── 测试：load_questions_meta_node 合并DB数据 ──
    # 需要一个真实的 exam_id（从数据库查）
    async def check_load_meta():
        EXAM_ID = "e0000001-0000-0000-0000-000000000001"
        state = {
            "exam_id": EXAM_ID,
            "parsed_questions": parsed,
        }
        result = await load_questions_meta_node(state)
        print(f'\n➡️ 节点产出: {result}')
        merged = result["parsed_questions"]
        print(f'\n合并后共 {len(merged)} 道题：')
        for q in merged:
            print(f'  题{q["question_no"]} '
                  f'[{q["question_type"]}] '
                  f'满分{q["full_score"]} '
                  f'学员答案={repr(q["student_answer"])}')

    asyncio.run(check_load_meta())