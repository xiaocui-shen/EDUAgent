# scripts/build_knowledge_base.py
# -------------------------------------------------------
# Part 1: pdf md 文档加载
# -------------------------------------------------------
# 5.3 补充分块，5.4 补充嵌入，5.5 完整版

from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader)
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """加载 PDF 文档，每页返回一个 Document。
    只提取文字层内容；图片/扫描件页面 page_content 为空字符串，
    不报错（分块时会过滤掉空页）。
    Args:
        file_path: PDF 文件的本地路径
    Returns:
        list[Document]，每个 Document 对应一页
        metadata 包含 source（文件路径）和 page（页码，从 0 开始）
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"  [PDF] 加载完成：{len(pages)} 页 ← {Path(file_path).name}")
    return pages


def load_markdown(file_path: str) -> list[Document]:
    """加载 Markdown 文档，整个文件作为一个 Document 返回。
    不在这里做切分——那是后续分块步骤的工作。
    这里只负责把文件内容读进内存。
    Args:
        file_path: Markdown 文件的本地路径（.md 或 .markdown）
    Returns:
        list[Document]，只有一个元素，page_content 为文件全文
        metadata 包含 source（文件路径）
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    char_count = len(docs[0].page_content)
    print(f"  [MD] 加载完成：{char_count} 字符 ← {Path(file_path).name}")
    return docs


def load_document(file_path: str) -> list[Document]:
    """统一文档加载入口。
    根据文件扩展名自动选择加载器：
        .pdf            → PyPDFLoader（文字层提取）
        .md / .markdown → TextLoader（纯文本读取）
    Args:
        file_path: 文档本地路径
    Returns:
        list[Document]
    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = path.suffix.lower() # 取出文件扩展名并转为小写
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".md", ".markdown"):
        return load_markdown(file_path)
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"
            f"当前支持：.pdf / .md / .markdown\n"
            f"提示：可用 markitdown 将 Word/PPT 转换为 .md 后再导入"
        )

# -------------------------------------------------------
# Part 2: pdf md 文档分割
# -------------------------------------------------------
# scripts/build_knowledge_base.py（阶段版：文档加载 + 分块，5.4 / 5.5 继续补全）


from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── 模块级分块器单例 ──────────────────────────────────────────
_MD_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "H1"),
        ("##",  "H2"),
        ("###", "H3"),
    ],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

# ── PDF 分块 ──────────────────────────────────────────────────
def split_pdf_documents(pages: list[Document]) -> list[Document]:
    """PDF 文档分块：过滤空页 + RecursiveCharacterTextSplitter"""
    non_empty_pages = [p
                       for p in pages
                       if len(p.page_content.strip()) > 20]
    skipped = len(pages) - len(non_empty_pages)
    if skipped > 0:
        print(f"  过滤空页：{skipped} 页（图片/扫描件页）")

    chunks = _CHAR_SPLITTER.split_documents(non_empty_pages)

    for chunk in chunks:
        # 纯文件名，去除路径和拓展名
        filename = Path(chunk.metadata.get("source", "未知文件")).stem
        page_num = chunk.metadata.get("page", 0) + 1
        # 元数据["source_name"] = 纯文件名 + 页码
        chunk.metadata["source_name"] = f"{filename} 第{page_num}页"

    print(f"  [PDF] 分块完成：{len(non_empty_pages)} 页 → {len(chunks)} 个 chunk")
    return chunks


# ── Markdown 分块 ─────────────────────────────────────────────

def split_markdown_documents(docs: list[Document]) -> list[Document]:
    """
    Markdown 文档分层分块逻辑
    1. 先按 Markdown 一级/二级/三级标题切割，保留标题层级元数据
    2. 对标题分段后的大块文本，再用递归字符分割做二次细切分
    3. 统一填充自定义元数据 source_name：文件名 > H1 > H2 > H3，方便检索溯源
    :param docs: 原始加载完成的MD文档列表
    :return: 经过两级分割、携带完整层级元数据的细分片Document列表
    """
    # 存储第一层：按标题分割后的分段文档
    header_chunks: list[Document] = []

    # 遍历每个原始MD文件
    for doc in docs:
        # 第一步：按#、##、###标题切分文本，每个分段自动携带 H1/H2/H3 标题元数据
        sections = _MD_HEADER_SPLITTER.split_text(doc.page_content)
        # 获取当前文档原始文件路径
        source_path = doc.metadata.get("source", "")

        # 给每个标题分段绑定来源文件路径元数据
        for section in sections:
            section.metadata["source"] = source_path
        # 收集所有标题分段
        # 合并列表 [Document] + [Document] ==> [多个Document]
        header_chunks.extend(sections)

    # 第二步：切子块，按符号顺序做递归切割、直到分成512字符数的块
    final_chunks = _CHAR_SPLITTER.split_documents(header_chunks)

    # 遍历所有最终分片，组装可读的层级文件名标识存入元数据
    for chunk in final_chunks:
        # 取出文件原始路径
        source_path = chunk.metadata.get("source", "")
        # 提取无后缀纯文件名，无路径则填充默认值
        filename = Path(source_path).stem \
            if source_path else "未知文件"
        # 提取当前分片携带的三级标题内容
        parts = [
            chunk.metadata.get("H1", ""),
            chunk.metadata.get("H2", ""),
            chunk.metadata.get("H3", ""),
        ]
        # 过滤空标题，只保留有内容的层级
        parts = [p for p in parts if p]

        # 拼接格式：文件名 > 一级标题 > 二级标题 > 三级标题
        if parts:
            # 添加元数据
            chunk.metadata["source_name"] = f"{filename} > {' > '.join(parts)}"
        else:
            # 无任何标题时仅保留文件名
            chunk.metadata["source_name"] = filename

    # 打印分块统计日志
    print(f"  [MD]  分块完成：{len(docs)} 个文件 → {len(final_chunks)} 个 chunk")
    return final_chunks


# ── 统一分块入口 ──────────────────────────────────────────────

def split_documents(docs: list[Document], file_path: str) -> list[Document]:
    """统一分块入口，根据文件类型自动选择分块策略"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return split_pdf_documents(docs)
    elif ext in (".md", ".markdown"):
        return split_markdown_documents(docs)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


# -------------------------------------------------------
# Part 3: 补充嵌入
# -------------------------------------------------------
import sys

# 把项目根目录加入导包路径，使得build_knowledge_base.py 能import backend.*
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.core.knowledge_base import (
    BGEMEmbedder, DocumentChunk, generate_chunk_id)

BATCH_SIZE = 12   # BGE-M3 批量推理大小（8-12是平衡点）

def embed_chunks(
    chunks: list[Document],
    course_id: str,
    document_id: str,
    tenant_id: str = "tenant_default",
    version: str = "1.0",
) -> list[DocumentChunk]:
    """对 split_documents() 产出的 chunk 列表做 BGE-M3 嵌入
    返回 DocumentChunk 列表。
    BGE-M3 推理为 CPU / GPU-bound，按 BATCH_SIZE 批量处理：
    - 减少模型推理次数（每次推理有固定启动开销）
    - 控制显存/内存峰值（整批一次性推理会爆显存）
    Args:
        chunks:      split_documents() 返回的 list[Document]
        course_id:   所属课程 UUID
        document_id: 文档的 UUID（用于 Milvus 幂等更新，删旧插新）
        tenant_id:   租户 ID，用于 Milvus 多租户过滤
        version:     课程版本号
    Returns:
        list[DocumentChunk]，每项包含dense+sparse向量，可直接写入 Milvus
    """
    embedder = BGEMEmbedder.get_instance() # BEG-M3单例，首次调用加载模型
    all_doc_chunks: list[DocumentChunk] = []

    total = len(chunks) # 文档块总数
    for batch_start in range(0, total, BATCH_SIZE):
        # 按下表取BATCH_SIZE个文档块列表
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        # 取出文档块文本内容，构造[str]
        texts = [c.page_content for c in batch]

        # BGE-M3 批量推理：同时拿到 dense 和 sparse
        dense_vecs, sparse_vecs = (
            embedder.encode(texts, batch_size=BATCH_SIZE))

        # batch, dense_vecs, sparse_vecs 配对，一起被遍历
        for i, (chunk, dense, sparse) in (
                enumerate(zip(batch, dense_vecs, sparse_vecs))):
            global_index = batch_start + i # 在整个文档中的顺序编号

            all_doc_chunks.append(DocumentChunk(
                id=generate_chunk_id(
                    chunk.page_content,
                    document_id,
                    global_index
                ), # md5 id
                content=chunk.page_content, # 块文本内容
                embedding=dense, # 稠密向量
                sparse_embedding=sparse, # 稀疏向量
                course_id=course_id, # 所属课程ID
                document_id=document_id, # 文档ID
                source_name=chunk.metadata.get("source_name", ""), # 源文件名
                chunk_type=chunk.metadata.get("chunk_type", "text"), # 块类型
                chunk_index=global_index, # 块在所有文档块中的顺序编号
                version=version, # 课程版本号
                tenant_id=tenant_id, # 租户ID
            ))

        done = min(batch_start + BATCH_SIZE, total) # 取最小值
        print(f"  嵌入进度：{done}/{total}")

    print(f"  嵌入完成：{len(all_doc_chunks)} 个 DocumentChunk")
    return all_doc_chunks # 返回所有文档块构成的列表 [DocumentChunk]


# -------------------------------------------------------
# Part 4: RAG 上下文增强
# -------------------------------------------------------
import asyncio
import uuid
import argparse
from backend.core.llm_factory import get_llm
from backend.core.knowledge_base import KnowledgeBaseClient

# 上下文生成 Prompt
CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{document_text}
</document>

以下是需要在整个文档中定位的 chunk：
<chunk>
{chunk_content}
</chunk>

请用一句简洁的中文，描述这段内容在整个文档中的位置和作用，以便改善检索效果。
只输出这一句描述，不要加任何前缀或标签。"""

async def generate_chunk_context(
        llm,
        document_text: str,
        chunk_content: str,
        semaphore: asyncio.Semaphore,
    ) -> str:
    """用 LLM 为单个 chunk 生成一句定位描述。
    失败时返回空字符串，调用方保留原始 chunk 文本（降级处理）。
    Args:
        llm:           DeepSeek LLM 实例（via get_llm）
        document_text: 整篇文档全文（截断至 8000 字）
        chunk_content: 当前 chunk 的原始文本
        semaphore: 并发限流（最多 MAX_CONTEXT_CONCURRENCY 个 LLM 请求同时进行）
    """
    async with semaphore: # 限流：最多 concurrency 个请求同时进行
        try:
            from langchain_core.messages import HumanMessage
            prompt = CONTEXTUAL_CHUNK_PROMPT.format( # 拼接提示词
                document_text=document_text,
                chunk_content=chunk_content,
            )
            resp = await llm.ainvoke(
                [HumanMessage(content=prompt)])
            # 兼容不同版本 LangChain 的返回值格式
            ctx = (
                resp.text
                if hasattr(resp, "text")
                   and not callable(resp.text)
                else str(resp.content)
            ).strip()
            return ctx
        except Exception as e:
            print(f"   [warning] 上下文生成失败，保留原始 chunk：{e}")
            return "" # 降级：不添加上下文，原始chunk继续使用

MAX_CONTEXT_CONCURRENCY = 5 # 上下文生成的最大并发 LLM 请求数
async def add_context(
        chunks: list[Document],
        docs: list[Document],
        concurrency: int = MAX_CONTEXT_CONCURRENCY,
    ) -> list[Document]:
    """Contextual RAG：并发为所有 chunk 生成上下文描述，拼接到 chunk 文本前方。
    拼接后格式：
        "<上下文描述一句话>\\n\\n<原始 chunk 文本>"
    拼接后再做嵌入（embed_chunks），向量同时编码"在哪里"和"说了什么"两层信息。
    Args:
        chunks: split_documents() 输出的 list[Document]
        docs: load_document() 输出的原始 list[Document]（用于构建全文参考）
        concurrency: 最大并发 LLM 请求数（默认 5，防止触发 API 限流）
    Returns:
        page_content 已被就地修改（拼接上下文）的 list[Document]
    """
    # 拼接全文供 LLM 参考（截断 8000 字，避免超出模型 context 长度）
    full_doc_text = "\n\n".join( # 用双换行拼接所有页的文本
        d.page_content for d in docs)[:8000]

    llm = get_llm("qa", temperature=0) # 获取LLM实例
    semaphore = asyncio.Semaphore(concurrency) # 信号量控制并发数

    # 并发调用 LLM，为每个 chunk 生成上下文描述
    contexts = await asyncio.gather(
        *[generate_chunk_context( # 为每个chunk生成一个任务
            llm, full_doc_text, c.page_content, semaphore)
          for c in chunks]
    )

    enriched = 0
    for chunk, ctx in zip(chunks, contexts): # 配对原文和生成上下文
        if ctx:
            # 拼接格式：<上下文描述>\n\n<原始 chunk 文本>
            chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
            enriched += 1

    print(f"  上下文增强完成：{enriched}/{len(chunks)} 个 chunk 已添加描述")
    return chunks


# ── Step 4：写入 Milvus ────────────────────────────────────────

def write_to_milvus(doc_chunks: list[DocumentChunk]) -> None:
    """将 embed_chunks() 产出的 DocumentChunk 列表写入 Milvus。
    先按 document_id 删除同文档旧版本 chunk，再批量 upsert，
    保证文档更新时不残留旧数据。"""
    if not doc_chunks:
        print("  ⚠️  无 chunk 可写入，跳过")
        return
    kb          = KnowledgeBaseClient() # 单例客户端
    document_id = doc_chunks[0].document_id # 取第一个chunk的文档ID

    # 先删旧版本，再批量插入，保证幂等更新
    print(f"  🗑️  删除旧版本 chunk（document_id={document_id[:8]}…）")
    kb.delete_document_chunks(document_id) # 按文档ID删旧数据

    written = kb.upsert_chunks(doc_chunks) # 批量写入新数据
    print(f"  ✅ 写入完成：{written} 个 chunk → knowledge_domain")


# ── 主流水线 ─────────────────────────────────────────────────
async def build_pipeline(
    file_path:   str,
    course_id:   str,
    document_id: str,
    tenant_id:   str = "tenant_default",
    version:     str = "1.0",
    use_context: bool = True,
) -> None:
    """知识库建库完整流水线（五步）：
      Step 1   读取文档（PyPDFLoader / TextLoader）
      Step 2   智能分块（MarkdownHeaderTextSplitter / RecursiveCharacterTextSplitter）
      Step 2.5 Contextual RAG 上下文增强（LLM 并发，可跳过）
      Step 3   BGE-M3 嵌入（dense + sparse 双向量）
      Step 4   写入 Milvus（MilvusClient upsert）
    """
    print(f"\n{'='*55}")
    print(f" EduAgent 知识库构建")
    print(f" 文件      ：{file_path}")
    print(f" 课程      ：{course_id}")
    print(f" 文档 ID   ：{document_id}")
    print(f" 租户      ：{tenant_id}")
    print(f" Contextual RAG：{'启用' if use_context else '跳过（--no-context）'}")
    print(f"{'='*55}\n")

    # Step 1：读取
    print("📖 Step 1/4  读取文档…")
    docs = load_document(file_path)

    # Step 2：分块
    print("\n✂️  Step 2/4  智能分块…")
    chunks = split_documents(docs, file_path)

    # Step 2.5：Contextual RAG（可选）
    if use_context and chunks:
        print(f"\n🧠 Step 2.5  Contextual RAG 上下文增强"
              f"（并发={MAX_CONTEXT_CONCURRENCY}）…")
        chunks = await add_context(chunks, docs)

    # Step 3：嵌入
    print("\n🔢 Step 3/4  BGE-M3 嵌入…")
    doc_chunks = embed_chunks(
        chunks,
        course_id=course_id,
        document_id=document_id,
        tenant_id=tenant_id,
        version=version,
    )

    # Step 4：写入
    print("\n💾 Step 4/4  写入 Milvus…")
    write_to_milvus(doc_chunks)

    print(f"\n🎉 完成！共处理 {len(doc_chunks)} 个 chunk")
    print(f"   document_id = {document_id}")
    print(f"   ⚠️  更新此文档时请保留此 document_id")


# ── CLI 入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EduAgent 知识库构建：PDF / Markdown 导入 Milvus"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="文档路径（.pdf / .md / .markdown）")
    parser.add_argument(
        "--course_id",
        required=True,
        help="所属课程 UUID")
    parser.add_argument(
        "--document_id",
        default=None,
        help="文档 UUID（不传则自动生成；更新同一文档时传相同 ID）")
    parser.add_argument(
        "--tenant_id",
        default="tenant_default",
        help="租户 ID（默认 tenant_default）")
    parser.add_argument(
        "--version",
        default="1.0",
        help="课程版本号（默认 1.0）")
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="跳过 Contextual RAG 上下文生成（快速调试用）")
    args = parser.parse_args()

    # 如果没传 document_id，自动生成 UUID
    doc_id = args.document_id or str(uuid.uuid4())

    # 启动异步流水线
    asyncio.run(build_pipeline(
        file_path=args.file,           # 文档路径
        course_id=args.course_id,      # 课程ID
        document_id=doc_id,            # 文档ID
        tenant_id=args.tenant_id,      # 租户ID
        version=args.version,          # 版本号
        use_context=not args.no_context, # 是否启用上下文增强
    ))


