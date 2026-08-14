# backend/core/reranker.py

from dataclasses import dataclass
from typing import Optional
import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.config import get_abs_path
from backend.core.logger import get_logger
logger = get_logger(__name__)

# 截断过长文档，防止超出CrossEncoder的最大输入长度限制
# max_length=512tokens
# 1200字符大约对应512tokens，留有一定安全余量，避免截断导致语义丢失
RERANK_MAX_INPUT_CHARS = 1200


@dataclass
class RankedDocument:
    """
    精排后的单个文档结果数据类。
    用于在内部模块间传递精排结果，保持数据结构的一致性。
    """
    content: str  # 文档文本内容
    score: float  # BGE-Reranker 输出的相关性概率，范围 [0, 1]
    # 在原始召回列表中的位置索引（从 0 开始），便于后续溯源
    original_index: int
    metadata: dict  # 来源元数据（如 source_name / chunk_type / course_id 等）


class BGEReranker:
    """
    BGE-Reranker精排服务（单例模式）。

    核心作用：
    对 Hybrid（混合检索）召回的候选文档进行 CrossEncoder 交叉编码精排。
    由于 CrossEncoder 模型默认经过 Sigmoid 激活，直接输出 [0, 1] 的相关性概率，
    因此无需在应用层做额外的归一化处理。

    用法示例：
        reranker = BGEReranker.get_instance()
        docs, confidence = reranker.rerank_with_confidence(
            query="什么是 Spring IOC？",
            documents=candidates,
            top_k=3,
        )
    """

    # 类变量：用于保存单例实例，确保全局只有一个模型加载在内存中
    _instance: Optional["BGEReranker"] = None

    def __init__(self):
        # 从配置中获取BGE-RERANKER型路径，转为绝对路径
        model_path = get_abs_path(
            get_settings().reranker_model_path)
        # 自动检测设备：优先使用 CUDA GPU 加速，否则回退到 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("reranker.loading",
                    model_path=model_path, device=device)
        # 初始化 CrossEncoder，限制最大序列长度为 512 tokens（BGE-Reranker 的上限）
        self._model = CrossEncoder(
            model_path, device=device, max_length=512,
            # 强制transformers直接在CPU上分配真实权重，跳过meta device
            automodel_args={"low_cpu_mem_usage": False},
        )
        logger.info("reranker.loaded",
                    model_path=model_path,
                    device=device,
                    max_length=512)

    @classmethod
    def get_instance(cls) -> "BGEReranker":
        """
        获取 Reranker 单例。
        首次调用时触发模型加载（懒加载机制），后续调用直接返回已加载的实例，
        避免重复加载模型导致显存溢出或内存浪费。
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # 执行精排并返回 Top-K 结果及置信度
    def rerank_with_confidence(
            self,
            query: str,
            documents: list[dict],
            top_k: int = 3,
        ) -> tuple[list[RankedDocument], float]:
        """
        执行精排并返回 Top-K 结果及置信度。

        Args:
            query:     用户输入的原始 Query
            documents: 候选文档列表，每项必须包含 "content" 字段
            top_k:     返回的相关性最高的文档数量，默认为 3

        Returns:
            (ranked_docs, confidence):
            - ranked_docs:  按相关性分数降序排列的 RankedDocument 列表，长度 <= top_k
            - confidence:   Top-1 文档的 BGE 相关性概率 [0, 1]。
                            业务约定：
                            ≥ 0.75 → 高置信度，直接走 LLM 生成回答；
                            < 0.75 → 低置信度，触发 Web 搜索兜底或提示用户澄清。
        """
        # 防御性编程：如果候选文档为空，直接返回空结果和 0 置信度
        if not documents:
            return [], 0.0

        # 构造 CrossEncoder 的输入对：(query, document)

        pairs = [ # 截断长文档，防止超出 max_length=512 导致报错或性能下降
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]

        # 批量推理：predict() 直接输出 [0, 1] 概率
        scores: list[float] = self._model.predict(pairs).tolist()

        # 将原始文档与精排分数绑定，并按分数降序排序
        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""), # 文档文本内容
                    score=scores[i], # BGE-Reranker输出的相关性概率，范围 [0, 1]
                    original_index=i, # 在原始召回列表中的位置索引（从 0 开始）
                    metadata=documents[i].get("metadata", {}), # 元数据
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,  # 降序排列，分数最高的排在最前面
        )

        # 截取 Top-K 结果
        top_results = ranked[:top_k]
        # 提取 Top-1 的分数作为整体置信度，若结果为空则默认为 0.0
        confidence = top_results[0].score if top_results else 0.0

        # 记录精排完成的日志，便于后续监控和排查问题
        logger.info(
            "reranker.done",
            candidates=len(documents),  # 原始候选文档数量
            top_k=top_k,  # 请求返回的数量
            confidence=round(confidence, 4),  # 保留 4 位小数，便于阅读
        )
        # 返回 tok_k 的结果及最高得分（置信度）
        return top_results, confidence


def retrieve(
        query: str,
        tenant_id: str,
        course_id: Optional[str] = None,
        recall_top_k: int = 10,
        rerank_top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
    """
    Hybrid 召回 → BGE 精排一体化 Pipeline。

    核心设计：
    调用方只需传入 Query 文本和租户/课程参数，无需关心底层的向量化、Milvus 检索及精排逻辑，
    所有中间步骤均在此函数内部闭环完成。

    Args:
        query:        用户输入的原始 Query 文本
        tenant_id:    租户 ID（用于 Milvus 的硬过滤，确保数据隔离）
        course_id:    课程 ID（可选，进一步缩小检索范围，提升精准度）
        recall_top_k: Hybrid 召回数量（默认 10，即从向量库中捞取的候选文档数，送给精排）
        rerank_top_k: 精排后最终返回数量（默认 3，送给 LLM 作为上下文）

    Returns:
        (ranked_docs, confidence):
        - ranked_docs:  精排后 Top-rerank_top_k 的文档列表
        - confidence:   Top-1 文档的 BGE 置信度 [0, 1]，用于下游决策（如是否触发兜底）
    """
    # 延迟导入（Lazy Import）：
    # 1. 避免模块间的循环依赖（Circular Import）问题；
    # 2. 将重量级模型（Embedder/KB Client）的加载推迟到首次调用时，加快服务启动速度。
    from backend.core.knowledge_base import BGEMEmbedder, KnowledgeBaseClient

    # ── 第一步：Query 向量化（在 Pipeline 内部完成，调用方无需感知）──────
    # 获取 Embedder 单例，将用户文本同时转换为 Dense 和 Sparse 向量
    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    # ── 第二步：Hybrid 召回（混合检索） ───────────────────────────────
    kb = KnowledgeBaseClient()
    # 根据租户和课程参数构建 Milvus过滤表达式
    # 如 "tenant_id == 'xxx' and course_id == 'yyy'"
    filters = kb._build_filter(tenant_id, course_id)

    # 执行混合检索：同时利用 Dense 向量（语义匹配）和 Sparse 向量（精确关键词匹配）
    documents = kb._hybrid_search(
        query_embedding=dense_vec,  # 稠密向量
        query_sparse=sparse_vec,  # 稀疏向量
        top_k=recall_top_k,  # 召回候选文档数量，默认 10
        filters=filters,  # 权限/范围过滤条件
    )

    # 防御性编程：如果召回结果为空，记录日志并直接返回空结果，避免后续无效计算
    if not documents:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    # ── 第三步：CrossEncoder 精排 ────────────────────────────────────────
    # 获取 Reranker 单例，对召回的候选文档进行深度交叉编码，返回 Top-K 结果及置信度
    reranker = BGEReranker.get_instance()
    # 返回rerank_top_k（3）个重排序结果及最高得分（置信度）
    return reranker.rerank_with_confidence(
        query, documents, top_k=rerank_top_k)

if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv("../../.env.local") # 加载环境变量文件

    ranked_docs, confidence = retrieve(
        query="什么是HNSW？",
        tenant_id="tenant_default",
        course_id=None,  # 不限课程
    )

    print(f"置信度：{confidence:.4f}"
          f"（{'高置信度' if confidence >= 0.75 else '低置信度'}）\n")
    for i, doc in enumerate(ranked_docs):
        print(f"[{i + 1}] score={doc.score:.4f}  "
              f"来源：{doc.metadata.get('source_name', '')}")
        print(f"     {doc.content[:80]}\n")
