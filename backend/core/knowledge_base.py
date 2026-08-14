# backend/core/knowledge_base.py

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双输出）
# ──────────────────────────────────────────────────────────────

class BGEMEmbedder:
    """BGE-M3 本地嵌入模型单例。
    一次推理同时输出：
      - dense 向量（1024 维浮点数组，用于语义相似度检索）
      - sparse 向量（{token_id: weight} 字典，用于关键词精确检索）
    进程内单例：首次调用 get_instance() 时加载模型（约5-15秒），
    后续调用直接返回同一实例，不重复加载。
    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("什么是 Spring IOC？")
    """

    _instance: Optional["BGEMEmbedder"] = None   # 单例持有

    def __init__(self, model_path: str):
        # ── 兼容性补丁：FlagEmbedding 1.3.x 依赖 transformers 内部函数 ──
        # transformers>=5.0 移除了 is_torch_fx_available，
        # 但当前锁定 transformers==4.51.0 不受影响。
        # 此补丁作为保险，避免未来升级时报 ImportError。
        # import importlib.util as _ilu
        # from transformers.utils import import_utils as _tf_iu
        # if not hasattr(_tf_iu, "is_torch_fx_available"):
        #     _tf_iu.is_torch_fx_available = (
        #         lambda: _ilu.find_spec("torch.fx") is not None
        #     )

        from FlagEmbedding import BGEM3FlagModel
        from backend.config import get_abs_path

        logger.info("bge_m3.loading", model_path=model_path)

        # ── fp16只能在CUDA上启用，CPU不启用 ──
        # CPU 模式下用 fp32，速度稍慢但稳定。
        # _use_fp16 = torch.cuda.is_available() # True or False
        _use_fp16 = False  # True or False

        # BGE-M3 模型对象
        self._model = BGEM3FlagModel(
            # 必须绝对路径，否则会把相对路径当做线上的模型仓库、导致报错
            model_name_or_path=get_abs_path(model_path),
            use_fp16=_use_fp16,
            # normalize_embeddings=True # 开启稠密向量归一化
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """获取单例（首次调用时加载模型，后续复用）"""
        if cls._instance is None:
            cls._instance = (
                # 实例化自己，传入模型路径
                BGEMEmbedder(get_settings().bge_m3_model_path)
            )
        return cls._instance # 返回单例

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """批量编码文本，同时返回 dense 和 sparse 两种向量。
        Args:
            texts:      待编码的文本列表
            batch_size: 单次推理批大小，越大速度越快但显存占用越多；
                        12 个人电脑一般 8~16
        Returns:
            (dense_vecs, sparse_vecs)
            dense_vecs:  list of 1024-dim float 稠密向量，每项对应 texts[i]
            sparse_vecs: list of {token_id: weight} 稀疏向量字典，每项对应 texts[i]
        """
        # 对文本进行向量化
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192, # BGE-M3支持最长8192token，覆盖大多数chunk
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏关键词向量
            return_colbert_vecs=False,  # 不返回多向量，本项目不用
        )
        # 稠密向量：numpy.float32 → Python list
        dense_vecs = output["dense_vecs"].tolist()

        # 稀疏向量：dict[numpy.int, numpy.float]
        # → Python dict[int, float] ，key是token id，value是权重
        # 必须转换！LangGraph MemorySaver 用msgpack序列化State，
        # msgpack 不支持 numpy.float，会在运行时抛 TypeError。
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]

        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """编码单条查询，返回 (dense_vec, sparse_vec)。
        查询时调用此方法（而非 encode），batch_size=1 避免不必要的 padding。
        Returns:(dense_vec, sparse_vec)
            dense_vec:  1024-dim float 列表
            sparse_vec: {token_id: weight} 字典"""
        dense_list, sparse_list = self.encode(
            [text], batch_size=1)
        return dense_list[0], sparse_list[0]


# ──────────────────────────────────────────────────────────────
# 数据类：DocumentChunk（建库写入 Milvus 的数据结构）
# ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """准备写入 Milvus 的单个文档块，字段与 Milvus Schema 一一对应。
    id: 全局唯一 ID（MD5 of content + document_id + chunk_index）
    content: chunk 文本（Contextual RAG 模式下含 LLM 生成的上下文描述前缀）
    embedding: Dense 向量（BGE-M3，1024 维）
    sparse_embedding: 稀疏向量{token_id: weight}
    source_name: 元数据来源标注 检索结果展示用，如 "Java讲义>第3章>3.1 IOC"
    """
    id:               str
    content:          str
    embedding:        list[float]
    sparse_embedding: dict
    course_id:        str
    document_id:      str
    source_name:      str
    chunk_type:       str # "text" / "code" / "table"
    chunk_index:      int
    version:          str
    tenant_id:        str = "tenant_default"
    updated_at: int = field(default_factory=lambda: int(time.time()))


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """生成 chunk文档块 全局唯一 ID（MD5）。
    用 document_id + chunk_index + content 前缀组合，确保：
    - 同一文档不同位置的 chunk 不冲突
    - 内容不变时 ID 稳定（重建时不会重复插入）
    注意：此函数后续会作为 KnowledgeBaseClient 的静态方法重新出现，
        届时 build_knowledge_base.py 中会改为调用
            KnowledgeBaseClient.generate_chunk_id()
    """
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


# -------------------------------------------------------
# milvus：连接、加载集合、（批量）插入或更新、删除document_id的数据
# -------------------------------------------------------

import hashlib
from typing import Optional
from pymilvus import (
    MilvusClient,
    AnnSearchRequest,
    WeightedRanker
)

COLLECTION_NAME = get_settings().milvus_collection_name # 集合名

class KnowledgeBaseClient:
    """Milvus 知识库客户端（MilvusClient 版）。
    单 Collection 设计（knowledge_domain），按 tenant_id 字段过滤实现多租户隔离。
    单例连接：_client 是类变量，整个进程只创建一次 MilvusClient 连接。
    """
    _client: Optional["MilvusClient"] = None
    _loaded: bool = False # 是否已加载集合

    # 检索配置
    VECTOR_TOP_K = 10 # 向量搜索结果数量
    RERANK_TOP_K = 3 # 重新排序结果数量
    HIGH_CONFIDENCE_THRESHOLD = 0.75 # 相关性概率阈值（置信度）
    ANN_EF = 64  # 向量按HNSW图索引搜索邻居的数量参数

    def __init__(self):
        # ── 单例连接：整个进程只创建一次 MilvusClient ──
        if KnowledgeBaseClient._client is None:
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            KnowledgeBaseClient._client = MilvusClient( # 创建客户端
                uri=uri,
                db_name=settings.milvus_db_name, # Milvus数据库名
            )
            logger.info("milvus.connected",
                        uri=uri, db_name=settings.milvus_db_name)

        # ── 只加载一次集合到内存 ──
        if not KnowledgeBaseClient._loaded:
            try:
                KnowledgeBaseClient._client.load_collection(
                    COLLECTION_NAME) # 加载集合到内存
            except Exception as e:
                logger.warning("knowledge_base.load_skipped",
                               reason="集合可能尚未创建", error=str(e))
                raise
            KnowledgeBaseClient._loaded = True

    # ── 写入：批量 Upsert ────────────────────────────────────
    def upsert_chunks(self, chunks: list) -> int:
        """批量写入文档块（Upsert：primary key 存在则更新，不存在则插入）。
        MilvusClient 行格式写入：每行一个 dict，key = 字段名，字段顺序无关。"""
        if not chunks:
            return 0
        # 将 DocumentChunk 列表转为 Milvus 行格式 dict 列表
        data = [
            {
                "id":               c.id,                # chunk唯一ID
                "embedding":        c.embedding,         # 稠密向量
                "sparse_embedding": c.sparse_embedding,  # 稀疏向量
                "content":          c.content[:4096],    # 文本内容截断4096
                "chunk_index":      c.chunk_index,       # 分块序号
                "document_id":      c.document_id,       # 文档ID
                "course_id":        c.course_id,         # 课程ID
                "tenant_id":        c.tenant_id,         # 租户ID
                "source_name":      c.source_name,       # 来源标注
                "chunk_type":       c.chunk_type,        # text/code/table
                "version":          c.version,           # 版本号
                "updated_at":       c.updated_at,        # 更新时间戳
            }
            for c in chunks
        ]
        # 批量插入或更新
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            data=data
        )
        logger.info("knowledge_base.chunks_upserted",
                    count=len(chunks))
        return len(chunks)

    # ── 删除指定文档的所有 chunk ──────────────────────
    def delete_document_chunks(self, document_id: str) -> None:
        """删除指定文档的所有 chunk（文档更新时先删后插，幂等重建）。
        对 document_id 转义，防止 filter 表达式注入。
        """
        # 转义双引号防注入
        safe_id = document_id.replace('"', '\\"')
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'document_id == "{safe_id}"', # 按文档ID过滤删除
        )
        logger.info("knowledge_base.document_deleted",
                    document_id=document_id)

    # ── 工具方法 ─────────────────────────────────────────────
    @staticmethod
    def generate_chunk_id(
            content: str, document_id: str, chunk_index: int) -> str:
        """生成 chunk唯一ID（MD5）。内容+位置不变则ID不变"""
        raw = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()

    # --------------------------------------------------------
    #  实现 _hybrid_search()
    # --------------------------------------------------------
    # ── 检索配置 ─────────────────────────────────────────────
    def _hybrid_search(
            self,
            query_embedding: list[float],
            query_sparse: dict,
            top_k: int,
            filters: Optional[str] = None,
        ) -> list[dict]:
        """对 knowledge_domain 做 Hybrid 检索
        （Dense + Sparse → WeightedRanker 融合）。
        两个 AnnSearchRequest 分别构造 Dense 和 Sparse 检索请求，
        由 Milvus 在服务端并行执行后，用 WeightedRanker 加权融合排序。
        Args:
            query_embedding:
                Dense Query 向量
                    1024 维，来自 encode_query
            query_sparse:
                Sparse Query 向量
                    {token_id: weight}，来自 encode_query
            top_k: 每路召回数量（融合后同样取 top_k）
            filters: Milvus bool 表达式，如 'tenant_id == "xxx"'
        Returns:
            候选文档列表，每项含 "content" / "score" / "metadata"。
            score 是 WeightedRanker 的加权排序信号，不是概率，
            直接交给 Reranker 做精细打分。
        """
        try:
            # ── Dense ANN 检索请求 ─────────────────────────────────────
            # COSINE度量匹配稠密向量（L2归一化后等价于COSINE余弦相似度）
            # ef=64：HNSW搜索时的候选集大小，越大精度越高，64是精度/速度平衡点
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding", # 稠密向量字段名
                param={
                    "metric_type": "COSINE", # 度量类型：语义余弦相似度
                    "params": {"ef": self.ANN_EF}, # ef=64 搜索范围
                },
                limit=top_k,
                expr=filters,
            )

            # ── Sparse 关键词检索请求 ──────────────────────────────────
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_embedding", # 稀疏向量字段名
                param={"metric_type": "IP"}, # 度量方式 IP内积
                limit=top_k,
                expr=filters,
            )

            # 指定返回字段
            output_fields = [
                "content", "source_name", "chunk_type",
                "course_id", "document_id", "chunk_index",
            ]

            # ── WeightedRanker(0.7, 0.3) ──────────────────────────────
            # 第一个权重对应第一个请求（Dense）
            # 第二个对应第二个请求（Sparse）
            # 两路结果在 Milvus 服务端并行检索，融合后返回
            results = self._client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )

            # ── 解析结果 ───────────────────────────────────────────────
            # MilvusClient的hybrid_search结果用distance字段存融合后的分数
            # 这个分数是排序信号，不是概率，不做任何额外处理，直接传给 Reranker
            # 构造结果列表
            candidates = []
            for hit in results[0]:
                candidates.append({
                    # 内容
                    "content": hit.get("content") or "",
                    # WeightedRanker得分
                    "score":   hit.get("distance") or 0.0,
                    # 元数据
                    "metadata": {
                        "source_name": hit.get("source_name") or "",
                        "chunk_type":
                            hit.get("chunk_type") or "text",
                        "course_id":   hit.get("course_id") or "",
                        "document_id": hit.get("document_id") or "",
                        "chunk_index": hit.get("chunk_index") or 0,
                    },
                })

            logger.info(
                "knowledge_base.hybrid_search_done",
                candidates=len(candidates),
            )
            return candidates

        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed",
                         error=str(e))
            return []

    @staticmethod
    def _build_filter(
            tenant_id: str,
            course_id: Optional[str] = None) -> str:
        """构建 Milvus bool 过滤表达式。
        对 tenant_id / course_id 做转义，防止 filter 表达式注入。"""
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr


if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv("../../.env.local") # 加载环境变量文件
    query = "什么是HNSW？"

    embedder = BGEMEmbedder.get_instance()
    dense, sparse = embedder.encode_query(query)

    kb = KnowledgeBaseClient()
    candidates = kb._hybrid_search(
        query_embedding=dense,
        query_sparse=sparse,
        top_k=10,
        filters='tenant_id == "tenant_default"',
    )

    print(f"召回候选数：{len(candidates)}")
    for i, c in enumerate(candidates[:4]):
        print(f"\n[{i + 1}] score={c['score']:.4f}  "
              f"来源：{c['metadata']['source_name']}")
        print(f"     {c['content'][:80]}...")




