# scripts/init_milvus.py
# 执行：python scripts/init_milvus.py
import os
from pymilvus import MilvusClient, DataType
from backend.config import get_settings


MILVUS_URI = (f"http://{get_settings().milvus_host}:"
              f"{get_settings().milvus_port}")
VECTOR_DIM = 1024                  # BGE-M3 稠密向量维度
COLLECTION_NAME = "knowledge_base" # 集合名字


def build_schema(client: MilvusClient):
    """构建集合 schema：稠密 + 稀疏双向量 + 标量字段（含 tenant_id）。"""
    schema = client.create_schema(
        auto_id=False, enable_dynamic_field=True)
    schema.add_field(
        "id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field( # 稠密
        "embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field( # 稀疏
        "sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field( # 本本内容
        "content", DataType.VARCHAR, max_length=4096)
    schema.add_field( # 多租户隔离
        "tenant_id", DataType.VARCHAR, max_length=64) 
    schema.add_field( # 分块序号
        "chunk_index", DataType.INT64)
    schema.add_field( # 文档ID
        "document_id", DataType.VARCHAR, max_length=64)
    schema.add_field( # 课程ID
        "course_id", DataType.VARCHAR, max_length=64)
    schema.add_field( # 来源标注
        "source_name", DataType.VARCHAR, max_length=256)
    schema.add_field( # 分块类型：text/code/table
        "chunk_type", DataType.VARCHAR, max_length=32)
    schema.add_field( # 版本号
        "version", DataType.VARCHAR, max_length=32)
    schema.add_field( # 更新时间戳
        "updated_at", DataType.INT64)
    return schema


def build_index_params(client: MilvusClient):
    """构建索引：稠密 HNSW、稀疏 SPARSE_INVERTED、标量 INVERTED。"""
    index_params = client.prepare_index_params() # 索引参数对象
    # 稠密向量索引：HNSW + COSINE（语义相似度）
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW", # 索引类型，图
        metric_type="COSINE", # 距离度量
        # 图中每个节点有16个邻居边 256是搜索范围
        params={"M": 16, "efConstruction": 256}
    )
    # 稀疏向量索引：SPARSE_INVERTED_INDEX + IP（内积）
    # drop_ratio_build 丢弃最低20%权重的token省存储
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX", # 索引类型，倒排索引
        metric_type="IP", # 距离度量
        params={"drop_ratio_build": 0.2}
    )
    # 标量字段：INVERTED 索引，加速 filter 过滤
    index_params.add_index(
        field_name="tenant_id",
        index_type="INVERTED" # 索引类型，倒排索引
    )
    index_params.add_index(
        field_name="course_id",
        index_type="INVERTED" # 索引类型，倒排索引
    )
    return index_params


def main():
    print(f"连接 Milvus：{MILVUS_URI}")
    client = MilvusClient(
        uri=MILVUS_URI,
        db_name=get_settings().milvus_db_name, # Milvus数据库名
    )

    if client.has_collection(COLLECTION_NAME):
        print(f"🗑️  删除旧集合 '{COLLECTION_NAME}'...")
        client.drop_collection(COLLECTION_NAME)

    # create_collection 传 index_params 会一并建索引并加载
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=build_schema(client),
        index_params=build_index_params(client),
    )
    print(f"✅ 集合 '{COLLECTION_NAME}' 创建完成（含索引，已加载）")
    print("当前集合：", client.list_collections()[0])
    print("⚠️  集合已重建，原有数据已清空，请重新运行 build_knowledge_base.py 导入。")


if __name__ == "__main__":
    main()
