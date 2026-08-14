
import asyncio
import uuid
from scripts.build_knowledge_base import build_pipeline

if __name__ == '__main__':

    # course_id = str(uuid.uuid4())  # 课程ID
    # document_id = str(uuid.uuid4())  # 文档ID
    '''
    课程ID：5abae311-f79f-4da1-a6ae-a11fcbebe25b
    文档ID：2f46bd48-6f2a-4b07-ab21-9ea86979a093
    '''
    course_id = '5abae311-f79f-4da1-a6ae-a11fcbebe25b' # 课程ID
    document_id = '2f46bd48-6f2a-4b07-ab21-9ea86979a093' # 文档ID

    # 启动异步流水线，构建知识库
    asyncio.run(build_pipeline(
        file_path='../tmp/Milvus索引原理.md',  # 文档路径
        course_id=course_id,  # 课程ID
        document_id=document_id,  # 文档ID
        tenant_id='tenant_default',  # 租户ID 默认 tenant_default
        version='7.9',  # 课程版本号 默认 1.0
        use_context=False,  # 是否启用上下文增强
    ))


    input('输入回车 ➡️ 开始查询验证：')
    import os
    from dotenv import load_dotenv
    load_dotenv("../.env.local") # 加载环境变量文件
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f"http://{os.getenv('MILVUS_HOST')}:"
            f"{os.getenv('MILVUS_PORT')}",
        db_name=os.getenv('MILVUS_DB_NAME'),
    )
    rows = client.query(
        collection_name=os.getenv('MILVUS_COLLECTION_NAME'),
        filter='tenant_id == "tenant_default"',
        output_fields=["id", "source_name", "chunk_index", "content"],
        limit=5,
    )
    print(f"查到 {len(rows)} 条（展示前5）：")
    for r in rows:
        print(f"  [{r['chunk_index']}] {r['source_name']} ")


