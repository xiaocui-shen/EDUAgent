# scripts/seed_interview_questions.py
# 向interview_questions表插入10道AI大模型开发工程师面试题
# 运行：scripts/seed_interview_questions.py

import sys, asyncio, json
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(".env.local")

from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal

TARGET_POSITION = "AI大模型开发工程师"

QUESTIONS = [
    {
        "content": "请解释 Transformer 的自注意力机制（Self-Attention）原理，并说明 Q、K、V 矩阵的含义和计算过程。",
        "difficulty": "medium",
        "tags": ["Transformer", "Self-Attention", "深度学习基础"],
    },
    {
        "content": "大模型微调有哪些主流方法？请对比 Full Fine-tuning、LoRA、QLoRA 的适用场景和资源消耗差异。",
        "difficulty": "medium",
        "tags": ["微调", "LoRA", "QLoRA", "大模型训练"],
    },
    {
        "content": "什么是 RAG（检索增强生成）？请描述其完整流程，并说明为什么 RAG 比直接 Prompt 更适合处理私域知识问答。",
        "difficulty": "easy",
        "tags": ["RAG", "向量检索", "知识库"],
    },
    {
        "content": "向量数据库的作用是什么？Faiss、Milvus、Chroma 各有什么特点，你会如何在生产环境中选型？",
        "difficulty": "medium",
        "tags": ["向量数据库", "Faiss", "Milvus", "Chroma"],
    },
    {
        "content": "LLM 推理时的 KV Cache 是什么机制？它如何减少重复计算，对推理速度有什么影响？",
        "difficulty": "hard",
        "tags": ["KV Cache", "推理优化", "大模型部署"],
    },
    {
        "content": "请解释 LangChain 中 Chain 和 Agent 的区别，并举例说明你会在什么场景下选择 Agent 而非固定 Chain。",
        "difficulty": "medium",
        "tags": ["LangChain", "Agent", "工程实践"],
    },
    {
        "content": "如何评估一个 RAG 系统的效果？你了解哪些 RAG 评测指标（如 Faithfulness、Answer Relevancy），如何用 RAGAS 进行评测？",
        "difficulty": "hard",
        "tags": ["RAG评测", "RAGAS", "Faithfulness"],
    },
    {
        "content": "Prompt Engineering 有哪些常用技巧？请对比 Zero-shot、Few-shot、Chain-of-Thought 各自的适用场景。",
        "difficulty": "easy",
        "tags": ["Prompt Engineering", "CoT", "Few-shot"],
    },
    {
        "content": "大模型部署时如何做量化（Quantization）？INT8 和 INT4 量化各有什么优缺点，对精度和速度的影响如何？",
        "difficulty": "hard",
        "tags": ["量化", "INT8", "INT4", "大模型部署"],
    },
    {
        "content": "LangGraph 与 LangChain 的 AgentExecutor 有什么区别？LangGraph 适合解决什么类型的问题，请举一个实际应用场景。",
        "difficulty": "medium",
        "tags": ["LangGraph", "状态机", "多步推理"],
    },
]


async def main():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for q in QUESTIONS:
                await session.execute(
                    text("""
                        INSERT INTO interview_questions
                            (content, difficulty, tags, target_position, tenant_id, is_active)
                        VALUES
                            (:content, :difficulty, :tags, :target_position, 'tenant_default', TRUE)
                    """),
                    {
                        "content":         q["content"],
                        "difficulty":      q["difficulty"],
                        "tags":            json.dumps(q["tags"], ensure_ascii=False),
                        "target_position": TARGET_POSITION,
                    },
                )

    # 验证写入
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT id, difficulty, LEFT(content, 30) AS preview FROM interview_questions WHERE target_position = :pos ORDER BY created_at"),
            {"pos": TARGET_POSITION},
        )
        rows = result.mappings().all()

    print(f"\n插入成功，共 {len(rows)} 条题目：")
    for i, r in enumerate(rows, 1):
        print(f"  {i:2d}. [{r['difficulty']:6s}] {r['preview']}...")


asyncio.run(main())
