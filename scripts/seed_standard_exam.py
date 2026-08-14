"""
把「标准试卷」（题目 + 正确答案 + 得分点）灌进数据库 exams / questions / scoring_points。
这是第 6 章批改前的「数据准备」——库里有了这份标准卷，load_questions_meta 才有东西可读。

- 用 SQLAlchemy + text()，复用项目的 AsyncSessionLocal（与课件正文节点代码同一种写法）
- 幂等：先按固定 exam_id 删除（级联清掉 questions/scoring_points），再插入
用法：直接运行scripts/seed_standard_exam.py
"""
import asyncio
import json

from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal

TENANT = "tenant_default"

# ── 固定测试用试卷考试ID──────────────
EXAM_ID = "e0000001-0000-0000-0000-000000000001"
# ── 固定测试用题目的ID──────────────
Q = {  # question_no -> 固定 question_id
    1: "e0000001-0000-0000-0000-0000000000a1",
    2: "e0000001-0000-0000-0000-0000000000a2",
    3: "e0000001-0000-0000-0000-0000000000a3",
    4: "e0000001-0000-0000-0000-0000000000a4",
    5: "e0000001-0000-0000-0000-0000000000a5",
}

# ── 试卷：Java 基础测试卷（5 题，总分 35）────────────────────
QUESTIONS = [
    # (no, type, content, correct_answer, score, knowledge_tag)
    # (序号, 题型, 题目内容, 答案, 分数, 知识点标签)
    (1, "single_choice",
     "下列关于 Java 中 final 关键字的说法，正确的是（ ）\n"
     "A. final 修饰的变量不能重新赋值\nB. final 修饰的方法可以被重写\n"
     "C. final 修饰的类可以被继承\nD. 以上都不对",
     "A", 5, "Java基础"),
    (2, "multi_choice",
     "下列关于 Java 集合的说法正确的是（多选）\n"
     "A. ArrayList 底层是数组\nB. LinkedList 支持随机访问\n"
     "C. HashMap 允许 null key\nD. HashSet 不允许重复元素",
     "ACD", 6, "Java集合"),
    (3, "judge", "判断：Java 中 int 类型的成员变量默认值是 0。", "正确", 4, "Java基础"),
    (4, "short_answer", "请解释 Spring IOC 的概念及其核心作用。", "", 10, "Spring IOC"),
    (5, "code",
     "编写一个 Java 方法 fib(int n)，计算斐波那契数列第 N 项（fib(0)=0, fib(1)=1）。",
     # 代码题答案存JSON形式的测试用例
     json.dumps([{"input": "10", "expected_output": "55"},
                 {"input": "1", "expected_output": "1"}], ensure_ascii=False),
     10, "算法-递归"),
]

# ── 第4题（简答）的得分点，共 10 分 ─────────────────────────────
SCORING_POINTS_4 = [
    ("正确说明 IOC 是「控制反转/依赖注入」的设计思想", 4),
    ("说明对象的创建与依赖管理交给 Spring 容器统一负责", 3),
    ("说明 IOC 的作用：降低耦合、便于测试与维护", 3),
]


async def main():
    async with AsyncSessionLocal() as session:
        # 先删试卷（级联清掉 questions/scoring_points）
        await session.execute(
            text("DELETE FROM exams WHERE id = :id"),
            {"id": EXAM_ID}
        )
        # 插入试卷
        await session.execute(
            text("""INSERT INTO exams 
            (id, tenant_id, title, description, is_active) 
            VALUES (:id, :tenant, :title, :desc, TRUE)"""
            ),
            {
                "id": EXAM_ID, "tenant": TENANT,
                "title": "Java 基础测试卷（第6章批改标准卷）",
                "desc": "单选/多选/判断/简答/代码五种题型，总分35",
            },
        )

        # 插入题目
        for no,qtype,content,correct,score,tag in QUESTIONS:
            await session.execute(
                text("""INSERT INTO questions 
                (id, tenant_id, exam_id, question_no, 
                question_type, content, correct_answer, 
                score, knowledge_tag) 
                VALUES 
                (:id, :tenant, :exam_id, :no, :qtype, 
                :content, :correct, :score, :tag)"""
                ),
                {
                    "id": Q[no], # 固定 question_id
                    "tenant": TENANT,
                    "exam_id": EXAM_ID,
                    "no": no, # 题目序号
                    "qtype": qtype,
                    "content": content,
                    "correct": correct,
                    "score": score,
                    "tag": tag,
                },
            )

        # 插入得分点
        for desc, pts in SCORING_POINTS_4:
            await session.execute(
                text("""INSERT INTO scoring_points 
                (question_id, point_desc, 
                point_score, is_active) 
                VALUES (:qid, :desc, :pts, TRUE)"""),
                {
                    "qid": Q[4], # 固定 question_id
                    "desc": desc, # 得分点描述
                    "pts": pts, # 得分点分数
                },
            )

        await session.commit()  # 提交落库

        # 自检
        total = (await session.execute(
            text("SELECT SUM(score) FROM questions "
                 "WHERE exam_id = :id"),
            {"id": EXAM_ID})).scalar()
        n_q = (await session.execute(
            text("SELECT COUNT(*) FROM questions "
                 "WHERE exam_id = :id"),
            {"id": EXAM_ID})).scalar()
        n_sp = (await session.execute(
            text("SELECT COUNT(*) FROM scoring_points "
                 "WHERE question_id = :qid"),
            {"qid": Q[4]})).scalar()

    print(f"✅ 已写入标准卷：exam_id={EXAM_ID}")
    print(f"\t题目 {n_q} 道，总分 {total}；简答题得分点 {n_sp} 个")
    print("\t EXAM_ID 设成上面这个，即可提交批改")


if __name__ == "__main__":
    asyncio.run(main())
