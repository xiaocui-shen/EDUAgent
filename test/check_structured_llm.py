# test/check_structured_llm.py
# 结构化输出精准测试：验证 Qwen3.7-plus 的 function_calling 是否正常工作。
# 运行方式：python test/check_structured_llm.py
# 依赖：阿里云百炼 API Key（环境变量 AliApiKey）

import asyncio
import sys
import os

# 确保项目根目录在 sys.path 里（这样能 import backend.*）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from backend.core.llm_factory import LLMFactory


# ══════════════════════════════════════════════
# 测试用 Pydantic Schema（从简单到复杂，逐级验证）
# ══════════════════════════════════════════════

# ── Level 1：最简 Schema（纯平字段）──
class StudentBasic(BaseModel):
    """最简结构化输出：验证 function_calling 基本能力"""
    name: str = Field(description="学生姓名")
    age: int = Field(description="学生年龄")
    major: str = Field(description="专业方向")


# ── Level 2：带列表的 Schema ──
class SkillItem(BaseModel):
    """单条技能"""
    skill_name: str = Field(description="技能名称")
    level: str = Field(description="熟练度：精通/熟练/了解")
    years: float = Field(description="使用年限")


class StudentSkills(BaseModel):
    """学生技能清单：验证 list[嵌套对象] 的解析"""
    name: str = Field(description="学生姓名")
    skills: list[SkillItem] = Field(description="技能列表")
    summary: str = Field(description="一句话评价该学生的技术水平")


# ── Level 3：直接复用项目真实 Schema ──
from backend.agents.resume.state import (
    ResumeStructured, EducationItem, ProjectItem
)
from backend.agents.exam.state import (
    WeakPoint, WeakPointsReport, ScoringPointResult, SubjectiveReviewResult
)


# ══════════════════════════════════════════════
# 测试函数
# ══════════════════════════════════════════════

async def check_1_simple():
    """Level 1：最简 Schema，验证 function_calling 基本能力"""
    print("\n" + "=" * 60)
    print("📝 测试 1：最简结构化输出（StudentBasic）")
    print("=" * 60)

    structured_llm = LLMFactory.get_structured_llm("resume", StudentBasic)
    result = await structured_llm.ainvoke([
        {"role": "user", "content": "李明，22岁，计算机科学与技术专业"}
    ])

    print(f"   name  = {result.name}")
    print(f"   age   = {result.age}")
    print(f"   major = {result.major}")

    # 基本校验
    assert isinstance(result.name, str) and len(result.name) > 0, "name 应为非空字符串"
    assert isinstance(result.age, int), "age 应为整数"
    assert isinstance(result.major, str) and len(result.major) > 0, "major 应为非空字符串"
    print("   ✅ 通过")
    return True


async def check_2_nested_list():
    """Level 2：嵌套列表 Schema，验证 list[嵌套对象]"""
    print("\n" + "=" * 60)
    print("📝 测试 2：嵌套列表结构化输出（StudentSkills）")
    print("=" * 60)

    structured_llm = LLMFactory.get_structured_llm("resume", StudentSkills)
    result = await structured_llm.ainvoke([
        {"role": "user", "content":
         "王强，5年开发经验。精通Java（5年）、熟练使用Spring Boot（4年）、"
         "了解Redis（1年）。整体来说是一个后端开发能力比较扎实的候选人。"}
    ])

    print(f"   name    = {result.name}")
    print(f"   技能数  = {len(result.skills)}")
    for i, s in enumerate(result.skills):
        print(f"     [{i+1}] {s.skill_name} | {s.level} | {s.years}年")
    print(f"   summary = {result.summary}")

    # 校验
    assert isinstance(result.skills, list) and len(result.skills) > 0, "skills 应为非空列表"
    assert isinstance(result.skills[0], SkillItem), "列表元素应为 SkillItem"
    assert isinstance(result.summary, str) and len(result.summary) > 0, "summary 应为非空字符串"
    print("   ✅ 通过")
    return True


async def check_3_resume_real_schema():
    """Level 3：复用项目真实 Schema（ResumeStructured）"""
    print("\n" + "=" * 60)
    print("📝 测试 3：项目真实 Schema（ResumeStructured 简历提取）")
    print("=" * 60)

    structured_llm = LLMFactory.get_structured_llm("resume", ResumeStructured)
    result = await structured_llm.ainvoke([
        {"role": "user", "content":
         "简历内容：\n"
         "姓名：赵六，手机：13800138000，邮箱：zhaoliu@qq.com\n"
         "求职意向：Java后端开发\n"
         "教育经历：武汉大学 计算机科学与技术 本科 2019.09-2023.06 GPA 3.5\n"
         "技术栈：Java, Spring Boot, MySQL, Redis, Docker\n"
         "项目经历：在线商城系统 | 后端开发 | 2022.03-2022.09 | "
         "技术栈：Spring Boot, Redis, MySQL | "
         "负责商品模块和订单模块开发，使用Redis缓存热点商品数据，QPS提升200%\n"
         "证书：CET-6\n"
         "自我评价：热爱编程，有良好的团队协作能力"}
    ])

    print(f"   name           = {result.name}")
    print(f"   target_position= {result.target_position}")
    print(f"   教育经历数      = {len(result.education)}")
    if result.education:
        e = result.education[0]
        print(f"     学校={e.school} 专业={e.major} 学历={e.degree}")
    print(f"   技能列表        = {result.skills_list}")
    print(f"   项目经历数      = {len(result.projects)}")
    if result.projects:
        p = result.projects[0]
        print(f"     项目={p.name} 角色={p.role} 技术栈={p.tech_stack}")
        print(f"     亮点={p.highlights}")

    # 校验
    assert isinstance(result.name, str) and len(result.name) > 0
    assert isinstance(result.education, list) and len(result.education) > 0
    assert isinstance(result.projects, list) and len(result.projects) > 0
    assert isinstance(result.skills_list, list) and len(result.skills_list) > 0
    print("   ✅ 通过")
    return True


async def check_4_exam_review():
    """Level 4：考试批改 Schema（SubjectiveReviewResult）"""
    print("\n" + "=" * 60)
    print("📝 测试 4：考试批改结构化输出（SubjectiveReviewResult）")
    print("=" * 60)

    structured_llm = LLMFactory.get_structured_llm("exam_subjective", SubjectiveReviewResult)
    result = await structured_llm.ainvoke([
        {"role": "user", "content":
         "题目ID: Q001\n"
         "题目：请解释 Spring IOC 和 AOP 的概念及其作用。\n"
         "满分：10分\n"
         "得分点：\n"
         "  - P01（4分）：答出 IOC（控制反转）的概念\n"
         "  - P02（3分）：答出 AOP（面向切面编程）的概念\n"
         "  - P03（3分）：答出 IOC 和 AOP 的实际作用\n\n"
         "学员答案：\n"
         "IOC就是控制反转，把对象的创建交给Spring容器管理，不用自己new了。"
         "AOP是面向切面编程，可以把日志、事务等公共逻辑抽出来统一处理。"
         "IOC的作用是解耦，AOP的作用是不侵入代码的情况下增加功能。"}
    ])

    print(f"   question_id   = {result.question_id}")
    print(f"   total_score   = {result.total_score}/{result.full_score}")
    print(f"   confidence    = {result.confidence}")
    print(f"   得分点明细：")
    for pr in result.point_results:
        status = "✅" if pr.earned else "❌"
        print(f"     {status} {pr.point_desc} | 得{pr.earned} | "
              f"证据:{pr.evidence[:30]}... | 失分:{pr.missing[:30]}...")
    print(f"   整体评语      = {result.overall_comment}")

    # 校验
    assert isinstance(result.total_score, int)
    assert isinstance(result.full_score, int)
    assert isinstance(result.point_results, list) and len(result.point_results) > 0
    assert isinstance(result.confidence, float)
    print("   ✅ 通过")
    return True


async def check_5_weak_points():
    """Level 5：薄弱点分析 Schema（WeakPointsReport）"""
    print("\n" + "=" * 60)
    print("📝 测试 5：薄弱点分析结构化输出（WeakPointsReport）")
    print("=" * 60)

    structured_llm = LLMFactory.get_structured_llm("exam_subjective", WeakPointsReport)
    result = await structured_llm.ainvoke([
        {"role": "user", "content":
         "学员张三的批改结果如下：\n"
         "Spring IOC 相关题目 3 道，错了 2 道（Q001、Q003）\n"
         "Spring AOP 相关题目 2 道，错了 1 道（Q005）\n"
         "MySQL 索引相关题目 2 道，全对\n"
         "Redis 缓存相关题目 3 道，错了 2 道（Q008、Q010）\n"
         "请分析该学员的知识薄弱点并给出复习建议。"}
    ])

    print(f"   薄弱点数 = {len(result.weak_points)}")
    for wp in result.weak_points:
        print(f"     📌 {wp.tag} | 错{wp.wrong_count}/{wp.total_count} "
              f"| 题号:{wp.question_nos}")
        print(f"        建议: {wp.suggestion}")
    print(f"   整体评价 = {result.overall_summary}")

    # 校验
    assert isinstance(result.weak_points, list)
    assert isinstance(result.overall_summary, str) and len(result.overall_summary) > 0
    print("   ✅ 通过")
    return True


# ══════════════════════════════════════════════
# 主测试流程
# ══════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("🔬 结构化输出精准测试（Qwen3.7-plus via function_calling）")
    print("=" * 60)

    # 先清缓存，确保用最新配置创建模型
    LLMFactory.clear_cache()

    results = {}
    tests = [
        ("1_最简Schema",       check_1_simple),
        ("2_嵌套列表Schema",    check_2_nested_list),
        ("3_项目真实简历Schema", check_3_resume_real_schema),
        ("4_考试批改Schema",    check_4_exam_review),
        ("5_薄弱点分析Schema",  check_5_weak_points),
    ]

    for name, check_fn in tests:
        try:
            results[name] = await check_fn()
        except Exception as e:
            print(f"   ❌ 失败: {type(e).__name__}: {e}")
            results[name] = False

    # 汇总
    print("\n" + "=" * 60)
    print("📊 结构化输出测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status} | {name}")

    total = len(results)
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  总计: {passed_count}/{total} 通过")
    print("=" * 60)

    # 清缓存
    LLMFactory.clear_cache()


if __name__ == "__main__":
    asyncio.run(main())
