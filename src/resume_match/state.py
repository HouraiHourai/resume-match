"""State 与结构化输出 schema 定义。

设计分工:
- AgentState 用 TypedDict: 它是节点之间传递的"数据黑板", 每个节点只部分更新, 运行时就是普通 dict, 轻量、灵活
- LLM 产出的字段(JDRequirements / GapAnalysis / ResumeAdvice)用 Pydantic BaseModel: 需要运行时强校验(反幻觉), .with_structured_output() 也依赖 Pydantic schema
"""
from typing import Optional, Annotated, Any
from operator import add

from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------- Pydantic Schemas (LLM 结构化输出必须符合) ----------

class JDRequirements(BaseModel):
    """从 JD 抽取的结构化需求。"""
    role_title: str = Field(description="岗位名称,例如 'AI Agent 后端工程师'")
    company: Optional[str] = Field(default=None, description="公司名称,如未提及则 null")
    location: Optional[str] = Field(default=None, description="工作地点")
    seniority: str = Field(
        description="资历要求: 'intern' / 'junior' / 'mid' / 'senior' / 'unspecified'"
    )

    hard_skills: list[str] = Field(
        description="必须掌握的硬技能, 如 ['Python', 'LangGraph', 'Docker']"
    )
    nice_to_have: list[str] = Field(
        default_factory=list, description="加分项技能"
    )
    domain: Optional[str] = Field(
        default=None, description="业务领域, 如 'FinTech' / 'E-commerce' / 'Enterprise SaaS'"
    )
    years_of_experience: Optional[str] = Field(
        default=None, description="经验年限要求, 例如 '1-3年' / '3-5年' / '应届'"
    )
    education: Optional[str] = Field(
        default=None, description="学历要求, 例如 '本科' / '硕士优先'"
    )
    salary_range: Optional[str] = Field(
        default=None, description="薪资范围, 例如 '15-25K·14薪'"
    )
    key_responsibilities: list[str] = Field(
        description="核心工作职责列表, 5-10 条"
    )


class SkillMatch(BaseModel):
    """单个技能的匹配情况。"""
    skill: str
    status: str = Field(description="'matched' / 'partial' / 'missing'")
    evidence: Optional[str] = Field(
        default=None,
        description="如果 matched, 引用简历里的哪句话作为证据"
    )


class ScoreBreakdown(BaseModel):
    """投递成功率分数拆解 (总分100)"""
    tech_match: int = Field(
        ge=0, le=35,
        description="技术匹配度 /35分: 基于硬技能 matched/partial/missing 占比"
    )
    role_fit: int = Field(
        ge=0, le=20,
        description="职位种类倾向匹配度 /20分: 简历项目技术栈方向与岗位方向契合度"
    )
    project_relevance: int = Field(
        ge=0, le=25,
        description="项目匹配度 /25分: 项目复杂度和业务场景与 JD 职责契合度"
    )
    internship_match: int = Field(
        ge=0, le=10,
        description="实习经历匹配度 /10分: 相关实习经历的加成"
    )
    bonus: int = Field(
        ge=0, le=10,
        description="加分项 /10分: 开源贡献/论文/竞赛等超预期亮点"
    )


class GapAnalysis(BaseModel):
    """简历 vs JD 的差距分析。"""
    overall_match_score: int = Field(
        description="投递成功率总分 0-100 分", ge=0, le=100
    )
    score_breakdown: ScoreBreakdown = Field(
        description="成功率分数拆解: tech_match + role_fit + project_relevance + internship_match + bonus = overall_match_score"
    )
    skill_matches: list[SkillMatch] = Field(
        description="每个硬技能的匹配情况"
    )
    strengths: list[str] = Field(
        description="简历相对于 JD 的强项, 3-5 条"
    )
    critical_gaps: list[str] = Field(
        description="关键缺口 (会影响是否值得投递), 最多 5 条"
    )
    minor_gaps: list[str] = Field(
        default_factory=list, description="次要缺口"
    )


class BulletRewrite(BaseModel):
    """针对简历某条 bullet 的改写建议。"""
    original: str = Field(description="简历原文摘录 (不改动)")
    rewritten: str = Field(description="改写后的版本")
    rationale: str = Field(description="为什么这样改, 1-2 句话")


class SimilarHistory(BaseModel):
    """检索到的一条相似历史投递(给 advise 参考 + 前端展示)。"""
    history_id: int
    jd_summary: str
    match_score: Optional[int] = None
    should_apply: Optional[str] = None
    similarity_score: float = Field(ge=0, le=1, description="混合召回及业务特征融合分 0-1，不是概率或余弦相似度")
    created_at: str


class ResumeAdvice(BaseModel):
    """给用户的最终建议。"""
    should_apply: str = Field(
        description="'strongly_recommend' / 'worth_trying' / 'stretch' / 'not_recommended'"
    )
    should_apply_reason: str = Field(description="为什么建议这样, 2-3 句话")
    bullet_rewrites: list[BulletRewrite] = Field(
        description="针对当前 JD 的简历 bullet 改写建议, 3-5 条"
    )
    interview_prep_focus: list[str] = Field(
        description="岗位准备重点的技术点, 3-5 条"
    )
    questions_to_ask_interviewer: list[str] = Field(
        default_factory=list,
        description="可询问招聘方的问题, 2-3 条"
    )


# ---------- LangGraph State ----------

class AgentState(TypedDict, total=False):
    """整个工作流共享的 state。

    total=False: 初始 State 允许只填部分字段(如只给 jd_input/resume_raw_text), 其余字段由各节点跑完后逐步填上。
    """
    # 输入
    jd_input: str              # 原始输入: URL 或纯文本
    resume_raw_text: str       # 从 PDF 解析出的简历纯文本

    # 中间状态
    jd_raw_text: str           # pdf 输出
    company_name: Optional[str]  # 用户通过 [公司: XXX] 指定的公司名, 优先于 LLM 提取

    # 结构化产出
    jd_structured: JDRequirements
    gap_analysis: GapAnalysis
    advice: ResumeAdvice

    # RAG: advise 节点检索到的相似历史投递(给前端展示)
    similar_histories: list[SimilarHistory]

    # 元信息
    # reducer 拼接各节点上报的错误，保留前序节点的错误记录。
    errors: Annotated[list[str], add]

    # ReAct 状态

    # 推理历史（完整的 Thought-Action-Observation 序列）
    react_history: Annotated[list[dict], add]

    # 当前循环状态
    current_step: int  # 当前是第几步
    max_steps: int  # 最大步数（防止死循环，默认 10）

    # Agent 决策
    next_action: Optional[str]  # 下一步要执行的 action
    action_input: Optional[dict]  # action 的输入参数

    # 工具调用结果缓存
    tool_results: dict[str, Any]

    # 公司风险信息（check_company 工具输出）
    company_risk: Optional[dict]

    # 是否完成
    is_finished: bool
