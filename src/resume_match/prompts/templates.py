"""Prompt 模板集合。

设计原则:
- 明确 "Do NOT invent": 反幻觉(LLM 会脑补 JD 里没写的技能)
- 明确 "use null / empty list": 缺失字段有确定行为, 不让模型瞎填
- 只抽硬技能, 排除软技能: 下游 gap 分析才有意义
- 输出语言 = 中文: 界面是中文的, 自然语言字段(strengths/gaps/建议等)必须中文, 但从 JD 直接抽取的结构化字段和简历引用保持原文语言(见 _LANG_INSTRUCTION)
- Prompt 主体保持英文(指令精度更高), 只在末尾追加语言指令
"""

# 三个 prompt 共用的输出语言指令。追加到每个模板末尾。
_LANG_INSTRUCTION = """

IMPORTANT: Respond in Chinese (中文). All free-text fields in the output must be in Chinese, including:
- strengths, critical_gaps, minor_gaps (in analyze)
- rewritten text, rationale, should_apply_reason, interview_prep_focus, questions_to_ask_interviewer (in advise)

Structured fields extracted directly from the JD (role_title, company, hard_skills, nice_to_have) should PRESERVE the original language of the JD — if the JD is in Chinese, keep them in Chinese; if in English, keep in English. Do NOT translate these.

For skill_matches[].evidence, quote the resume text in its original language."""


EXTRACT_JD_PROMPT = """You are analyzing a job description (JD) to extract structured requirements.

Extract the following from the JD text below. Be precise and faithful to what's written:

- Do NOT invent skills or requirements not mentioned in the JD.
- If a field is not mentioned, use null (for optional fields) or an empty list.
- For hard_skills: extract concrete technical skills only (languages, frameworks, tools, protocols). Do not include soft skills.
- For nice_to_have: things explicitly marked as "加分项" / "preferred" / "bonus" / "plus".
- For key_responsibilities: 5-10 concrete bullet items describing the day-to-day work.

JD text:
---
{jd_text}
---

Extract now.""" + _LANG_INSTRUCTION


ANALYZE_PROMPT = """You are comparing a candidate's resume against a structured job description (JD).

JD (structured):
```json
{jd}
```

Resume (raw text):
---
{resume}
---

Your task:
1. For EACH hard_skill in the JD, determine if the resume shows: matched / partial / missing.
   - 'matched': resume explicitly demonstrates this skill (cite the evidence sentence).
   - 'partial': related but not exact (e.g., JD asks React, resume shows Vue — mark partial).
   - 'missing': no evidence in resume.

2. Calculate score_breakdown (投递成功率拆解, 总分100):
   a) tech_match (技术匹配度, /35分):
      - Based on hard_skills: matched=full points, partial=half points, missing=0
      - Example: 10 skills, 8 matched, 1 partial, 1 missing → ~30/35

   b) role_fit (职位种类倾向匹配度, /20分):
      - Does resume's tech stack direction align with JD's role_title and domain?
      - Example: Backend role but resume shows only frontend projects → low score
      - Example: Agent development role and resume has Agent projects → high score

   c) project_relevance (项目匹配度, /25分):
      - Project complexity and business scenarios vs JD's key_responsibilities
      - Multiple deep relevant projects → 22-25, one relevant → 12-15, unrelated → 0-8

   d) internship_match (实习经历匹配度, /10分):
      - Relevant internship experience bonus
      - Has relevant internship → 8-10, no internship or unrelated → 0-5

   e) bonus (加分项, /10分):
      - Open source contributions, papers, competitions, advanced skills beyond JD
      - Corresponds to 'strengths' field

   IMPORTANT: tech_match + role_fit + project_relevance + internship_match + bonus MUST equal overall_match_score

3. Identify 3-5 strengths where the candidate's resume goes ABOVE the JD requirements.

4. Identify critical_gaps (up to 5) that would materially affect whether to apply.

5. Give overall_match_score 0-100 (投递成功率):
   - 90+: strongly matched, almost certain interview
   - 70-89: solid match, worth applying
   - 50-69: stretch, some critical gaps
   - <50: significant mismatch

Be honest. Do not inflate the score to be encouraging.""" + _LANG_INSTRUCTION


ADVISE_PROMPT = """You are a career coach helping a job seeker tailor their resume for a specific JD.

JD (structured):
```json
{jd}
```

Gap Analysis:
```json
{gap}
```

Resume (raw text):
---
{resume}
---

Your task:
1. Decide should_apply:
   - 'strongly_recommend': match_score >= 80, no critical_gaps
   - 'worth_trying': 60-79, some gaps but strengths compensate
   - 'stretch': 50-59, apply only if user has time and confidence
   - 'not_recommended': <50 or has deal-breaking gaps

2. Suggest 3-5 bullet_rewrites:
   - Pick the resume bullets MOST relevant to this JD.
   - Rewrite each to emphasize keywords from the JD (only if the underlying experience is real).
   - IMPORTANT: Do NOT invent achievements. Only rephrase existing bullets to highlight relevant angles.
   - Include a 1-sentence rationale for each rewrite.

3. Suggest 3-5 interview_prep_focus items:
   - Based on the JD requirements + gaps, what tech topics should the candidate deep-review?
   - Be specific (e.g., "Study LangGraph checkpointer & thread persistence" not "Review LangGraph").

4. Suggest 2-3 questions_to_ask_interviewer:
   - Questions that show engagement AND help the candidate assess fit.
   - Avoid generic "what's your culture" questions.

## User's Similar Past Applications

{similar_histories_context}

Consider these past applications when generating advice:
- If the user has applied to similar roles before with consistent gaps (e.g., always missing "Milvus" or "gRPC"), acknowledge this pattern: "This is a recurring gap across your applications, worth investing time to learn."
- If a similar past application scored much lower/higher, briefly note what changed.
- Do NOT invent patterns if there's only 1 similar application; only comment on patterns with 2+ similar past entries.
- Include pattern observations in should_apply_reason and/or interview_prep_focus.""" + _LANG_INSTRUCTION
