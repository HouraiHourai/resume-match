"""JD 抓取服务：获取 JD 纯文本。

关键决策:
- 不用 LLM，纯 IO + 简单启发式判断，快且零成本
- 分层抓取: httpx 静态优先（快、轻），失败或正文过短才 fallback 到 Playwright（慢、重）
- 失败时不 raise，而是往 state['errors'] 里塞错误信息，让上层决定如何处理
"""
from resume_match.state import AgentState
from resume_match.adapters import web_scraper

# httpx 抓到的正文短于此长度，视为"静态抓取无效"（SPA 空壳/被挡），触发 Playwright
_MIN_STATIC_LEN = 500
# 最终正文短于此长度，判定为抓取失败
_MIN_VALID_LEN = 200


async def _fetch_from_url(url: str) -> tuple[str, list[str]]:
    """URL 抓取: httpx 优先，失败或过短 fallback 到 Playwright。返回 (text, notes)。"""
    notes: list[str] = []

    # httpx 静态抓取
    static_text = ""
    try:
        static_text = await web_scraper.fetch_and_extract(url)
        if len(static_text) >= _MIN_STATIC_LEN:
            return static_text, notes
        notes.append(f"static fetch too short ({len(static_text)} chars), trying Playwright")
    except Exception as e:
        notes.append(f"static fetch failed ({e}), trying Playwright")

    # Playwright 动态抓取
    try:
        dynamic_text = await web_scraper.fetch_with_playwright(url)
        if len(dynamic_text) >= len(static_text):
            return dynamic_text, notes
        return static_text, notes
    except Exception as e:
        notes.append(f"Playwright fetch failed ({e})")
        return static_text, notes


async def fetch_jd(state: AgentState) -> dict:
    """获取 JD 纯文本。

    分支:
      1. jd_input 是 URL: httpx -> Playwright；仍拿不到足够正文则报错让用户手动粘贴
      2. jd_input 是纯文本(>= 200 字符): 直接用
      3. 支持 [公司: XXX] 前缀指定公司名（会在结构化服务中使用）
    """
    jd_input = state["jd_input"]
    company_override = None

    # 检查是否有 [公司: XXX] 前缀
    if jd_input.startswith("[公司:") or jd_input.startswith("[Company:"):
        lines = jd_input.split("\n", 1)
        if len(lines) == 2:
            # 提取公司名
            first_line = lines[0]
            if first_line.startswith("[公司:"):
                company_override = first_line[4:].strip()
                if company_override.endswith("]"):
                    company_override = company_override[:-1].strip()
            elif first_line.startswith("[Company:"):
                company_override = first_line[9:].strip()
                if company_override.endswith("]"):
                    company_override = company_override[:-1].strip()
            # 剩余部分作为实际输入
            jd_input = lines[1].strip()

    if jd_input.startswith(("http://", "https://")):
        # 检查是否是搜索列表页
        list_page_patterns = [
            "jobs?",  # Boss 直聘、智联等搜索页
            "pageMode=search",  # 智联搜索
            "query=",  # 通用搜索参数
            "/sou/",  # 51job 搜索
            "/zhaopin/",  # 通用招聘列表
            "/campus/post?",  # 腾讯音乐等校招列表
            "/campus/jobs",  # 校招职位列表
            "/social/jobs",  # 社招职位列表
            "/positions?",  # 职位列表
        ]

        if any(pattern in jd_input for pattern in list_page_patterns):
            return {
                "errors": [
                    "检测到搜索列表页 URL，请改为输入单个岗位详情页 URL。\n"
                    "或者：\n"
                    "1. 在浏览器中打开岗位详情页\n"
                    "2. 按 F12 打开开发者工具 → Console\n"
                    "3. 输入: copy(document.body.innerText)\n"
                    "4. 回车后，JD 文本已复制到剪贴板\n"
                    "5. 粘贴到这里进行分析"
                ]
            }

        text, notes = await _fetch_from_url(jd_input)
        if len(text) < _MIN_VALID_LEN:
            # 检测是否是已知的强防护网站
            protected_sites = {
                "zhipin.com": "Boss 直聘",
                "zhaopin.com": "智联招聘",
                "51job.com": "前程无忧",
            }
            site_name = None
            for domain, name in protected_sites.items():
                if domain in jd_input:
                    site_name = name
                    break

            if site_name:
                error_msg = (
                    f"{site_name} 有强验证保护，自动抓取无法获取内容。\n"
                    f"请手动复制 JD 文本：\n"
                    f"1. 在浏览器中打开岗位详情页\n"
                    f"2. 按 F12 打开开发者工具 → Console\n"
                    f"3. 输入: copy(document.body.innerText)\n"
                    f"4. 回车后，JD 文本已复制到剪贴板\n"
                    f"5. 粘贴到这里进行分析"
                )
            else:
                detail = "; ".join(notes) if notes else "no content"
                error_msg = (
                    f"JD fetch failed ({detail}). This site may be JS-rendered or anti-bot "
                    f"protected — please paste the JD text directly."
                )
            return {"errors": [error_msg]}
        result = {"jd_raw_text": text}
        if company_override:
            result["company_name"] = company_override
        return result

    text = jd_input  # 假设是原文
    if len(text) < _MIN_VALID_LEN:
        return {"errors": ["JD text too short (<200 chars). Provide the full JD."]}
    result = {"jd_raw_text": text}
    if company_override:
        result["company_name"] = company_override
    return result
