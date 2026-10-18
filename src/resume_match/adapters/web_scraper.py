"""JD 网页抓取：优先使用异步 HTTP，必要时回退到 Playwright。"""
import httpx
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,en-US;q=0.9,en;q=0.8",
}


def _clean_html_to_text(html: str) -> str:
    """用 BeautifulSoup 剥离噪音标签, 提取可见正文, 折叠连续空行。"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


async def fetch_and_extract(url: str, timeout: int = 15) -> str:
    """静态抓取: httpx 拿 HTML + BeautifulSoup 提取正文。

    抓不到就 raise(httpx 的 HTTPStatusError / 超时等), 由上层 fetch_jd 决定是否 fallback。
    """
    async with httpx.AsyncClient(
        headers=HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    return _clean_html_to_text(resp.text)


async def fetch_with_playwright(url: str, timeout: int = 30000) -> str:
    """动态抓取 fallback: 用 headless Chromium 等 JS 渲染完再抓正文。

    适用: Lever / Greenhouse / Workday 等 SPA 招聘页(静态 HTML 是空壳)。
    - 等 networkidle, 让前端把 JD 渲染出来。
    - 抓 body innerText(浏览器已渲染的可见文本), 不用再 BeautifulSoup。
    - 每次用完关掉 context/browser, 不复用(避免 session 污染)。
    - 不做主动反爬伪装(不装绕过插件、不用第三方服务), 只设常规 UA/语言。

    注意: Python 3.14 在 Windows 上的 asyncio 子进程有问题，使用同步 API + 线程池绕过。
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _sync_fetch(url: str, timeout: int) -> str:
        """同步版本的 Playwright 抓取，在线程池中运行避开 asyncio 子进程问题。"""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    user_agent=HEADERS["User-Agent"],
                    locale="zh-CN",
                    extra_http_headers={"Accept-Language": HEADERS["Accept-Language"]},
                )
                page = context.new_page()
                page.goto(url, timeout=timeout)
                page.wait_for_load_state("networkidle", timeout=timeout)
                text = page.evaluate("() => document.body.innerText")
                context.close()
                return text
            finally:
                browser.close()

    # 在线程池中运行同步版本
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        text = await loop.run_in_executor(executor, _sync_fetch, url, timeout)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
