"""服务器启动脚本，解决 Windows + Python 3.14 的 asyncio 子进程问题。"""
import sys
import asyncio

# Windows 上必须使用 ProactorEventLoop 才能创建子进程
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "resume_match.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
