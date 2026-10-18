# 基于 python:3.14-slim，与项目部署基线保持一致。
FROM python:3.14-slim

# 从官方镜像复制 uv。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先安装依赖，便于源码变更时复用依赖层缓存。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 安装 Chromium 及其系统依赖。
RUN uv run --no-sync playwright install chromium --with-deps

# 复制源码并安装项目，使 python -m resume_match.cli 可用。
COPY src/ ./src/
RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1

# 运行时复用已安装的环境；docker run 可追加 --jd、--resume 等 CLI 参数。
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "resume_match.cli"]
