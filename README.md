# ResumeMatch

**简体中文** | [English](README.en.md)

AI Agent 驱动的简历-JD 匹配分析系统。
本项目受到 [JD-Scout](https://github.com/AlanYy-42/JD-Scout) 的启发，在其 MIT 许可的开源实现基础上，进一步扩展了 Agent 编排、混合检索、缓存、模型降级、流式接口和离线评测等能力。

## 界面展示

![界面展示](assets/demo.png)

上传 PDF 简历并输入岗位描述或链接，获取匹配分析与投递建议。系统先评估岗位本身的质量，只有信息充分、要求合理的岗位才继续生成简历改写。截图使用合成示例数据。

## Agent 核心执行流程

![Agent 核心执行流程](assets/agent-flow.svg)

Agent 通过推理与工具执行循环完成分析；建议生成阶段检索历史记录，并将相关上下文交给模型生成建议。

## 功能特性

### 核心功能

- **智能 JD 抓取** - 支持 URL 输入或直接粘贴文本，自动识别并抓取职位描述
- **结构化分析** - 自动提取岗位要求、技能清单、工作职责等关键信息
- **岗位质量判断** - 根据 JD 原文评估职责清晰度、要求合理性和招聘风险；质量不高或信息不明确时仍生成建议，仅跳过改写
- **技能匹配评估** - 逐项对比简历与 JD 要求，给出 matched/partial/missing 评级
- **差距分析** - 识别技能缺口和优势项，输出综合匹配评分（0–100 分）
- **简历优化建议** - 仅在岗位质量为 `high` 时基于真实经历生成改写建议
- **岗位准备建议** - 提供技术准备重点和可询问招聘方的问题

### 技术亮点

- **ReAct Agent 架构** - 自主推理和动态工具调用
- **混合 RAG 检索** - 向量检索（Chroma）+ BM25 关键词匹配
- **混合排序** - 双路独立召回 + 归一化 RRF 排名融合 + 业务特征精排
- **多模型降级** - 结构化业务调用支持有限重试和备用模型降级
- **实时流式输出** - Server-Sent Events 推送进度

## 技术栈

### 后端

- **Agent 框架**: LangGraph 0.2+, LangChain 0.3+
- **LLM**: DeepSeek V4 Flash（主模型）+ Qwen 3.8 Max（结构化业务调用降级），模型标识见 `src/resume_match/config.py`
- **Embedding**: BGE-M3 (SiliconFlow API)
- **向量数据库**: Chroma 0.5+
- **文本检索**: BM25 (rank-bm25)
- **Web 框架**: FastAPI + uvicorn
- **PDF 解析**: pdfplumber
- **网页抓取**: httpx + Playwright

### 前端

- **技术栈**: 原生 HTML/CSS/JavaScript
- **实时通信**: Server-Sent Events (SSE)
- **Markdown 渲染**: Marked.js

### 数据存储

- **历史记录**: SQLite（默认 `~/.resume_match/history.db`），作为历史数据的事实源
- **向量索引**: Chroma（默认 `~/.resume_match/chroma`）
- **BM25 索引**: pickle 持久化（默认 `~/.resume_match/bm25_index.pkl`）

## 快速开始

### 环境要求

- Python 3.14（`>=3.14,<3.15`）
- 推荐使用 uv 包管理器

### 安装步骤

1. **克隆项目**

   将 `<repository-url>` 替换为本仓库的 GitHub 地址。

   ```bash
   git clone <repository-url> resume_match
   cd resume_match
   ```

2. **安装依赖**

   ```bash
   # 使用 uv (推荐)
   uv sync --locked

   # 或使用 pip
   pip install -e .
   ```

3. **安装浏览器驱动**

   ```bash
   uv run playwright install chromium
   ```

   使用 pip 安装依赖时，在对应 Python 环境中运行 `python -m playwright install chromium`。

4. **配置环境变量**

   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入以下 API keys
   ```

   Windows PowerShell 可使用：

   ```powershell
   Copy-Item .env.example .env
   ```

### 环境变量配置

按默认主模型、备用模型和 Embedding 服务配置以下 API key：

```bash
# DeepSeek API (主模型)
DEEPSEEK_API_KEY=your_deepseek_api_key

# 阿里云 DashScope API (备用模型)
DASHSCOPE_API_KEY=your_dashscope_api_key

# SiliconFlow API (BGE-M3 嵌入模型)
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

可选配置：

```bash
# 未显式指定模型的调用使用此默认值，例如公司检查
DEFAULT_MODEL=deepseek

# 进程内 Embedding 缓存；任一项为 0 时关闭
EMBEDDING_CACHE_MAX_ENTRIES=256
EMBEDDING_CACHE_TTL_SECONDS=3600
```

Agent 推理、JD 提取、岗位质量评估、差距分析和建议生成目前显式使用 DeepSeek，`DEFAULT_MODEL` 不会切换这些调用。模型标识和服务地址在 `src/resume_match/config.py` 中配置。

存储位置可通过 `SQLITE_PATH`、`CHROMA_DIR` 和 `BM25_INDEX_PATH` 设置。自定义时请使用绝对路径；不设置时使用上面的用户目录默认值。环境变量中的 `~` 不会自动展开。

### 启动服务

```bash
# 使用 uv
uv run uvicorn resume_match.api.server:app --reload --port 8000

# 或直接使用已安装的 uvicorn
uvicorn resume_match.api.server:app --reload --port 8000

# 服务默认运行在 http://localhost:8000
```

## 使用指南

### Web 界面

1. 打开浏览器访问 http://localhost:8000
2. 上传简历 PDF
3. 输入 JD URL 或粘贴 JD 文本
4. 点击"开始分析"
5. 实时查看分析进度和结果

直接粘贴 JD 时至少需要 200 个字符。输入岗位 URL 时公司名称留空；粘贴文本时可以填写公司名称。当前网页界面为中文。

历史报告上方的“本次分析资料”支持查看、展开和复制原始 JD 文本，或复制、打开输入的岗位 URL，同时显示分析时间。资料区不显示简历 PDF。当前只保存 URL 输入，不保存当时抓取的网页正文；页面会提示原网页可能变化。

![历史报告的分析资料](assets/history-materials.png)

识别到公司名称后，Agent 会在生成建议前完成公司背景检查。公司栏显示公司名称、企业类型和简短分析正文，不显示重复标签或“简要分析：”前缀。知名企业也生成具体业务背景；模型不可用时，仅返回名称匹配线索并明确提示详细分析不可用。没有保存背景结果的旧记录显示暂无结果，重新分析后才会生成新内容。公司背景来自模型知识或名单线索，不等同于联网风险调查。

| 质量等级 | 分析与建议 | 简历改写 | 网页展示 |
| --- | --- | --- | --- |
| `high` | 正常生成 | 正常生成 | 隐藏岗位质量报告 |
| `low` | 继续生成投递建议、准备重点和核实问题 | 跳过 | 显示风险依据与建议 |
| `uncertain` | 继续生成投递建议、准备重点和核实问题 | 跳过 | 标题为“JD 描述不够明确”，显示问题说明圆点列表及招聘方核实问题 |

![描述不明确时的结果](assets/jd-uncertain.png)

`uncertain` 卡片以圆点列表逐条分行展示问题说明，每条以“。”结尾，不显示线索前缀或 JD 原文引用；没有问题说明时显示空状态提示。保留卡片内的“建议向招聘方核实”问题列表，以及下方的投递建议、准备重点和招聘方问题。

### CLI 模式

```bash
uv run python -m resume_match.cli \
    --jd "$(cat examples/jd-high.txt)" \
    --resume tests/fixtures/sample_resume.pdf \
    --output result.json \
    --output-md result.md
```

CLI 支持 JSON 和 Markdown 报告输出，不会自动存档到 Web 历史记录。

以上为 Bash 示例。PowerShell 单行示例：

```powershell
uv run python -m resume_match.cli --jd (Get-Content examples/jd-high.txt -Raw -Encoding utf8) --resume tests/fixtures/sample_resume.pdf --output result.json --output-md result.md
```

`examples/` 提供完整的合成 JD，可用于复现三种质量分支；这些命令会调用实际模型服务。

### Docker

配置好 `.env` 后运行：

```bash
docker compose up --build
```

默认映射端口为 8000，历史记录存放在本机 `~/.resume_match/`。

## 项目结构

### 模块结构图

下图展示主项目的模块职责与调用关系；`tools/` 中只有五个 Agent 工具，`tools/core/` 提供接口和注册机制。

```mermaid
flowchart TB
    subgraph entry["入口与界面"]
        WEB["web/static/<br/>页面、结果展示、历史记录"]
        API["api/<br/>FastAPI 与 SSE 接口"]
        CLI["cli.py<br/>命令行入口"]
        WEB --> API
    end

    subgraph orchestration["Agent 编排"]
        GRAPH["graph.py<br/>LangGraph 主图"]
        AGENT["nodes/agent.py<br/>根据状态选择下一步"]
        EXECUTOR["nodes/tool_executor.py<br/>执行工具并记录观察"]
        CORE["tools/core/<br/>base.py 接口 · registry.py 注册"]
        GRAPH --> AGENT
        AGENT -->|选择工具| EXECUTOR
        EXECUTOR -->|观察结果与状态更新| AGENT
        EXECUTOR -->|查找工具| CORE
    end

    subgraph business["tools/ · 五个业务工具"]
        EXTRACT["extract_jd.py<br/>获取并结构化 JD"]
        QUALITY["assess_jd_quality.py<br/>评估岗位质量"]
        COMPANY["check_company.py<br/>公司背景分析"]
        GAP["analyze_gap.py<br/>匹配与差距分析"]
        ADVICE["generate_advice.py<br/>建议与简历改写"]
    end

    subgraph support["业务服务与数据支持"]
        SERVICES["services/<br/>JD 处理、质量评估、匹配和建议服务"]
        ADAPTERS["adapters/<br/>PDF 解析 · 网页抓取"]
        RAG["rag/<br/>混合检索、排序与嵌入缓存"]
        HISTORY["history/<br/>SQLite 历史记录"]
        MODEL["config.py · 模型服务<br/>模型配置与调用"]
    end

    API --> GRAPH
    CLI --> GRAPH
    API -->|解析简历| ADAPTERS
    CLI -->|解析简历| ADAPTERS
    CORE --> EXTRACT
    CORE --> QUALITY
    CORE --> COMPANY
    CORE --> GAP
    CORE --> ADVICE
    EXTRACT --> SERVICES
    QUALITY --> SERVICES
    GAP --> SERVICES
    ADVICE --> SERVICES
    COMPANY --> MODEL
    SERVICES --> MODEL
    SERVICES -->|获取网页正文| ADAPTERS
    SERVICES -->|建议生成前检索相似记录| RAG
    API -->|保存、读取与删除历史| HISTORY
```

图中的工具连线表示调用关系，不表示五个工具并行运行。实际执行顺序由 Agent 和工具前置条件决定；只有岗位质量为 `high` 时生成简历改写，`low / uncertain` 仍生成投递与准备建议。

### 目录结构

```text
resume_match/
├── README.md             # 中文说明与 Mermaid 架构图
├── README.en.md          # 英文说明
├── assets/               # 合成截图与 SVG 流程图
├── examples/             # 三种质量分支的合成 JD
├── .github/workflows/    # GitHub Actions 离线检查
├── src/resume_match/
│   ├── api/              # FastAPI 服务端
│   ├── nodes/            # LangGraph 图节点
│   │   ├── agent.py      # Agent 推理节点
│   │   └── tool_executor.py # 工具执行节点
│   ├── services/         # 工具调用的业务服务
│   │   ├── jd_fetcher.py     # JD 抓取
│   │   ├── jd_extractor.py  # 结构化提取
│   │   ├── jd_quality.py    # 岗位质量评估与证据校验
│   │   ├── gap_analyzer.py  # 差距分析
│   │   ├── adviser.py       # 建议生成
│   │   └── structured_llm.py # 结构化 LLM 重试/降级
│   ├── rag/              # RAG 检索系统
│   │   ├── embedder.py   # 嵌入模型
│   │   ├── embedding_cache.py # 嵌入缓存
│   │   ├── vector_store.py   # 向量库
│   │   ├── bm25_store.py     # BM25 索引
│   │   ├── coarse_ranker.py  # 粗排
│   │   ├── fine_ranker.py    # 精排
│   │   └── retriever.py      # 检索器
│   ├── tools/            # 5 个 Agent 可调用工具（__init__.py 仅用于声明包）
│   │   ├── extract_jd.py        # 获取并结构化 JD
│   │   ├── assess_jd_quality.py # 评估岗位质量
│   │   ├── check_company.py     # 公司背景与简要分析
│   │   ├── analyze_gap.py       # 简历匹配与差距分析
│   │   ├── generate_advice.py   # 投递建议与简历改写
│   │   └── core/               # 工具基础设施，不是业务工具
│   │       ├── base.py         # Tool、ToolInput、ToolOutput
│   │       └── registry.py     # 注册与发现工具
│   ├── adapters/         # 外部输入与第三方 I/O 适配
│   │   ├── pdf_parser.py
│   │   └── web_scraper.py
│   ├── history/          # 历史记录管理
│   ├── prompts/          # Prompt 模板
│   ├── evaluation/       # 检索评测代码
│   ├── web/static/      # HTML、CSS、JavaScript 前端
│   ├── cli.py            # 命令行入口
│   ├── config.py         # 配置管理
│   ├── state.py          # 状态定义
│   └── graph.py          # LangGraph 主图
├── scripts/              # 启动、索引维护、评测和发布脚本
├── evaluation/           # 公开评测数据与复现说明
└── tests/                # 测试文件
```

`tools/` 的五个业务文件与注册中心中的五个工具一一对应；`__init__.py` 是包声明，`core/` 是工具基础设施，不计入工具数量。
PDF 解析和网页抓取统一从 `adapters/` 导入，不作为独立的 Agent 工具注册，也不再保留 `tools/` 下的旧兼容入口。

各层职责：`api/cli → graph/nodes → tools → services/adapters`；`tools/core` 提供统一接口和注册，`rag` 负责检索，`history` 负责存档。

## 核心架构

### ReAct Agent 流程

```text
JD 文本或 URL + PDF 简历
   ↓
解析输入 → 初始化 AgentState
   ↓
agent_think
   ├─ 继续 → execute_tool → Observation → agent_think
   └─ 结束 → 组装报告 → Web 按条件存档 → 返回最终结果
```

`execute_tool` 调用已注册的 `extract_jd`、`assess_jd_quality`、`check_company`、`analyze_gap` 或 `generate_advice`。RAG 位于 `generate_advice` 内部：检索相似历史记录，将最多 3 条结果注入 Prompt，再调用模型生成建议。检索无结果或失败时，仍可在不附加历史上下文的情况下生成建议。

JD 提取后必须先通过质量评估，再继续差距分析和建议生成；Agent 根据当前状态选择允许执行的工具；有公司名称时，必须在生成建议前完成公司背景检查，公司检查与差距分析的先后可由 Agent 选择。没有公司名称时跳过公司检查。高质量使用包含改写的输出结构，`low / uncertain` 使用不包含改写字段的结构，对外仍返回 `advice.bullet_rewrites=[]`。

质量判断基于 JD 文本，要求提供可核对的原文证据；引用无法核对时降为 `uncertain`。缺薪资或公司不知名不会单独导致低质量，匹配分数也不代表岗位可信度或录取概率。该判断不等同于招聘方身份、实际待遇的外部核验。

`rewrite_skipped_reason` 表示只跳过改写；`stop_reason` 表示流程提前结束。低质量或信息不明确不会直接结束分析。

Web 流式接口在执行过程中推送 SSE 进度，最终结果在按条件存档后发送。CLI 直接返回报告。

### RAG 检索流程

```text
查询 JD
   ↓
确定每路候选预算（索引数量较大值的 35%，15–80 条；小库取全部）
   ↓
双路召回
   ├─ 向量检索（Chroma）
   └─ BM25 检索
   ↓
按历史 ID 合并候选并过滤重复记录
   ↓
精排（多因子打分）
   ├─ 双路归一化 RRF 45%
   ├─ 技能匹配   35%
   ├─ 资历匹配   10%
   ├─ 领域匹配   5%
   └─ 公司匹配   5%
   ↓
阈值过滤 → Top-K 结果（建议生成阶段最多 3 条）
```

### 多模型降级策略

以下策略适用于 JD 提取、岗位质量评估、差距分析、建议生成及公司检查的结构化 LLM 分支。Agent 决策节点本身不使用这一重试和降级封装。

```text
结构化业务调用
   ↓
主模型（默认 DeepSeek V4 Flash）最多尝试 3 次
   ↓ 全部失败
备用模型 Qwen 3.8 Max 最多尝试 3 次
   ↓ 全部失败
返回错误
```

每个模型的尝试次数包含首次调用；默认两次重试等待分别为 0.5 秒和 1.0 秒。

## API 文档

### 分析接口

**POST /api/analyze**

使用 `multipart/form-data` 提交两个必填字段：`jd_text`（JD 文本或 URL）和 `resume_file`（简历 PDF）。

```bash
curl -X POST http://localhost:8000/api/analyze \
    --form "jd_text=<examples/jd-high.txt" \
    --form "resume_file=@tests/fixtures/sample_resume.pdf"
```

分析完成后返回 JSON，包含 `jd`、`jd_quality`、`gap`、`advice`、`rewrite_skipped_reason`、`stop_reason`、`company_risk`、`errors`、`history_id` 和 `similar_histories`。

### 流式分析接口

**POST /api/analyze/stream**

请求字段与 `/api/analyze` 相同，响应为 SSE 事件流：

```bash
curl -N -X POST http://localhost:8000/api/analyze/stream \
    --form "jd_text=<examples/jd-high.txt" \
    --form "resume_file=@tests/fixtures/sample_resume.pdf"
```

进度事件示例：

```text
event: node_start
data: {"node": "fetch_jd"}

event: node_end
data: {"node": "fetch_jd"}
```

`final_result` 事件携带完整分析结果；`error` 事件携带错误信息。交互式接口文档见 http://localhost:8000/docs。

进度中的 `node` 是前端步骤标识，可为 `fetch_jd`、`extract`、`quality`、`analyze`、`advise`；实际 LangGraph 节点是 `agent_think` 和 `execute_tool`。

### 历史记录接口

**GET /api/history** - 获取历史记录列表

**GET /api/history/{id}** - 获取单条历史记录

**DELETE /api/history/{id}** - 删除历史记录

## 开发指南

### 添加新工具

1. 在 `src/resume_match/tools/` 创建工具文件
2. 继承 `Tool` 基类
3. 实现 `name`、`description`、`input_schema` 属性和 `execute` 方法
4. 在 `tools/core/registry.py` 注册工具
5. 将工具加入 `nodes/agent.py` 的可用动作规则，并在 `nodes/tool_executor.py` 中声明允许写回的状态字段；需要网页进度时补充 API 的步骤映射

示例：

```python
from resume_match.tools.core.base import Tool, ToolInput, ToolOutput


class MyToolInput(ToolInput):
    text: str


class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "去除输入文本首尾的空白字符。"

    @property
    def input_schema(self) -> type[ToolInput]:
        return MyToolInput

    async def execute(self, input: MyToolInput, state: dict) -> ToolOutput:
        return ToolOutput(success=True, result={"text": input.text.strip()})
```

### 修改 Prompt

编辑 `src/resume_match/prompts/templates.py`：

- `EXTRACT_JD_PROMPT` - JD 结构化提取
- `ANALYZE_PROMPT` - 差距分析
- `ADVISE_PROMPT` - 建议生成

岗位质量评估 Prompt 位于 `services/jd_quality.py`。

### 调整检索策略

修改 `src/resume_match/rag/fine_ranker.py` 中的权重：

```python
final_score = (
    0.45 * retrieval_score +   # 双路归一化 RRF
    0.35 * skill_score +       # 技能匹配
    0.05 * company_score +     # 公司匹配
    0.10 * seniority_score +   # 资历匹配
    0.05 * domain_score        # 领域匹配
)
```

## 常见问题

### 网页抓取失败

部分招聘网站有强反爬保护（如 BOSS直聘），建议直接粘贴 JD 文本。

### API 调用超时

检查网络连接、API key 配额、`config.py` 中的模型配置以及模型服务的运行状态。

### 向量检索结果不准确

- 增加历史记录数量（历史会持续入库供检索，模型权重不会自动训练）
- 调整精排阈值（`fine_ranker.py` 中的 `FINE_THRESHOLD`）

### 模型返回空结果

结构化业务调用会对空结果和异常进行有限重试，并在主模型尝试耗尽后调用备用模型；如仍失败，请检查 API key 配额和模型配置。

## 性能优化

- **Embedding 缓存**: 进程内 LRU/TTL 缓存，相同输入与模型配置可复用向量，同键并发共享调用
- **索引持久化**: Chroma 和 BM25 索引写入磁盘，作为 SQLite 历史记录的派生索引
- **并发处理**: FastAPI 异步 I/O
- **模型降级**: 结构化业务调用的主模型尝试耗尽后可调用备用模型；调用比例和成本需实际测量
- **有限重试**: 默认最多每模型 3 次尝试，指数等待 0.5/1.0 秒；增加尝试次数时后续为 2.0/4.0 秒，切换备用模型后重新计数；尚未实现按错误类型分类重试

## 致谢

感谢开源社区提供的灵感和参考实现。

- LangGraph & LangChain 团队提供的 Agent 框架
- DeepSeek 提供高性价比的 LLM 推理
- 阿里云提供通义千问模型
- SiliconFlow 提供 BGE-M3 嵌入模型 API
- Chroma 的嵌入式向量库

## 许可证

本项目使用 MIT License。上游 JD-Scout 的原始代码版权归
Ziming "Alan" Yi 所有；本仓库新增和修改部分版权归 Eirudy 所有。完整许可文本和
归属信息见 [LICENSE](LICENSE)。

## 上游来源与扩展

本项目受到 Ziming "Alan" Yi 的 [JD-Scout](https://github.com/AlanYy-42/JD-Scout) 启发，并基于其 MIT 许可的开源实现进行扩展。上游提供了基础的简历与 JD 分析流程，本仓库进一步扩展了以下能力：

- ReAct 风格的 Agent 编排和隔离工具执行
- 基于原文证据的岗位质量判断与按质量选择简历改写
- BGE-M3 与 BM25 混合检索、RRF 融合和业务特征精排
- Embedding LRU/TTL 缓存与同键请求共享
- 结构化模型调用的重试与降级处理
- SSE 进度推送
- BM25 索引生命周期管理与重建
- 离线评测和回归测试

请勿提交真实简历、API key、本地数据库、缓存或私有评测数据。

## 回归测试

```powershell
uv sync --locked --extra dev
uv run --extra dev python -m pytest -q
```

默认测试禁止真实 HTTP，联网 E2E 需显式设置 `RUN_E2E=1`。
GitHub Actions 使用 Python 3.14 运行离线回归、Ruff 和 JavaScript 语法检查，无需配置 API keys。测试覆盖岗位质量分支、公司背景检查顺序、模型异常处理、历史记录和检索等行为；离线测试不用于衡量模型准确率。

若 Windows 的系统 pytest 临时目录权限异常，可指定一个全新的目录：`uv run --extra dev python -m pytest -q --basetemp temp/pytest-local -o cache_dir=temp/pytest-cache`。该临时目录会由 pytest 管理，不要指向已有个人文件目录。

## 数据处理

上传的 PDF 在文本提取后删除临时副本；提取的简历文本和 JD 会发送到配置的模型服务。Web 历史保存分析结果，其中可能包含原简历的片段；SQLite、Chroma 和 BM25 默认保存在本机。未配置账号认证，默认启动方式适合本地单用户使用。演示截图、JD 和测试简历均为合成样例。

## 索引维护

正常 Web 存档/删除会独立更新 Chroma 和 BM25；存储首次初始化会从 SQLite 完整重建 BM25，清除旧的分词/元数据和已删除 ID。
这不是跨存储事务，也不支持多个进程同时写同一个 BM25 文件。

Embedding 客户端已设置 `check_embedding_ctx_length=False`，确保向 SiliconFlow 发送文本而非 OpenAI token ID；原有 Chroma 向量不会自动更新。
如需重新生成，先停止应用、备份历史数据，再运行 `uv run python scripts/rebuild_vector_index.py`。该操作会调用实际 Embedding API 并产生费用。

RAG 评测脚本、合成 smoke 数据、人工盲标导入导出和真实 BGE 评测步骤见 [evaluation/README.md](evaluation/README.md)。合成/假向量报告只验证流程，不能作为线上效果；真实指标必须基于人工复核的冻结测试集。

Embedding 支持有界进程内 LRU/TTL 缓存（默认 256 条、1 小时），复用相同输入和模型配置的向量，同键并发共享调用、失败不缓存。容量或 TTL 设为 0 可关闭。
