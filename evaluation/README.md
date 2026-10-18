# RAG 离线评测

本目录提供检索评测的合成数据和运行说明。

| 文件 | 用途 |
| --- | --- |
| `synthetic_jd_pilot.json` | 32 条候选 JD、30 条查询及相关性标签；查询划分为 6 条开发集和 24 条测试集。 |
| `README.md` | 评测命令、标注流程和指标定义。 |

样本及初始标签由 AI 生成，用于验证数据校验、检索排序和报告生成流程。数据来源标记为 `synthetic`，不能据此估计真实招聘场景中的检索效果。回归测试 `tests/test_evaluation.py` 使用该数据集。

## 运行方式

在项目根目录安装依赖后执行：

```powershell
uv run python -m resume_match.evaluation.runner `
  --dataset evaluation/synthetic_jd_pilot.json `
  --output evaluation/local/runs/smoke-v1 `
  --backend smoke --split test --k 3 --repeats 3
```

`smoke` 后端使用确定性的哈希词袋向量，不调用远程 Embedding 或 LLM API。索引位于独立临时目录，不读取个人历史库。`--output` 必须指向尚不存在的目录。

输出包含 `result.json`、`per_query.csv`、`report.md` 和 `dataset.snapshot.json`。本地数据、缓存及运行结果保存在 Git 忽略的 `evaluation/local/` 中。

## 数据标注

真实数据集需记录来源和授权信息，并移除个人联系方式。开发集与测试集应在调参前划分，避免重复文本或相同结构化输入跨集合出现。候选集合应覆盖相关岗位、难负样本及无相关候选的查询。

生成待标注数据并导出标注表：

```powershell
uv run python -m resume_match.evaluation.annotation draft `
  --input raw_dataset.json --output evaluation/local/draft.json
uv run python -m resume_match.evaluation.annotation export `
  --dataset evaluation/local/draft.json --output evaluation/local/annotation.json
```

标注表不包含检索排名。每个查询与候选组合需填写 `grade` 和 `reason`：`2` 表示高度相关，`1` 表示部分相关，`0` 表示不相关。草稿中的占位值不代表已完成标注。

导入人工标注：

```powershell
uv run python -m resume_match.evaluation.annotation import `
  --dataset evaluation/local/draft.json `
  --annotations evaluation/local/annotation.json `
  --output evaluation/local/reviewed.json
```

配置 SiliconFlow 凭据后，可使用 BGE-M3 评测：

```powershell
uv run python -m resume_match.evaluation.runner `
  --dataset evaluation/local/reviewed.json `
  --output evaluation/local/runs/bge-test-v1 `
  --backend siliconflow --split test --k 3 --repeats 3
```

该后端将去重后的结构化 JD 文本发送至配置的 SiliconFlow API，可能产生调用费用。人工标注不会改变数据集原有的合成来源属性。

## 指标定义

评测包含 `vector`、`bm25`、`vector_rules`、`rrf_only` 和 `hybrid` 五种方法。其中 `vector_rules` 用于分析 BM25 的贡献，`rrf_only` 用于分析业务精排规则的贡献。

- `Hit@k`、`Recall@k`、`MRR@k` 和 `nDCG@k`：在存在相关候选的查询上计算宏平均。
- `Precision@k`：分母固定为 `k`，对全部查询计算宏平均。
- 无相关候选的查询：单独统计误召回率。
- 检索时延：每条查询重复测量后取中位数，再计算查询级 P95；不包含远程 Embedding、索引构建、JD 提取、Agent 执行和结果生成。

合成数据结果用于功能验证。真实数据结果需结合样本规模、标注质量、测试集独立性和数据代表性解释。

## 扩展实验

以下脚本需要额外输入，相关数据未包含在仓库中。

- `scripts/benchmark_embedding_cache.py`：通过 `--dataset` 指定含 10 条查询的数据集。历史向量读取自 `evaluation/local/jd-pilot/bge-responses.json` 或 `evaluation/local/jd-pilot-v2/embedding-cache/*.json`。`--offline` 模式还需要对应的查询向量；其他模式调用真实 Embedding API。
- `scripts/public_job_benchmark.py`：需要可选依赖 `pyarrow`，以及 [Avature/Job-Title-Similarity](https://huggingface.co/datasets/Avature/Job-Title-Similarity) 的中文数据。`corpus.parquet`、`queries.parquet` 和含版本 `sha` 的 `metadata.json` 放在 `evaluation/local/public-jobs/`。执行顺序为 `prepare`、`embed`、`dev`、`test`；Embedding 和重排阶段调用真实 API。
- `scripts/fallback_smoke.py --output evaluation/local/runs/fallback-v1`：使用主备模型凭据验证结构化输出与降级流程，会发送真实模型请求；输出目录必须尚不存在。
