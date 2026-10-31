# 合成演示输入 / Synthetic examples

| 文件 / File | 用途 / Scenario |
| --- | --- |
| `jd-high.txt` | 职责与要求明确，预期继续改写 / Clear and reasonable requirements |
| `jd-low.txt` | 包含入职收费及扣押证件等风险，预期仅生成建议 / Explicit recruitment risks |
| `jd-uncertain.txt` | 职责笼统、工作安排矛盾，预期仅生成建议 / Unclear and contradictory description |

这些 JD 均为合成文本，不代表真实招聘岗位。每份文本满足至少 200 字符的输入要求。
搭配 `tests/fixtures/sample_resume.pdf` 可测试上传、分析和结果展示。模型判断可能随模型及输入变化，固定的分支约束由离线回归测试覆盖。

These are synthetic JDs, not real vacancies. Each meets the 200-character minimum.
Use the synthetic PDF in `tests/fixtures/` to exercise the application. Running analysis sends data to the configured model providers and may incur API charges.

