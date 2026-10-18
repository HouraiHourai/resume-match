"""检查公司可信度的工具（白名单 + LLM 判断）。"""
from resume_match.tools.base import Tool, ToolInput, ToolOutput
from resume_match.config import get_llm
from pydantic import BaseModel, Field


# 知名企业白名单（快速路径，无需 LLM 判断）
WELL_KNOWN_COMPANIES = {
    # 互联网大厂
    "字节跳动", "ByteDance", "抖音", "今日头条",
    "腾讯", "Tencent", "微信", "QQ",
    "阿里巴巴", "Alibaba", "阿里", "淘宝", "天猫", "阿里云",
    "百度", "Baidu",
    "美团", "Meituan",
    "京东", "JD.com", "京东集团",
    "拼多多", "PDD", "Pinduoduo",
    "小米", "Xiaomi", "米家",
    "华为", "Huawei",
    "网易", "NetEase",
    "bilibili", "B站", "哔哩哔哩",
    "快手", "Kuaishou",
    "滴滴", "Didi", "滴滴出行",
    "携程", "Ctrip", "去哪儿",
    "新浪", "Sina", "微博", "Weibo",
    "搜狐", "Sohu",
    "360", "奇虎360",
    "小红书", "RED", "Xiaohongshu",
    "知乎", "Zhihu",
    "陌陌", "Momo",
    "爱奇艺", "iQIYI",
    "优酷", "Youku",

    # 外企科技公司
    "Google", "谷歌", "Microsoft", "微软", "Apple", "苹果",
    "Amazon", "亚马逊", "Meta", "Facebook", "脸书",
    "Tesla", "特斯拉", "Netflix", "奈飞",
    "IBM", "Oracle", "甲骨文", "SAP",
    "Intel", "英特尔", "AMD", "NVIDIA", "英伟达",

    # 金融/电信/国企
    "工商银行", "建设银行", "农业银行", "中国银行",
    "招商银行", "中信银行", "浦发银行", "民生银行",
    "中国移动", "中国联通", "中国电信",
    "中国石油", "中国石化", "国家电网",
    "中国平安", "中国人寿",
}


class CheckCompanyInput(ToolInput):
    """检查公司的输入"""
    company_name: str = Field(description="公司名称")


class CompanyCheckResult(BaseModel):
    """公司检查结果的结构化输出"""
    is_reliable: bool = Field(description="公司是否可靠（500强或知名大厂）")
    company_type: str = Field(description="公司类型，如：世界500强、中国500强、知名互联网大厂、普通企业")
    reasoning: str = Field(description="判断理由")


class CheckCompanyTool(Tool):
    """查询公司的可信度信息（基于 LLM 判断是否为500强/知名大厂）。"""

    @property
    def name(self) -> str:
        return "check_company"

    @property
    def description(self) -> str:
        return """判断公司是否为知名企业（世界500强、中国500强、知名互联网大厂等）。

使用场景：
- 对于不熟悉的公司，调用此工具判断可信度
- 知名公司会被标记为可靠
- 不知名公司会提示用户注意核实背景

输出：
- is_reliable: 是否为知名企业
- company_type: 企业类型（500强/知名大厂/普通企业）
- reasoning: 判断理由和建议"""

    @property
    def input_schema(self) -> type[ToolInput]:
        return CheckCompanyInput

    async def execute(self, input: CheckCompanyInput, state: dict) -> ToolOutput:
        try:
            # 快速路径：检查是否在知名企业白名单中
            if self._is_in_whitelist(input.company_name):
                return ToolOutput(
                    success=True,
                    result={
                        "company_name": input.company_name,
                        "is_reliable": True,
                        "company_type": "知名企业",
                        "message": "✅ 知名企业，可信度高，可以放心投递。",
                        "reasoning": "该公司在知名企业白名单中（世界500强/知名大厂/知名上市公司）",
                    }
                )

            # 慢速路径: 使用 LLM 判断公司类型
            result = await self._check_company_with_llm(input.company_name)

            # 根据判断结果生成提示信息
            if result.is_reliable:
                message = f"✅ {result.company_type}，可信度高，可以放心投递。"
            else:
                message = (
                    f"⚠️ 该公司不属于知名企业（{result.company_type}），建议核实以下信息：\n"
                    f"1. 在国家企业信用信息公示系统查询工商信息\n"
                    f"2. 在招聘平台查看公司评价和员工反馈\n"
                    f"3. 搜索相关新闻，确认公司真实性\n"
                    f"4. 注意招聘信息是否存在异常（如异常高薪、模糊的岗位描述等）"
                )

            return ToolOutput(
                success=True,
                result={
                    "company_name": input.company_name,
                    "is_reliable": result.is_reliable,
                    "company_type": result.company_type,
                    "message": message,
                    "reasoning": result.reasoning,
                }
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                result=None,
                error=f"查询公司信息失败: {str(e)}"
            )

    def _is_in_whitelist(self, company_name: str) -> bool:
        """检查公司名是否在白名单中（模糊匹配）"""
        # 规范化公司名称（去除常见后缀）
        normalized = company_name.replace("有限公司", "").replace("股份", "").replace("集团", "").strip()

        # 检查是否包含白名单中的任何关键词
        for keyword in WELL_KNOWN_COMPANIES:
            if keyword in company_name or keyword in normalized:
                return True
        return False

    async def _check_company_with_llm(self, company_name: str) -> CompanyCheckResult:
        """使用 LLM 判断公司是否为500强或知名大厂。"""
        from resume_match.services.structured_llm import ainvoke_structured_with_retry

        llm = get_llm(temperature=0.1)  # 低温度，保证判断稳定

        # 使用结构化输出（function_calling 方式）
        structured_llm = llm.with_structured_output(CompanyCheckResult, method="function_calling")

        prompt = f"""请判断以下公司是否为知名企业。

公司名称：{company_name}

判断标准：
1. **可靠企业**（is_reliable=true）：
   - 世界500强企业
   - 中国500强企业
   - 知名互联网大厂（如BAT、字节跳动、美团、拼多多、小米、华为、网易、bilibili、快手、滴滴等）
   - 知名上市公司
   - 大型国企、央企

2. **普通企业**（is_reliable=false）：
   - 不在以上范围的公司

注意：
- 只要曾经是或现在是500强/知名大厂，就判断为可靠
- 如果不确定，偏向判断为普通企业（谨慎原则）
- company_type 要具体描述（如"世界500强科技公司"、"知名互联网大厂"、"普通企业"）
- reasoning 要简洁说明判断依据

请输出结构化结果。"""

        # 使用项目的重试机制，支持多模型降级
        result = await ainvoke_structured_with_retry(
            structured_llm,
            prompt,
            attempts=3,
            fallback_model="qwen",
            schema_class=CompanyCheckResult
        )
        return result
