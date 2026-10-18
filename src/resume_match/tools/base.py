"""Tool 基类和输入输出定义。"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel


class ToolInput(BaseModel):
    """工具输入的基类"""
    pass


class ToolOutput(BaseModel):
    """工具输出的基类"""
    success: bool
    result: Any
    error: Optional[str] = None


class Tool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（给 LLM 看的）"""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[ToolInput]:
        """输入参数 schema"""
        pass

    @abstractmethod
    async def execute(self, input: ToolInput, state: dict) -> ToolOutput:
        """执行工具"""
        pass
