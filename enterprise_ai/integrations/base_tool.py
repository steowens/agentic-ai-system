"""
Base tool interface following SOLID principles.
"""
from abc import ABC, abstractmethod
from autogen_core.tools import FunctionTool


class BaseTool(ABC):
    """Abstract base class for all tools - Interface Segregation Principle"""

    @abstractmethod
    def get_function_tool(self) -> FunctionTool:
        """Return the AutoGen FunctionTool wrapper"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return tool description"""
        pass