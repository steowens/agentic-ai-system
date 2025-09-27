"""
Tool factory for creating and managing all available tools.
Follows the Factory Pattern and Dependency Inversion Principle.
"""
from typing import Dict
from .base_tool import BaseTool
from .math_tool import MathCalculatorTool
from .file_tool import FileSystemTool
from .wordle_tool import WordleTool


class ToolFactory:
    """Factory for creating tools - Dependency Inversion Principle"""

    @staticmethod
    def create_math_tool() -> MathCalculatorTool:
        """Create a mathematical calculation tool"""
        return MathCalculatorTool()

    @staticmethod
    def create_file_tool() -> FileSystemTool:
        """Create a file system operations tool"""
        return FileSystemTool()

    @staticmethod
    def create_wordle_tool() -> WordleTool:
        """Create a Wordle constraint solver tool"""
        return WordleTool()

    @staticmethod
    def get_all_tools() -> Dict[str, BaseTool]:
        """Get all available tools as a dictionary"""
        return {
            "math": ToolFactory.create_math_tool(),
            "file": ToolFactory.create_file_tool(),
            "wordle": ToolFactory.create_wordle_tool()
        }

    @staticmethod
    def get_available_tool_types() -> list:
        """Get list of all available tool types"""
        return list(ToolFactory.get_all_tools().keys())

    @staticmethod
    def create_tool_by_name(tool_name: str) -> BaseTool:
        """Create a specific tool by name"""
        tools_map = {
            "math": ToolFactory.create_math_tool,
            "file": ToolFactory.create_file_tool,
            "wordle": ToolFactory.create_wordle_tool
        }

        if tool_name not in tools_map:
            raise ValueError(f"Unknown tool type: {tool_name}. Available: {list(tools_map.keys())}")

        return tools_map[tool_name]()