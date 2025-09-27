"""
File system operations tool.
Handles reading, writing, and listing files and directories.
"""
import os
from autogen_core.tools import FunctionTool
from .base_tool import BaseTool


class FileSystemTool(BaseTool):
    """File system operations tool - Single Responsibility Principle"""

    def __init__(self):
        self._description = "File system operations - read, write, and list files"

    def operate(self, action: str, filename: str, content: str = "") -> str:
        """File system access operations"""
        try:
            if action == "read":
                with open(filename, 'r') as f:
                    return f"FILE CONTENT:\n{f.read()[:1000]}"
            elif action == "write":
                with open(filename, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {filename}"
            elif action == "list":
                files = os.listdir(filename if filename else ".")
                return f"FILES: {', '.join(files[:20])}"
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"File error: {e}"

    def get_function_tool(self) -> FunctionTool:
        return FunctionTool(self.operate, description=self._description)

    def get_description(self) -> str:
        return self._description