"""
Tool interfaces and implementations following SOLID principles.
Each tool has a single responsibility and can be easily extended.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
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


class MathCalculatorTool(BaseTool):
    """Mathematical calculation tool - Single Responsibility Principle"""
    
    def __init__(self):
        self._description = "Advanced mathematical expression calculator - handles complex expressions, functions, and constants"
    
    def calculate(self, expression: str) -> str:
        """
        Advanced mathematical expression parser and calculator.
        Handles complex expressions, unit conversions, and mathematical operations.
        """
        import math
        import re
        
        try:
            # Clean and normalize the expression
            expr = expression.strip()
            
            # Handle common mathematical notation
            replacements = {
                '^': '**',  # Power notation
                'π': 'pi', 'PI': 'pi',
                'e': 'math.e', 
                'sqrt': 'math.sqrt',
                'sin': 'math.sin', 'cos': 'math.cos', 'tan': 'math.tan',
                'log': 'math.log', 'ln': 'math.log',
                'exp': 'math.exp',
                'abs': 'abs'
            }
            
            for old, new in replacements.items():
                expr = expr.replace(old, new)
            
            # Safe evaluation environment
            allowed_names = {
                '__builtins__': {},
                # Math functions
                'math': math, 'pi': math.pi, 'e': math.e,
                # Basic functions
                'abs': abs, 'round': round, 'pow': pow, 'sum': sum,
                'min': min, 'max': max, 'int': int, 'float': float,
                # Constants for common calculations
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'exp': math.exp, 'ceil': math.ceil, 'floor': math.floor
            }
            
            # Evaluate the expression
            result = eval(expr, allowed_names)
            
            # Format result intelligently
            if isinstance(result, float):
                # Show appropriate precision
                if abs(result) >= 1000000 or (abs(result) < 0.001 and result != 0):
                    formatted_result = f"{result:.6e}"  # Scientific notation
                elif result == int(result):
                    formatted_result = str(int(result))  # Integer if whole number
                else:
                    formatted_result = f"{result:.8f}".rstrip('0').rstrip('.')  # Clean decimal
            else:
                formatted_result = str(result)
            
            return f"CALCULATION RESULT:\nExpression: {expression}\nProcessed: {expr}\nResult: {formatted_result}"
            
        except Exception as e:
            return f"CALCULATION ERROR: Could not evaluate '{expression}'\nError: {str(e)}\nPlease check the mathematical expression format."
    
    def get_function_tool(self) -> FunctionTool:
        return FunctionTool(self.calculate, description=self._description)
    
    def get_description(self) -> str:
        return self._description


class FileSystemTool(BaseTool):
    """File system operations tool - Single Responsibility Principle"""
    
    def __init__(self):
        self._description = "File system operations - read, write, and list files"
    
    def operate(self, action: str, filename: str, content: str = "") -> str:
        """File system access operations"""
        import os
        
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


class ToolFactory:
    """Factory for creating tools - Dependency Inversion Principle"""
    
    @staticmethod
    def create_math_tool() -> MathCalculatorTool:
        return MathCalculatorTool()
    
    @staticmethod
    def create_file_tool() -> FileSystemTool:
        return FileSystemTool()
    
    @staticmethod
    def get_all_tools() -> Dict[str, BaseTool]:
        return {
            "math": ToolFactory.create_math_tool(),
            "file": ToolFactory.create_file_tool()
        }