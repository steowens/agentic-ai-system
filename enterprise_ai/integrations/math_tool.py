"""
Mathematical calculation tool.
Handles complex mathematical expressions and calculations.
"""
import math
import re
import logging
from autogen_core.tools import FunctionTool
from .base_tool import BaseTool


class MathCalculatorTool(BaseTool):
    """Mathematical calculation tool - Single Responsibility Principle"""

    def __init__(self):
        self._description = "Advanced mathematical expression calculator - handles complex expressions, functions, and constants"

    def calculate(self, expression: str) -> str:
        """
        Advanced mathematical expression parser and calculator.
        Handles complex expressions, unit conversions, and mathematical operations.
        """
        # Log the input expression using the enterprise_ai logger
        logger = logging.getLogger("enterprise_ai")
        logger.info(f"Math calculation input: {expression}")

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

            return f"The calculation result is: **{formatted_result}**"

        except Exception as e:
            return f"CALCULATION ERROR: Could not evaluate '{expression}'\nError: {str(e)}\nPlease check the mathematical expression format."

    def get_function_tool(self) -> FunctionTool:
        return FunctionTool(self.calculate, description=self._description)

    def get_description(self) -> str:
        return self._description