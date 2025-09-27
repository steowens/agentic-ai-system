import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

class RealWorldAgentSystem:
    """
    This is how you ACTUALLY solve the routing problem:
    1. Create agents that KNOW when to use tools vs reasoning
    2. Build a classifier that routes intelligently  
    3. Use REAL tools, not toy examples
    """
    
    def __init__(self):
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self.tools = self._create_real_tools()
        self.agents = self._create_smart_agents()
    
    def _create_real_tools(self):
        """Real tools that do things LLMs can't do well"""
        
        def smart_math_calculator(expression: str) -> str:
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
        
        def file_operations(action: str, filename: str, content: str = "") -> str:
            """File system access - LLMs can't access files"""
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
        
        return {
            "math_tool": FunctionTool(smart_math_calculator, description="Advanced mathematical expression calculator - handles complex expressions, functions, and constants"),
            "file_tool": FunctionTool(file_operations, description="File system operations")
        }
    
    def _create_smart_agents(self):
        """Agents that know when to use tools vs their knowledge"""
        
        # Math Agent - cautious and precise
        math_agent = AssistantAgent(
            "math_agent",
            model_client=self.model_client,
            system_message="""You are a careful mathematics expert with access to a precise calculator.

CRITICAL DECISION PROCESS:
1. ANALYZE the user's request carefully
2. IDENTIFY if there are mathematical expressions to calculate
3. If UNSURE whether calculation is needed: ASK CLARIFYING QUESTIONS first
4. NEVER assume - be explicit about what you're calculating

WHEN TO USE THE CALCULATOR TOOL:
- Clear numerical expressions: "What is 123 * 456?"
- Mathematical functions: "Calculate sin(30 degrees)"  
- Complex expressions: "What's the result of (5^3 + sqrt(144)) / 2?"
- Unit conversions with math: "Convert 75 mph to m/s"

WHEN TO USE YOUR KNOWLEDGE:
- Conceptual explanations: "What is a derivative?"
- Theoretical questions: "Prove the Pythagorean theorem"
- Method explanations: "How do you integrate by parts?"

WHEN TO ASK QUESTIONS:
- Ambiguous requests: "Help me with calculus" → Ask: "What specific calculus problem would you like help with?"
- Unclear expressions: "What's that formula result?" → Ask: "Which formula and with what values?"
- Missing information: "Calculate the area" → Ask: "Area of what shape and with what dimensions?"

TOOL USAGE FORMAT:
When you identify a clear mathematical expression, extract it and pass it to the calculator tool.
For example: "Calculate the integral of x^2 from 0 to 5" → Use tool with "integrate(x**2, 0, 5)" if you can express it, otherwise ask for the numerical setup.

NEVER GUESS at calculations - always be explicit about what you're computing and why.""",
            tools=[self.tools["math_tool"]]
        )
        
        # System Agent - uses tools for file operations, knowledge for advice
        system_agent = AssistantAgent(
            "system_agent", 
            model_client=self.model_client,
            system_message="""You are a system expert with file access.

CRITICAL RULES:
- For checking actual files/directories: USE FILE TOOL
- For general system advice: Use your knowledge
- For reading file contents: USE FILE TOOL
- Always distinguish between current system state vs general advice

Examples:
- "What files are here?" → USE TOOL: file_operations("list", ".")  
- "Best practices for backups?" → USE KNOWLEDGE
- "Read config.txt" → USE TOOL: file_operations("read", "config.txt")
""",
            tools=[self.tools["file_tool"]]
        )
        
        # Convert to AgentTools for delegation
        return {
            "math": AgentTool(math_agent),
            "system": AgentTool(system_agent)
        }
    
    async def smart_dispatch(self, question: str) -> str:
        """
        This is the KEY: intelligent question routing
        Instead of a complex orchestrator, use pattern matching + keywords
        """
        
        question_lower = question.lower()
        
        # Math keywords and patterns - be more specific about calculation requests
        calculation_indicators = [
            'calculate', 'compute', 'what is', 'what\'s', 'solve for', 'find the value',
            'multiply', 'divide', 'add', 'subtract', 'equals', '=',
            '*', '+', '-', '/', '^', 'pow', '**',
            'sqrt', 'square root', 'sin(', 'cos(', 'tan(', 'log(', 'ln(',
            'result of', 'answer to', 'evaluate'
        ]
        
        # Check for actual mathematical expressions (numbers + operators)
        import re
        has_math_expression = bool(re.search(r'\d+\s*[+\-*/^]\s*\d+', question)) or \
                             bool(re.search(r'(sin|cos|tan|sqrt|log)\s*\(', question_lower)) or \
                             bool(re.search(r'\d+\s*\*\*\s*\d+', question))  # Power notation
        
        # System/File keywords  
        system_indicators = [
            'file', 'directory', 'folder', 'read', 'write', 'list', 'files',
            'current directory', 'what files', 'file system', '.txt', '.py', '.json'
        ]
        
        # Check for math question - either has calculation indicators OR mathematical expressions
        if (any(indicator in question_lower for indicator in calculation_indicators) or has_math_expression):
            print("🧮 ROUTING TO: Math Agent")
            coordinator = AssistantAgent(
                "math_coordinator",
                model_client=self.model_client,
                system_message="Route math questions to the math agent. Be direct.",
                tools=[self.agents["math"]]
            )
            response = await coordinator.run(task=question)
            return response.messages[-1].content
        
        # Check for system question
        elif any(indicator in question_lower for indicator in system_indicators):
            print("💻 ROUTING TO: System Agent") 
            coordinator = AssistantAgent(
                "system_coordinator",
                model_client=self.model_client,
                system_message="Route system questions to the system agent. Be direct.",
                tools=[self.agents["system"]]
            )
            response = await coordinator.run(task=question)
            return response.messages[-1].content
        
        # General question - handle with general knowledge
        else:
            print("🧠 ROUTING TO: General Knowledge")
            general_agent = AssistantAgent(
                "general_agent",
                model_client=self.model_client,
                system_message="You are a helpful assistant. Answer general questions with your knowledge."
            )
            response = await general_agent.run(task=question)
            return response.messages[-1].content

async def test_real_routing():
    """Test the actual routing system"""
    
    system = RealWorldAgentSystem()
    
    print("🎯 REAL AGENT ROUTING SYSTEM")
    print("=" * 50)
    
    test_cases = [
        # Should route to Math Agent + use calculator tool
        "What is 123456 * 789012?",
        "Calculate the square root of 50",
        
        # Should route to Math Agent but use knowledge  
        "Explain what a derivative is",
        
        # Should route to System Agent + use file tool
        "What files are in the current directory?", 
        "List files here",
        
        # Should route to System Agent but use knowledge
        "What are best practices for file organization?",
        
        # Should route to General Agent
        "What's the capital of France?",
        "How does photosynthesis work?"
    ]
    
    for question in test_cases:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        try:
            answer = await system.smart_dispatch(question)
            print(f"✅ Answer: {answer[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    await system.model_client.close()

if __name__ == "__main__":
    asyncio.run(test_real_routing())