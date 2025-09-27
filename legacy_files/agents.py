"""
Agent factory and configurations following SOLID principles.
Separates agent creation from business logic.
"""
from typing import Dict, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import BaseTool


class AgentConfig:
    """Configuration for an agent - Single Responsibility Principle"""
    
    def __init__(self, name: str, system_message: str, tools: List[str]):
        self.name = name
        self.system_message = system_message
        self.required_tools = tools


class AgentFactory:
    """Factory for creating agents - Open/Closed Principle"""
    
    def __init__(self, model_client: OpenAIChatCompletionClient, available_tools: Dict[str, BaseTool]):
        self.model_client = model_client
        self.available_tools = available_tools
    
    def create_agent(self, config: AgentConfig) -> AssistantAgent:
        """Create an agent based on configuration"""
        # Get required tools for this agent
        agent_tools = []
        for tool_name in config.required_tools:
            if tool_name in self.available_tools:
                agent_tools.append(self.available_tools[tool_name].get_function_tool())
        
        return AssistantAgent(
            name=config.name,
            model_client=self.model_client,
            system_message=config.system_message,
            tools=agent_tools
        )
    
    def create_agent_tool(self, config: AgentConfig) -> AgentTool:
        """Create an AgentTool wrapper for delegation"""
        agent = self.create_agent(config)
        return AgentTool(agent)


class AgentConfigurationProvider:
    """Provides agent configurations - Single Responsibility Principle"""
    
    @staticmethod
    def get_math_agent_config() -> AgentConfig:
        return AgentConfig(
            name="math_agent",
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

MATHEMATICAL FORMATTING:
- Use proper mathematical notation in responses
- For inline math: $x^2$, $\sin(\pi/4)$, $\int x dx$
- For display math: $$\int_0^5 x^2 dx = \frac{x^3}{3}\Big|_0^5 = \frac{125}{3}$$
- Use LaTeX symbols: $\pi$, $\infty$, $\sum$, $\prod$, $\sqrt{x}$, $\frac{a}{b}$

NEVER GUESS at calculations - always be explicit about what you're computing and why.""",
            tools=["math"]
        )
    
    @staticmethod
    def get_system_agent_config() -> AgentConfig:
        return AgentConfig(
            name="system_agent",
            system_message="""You are a system expert with file access.

CRITICAL RULES:
- For checking actual files/directories: USE FILE TOOL
- For general system advice: Use your knowledge
- For reading file contents: USE FILE TOOL
- Always distinguish between current system state vs general advice

Examples:
- "What files are here?" → USE TOOL: operate("list", ".")  
- "Best practices for backups?" → USE KNOWLEDGE
- "Read config.txt" → USE TOOL: operate("read", "config.txt")
""",
            tools=["file"]
        )
    
    @staticmethod
    def get_general_agent_config() -> AgentConfig:
        return AgentConfig(
            name="general_agent",
            system_message="You are a helpful assistant. Answer general questions with your knowledge.",
            tools=[]
        )