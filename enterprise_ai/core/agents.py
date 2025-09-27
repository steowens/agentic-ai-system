"""
Agent factory and configurations following SOLID principles.
Separates agent creation from business logic.
"""
from typing import Dict, List, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from ..integrations import BaseTool
from .compressed_agent import CompressedAssistantAgent, ContextCompressionService


class AgentConfig:
    """Configuration for an agent - Single Responsibility Principle"""
    
    def __init__(self, name: str, system_message: str, tools: List[str]):
        self.name = name
        self.system_message = system_message
        self.required_tools = tools


class AgentFactory:
    """Factory for creating agents - Open/Closed Principle"""

    def __init__(self, model_client: OpenAIChatCompletionClient, available_tools: Dict[str, BaseTool],
                 enable_compression: bool = True, max_tokens: int = 100000):
        self.model_client = model_client
        self.available_tools = available_tools
        self.enable_compression = enable_compression
        self.max_tokens = max_tokens
    
    def create_agent(self, config: AgentConfig) -> AssistantAgent:
        """Create an agent based on configuration"""
        # Get required tools for this agent
        agent_tools = []
        for tool_name in config.required_tools:
            if tool_name in self.available_tools:
                agent_tools.append(self.available_tools[tool_name].get_function_tool())

        # Create compressed or regular agent based on configuration
        if self.enable_compression:
            print(f"🗜️ Creating CompressedAssistantAgent: {config.name} (max_tokens: {self.max_tokens})")
            return CompressedAssistantAgent(
                name=config.name,
                model_client=self.model_client,
                system_message=config.system_message,
                tools=agent_tools,
                max_tokens=self.max_tokens,
                compression_ratio=0.7  # Compress at 70% of max tokens
            )
        else:
            print(f"📝 Creating standard AssistantAgent: {config.name}")
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

    def get_compression_stats(self, agent: AssistantAgent) -> Dict[str, Any]:
        """Get compression statistics for an agent"""
        if isinstance(agent, CompressedAssistantAgent):
            return ContextCompressionService.get_compression_stats(agent)
        else:
            return {"status": "compression_disabled", "agent_type": "standard"}


class AgentConfigurationProvider:
    """Provides agent configurations - Single Responsibility Principle"""
    
    @staticmethod
    def get_math_agent_config() -> AgentConfig:
        return AgentConfig(
            name="math_agent",
            system_message=r"""You are a careful mathematics expert with access to a precise calculator.

MANDATORY CALCULATOR USAGE:
For ANY question involving numbers, calculations, measurements, or mathematical expressions, you MUST use the calculator tool FIRST before providing any response.

EXAMPLES REQUIRING CALCULATOR:
- "What is 2+2?" → Use calculator: "2+2"
- "Height of roof truss with 30 foot span and 4:12 pitch?" → Use calculator: "30 * (4/12) / 2"
- "What is sin(30)?" → Use calculator: "sin(30)"
- "Convert 75 mph to m/s" → Use calculator: "75 * 0.44704"
- ANY question with numbers → ALWAYS use calculator tool first

ONLY USE YOUR KNOWLEDGE WITHOUT CALCULATOR FOR:
- Pure conceptual explanations: "What is a derivative?"
- Theoretical proofs: "Prove the Pythagorean theorem"
- Method explanations: "How do you integrate by parts?"

WHEN TO ASK QUESTIONS:
- Ambiguous requests: "Help me with calculus" → Ask: "What specific calculus problem would you like help with?"
- Unclear expressions: "What's that formula result?" → Ask: "Which formula and with what values?"
- Missing information: "Calculate the area" → Ask: "Area of what shape and with what dimensions?"

GEOMETRY RELATIONSHIPS:
- For a circle: diameter = circumference / π, radius = diameter / 2, area = π * r²
- "Circle of X" typically means circumference = X unless stated otherwise
- "Average diameter" for a perfect circle is just the diameter

TOOL USAGE FORMAT:
When you identify a clear mathematical expression, extract it and pass it to the calculator tool.
For example: "Calculate the integral of x^2 from 0 to 5" → Use tool with "integrate(x**2, 0, 5)" if you can express it, otherwise ask for the numerical setup.

MATHEMATICAL FORMATTING:
- Use proper mathematical notation in responses
- For inline math: $x^2$, $\sin(\pi/4)$, $\int x dx$
- For display math: $$\int_0^5 x^2 dx = \frac{x^3}{3}\Big|_0^5 = \frac{125}{3}$$
- Use LaTeX symbols: $\pi$, $\infty$, $\sum$, $\prod$, $\sqrt{x}$, $\frac{a}{b}$

RESPONSE ATTRIBUTION (MANDATORY):
You MUST end EVERY response with exactly one of these method indicators:

If you used the calculator tool:
"🧮 **Method: Calculator Tool** - Used precise calculation function"

If you solved it with your knowledge:  
"🧠 **Method: Mathematical Knowledge** - Solved using analytical methods"

If you need more information:
"❓ **Method: Clarification Needed** - Requesting more information"

NO EXCEPTIONS - every math response must have a method indicator.

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

    @staticmethod
    def get_wordle_agent_config() -> AgentConfig:
        return AgentConfig(
            name="wordle_agent",
            system_message="""You are a Wordle puzzle expert with access to a constraint solver.

MANDATORY TOOL USAGE:
For ANY Wordle-related question, you MUST use the solve_wordle tool.

EXAMPLES REQUIRING TOOL:
- "Find a word without letters CORTI or U, containing P and S, with P at position 2"
  → Use tool: solve_wordle(excluded_letters="CORTIU", included_letters="PS", letters_at="P:2")

- "I need a word with two E's and no A, B, C"
  → Use tool: solve_wordle(excluded_letters="ABC", included_letters="EE", letters_at="")

- "Word with R at position 3, S at position 5, no vowels except E"
  → Use tool: solve_wordle(excluded_letters="AIOU", included_letters="E", letters_at="R:3,S:5")

PARAMETER FORMAT:
- excluded_letters: String of letters to exclude (e.g., "ABCD")
- included_letters: String of required letters, duplicates for count (e.g., "EE" means two E's)
- letters_at: Position constraints as "letter:position" pairs (e.g., "P:2,S:4")

RESPONSE FORMAT:
Always use the tool first, then explain the results in a user-friendly way.""",
            tools=["wordle"]
        )