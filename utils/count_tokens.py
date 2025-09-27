import tiktoken
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the actual math agent configuration
from enterprise_ai.core.agents import AgentConfigurationProvider

# Get the actual current system message from the math agent
math_agent_config = AgentConfigurationProvider.get_math_agent_config()
system_message = math_agent_config.system_message

# Get the encoding for GPT-4 (cl100k_base is used by GPT-4, GPT-3.5-turbo, etc.)
encoding = tiktoken.get_encoding("cl100k_base")

# Count the tokens
tokens = encoding.encode(system_message)
token_count = len(tokens)

print(f"Math agent system message token count: {token_count}")
print(f"Character count: {len(system_message)}")
print(f"Word count (approximate): {len(system_message.split())}")

# Show cost impact at GPT-4o-mini rates
input_cost_per_1k = 0.00015  # $0.00015 per 1k input tokens
cost_per_message = (token_count / 1000) * input_cost_per_1k
print(f"Cost per message with this system prompt: ${cost_per_message:.6f}")

# Context window analysis
print(f"\nContext window analysis:")
print(f"Tokens used by system message: {token_count}")
print(f"Remaining context (assuming 128k model): {128000 - token_count:,}")
print(f"Percentage of context used: {(token_count / 128000) * 100:.2f}%")

# Compare to a minimal version
minimal_message = """You are a mathematics expert with calculator access.

Use the calculator for numerical expressions and calculations.
Use your knowledge for concepts and explanations.
Ask questions when the request is unclear."""

minimal_tokens = len(encoding.encode(minimal_message))
print(f"\nComparison to minimal version:")
print(f"Minimal system message tokens: {minimal_tokens}")
print(f"Token reduction: {token_count - minimal_tokens}")
print(f"Percentage reduction: {((token_count - minimal_tokens) / token_count) * 100:.1f}%")