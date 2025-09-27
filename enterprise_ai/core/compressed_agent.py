"""
Compressed Assistant Agent with automatic context management.
Prevents token limit errors by intelligently compressing conversation history.
"""
import tiktoken
from typing import List, Dict, Any, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ChatMessage


class CompressedAssistantAgent(AssistantAgent):
    """
    Assistant Agent with automatic context compression to prevent token limit errors.

    Features:
    - Token-based compression using tiktoken
    - Sliding window strategy
    - System message preservation
    - Configurable compression thresholds
    """

    def __init__(self,
                 max_tokens: int = 100000,  # Leave room for response
                 compression_ratio: float = 0.7,  # When to compress (70% of max)
                 min_messages_to_keep: int = 6,  # Always keep recent exchanges
                 **kwargs):
        """
        Initialize compressed agent.

        Args:
            max_tokens: Maximum tokens before compression (default: 100k)
            compression_ratio: Trigger compression at this ratio of max_tokens
            min_messages_to_keep: Minimum recent messages to preserve
        """
        super().__init__(**kwargs)

        self.max_tokens = max_tokens
        self.compression_threshold = int(max_tokens * compression_ratio)
        self.min_messages_to_keep = min_messages_to_keep

        # Get the appropriate tokenizer for the model
        try:
            # Try to get model from model_client if it exists
            if hasattr(self, 'model_client') and self.model_client:
                model_name = getattr(self.model_client, 'model', 'gpt-4o-mini')
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            else:
                # Fallback to cl100k_base encoding
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except (KeyError, AttributeError):
            # Fallback to cl100k_base encoding (used by GPT-4 family)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        print(f"🗜️ CompressedAgent initialized: max_tokens={max_tokens}, threshold={self.compression_threshold}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if isinstance(text, list):
            # Handle message list
            total = 0
            for msg in text:
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                elif hasattr(msg, 'content'):
                    content = msg.content
                else:
                    content = str(msg)
                total += len(self.tokenizer.encode(content))
            return total
        else:
            return len(self.tokenizer.encode(str(text)))

    def count_messages_tokens(self, messages: List[ChatMessage]) -> int:
        """Count total tokens in message list."""
        total_tokens = 0
        for message in messages:
            # Count role and content
            role_tokens = len(self.tokenizer.encode(message.role))
            content_tokens = len(self.tokenizer.encode(message.content or ""))
            total_tokens += role_tokens + content_tokens + 3  # 3 tokens for message formatting
        return total_tokens

    def compress_conversation(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Compress conversation using sliding window strategy.

        Strategy:
        1. Always preserve system message (first message)
        2. Keep recent message pairs (user + assistant)
        3. If still too long, remove oldest pairs
        """
        if len(messages) <= 3:  # System + user + assistant minimum
            return messages

        current_tokens = self.count_messages_tokens(messages)

        if current_tokens <= self.compression_threshold:
            return messages  # No compression needed

        print(f"🗜️ COMPRESSING: {current_tokens} tokens > {self.compression_threshold} threshold")

        # Always keep system message
        compressed = [messages[0]] if messages[0].role == 'system' else []

        # Calculate how many recent messages to keep
        messages_to_check = messages[1:] if compressed else messages

        # Keep pairs of user/assistant messages from the end
        kept_messages = []
        tokens_kept = len(self.tokenizer.encode(compressed[0].content)) if compressed else 0

        # Work backwards through messages, keeping pairs
        for i in range(len(messages_to_check) - 1, -1, -1):
            msg = messages_to_check[i]
            msg_tokens = self.count_messages_tokens([msg])

            # Check if we can afford to keep this message
            if tokens_kept + msg_tokens <= self.compression_threshold:
                kept_messages.insert(0, msg)
                tokens_kept += msg_tokens
            else:
                break

        # Ensure we keep minimum number of recent messages
        if len(kept_messages) < self.min_messages_to_keep:
            # Keep at least the last N messages regardless of token count
            kept_messages = messages_to_check[-self.min_messages_to_keep:]

        final_messages = compressed + kept_messages
        final_tokens = self.count_messages_tokens(final_messages)

        removed_count = len(messages) - len(final_messages)
        print(f"🗜️ COMPRESSED: Removed {removed_count} messages, {current_tokens} → {final_tokens} tokens")

        return final_messages

    async def a_run(self, *args, **kwargs):
        """Override run method to compress before processing."""
        # Compress conversation before running
        if hasattr(self, '_messages') and self._messages:
            self._messages = self.compress_conversation(self._messages)

        return await super().a_run(*args, **kwargs)

    async def run(self, *args, **kwargs):
        """Override run method to compress before processing."""
        # Check if we have messages to compress
        if hasattr(self, '_chat_messages') and self._chat_messages:
            original_count = len(self._chat_messages)
            original_tokens = self.count_messages_tokens(self._chat_messages)

            # Compress if needed
            if original_tokens > self.compression_threshold:
                self._chat_messages = self.compress_conversation(self._chat_messages)
                new_count = len(self._chat_messages)
                new_tokens = self.count_messages_tokens(self._chat_messages)

                print(f"🗜️ PRE-RUN COMPRESSION: {original_count} → {new_count} messages, {original_tokens} → {new_tokens} tokens")

        return await super().run(*args, **kwargs)


class ContextCompressionService:
    """Service for managing context compression across the system."""

    @staticmethod
    def create_compressed_agent(agent_config, model_client, tools,
                               max_tokens: int = 100000,
                               compression_ratio: float = 0.7) -> CompressedAssistantAgent:
        """Factory method to create compressed agents."""
        return CompressedAssistantAgent(
            name=agent_config.name,
            model_client=model_client,
            system_message=agent_config.system_message,
            tools=tools,
            max_tokens=max_tokens,
            compression_ratio=compression_ratio
        )

    @staticmethod
    def get_compression_stats(agent: CompressedAssistantAgent) -> Dict[str, Any]:
        """Get compression statistics for an agent."""
        if not hasattr(agent, '_chat_messages'):
            return {"status": "no_messages"}

        current_tokens = agent.count_messages_tokens(agent._chat_messages)
        return {
            "current_tokens": current_tokens,
            "max_tokens": agent.max_tokens,
            "threshold": agent.compression_threshold,
            "utilization": current_tokens / agent.max_tokens,
            "needs_compression": current_tokens > agent.compression_threshold,
            "message_count": len(agent._chat_messages)
        }