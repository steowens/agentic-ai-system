#!/usr/bin/env python3
"""
Test script for context compression functionality.
Simulates a long conversation to trigger compression.
"""
import asyncio
import os
from dotenv import load_dotenv
from enterprise_ai.core.orchestrator_simple import SystemOrchestrator

async def test_compression():
    """Test context compression with a simulated long conversation."""
    load_dotenv()

    # Set compression to a low threshold for testing
    os.environ["ENTERPRISE_AI_MAX_TOKENS"] = "1000"  # Very low for testing
    os.environ["ENTERPRISE_AI_ENABLE_COMPRESSION"] = "true"

    print("🧪 Testing Context Compression")
    print("=" * 50)

    # Initialize orchestrator
    orchestrator = SystemOrchestrator()

    # Test questions that will accumulate context
    test_questions = [
        "What is 2+2?",
        "What about 3+3?",
        "Can you calculate 5+5?",
        "What's 10+10?",
        "How about 15+15?",
        "What is 20+20?",
        "Can you tell me 25+25?",
        "What about 30+30?",
        "Calculate 35+35 please",
        "What is 40+40?",
        "Final question: what is 50+50?"
    ]

    print(f"Testing with {len(test_questions)} questions...")
    print("Watch for compression messages (🗜️)")
    print()

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")

        try:
            result = await orchestrator.process_question(question)

            if result.get('success', True):
                response = result.get('response', 'No response')
                agent_used = result.get('agent_used', 'unknown')

                print(f"🤖 Agent: {agent_used}")
                print(f"💬 Response: {response[:100]}{'...' if len(response) > 100 else ''}")

                # Try to get compression stats if available
                if hasattr(orchestrator, 'agent_factory'):
                    try:
                        agent = orchestrator.agents.get(agent_used)
                        if agent:
                            stats = orchestrator.agent_factory.get_compression_stats(agent)
                            if stats.get('status') != 'compression_disabled':
                                print(f"📊 Stats: {stats}")
                    except Exception as e:
                        print(f"⚠️  Could not get stats: {e}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Exception: {e}")
            break

        # Small delay to see output clearly
        await asyncio.sleep(0.5)

    print("\n🏁 Compression test completed!")

if __name__ == "__main__":
    asyncio.run(test_compression())