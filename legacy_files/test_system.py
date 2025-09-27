"""
Simple test script for the refactored system
"""
from main import SystemOrchestrator

async def test_system():
    """Test the system with a few sample questions"""
    try:
        # Initialize the system
        system = SystemOrchestrator()
        
        print("🤖 INTELLIGENT AGENT SYSTEM TEST")
        print("="*50)
        
        # Test questions
        test_questions = [
            "What is 2 + 3?",
            "What is the integral of x^2?", 
            "List files in current directory",
            "What is machine learning?"
        ]
        
        for question in test_questions:
            print(f"\n❓ TESTING: {question}")
            print("-" * 30)
            
            result = await system.process_question(question, verbose=True)
            response_text = result.get('response', str(result))
            print(f"🤖 RESPONSE: {response_text[:100]}...")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_system())