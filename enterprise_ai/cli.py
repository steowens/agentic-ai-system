"""
Command Line Interface for Enterprise AI System
"""
import argparse
import asyncio
import sys
from pathlib import Path

from enterprise_ai import EnterpriseAISystem, get_system_status
from enterprise_ai.learning import routing_engine, feedback_collector
from enterprise_ai.monitoring import get_cost_summary
from enterprise_ai.dashboard import start_dashboard


async def process_single_question(question: str):
    """Process a single question and return results"""
    system = EnterpriseAISystem()
    await system.initialize()
    
    result = await system.process_question(question)
    
    print(f"Question: {question}")
    print(f"Agent: {result['agent_used']}")
    print(f"Confidence: {result['routing_confidence']:.3f}")
    print(f"Response: {result['response']}")
    print(f"Time: {result['processing_time']:.3f}s")
    print(f"Cost: ${result['cost']:.6f}")


async def interactive_mode():
    """Start interactive question-answer mode"""
    system = EnterpriseAISystem()
    await system.initialize()
    
    print("Enterprise AI Interactive Mode")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            question = input("\n> ").strip()
            
            if question.lower() in ['exit', 'quit']:
                break
            elif question.lower() == 'help':
                print("Commands:")
                print("  help     - Show this help")
                print("  status   - Show system status")
                print("  stats    - Show usage statistics")
                print("  exit     - Exit interactive mode")
                continue
            elif question.lower() == 'status':
                status = get_system_status()
                print(f"Status: {status}")
                continue
            elif question.lower() == 'stats':
                stats = get_cost_summary()
                print(f"Statistics: {stats}")
                continue
            elif not question:
                continue
            
            result = await system.process_question(question)
            
            print(f"Agent: {result['agent_used']} (confidence: {result['routing_confidence']:.3f})")
            print(f"Response: {result['response']}")
            print(f"Time: {result['processing_time']:.3f}s | Cost: ${result['cost']:.6f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def train_models():
    """Train ML routing models"""
    try:
        print("Training ML routing models...")
        routing_engine.ml_engine.train_models()
        print("Training completed successfully!")
        
        # Show model performance
        if routing_engine.ml_engine.is_trained:
            print(f"Best model: {routing_engine.ml_engine.best_model_name}")
            print(f"Accuracy: {routing_engine.ml_engine.model_accuracy:.3f}")
    except Exception as e:
        print(f"Training failed: {e}")


def show_statistics():
    """Show system usage statistics"""
    print("=== Cost Statistics ===")
    cost_stats = get_cost_summary()
    print(f"Total cost: ${cost_stats['total_cost']:.6f}")
    print(f"Total requests: {cost_stats['total_requests']}")
    print(f"Average cost per request: ${cost_stats['avg_cost_per_request']:.6f}")
    
    if cost_stats['agent_breakdown']:
        print("\nAgent Cost Breakdown:")
        for agent, data in cost_stats['agent_breakdown'].items():
            print(f"  {agent}: ${data['total_cost']:.6f} ({data['request_count']} requests)")
    
    print("\n=== Feedback Statistics ===")
    feedback_stats = feedback_collector.get_performance_summary()
    print(f"Total feedback entries: {feedback_stats.get('total_feedback', 0)}")
    print(f"Average rating: {feedback_stats.get('avg_rating', 0):.2f}/5")
    print(f"Success rate: {feedback_stats.get('success_rate', 0):.1f}%")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enterprise AI Routing System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  enterprise-ai --question "What is 2 + 2?"
  enterprise-ai --interactive
  enterprise-ai --train
  enterprise-ai --dashboard
  enterprise-ai --stats
        """
    )
    
    parser.add_argument(
        '--question', '-q',
        help='Process a single question'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train ML routing models'
    )
    
    parser.add_argument(
        '--dashboard', '-d',
        action='store_true',
        help='Start web dashboard'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show usage statistics'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version information'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        if args.version:
            status = get_system_status()
            print(f"Enterprise AI System v{status['version']}")
            print(f"Components: {', '.join(status['components'])}")
        
        elif args.question:
            asyncio.run(process_single_question(args.question))
        
        elif args.interactive:
            asyncio.run(interactive_mode())
        
        elif args.train:
            train_models()
        
        elif args.dashboard:
            print("Starting web dashboard at http://localhost:8000")
            print("Press Ctrl+C to stop")
            asyncio.run(start_dashboard())
        
        elif args.stats:
            show_statistics()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()