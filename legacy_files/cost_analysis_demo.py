"""
OpenAI Cost Analysis Demo - Shows the critical difference between input and output pricing
"""
from logging_system import MetricsCollector

def demonstrate_input_output_cost_difference():
    """Show how input vs output token costs differ dramatically"""
    
    metrics = MetricsCollector()
    
    print("💰 OPENAI INPUT vs OUTPUT TOKEN COST ANALYSIS")
    print("=" * 60)
    
    # Example scenarios
    scenarios = [
        {
            "name": "Simple Math Question",
            "prompt_tokens": 10,    # "What is 2+3?"
            "output_tokens": 20     # Short response
        },
        {
            "name": "Code Generation Request", 
            "prompt_tokens": 50,    # Detailed request
            "output_tokens": 500    # Large code block
        },
        {
            "name": "Engineering Analysis",
            "prompt_tokens": 100,   # Complex problem
            "output_tokens": 800    # Detailed explanation
        }
    ]
    
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
    
    for model in models:
        print(f"\n🤖 MODEL: {model.upper()}")
        print("-" * 40)
        
        model_costs = metrics.MODEL_COSTS[model]
        print(f"📥 Input cost:  ${model_costs['input']:.6f} per 1K tokens")
        print(f"📤 Output cost: ${model_costs['output']:.6f} per 1K tokens")
        print(f"⚖️  Ratio: {model_costs['ratio']} (output vs input)")
        print()
        
        for scenario in scenarios:
            cost_breakdown = metrics.calculate_cost(
                model,
                scenario["prompt_tokens"],
                scenario["output_tokens"]
            )
            
            print(f"📋 {scenario['name']}:")
            print(f"   📥 Input:  {scenario['prompt_tokens']:3d} tokens × ${cost_breakdown['input_rate_per_1k']:.6f} = ${cost_breakdown['input_cost']:.8f}")
            print(f"   📤 Output: {scenario['output_tokens']:3d} tokens × ${cost_breakdown['output_rate_per_1k']:.6f} = ${cost_breakdown['output_cost']:.8f}")
            print(f"   💵 Total: ${cost_breakdown['total_cost']:.8f}")
            
            # Show percentage breakdown
            if cost_breakdown['total_cost'] > 0:
                input_pct = (cost_breakdown['input_cost'] / cost_breakdown['total_cost']) * 100
                output_pct = (cost_breakdown['output_cost'] / cost_breakdown['total_cost']) * 100
                print(f"   📊 Input: {input_pct:.1f}% | Output: {output_pct:.1f}%")
            print()

def show_cost_optimization_tips():
    """Show how understanding input/output costs helps optimize usage"""
    
    print("\n🎯 COST OPTIMIZATION INSIGHTS")
    print("=" * 60)
    
    print("""
💡 KEY INSIGHTS:

1. OUTPUT TOKENS ARE MUCH MORE EXPENSIVE!
   - GPT-4o-mini: Output costs 4x more than input
   - GPT-4o: Output costs 4x more than input  
   - GPT-4: Output costs 2x more than input

2. COST OPTIMIZATION STRATEGIES:
   ✅ Use shorter, more focused prompts
   ✅ Ask for concise responses when possible
   ✅ Use cheaper models (gpt-4o-mini) for simple tasks
   ✅ Batch multiple questions in one request
   ✅ Cache results to avoid repeat generation

3. ENGINEERING APPLICATIONS:
   - For code generation: High output cost (500+ tokens)
   - For validation: Low output cost (yes/no answers)  
   - For explanations: Medium output cost (100-300 tokens)

4. BUDGET PLANNING:
   - Simple queries: ~$0.00001 per request
   - Code generation: ~$0.0003 per request
   - Complex analysis: ~$0.0006 per request
    """)

if __name__ == "__main__":
    demonstrate_input_output_cost_difference()
    show_cost_optimization_tips()