"""
Simplified system orchestrator that works with current package structure.
"""
import os
import time
from typing import Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient

from .agents import AgentFactory, AgentConfigurationProvider
from ..integrations import ToolFactory
from .routing import RoutingService, QuestionAnalyzer


class SystemOrchestrator:
    """
    Main system coordinator - simplified version that works
    """
    
    def __init__(self):
        load_dotenv()
        
        # Initialize core components (no server-side tracking)
        self.model_client = self._create_model_client()
        self.tools = self._create_tools()
        # Configure compression settings
        enable_compression = os.getenv("ENTERPRISE_AI_ENABLE_COMPRESSION", "true").lower() == "true"
        max_tokens = int(os.getenv("ENTERPRISE_AI_MAX_TOKENS", "100000"))

        self.agent_factory = AgentFactory(
            self.model_client,
            self.tools,
            enable_compression=enable_compression,
            max_tokens=max_tokens
        )
        self.routing_service = RoutingService()
        
        # Create agents
        self.agents = self._create_agents()
        
        print("✅ SystemOrchestrator initialized")
    
    def _create_model_client(self) -> OpenAIChatCompletionClient:
        """Create OpenAI model client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-openai-api-key":
            print("⚠️  Warning: No valid OpenAI API key found")
            # Return a mock client for testing
            return None
        
        model = os.getenv("ENTERPRISE_AI_MODEL", "gpt-4o-mini")
        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key
        )
    
    def _create_tools(self) -> Dict[str, Any]:
        """Create tools"""
        return ToolFactory.get_all_tools()
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create all configured agents"""
        if not self.model_client:
            return {}
        
        configs = AgentConfigurationProvider()
        
        return {
            "math": self.agent_factory.create_agent(configs.get_math_agent_config()),
            "system": self.agent_factory.create_agent(configs.get_system_agent_config()),
            "general": self.agent_factory.create_agent(configs.get_general_agent_config()),
            "wordle": self.agent_factory.create_agent(configs.get_wordle_agent_config())
        }
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the routing system"""
        start_time = time.time()
        
        # Route the question
        agent_type, reasoning, confidence = self.routing_service.get_routing_info(question)
        
        try:
            if not self.model_client:
                # Demo mode without API key
                response = f"[DEMO MODE] Would route to {agent_type} agent: {reasoning}"
                cost = 0.0
            else:
                # Get the appropriate agent
                agent = self.agents.get(agent_type, self.agents["general"])
                
                # Process with the agent
                result = await agent.run(task=question)
                
                # Extract clean text from AutoGen response
                if hasattr(result, 'messages') and result.messages:
                    # Get the last message content
                    last_message = result.messages[-1]
                    if hasattr(last_message, 'content'):
                        response = last_message.content
                    else:
                        response = str(last_message)
                else:
                    response = str(result)
                
                # Estimate cost (simplified)
                input_tokens = len(question.split()) * 1.3  # Rough estimate
                output_tokens = len(response.split()) * 1.3
                cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000
            
            processing_time = time.time() - start_time
            
            # No server-side tracking - client handles all metrics via localStorage
            
            return {
                "response": response,
                "agent_type": agent_type,
                "agent_used": agent_type,  # Keep for backward compatibility
                "routing_reasoning": reasoning,
                "reasoning": reasoning,  # Frontend expects this
                "routing_confidence": confidence,
                "confidence": confidence,  # Frontend expects this
                "processing_time": processing_time,
                "execution_time_ms": processing_time * 1000,  # Frontend expects this in ms
                "cost": cost,
                "token_usage": {
                    "total_tokens": int((len(question.split()) + len(response.split())) * 1.3),
                    "prompt_tokens": int(len(question.split()) * 1.3),
                    "completion_tokens": int(len(response.split()) * 1.3),
                    "estimated_cost_usd": cost,
                    "input_cost_usd": cost * 0.25,  # Rough estimate
                    "output_cost_usd": cost * 0.75,  # Rough estimate
                    "input_rate_per_1k": 0.00015,
                    "output_rate_per_1k": 0.0006,
                    "cost_ratio": "4:1"
                } if cost > 0 else None,
                "request_id": f"req_{int(time.time())}"
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "response": f"Error processing question: {str(e)}",
                "agent_type": agent_type,
                "agent_used": agent_type,
                "routing_reasoning": reasoning,
                "reasoning": reasoning,
                "routing_confidence": confidence,
                "confidence": confidence,
                "processing_time": processing_time,
                "execution_time_ms": processing_time * 1000,
                "cost": 0.0,
                "token_usage": None,
                "request_id": f"req_{int(time.time())}"
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get system metrics summary for dashboard"""
        total_cost = getattr(self, '_total_cost', 0.0)
        agent_usage = getattr(self, '_agent_usage', {})
        total_queries = getattr(self, '_total_queries', 0)
        
        return {
            # Frontend expects these specific fields
            "total_costs": {"total": total_cost},  # Frontend sums values
            "total_tokens": {"total": total_queries * 100},  # Rough estimate
            "routing_stats": {
                "total_requests": total_queries,
                "average_confidence": 0.85,  # Default confidence
                "agent_distribution": agent_usage
            },
            "recent_activity": [],  # Empty for now
            
            # Keep original fields for compatibility
            "total_queries": total_queries,
            "total_cost": total_cost,
            "avg_processing_time": getattr(self, '_avg_processing_time', 0.0),
            "agent_usage": agent_usage,
            "system_status": "operational", 
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }


class EnterpriseAISystem:
    """
    Simplified enterprise AI system that works with current package
    """
    
    def __init__(self):
        self.orchestrator = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the system"""
        if not self.initialized:
            self.orchestrator = SystemOrchestrator()
            self.initialized = True
            print("✅ EnterpriseAISystem initialized")
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question"""
        if not self.initialized:
            await self.initialize()
        
        return await self.orchestrator.process_question(question)