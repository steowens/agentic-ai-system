"""
Main system orchestrator with enhanced enterprise features.
Integrates all components with SOLID principles and comprehensive monitoring.
"""
import os
import time
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.tools import AgentTool

# Import from our organized packages
from ..monitoring.metrics import MetricsCollector
from ..learning.feedback import feedback_collector
from ..learning.routing_engine import routing_engine
from ..integrations.mcp_framework import mcp_orchestrator, process_mcp_request

from .agents import AgentFactory, AgentConfigurationProvider
from ..integrations.tools import ToolFactory, BaseTool
from .routing import RoutingService, QuestionAnalyzer


class SystemOrchestrator:
    """
    Main system coordinator following SOLID principles:
    - Single Responsibility: Coordinates system components
    - Open/Closed: Extensible through dependency injection
    - Liskov Substitution: Works with any tool/agent implementations
    - Interface Segregation: Depends only on needed abstractions
    - Dependency Inversion: Depends on abstractions, not concretions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        # Initialize logging and metrics
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger()
        
        # Initialize core dependencies
        self.model_client = self._create_model_client(api_key)
        self.tools = self._create_tools()
        self.agent_factory = AgentFactory(self.model_client, self.tools)
        self.routing_service = RoutingService()
        self.question_analyzer = QuestionAnalyzer()
        
        # Create agent tools for delegation
        self.agents = self._create_agents()
    
    def _create_model_client(self, api_key: Optional[str] = None) -> OpenAIChatCompletionClient:
        """Create OpenAI model client with error handling"""
        try:
            final_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not final_api_key:
                raise ValueError("OPENAI_API_KEY must be provided either as parameter or environment variable")
            
            return OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                api_key=final_api_key
            )
        except Exception as e:
            self.logger.error(f"Failed to create model client: {e}")
            raise
    
    def _create_tools(self) -> Dict[str, BaseTool]:
        """Create and configure system tools"""
        tool_factory = ToolFactory()
        return {
            "math_calculator": tool_factory.create_tool("math_calculator"),
            "file_system": tool_factory.create_tool("file_system")
        }
    
    def _create_agents(self) -> Dict[str, AgentTool]:
        """Create agent tools for the routing system"""
        config_provider = AgentConfigurationProvider()
        
        agents = {}
        for agent_type in ["math", "system", "general"]:
            config = config_provider.get_configuration(agent_type)
            agent = self.agent_factory.create_agent(config)
            
            # Wrap agent as a tool for routing
            agents[f"{agent_type}_agent"] = AgentTool(
                name=f"{agent_type}_agent",
                description=config.description,
                func=lambda question, agent=agent: self._execute_agent(agent, question)
            )
        
        return agents
    
    def _execute_agent(self, agent, question: str) -> str:
        """Execute agent with comprehensive tracking"""
        start_time = time.time()
        
        try:
            # Execute the agent
            result = agent.run(
                task=question,
                termination_condition=lambda messages: len(messages) >= 10
            )
            
            processing_time = time.time() - start_time
            
            # Extract response and token usage
            if hasattr(result, 'messages') and result.messages:
                response = result.messages[-1].content if result.messages[-1].content else "No response generated"
            else:
                response = str(result) if result else "No response generated"
            
            # Track token usage
            token_usage = self._estimate_token_usage(question, response)
            
            # Log execution
            execution = AgentExecution(
                agent_name=agent.name if hasattr(agent, 'name') else "unknown_agent",
                question=question,
                response=response,
                processing_time=processing_time,
                token_usage=token_usage,
                timestamp=datetime.now(),
                success=True
            )
            
            self.metrics.log_agent_execution(execution)
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Agent execution failed: {str(e)}"
            
            # Log failed execution
            execution = AgentExecution(
                agent_name=agent.name if hasattr(agent, 'name') else "unknown_agent",
                question=question,
                response=error_msg,
                processing_time=processing_time,
                token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
            
            self.metrics.log_agent_execution(execution)
            self.logger.error(f"Agent execution error: {e}")
            return error_msg
    
    def _estimate_token_usage(self, question: str, response: str) -> TokenUsage:
        """Estimate token usage (simplified - would use tiktoken in production)"""
        # Simple word-based estimation (4 chars ≈ 1 token)
        input_tokens = len(question) // 4
        output_tokens = len(response) // 4
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process user question with comprehensive routing and tracking.
        Enhanced with better cost tracking and routing decisions.
        """
        start_time = time.time()
        
        try:
            # Analyze question
            analysis = self.question_analyzer.analyze_question(question)
            
            # Get routing decision with confidence
            routing_decision = self.routing_service.route_question(question, self.agents)
            
            # Log routing decision
            self.metrics.log_routing_decision(RoutingDecision(
                question=question,
                selected_agent=routing_decision['agent_name'],
                confidence=routing_decision['confidence'],
                reasoning=routing_decision['reasoning'],
                available_agents=list(self.agents.keys()),
                question_analysis=analysis,
                timestamp=datetime.now()
            ))
            
            # Execute selected agent
            selected_agent_tool = self.agents[routing_decision['agent_name']]
            response = selected_agent_tool.func(question)
            
            processing_time = time.time() - start_time
            
            # Calculate enhanced cost breakdown
            token_usage = self._estimate_token_usage(question, response)
            cost_breakdown = self.metrics.calculate_cost(token_usage, model="gpt-4o-mini")
            
            # Return comprehensive response
            return {
                "response": response,
                "agent_used": routing_decision['agent_name'],
                "routing_confidence": routing_decision['confidence'],
                "routing_reasoning": routing_decision['reasoning'],
                "question_analysis": analysis,
                "tokens_used": {
                    "input": token_usage.input_tokens,
                    "output": token_usage.output_tokens, 
                    "total": token_usage.total_tokens
                },
                "cost": cost_breakdown.get("total_cost", 0.0),
                "cost_breakdown": cost_breakdown,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_response = {
                "response": f"System error: {str(e)}",
                "agent_used": "error_handler",
                "routing_confidence": 0.0,
                "routing_reasoning": "Error occurred during processing",
                "tokens_used": {"input": 0, "output": 0, "total": 0},
                "cost": 0.0,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }
            
            self.logger.error(f"Question processing error: {e}")
            return error_response
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return self.metrics.get_summary()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "status": "operational",
            "agents_available": len(self.agents),
            "tools_available": len(self.tools),
            "model_client": "gpt-4o-mini",
            "uptime": "system_running",
            "metrics": self.get_metrics_summary()
        }


class EnterpriseAISystem:
    """
    Complete enterprise AI system integrating all components.
    This is the main entry point for the entire system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Core components
        self.base_orchestrator = SystemOrchestrator()
        self.secure_orchestrator = SecureAgentOrchestrator(self.base_orchestrator)
        self.feedback_collector = FeedbackCollector()
        self.learning_engine = RoutingLearningEngine()
        self.mcp_orchestrator = MCPOrchestrator()
        self.cost_tracker = MultiModalCostTracker()
        
        # Session management
        self.active_sessions = {}
        
        # Set security configuration
        security_level = self.config.get("security_level", "balanced")
        if security_level in SECURITY_CONFIGS:
            self.secure_orchestrator.security_config = SECURITY_CONFIGS[security_level]
    
    async def initialize(self):
        """Initialize all system components"""
        print("🚀 Initializing Enterprise AI System...")
        
        # Setup MCP integrations
        esri_url = self.config.get("esri_url", "https://services.arcgisonline.com/arcgis")
        self.mcp_orchestrator.register_esri_integration(esri_url)
        
        db_config = self.config.get("database", {"path": "engineering.db"})
        self.mcp_orchestrator.register_database_integration(db_config)
        
        await self.mcp_orchestrator.initialize_integrations()
        
        # Load or train learning models
        if not self.learning_engine.load_models():
            print("🎓 Training routing models...")
            self.learning_engine.train_models()
        
        print("✅ Enterprise AI System ready!")
    
    async def process_request(self, 
                            user_question: str,
                            session_id: str,
                            user_id: str = None) -> Dict[str, Any]:
        """Process user request with full enterprise features"""
        
        request_id = f"{session_id}_{len(self.active_sessions.get(session_id, []))}"
        
        # Get ML routing prediction
        ml_prediction = self.learning_engine.predict_agent(user_question)
        
        # Check if this should go to MCP instead of LLM
        mcp_route = await self._check_mcp_routing(user_question, ml_prediction)
        
        if mcp_route:
            # Handle via MCP (no token cost!)
            response = await self._handle_mcp_request(
                user_question, mcp_route, session_id, request_id
            )
        else:
            # Handle via secure LLM routing
            response = await self._handle_llm_request(
                user_question, ml_prediction, session_id, request_id
            )
        
        # Track costs and performance
        await self._track_request_metrics(response, session_id)
        
        # Store for feedback collection
        self._prepare_feedback_collection(response, request_id, session_id)
        
        return response
    
    async def _check_mcp_routing(self, question: str, ml_prediction) -> Optional[Dict]:
        """Check if question should be routed to MCP instead of LLM"""
        question_lower = question.lower()
        
        # ESRI/GIS routing
        if any(term in question_lower for term in ['map', 'gis', 'spatial', 'coordinate', 'elevation']):
            return {
                "type": "esri",
                "operation": "generate_map" if "show" in question_lower or "display" in question_lower else "query_features",
                "reasoning": "Geographic/spatial request - route to ESRI"
            }
        
        # Database routing
        if any(term in question_lower for term in ['database', 'query', 'records', 'table', 'data']):
            return {
                "type": "database", 
                "operation": "execute_procedure",
                "reasoning": "Database query request"
            }
        
        return None
    
    async def _handle_mcp_request(self, question: str, mcp_route: Dict, session_id: str, request_id: str) -> Dict:
        """Handle request via MCP (zero token cost)"""
        start_time = time.time()
        
        # Create appropriate MCP request
        if mcp_route["type"] == "esri":
            mcp_request = MCPRequest(
                request_id=request_id,
                resource_id="esri_map_service",
                operation=mcp_route["operation"],
                parameters={"extent": {"xmin": -120, "ymin": 35, "xmax": -119, "ymax": 36}, "layers": ["0"]}
            )
        elif mcp_route["type"] == "database":
            mcp_request = MCPRequest(
                request_id=request_id,
                resource_id="engineering_database", 
                operation="execute_procedure",
                parameters={"procedure": "get_engineering_data", "params": {"material": "steel"}}
            )
        
        # Execute MCP request
        mcp_response = await self.mcp_orchestrator.execute_mcp_request(mcp_request)
        processing_time = time.time() - start_time
        
        # Track MCP usage
        if mcp_route["type"] == "esri":
            self.cost_tracker.track_image_generation(
                session_id=session_id,
                provider="esri",
                image_count=1 if mcp_response.success else 0
            )
        elif mcp_route["type"] == "database":
            self.cost_tracker.track_database_operation(
                session_id=session_id,
                operation_type="mcp_query",
                query_count=1,
                record_count=len(mcp_response.content) if mcp_response.content else 0,
                response_time=processing_time
            )
        
        return {
            "success": mcp_response.success,
            "response": self._format_mcp_response(mcp_response),
            "agent_used": f"mcp_{mcp_route['type']}",
            "routing_confidence": 0.95,
            "response_type": mcp_response.response_type.value,
            "file_path": mcp_response.file_path,
            "processing_time": processing_time,
            "cost": 0.0,  # Zero token cost!
            "cost_breakdown": {"mcp_operation": 0.0, "token_savings": "High"},
            "routing_reasoning": mcp_route["reasoning"]
        }
    
    async def _handle_llm_request(self, question: str, ml_prediction, session_id: str, request_id: str) -> Dict:
        """Handle request via secure LLM routing"""
        secure_response = await self.secure_orchestrator.secure_process_question(question)
        
        if secure_response.get("blocked_by_security"):
            return secure_response
        
        # Track token usage
        if secure_response.get("tokens_used"):
            tokens = secure_response["tokens_used"]
            self.cost_tracker.track_token_usage(
                session_id=session_id,
                model="gpt-4o-mini",
                input_tokens=tokens.get("input", 0),
                output_tokens=tokens.get("output", 0)
            )
        
        return secure_response
    
    def _format_mcp_response(self, mcp_response) -> str:
        """Format MCP response for user"""
        if mcp_response.success:
            if mcp_response.response_type.value == "image":
                return f"Generated map image: {mcp_response.file_path}"
            elif mcp_response.response_type.value == "tabular_data":
                return f"Retrieved {len(mcp_response.content)} records from database"
            else:
                return "Successfully processed via MCP"
        else:
            return f"MCP Error: {mcp_response.error_message}"
    
    async def _track_request_metrics(self, response: Dict, session_id: str):
        """Track comprehensive request metrics"""
        pass  # Implementation would track detailed metrics
    
    def _prepare_feedback_collection(self, response: Dict, request_id: str, session_id: str):
        """Prepare data for feedback collection"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        
        self.active_sessions[session_id].append({
            "request_id": request_id,
            "response": response,
            "timestamp": datetime.now()
        })
    
    async def submit_feedback(self, 
                            request_id: str,
                            session_id: str, 
                            feedback_type: str,
                            rating: int = None,
                            comment: str = "",
                            suggested_agent: str = None) -> str:
        """Submit user feedback and trigger learning"""
        
        # Collect feedback
        if feedback_type == "quick":
            feedback_id = self.feedback_collector.collect_quick_feedback(
                question_id=request_id,
                session_id=session_id,
                is_positive=rating >= 4,
                routing_info={"agent_used": "system", "confidence": 0.8},
                response_info={"question": "", "response": "", "processing_time": 1.0, "cost": 0.001}
            )
        else:
            feedback_id = self.feedback_collector.collect_detailed_feedback(
                question_id=request_id,
                session_id=session_id,
                rating=rating,
                routing_correction=suggested_agent,
                user_comment=comment,
                what_worked="",
                what_failed="",
                improvement_suggestion="",
                routing_info={"agent_used": "system", "confidence": 0.8},
                response_info={"question": "", "response": "", "processing_time": 1.0, "cost": 0.001}
            )
        
        # Update learning engine
        self.learning_engine.update_from_feedback({
            "routing_outcome": "wrong_agent" if suggested_agent else "success",
            "suggested_agent": suggested_agent,
            "original_question": "User feedback"
        })
        
        return feedback_id
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "routing_engine": {
                "ml_trained": self.learning_engine.is_trained,
                "best_model": self.learning_engine.current_best_model,
            },
            "mcp_integrations": {
                "total_resources": len(self.mcp_orchestrator.resources)
            },
            "cost_tracking": {
                "total_sessions": len(self.active_sessions),
            },
            "feedback_system": {
                "collector_ready": True,
            },
            "security": {
                "data_classification": "active",
            }
        }