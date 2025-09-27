"""
Comprehensive logging and metrics system for agent operations.
Separates decision logic visibility from user interface.
"""
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class RoutingDecision:
    """Structured data for routing decisions"""
    question: str
    agent_type: str
    confidence: float
    reasoning: str
    expressions_found: List[str]
    needs_calculation: bool
    is_conceptual: bool
    routing_time_ms: float


@dataclass
class TokenUsage:
    """Token usage tracking per request"""
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    timestamp: datetime


@dataclass
class AgentExecution:
    """Agent execution metrics"""
    request_id: str
    agent_type: str
    question: str
    response_length: int
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MetricsCollector:
    """Collects and manages system metrics"""
    
    # OpenAI Pricing (per 1000 tokens) - Updated September 2025
    # NOTE: Output tokens are significantly more expensive than input tokens!
    MODEL_COSTS = {
        "gpt-4o-mini": {
            "input": 0.00015,   # $0.15 per 1M INPUT tokens
            "output": 0.0006,   # $0.60 per 1M OUTPUT tokens (4x more expensive!)
            "ratio": "1:4"      # Output costs 4x more than input
        },
        "gpt-4o": {
            "input": 0.0025,    # $2.50 per 1M INPUT tokens
            "output": 0.01,     # $10.00 per 1M OUTPUT tokens (4x more expensive!)
            "ratio": "1:4"
        },
        "gpt-4-turbo": {
            "input": 0.01,      # $10.00 per 1M INPUT tokens
            "output": 0.03,     # $30.00 per 1M OUTPUT tokens (3x more expensive!)
            "ratio": "1:3"
        },
        "gpt-4": {
            "input": 0.03,      # $30.00 per 1M INPUT tokens  
            "output": 0.06,     # $60.00 per 1M OUTPUT tokens (2x more expensive!)
            "ratio": "1:2"
        },
        "gpt-3.5-turbo": {
            "input": 0.0015,    # $1.50 per 1M INPUT tokens
            "output": 0.002,    # $2.00 per 1M OUTPUT tokens (1.33x more expensive)
            "ratio": "1:1.33"
        }
    }
    
    def __init__(self):
        self.routing_decisions: List[RoutingDecision] = []
        self.token_usage: List[TokenUsage] = []
        self.agent_executions: List[AgentExecution] = []
        self._lock = threading.Lock()
        self._request_counter = 0
    
    def generate_request_id(self) -> str:
        """Generate unique request ID"""
        with self._lock:
            self._request_counter += 1
            return f"req_{int(time.time())}_{self._request_counter}"
    
    def log_routing_decision(self, decision: RoutingDecision):
        """Log routing decision with full context"""
        with self._lock:
            self.routing_decisions.append(decision)
    
    def log_token_usage(self, usage: TokenUsage):
        """Log token usage with cost calculation"""
        with self._lock:
            self.token_usage.append(usage)
    
    def log_agent_execution(self, execution: AgentExecution):
        """Log agent execution metrics"""
        with self._lock:
            self.agent_executions.append(execution)
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Dict:
        """Calculate detailed cost breakdown for token usage"""
        if model not in self.MODEL_COSTS:
            return {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "cost_ratio": "unknown",
                "error": f"Unknown model: {model}"
            }
        
        costs = self.MODEL_COSTS[model]
        input_cost = (prompt_tokens / 1000) * costs["input"]
        output_cost = (completion_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "input_rate_per_1k": costs["input"],
            "output_rate_per_1k": costs["output"],
            "cost_ratio": costs.get("ratio", "unknown"),
            "model": model
        }
    
    def get_total_costs(self) -> Dict[str, float]:
        """Get total costs by model"""
        costs = {}
        for usage in self.token_usage:
            if usage.model not in costs:
                costs[usage.model] = 0.0
            costs[usage.model] += usage.estimated_cost_usd
        return costs
    
    def get_total_tokens(self) -> Dict[str, Dict[str, int]]:
        """Get total token usage by model"""
        tokens = {}
        for usage in self.token_usage:
            if usage.model not in tokens:
                tokens[usage.model] = {"prompt": 0, "completion": 0, "total": 0}
            tokens[usage.model]["prompt"] += usage.prompt_tokens
            tokens[usage.model]["completion"] += usage.completion_tokens
            tokens[usage.model]["total"] += usage.total_tokens
        return tokens
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing decision statistics"""
        if not self.routing_decisions:
            return {}
        
        agents = [d.agent_type for d in self.routing_decisions]
        agent_counts = {agent: agents.count(agent) for agent in set(agents)}
        
        avg_confidence = sum(d.confidence for d in self.routing_decisions) / len(self.routing_decisions)
        avg_routing_time = sum(d.routing_time_ms for d in self.routing_decisions) / len(self.routing_decisions)
        
        return {
            "total_requests": len(self.routing_decisions),
            "agent_distribution": agent_counts,
            "average_confidence": avg_confidence,
            "average_routing_time_ms": avg_routing_time
        }
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent activity for dashboard"""
        recent = []
        
        # Combine recent executions with routing decisions
        for execution in self.agent_executions[-limit:]:
            # Find corresponding routing decision
            routing = next((r for r in self.routing_decisions if execution.question == r.question), None)
            
            activity = {
                "request_id": execution.request_id,
                "timestamp": execution.timestamp.isoformat(),
                "question": execution.question[:100] + "..." if len(execution.question) > 100 else execution.question,
                "agent_type": execution.agent_type,
                "success": execution.success,
                "execution_time_ms": execution.execution_time_ms,
                "confidence": routing.confidence if routing else 0.0,
                "response_length": execution.response_length
            }
            recent.append(activity)
        
        return recent[::-1]  # Most recent first


class StructuredLogger:
    """Structured logging system for agent operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with JSON formatting"""
        logger = logging.getLogger("agentic_system")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler for structured logs
        file_handler = logging.FileHandler("agentic_system.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_routing_decision(self, decision: RoutingDecision):
        """Log routing decision"""
        self.metrics.log_routing_decision(decision)
        
        log_data = {
            "event": "routing_decision",
            "agent_type": decision.agent_type,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "routing_time_ms": decision.routing_time_ms
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_agent_start(self, request_id: str, agent_type: str, question: str):
        """Log agent execution start"""
        log_data = {
            "event": "agent_start",
            "request_id": request_id,
            "agent_type": agent_type,
            "question_length": len(question)
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_agent_complete(self, execution: AgentExecution):
        """Log agent execution completion"""
        self.metrics.log_agent_execution(execution)
        
        log_data = {
            "event": "agent_complete",
            "request_id": execution.request_id,
            "agent_type": execution.agent_type,
            "success": execution.success,
            "execution_time_ms": execution.execution_time_ms,
            "response_length": execution.response_length
        }
        
        if execution.error_message:
            log_data["error"] = execution.error_message
        
        self.logger.info(json.dumps(log_data))
    
    def log_token_usage(self, usage: TokenUsage):
        """Log token usage and costs"""
        self.metrics.log_token_usage(usage)
        
        log_data = {
            "event": "token_usage",
            "request_id": usage.request_id,
            "model": usage.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost_usd": usage.estimated_cost_usd
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_system_event(self, event: str, data: Dict):
        """Log general system events"""
        log_data = {
            "event": event,
            **data
        }
        
        self.logger.info(json.dumps(log_data))


# Global metrics collector instance
metrics_collector = MetricsCollector()
structured_logger = StructuredLogger(metrics_collector)