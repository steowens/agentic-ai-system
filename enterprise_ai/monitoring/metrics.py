"""
Comprehensive metrics and logging system for enterprise AI operations.
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
    selected_agent: str
    confidence: float
    reasoning: str
    available_agents: List[str]
    question_analysis: Dict[str, Any]
    timestamp: datetime


@dataclass
class TokenUsage:
    """Token usage tracking"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: Optional[str] = None
    estimated_cost: Optional[float] = None


@dataclass
class AgentExecution:
    """Agent execution metrics"""
    agent_name: str
    question: str
    response: str
    processing_time: float
    token_usage: TokenUsage
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None


class StructuredLogger:
    """Structured logging for system operations"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger("enterprise_ai")
        self.logger.setLevel(log_level)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Log info message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra)}"
        self.logger.info(message)
    
    def error(self, message: str, extra: Optional[Dict] = None):
        """Log error message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra)}"
        self.logger.error(message)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Log debug message with optional structured data"""
        if extra:
            message = f"{message} | Data: {json.dumps(extra)}"
        self.logger.debug(message)


class MetricsCollector:
    """Comprehensive metrics collection and analysis"""
    
    def __init__(self):
        self.routing_decisions = []
        self.agent_executions = []
        self.token_usage_history = []
        self._lock = threading.Lock()
        
        # Enhanced cost models with input/output differentiation
        self.MODEL_COSTS = {
            "gpt-4o-mini": {
                "input_per_1k": 0.00015,
                "output_per_1k": 0.0006,
                "ratio": 4.0  # output is 4x more expensive than input
            },
            "gpt-4o": {
                "input_per_1k": 0.005,
                "output_per_1k": 0.015,
                "ratio": 3.0
            },
            "gpt-3.5-turbo": {
                "input_per_1k": 0.0005,
                "output_per_1k": 0.0015,
                "ratio": 3.0
            }
        }
    
    def log_routing_decision(self, decision: RoutingDecision):
        """Log routing decision for analysis"""
        with self._lock:
            self.routing_decisions.append(decision)
    
    def log_agent_execution(self, execution: AgentExecution):
        """Log agent execution metrics"""
        with self._lock:
            self.agent_executions.append(execution)
    
    def calculate_cost(self, token_usage: TokenUsage, model: str = "gpt-4o-mini") -> Dict[str, float]:
        """
        Calculate detailed cost breakdown with input/output pricing.
        Returns comprehensive cost analysis.
        """
        if model not in self.MODEL_COSTS:
            # Fallback to default model
            model = "gpt-4o-mini"
        
        costs = self.MODEL_COSTS[model]
        
        # Calculate individual costs
        input_cost = (token_usage.input_tokens / 1000) * costs["input_per_1k"]
        output_cost = (token_usage.output_tokens / 1000) * costs["output_per_1k"]
        total_cost = input_cost + output_cost
        
        # Calculate percentages
        input_percentage = (input_cost / total_cost * 100) if total_cost > 0 else 0
        output_percentage = (output_cost / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "model": model,
            "input_tokens": token_usage.input_tokens,
            "output_tokens": token_usage.output_tokens,
            "total_tokens": token_usage.total_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "input_percentage": round(input_percentage, 1),
            "output_percentage": round(output_percentage, 1),
            "cost_ratio": costs["ratio"],
            "cost_per_token": {
                "input": costs["input_per_1k"] / 1000,
                "output": costs["output_per_1k"] / 1000
            }
        }
    
    def get_routing_accuracy(self, recent_count: int = 100) -> Dict[str, Any]:
        """Calculate routing accuracy metrics"""
        if not self.routing_decisions:
            return {"accuracy": 0.0, "sample_size": 0}
        
        recent_decisions = self.routing_decisions[-recent_count:]
        
        # Calculate average confidence
        avg_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions)
        
        # Analyze confidence distribution
        high_confidence = len([d for d in recent_decisions if d.confidence > 0.8])
        medium_confidence = len([d for d in recent_decisions if 0.5 < d.confidence <= 0.8])
        low_confidence = len([d for d in recent_decisions if d.confidence <= 0.5])
        
        return {
            "average_confidence": round(avg_confidence, 3),
            "sample_size": len(recent_decisions),
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence, 
                "low": low_confidence
            }
        }
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Analyze agent performance metrics"""
        if not self.agent_executions:
            return {}
        
        performance = {}
        
        for execution in self.agent_executions:
            agent = execution.agent_name
            
            if agent not in performance:
                performance[agent] = {
                    "executions": 0,
                    "total_time": 0.0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "success_count": 0
                }
            
            perf = performance[agent]
            perf["executions"] += 1
            perf["total_time"] += execution.processing_time
            perf["total_tokens"] += execution.token_usage.total_tokens
            
            if execution.token_usage.estimated_cost:
                perf["total_cost"] += execution.token_usage.estimated_cost
            
            if execution.success:
                perf["success_count"] += 1
        
        # Calculate averages
        for agent, perf in performance.items():
            if perf["executions"] > 0:
                perf["avg_time"] = round(perf["total_time"] / perf["executions"], 2)
                perf["avg_tokens"] = round(perf["total_tokens"] / perf["executions"])
                perf["avg_cost"] = round(perf["total_cost"] / perf["executions"], 4)
                perf["success_rate"] = round(perf["success_count"] / perf["executions"] * 100, 1)
        
        return performance
    
    def get_cost_analysis(self, recent_count: int = 100) -> Dict[str, Any]:
        """Comprehensive cost analysis"""
        if not self.agent_executions:
            return {"total_cost": 0.0, "sample_size": 0}
        
        recent_executions = self.agent_executions[-recent_count:]
        
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for execution in recent_executions:
            if execution.token_usage.estimated_cost:
                total_cost += execution.token_usage.estimated_cost
            
            total_input_tokens += execution.token_usage.input_tokens
            total_output_tokens += execution.token_usage.output_tokens
        
        return {
            "total_cost": round(total_cost, 4),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "sample_size": len(recent_executions),
            "avg_cost_per_request": round(total_cost / len(recent_executions), 4) if recent_executions else 0.0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "routing_accuracy": self.get_routing_accuracy(),
            "agent_performance": self.get_agent_performance(),
            "cost_analysis": self.get_cost_analysis(),
            "total_decisions": len(self.routing_decisions),
            "total_executions": len(self.agent_executions)
        }


# Global instances for backward compatibility
metrics_collector = MetricsCollector()
structured_logger = StructuredLogger()