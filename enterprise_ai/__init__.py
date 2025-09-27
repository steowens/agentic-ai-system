"""
Enterprise AI Routing System
============================

A production-grade, self-improving AI orchestration system that intelligently routes 
requests to specialized backends including databases, ESRI services, and LLM agents.

Key Features:
- Intelligent routing with machine learning optimization
- Zero-token-cost operations via MCP integrations
- Comprehensive cost tracking and optimization
- Real-time feedback collection and system improvement
- Enterprise security controls and data classification
- Web-based monitoring dashboard with live updates

Quick Start:
-----------
```python
from enterprise_ai import EnterpriseAISystem

# Initialize the system
ai_system = EnterpriseAISystem()
await ai_system.initialize()

# Process requests
response = await ai_system.process_request(
    "Show me a map of California", 
    session_id="user_001"
)

# Submit feedback for continuous improvement
await ai_system.submit_feedback(
    request_id="req_001",
    session_id="user_001", 
    rating=5,
    comment="Perfect map generation!"
)
```

Modules:
--------
- core: Core orchestration and routing logic
- integrations: MCP connectors for databases, ESRI, APIs
- monitoring: Cost tracking, performance metrics, feedback collection  
- security: Data classification, sanitization, access controls
- learning: Machine learning for routing optimization
- dashboard: Web interface for monitoring and management
"""

__version__ = "1.0.0"
__author__ = "Enterprise AI Team"
__license__ = "MIT"

# Core exports for easy access
from .core.orchestrator import SystemOrchestrator, EnterpriseAISystem
from .core.agents import AgentFactory, AgentConfig
from .integrations import ToolFactory, BaseTool

from .integrations.mcp_framework import mcp_orchestrator, process_mcp_request
from .monitoring.metrics import MetricsCollector

from .learning.feedback import feedback_collector
from .learning.routing_engine import routing_engine

# Package-level configuration
class Config:
    """Global configuration for the Enterprise AI system"""
    
    # Default database paths
    FEEDBACK_DB = "enterprise_ai_feedback.db"
    COST_TRACKING_DB = "enterprise_ai_costs.db"
    LEARNING_DB = "enterprise_ai_learning.db"
    
    # Default model settings
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Security settings
    DEFAULT_SECURITY_LEVEL = "balanced"
    ENABLE_DATA_CLASSIFICATION = True
    ENABLE_AUDIT_LOGGING = True
    
    # Performance settings
    DEFAULT_TIMEOUT = 30
    MAX_CONCURRENT_REQUESTS = 100
    ENABLE_CACHING = True
    
    # Dashboard settings
    DASHBOARD_HOST = "0.0.0.0"
    DASHBOARD_PORT = 8000
    ENABLE_WEBSOCKETS = True

# Convenience functions
def create_ai_system():
    """
    Create and configure an Enterprise AI System instance.
    
    Returns:
        EnterpriseAISystem: Configured AI system instance
    """
    return EnterpriseAISystem()

def get_version():
    """Get the current package version"""
    return __version__

def get_package_info():
    """Get comprehensive package information"""
    return {
        "name": "enterprise_ai",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "modules": [
            "core",
            "integrations", 
            "monitoring",
            "security",
            "learning",
            "dashboard"
        ],
        "capabilities": [
            "Intelligent agent routing",
            "MCP integrations (ESRI, databases)",
            "Multi-modal cost tracking", 
            "Machine learning optimization",
            "Security controls",
            "Real-time monitoring",
            "Feedback-driven improvement"
        ]
    }

# Module imports for backward compatibility
# (Keep existing code working during transition)
try:
    from .legacy_imports import *
    import warnings
    warnings.warn(
        "Direct module imports are deprecated. Use enterprise_ai.core, "
        "enterprise_ai.integrations, etc. instead.",
        DeprecationWarning,
        stacklevel=2
    )
except ImportError:
    # Legacy imports not available, which is fine
    pass