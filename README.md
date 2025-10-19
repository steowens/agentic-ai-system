# Enterprise AI Routing System

A production-grade AI routing system with MCP integrations, ML optimization, and comprehensive enterprise features.

## Features

- **Intelligent Agent Routing**: Smart routing based on question analysis and machine learning
- **MCP Integration**: Connect to databases, ESRI services, and other external systems
- **Cost Optimization**: Detailed token tracking and cost analysis with input/output pricing
- **Security Framework**: Data classification and sanitization for sensitive information
- **Feedback Learning**: ML-driven routing improvement based on user feedback
- **Real-time Dashboard**: Web-based monitoring and analytics interface
- **Enterprise Ready**: SOLID principles, comprehensive logging, and production monitoring

## Quick Start

### Installation

1. **Install the package:**
   ```bash
   pip install -e .
   ```

2. **Create a .env file with your OpenAI API key:**
   ```bash
   echo "OPENAI_API_KEY=your-openai-api-key" > .env
   ```
   
   Or create a `.env` file manually:
   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

3. **Start the system:**
   ```bash
   python main.py
   ```

### Usage

```bash
# Start the web interface (single entry point)
python main.py
```

The system will launch a web interface at:
- **Dashboard**: http://localhost:8000

## Package Structure

```
enterprise_ai/
├── __init__.py              # Main package exports and configuration
├── core/                    # Core orchestration and routing
│   ├── orchestrator.py      # SystemOrchestrator & EnterpriseAISystem
│   ├── agents.py           # Agent factory and configurations  
│   └── routing.py          # Intelligent routing service
├── integrations/           # External system connectors
│   ├── tools.py           # Math and filesystem tools
│   └── mcp_framework.py   # MCP orchestration framework
├── monitoring/            # Metrics and cost tracking
│   └── metrics.py        # MetricsCollector and cost analysis
├── security/             # Data protection and sanitization
│   └── data_security.py  # DataSanitizer and security controls
├── learning/            # ML routing and feedback
│   ├── feedback.py     # User feedback collection
│   └── routing_engine.py # ML routing optimization
└── dashboard/          # Web interface and monitoring
    └── web_app.py     # FastAPI dashboard application
```

## Programming Interface

### Basic Usage

```python
from enterprise_ai import EnterpriseAISystem

# Initialize system
system = EnterpriseAISystem()
await system.initialize()

# Process questions
result = await system.process_question("Calculate 25 * 17")

print(f"Agent: {result['agent_used']}")
print(f"Response: {result['response']}")
print(f"Cost: ${result['cost']:.6f}")
```

### MCP Integration

```python
from enterprise_ai.integrations import process_mcp_request

# Database query
db_result = await process_mcp_request(
    resource_id="main_database",
    operation="query", 
    parameters={"sql": "SELECT * FROM customers"}
)

# ESRI GIS query
gis_result = await process_mcp_request(
    resource_id="esri_feature_service",
    operation="get_features",
    parameters={"layer": "cities", "where": "population > 1000000"}
)
```

### Feedback and Learning

```python
from enterprise_ai.learning import feedback_collector, routing_engine

# Collect user feedback
feedback_collector.collect_simple_feedback(
    question="What is 2+2?",
    response="The result is 4",
    agent_used="math",
    rating=5,
    processing_time=1.2,
    cost=0.0001
)

# Train ML models
routing_engine.ml_engine.train_models()

# Get smart routing prediction
prediction = routing_engine.predict_optimal_routing("Calculate derivatives")
```

### Cost Monitoring

```python
from enterprise_ai.monitoring import get_cost_summary, MetricsCollector

# Get cost analysis
costs = get_cost_summary()
print(f"Total cost: ${costs['total_cost']:.6f}")

# Detailed metrics
collector = MetricsCollector()
summary = collector.get_summary()
```

### Security Features

```python
from enterprise_ai.security import DataSanitizer

sanitizer = DataSanitizer()

# Classify data sensitivity
classification = sanitizer.classify_sensitivity(user_input)

# Sanitize sensitive data  
clean_data = sanitizer.sanitize_data(user_input)
```

## Web Dashboard

Access the comprehensive web dashboard at `http://localhost:8000/dashboard`:

- **Real-time Metrics**: Live cost tracking and performance monitoring
- **Agent Performance**: Detailed breakdown by agent type
- **Feedback Analysis**: User satisfaction and routing accuracy
- **MCP Statistics**: External system integration metrics
- **Interactive Chat**: Test the system with mathematical formula rendering

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY="your-openai-api-key"

# Optional
ENTERPRISE_AI_LOG_LEVEL="INFO"
ENTERPRISE_AI_DB_PATH="./enterprise_data.db"  
ENTERPRISE_AI_MODEL="gpt-4o-mini"
```

### Programmatic Configuration

```python
from enterprise_ai import configure_system

configure_system(
    model="gpt-4o-mini",
    log_level="DEBUG",
    enable_security=True,
    enable_feedback=True
)
```

## Advanced Features

### Custom Agent Types

```python
from enterprise_ai.core import AgentConfig, AgentConfigurationProvider

# Define custom agent
custom_config = AgentConfig(
    name="engineering_agent",
    system_message="You are a structural engineering expert...",
    tools=["math", "file"]
)

# Register with system
system.orchestrator.agent_factory.create_agent(custom_config)
```

### Custom MCP Resources

```python
from enterprise_ai.integrations import MCPResource, MCPResourceType

# Define custom resource
custom_resource = MCPResource(
    resource_id="custom_api",
    name="Custom Engineering API", 
    resource_type=MCPResourceType.WEB_API,
    endpoint="https://api.engineering-system.com",
    description="Specialized engineering calculations"
)

# Register resource
mcp_orchestrator.register_resource(custom_resource)
```

### ML Model Customization

```python
from enterprise_ai.learning import MLRoutingEngine
from sklearn.ensemble import GradientBoostingClassifier

# Custom ML model
engine = MLRoutingEngine()
engine.models['gradient_boost'] = GradientBoostingClassifier()
engine.train_models()
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["enterprise-ai", "--dashboard"]
```

### Monitoring and Logging

The system provides comprehensive structured logging and metrics:

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
```

### Security Considerations

- **Data Classification**: Automatic detection of sensitive information
- **API Key Management**: Secure handling of credentials
- **Input Sanitization**: Protection against prompt injection
- **Audit Logging**: Comprehensive request/response logging

## Performance Optimization

### Cost Optimization

- **Input/Output Token Tracking**: Separate pricing for input vs output tokens
- **MCP Zero-Cost Operations**: Route suitable queries to external systems
- **Agent Selection**: Choose most cost-effective agent for each query type
- **Batch Processing**: Optimize for multiple related queries

### Caching

```python
from enterprise_ai import configure_caching

configure_caching(
    enable_response_cache=True,
    cache_ttl=3600,  # 1 hour
    max_cache_size=1000
)
```

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Database Permissions**
   ```bash
   chmod 644 agent_feedback.db
   ```

3. **Port Already in Use**
   ```bash
   enterprise-ai --dashboard --port 8001
   ```

### Debug Mode

```bash
ENTERPRISE_AI_LOG_LEVEL=DEBUG enterprise-ai --interactive
```

### Model Training Issues

- Ensure sufficient feedback data (minimum 10 examples)
- Check for balanced agent usage in training data
- Verify database connectivity for feedback storage

## Contributing

1. **Development Setup**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

2. **Testing**
   ```bash
   pytest tests/
   ```

3. **Code Quality**
   ```bash
   black enterprise_ai/
   flake8 enterprise_ai/
   mypy enterprise_ai/
   ```

## License

see LICENSE file for details.

## Support

- **Documentation**: https://enterprise-ai-routing.readthedocs.io/
- **Issues**: https://github.com/company/enterprise-ai-routing/issues
- **Discussions**: https://github.com/company/enterprise-ai-routing/discussions
