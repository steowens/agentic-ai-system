# Developer Guide

## Quick Start
```bash
cd agentic-ai-system
python main.py
# Visit http://localhost:8000
```

## Project Structure

### Core Architecture
```
enterprise_ai/
├── core/                    # Main system components
│   ├── orchestrator_simple.py   # SystemOrchestrator - main routing logic
│   ├── agents.py               # Agent configurations (math, system, code)
│   ├── routing.py              # Agent routing service
│   └── tools.py                # Tool factory and implementations
├── dashboard/               # Web interface
│   └── web_interface.py        # FastAPI server (HTTP only, no WebSockets)
├── integrations/            # External service integrations
├── monitoring/              # Metrics and cost tracking
└── security/                # Security implementations
```

### Entry Points
- **`main.py`** - Simplified 55-line entry point that launches the web server
- **Web Interface** - http://localhost:8000 with real-time metrics via HTTP polling

### Utilities
```
utils/
├── count_tokens.py          # Token counting utility for agent prompts
└── [future utilities]       # Performance profilers, analyzers, etc.
```

## Key Technical Decisions

### WebSocket Removal (CRITICAL)
** DO NOT re-add WebSockets** - They caused connection spam issues ("an overwehlming nuber of connections opening")
- **Solution**: HTTP polling every 30 seconds in frontend
- **Files affected**: `web_interface.py`, `templates/index.html`
- **Why**: Simpler, more reliable, eliminates connection leaks

### Agent Method Signatures (IMPORTANT)
**Correct usage**: `agent.run(task=question)`
- **Wrong**: `agent.run(question)` 
- **Error**: "BaseChatAgent.run() takes 1 positional argument but 2 were given"

### Response Parsing (IMPORTANT)
**Correct extraction**: `result.messages[-1].content`
- AutoGen returns message objects, need to extract `.content`
- **Wrong**: Using raw result object in JSON responses

## Agent Configuration

### Token Counts (as of analysis)
- **Math Agent**: 558 tokens (detailed guidance)
- **Minimal version**: 34 tokens (93.9% reduction possible)
- **Cost impact**: $0.000084 per message (negligible)
- **Context usage**: 0.44% of 128k window

### Math Agent Philosophy
- **Detailed guidance works better** than minimal prompts
- 558 tokens provides clear decision-making framework
- No "explosion" risk - cognitive load is manageable
- Includes geometry relationships, LaTeX formatting, tool usage rules

## Cost Tracking

### Current Rates (GPT-4o-mini)
- Input: $0.00015 per 1k tokens
- Output: $0.0006 per 1k tokens
- Tracking: Input/output differentiated in metrics

### Database
- SQLite: `autogen04202.db`
- Tables: conversations, cost_tracking
- Web metrics: Real-time cost display

## Development Workflow

### Testing Agent Changes
1. Edit agent in `enterprise_ai/core/agents.py`
2. Restart `main.py`
3. Test via web interface
4. Check metrics for cost/performance impact

### Adding New Utilities
1. Create in `utils/` directory
2. Follow naming convention: `[purpose]_[tool].py`
3. Include token analysis if relevant
4. Document in this guide

### Debugging Connection Issues
1. **First check**: Are WebSockets involved? (Remove them!)
2. Check terminal for connection spam
3. Verify HTTP polling intervals in frontend
4. Use browser dev tools for network analysis

## Configuration

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_key_here
```

### Dependencies
```bash
pip install -r requirements.txt
# Key: autogen, openai, fastapi, uvicorn, tiktoken
```

## Known Issues & Solutions

### Issue: WebSocket Connection Spam
**Status**: SOLVED ✅
**Solution**: Completely removed WebSockets, use HTTP polling

### Issue: Method Signature Errors
**Status**: SOLVED ✅  
**Solution**: Use `agent.run(task=question)` format

### Issue: Response Parsing
**Status**: SOLVED ✅
**Solution**: Extract with `result.messages[-1].content`

### Issue: Undefined JavaScript Variables
**Status**: SOLVED ✅
**Solution**: Added missing `reasoning` field to responses

## Performance Notes

### System Metrics
- Real-time agent routing decisions
- Token consumption tracking
- Response time monitoring
- Cost analysis per conversation

### Optimization Tips
1. Monitor token usage with `utils/count_tokens.py`
2. Test agent prompt changes incrementally
3. Use detailed system messages (don't over-optimize)
4. HTTP polling is sufficient for UI updates

## Future Enhancements

### Planned Utilities
- Agent performance profiler
- Cost optimization analyzer  
- Response quality metrics
- Token usage trends

### Architecture Improvements
- Agent specialization routing
- Dynamic prompt optimization
- Multi-model cost comparison
- Advanced security features

---

## Quick Reference Commands

```bash
# Run system
python main.py

# Count tokens in math agent
python utils/count_tokens.py

# Check project structure
tree /F enterprise_ai

# Monitor costs
# Visit http://localhost:8000 metrics section
```

## Emergency Debugging

If the system breaks:
1. Check for WebSocket code (remove it)
2. Verify method signatures match `agent.run(task=...)`
3. Ensure response extraction uses `.content`
4. Check agent configurations in `enterprise_ai/core/agents.py`
5. Restart from scratch: `python main.py`

## AI Assistant Guidelines

**CRITICAL: Prevent over-engineering and cost inflation**

### Simplicity Rules
- **Text change = text change**: Don't refactor entire systems for string updates
- **2-line fix stays 2 lines**: Don't turn simple edits into complex refactors
- **Challenge complexity**: Ask "What's the minimal change needed?"

### Red Flags to Avoid
- ❌ Rewriting utilities when you only need to update hardcoded strings
- ❌ Adding imports and classes for simple text additions  
- ❌ Dynamic solutions when static solutions work fine
- ❌ "Let me make this more elegant" when it already works

### Cost-Conscious Approach
- **Be collaborative**: If you are even remotely in doubt 
explain your plan and then get approval.
- **Budget tool calls**: Simple tasks should use 1-3 tool calls maximum
- **Question complexity**: If solution needs >5 steps, something's wrong
- **Immediate pushback**: "Stop. You're over-engineering. What's the simplest fix?"

### Examples of RIGHT vs WRONG

**WRONG** (Token counter update):
```
- Rewrite entire script with dynamic imports
- Add class lookups and sys.path manipulation
- Debug import issues for 10 minutes
- Create maintenance overhead
```

**RIGHT** (Token counter update):  
```
- Copy new system message text
- Paste into hardcoded string
- Done in 30 seconds
```

### Enforcement Phrases
Use these to stop over-engineering:
- "What's the minimal change?"
- "Don't refactor, just fix"
- "Budget: max 2 tool calls"
- "Simple text edit only"

**Remember**: Every unnecessary tool call costs money and time. Simple problems deserve simple solutions.

---
*Last updated: September 26, 2025*  
*System status: Fully functional with HTTP-based architecture*