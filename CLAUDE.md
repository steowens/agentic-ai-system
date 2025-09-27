# Claude Code Custom Instructions for This Project

## Critical Collaboration Rules

### 1. BE COLLABORATIVE, NOT PRESUMPTUOUS
- **ALWAYS explain your plan before implementing** - Don't just start coding
- If you're even remotely in doubt, get approval first
- Ask "What's the minimal change needed?" before complex solutions
- The user often has better insights into their own codebase

### 2. TEST BEFORE YOU BREAK
- **NEVER remove or delete existing code until you've verified the replacement works**
- Always test imports, functionality, and end-to-end behavior
- Run the actual system (`python main.py`) to verify changes work
- Don't assume your refactoring preserved functionality

### 3. UNDERSTAND EXISTING ARCHITECTURE FIRST
- **Read DEVELOPER_GUIDE.md before making any structural changes**
- Check for duplicate classes/functions before creating new ones
- Understand which files are actually being used vs legacy
- Look for existing patterns and conventions in the codebase

## Project-Specific Context

### Architecture Overview
- **Main entry**: `main.py` - simplified 55-line launcher
- **Core orchestrator**: `enterprise_ai/core/orchestrator_simple.py` (currently active)
- **Full orchestrator**: `enterprise_ai/core/orchestrator.py` (enterprise features, some broken)
- **Web interface**: HTTP polling only, **NO WEBSOCKETS** (they caused connection spam)

### Key Technical Constraints
- **Agent method signatures**: Use `agent.run(task=question)` format
- **Response parsing**: Extract with `result.messages[-1].content`
- **Tool factory**: Located in `enterprise_ai/integrations/tool_factory.py`
- **Environment variables**: Model selection via `ENTERPRISE_AI_MODEL`

### Anti-Patterns to Avoid
- ❌ **Over-engineering simple text changes** - Don't refactor entire systems for string updates
- ❌ **Creating duplicate classes** without checking existing architecture
- ❌ **Deleting old code** before testing new code works
- ❌ **Adding WebSockets** - they're explicitly banned due to connection issues
- ❌ **Assuming imports work** - always test after refactoring

### User's Preferred Workflow
1. **Minimal viable changes** - Simple problems get simple solutions
2. **Collaborative planning** - Explain approach, get approval for complex changes
3. **Test-driven refactoring** - Prove new code works before removing old
4. **Architecture-aware** - Understand existing patterns before changing them

## Quality Standards

### Code Changes
- Follow SOLID principles when refactoring
- Preserve existing naming conventions and patterns
- Test imports and functionality after structural changes
- Keep changes focused - don't expand scope unnecessarily

### Communication Style
- Be direct about what you're doing and why
- Ask for guidance when multiple approaches are possible
- Admit mistakes quickly and fix them properly
- Focus on the user's actual request, not what you think they need

## Project History Context
- Started with monolithic `tools.py`, refactored to modular architecture
- Had issues with duplicate `ToolFactory` classes during refactoring
- WebSocket support was removed due to connection spam problems
- Math calculator tool has detailed prompts that work better than minimal ones
- System supports context compression to prevent token limit errors

## Emergency Debugging Checklist
When things break:
1. Check for missing imports after refactoring
2. Verify no duplicate classes exist
3. Test with `python main.py` to see actual errors
4. Check DEVELOPER_GUIDE.md for known solutions
5. Look for WebSocket code that shouldn't be there

---

**Remember**: The user values collaboration over speed, testing over assumptions, and simplicity over elegance. When in doubt, ask first.