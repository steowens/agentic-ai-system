# Legacy Files

This folder contains the original flat file structure before the code was refactored into the `enterprise_ai` package.

These files are kept for reference but are no longer used. The functionality has been reorganized into:

- `enterprise_ai/core/` - Main orchestration (agents.py, routing.py, etc.)
- `enterprise_ai/integrations/` - Tools and MCP framework
- `enterprise_ai/monitoring/` - Cost tracking and metrics (production_cost_tracker.py, etc.)
- `enterprise_ai/learning/` - Feedback and ML routing (feedback_system.py, routing_learning_engine.py)
- `enterprise_ai/security/` - Data security framework
- `enterprise_ai/dashboard/` - Web interface components

## Current System Entry Point

Use `python main.py` in the parent directory to start the web interface.

## Files in this folder:
- Original implementation files before package refactoring
- Demo and test files
- Standalone scripts that are now integrated into the package