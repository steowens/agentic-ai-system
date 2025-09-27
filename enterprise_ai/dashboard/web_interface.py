"""
FastAPI web interface for the intelligent agent system.
Provides markdown input/output with real-time metrics and logging.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import json
import asyncio
from datetime import datetime
from typing import Dict, List
import markdown
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path

from ..core.orchestrator_simple import SystemOrchestrator


class QueryRequest(BaseModel):
    question: str
    markdown: bool = True


# No WebSocket needed - using simple HTTP polling


# Initialize FastAPI app
app = FastAPI(title="Agentic AI System", version="1.0.0")

# Get the project root directory (where main.py is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Mount static files and templates with absolute paths
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize system
system = SystemOrchestrator()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = PROJECT_ROOT / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    else:
        # Return a simple default response if no favicon exists
        return FileResponse(str(PROJECT_ROOT / "static" / "favicon.ico"), status_code=204)


@app.post("/api/query")
async def process_query(query: QueryRequest) -> Dict:
    """Process a query and return structured response"""
    try:
        result = await system.process_question(query.question)
        
        # Convert response to HTML if markdown requested
        if query.markdown and isinstance(result.get('response'), str):
            # Process the response to ensure proper math formatting
            response_text = result['response']
            
            # Convert LaTeX parentheses to MathJax-friendly format
            import re
            
            # Replace \( ... \) with $ ... $ for inline math
            response_text = re.sub(r'\\\((.*?)\\\)', r'$\1$', response_text)
            
            # Replace \[ ... \] with $$ ... $$ for display math
            response_text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', response_text)
            
            # Ensure single parentheses math expressions are properly formatted
            response_text = re.sub(r'\(([^)]*[x\^0-9\+\-\*/=][^)]*)\)', r'$(\1)$', response_text)
            
            result['response_html'] = markdown.markdown(
                response_text,
                extensions=['codehilite', 'fenced_code', 'tables']
            )
        
        # No WebSocket broadcasting needed
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# No metrics endpoint needed - client tracks everything via localStorage


if __name__ == "__main__":
    uvicorn.run(
        "web_interface:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )