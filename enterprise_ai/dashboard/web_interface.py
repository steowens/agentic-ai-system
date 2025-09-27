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
from typing import Optional
import uvicorn
import os
from pathlib import Path

from ..core.orchestrator import SystemOrchestrator


class QueryRequest(BaseModel):
    question: str
    markdown: bool = True


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int
    feedback_type: str  # 'quick' or 'detailed'
    is_positive: Optional[bool] = None
    suggested_agent: Optional[str] = None
    comment: Optional[str] = None


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

# Store request context for feedback
request_context = {}


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
        
        # Store request context for feedback
        if result.get('request_id'):
            request_context[result['request_id']] = {
                'question': query.question,
                'agent_used': result.get('agent_used', 'unknown'),
                'response': result.get('response', ''),
                'confidence': result.get('confidence', 0.8),
                'processing_time': result.get('processing_time', 1.0),
                'cost': result.get('cost', 0.001),
                'timestamp': datetime.now().isoformat()
            }

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


@app.post("/api/feedback/submit")
async def submit_feedback(feedback: FeedbackRequest) -> Dict:
    """Submit user feedback for learning system"""
    try:
        # Import the learning system components
        from ..learning.feedback import feedback_collector
        from ..learning.routing_engine import routing_engine

        # Get the original question and agent info from stored context
        context = request_context.get(feedback.request_id)
        if not context:
            raise ValueError(f"No context found for request_id: {feedback.request_id}")

        routing_info = {
            "agent_used": context['agent_used'],
            "confidence": context['confidence']
        }

        response_info = {
            "question": context['question'],
            "response": context['response'],
            "processing_time": context['processing_time'],
            "cost": context['cost']
        }

        if feedback.feedback_type == "quick":
            # Simple thumbs up/down feedback
            feedback_id = feedback_collector.collect_simple_feedback(
                question_id=feedback.request_id,
                session_id="default_session",
                is_positive=feedback.is_positive,
                routing_info=routing_info,
                response_info=response_info
            )
        else:
            # Detailed feedback with rating and suggestions
            feedback_id = feedback_collector.collect_detailed_feedback(
                question_id=feedback.request_id,
                session_id="default_session",
                rating=feedback.rating,
                routing_correction=feedback.suggested_agent,
                user_comment=feedback.comment or "",
                what_worked="",
                what_failed="",
                improvement_suggestion=feedback.comment or "",
                routing_info=routing_info,
                response_info=response_info
            )

        # Update the learning engine with feedback
        if feedback.suggested_agent:
            routing_engine.update_from_feedback({
                'routing_outcome': 'wrong_agent',
                'suggested_agent': feedback.suggested_agent,
                'original_question': context['question']
            })

        # Log feedback for learning system
        print(f"📋 FEEDBACK: Q='{context['question'][:50]}...' Agent={context['agent_used']} → Rating={feedback.rating} SuggestedAgent={feedback.suggested_agent}")

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ FEEDBACK ERROR: {str(e)}")
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