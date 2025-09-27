"""
Advanced Feedback Collection and Routing Improvement System
Collects user feedback to improve routing decisions and system performance over time.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path


class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    DETAILED = "detailed"
    ROUTING_CORRECTION = "routing_correction"
    PERFORMANCE_ISSUE = "performance_issue"
    COST_CONCERN = "cost_concern"


class RoutingOutcome(Enum):
    SUCCESS = "success"
    WRONG_AGENT = "wrong_agent" 
    INCOMPLETE = "incomplete"
    ERROR = "error"
    TOO_EXPENSIVE = "too_expensive"
    TOO_SLOW = "too_slow"


@dataclass
class UserFeedback:
    """Structured user feedback data"""
    feedback_id: str
    session_id: str
    question_id: str
    user_id: Optional[str]
    timestamp: datetime
    feedback_type: FeedbackType
    rating: Optional[int]  # 1-5 scale
    
    # Routing-specific feedback
    actual_agent_used: str
    suggested_agent: Optional[str]
    routing_confidence: float
    routing_outcome: RoutingOutcome
    
    # Detailed feedback
    user_comment: Optional[str]
    what_worked: Optional[str]
    what_failed: Optional[str]
    improvement_suggestion: Optional[str]
    
    # Context
    original_question: str
    agent_response: str
    processing_time: float
    cost: float
    
    # System metrics
    response_quality_score: Optional[float]
    routing_accuracy_score: Optional[float]


@dataclass
class RoutingPattern:
    """Learned routing patterns from feedback"""
    pattern_id: str
    question_keywords: List[str]
    question_type: str
    preferred_agent: str
    confidence_threshold: float
    success_rate: float
    sample_count: int
    last_updated: datetime
    
    # Performance metrics
    avg_processing_time: float
    avg_cost: float
    avg_user_rating: float


class FeedbackDatabase:
    """SQLite database for storing and analyzing feedback"""
    
    def __init__(self, db_path: str = "agent_feedback.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT,
                question_id TEXT,
                user_id TEXT,
                timestamp TEXT,
                feedback_type TEXT,
                rating INTEGER,
                actual_agent_used TEXT,
                suggested_agent TEXT,
                routing_confidence REAL,
                routing_outcome TEXT,
                user_comment TEXT,
                what_worked TEXT,
                what_failed TEXT,
                improvement_suggestion TEXT,
                original_question TEXT,
                agent_response TEXT,
                processing_time REAL,
                cost REAL,
                response_quality_score REAL,
                routing_accuracy_score REAL
            )
        """)
        
        # Routing patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_patterns (
                pattern_id TEXT PRIMARY KEY,
                question_keywords TEXT,
                question_type TEXT,
                preferred_agent TEXT,
                confidence_threshold REAL,
                success_rate REAL,
                sample_count INTEGER,
                last_updated TEXT,
                avg_processing_time REAL,
                avg_cost REAL,
                avg_user_rating REAL
            )
        """)
        
        # Agent performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                agent_name TEXT,
                date TEXT,
                total_requests INTEGER,
                success_rate REAL,
                avg_rating REAL,
                avg_cost REAL,
                avg_processing_time REAL,
                user_satisfaction REAL,
                PRIMARY KEY (agent_name, date)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: UserFeedback):
        """Store user feedback in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feedback VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            feedback.feedback_id,
            feedback.session_id,
            feedback.question_id,
            feedback.user_id,
            feedback.timestamp.isoformat(),
            feedback.feedback_type.value,
            feedback.rating,
            feedback.actual_agent_used,
            feedback.suggested_agent,
            feedback.routing_confidence,
            feedback.routing_outcome.value,
            feedback.user_comment,
            feedback.what_worked,
            feedback.what_failed,
            feedback.improvement_suggestion,
            feedback.original_question,
            feedback.agent_response,
            feedback.processing_time,
            feedback.cost,
            feedback.response_quality_score,
            feedback.routing_accuracy_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_agent_feedback_summary(self, agent_name: str, days: int = 30) -> Dict:
        """Get feedback summary for specific agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                AVG(routing_confidence) as avg_confidence,
                AVG(processing_time) as avg_time,
                AVG(cost) as avg_cost,
                COUNT(CASE WHEN routing_outcome = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                COUNT(CASE WHEN feedback_type = 'thumbs_up' THEN 1 END) as thumbs_up,
                COUNT(CASE WHEN feedback_type = 'thumbs_down' THEN 1 END) as thumbs_down
            FROM feedback 
            WHERE actual_agent_used = ? AND timestamp >= ?
        """, (agent_name, since_date))
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0] == 0:  # No feedback
            return {"agent": agent_name, "no_feedback": True}
        
        return {
            "agent": agent_name,
            "total_feedback": result[0],
            "avg_rating": round(result[1] or 0, 2),
            "avg_confidence": round(result[2] or 0, 3),
            "avg_processing_time": round(result[3] or 0, 2),
            "avg_cost": round(result[4] or 0, 4),
            "success_rate": round(result[5] or 0, 1),
            "thumbs_up": result[6],
            "thumbs_down": result[7],
            "satisfaction_ratio": round((result[6] / (result[6] + result[7])) * 100, 1) if (result[6] + result[7]) > 0 else 0
        }
    
    def get_routing_improvement_suggestions(self) -> List[Dict]:
        """Analyze feedback to suggest routing improvements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find questions with consistent wrong routing
        cursor.execute("""
            SELECT 
                original_question,
                actual_agent_used,
                suggested_agent,
                COUNT(*) as occurrences,
                AVG(rating) as avg_rating,
                AVG(routing_confidence) as avg_confidence
            FROM feedback 
            WHERE routing_outcome = 'wrong_agent' 
                AND suggested_agent IS NOT NULL
                AND timestamp >= datetime('now', '-30 days')
            GROUP BY original_question, actual_agent_used, suggested_agent
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC, AVG(rating) ASC
        """)
        
        suggestions = []
        for row in cursor.fetchall():
            suggestions.append({
                "question_pattern": row[0],
                "current_agent": row[1],
                "suggested_agent": row[2],
                "frequency": row[3],
                "avg_rating": round(row[4], 2),
                "avg_confidence": round(row[5], 3),
                "recommendation": f"Route '{row[0][:50]}...' to {row[2]} instead of {row[1]}"
            })
        
        conn.close()
        return suggestions


class FeedbackCollector:
    """Main feedback collection and analysis system"""
    
    def __init__(self, db_path: str = "agent_feedback.db"):
        self.db = FeedbackDatabase(db_path)
        self.session_feedback = {}  # Temporary storage for session
    
    def collect_quick_feedback(
        self, 
        question_id: str,
        session_id: str,
        is_positive: bool,
        routing_info: Dict,
        response_info: Dict
    ) -> str:
        """Collect simple thumbs up/down feedback"""
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            question_id=question_id,
            user_id=None,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.THUMBS_UP if is_positive else FeedbackType.THUMBS_DOWN,
            rating=5 if is_positive else 2,
            actual_agent_used=routing_info.get("agent_used", "unknown"),
            suggested_agent=None,
            routing_confidence=routing_info.get("confidence", 0.0),
            routing_outcome=RoutingOutcome.SUCCESS if is_positive else RoutingOutcome.WRONG_AGENT,
            user_comment=None,
            what_worked=None,
            what_failed=None,
            improvement_suggestion=None,
            original_question=response_info.get("question", ""),
            agent_response=response_info.get("response", ""),
            processing_time=response_info.get("processing_time", 0.0),
            cost=response_info.get("cost", 0.0),
            response_quality_score=None,
            routing_accuracy_score=None
        )
        
        self.db.store_feedback(feedback)
        return feedback.feedback_id
    
    def collect_detailed_feedback(
        self,
        question_id: str,
        session_id: str,
        rating: int,
        routing_correction: Optional[str],
        user_comment: str,
        what_worked: str,
        what_failed: str,
        improvement_suggestion: str,
        routing_info: Dict,
        response_info: Dict
    ) -> str:
        """Collect detailed user feedback with routing correction"""
        
        # Determine routing outcome based on feedback
        if rating >= 4:
            outcome = RoutingOutcome.SUCCESS
        elif routing_correction:
            outcome = RoutingOutcome.WRONG_AGENT
        elif "slow" in user_comment.lower() or "time" in what_failed.lower():
            outcome = RoutingOutcome.TOO_SLOW
        elif "expensive" in user_comment.lower() or "cost" in what_failed.lower():
            outcome = RoutingOutcome.TOO_EXPENSIVE
        elif "incomplete" in user_comment.lower() or "partial" in what_failed.lower():
            outcome = RoutingOutcome.INCOMPLETE
        else:
            outcome = RoutingOutcome.ERROR
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            question_id=question_id,
            user_id=None,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.DETAILED,
            rating=rating,
            actual_agent_used=routing_info.get("agent_used", "unknown"),
            suggested_agent=routing_correction,
            routing_confidence=routing_info.get("confidence", 0.0),
            routing_outcome=outcome,
            user_comment=user_comment,
            what_worked=what_worked,
            what_failed=what_failed,
            improvement_suggestion=improvement_suggestion,
            original_question=response_info.get("question", ""),
            agent_response=response_info.get("response", ""),
            processing_time=response_info.get("processing_time", 0.0),
            cost=response_info.get("cost", 0.0),
            response_quality_score=None,
            routing_accuracy_score=None
        )
        
        self.db.store_feedback(feedback)
        return feedback.feedback_id
    
    def analyze_routing_performance(self) -> Dict:
        """Comprehensive routing performance analysis"""
        
        analysis = {
            "overall_metrics": {},
            "agent_performance": {},
            "routing_suggestions": [],
            "trends": {},
            "problem_areas": []
        }
        
        # Get performance for each agent
        agents = ["math_agent", "system_agent", "general_agent"]  # Add your agents
        
        for agent in agents:
            performance = self.db.get_agent_feedback_summary(agent)
            analysis["agent_performance"][agent] = performance
        
        # Get routing improvement suggestions
        analysis["routing_suggestions"] = self.db.get_routing_improvement_suggestions()
        
        # Calculate overall metrics
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                COUNT(CASE WHEN routing_outcome = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                AVG(cost) as avg_cost,
                AVG(processing_time) as avg_time
            FROM feedback 
            WHERE timestamp >= datetime('now', '-30 days')
        """)
        
        overall = cursor.fetchone()
        if overall and overall[0] > 0:
            analysis["overall_metrics"] = {
                "total_feedback_30_days": overall[0],
                "average_rating": round(overall[1], 2),
                "routing_success_rate": round(overall[2], 1),
                "average_cost": round(overall[3], 4),
                "average_processing_time": round(overall[4], 2)
            }
        
        conn.close()
        return analysis
    
    def get_feedback_trends(self, days: int = 30) -> Dict:
        """Get feedback trends over time"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Daily feedback trends
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                COUNT(CASE WHEN routing_outcome = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM feedback 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """.format(days))
        
        trends = []
        for row in cursor.fetchall():
            trends.append({
                "date": row[0],
                "total_feedback": row[1],
                "avg_rating": round(row[2], 2),
                "success_rate": round(row[3], 1)
            })
        
        conn.close()
        return {"daily_trends": trends}


# Example usage and testing
if __name__ == "__main__":
    
    print("🎯 FEEDBACK COLLECTION SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize feedback system
    feedback_collector = FeedbackCollector("demo_feedback.db")
    
    # Simulate some feedback data
    sample_feedback = [
        {
            "question_id": "q1",
            "session_id": "s1", 
            "is_positive": True,
            "routing_info": {"agent_used": "math_agent", "confidence": 0.95},
            "response_info": {"question": "What is the integral of x^2?", "response": "The integral is x^3/3 + C", "processing_time": 1.2, "cost": 0.001}
        },
        {
            "question_id": "q2",
            "session_id": "s1",
            "is_positive": False, 
            "routing_info": {"agent_used": "general_agent", "confidence": 0.6},
            "response_info": {"question": "Calculate stress on steel beam", "response": "I can help with general questions...", "processing_time": 2.1, "cost": 0.002}
        }
    ]
    
    # Collect quick feedback
    for sample in sample_feedback:
        feedback_id = feedback_collector.collect_quick_feedback(
            sample["question_id"],
            sample["session_id"], 
            sample["is_positive"],
            sample["routing_info"],
            sample["response_info"]
        )
        print(f"✅ Collected feedback: {feedback_id}")
    
    # Collect detailed feedback
    detailed_id = feedback_collector.collect_detailed_feedback(
        question_id="q3",
        session_id="s1",
        rating=2,
        routing_correction="math_agent",
        user_comment="The response was generic and didn't help with my engineering calculation",
        what_worked="The system responded quickly",
        what_failed="Wrong agent was selected - needed specialized math help",
        improvement_suggestion="Better detection of engineering/math questions",
        routing_info={"agent_used": "general_agent", "confidence": 0.7},
        response_info={"question": "Calculate moment of inertia", "response": "I can help...", "processing_time": 1.5, "cost": 0.0015}
    )
    print(f"📝 Detailed feedback collected: {detailed_id}")
    
    # Analyze performance
    analysis = feedback_collector.analyze_routing_performance()
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"Overall metrics: {analysis['overall_metrics']}")
    print(f"Routing suggestions: {len(analysis['routing_suggestions'])} improvements identified")
    
    for suggestion in analysis["routing_suggestions"]:
        print(f"💡 {suggestion['recommendation']}")
    
    print("\n🎉 Feedback system ready for integration!")