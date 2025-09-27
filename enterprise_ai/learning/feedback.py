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
        
        # Feedback table - simplified schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT,
                question_id TEXT,
                timestamp DATETIME,
                feedback_type TEXT,
                rating INTEGER,
                actual_agent TEXT,
                suggested_agent TEXT,
                routing_confidence REAL,
                routing_outcome TEXT,
                user_comment TEXT,
                original_question TEXT,
                agent_response TEXT,
                processing_time REAL,
                cost REAL
            )
        """)
        
        # Routing patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_patterns (
                pattern_id TEXT PRIMARY KEY,
                question_type TEXT,
                preferred_agent TEXT,
                confidence_threshold REAL,
                success_rate REAL,
                sample_count INTEGER,
                last_updated DATETIME,
                avg_processing_time REAL,
                avg_cost REAL,
                avg_user_rating REAL,
                keywords TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: UserFeedback):
        """Store user feedback in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feedback 
            (feedback_id, session_id, question_id, timestamp, feedback_type, rating,
             actual_agent, suggested_agent, routing_confidence, routing_outcome,
             user_comment, original_question, agent_response, processing_time, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.feedback_id, feedback.session_id, feedback.question_id,
            feedback.timestamp, feedback.feedback_type.value, feedback.rating,
            feedback.actual_agent_used, feedback.suggested_agent,
            feedback.routing_confidence, feedback.routing_outcome.value,
            feedback.user_comment, feedback.original_question,
            feedback.agent_response, feedback.processing_time, feedback.cost
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                AVG(processing_time) as avg_time,
                AVG(cost) as avg_cost,
                COUNT(CASE WHEN routing_outcome = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM feedback 
            WHERE timestamp > ?
        """, (since_date,))
        
        stats = cursor.fetchone()
        
        # Agent performance
        cursor.execute("""
            SELECT 
                actual_agent,
                COUNT(*) as usage_count,
                AVG(rating) as avg_rating,
                AVG(processing_time) as avg_time,
                AVG(cost) as avg_cost
            FROM feedback 
            WHERE timestamp > ?
            GROUP BY actual_agent
        """, (since_date,))
        
        agent_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_feedback": stats[0] or 0,
            "avg_rating": stats[1] or 0,
            "avg_processing_time": stats[2] or 0,
            "avg_cost": stats[3] or 0,
            "success_rate": stats[4] or 0,
            "agent_performance": [
                {
                    "agent": row[0],
                    "usage_count": row[1],
                    "avg_rating": row[2] or 0,
                    "avg_time": row[3] or 0,
                    "avg_cost": row[4] or 0
                }
                for row in agent_stats
            ]
        }


class FeedbackCollector:
    """Main interface for collecting and processing feedback"""
    
    def __init__(self, db_path: str = "agent_feedback.db"):
        self.db = FeedbackDatabase(db_path)
        self.session_id = str(uuid.uuid4())
    
    def collect_simple_feedback(
        self,
        question: str,
        response: str,
        agent_used: str,
        rating: int,
        processing_time: float,
        cost: float,
        routing_confidence: float = 0.5
    ) -> str:
        """Collect simple thumbs up/down feedback"""
        feedback_type = FeedbackType.THUMBS_UP if rating >= 3 else FeedbackType.THUMBS_DOWN
        outcome = RoutingOutcome.SUCCESS if rating >= 3 else RoutingOutcome.INCOMPLETE
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            session_id=self.session_id,
            question_id=str(uuid.uuid4()),
            user_id=None,
            timestamp=datetime.now(),
            feedback_type=feedback_type,
            rating=rating,
            actual_agent_used=agent_used,
            suggested_agent=None,
            routing_confidence=routing_confidence,
            routing_outcome=outcome,
            user_comment=None,
            what_worked=None,
            what_failed=None,
            improvement_suggestion=None,
            original_question=question,
            agent_response=response,
            processing_time=processing_time,
            cost=cost,
            response_quality_score=rating / 5.0,
            routing_accuracy_score=None
        )
        
        self.db.store_feedback(feedback)
        return feedback.feedback_id
    
    def collect_detailed_feedback(
        self,
        question: str,
        response: str,
        agent_used: str,
        rating: int,
        processing_time: float,
        cost: float,
        comment: Optional[str] = None,
        suggested_agent: Optional[str] = None,
        routing_confidence: float = 0.5
    ) -> str:
        """Collect detailed user feedback with comments and suggestions"""
        outcome = self._determine_outcome(rating, suggested_agent, agent_used)
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            session_id=self.session_id,
            question_id=str(uuid.uuid4()),
            user_id=None,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.DETAILED,
            rating=rating,
            actual_agent_used=agent_used,
            suggested_agent=suggested_agent,
            routing_confidence=routing_confidence,
            routing_outcome=outcome,
            user_comment=comment,
            what_worked=None,
            what_failed=None,
            improvement_suggestion=None,
            original_question=question,
            agent_response=response,
            processing_time=processing_time,
            cost=cost,
            response_quality_score=rating / 5.0,
            routing_accuracy_score=None
        )
        
        self.db.store_feedback(feedback)
        return feedback.feedback_id
    
    def _determine_outcome(self, rating: int, suggested_agent: Optional[str], actual_agent: str) -> RoutingOutcome:
        """Determine routing outcome based on rating and suggestions"""
        if rating >= 4:
            return RoutingOutcome.SUCCESS
        elif suggested_agent and suggested_agent != actual_agent:
            return RoutingOutcome.WRONG_AGENT
        elif rating <= 2:
            return RoutingOutcome.INCOMPLETE
        else:
            return RoutingOutcome.SUCCESS
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get recent performance summary"""
        return self.db.get_feedback_stats(days)


# Global feedback collector instance
feedback_collector = FeedbackCollector()


def log_interaction(
    question: str,
    response: str,
    agent_used: str,
    processing_time: float,
    cost: float,
    rating: Optional[int] = None,
    routing_confidence: float = 0.5
):
    """Convenience function to log an interaction"""
    if rating is not None:
        return feedback_collector.collect_simple_feedback(
            question, response, agent_used, rating,
            processing_time, cost, routing_confidence
        )
    return None


def get_feedback_stats(days: int = 7) -> Dict[str, Any]:
    """Convenience function to get feedback statistics"""
    return feedback_collector.get_performance_summary(days)