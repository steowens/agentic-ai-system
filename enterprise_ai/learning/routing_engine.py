"""
Machine Learning-Based Routing Improvement Engine
Learns from user feedback to improve routing decisions over time.
"""
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sqlite3
import re
from pathlib import Path


@dataclass
class RoutingRule:
    """Dynamic routing rule learned from feedback"""
    rule_id: str
    keywords: List[str]
    patterns: List[str]
    preferred_agent: str
    confidence_threshold: float
    success_rate: float
    priority: int
    created_date: datetime
    last_updated: datetime
    sample_count: int
    
    # Performance metrics
    avg_user_rating: float
    avg_processing_time: float
    avg_cost: float


@dataclass
class RoutingPrediction:
    """Routing prediction with confidence and reasoning"""
    predicted_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[Tuple[str, float]]
    rule_matches: List[str]
    ml_prediction: Optional[str]
    ml_confidence: Optional[float]


class QuestionFeatureExtractor:
    """Extract features from questions for ML routing"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.feature_names = []
        
        # Engineering and domain-specific keywords
        self.domain_keywords = {
            'math': ['calculate', 'integral', 'derivative', 'equation', 'formula', 'solve', 'compute'],
            'engineering': ['stress', 'strain', 'load', 'beam', 'truss', 'material', 'design', 'analysis'],
            'database': ['query', 'select', 'table', 'database', 'sql', 'data', 'record', 'join'],
            'system': ['file', 'directory', 'system', 'process', 'network', 'server', 'config'],
            'esri': ['map', 'gis', 'spatial', 'coordinate', 'layer', 'feature', 'geometry', 'projection']
        }
    
    def extract_features(self, question: str) -> Dict[str, float]:
        """Extract comprehensive features from a question"""
        features = {}
        
        # Basic text features
        features['question_length'] = len(question)
        features['word_count'] = len(question.split())
        features['has_numbers'] = float(bool(re.search(r'\d+', question)))
        features['has_equation'] = float(bool(re.search(r'[+=\-*/^]', question)))
        features['question_mark_count'] = question.count('?')
        
        # Domain-specific keyword features
        question_lower = question.lower()
        for domain, keywords in self.domain_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in question_lower)
            features[f'{domain}_keywords'] = keyword_count
            features[f'{domain}_score'] = keyword_count / len(keywords)
        
        # Complexity indicators
        features['technical_terms'] = self._count_technical_terms(question)
        features['complexity_score'] = self._calculate_complexity_score(question)
        
        return features
    
    def _count_technical_terms(self, question: str) -> float:
        """Count technical terms in question"""
        technical_terms = [
            'algorithm', 'optimization', 'parameter', 'variable', 'function',
            'matrix', 'vector', 'coefficient', 'analysis', 'simulation'
        ]
        return sum(1 for term in technical_terms if term in question.lower())
    
    def _calculate_complexity_score(self, question: str) -> float:
        """Calculate overall complexity score"""
        complexity_indicators = [
            len(question) > 100,  # Long questions
            question.count('(') > 0,  # Parentheses
            question.count(',') > 2,  # Multiple clauses
            bool(re.search(r'\b(how|why|explain|analyze|compare)\b', question.lower())),  # Complex queries
        ]
        return sum(complexity_indicators) / len(complexity_indicators)


class MLRoutingEngine:
    """Machine Learning engine for routing prediction"""
    
    def __init__(self, db_path: str = "agent_feedback.db"):
        self.db_path = db_path
        self.feature_extractor = QuestionFeatureExtractor()
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.model_accuracy = 0.0
        self.is_trained = False
        self.model_path = Path("routing_model.joblib")
        
        # Load existing model if available
        self.load_model()
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get successful routing examples
        cursor.execute("""
            SELECT original_question, actual_agent, rating
            FROM feedback
            WHERE rating IS NOT NULL AND rating >= 3
            ORDER BY timestamp DESC
            LIMIT 1000
        """)
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) < 10:
            raise ValueError("Insufficient training data. Need at least 10 examples.")
        
        # Extract features and labels
        questions = [row[0] for row in data]
        agents = [row[1] for row in data]
        
        # Create feature vectors
        question_features = []
        for question in questions:
            features = self.feature_extractor.extract_features(question)
            question_features.append(list(features.values()))
        
        X = np.array(question_features)
        y = np.array(agents)
        
        return X, y, questions
    
    def train_models(self):
        """Train all ML models and select the best one"""
        try:
            X, y, questions = self.prepare_training_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            best_score = 0
            model_results = {}
            
            # Train and evaluate each model
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    model_results[name] = {
                        'accuracy': accuracy,
                        'model': model
                    }
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        self.best_model = model
                        self.best_model_name = name
                        self.model_accuracy = accuracy
                    
                    print(f"Model {name}: Accuracy = {accuracy:.3f}")
                
                except Exception as e:
                    print(f"Error training {name}: {e}")
            
            if self.best_model:
                self.is_trained = True
                self.save_model()
                print(f"Best model: {self.best_model_name} (accuracy: {self.model_accuracy:.3f})")
            else:
                print("No models trained successfully")
                
        except Exception as e:
            print(f"Training error: {e}")
            self.is_trained = False
    
    def predict_routing(self, question: str) -> RoutingPrediction:
        """Predict optimal agent routing for a question"""
        if not self.is_trained or not self.best_model:
            return RoutingPrediction(
                predicted_agent="general",
                confidence=0.5,
                reasoning="ML model not trained",
                alternative_agents=[],
                rule_matches=[],
                ml_prediction=None,
                ml_confidence=None
            )
        
        # Extract features
        features = self.feature_extractor.extract_features(question)
        X = np.array([list(features.values())])
        
        # Make prediction
        try:
            prediction = self.best_model.predict(X)[0]
            
            # Get confidence if model supports it
            confidence = 0.7  # Default confidence
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X)[0]
                confidence = max(probabilities)
                
                # Get alternative agents
                classes = self.best_model.classes_
                agent_probs = list(zip(classes, probabilities))
                agent_probs.sort(key=lambda x: x[1], reverse=True)
                alternatives = agent_probs[1:3]  # Top 2 alternatives
            else:
                alternatives = []
            
            return RoutingPrediction(
                predicted_agent=prediction,
                confidence=confidence,
                reasoning=f"ML prediction using {self.best_model_name} (accuracy: {self.model_accuracy:.3f})",
                alternative_agents=alternatives,
                rule_matches=[],
                ml_prediction=prediction,
                ml_confidence=confidence
            )
            
        except Exception as e:
            return RoutingPrediction(
                predicted_agent="general",
                confidence=0.5,
                reasoning=f"ML prediction failed: {e}",
                alternative_agents=[],
                rule_matches=[],
                ml_prediction=None,
                ml_confidence=None
            )
    
    def save_model(self):
        """Save trained model to disk"""
        if self.best_model and self.is_trained:
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'accuracy': self.model_accuracy,
                'feature_extractor': self.feature_extractor
            }
            joblib.dump(model_data, self.model_path)
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)
                self.best_model = model_data['model']
                self.best_model_name = model_data['model_name']
                self.model_accuracy = model_data['accuracy']
                self.feature_extractor = model_data.get('feature_extractor', self.feature_extractor)
                self.is_trained = True
                print(f"Loaded model: {self.best_model_name} (accuracy: {self.model_accuracy:.3f})")
        except Exception as e:
            print(f"Could not load model: {e}")
            self.is_trained = False
    
    def retrain_if_needed(self, min_new_samples: int = 50):
        """Retrain model if sufficient new feedback is available"""
        if not self.is_trained:
            try:
                self.train_models()
            except Exception as e:
                print(f"Initial training failed: {e}")
                return False
        
        # Check for new feedback samples
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM feedback 
            WHERE timestamp > datetime('now', '-7 days')
            AND rating IS NOT NULL AND rating >= 3
        """)
        
        new_samples = cursor.fetchone()[0]
        conn.close()
        
        if new_samples >= min_new_samples:
            print(f"Retraining with {new_samples} new samples...")
            try:
                self.train_models()
                return True
            except Exception as e:
                print(f"Retraining failed: {e}")
                return False
        
        return False


class EnhancedRoutingEngine:
    """Enhanced routing engine combining rules and ML"""
    
    def __init__(self, db_path: str = "agent_feedback.db"):
        self.ml_engine = MLRoutingEngine(db_path)
        self.rules = []
        self.load_routing_rules()
    
    def load_routing_rules(self):
        """Load learned routing rules from database"""
        # Simplified rule loading - would be more complex in full implementation
        self.rules = [
            RoutingRule(
                rule_id="math_rule_1",
                keywords=["calculate", "solve", "compute", "equation"],
                patterns=[r"\d+\s*[+\-*/]\s*\d+", r"what\s+is\s+\d+"],
                preferred_agent="math",
                confidence_threshold=0.8,
                success_rate=0.9,
                priority=1,
                created_date=datetime.now(),
                last_updated=datetime.now(),
                sample_count=100,
                avg_user_rating=4.2,
                avg_processing_time=1.5,
                avg_cost=0.05
            )
        ]
    
    def predict_optimal_routing(self, question: str) -> RoutingPrediction:
        """Get optimal routing prediction combining rules and ML"""
        # Get ML prediction
        ml_prediction = self.ml_engine.predict_routing(question)
        
        # Apply rule-based routing
        rule_matches = []
        rule_agent = None
        rule_confidence = 0.0
        
        question_lower = question.lower()
        for rule in self.rules:
            # Check keyword matches
            keyword_matches = sum(1 for keyword in rule.keywords if keyword in question_lower)
            if keyword_matches > 0:
                rule_matches.append(f"Rule {rule.rule_id}: {keyword_matches} keywords")
                if rule.success_rate > rule_confidence:
                    rule_agent = rule.preferred_agent
                    rule_confidence = rule.success_rate
        
        # Combine predictions
        if rule_agent and rule_confidence > ml_prediction.confidence:
            final_agent = rule_agent
            final_confidence = rule_confidence
            reasoning = f"Rule-based routing (confidence: {rule_confidence:.3f})"
        else:
            final_agent = ml_prediction.predicted_agent
            final_confidence = ml_prediction.confidence
            reasoning = ml_prediction.reasoning
        
        return RoutingPrediction(
            predicted_agent=final_agent,
            confidence=final_confidence,
            reasoning=reasoning,
            alternative_agents=ml_prediction.alternative_agents,
            rule_matches=rule_matches,
            ml_prediction=ml_prediction.ml_prediction,
            ml_confidence=ml_prediction.ml_confidence
        )


# Global routing engine instance
routing_engine = EnhancedRoutingEngine()


def get_smart_routing_prediction(question: str) -> RoutingPrediction:
    """Convenience function to get routing prediction"""
    return routing_engine.predict_optimal_routing(question)


def train_routing_models():
    """Convenience function to train routing models"""
    routing_engine.ml_engine.train_models()


def retrain_if_needed():
    """Convenience function to retrain if needed"""
    return routing_engine.ml_engine.retrain_if_needed()