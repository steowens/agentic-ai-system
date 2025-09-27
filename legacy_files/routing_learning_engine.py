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
        technical_patterns = [
            r'\b\w+\s*(analysis|calculation|computation|optimization)\b',
            r'\b(formula|equation|algorithm|method|procedure)\b',
            r'\b\w+\s*(factor|coefficient|parameter|variable)\b',
            r'\b(database|query|table|schema|procedure)\b'
        ]
        
        count = 0
        for pattern in technical_patterns:
            count += len(re.findall(pattern, question, re.IGNORECASE))
        
        return float(count)
    
    def _calculate_complexity_score(self, question: str) -> float:
        """Calculate question complexity based on various indicators"""
        complexity_indicators = [
            (r'\bmultiple\s+\w+', 0.2),  # Multiple items
            (r'\bcompare\s+\w+', 0.3),   # Comparison requests
            (r'\banalyze\s+\w+', 0.4),   # Analysis requests
            (r'\boptimize\s+\w+', 0.5),  # Optimization requests
            (r'\bintegrate\s+\w+', 0.3), # Integration requests
            (r'\bcalculate.*and.*', 0.3), # Multiple calculations
        ]
        
        score = 0.0
        for pattern, weight in complexity_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                score += weight
        
        return min(score, 1.0)  # Cap at 1.0


class RoutingLearningEngine:
    """Machine learning engine for improving routing decisions"""
    
    def __init__(self, feedback_db_path: str = "agent_feedback.db"):
        self.feedback_db_path = feedback_db_path
        self.feature_extractor = QuestionFeatureExtractor()
        
        # ML models
        self.nb_model = MultinomialNB()
        self.lr_model = LogisticRegression(random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Model performance tracking
        self.model_performance = {}
        self.current_best_model = None
        
        # Dynamic routing rules
        self.routing_rules = []
        self.rule_performance = {}
        
        # Training data cache
        self.training_data = None
        self.is_trained = False
        
    def load_training_data(self, min_samples_per_agent: int = 5) -> Tuple[List[str], List[str]]:
        """Load training data from feedback database"""
        
        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()
        
        # Get successful routing examples (rating >= 4 or positive feedback)
        cursor.execute("""
            SELECT original_question, actual_agent_used, rating, routing_outcome
            FROM feedback 
            WHERE (rating >= 4 OR feedback_type = 'thumbs_up' OR routing_outcome = 'success')
                AND original_question != ''
                AND actual_agent_used != ''
        """)
        
        positive_samples = cursor.fetchall()
        
        # Get negative examples with suggested corrections
        cursor.execute("""
            SELECT original_question, suggested_agent, rating, routing_outcome
            FROM feedback 
            WHERE suggested_agent IS NOT NULL 
                AND suggested_agent != ''
                AND rating < 4
        """)
        
        negative_samples = cursor.fetchall()
        conn.close()
        
        # Combine and prepare training data
        questions = []
        agents = []
        
        # Add positive samples
        for question, agent, rating, outcome in positive_samples:
            questions.append(question)
            agents.append(agent)
        
        # Add corrected samples (what should have been used)
        for question, suggested_agent, rating, outcome in negative_samples:
            questions.append(question)
            agents.append(suggested_agent)
        
        # Filter agents with insufficient samples
        agent_counts = {}
        for agent in agents:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        filtered_questions = []
        filtered_agents = []
        
        for question, agent in zip(questions, agents):
            if agent_counts[agent] >= min_samples_per_agent:
                filtered_questions.append(question)
                filtered_agents.append(agent)
        
        self.training_data = (filtered_questions, filtered_agents)
        return filtered_questions, filtered_agents
    
    def train_models(self, retrain: bool = False) -> Dict[str, float]:
        """Train all ML models and compare performance"""
        
        if not retrain and self.is_trained:
            return self.model_performance
        
        questions, agents = self.load_training_data()
        
        if len(questions) < 10:
            print("⚠️ Insufficient training data. Need at least 10 samples.")
            return {}
        
        print(f"📚 Training with {len(questions)} samples across {len(set(agents))} agents")
        
        # Extract features
        feature_vectors = []
        for question in questions:
            features = self.feature_extractor.extract_features(question)
            feature_vectors.append(list(features.values()))
        
        X = np.array(feature_vectors)
        y = np.array(agents)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        
        # Train models
        models = {
            'naive_bayes': self.nb_model,
            'logistic_regression': self.lr_model,
            'random_forest': self.rf_model
        }
        
        performance = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                performance[name] = {
                    'accuracy': accuracy,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'agents': list(set(y))
                }
                
                print(f"✅ {name}: {accuracy:.3f} accuracy")
                
            except Exception as e:
                print(f"❌ Failed to train {name}: {e}")
                performance[name] = {'accuracy': 0.0, 'error': str(e)}
        
        # Select best model
        best_model_name = max(performance.keys(), key=lambda k: performance[k].get('accuracy', 0))
        self.current_best_model = best_model_name
        self.model_performance = performance
        self.is_trained = True
        
        print(f"🏆 Best model: {best_model_name} ({performance[best_model_name]['accuracy']:.3f})")
        
        # Save models
        self._save_models()
        
        return performance
    
    def predict_agent(self, question: str) -> RoutingPrediction:
        """Predict best agent for a question using ML and rules"""
        
        # Extract features
        features = self.feature_extractor.extract_features(question)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # ML prediction (if available)
        ml_prediction = None
        ml_confidence = 0.0
        
        if self.is_trained and self.current_best_model:
            try:
                model = getattr(self, f"{self.current_best_model.replace('_', '_')}_model")
                
                # Get prediction
                prediction = model.predict(feature_vector)[0]
                
                # Get confidence (if model supports it)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_vector)[0]
                    ml_confidence = float(np.max(probabilities))
                else:
                    ml_confidence = 0.8  # Default confidence for models without probability
                
                ml_prediction = prediction
                
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
        
        # Rule-based prediction
        rule_matches = []
        rule_prediction = self._apply_routing_rules(question, features)
        
        # Combine predictions
        if ml_prediction and rule_prediction:
            # Weighted combination based on confidence
            if ml_confidence > 0.8:
                final_prediction = ml_prediction
                confidence = ml_confidence
                reasoning = f"High-confidence ML prediction ({self.current_best_model})"
            else:
                final_prediction = rule_prediction['agent']
                confidence = rule_prediction['confidence']
                reasoning = f"Rule-based routing: {rule_prediction['reasoning']}"
        elif ml_prediction:
            final_prediction = ml_prediction
            confidence = ml_confidence
            reasoning = f"ML prediction ({self.current_best_model})"
        elif rule_prediction:
            final_prediction = rule_prediction['agent']
            confidence = rule_prediction['confidence']
            reasoning = f"Rule-based routing: {rule_prediction['reasoning']}"
        else:
            # Fallback to simple heuristics
            final_prediction = self._fallback_prediction(question, features)
            confidence = 0.3
            reasoning = "Fallback heuristic routing"
        
        # Generate alternative suggestions
        alternatives = self._get_alternative_agents(question, features, exclude=final_prediction)
        
        return RoutingPrediction(
            predicted_agent=final_prediction,
            confidence=confidence,
            reasoning=reasoning,
            alternative_agents=alternatives,
            rule_matches=rule_matches,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence
        )
    
    def _apply_routing_rules(self, question: str, features: Dict) -> Optional[Dict]:
        """Apply learned routing rules"""
        
        # Simple rule-based routing based on domain keywords
        domain_scores = {}
        
        for domain in ['math', 'engineering', 'database', 'system', 'esri']:
            domain_scores[domain] = features.get(f'{domain}_score', 0.0)
        
        # Find highest scoring domain
        best_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
        best_score = domain_scores[best_domain]
        
        if best_score > 0.1:  # Threshold for rule activation
            # Map domains to agents (customize for your system)
            domain_agent_map = {
                'math': 'math_agent',
                'engineering': 'math_agent',  # Math agent handles engineering calculations
                'database': 'system_agent',
                'system': 'system_agent', 
                'esri': 'esri_agent'
            }
            
            agent = domain_agent_map.get(best_domain, 'general_agent')
            confidence = min(best_score * 2, 0.9)  # Scale confidence
            
            return {
                'agent': agent,
                'confidence': confidence,
                'reasoning': f"Domain match: {best_domain} (score: {best_score:.2f})"
            }
        
        return None
    
    def _fallback_prediction(self, question: str, features: Dict) -> str:
        """Simple fallback prediction when no ML or rules apply"""
        
        question_lower = question.lower()
        
        # Simple keyword-based fallback
        if any(word in question_lower for word in ['calculate', 'solve', 'equation', 'integral']):
            return 'math_agent'
        elif any(word in question_lower for word in ['file', 'system', 'directory', 'process']):
            return 'system_agent'
        elif any(word in question_lower for word in ['map', 'gis', 'spatial', 'coordinate']):
            return 'esri_agent'  
        else:
            return 'general_agent'
    
    def _get_alternative_agents(self, question: str, features: Dict, exclude: str) -> List[Tuple[str, float]]:
        """Get alternative agent suggestions with confidence scores"""
        
        alternatives = []
        agents = ['math_agent', 'system_agent', 'general_agent', 'esri_agent']
        
        for agent in agents:
            if agent != exclude:
                # Simple scoring based on domain features
                score = 0.0
                
                if agent == 'math_agent':
                    score = features.get('math_score', 0.0) + features.get('engineering_score', 0.0)
                elif agent == 'system_agent':
                    score = features.get('system_score', 0.0) + features.get('database_score', 0.0)
                elif agent == 'esri_agent':
                    score = features.get('esri_score', 0.0)
                else:  # general_agent
                    score = 0.3  # Default baseline
                
                if score > 0.1:
                    alternatives.append((agent, min(score, 0.8)))
        
        return sorted(alternatives, key=lambda x: x[1], reverse=True)[:2]
    
    def update_from_feedback(self, feedback_data: Dict) -> bool:
        """Update routing performance based on new feedback"""
        
        # This would typically trigger model retraining or rule updates
        # For now, we'll implement a simple rule learning mechanism
        
        if feedback_data.get('routing_outcome') == 'wrong_agent' and feedback_data.get('suggested_agent'):
            # Learn from routing corrections
            question = feedback_data['original_question']
            correct_agent = feedback_data['suggested_agent']
            
            features = self.feature_extractor.extract_features(question)
            
            # Simple rule learning: identify strong domain signals
            for domain in ['math', 'engineering', 'database', 'system', 'esri']:
                if features.get(f'{domain}_score', 0.0) > 0.3:
                    print(f"📝 Learning: Questions with {domain} keywords should go to {correct_agent}")
                    # In a full implementation, this would update routing rules database
            
            return True
        
        return False
    
    def _save_models(self):
        """Save trained models to disk"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        for name, model in [
            ('nb', self.nb_model),
            ('lr', self.lr_model), 
            ('rf', self.rf_model)
        ]:
            try:
                joblib.dump(model, models_dir / f"routing_{name}_model.pkl")
            except Exception as e:
                print(f"⚠️ Failed to save {name} model: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        models_dir = Path("models")
        
        if not models_dir.exists():
            return False
        
        try:
            self.nb_model = joblib.load(models_dir / "routing_nb_model.pkl")
            self.lr_model = joblib.load(models_dir / "routing_lr_model.pkl")
            self.rf_model = joblib.load(models_dir / "routing_rf_model.pkl")
            self.is_trained = True
            print("✅ Loaded pre-trained routing models")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load models: {e}")
            return False
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress and model performance"""
        
        return {
            "is_trained": self.is_trained,
            "current_best_model": self.current_best_model,
            "model_performance": self.model_performance,
            "training_data_size": len(self.training_data[0]) if self.training_data else 0,
            "available_agents": list(set(self.training_data[1])) if self.training_data else [],
            "last_updated": datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    
    print("🤖 ROUTING LEARNING ENGINE DEMO")
    print("=" * 50)
    
    # Initialize learning engine
    learning_engine = RoutingLearningEngine("demo_feedback.db")
    
    # Try to load existing models or train new ones
    if not learning_engine.load_models():
        print("🎓 Training new models...")
        performance = learning_engine.train_models()
        
        if performance:
            print(f"📊 Model performance: {performance}")
        else:
            print("⚠️ No training data available. Using rule-based routing.")
    
    # Test predictions
    test_questions = [
        "What is the integral of x^2?",
        "Calculate the stress in a steel beam under 1000N load",
        "Show me all files in the system directory", 
        "Create a map showing elevation data for California",
        "What's the weather like today?"
    ]
    
    print("\n🧪 TESTING ROUTING PREDICTIONS:")
    print("-" * 40)
    
    for question in test_questions:
        prediction = learning_engine.predict_agent(question)
        
        print(f"\nQ: {question}")
        print(f"🎯 Predicted: {prediction.predicted_agent} (confidence: {prediction.confidence:.3f})")
        print(f"💭 Reasoning: {prediction.reasoning}")
        
        if prediction.alternative_agents:
            alternatives = ", ".join([f"{agent} ({conf:.2f})" for agent, conf in prediction.alternative_agents])
            print(f"🔄 Alternatives: {alternatives}")
    
    # Show learning summary
    summary = learning_engine.get_learning_summary()
    print(f"\n📈 LEARNING SUMMARY:")
    print(f"Models trained: {summary['is_trained']}")
    print(f"Best model: {summary.get('current_best_model', 'None')}")
    print(f"Training samples: {summary['training_data_size']}")
    print(f"Available agents: {', '.join(summary.get('available_agents', []))}")
    
    print("\n🎉 Learning engine ready for integration!")