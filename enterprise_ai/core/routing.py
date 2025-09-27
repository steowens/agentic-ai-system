"""
Intelligent routing service following SOLID principles.
Separates routing logic from agent management and tool concerns.
"""
from typing import Dict, Tuple
import re


class RoutingService:
    """Service for routing questions to appropriate agents - Single Responsibility Principle"""
    
    # Mathematical keywords for classification
    MATH_KEYWORDS = [
        'calculate', 'compute', 'solve', 'what is', 'find', 'evaluate', 'determine',
        'multiply', 'divide', 'add', 'subtract', 'square', 'sqrt', 'power', 'exponent',
        'derivative', 'integral', 'integrate', 'differentiate', 'limit', 'series',
        'equation', 'formula', 'function', 'graph', 'plot', 'sin', 'cos', 'tan',
        'logarithm', 'log', 'ln', 'factorial', 'fibonacci', 'prime', 'matrix',
        'vector', 'geometry', 'algebra', 'calculus', 'statistics', 'probability',
        'degrees', 'radians', 'convert', 'unit', 'measurement', 'height', 'width',
        'length', 'area', 'volume', 'distance', 'pitch', 'slope', 'angle', 'ratio',
        'span', 'truss', 'beam', 'load', 'force', 'pressure', 'dimension'
    ]
    
    # File system keywords
    FILE_KEYWORDS = [
        'file', 'directory', 'folder', 'path', 'read', 'write', 'list', 'ls', 'dir',
        'create', 'delete', 'move', 'copy', 'exists', 'size', 'permissions',
        'current directory', 'working directory', 'contents', 'text file'
    ]

    # Wordle keywords
    WORDLE_KEYWORDS = [
        'wordle', 'word', 'letters', 'position', 'constraint', 'exclude', 'include',
        'contains', 'letter at', 'position', 'must contain', 'cannot contain',
        'excluded', 'included', 'puzzle', 'five letter', '5 letter'
    ]
    
    def __init__(self):
        # Compile regex patterns for better performance
        self.math_patterns = [
            r'\b\d+\s*[\+\-\*/\^]\s*\d+',  # Basic arithmetic expressions
            r'\b\d+\s*\*\*\s*\d+',         # Power notation
            r'\bsin\s*\(|\bcos\s*\(|\btan\s*\(',  # Trig functions
            r'\bsqrt\s*\(|\blog\s*\(|\bln\s*\(',  # Other math functions
            r'\bintegral|\bderivative|\blimit',      # Calculus terms
            r'\d+\s*(degrees?|radians?|mph|m\/s|kg|pounds?)',  # Units
        ]
        self.compiled_math_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.math_patterns]
    
    def classify_question(self, question: str) -> str:
        """Classify a question to determine appropriate agent"""
        question_lower = question.lower()
        
        # Check for mathematical expressions using regex patterns
        for pattern in self.compiled_math_patterns:
            if pattern.search(question):
                return "math"
        
        # Check for mathematical keywords
        math_score = sum(1 for keyword in self.MATH_KEYWORDS if keyword in question_lower)

        # Check for file system keywords
        file_score = sum(1 for keyword in self.FILE_KEYWORDS if keyword in question_lower)

        # Check for Wordle keywords
        wordle_score = sum(1 for keyword in self.WORDLE_KEYWORDS if keyword in question_lower)

        # Decide based on scores
        if math_score > max(file_score, wordle_score) and math_score > 0:
            return "math"
        elif wordle_score > max(math_score, file_score) and wordle_score > 0:
            return "wordle"
        elif file_score > 0:
            return "system"
        else:
            return "general"
    
    def get_routing_info(self, question: str) -> Tuple[str, str, float]:
        """
        Get detailed routing information
        Returns: (agent_type, reasoning, confidence)
        """
        question_lower = question.lower()
        
        # Pattern matching for high confidence routing
        for pattern in self.compiled_math_patterns:
            if pattern.search(question):
                return ("math", "Mathematical expression detected", 0.9)
        
        # Keyword analysis
        math_score = sum(1 for keyword in self.MATH_KEYWORDS if keyword in question_lower)
        file_score = sum(1 for keyword in self.FILE_KEYWORDS if keyword in question_lower)
        wordle_score = sum(1 for keyword in self.WORDLE_KEYWORDS if keyword in question_lower)

        # Find the highest scoring category
        scores = [
            (math_score, "math", "Math keywords detected"),
            (wordle_score, "wordle", "Wordle keywords detected"),
            (file_score, "system", "File system keywords detected")
        ]

        # Sort by score (highest first)
        scores.sort(key=lambda x: x[0], reverse=True)
        top_score, top_agent, top_reason = scores[0]

        if top_score > 0:
            confidence = min(0.8, 0.3 + (top_score * 0.1))
            return (top_agent, f"{top_reason} (score: {top_score})", confidence)
        else:
            return ("general", "No specific domain keywords detected", 0.5)


class QuestionAnalyzer:
    """Analyzes questions for routing decisions - Single Responsibility Principle"""
    
    @staticmethod
    def extract_mathematical_expressions(question: str) -> list:
        """Extract potential mathematical expressions from a question"""
        expressions = []
        
        # Common mathematical expression patterns
        patterns = [
            r'\b\d+\s*[\+\-\*/\^]\s*\d+(?:\s*[\+\-\*/\^]\s*\d+)*',
            r'\b\d+\s*\*\*\s*\d+',
            r'\b(?:sin|cos|tan|sqrt|log|ln)\s*\([^)]+\)',
            r'\b\d+\s*[a-zA-Z]+(?:\s*to\s*[a-zA-Z]+)?'  # Unit conversions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            expressions.extend(matches)
        
        return expressions
    
    @staticmethod
    def is_calculation_needed(question: str) -> bool:
        """Determine if a question requires actual calculation"""
        calculation_indicators = [
            'what is', 'calculate', 'compute', 'solve for', 'find the value',
            'result of', 'equals', '=', 'answer'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in calculation_indicators)
    
    @staticmethod
    def is_conceptual_question(question: str) -> bool:
        """Determine if a question is asking for explanation rather than calculation"""
        conceptual_indicators = [
            'what is a', 'what are', 'explain', 'how does', 'why is',
            'definition of', 'meaning of', 'prove that', 'show that'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in conceptual_indicators)