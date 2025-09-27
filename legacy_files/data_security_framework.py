"""
Data Security and Privacy Framework for LLM-based Engineering Applications
Implements multiple layers of protection for sensitive data processing.
"""
import re
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class SensitivityLevel:
    """Define data sensitivity classifications"""
    PUBLIC = "public"           # Can be sent to external APIs
    INTERNAL = "internal"       # Company internal, sanitize before API
    CONFIDENTIAL = "confidential"  # Remove/mask before API  
    RESTRICTED = "restricted"   # Never send to external APIs


@dataclass 
class DataClassification:
    """Result of data sensitivity analysis"""
    level: str
    detected_patterns: List[str]
    risks: List[str]
    recommendations: List[str]
    sanitized_content: Optional[str] = None


class DataSanitizer:
    """Sanitize sensitive data before sending to LLM APIs"""
    
    # Patterns for detecting sensitive information
    SENSITIVE_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        "file_path": r'[C-Z]:\\[^\\/:*?"<>|\r\n]+',
        "server_name": r'\b[a-zA-Z0-9-]+\.(local|corp|internal|dev|staging|prod)\b',
        "api_key": r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)[\s=:]["\']*([a-zA-Z0-9_-]{20,})',
        "password": r'(?i)(password|pwd|pass)[\s=:]["\']*([^\s"\']{8,})',
        
        # Engineering-specific patterns
        "blueprint_id": r'\b(BP|BLUEPRINT|DWG)[-_]?\d{4,}\b',
        "part_number": r'\b(P/N|PN|PART)[-_:#]?[A-Z0-9-]{6,}\b',
        "project_code": r'\b(PROJ|PROJECT)[-_]?[A-Z0-9]{4,8}\b',
        "material_spec": r'\b(ASTM|ISO|ANSI|DIN)[-_]?[A-Z0-9]{3,}\b',
        "coordinate": r'\b\d+[°′″]\d+[′″]?\d*[″]?\s*[NSEW]\b',
        "dimension": r'\b\d+(?:\.\d+)?[\s]*(?:mm|cm|m|in|ft|′|″)\b',
        
        # Company-specific (customizable)
        "employee_id": r'\bEMP[-_]?\d{4,8}\b',
        "contract_id": r'\bCON[-_]?\d{6,10}\b',
        "facility_code": r'\bFAC[-_]?[A-Z0-9]{4,6}\b'
    }
    
    # Engineering terms that might be sensitive in context
    ENGINEERING_SENSITIVE_TERMS = [
        "proprietary", "confidential", "trade secret", "patent pending",
        "internal use only", "restricted access", "classified",
        "stress analysis", "failure mode", "design defect", "safety factor",
        "load capacity", "material properties", "test results"
    ]
    
    def __init__(self, custom_patterns: Optional[Dict] = None):
        self.patterns = self.SENSITIVE_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Setup logging
        self.logger = logging.getLogger("data_sanitizer")
        self.audit_log = []
    
    def classify_data_sensitivity(self, content: str) -> DataClassification:
        """Analyze content and classify its sensitivity level"""
        
        detected_patterns = []
        risks = []
        content_lower = content.lower()
        
        # Check for sensitive patterns
        for pattern_name, pattern_regex in self.patterns.items():
            matches = re.findall(pattern_regex, content, re.IGNORECASE)
            if matches:
                detected_patterns.append(f"{pattern_name}: {len(matches)} matches")
        
        # Check for engineering sensitive terms
        sensitive_terms_found = []
        for term in self.ENGINEERING_SENSITIVE_TERMS:
            if term.lower() in content_lower:
                sensitive_terms_found.append(term)
        
        # Determine sensitivity level and risks
        if any(p.startswith(('api_key', 'password', 'credit_card', 'ssn')) for p in detected_patterns):
            level = SensitivityLevel.RESTRICTED
            risks.extend([
                "Contains authentication credentials",
                "May violate data protection regulations",
                "High risk of unauthorized access"
            ])
        
        elif any(p.startswith(('blueprint_id', 'part_number', 'project_code')) for p in detected_patterns):
            level = SensitivityLevel.CONFIDENTIAL
            risks.extend([
                "Contains proprietary engineering data",
                "May reveal competitive information",
                "Could violate NDAs or trade secrets"
            ])
        
        elif sensitive_terms_found or any(p.startswith(('email', 'phone', 'server_name')) for p in detected_patterns):
            level = SensitivityLevel.INTERNAL
            risks.extend([
                "Contains internal company information",
                "May reveal business processes",
                "Could be used for social engineering"
            ])
        
        else:
            level = SensitivityLevel.PUBLIC
        
        # Generate recommendations
        recommendations = self._get_recommendations(level, detected_patterns)
        
        return DataClassification(
            level=level,
            detected_patterns=detected_patterns,
            risks=risks,
            recommendations=recommendations
        )
    
    def sanitize_content(self, content: str, classification: DataClassification) -> str:
        """Sanitize content based on sensitivity classification"""
        
        if classification.level == SensitivityLevel.RESTRICTED:
            return "[CONTENT BLOCKED: Contains restricted information]"
        
        sanitized = content
        
        # Apply sanitization based on sensitivity level
        if classification.level in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.INTERNAL]:
            
            # Replace sensitive patterns with placeholders
            for pattern_name, pattern_regex in self.patterns.items():
                if pattern_name in ['api_key', 'password']:
                    sanitized = re.sub(pattern_regex, '[CREDENTIALS_REMOVED]', sanitized, flags=re.IGNORECASE)
                elif pattern_name in ['email']:
                    sanitized = re.sub(pattern_regex, '[EMAIL_REMOVED]', sanitized)
                elif pattern_name in ['phone']:
                    sanitized = re.sub(pattern_regex, '[PHONE_REMOVED]', sanitized)
                elif pattern_name in ['ip_address']:
                    sanitized = re.sub(pattern_regex, '[IP_REMOVED]', sanitized)
                elif pattern_name in ['file_path']:
                    sanitized = re.sub(pattern_regex, '[PATH_REMOVED]', sanitized)
                elif pattern_name in ['server_name']:
                    sanitized = re.sub(pattern_regex, '[SERVER_REMOVED]', sanitized)
                elif pattern_name in ['blueprint_id', 'part_number', 'project_code']:
                    # For engineering data, use generic placeholders
                    sanitized = re.sub(pattern_regex, f'[{pattern_name.upper()}_REMOVED]', sanitized, flags=re.IGNORECASE)
        
        # Log sanitization action
        self._log_sanitization(content, sanitized, classification)
        
        return sanitized
    
    def _get_recommendations(self, level: str, detected_patterns: List[str]) -> List[str]:
        """Generate security recommendations based on classification"""
        
        recommendations = []
        
        if level == SensitivityLevel.RESTRICTED:
            recommendations.extend([
                "DO NOT send to external APIs",
                "Use local/on-premises LLM instead", 
                "Implement additional encryption",
                "Require explicit approval for processing"
            ])
        
        elif level == SensitivityLevel.CONFIDENTIAL:
            recommendations.extend([
                "Sanitize before sending to APIs",
                "Use data masking techniques",
                "Consider on-premises deployment",
                "Implement audit logging"
            ])
        
        elif level == SensitivityLevel.INTERNAL:
            recommendations.extend([
                "Apply data sanitization",
                "Monitor API usage",
                "Use business-tier API with data controls",
                "Regular security reviews"
            ])
        
        else:  # PUBLIC
            recommendations.extend([
                "Safe to use with standard APIs",
                "Still monitor for unexpected sensitive data"
            ])
        
        return recommendations
    
    def _log_sanitization(self, original: str, sanitized: str, classification: DataClassification):
        """Log sanitization actions for audit purposes"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sensitivity_level": classification.level,
            "patterns_detected": classification.detected_patterns,
            "risks": classification.risks,
            "original_length": len(original),
            "sanitized_length": len(sanitized),
            "sanitization_applied": original != sanitized,
            "hash_original": hashlib.sha256(original.encode()).hexdigest()[:16]
        }
        
        self.audit_log.append(log_entry)
        self.logger.info(f"Data sanitization: {classification.level} - {len(classification.detected_patterns)} patterns detected")
    
    def get_audit_log(self) -> List[Dict]:
        """Get sanitization audit log"""
        return self.audit_log.copy()


class SecureLLMInterface:
    """Secure wrapper for LLM API calls with data protection"""
    
    def __init__(self, sanitizer: DataSanitizer):
        self.sanitizer = sanitizer
        self.logger = logging.getLogger("secure_llm")
    
    def secure_process_question(self, question: str, bypass_security: bool = False) -> Dict:
        """Process question with security checks and data sanitization"""
        
        # Step 1: Classify data sensitivity
        classification = self.sanitizer.classify_data_sensitivity(question)
        
        # Step 2: Security decision
        if not bypass_security:
            if classification.level == SensitivityLevel.RESTRICTED:
                return {
                    "blocked": True,
                    "reason": "Content contains restricted information",
                    "classification": classification,
                    "recommendations": classification.recommendations
                }
        
        # Step 3: Sanitize content if needed
        processed_question = question
        if classification.level in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.INTERNAL]:
            processed_question = self.sanitizer.sanitize_content(question, classification)
        
        # Step 4: Log security analysis
        self.logger.info(f"Question processed - Sensitivity: {classification.level}, Sanitized: {processed_question != question}")
        
        return {
            "blocked": False,
            "original_question": question,
            "processed_question": processed_question,
            "classification": classification,
            "sanitization_applied": processed_question != question
        }


# Example usage and testing
if __name__ == "__main__":
    
    print("🔒 DATA SECURITY & PRIVACY FRAMEWORK DEMO")
    print("=" * 60)
    
    # Initialize security framework
    sanitizer = DataSanitizer()
    secure_interface = SecureLLMInterface(sanitizer)
    
    # Test cases with different sensitivity levels
    test_questions = [
        # Public - safe
        "What is the integral of x^2?",
        
        # Internal - contains internal info
        "How do I connect to server-01.corp using admin credentials?",
        
        # Confidential - engineering data
        "Analyze stress test results for part PN-12345 in project PROJ-2024-BRIDGE",
        
        # Restricted - contains credentials
        "My API key is sk-abc123xyz456 and password is SecretPass2024, help me debug"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🧪 TEST {i}: {question[:50]}...")
        print("-" * 40)
        
        result = secure_interface.secure_process_question(question)
        
        if result["blocked"]:
            print("🚫 BLOCKED - Content too sensitive for external API")
            print(f"📋 Reason: {result['reason']}")
        else:
            classification = result["classification"]
            print(f"🎯 Sensitivity: {classification.level}")
            print(f"🔍 Patterns: {', '.join(classification.detected_patterns) if classification.detected_patterns else 'None'}")
            print(f"🛡️ Sanitized: {result['sanitization_applied']}")
            
            if result["sanitization_applied"]:
                print(f"📝 Processed: {result['processed_question'][:100]}...")
        
        print(f"💡 Recommendations: {', '.join(result['classification'].recommendations[:2])}")
    
    print(f"\n📊 Audit log entries: {len(sanitizer.get_audit_log())}")