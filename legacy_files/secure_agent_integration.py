"""
Security Integration for AutoGen Agent System
Extends the existing system with comprehensive data protection.
"""
import os
from typing import Dict, Optional
from data_security_framework import DataSanitizer, SecureLLMInterface, SensitivityLevel


class SecureAgentOrchestrator:
    """Enhanced SystemOrchestrator with security controls"""
    
    def __init__(self, original_orchestrator):
        self.orchestrator = original_orchestrator
        self.sanitizer = DataSanitizer(self._get_custom_patterns())
        self.secure_interface = SecureLLMInterface(self.sanitizer)
        
        # Security configuration
        self.security_config = {
            "enforce_data_classification": True,
            "block_restricted_content": True,
            "sanitize_confidential": True,
            "audit_all_requests": True,
            "require_approval_for_sensitive": False,  # Set to True for production
            "allowed_sensitivity_levels": [
                SensitivityLevel.PUBLIC,
                SensitivityLevel.INTERNAL,
                SensitivityLevel.CONFIDENTIAL
                # SensitivityLevel.RESTRICTED not included - will be blocked
            ]
        }
    
    def _get_custom_patterns(self) -> Dict:
        """Define company-specific sensitive patterns"""
        return {
            # Add your company's specific patterns here
            "internal_project": r'\b(ACME|CORP|INTERNAL)[-_]?[A-Z0-9]{4,}\b',
            "customer_code": r'\bCUST[-_]?\d{6,}\b',
            "proprietary_term": r'\b(proprietary|confidential|trade secret)\b',
            
            # Engineering-specific for your domain
            "truss_design": r'\bTRUSS[-_]?[A-Z0-9]{4,}\b',
            "load_calculation": r'\b\d+(?:\.\d+)?\s*(?:kN|lbs|tons?)\b',
            "safety_factor": r'\bSF\s*=\s*\d+(?:\.\d+)?\b'
        }
    
    async def secure_process_question(
        self, 
        question: str, 
        bypass_security: bool = False,
        user_approval: bool = False
    ) -> Dict:
        """Process question with comprehensive security checks"""
        
        # Step 1: Security analysis
        security_result = self.secure_interface.secure_process_question(
            question, 
            bypass_security=bypass_security
        )
        
        # Step 2: Apply security policy
        if security_result["blocked"] and self.security_config["block_restricted_content"]:
            return {
                "success": False,
                "blocked_by_security": True,
                "security_analysis": security_result,
                "response": "❌ This request contains sensitive information that cannot be processed via external APIs.",
                "recommendations": self._get_security_recommendations(security_result["classification"])
            }
        
        # Step 3: Check if user approval is required
        classification = security_result["classification"]
        if (classification.level == SensitivityLevel.CONFIDENTIAL and 
            self.security_config["require_approval_for_sensitive"] and 
            not user_approval):
            
            return {
                "success": False,
                "requires_approval": True,
                "security_analysis": security_result,
                "approval_message": "⚠️ This request contains confidential information. Do you approve processing with external APIs?",
                "risks": classification.risks
            }
        
        # Step 4: Process with original system using sanitized content
        processed_question = security_result["processed_question"]
        
        try:
            # Call original orchestrator with sanitized content
            agent_response = await self.orchestrator.process_question(processed_question)
            
            # Step 5: Enhance response with security metadata
            return {
                "success": True,
                "response": agent_response["response"],
                "agent_used": agent_response.get("agent_used"),
                "routing_confidence": agent_response.get("routing_confidence"),
                "security_analysis": security_result,
                "data_sanitized": security_result["sanitization_applied"],
                "sensitivity_level": classification.level,
                "tokens_used": agent_response.get("tokens_used"),
                "cost": agent_response.get("cost"),
                "processing_time": agent_response.get("processing_time")
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "security_analysis": security_result
            }
    
    def _get_security_recommendations(self, classification) -> Dict:
        """Get actionable security recommendations"""
        
        recommendations = {
            "immediate_actions": [],
            "policy_suggestions": [],
            "alternative_approaches": []
        }
        
        if classification.level == SensitivityLevel.RESTRICTED:
            recommendations["immediate_actions"].extend([
                "Use local/on-premises LLM deployment",
                "Implement additional data encryption",
                "Review and remove sensitive credentials"
            ])
            
            recommendations["alternative_approaches"].extend([
                "Use Azure OpenAI with private endpoints",
                "Deploy open-source LLM (Llama, Mistral) locally",
                "Implement federated learning approach"
            ])
        
        elif classification.level == SensitivityLevel.CONFIDENTIAL:
            recommendations["policy_suggestions"].extend([
                "Implement data loss prevention (DLP) policies", 
                "Use OpenAI Business tier with enhanced controls",
                "Regular security audits and access reviews"
            ])
        
        return recommendations
    
    def get_security_dashboard(self) -> Dict:
        """Get security metrics and status"""
        
        audit_log = self.sanitizer.get_audit_log()
        
        # Calculate security metrics
        total_requests = len(audit_log)
        sensitive_requests = sum(1 for entry in audit_log 
                               if entry["sensitivity_level"] != SensitivityLevel.PUBLIC)
        sanitized_requests = sum(1 for entry in audit_log 
                               if entry["sanitization_applied"])
        
        return {
            "total_requests_processed": total_requests,
            "sensitive_requests": sensitive_requests,
            "sanitization_rate": f"{(sanitized_requests/total_requests*100):.1f}%" if total_requests > 0 else "0%",
            "security_config": self.security_config,
            "recent_patterns_detected": self._get_recent_pattern_summary(audit_log[-10:]),
            "risk_level": self._calculate_overall_risk_level(audit_log)
        }
    
    def _get_recent_pattern_summary(self, recent_logs: list) -> Dict:
        """Summarize recently detected sensitive patterns"""
        pattern_counts = {}
        
        for entry in recent_logs:
            for pattern in entry.get("patterns_detected", []):
                pattern_type = pattern.split(":")[0]
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return pattern_counts
    
    def _calculate_overall_risk_level(self, audit_log: list) -> str:
        """Calculate overall security risk level"""
        
        if not audit_log:
            return "LOW"
        
        recent_entries = audit_log[-20:]  # Last 20 requests
        
        restricted_count = sum(1 for entry in recent_entries 
                             if entry["sensitivity_level"] == SensitivityLevel.RESTRICTED)
        confidential_count = sum(1 for entry in recent_entries 
                               if entry["sensitivity_level"] == SensitivityLevel.CONFIDENTIAL)
        
        if restricted_count > 0:
            return "HIGH"
        elif confidential_count > 5:
            return "MEDIUM"
        else:
            return "LOW"


# Security configuration templates
SECURITY_CONFIGS = {
    "strict": {
        "enforce_data_classification": True,
        "block_restricted_content": True,
        "sanitize_confidential": True,
        "require_approval_for_sensitive": True,
        "allowed_sensitivity_levels": [SensitivityLevel.PUBLIC]
    },
    
    "balanced": {
        "enforce_data_classification": True,
        "block_restricted_content": True,
        "sanitize_confidential": True,
        "require_approval_for_sensitive": False,
        "allowed_sensitivity_levels": [
            SensitivityLevel.PUBLIC,
            SensitivityLevel.INTERNAL,
            SensitivityLevel.CONFIDENTIAL
        ]
    },
    
    "permissive": {
        "enforce_data_classification": False,
        "block_restricted_content": False,
        "sanitize_confidential": False,
        "require_approval_for_sensitive": False,
        "allowed_sensitivity_levels": [
            SensitivityLevel.PUBLIC,
            SensitivityLevel.INTERNAL,
            SensitivityLevel.CONFIDENTIAL,
            SensitivityLevel.RESTRICTED
        ]
    }
}


if __name__ == "__main__":
    print("🛡️ SECURE AGENT ORCHESTRATOR")
    print("This module extends your existing AutoGen system with security controls.")
    print("Integration example:")
    print("""
    from main import SystemOrchestrator
    from secure_agent_integration import SecureAgentOrchestrator
    
    # Create your existing orchestrator
    orchestrator = SystemOrchestrator()
    
    # Wrap it with security
    secure_orchestrator = SecureAgentOrchestrator(orchestrator)
    
    # Process questions securely
    result = await secure_orchestrator.secure_process_question("Your question here")
    """)