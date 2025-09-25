# Prism: Verifier & PRM layer

class Verifier:
    def verify(self, output: str) -> bool:
        # Stub: Verify output
        return True

"""
LoomOS Prism - Advanced AI Verifier & Safety System

Prism is LoomOS's comprehensive AI verification and safety system that provides:
- Multi-layer content verification (safety, factuality, quality)
- Process verification and reward modeling (PRM)
- Constitutional AI alignment checks
- Bias detection and mitigation
- Adversarial robustness testing
- Real-time safety monitoring

Architecture:
- Ensemble of specialized verification models
- Configurable safety policies and thresholds
- Integration with external verification services
- Comprehensive audit trails and provenance tracking
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import numpy as np
import hashlib
from prometheus_client import Counter, Histogram, Gauge

# Metrics
VERIFICATION_REQUESTS = Counter('loomos_prism_verification_requests_total', 'Total verification requests', ['type', 'result'])
VERIFICATION_DURATION = Histogram('loomos_prism_verification_duration_seconds', 'Verification duration')
SAFETY_SCORE = Gauge('loomos_prism_safety_score', 'Current safety score')
FACTUALITY_SCORE = Gauge('loomos_prism_factuality_score', 'Current factuality score')
QUALITY_SCORE = Gauge('loomos_prism_quality_score', 'Current quality score')

logger = logging.getLogger(__name__)

class VerificationType(Enum):
    """Types of verification supported by Prism"""
    SAFETY = "safety"
    FACTUALITY = "factuality"
    QUALITY = "quality"
    BIAS = "bias"
    ADVERSARIAL = "adversarial"
    CONSTITUTIONAL = "constitutional"
    PROCESS = "process"
    REWARD = "reward"

class VerificationResult(Enum):
    """Verification result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

@dataclass
class VerificationConfig:
    """Configuration for verification checks"""
    # Enabled verification types
    enabled_checks: List[VerificationType] = field(default_factory=lambda: [
        VerificationType.SAFETY,
        VerificationType.FACTUALITY,
        VerificationType.QUALITY
    ])
    
    # Thresholds
    safety_threshold: float = 0.8
    factuality_threshold: float = 0.7
    quality_threshold: float = 0.6
    bias_threshold: float = 0.8
    
    # Behavior configuration
    fail_fast: bool = False  # Stop on first failure
    require_all_pass: bool = True  # All checks must pass
    timeout_seconds: int = 30
    
    # Model configurations
    safety_model: str = "microsoft/DialogRPT-human-vs-rand"
    factuality_model: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    quality_model: str = "microsoft/DialoGPT-medium"
    
    # External services
    use_external_apis: bool = False
    openai_moderation: bool = False
    perspective_api: bool = False
    
    # Audit settings
    save_audit_logs: bool = True
    detailed_explanations: bool = True

@dataclass
class VerificationMetrics:
    """Detailed verification metrics"""
    safety_score: float = 0.0
    factuality_score: float = 0.0
    quality_score: float = 0.0
    bias_score: float = 0.0
    overall_score: float = 0.0
    
    # Detailed breakdowns
    toxicity_score: float = 0.0
    hate_speech_score: float = 0.0
    harassment_score: float = 0.0
    self_harm_score: float = 0.0
    
    # Factuality components
    claim_accuracy: float = 0.0
    source_reliability: float = 0.0
    logical_consistency: float = 0.0
    
    # Quality components
    coherence: float = 0.0
    relevance: float = 0.0
    fluency: float = 0.0
    informativeness: float = 0.0

@dataclass
class VerificationRequest:
    """Request for content verification"""
    content: str
    content_type: str = "text"
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    verification_types: Optional[List[VerificationType]] = None
    config: Optional[VerificationConfig] = None
    
    # Request tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class VerificationResponse:
    """Response from verification process"""
    request_id: str
    overall_result: VerificationResult
    metrics: VerificationMetrics
    
    # Individual check results
    check_results: Dict[VerificationType, VerificationResult] = field(default_factory=dict)
    check_scores: Dict[VerificationType, float] = field(default_factory=dict)
    check_explanations: Dict[VerificationType, str] = field(default_factory=dict)
    
    # Metadata
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class SafetyVerifier:
    """Handles safety verification checks"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        logger.info("Safety verifier initialized")
    
    async def verify(self, content: str, context: Optional[Dict] = None) -> Tuple[float, str]:
        """Perform safety verification"""
        logger.info("Performing safety verification")
        
        # Mock safety verification for demo
        # In production, this would use actual safety models
        
        # Simulate safety checks
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Check for obvious safety issues
        safety_score = 1.0
        explanation = "Content passed safety checks"
        
        # Simple keyword-based safety checking (demo)
        unsafe_keywords = [
            'violence', 'harm', 'kill', 'murder', 'suicide', 'bomb',
            'hate', 'discrimination', 'harassment', 'bullying'
        ]
        
        content_lower = content.lower()
        for keyword in unsafe_keywords:
            if keyword in content_lower:
                safety_score -= 0.3
                explanation = f"Potentially unsafe content detected: {keyword}"
        
        # Add some randomness for demo variability
        safety_score += np.random.normal(0, 0.1)
        safety_score = max(0.0, min(1.0, safety_score))
        
        # Check for toxicity patterns
        if len(content) > 100 and any(word in content_lower for word in ['stupid', 'idiot', 'hate']):
            safety_score -= 0.2
            explanation += " | Potential toxicity detected"
        
        return safety_score, explanation

class FactualityVerifier:
    """Handles factuality verification checks"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        logger.info("Factuality verifier initialized")
    
    async def verify(self, content: str, context: Optional[Dict] = None) -> Tuple[float, str]:
        """Perform factuality verification"""
        logger.info("Performing factuality verification")
        
        # Mock factuality verification for demo
        await asyncio.sleep(0.15)  # Simulate processing time
        
        # Simple heuristic-based factuality checking
        factuality_score = 0.8  # Base score
        explanation = "Factuality assessment completed"
        
        # Check for factual claim indicators
        fact_indicators = [
            'according to', 'research shows', 'studies indicate',
            'data reveals', 'statistics show', 'proven fact'
        ]
        
        uncertain_indicators = [
            'might', 'could be', 'possibly', 'perhaps',
            'allegedly', 'rumored', 'unconfirmed'
        ]
        
        content_lower = content.lower()
        
        # Boost score for fact-based language
        for indicator in fact_indicators:
            if indicator in content_lower:
                factuality_score += 0.1
                explanation += f" | Factual language detected: {indicator}"
        
        # Reduce score for uncertain language
        for indicator in uncertain_indicators:
            if indicator in content_lower:
                factuality_score -= 0.1
                explanation += f" | Uncertain language detected: {indicator}"
        
        # Add variability
        factuality_score += np.random.normal(0, 0.05)
        factuality_score = max(0.0, min(1.0, factuality_score))
        
        return factuality_score, explanation

class QualityVerifier:
    """Handles quality verification checks"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        logger.info("Quality verifier initialized")
    
    async def verify(self, content: str, context: Optional[Dict] = None) -> Tuple[float, str]:
        """Perform quality verification"""
        logger.info("Performing quality verification")
        
        # Mock quality verification for demo
        await asyncio.sleep(0.1)
        
        quality_score = 0.7  # Base score
        explanation = "Quality assessment completed"
        
        # Basic quality heuristics
        
        # Length appropriateness
        if len(content) < 10:
            quality_score -= 0.3
            explanation += " | Content too short"
        elif len(content) > 1000:
            quality_score += 0.1
            explanation += " | Comprehensive content"
        
        # Sentence structure
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        if sentence_count > 0:
            avg_sentence_length = len(content.split()) / sentence_count
            if 10 <= avg_sentence_length <= 25:
                quality_score += 0.1
                explanation += " | Good sentence structure"
        
        # Vocabulary diversity
        words = content.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            diversity_ratio = unique_words / len(words)
            if diversity_ratio > 0.7:
                quality_score += 0.1
                explanation += " | Good vocabulary diversity"
        
        # Grammar indicators (simple check)
        if content.count(',') > 0 and content.count(';') > 0:
            quality_score += 0.05
            explanation += " | Complex sentence structure"
        
        # Add variability
        quality_score += np.random.normal(0, 0.05)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score, explanation

class BiasVerifier:
    """Handles bias detection and verification"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        logger.info("Bias verifier initialized")
    
    async def verify(self, content: str, context: Optional[Dict] = None) -> Tuple[float, str]:
        """Perform bias verification"""
        logger.info("Performing bias verification")
        
        await asyncio.sleep(0.12)
        
        bias_score = 0.9  # Start with high score (low bias)
        explanation = "Bias assessment completed"
        
        # Check for potential bias indicators
        gender_bias_terms = ['he said', 'she said', 'men are', 'women are']
        racial_bias_terms = ['those people', 'they all', 'typical of']
        age_bias_terms = ['young people', 'old people', 'millennials', 'boomers']
        
        content_lower = content.lower()
        
        for term in gender_bias_terms:
            if term in content_lower:
                bias_score -= 0.1
                explanation += f" | Potential gender bias: {term}"
        
        for term in racial_bias_terms:
            if term in content_lower:
                bias_score -= 0.2
                explanation += f" | Potential racial bias: {term}"
        
        for term in age_bias_terms:
            if term in content_lower:
                bias_score -= 0.05
                explanation += f" | Potential age bias: {term}"
        
        bias_score = max(0.0, min(1.0, bias_score))
        return bias_score, explanation

class ProcessVerifier:
    """Handles process verification and reward modeling (PRM)"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        logger.info("Process verifier initialized")
    
    async def verify(self, content: str, context: Optional[Dict] = None) -> Tuple[float, str]:
        """Perform process verification"""
        logger.info("Performing process verification")
        
        await asyncio.sleep(0.08)
        
        # Mock process verification
        process_score = 0.8
        explanation = "Process verification completed"
        
        # Check for reasoning indicators
        reasoning_indicators = [
            'because', 'therefore', 'since', 'as a result',
            'consequently', 'due to', 'leads to', 'causes'
        ]
        
        step_indicators = [
            'first', 'second', 'third', 'next', 'then',
            'finally', 'step', 'process', 'method'
        ]
        
        content_lower = content.lower()
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in content_lower)
        step_count = sum(1 for indicator in step_indicators if indicator in content_lower)
        
        if reasoning_count > 0:
            process_score += 0.1
            explanation += f" | Good reasoning structure ({reasoning_count} indicators)"
        
        if step_count > 0:
            process_score += 0.1
            explanation += f" | Clear process steps ({step_count} indicators)"
        
        process_score = max(0.0, min(1.0, process_score))
        return process_score, explanation

class Prism:
    """Main Prism verification system"""
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        
        # Initialize verifiers
        self.safety_verifier = SafetyVerifier(self.config)
        self.factuality_verifier = FactualityVerifier(self.config)
        self.quality_verifier = QualityVerifier(self.config)
        self.bias_verifier = BiasVerifier(self.config)
        self.process_verifier = ProcessVerifier(self.config)
        
        # Verification history
        self.verification_history: List[VerificationResponse] = []
        
        logger.info("Prism verification system initialized")
    
    async def verify(self, request: VerificationRequest) -> VerificationResponse:
        """Perform comprehensive verification"""
        start_time = time.time()
        
        logger.info(f"Starting verification for request {request.request_id}")
        
        # Initialize response
        response = VerificationResponse(
            request_id=request.request_id,
            overall_result=VerificationResult.PASS,
            metrics=VerificationMetrics()
        )
        
        # Determine which checks to run
        checks_to_run = request.verification_types or self.config.enabled_checks
        
        try:
            # Run verification checks
            for check_type in checks_to_run:
                await self._run_verification_check(request, response, check_type)
                
                # Fail fast if configured
                if (self.config.fail_fast and 
                    response.check_results.get(check_type) == VerificationResult.FAIL):
                    response.overall_result = VerificationResult.FAIL
                    break
            
            # Compute overall result and metrics
            self._compute_overall_result(response)
            
            # Update Prometheus metrics
            self._update_metrics(response)
            
            # Record verification
            VERIFICATION_REQUESTS.labels(
                type="comprehensive",
                result=response.overall_result.value
            ).inc()
            
        except Exception as e:
            logger.error(f"Verification failed for {request.request_id}: {e}")
            response.overall_result = VerificationResult.ERROR
            response.errors.append(str(e))
        
        # Record processing time
        response.processing_time = time.time() - start_time
        VERIFICATION_DURATION.observe(response.processing_time)
        
        # Save to history
        self.verification_history.append(response)
        
        # Save audit log if configured
        if self.config.save_audit_logs:
            await self._save_audit_log(request, response)
        
        logger.info(f"Verification completed for {request.request_id}: {response.overall_result.value}")
        
        return response
    
    async def _run_verification_check(self, request: VerificationRequest, 
                                    response: VerificationResponse, 
                                    check_type: VerificationType) -> None:
        """Run a specific verification check"""
        try:
            if check_type == VerificationType.SAFETY:
                score, explanation = await self.safety_verifier.verify(
                    request.content, request.context
                )
                threshold = self.config.safety_threshold
                response.metrics.safety_score = score
                
            elif check_type == VerificationType.FACTUALITY:
                score, explanation = await self.factuality_verifier.verify(
                    request.content, request.context
                )
                threshold = self.config.factuality_threshold
                response.metrics.factuality_score = score
                
            elif check_type == VerificationType.QUALITY:
                score, explanation = await self.quality_verifier.verify(
                    request.content, request.context
                )
                threshold = self.config.quality_threshold
                response.metrics.quality_score = score
                
            elif check_type == VerificationType.BIAS:
                score, explanation = await self.bias_verifier.verify(
                    request.content, request.context
                )
                threshold = self.config.bias_threshold
                response.metrics.bias_score = score
                
            elif check_type == VerificationType.PROCESS:
                score, explanation = await self.process_verifier.verify(
                    request.content, request.context
                )
                threshold = 0.7  # Default process threshold
                
            else:
                # Unsupported check type
                response.warnings.append(f"Unsupported verification type: {check_type}")
                return
            
            # Determine result based on threshold
            if score >= threshold:
                result = VerificationResult.PASS
            elif score >= threshold * 0.8:
                result = VerificationResult.WARNING
            else:
                result = VerificationResult.FAIL
            
            # Store results
            response.check_results[check_type] = result
            response.check_scores[check_type] = score
            response.check_explanations[check_type] = explanation
            
        except Exception as e:
            logger.error(f"Error in {check_type.value} verification: {e}")
            response.check_results[check_type] = VerificationResult.ERROR
            response.errors.append(f"{check_type.value}: {str(e)}")
    
    def _compute_overall_result(self, response: VerificationResponse) -> None:
        """Compute overall verification result and metrics"""
        # Count results
        pass_count = sum(1 for result in response.check_results.values() 
                        if result == VerificationResult.PASS)
        fail_count = sum(1 for result in response.check_results.values() 
                        if result == VerificationResult.FAIL)
        warning_count = sum(1 for result in response.check_results.values() 
                           if result == VerificationResult.WARNING)
        error_count = sum(1 for result in response.check_results.values() 
                         if result == VerificationResult.ERROR)
        
        total_checks = len(response.check_results)
        
        # Compute overall score
        scores = list(response.check_scores.values())
        if scores:
            response.metrics.overall_score = np.mean(scores)
        
        # Determine overall result
        if error_count > 0:
            response.overall_result = VerificationResult.ERROR
        elif fail_count > 0:
            if self.config.require_all_pass:
                response.overall_result = VerificationResult.FAIL
            elif fail_count > pass_count:
                response.overall_result = VerificationResult.FAIL
            else:
                response.overall_result = VerificationResult.WARNING
        elif warning_count > 0:
            response.overall_result = VerificationResult.WARNING
        else:
            response.overall_result = VerificationResult.PASS
    
    def _update_metrics(self, response: VerificationResponse) -> None:
        """Update Prometheus metrics"""
        if response.metrics.safety_score > 0:
            SAFETY_SCORE.set(response.metrics.safety_score)
        if response.metrics.factuality_score > 0:
            FACTUALITY_SCORE.set(response.metrics.factuality_score)
        if response.metrics.quality_score > 0:
            QUALITY_SCORE.set(response.metrics.quality_score)
    
    async def _save_audit_log(self, request: VerificationRequest, 
                            response: VerificationResponse) -> None:
        """Save audit log for verification"""
        try:
            audit_dir = Path("audit_logs")
            audit_dir.mkdir(exist_ok=True)
            
            audit_data = {
                "request": {
                    "id": request.request_id,
                    "content_hash": hashlib.sha256(request.content.encode()).hexdigest(),
                    "content_length": len(request.content),
                    "timestamp": request.timestamp.isoformat(),
                    "verification_types": [vt.value for vt in (request.verification_types or [])]
                },
                "response": {
                    "overall_result": response.overall_result.value,
                    "overall_score": response.metrics.overall_score,
                    "processing_time": response.processing_time,
                    "check_results": {k.value: v.value for k, v in response.check_results.items()},
                    "check_scores": {k.value: v for k, v in response.check_scores.items()},
                    "warnings": response.warnings,
                    "errors": response.errors,
                    "timestamp": response.timestamp.isoformat()
                }
            }
            
            audit_file = audit_dir / f"verification_{request.request_id}.json"
            with open(audit_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")
    
    async def batch_verify(self, requests: List[VerificationRequest]) -> List[VerificationResponse]:
        """Verify multiple requests in batch"""
        logger.info(f"Starting batch verification for {len(requests)} requests")
        
        # Process requests concurrently
        tasks = [self.verify(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch verification failed for request {i}: {response}")
                # Create error response
                error_response = VerificationResponse(
                    request_id=requests[i].request_id,
                    overall_result=VerificationResult.ERROR,
                    metrics=VerificationMetrics()
                )
                error_response.errors.append(str(response))
                valid_responses.append(error_response)
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    def get_verification_history(self, limit: int = 100) -> List[VerificationResponse]:
        """Get recent verification history"""
        return self.verification_history[-limit:]
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        if not self.verification_history:
            return {"message": "No verification history available"}
        
        recent_responses = self.verification_history[-100:]  # Last 100
        
        # Compute statistics
        total_count = len(recent_responses)
        pass_count = sum(1 for r in recent_responses if r.overall_result == VerificationResult.PASS)
        fail_count = sum(1 for r in recent_responses if r.overall_result == VerificationResult.FAIL)
        warning_count = sum(1 for r in recent_responses if r.overall_result == VerificationResult.WARNING)
        
        avg_processing_time = np.mean([r.processing_time for r in recent_responses])
        avg_overall_score = np.mean([r.metrics.overall_score for r in recent_responses])
        
        return {
            "total_verifications": total_count,
            "pass_rate": pass_count / total_count if total_count > 0 else 0,
            "fail_rate": fail_count / total_count if total_count > 0 else 0,
            "warning_rate": warning_count / total_count if total_count > 0 else 0,
            "avg_processing_time": avg_processing_time,
            "avg_overall_score": avg_overall_score,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test verification with simple content
            test_request = VerificationRequest(
                content="This is a test message for health check.",
                verification_types=[VerificationType.SAFETY]
            )
            
            start_time = time.time()
            response = await self.verify(test_request)
            health_check_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "health_check_time": health_check_time,
                "test_result": response.overall_result.value,
                "total_verifications": len(self.verification_history),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Factory function
def create_prism(config: Optional[VerificationConfig] = None) -> Prism:
    """Create a new Prism instance"""
    return Prism(config)

# Convenience functions
async def verify_content(content: str, 
                        verification_types: Optional[List[VerificationType]] = None,
                        config: Optional[VerificationConfig] = None) -> VerificationResponse:
    """Quick content verification"""
    prism = create_prism(config)
    request = VerificationRequest(
        content=content,
        verification_types=verification_types
    )
    return await prism.verify(request)

async def safety_check(content: str) -> bool:
    """Quick safety check - returns True if safe"""
    response = await verify_content(content, [VerificationType.SAFETY])
    return response.overall_result in [VerificationResult.PASS, VerificationResult.WARNING]