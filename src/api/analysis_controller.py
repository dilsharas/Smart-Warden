"""
Analysis controller for orchestrating smart contract security analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback
import json
import hashlib
from dataclasses import dataclass, asdict

from ..features.feature_extractor import SolidityFeatureExtractor
from ..models.random_forest import RandomForestVulnerabilityDetector
from ..models.multiclass_classifier import MultiClassVulnerabilityClassifier
from ..integration.slither_runner import SlitherAnalyzer
from ..integration.mythril_runner import MythrilAnalyzer
from ..integration.tool_comparator import ToolComparator
from .utils import get_file_hash, create_analysis_id

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """Request for smart contract analysis."""
    contract_code: str
    filename: str
    analysis_options: Dict[str, bool]
    include_tool_comparison: bool = True
    generate_pdf_report: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class VulnerabilityFinding:
    """Represents a vulnerability finding."""
    vulnerability_type: str
    severity: str
    confidence: float
    line_number: int
    description: str
    recommendation: str
    code_snippet: str
    tool_source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    analysis_id: str
    contract_hash: str
    overall_risk_score: int
    is_vulnerable: bool
    confidence_level: float
    vulnerabilities: List[VulnerabilityFinding]
    feature_importance: Dict[str, float]
    tool_comparison: Dict[str, Any]
    analysis_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['vulnerabilities'] = [vuln.to_dict() for vuln in self.vulnerabilities]
        return result


class AnalysisController:
    """
    Orchestrates smart contract security analysis using multiple tools and AI models.
    
    Features:
    - Coordinates feature extraction, ML prediction, and tool integration
    - Manages analysis caching and result storage
    - Handles asynchronous analysis execution
    - Provides comprehensive vulnerability reporting
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 cache_enabled: bool = True,
                 max_workers: int = 4,
                 analysis_timeout: int = 300):
        """
        Initialize the analysis controller.
        
        Args:
            models_dir: Directory containing trained ML models
            cache_enabled: Whether to enable result caching
            analysis_timeout: Maximum analysis time in seconds
            max_workers: Maximum number of worker threads
        """
        self.models_dir = Path(models_dir)
        self.cache_enabled = cache_enabled
        self.analysis_timeout = analysis_timeout
        self.max_workers = max_workers
        
        # Initialize components
        self.feature_extractor = SolidityFeatureExtractor()
        self.tool_comparator = ToolComparator()
        
        # Load AI models
        self.ai_binary_model = None
        self.ai_multiclass_model = None
        self._load_models()
        
        # Analysis cache
        self.analysis_cache = {}
        
        # Thread pool for concurrent analysis
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("Analysis controller initialized")
    
    def _load_models(self):
        """Load trained AI models."""
        try:
            # Load binary classifier
            binary_model_path = self.models_dir / "vulnerability_detector.pkl"
            if binary_model_path.exists():
                self.ai_binary_model = RandomForestVulnerabilityDetector.load_model(str(binary_model_path))
                logger.info("Loaded binary AI model")
            else:
                logger.warning(f"Binary model not found at {binary_model_path}")
            
            # Load multi-class classifier
            multiclass_model_path = self.models_dir / "multiclass_detector.pkl"
            if multiclass_model_path.exists():
                self.ai_multiclass_model = MultiClassVulnerabilityClassifier.load_model(str(multiclass_model_path))
                logger.info("Loaded multi-class AI model")
            else:
                logger.warning(f"Multi-class model not found at {multiclass_model_path}")
            
            # Load models into tool comparator
            self.tool_comparator.load_ai_models(
                binary_model_path=str(binary_model_path) if binary_model_path.exists() else None,
                multiclass_model_path=str(multiclass_model_path) if multiclass_model_path.exists() else None
            )
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def analyze_contract(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive analysis of a smart contract.
        
        Args:
            request: Analysis request containing contract code and options
            
        Returns:
            AnalysisResult with comprehensive findings
        """
        start_time = time.time()
        analysis_id = create_analysis_id()
        contract_hash = get_file_hash(request.contract_code)
        
        logger.info(f"Starting analysis {analysis_id} for contract {request.filename}")
        
        # Check cache first
        if self.cache_enabled and contract_hash in self.analysis_cache:
            cached_result = self.analysis_cache[contract_hash]
            logger.info(f"Returning cached result for contract {contract_hash}")
            return cached_result
        
        try:
            # Run analysis with timeout
            result = await asyncio.wait_for(
                self._run_analysis(request, analysis_id, contract_hash),
                timeout=self.analysis_timeout
            )
            
            # Cache successful results
            if self.cache_enabled and result.success:
                self.analysis_cache[contract_hash] = result
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Analysis timed out after {self.analysis_timeout} seconds"
            logger.error(error_msg)
            
            return AnalysisResult(
                analysis_id=analysis_id,
                contract_hash=contract_hash,
                overall_risk_score=0,
                is_vulnerable=False,
                confidence_level=0.0,
                vulnerabilities=[],
                feature_importance={},
                tool_comparison={},
                analysis_time=time.time() - start_time,
                timestamp=datetime.now(),
                success=False,
                error_message=error_msg
            )
        
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"Analysis error: {e}")
            logger.debug(traceback.format_exc())
            
            return AnalysisResult(
                analysis_id=analysis_id,
                contract_hash=contract_hash,
                overall_risk_score=0,
                is_vulnerable=False,
                confidence_level=0.0,
                vulnerabilities=[],
                feature_importance={},
                tool_comparison={},
                analysis_time=time.time() - start_time,
                timestamp=datetime.now(),
                success=False,
                error_message=error_msg
            )
    
    async def _run_analysis(self, request: AnalysisRequest, analysis_id: str, contract_hash: str) -> AnalysisResult:
        """
        Execute the actual analysis workflow.
        
        Args:
            request: Analysis request
            analysis_id: Unique analysis identifier
            contract_hash: Hash of the contract code
            
        Returns:
            AnalysisResult
        """
        start_time = time.time()
        
        # Step 1: Extract features
        logger.info(f"Extracting features for analysis {analysis_id}")
        features = await self._extract_features(request.contract_code)
        
        # Step 2: Run AI models
        logger.info(f"Running AI models for analysis {analysis_id}")
        ai_results = await self._run_ai_models(features, request.contract_code)
        
        # Step 3: Run external tools (if requested)
        tool_results = {}
        if request.include_tool_comparison:
            logger.info(f"Running tool comparison for analysis {analysis_id}")
            tool_results = await self._run_tool_comparison(request.contract_code, request.filename)
        
        # Step 4: Aggregate results
        logger.info(f"Aggregating results for analysis {analysis_id}")
        aggregated_result = self._aggregate_results(
            ai_results, tool_results, features, request.contract_code
        )
        
        execution_time = time.time() - start_time
        
        return AnalysisResult(
            analysis_id=analysis_id,
            contract_hash=contract_hash,
            overall_risk_score=aggregated_result['risk_score'],
            is_vulnerable=aggregated_result['is_vulnerable'],
            confidence_level=aggregated_result['confidence'],
            vulnerabilities=aggregated_result['vulnerabilities'],
            feature_importance=aggregated_result['feature_importance'],
            tool_comparison=tool_results,
            analysis_time=execution_time,
            timestamp=datetime.now(),
            success=True
        )
    
    async def _extract_features(self, contract_code: str) -> Dict[str, float]:
        """
        Extract features from contract code.
        
        Args:
            contract_code: Solidity source code
            
        Returns:
            Dictionary of extracted features
        """
        loop = asyncio.get_event_loop()
        
        # Run feature extraction in thread pool
        features = await loop.run_in_executor(
            self.executor,
            self.feature_extractor.extract_features,
            contract_code
        )
        
        return features
    
    async def _run_ai_models(self, features: Dict[str, float], contract_code: str) -> Dict[str, Any]:
        """
        Run AI models on extracted features.
        
        Args:
            features: Extracted features
            contract_code: Original contract code
            
        Returns:
            Dictionary with AI model results
        """
        results = {
            'binary_prediction': None,
            'multiclass_prediction': None,
            'feature_importance': {},
            'confidence_scores': {}
        }
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Run binary classifier
        if self.ai_binary_model:
            try:
                predictions, probabilities = self.ai_binary_model.predict(features_df)
                confidence = np.max(probabilities[0])
                
                results['binary_prediction'] = {
                    'prediction': predictions[0],
                    'confidence': confidence,
                    'probabilities': probabilities[0].tolist()
                }
                
                # Get feature importance
                importance_df = self.ai_binary_model.get_feature_importance(top_n=10)
                results['feature_importance'] = dict(zip(importance_df['feature'], importance_df['importance']))
                
            except Exception as e:
                logger.error(f"Binary model prediction failed: {e}")
        
        # Run multi-class classifier
        if self.ai_multiclass_model:
            try:
                predictions, probabilities = self.ai_multiclass_model.predict(features_df)
                
                # Get top-k predictions
                top_k = self.ai_multiclass_model.predict_top_k(features_df, k=3)[0]
                
                results['multiclass_prediction'] = {
                    'prediction': predictions[0],
                    'top_predictions': top_k,
                    'probabilities': probabilities[0].tolist(),
                    'class_names': self.ai_multiclass_model.label_encoder.classes_.tolist()
                }
                
            except Exception as e:
                logger.error(f"Multi-class model prediction failed: {e}")
        
        return results
    
    async def _run_tool_comparison(self, contract_code: str, filename: str) -> Dict[str, Any]:
        """
        Run external tool comparison.
        
        Args:
            contract_code: Solidity source code
            filename: Contract filename
            
        Returns:
            Dictionary with tool comparison results
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Run tool comparison in thread pool
            comparison_result = await loop.run_in_executor(
                self.executor,
                self.tool_comparator.compare_tools,
                contract_code,
                filename
            )
            
            return comparison_result.to_dict()
            
        except Exception as e:
            logger.error(f"Tool comparison failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tool_performances': {},
                'consensus_findings': [],
                'agreement_score': 0.0
            }
    
    def _aggregate_results(self, 
                          ai_results: Dict[str, Any],
                          tool_results: Dict[str, Any],
                          features: Dict[str, float],
                          contract_code: str) -> Dict[str, Any]:
        """
        Aggregate results from all analysis components.
        
        Args:
            ai_results: Results from AI models
            tool_results: Results from external tools
            features: Extracted features
            contract_code: Original contract code
            
        Returns:
            Aggregated analysis results
        """
        vulnerabilities = []
        overall_confidence = 0.0
        is_vulnerable = False
        risk_score = 0
        
        # Process AI model results
        if ai_results.get('binary_prediction'):
            binary_pred = ai_results['binary_prediction']
            
            # Check if contract is predicted as vulnerable
            if (isinstance(binary_pred['prediction'], str) and binary_pred['prediction'] == 'vulnerable') or \
               (isinstance(binary_pred['prediction'], (int, float)) and binary_pred['prediction'] > 0.5):
                is_vulnerable = True
                overall_confidence = binary_pred['confidence']
                risk_score = int(binary_pred['confidence'] * 100)
        
        # Process multi-class predictions
        if ai_results.get('multiclass_prediction'):
            multiclass_pred = ai_results['multiclass_prediction']
            
            if multiclass_pred['prediction'] != 'safe':
                is_vulnerable = True
                
                # Create vulnerability finding from AI prediction
                ai_vulnerability = VulnerabilityFinding(
                    vulnerability_type=multiclass_pred['prediction'],
                    severity=self._map_confidence_to_severity(multiclass_pred['top_predictions'][0][1]),
                    confidence=multiclass_pred['top_predictions'][0][1],
                    line_number=1,  # AI models don't provide line numbers
                    description=f"AI model detected {multiclass_pred['prediction']} vulnerability pattern",
                    recommendation=self._get_vulnerability_recommendation(multiclass_pred['prediction']),
                    code_snippet="",
                    tool_source="AI Multi-class Classifier"
                )
                vulnerabilities.append(ai_vulnerability)
        
        # Process external tool results
        if tool_results.get('tool_performances'):
            for tool_name, performance in tool_results['tool_performances'].items():
                if performance.get('success') and performance.get('vulnerabilities_found'):
                    for vuln_type in performance['vulnerabilities_found']:
                        tool_vulnerability = VulnerabilityFinding(
                            vulnerability_type=vuln_type,
                            severity="Medium",  # Default severity for external tools
                            confidence=0.8,  # Default confidence for external tools
                            line_number=1,
                            description=f"{tool_name} detected {vuln_type} vulnerability",
                            recommendation=self._get_vulnerability_recommendation(vuln_type),
                            code_snippet="",
                            tool_source=tool_name
                        )
                        vulnerabilities.append(tool_vulnerability)
        
        # Calculate overall risk score based on findings
        if vulnerabilities:
            severity_weights = {'Critical': 100, 'High': 80, 'Medium': 60, 'Low': 40}
            total_weight = sum(severity_weights.get(vuln.severity, 50) for vuln in vulnerabilities)
            risk_score = min(100, total_weight // len(vulnerabilities))
            is_vulnerable = True
        
        # Get feature importance
        feature_importance = ai_results.get('feature_importance', {})
        
        return {
            'vulnerabilities': vulnerabilities,
            'is_vulnerable': is_vulnerable,
            'confidence': overall_confidence,
            'risk_score': risk_score,
            'feature_importance': feature_importance
        }
    
    def _map_confidence_to_severity(self, confidence: float) -> str:
        """
        Map confidence score to severity level.
        
        Args:
            confidence: Confidence score (0.0 - 1.0)
            
        Returns:
            Severity level string
        """
        if confidence >= 0.9:
            return "Critical"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _get_vulnerability_recommendation(self, vulnerability_type: str) -> str:
        """
        Get recommendation for a specific vulnerability type.
        
        Args:
            vulnerability_type: Type of vulnerability
            
        Returns:
            Recommendation string
        """
        recommendations = {
            'reentrancy': 'Use the checks-effects-interactions pattern. Move external calls to the end of functions after state changes. Consider using reentrancy guards.',
            'access_control': 'Implement proper access control mechanisms. Use modifiers like onlyOwner and validate msg.sender. Avoid using tx.origin for authorization.',
            'arithmetic': 'Use SafeMath library for arithmetic operations (Solidity < 0.8.0) or upgrade to Solidity 0.8+ which has built-in overflow protection.',
            'bad_randomness': 'Do not use block.timestamp, block.number, or blockhash for randomness. Use a secure random number generator or oracle service.',
            'unchecked_calls': 'Always check the return value of external calls. Use require() statements or handle failures appropriately.',
            'denial_of_service': 'Avoid unbounded loops and expensive operations. Implement gas limits and consider using pull-over-push patterns.',
            'vulnerable': 'Review the contract for potential security issues. Consider getting a professional security audit.'
        }
        
        return recommendations.get(vulnerability_type, 'Review the code carefully and follow Solidity security best practices.')
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get the status of an ongoing analysis.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Status information
        """
        # This would typically check a database or cache for analysis status
        # For now, return a simple status
        return {
            'analysis_id': analysis_id,
            'status': 'completed',  # In a real implementation, this would track actual status
            'progress': 100,
            'message': 'Analysis completed successfully'
        }
    
    def get_cached_result(self, contract_hash: str) -> Optional[AnalysisResult]:
        """
        Get cached analysis result.
        
        Args:
            contract_hash: Hash of the contract code
            
        Returns:
            Cached result or None
        """
        return self.analysis_cache.get(contract_hash)
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            'cache_size': len(self.analysis_cache),
            'cache_enabled': self.cache_enabled
        }
    
    def shutdown(self):
        """Shutdown the analysis controller and cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Analysis controller shutdown complete")