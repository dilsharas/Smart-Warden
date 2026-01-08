"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class AnalysisOptions(BaseModel):
    """Analysis options configuration."""
    include_ai_analysis: bool = Field(True, description="Include AI model analysis")
    include_slither: bool = Field(True, description="Include Slither static analysis")
    include_mythril: bool = Field(True, description="Include Mythril symbolic execution")
    include_feature_importance: bool = Field(True, description="Include feature importance analysis")
    detailed_report: bool = Field(True, description="Generate detailed vulnerability report")


class ContractAnalysisRequest(BaseModel):
    """Request model for contract analysis."""
    contract_code: str = Field(..., min_length=10, max_length=1000000, description="Solidity contract source code")
    filename: str = Field("contract.sol", max_length=255, description="Contract filename")
    analysis_options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="Analysis configuration")
    include_tool_comparison: bool = Field(True, description="Include comparison with external tools")
    generate_pdf_report: bool = Field(False, description="Generate PDF report")
    
    @validator('contract_code')
    def validate_contract_code(cls, v):
        """Validate contract code."""
        if not v.strip():
            raise ValueError("Contract code cannot be empty")
        
        # Check for basic Solidity structure
        if 'pragma solidity' not in v.lower():
            raise ValueError("Contract must contain pragma solidity statement")
        
        if 'contract' not in v.lower():
            raise ValueError("Contract must contain contract definition")
        
        return v.strip()
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        if not v.endswith('.sol'):
            v = v + '.sol'
        
        # Remove any path components for security
        v = v.split('/')[-1].split('\\')[-1]
        
        # Sanitize filename
        import re
        v = re.sub(r'[^\w\-_\.]', '_', v)
        
        return v


class SeverityLevel(str, Enum):
    """Vulnerability severity levels."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class VulnerabilityType(str, Enum):
    """Supported vulnerability types."""
    REENTRANCY = "reentrancy"
    ACCESS_CONTROL = "access_control"
    ARITHMETIC = "arithmetic"
    UNCHECKED_CALLS = "unchecked_calls"
    DENIAL_OF_SERVICE = "denial_of_service"
    BAD_RANDOMNESS = "bad_randomness"
    OTHER = "other"


class VulnerabilityFinding(BaseModel):
    """Vulnerability finding model."""
    vulnerability_type: VulnerabilityType = Field(..., description="Type of vulnerability")
    severity: SeverityLevel = Field(..., description="Severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    line_number: int = Field(..., ge=1, description="Line number where vulnerability was found")
    description: str = Field(..., min_length=1, max_length=1000, description="Vulnerability description")
    recommendation: str = Field(..., min_length=1, max_length=2000, description="Remediation recommendation")
    code_snippet: str = Field("", max_length=500, description="Relevant code snippet")
    tool_source: str = Field(..., min_length=1, max_length=100, description="Tool that detected the vulnerability")


class ToolPerformance(BaseModel):
    """Tool performance metrics."""
    tool_name: str = Field(..., description="Name of the analysis tool")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    findings_count: int = Field(..., ge=0, description="Number of findings")
    vulnerabilities_found: List[str] = Field(default_factory=list, description="List of vulnerability types found")
    severity_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of findings by severity")
    success: bool = Field(..., description="Whether the tool executed successfully")
    error_message: Optional[str] = Field(None, description="Error message if tool failed")


class ToolComparison(BaseModel):
    """Tool comparison results."""
    tool_performances: Dict[str, ToolPerformance] = Field(..., description="Performance of each tool")
    consensus_findings: List[str] = Field(default_factory=list, description="Vulnerabilities found by multiple tools")
    unique_findings: Dict[str, List[str]] = Field(default_factory=dict, description="Unique findings per tool")
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="Agreement score between tools")


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    contract_hash: str = Field(..., description="Hash of the analyzed contract")
    overall_risk_score: int = Field(..., ge=0, le=100, description="Overall risk score (0-100)")
    is_vulnerable: bool = Field(..., description="Whether vulnerabilities were detected")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Overall confidence level")
    vulnerabilities: List[VulnerabilityFinding] = Field(default_factory=list, description="List of vulnerabilities found")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    tool_comparison: Optional[ToolComparison] = Field(None, description="Tool comparison results")
    analysis_time: float = Field(..., ge=0.0, description="Total analysis time in seconds")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    success: bool = Field(..., description="Whether analysis completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")


class AnalysisStatus(BaseModel):
    """Analysis status response."""
    analysis_id: str = Field(..., description="Analysis identifier")
    status: str = Field(..., description="Current status (pending, running, completed, failed)")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    available_tools: Dict[str, bool] = Field(..., description="Availability of analysis tools")
    model_status: Dict[str, bool] = Field(..., description="Status of ML models")
    uptime: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """ML model information."""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    is_loaded: bool = Field(..., description="Whether model is loaded")
    training_date: Optional[datetime] = Field(None, description="Model training date")
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model accuracy")
    feature_count: Optional[int] = Field(None, ge=0, description="Number of features")
    class_count: Optional[int] = Field(None, ge=0, description="Number of classes")


class ModelsInfoResponse(BaseModel):
    """Models information response."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_models: int = Field(..., description="Total number of models")
    loaded_models: int = Field(..., description="Number of loaded models")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ValidationError(BaseModel):
    """Validation error details."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="Invalid value provided")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = Field("validation_error", description="Error type")
    message: str = Field(..., description="General validation error message")
    validation_errors: List[ValidationError] = Field(..., description="List of validation errors")
    timestamp: datetime = Field(..., description="Error timestamp")


class BenchmarkRequest(BaseModel):
    """Request for benchmarking tools."""
    test_contracts: List[Dict[str, Any]] = Field(..., min_items=1, description="List of test contracts with ground truth")
    tools: Optional[List[str]] = Field(None, description="Tools to benchmark (default: all available)")
    
    @validator('test_contracts')
    def validate_test_contracts(cls, v):
        """Validate test contracts."""
        for i, contract in enumerate(v):
            if 'code' not in contract:
                raise ValueError(f"Contract {i} missing 'code' field")
            if 'name' not in contract:
                raise ValueError(f"Contract {i} missing 'name' field")
            if not contract['code'].strip():
                raise ValueError(f"Contract {i} has empty code")
        return v


class BenchmarkMetrics(BaseModel):
    """Benchmark metrics for a tool."""
    tool_name: str = Field(..., description="Tool name")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy score")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    false_positive_rate: float = Field(..., ge=0.0, le=1.0, description="False positive rate")
    false_negative_rate: float = Field(..., ge=0.0, le=1.0, description="False negative rate")
    avg_execution_time: float = Field(..., ge=0.0, description="Average execution time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    vulnerability_detection_rates: Dict[str, float] = Field(default_factory=dict, description="Detection rates per vulnerability type")


class BenchmarkResponse(BaseModel):
    """Benchmark results response."""
    benchmark_id: str = Field(..., description="Benchmark identifier")
    metrics: Dict[str, BenchmarkMetrics] = Field(..., description="Metrics for each tool")
    total_contracts: int = Field(..., description="Total number of test contracts")
    timestamp: datetime = Field(..., description="Benchmark timestamp")
    execution_time: float = Field(..., description="Total benchmark execution time")


class CacheStats(BaseModel):
    """Cache statistics."""
    cache_size: int = Field(..., ge=0, description="Number of cached results")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    hit_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cache hit rate")
    memory_usage: Optional[int] = Field(None, ge=0, description="Cache memory usage in bytes")


class SystemStats(BaseModel):
    """System statistics."""
    total_analyses: int = Field(..., ge=0, description="Total number of analyses performed")
    successful_analyses: int = Field(..., ge=0, description="Number of successful analyses")
    failed_analyses: int = Field(..., ge=0, description="Number of failed analyses")
    avg_analysis_time: float = Field(..., ge=0.0, description="Average analysis time")
    cache_stats: CacheStats = Field(..., description="Cache statistics")
    uptime: float = Field(..., ge=0.0, description="System uptime in seconds")


# Request validation utilities
def validate_contract_size(contract_code: str, max_size: int = 1000000) -> bool:
    """
    Validate contract code size.
    
    Args:
        contract_code: Contract source code
        max_size: Maximum allowed size in characters
        
    Returns:
        True if valid, False otherwise
    """
    return len(contract_code) <= max_size


def validate_solidity_syntax(contract_code: str) -> tuple[bool, Optional[str]]:
    """
    Basic Solidity syntax validation.
    
    Args:
        contract_code: Contract source code
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import re
    
    # Check for pragma statement
    if not re.search(r'pragma\s+solidity', contract_code, re.IGNORECASE):
        return False, "Missing pragma solidity statement"
    
    # Check for contract definition
    if not re.search(r'\bcontract\s+\w+', contract_code, re.IGNORECASE):
        return False, "Missing contract definition"
    
    # Check for balanced braces
    open_braces = contract_code.count('{')
    close_braces = contract_code.count('}')
    if open_braces != close_braces:
        return False, f"Unbalanced braces: {open_braces} opening, {close_braces} closing"
    
    # Check for balanced parentheses
    open_parens = contract_code.count('(')
    close_parens = contract_code.count(')')
    if open_parens != close_parens:
        return False, f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for security.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Replace invalid characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Ensure .sol extension
    if not filename.endswith('.sol'):
        filename = filename + '.sol'
    
    return filename


def create_error_response(error_type: str, message: str, details: Optional[str] = None) -> ErrorResponse:
    """
    Create standardized error response.
    
    Args:
        error_type: Type of error
        message: Error message
        details: Additional details
        
    Returns:
        ErrorResponse object
    """
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details,
        timestamp=datetime.now()
    )


def create_validation_error_response(validation_errors: List[Dict[str, Any]]) -> ValidationErrorResponse:
    """
    Create validation error response from Pydantic errors.
    
    Args:
        validation_errors: List of validation errors from Pydantic
        
    Returns:
        ValidationErrorResponse object
    """
    errors = []
    for error in validation_errors:
        field = '.'.join(str(loc) for loc in error['loc'])
        errors.append(ValidationError(
            field=field,
            message=error['msg'],
            invalid_value=error.get('input')
        ))
    
    return ValidationErrorResponse(
        message="Request validation failed",
        validation_errors=errors,
        timestamp=datetime.now()
    )


# Response formatting utilities
def format_analysis_response(result: Any) -> AnalysisResult:
    """
    Format analysis result for API response.
    
    Args:
        result: Raw analysis result
        
    Returns:
        Formatted AnalysisResult
    """
    # Convert internal result format to API response format
    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    else:
        result_dict = result
    
    # Ensure all required fields are present
    formatted_result = AnalysisResult(
        analysis_id=result_dict.get('analysis_id', ''),
        contract_hash=result_dict.get('contract_hash', ''),
        overall_risk_score=result_dict.get('overall_risk_score', 0),
        is_vulnerable=result_dict.get('is_vulnerable', False),
        confidence_level=result_dict.get('confidence_level', 0.0),
        vulnerabilities=[
            VulnerabilityFinding(**vuln) if isinstance(vuln, dict) else vuln
            for vuln in result_dict.get('vulnerabilities', [])
        ],
        feature_importance=result_dict.get('feature_importance', {}),
        tool_comparison=result_dict.get('tool_comparison'),
        analysis_time=result_dict.get('analysis_time', 0.0),
        timestamp=result_dict.get('timestamp', datetime.now()),
        success=result_dict.get('success', True),
        error_message=result_dict.get('error_message')
    )
    
    return formatted_result