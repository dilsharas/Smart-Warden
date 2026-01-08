"""
Solidity smart contract feature extraction for security analysis.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    include_basic_metrics: bool = True
    include_function_analysis: bool = True
    include_dangerous_patterns: bool = True
    include_control_flow: bool = True
    include_access_control: bool = True
    include_arithmetic: bool = True
    include_randomness: bool = True


class SolidityFeatureExtractor:
    """
    Extracts security-relevant features from Solidity smart contract code.
    
    Features are grouped into categories:
    - Basic metrics: Lines of code, complexity
    - Function analysis: Counts, visibility, modifiers
    - Dangerous patterns: External calls, delegatecall, selfdestruct
    - Control flow: Loops, conditionals, state changes
    - Access control: Modifiers, require/assert statements
    - Arithmetic: Operations indicating overflow risk
    - Randomness: Usage of predictable entropy sources
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration for feature extraction
        """
        self.config = config or FeatureConfig()
        
        # Define dangerous function patterns
        self.dangerous_functions = [
            r'\.call\s*\(',
            r'\.delegatecall\s*\(',
            r'\.staticcall\s*\(',
            r'selfdestruct\s*\(',
            r'suicide\s*\(',
            r'\.send\s*\(',
            r'\.transfer\s*\('
        ]
        
        # Randomness sources
        self.randomness_sources = [
            r'block\.timestamp',
            r'block\.number',
            r'block\.difficulty',
            r'block\.coinbase',
            r'blockhash\s*\(',
            r'now\b',
            r'tx\.origin'
        ]
        
        # Access control patterns
        self.access_patterns = [
            r'onlyOwner',
            r'require\s*\(',
            r'assert\s*\(',
            r'modifier\s+\w+',
            r'msg\.sender',
            r'tx\.origin'
        ]
        
        # Arithmetic operations
        self.arithmetic_ops = [
            r'\+\+',
            r'--',
            r'\+=',
            r'-=',
            r'\*=',
            r'/=',
            r'\*\*',
            r'SafeMath'
        ]
        self.dangerous_functions = [
            'selfdestruct', 'suicide', 'delegatecall', 'callcode',
            'call.value', 'send', 'transfer'
        ]
        
        # Access modifiers
        self.access_modifiers = [
            'public', 'private', 'internal', 'external',
            'view', 'pure', 'payable'
        ]
        
        # Vulnerability-specific keywords
        self.vulnerability_keywords = {
            'reentrancy': ['call', 'transfer', 'send', 'balance'],
            'overflow': ['+', '-', '*', '/', '**', '+=', '-=', '*=', '/='],
            'access_control': ['onlyOwner', 'require', 'modifier', 'msg.sender'],
            'randomness': ['block.timestamp', 'now', 'block.number', 'blockhash']
        }
        
    def extract_features(self, solidity_code: str) -> Dict[str, float]:
        """
        Extract all features from a single Solidity contract.
        
        Args:
            solidity_code: Raw Solidity source code as string
            
        Returns:
            Dictionary mapping feature names to numerical values
        """
        features = {}
        
        if self.config.include_basic_metrics:
            features.update(self._extract_basic_metrics(solidity_code))
        
        if self.config.include_function_analysis:
            features.update(self._extract_function_analysis(solidity_code))
        
        if self.config.include_dangerous_patterns:
            features.update(self._extract_dangerous_patterns(solidity_code))
        
        if self.config.include_control_flow:
            features.update(self._extract_control_flow(solidity_code))
        
        if self.config.include_access_control:
            features.update(self._extract_access_control(solidity_code))
        
        if self.config.include_arithmetic:
            features.update(self._extract_arithmetic_features(solidity_code))
        
        if self.config.include_randomness:
            features.update(self._extract_randomness_features(solidity_code))
        
        return features
    
    def _extract_basic_metrics(self, code: str) -> Dict[str, float]:
        """Extract basic code metrics."""
        features = {}
        
        # Line counts
        lines = code.split('\n')
        features['lines_of_code'] = len(lines)
        features['non_comment_lines'] = self._count_non_comment_lines(code)
        features['char_count'] = len(code)
        features['comment_ratio'] = self._calculate_comment_ratio(code)
        
        # Complexity metrics
        features['cyclomatic_complexity'] = self._calculate_cyclomatic_complexity(code)
        features['nesting_depth'] = self._calculate_max_nesting_depth(code)
        
        return features
    
    def _extract_function_analysis(self, code: str) -> Dict[str, float]:
        """Extract function-related features."""
        features = {}
        
        # Extract all functions
        functions = self._extract_functions(code)
        features['function_count'] = len(functions)
        
        # Count by visibility
        features['public_function_count'] = self._count_visibility(functions, 'public')
        features['private_function_count'] = self._count_visibility(functions, 'private')
        features['external_function_count'] = self._count_visibility(functions, 'external')
        features['internal_function_count'] = self._count_visibility(functions, 'internal')
        
        # Special function types
        features['payable_function_count'] = self._count_payable_functions(code)
        features['view_function_count'] = self._count_view_functions(code)
        features['pure_function_count'] = self._count_pure_functions(code)
        
        # Function ratios
        if features['function_count'] > 0:
            features['public_function_ratio'] = features['public_function_count'] / features['function_count']
            features['payable_function_ratio'] = features['payable_function_count'] / features['function_count']
        else:
            features['public_function_ratio'] = 0.0
            features['payable_function_ratio'] = 0.0
        
        return features
    
    def _extract_dangerous_patterns(self, code: str) -> Dict[str, float]:
        """Extract dangerous function usage patterns."""
        features = {}
        
        # Count dangerous functions
        for func in self.dangerous_functions:
            features[f'has_{func}'] = float(func in code)
        
        features['dangerous_function_count'] = sum(code.count(func) for func in self.dangerous_functions)
        
        # External call patterns
        features['external_call_count'] = self._count_external_calls(code)
        features['low_level_call_count'] = self._count_low_level_calls(code)
        features['call_in_loop'] = float(self._has_call_in_loop(code))
        features['state_change_after_call'] = float(self._detect_reentrancy_pattern(code))
        
        # Assembly usage
        features['has_assembly'] = float('assembly' in code)
        features['assembly_block_count'] = len(re.findall(r'assembly\s*\{', code))
        
        return features
    
    def _extract_control_flow(self, code: str) -> Dict[str, float]:
        """Extract control flow features."""
        features = {}
        
        # Loop counts
        features['for_loop_count'] = len(re.findall(r'\bfor\s*\(', code))
        features['while_loop_count'] = len(re.findall(r'\bwhile\s*\(', code))
        features['loop_count'] = features['for_loop_count'] + features['while_loop_count']
        
        # Conditional counts
        features['if_count'] = len(re.findall(r'\bif\s*\(', code))
        features['else_count'] = code.count('else')
        features['conditional_count'] = features['if_count']
        
        # Nested structures
        features['nested_loop_depth'] = self._calculate_max_loop_nesting(code)
        features['nested_if_depth'] = self._calculate_max_if_nesting(code)
        
        return features
    
    def _extract_access_control(self, code: str) -> Dict[str, float]:
        """Extract access control features."""
        features = {}
        
        # Modifier usage
        features['modifier_count'] = len(re.findall(r'modifier\s+\w+', code))
        features['require_count'] = code.count('require')
        features['assert_count'] = code.count('assert')
        features['revert_count'] = code.count('revert')
        
        # Owner patterns
        features['has_onlyOwner'] = float('onlyOwner' in code)
        features['msg_sender_checks'] = code.count('msg.sender')
        features['owner_variable'] = float(bool(re.search(r'\bowner\b', code, re.IGNORECASE)))
        
        # Constructor access
        features['has_constructor'] = float('constructor' in code)
        features['constructor_payable'] = float('constructor' in code and 'payable' in code)
        
        return features
    
    def _extract_arithmetic_features(self, code: str) -> Dict[str, float]:
        """Extract arithmetic operation features."""
        features = {}
        
        # Basic arithmetic operations
        features['addition_count'] = code.count('+') - code.count('++')  # Exclude increment
        features['subtraction_count'] = code.count('-') - code.count('--')  # Exclude decrement
        features['multiplication_count'] = code.count('*')
        features['division_count'] = code.count('/')
        features['modulo_count'] = code.count('%')
        features['exponentiation_count'] = code.count('**')
        
        # Compound assignments
        features['compound_assignment_count'] = (
            code.count('+=') + code.count('-=') + 
            code.count('*=') + code.count('/=') + code.count('%=')
        )
        
        # SafeMath usage
        features['has_safe_math'] = float('SafeMath' in code)
        features['safe_math_calls'] = len(re.findall(r'SafeMath\.\w+', code))
        
        # Solidity version (affects overflow protection)
        features['solidity_version'] = self._extract_version_number(code)
        features['is_safe_version'] = float(features['solidity_version'] >= 0.8)
        
        return features
    
    def _extract_randomness_features(self, code: str) -> Dict[str, float]:
        """Extract randomness-related features."""
        features = {}
        
        # Block-based randomness
        features['uses_block_timestamp'] = float('block.timestamp' in code or 'now' in code)
        features['uses_block_number'] = float('block.number' in code)
        features['uses_blockhash'] = float('blockhash' in code)
        features['uses_block_difficulty'] = float('block.difficulty' in code)
        
        # Hash functions for randomness
        features['keccak256_count'] = code.count('keccak256')
        features['sha256_count'] = code.count('sha256')
        features['ripemd160_count'] = code.count('ripemd160')
        
        # Random-like patterns
        features['random_keyword'] = float(bool(re.search(r'\brandom\b', code, re.IGNORECASE)))
        features['seed_keyword'] = float(bool(re.search(r'\bseed\b', code, re.IGNORECASE)))
        
        return features
    
    def _count_non_comment_lines(self, code: str) -> int:
        """Count lines excluding comments and blank lines."""
        lines = code.split('\n')
        non_comment = []
        in_multiline_comment = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle multiline comments
            if '/*' in line:
                in_multiline_comment = True
            if '*/' in line:
                in_multiline_comment = False
                continue
            
            if in_multiline_comment:
                continue
            
            # Skip single line comments
            if line.startswith('//'):
                continue
            
            non_comment.append(line)
        
        return len(non_comment)
    
    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculate ratio of comment lines to total lines."""
        total = len(code.split('\n'))
        non_comment = self._count_non_comment_lines(code)
        if total == 0:
            return 0.0
        return (total - non_comment) / total
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        # Count decision points
        complexity = 1  # Base complexity
        
        # Conditional statements
        complexity += len(re.findall(r'\bif\s*\(', code))
        complexity += len(re.findall(r'\belse\s+if\s*\(', code))
        complexity += len(re.findall(r'\bwhile\s*\(', code))
        complexity += len(re.findall(r'\bfor\s*\(', code))
        
        # Logical operators
        complexity += code.count('&&')
        complexity += code.count('||')
        
        # Exception handling
        complexity += code.count('require')
        complexity += code.count('assert')
        complexity += code.count('revert')
        
        return complexity
    
    def _calculate_max_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth of braces."""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _extract_functions(self, code: str) -> List[str]:
        """Extract all function definitions."""
        pattern = r'function\s+\w+\s*\([^)]*\)\s*[^{]*'
        return re.findall(pattern, code)
    
    def _count_visibility(self, functions: List[str], visibility: str) -> int:
        """Count functions with specific visibility modifier."""
        return sum(1 for func in functions if visibility in func)
    
    def _count_payable_functions(self, code: str) -> int:
        """Count payable functions."""
        return len(re.findall(r'function\s+\w+\s*\([^)]*\)[^{]*payable', code))
    
    def _count_view_functions(self, code: str) -> int:
        """Count view functions."""
        return len(re.findall(r'function\s+\w+\s*\([^)]*\)[^{]*view', code))
    
    def _count_pure_functions(self, code: str) -> int:
        """Count pure functions."""
        return len(re.findall(r'function\s+\w+\s*\([^)]*\)[^{]*pure', code))
    
    def _count_external_calls(self, code: str) -> int:
        """Count external contract calls."""
        patterns = [
            r'\.call\s*\(',
            r'\.delegatecall\s*\(',
            r'\.staticcall\s*\(',
            r'\.send\s*\(',
            r'\.transfer\s*\('
        ]
        return sum(len(re.findall(pattern, code)) for pattern in patterns)
    
    def _count_low_level_calls(self, code: str) -> int:
        """Count low-level calls."""
        patterns = [r'\.call\s*\(', r'\.delegatecall\s*\(', r'\.staticcall\s*\(']
        return sum(len(re.findall(pattern, code)) for pattern in patterns)
    
    def _has_call_in_loop(self, code: str) -> bool:
        """Detect if external calls occur inside loops."""
        # Find all loop blocks
        loop_pattern = r'(for|while)\s*\([^)]+\)\s*\{([^}]+)\}'
        loops = re.findall(loop_pattern, code, re.DOTALL)
        
        for loop_type, loop_body in loops:
            if any(call in loop_body for call in ['.call', '.send', '.transfer', '.delegatecall']):
                return True
        
        return False
    
    def _detect_reentrancy_pattern(self, code: str) -> int:
        """Detect potential reentrancy patterns."""
        lines = code.split('\n')
        risky_patterns = 0
        
        for i in range(len(lines) - 3):
            # Check if current line has external call
            if any(call in lines[i] for call in ['.call', '.send', '.transfer', '.delegatecall']):
                # Check next 3 lines for state changes
                for j in range(i+1, min(i+4, len(lines))):
                    if '=' in lines[j] and '==' not in lines[j] and '!=' not in lines[j]:
                        risky_patterns += 1
                        break
        
        return risky_patterns
    
    def _calculate_max_loop_nesting(self, code: str) -> int:
        """Calculate maximum loop nesting depth."""
        max_depth = 0
        current_depth = 0
        
        # Find all loop keywords and braces
        tokens = re.findall(r'\b(for|while)\b|\{|\}', code)
        
        in_loop = False
        for token in tokens:
            if token in ['for', 'while']:
                in_loop = True
            elif token == '{' and in_loop:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                in_loop = False
            elif token == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _calculate_max_if_nesting(self, code: str) -> int:
        """Calculate maximum if statement nesting depth."""
        max_depth = 0
        current_depth = 0
        
        # Find all if keywords and braces
        tokens = re.findall(r'\bif\b|\{|\}', code)
        
        in_if = False
        for token in tokens:
            if token == 'if':
                in_if = True
            elif token == '{' and in_if:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                in_if = False
            elif token == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _extract_version_number(self, code: str) -> float:
        """Extract Solidity version from pragma statement."""
        match = re.search(r'pragma\s+solidity\s+[\^>=<]*(\d+\.\d+)', code)
        if match:
            return float(match.group(1))
        return 0.0
    
    def extract_batch(self, contracts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from multiple contracts in a DataFrame.
        
        Args:
            contracts_df: DataFrame with columns ['filename', 'code', 'vulnerability', 'label']
            
        Returns:
            DataFrame with extracted features plus metadata columns
        """
        feature_list = []
        
        for idx, row in contracts_df.iterrows():
            try:
                features = self.extract_features(row['code'])
                features['filename'] = row['filename']
                features['vulnerability'] = row['vulnerability']
                features['label'] = row['label']
                feature_list.append(features)
            except Exception as e:
                logger.warning(f"Error extracting features from {row['filename']}: {e}")
                continue
        
        return pd.DataFrame(feature_list)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be extracted."""
        # Extract features from a dummy contract to get feature names
        dummy_code = """
        pragma solidity ^0.8.0;
        contract DummyContract {
            function test() public {}
        }
        """
        features = self.extract_features(dummy_code)
        return [name for name in features.keys() if name not in ['filename', 'vulnerability', 'label']]
    
    def validate_features(self, features: Dict) -> bool:
        """
        Validate extracted features for consistency.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            True if features are valid, False otherwise
        """
        # Check for required features
        required_features = ['lines_of_code', 'function_count', 'char_count']
        for feature in required_features:
            if feature not in features:
                logger.error(f"Missing required feature: {feature}")
                return False
        
        # Check for reasonable values
        if features['lines_of_code'] <= 0:
            logger.error("Invalid lines_of_code value")
            return False
        
        if features['char_count'] <= 0:
            logger.error("Invalid char_count value")
            return False
        
        # Check ratios are between 0 and 1
        ratio_features = [name for name in features.keys() if 'ratio' in name]
        for feature in ratio_features:
            if not 0 <= features[feature] <= 1:
                logger.error(f"Invalid ratio value for {feature}: {features[feature]}")
                return False
        
        return True


def main():
    """Example usage of SolidityFeatureExtractor."""
    # Sample Solidity contract
    sample_contract = """
    pragma solidity ^0.8.0;
    
    contract VulnerableContract {
        address public owner;
        mapping(address => uint256) public balances;
        
        constructor() {
            owner = msg.sender;
        }
        
        modifier onlyOwner() {
            require(msg.sender == owner, "Not owner");
            _;
        }
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            
            // Vulnerable to reentrancy
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            
            balances[msg.sender] -= amount;  // State change after external call
        }
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        
        function emergencyWithdraw() public onlyOwner {
            selfdestruct(payable(owner));
        }
    }
    """
    
    # Initialize feature extractor
    extractor = SolidityFeatureExtractor()
    
    # Extract features
    features = extractor.extract_features(sample_contract)
    
    print("Extracted Features:")
    print("-" * 50)
    for feature_name, value in sorted(features.items()):
        print(f"{feature_name:<30}: {value}")
    
    # Validate features
    is_valid = extractor.validate_features(features)
    print(f"\nFeatures valid: {is_valid}")
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"\nTotal features extracted: {len(feature_names)}")


if __name__ == "__main__":
    main() 
   
    def extract_features(self, contract_code: str) -> Dict[str, float]:
        """
        Extract all features from contract code.
        
        Args:
            contract_code: Solidity source code
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Clean and normalize code
        cleaned_code = self._clean_code(contract_code)
        
        if self.config.include_basic_metrics:
            features.update(self._extract_basic_metrics(cleaned_code))
        
        if self.config.include_function_analysis:
            features.update(self._extract_function_features(cleaned_code))
        
        if self.config.include_dangerous_patterns:
            features.update(self._extract_dangerous_patterns(cleaned_code))
        
        if self.config.include_control_flow:
            features.update(self._extract_control_flow(cleaned_code))
        
        if self.config.include_access_control:
            features.update(self._extract_access_control(cleaned_code))
        
        if self.config.include_arithmetic:
            features.update(self._extract_arithmetic_features(cleaned_code))
        
        if self.config.include_randomness:
            features.update(self._extract_randomness_features(cleaned_code))
        
        return features
    
    def _clean_code(self, code: str) -> str:
        """Clean and normalize contract code."""
        # Remove comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove string literals to avoid false positives
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        
        return code
    
    def _extract_basic_metrics(self, code: str) -> Dict[str, float]:
        """Extract basic code metrics."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'avg_line_length': np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0,
            'max_line_length': max([len(line) for line in non_empty_lines]) if non_empty_lines else 0,
            'contract_count': len(re.findall(r'\bcontract\s+\w+', code)),
            'interface_count': len(re.findall(r'\binterface\s+\w+', code)),
            'library_count': len(re.findall(r'\blibrary\s+\w+', code))
        }
    
    def _extract_function_features(self, code: str) -> Dict[str, float]:
        """Extract function-related features."""
        # Function patterns
        functions = re.findall(r'\bfunction\s+\w+', code)
        public_functions = re.findall(r'\bfunction\s+\w+[^{]*\bpublic\b', code)
        external_functions = re.findall(r'\bfunction\s+\w+[^{]*\bexternal\b', code)
        internal_functions = re.findall(r'\bfunction\s+\w+[^{]*\binternal\b', code)
        private_functions = re.findall(r'\bfunction\s+\w+[^{]*\bprivate\b', code)
        payable_functions = re.findall(r'\bfunction\s+\w+[^{]*\bpayable\b', code)
        view_functions = re.findall(r'\bfunction\s+\w+[^{]*\bview\b', code)
        pure_functions = re.findall(r'\bfunction\s+\w+[^{]*\bpure\b', code)
        
        # Modifiers
        modifiers = re.findall(r'\bmodifier\s+\w+', code)
        
        # Events
        events = re.findall(r'\bevent\s+\w+', code)
        
        return {
            'function_count': len(functions),
            'public_function_count': len(public_functions),
            'external_function_count': len(external_functions),
            'internal_function_count': len(internal_functions),
            'private_function_count': len(private_functions),
            'payable_function_count': len(payable_functions),
            'view_function_count': len(view_functions),
            'pure_function_count': len(pure_functions),
            'modifier_count': len(modifiers),
            'event_count': len(events),
            'public_external_ratio': (len(public_functions) + len(external_functions)) / max(len(functions), 1)
        }
    
    def _extract_dangerous_patterns(self, code: str) -> Dict[str, float]:
        """Extract dangerous pattern features."""
        features = {}
        
        # Count dangerous functions
        total_dangerous = 0
        for pattern in self.dangerous_functions:
            count = len(re.findall(pattern, code))
            feature_name = pattern.replace(r'\.', '').replace(r'\s*\(', '').replace('\\', '')
            features[f'{feature_name}_count'] = count
            total_dangerous += count
        
        features['total_dangerous_calls'] = total_dangerous
        
        # External calls
        external_calls = len(re.findall(r'\.call\s*\(', code))
        features['external_call_count'] = external_calls
        
        # Low-level calls
        low_level_calls = len(re.findall(r'\.(call|delegatecall|staticcall)\s*\(', code))
        features['low_level_call_count'] = low_level_calls
        
        # Assembly usage
        assembly_blocks = len(re.findall(r'\bassembly\s*\{', code))
        features['assembly_block_count'] = assembly_blocks
        
        return features
    
    def _extract_control_flow(self, code: str) -> Dict[str, float]:
        """Extract control flow features."""
        # Loops
        for_loops = len(re.findall(r'\bfor\s*\(', code))
        while_loops = len(re.findall(r'\bwhile\s*\(', code))
        
        # Conditionals
        if_statements = len(re.findall(r'\bif\s*\(', code))
        else_statements = len(re.findall(r'\belse\b', code))
        
        # State changes after external calls (reentrancy indicator)
        state_changes_after_calls = self._detect_state_changes_after_calls(code)
        
        return {
            'for_loop_count': for_loops,
            'while_loop_count': while_loops,
            'total_loop_count': for_loops + while_loops,
            'if_statement_count': if_statements,
            'else_statement_count': else_statements,
            'state_changes_after_calls': state_changes_after_calls,
            'cyclomatic_complexity': self._calculate_complexity(code)
        }
    
    def _extract_access_control(self, code: str) -> Dict[str, float]:
        """Extract access control features."""
        # Access control patterns
        require_count = len(re.findall(r'\brequire\s*\(', code))
        assert_count = len(re.findall(r'\bassert\s*\(', code))
        msg_sender_count = len(re.findall(r'\bmsg\.sender\b', code))
        tx_origin_count = len(re.findall(r'\btx\.origin\b', code))
        
        # Owner patterns
        owner_patterns = len(re.findall(r'\bowner\b', code, re.IGNORECASE))
        only_owner_patterns = len(re.findall(r'\bonlyOwner\b', code))
        
        return {
            'require_count': require_count,
            'assert_count': assert_count,
            'msg_sender_count': msg_sender_count,
            'tx_origin_count': tx_origin_count,
            'owner_pattern_count': owner_patterns,
            'only_owner_count': only_owner_patterns,
            'access_control_ratio': (require_count + assert_count) / max(len(re.findall(r'\bfunction\s+\w+', code)), 1)
        }
    
    def _extract_arithmetic_features(self, code: str) -> Dict[str, float]:
        """Extract arithmetic operation features."""
        # Arithmetic operations
        addition_ops = len(re.findall(r'\+(?!=)', code))
        subtraction_ops = len(re.findall(r'-(?!=)', code))
        multiplication_ops = len(re.findall(r'\*(?!=)', code))
        division_ops = len(re.findall(r'/(?!=)', code))
        
        # SafeMath usage
        safemath_usage = len(re.findall(r'\bSafeMath\b', code))
        
        # Overflow-prone patterns
        increment_ops = len(re.findall(r'\+\+', code))
        decrement_ops = len(re.findall(r'--', code))
        
        return {
            'addition_count': addition_ops,
            'subtraction_count': subtraction_ops,
            'multiplication_count': multiplication_ops,
            'division_count': division_ops,
            'safemath_usage': safemath_usage,
            'increment_ops': increment_ops,
            'decrement_ops': decrement_ops,
            'arithmetic_intensity': (addition_ops + subtraction_ops + multiplication_ops + division_ops) / max(len(code.split('\n')), 1)
        }
    
    def _extract_randomness_features(self, code: str) -> Dict[str, float]:
        """Extract randomness-related features."""
        features = {}
        
        total_randomness = 0
        for pattern in self.randomness_sources:
            count = len(re.findall(pattern, code))
            feature_name = pattern.replace(r'\.', '_').replace(r'\s*\(', '').replace('\\b', '').replace('\\', '')
            features[f'{feature_name}_usage'] = count
            total_randomness += count
        
        features['total_randomness_sources'] = total_randomness
        features['uses_block_timestamp'] = 1 if re.search(r'block\.timestamp|now\b', code) else 0
        
        return features
    
    def _detect_state_changes_after_calls(self, code: str) -> float:
        """Detect state changes after external calls (reentrancy pattern)."""
        # This is a simplified heuristic
        lines = code.split('\n')
        state_changes_after_calls = 0
        
        for i, line in enumerate(lines):
            if re.search(r'\.call\s*\(', line):
                # Look for state changes in next few lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if re.search(r'=(?!=)', lines[j]) and not re.search(r'==', lines[j]):
                        state_changes_after_calls += 1
                        break
        
        return state_changes_after_calls
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity."""
        # Count decision points
        decision_points = (
            len(re.findall(r'\bif\s*\(', code)) +
            len(re.findall(r'\bwhile\s*\(', code)) +
            len(re.findall(r'\bfor\s*\(', code)) +
            len(re.findall(r'\bcatch\b', code)) +
            len(re.findall(r'\b&&\b', code)) +
            len(re.findall(r'\|\|\b', code))
        )
        
        return decision_points + 1  # Base complexity is 1
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # This would return all possible feature names
        # For now, return a representative set
        return [
            'total_lines', 'code_lines', 'function_count', 'public_function_count',
            'external_function_count', 'payable_function_count', 'external_call_count',
            'low_level_call_count', 'require_count', 'assert_count', 'msg_sender_count',
            'uses_block_timestamp', 'state_changes_after_calls', 'cyclomatic_complexity',
            'arithmetic_intensity', 'safemath_usage', 'total_dangerous_calls'
        ]