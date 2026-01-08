"""
Abstract Syntax Tree parser for Solidity smart contracts.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """AST node types for Solidity constructs."""
    CONTRACT = "Contract"
    FUNCTION = "Function"
    MODIFIER = "Modifier"
    VARIABLE = "Variable"
    STATEMENT = "Statement"
    EXPRESSION = "Expression"
    CALL = "Call"
    LOOP = "Loop"
    CONDITIONAL = "Conditional"


@dataclass
class ASTNode:
    """Represents a node in the Abstract Syntax Tree."""
    node_type: NodeType
    name: str
    line_number: int
    children: List['ASTNode']
    attributes: Dict[str, Any]
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.attributes is None:
            self.attributes = {}


class SolidityASTParser:
    """
    Parses Solidity code to extract Abstract Syntax Tree information.
    
    Note: This is a simplified AST parser using regex patterns.
    For production use, consider using the Solidity compiler's AST output.
    """
    
    def __init__(self):
        """Initialize the AST parser."""
        self.current_line = 0
        self.nodes = []
        self.call_graph = {}
        self.control_flow_graph = {}
        
    def parse(self, code: str) -> ASTNode:
        """
        Parse Solidity code and return the root AST node.
        
        Args:
            code: Solidity source code
            
        Returns:
            Root AST node representing the contract
        """
        self.current_line = 0
        self.nodes = []
        
        # Clean and prepare code
        lines = code.split('\n')
        
        # Create root contract node
        contract_name = self._extract_contract_name(code)
        root = ASTNode(
            node_type=NodeType.CONTRACT,
            name=contract_name,
            line_number=0,
            children=[],
            attributes={'total_lines': len(lines)}
        )
        
        # Parse contract elements
        root.children.extend(self._parse_state_variables(code))
        root.children.extend(self._parse_functions(code))
        root.children.extend(self._parse_modifiers(code))
        
        # Build call graph
        self.call_graph = self._build_call_graph(root)
        
        # Build control flow graph
        self.control_flow_graph = self._build_control_flow_graph(root)
        
        return root
    
    def extract_control_flow_features(self, code: str) -> Dict[str, float]:
        """
        Extract control flow features from the AST.
        
        Args:
            code: Solidity source code
            
        Returns:
            Dictionary of control flow features
        """
        root = self.parse(code)
        features = {}
        
        # Function call analysis
        features['total_function_calls'] = self._count_function_calls(root)
        features['external_function_calls'] = self._count_external_calls(root)
        features['internal_function_calls'] = self._count_internal_calls(root)
        features['recursive_calls'] = self._count_recursive_calls(root)
        
        # Control flow complexity
        features['cyclomatic_complexity'] = self._calculate_cyclomatic_complexity(root)
        features['max_function_complexity'] = self._get_max_function_complexity(root)
        features['avg_function_complexity'] = self._get_avg_function_complexity(root)
        
        # Loop analysis
        features['nested_loop_depth'] = self._get_max_nested_loop_depth(root)
        features['loops_with_breaks'] = self._count_loops_with_breaks(root)
        features['infinite_loop_risk'] = self._detect_infinite_loop_risk(root)
        
        # Conditional analysis
        features['nested_conditional_depth'] = self._get_max_nested_conditional_depth(root)
        features['complex_conditionals'] = self._count_complex_conditionals(root)
        
        # Function dependency analysis
        features['function_dependency_depth'] = self._get_function_dependency_depth(root)
        features['circular_dependencies'] = self._detect_circular_dependencies(root)
        
        return features
    
    def extract_call_graph_features(self, code: str) -> Dict[str, float]:
        """
        Extract call graph features.
        
        Args:
            code: Solidity source code
            
        Returns:
            Dictionary of call graph features
        """
        root = self.parse(code)
        features = {}
        
        # Call graph metrics
        features['call_graph_nodes'] = len(self.call_graph)
        features['call_graph_edges'] = sum(len(calls) for calls in self.call_graph.values())
        features['max_fan_out'] = max(len(calls) for calls in self.call_graph.values()) if self.call_graph else 0
        features['max_fan_in'] = self._calculate_max_fan_in()
        
        # Function reachability
        features['unreachable_functions'] = self._count_unreachable_functions(root)
        features['dead_code_functions'] = self._count_dead_code_functions(root)
        
        # Call patterns
        features['callback_patterns'] = self._detect_callback_patterns(root)
        features['delegation_patterns'] = self._detect_delegation_patterns(root)
        
        return features
    
    def get_function_call_graph(self) -> Dict[str, List[str]]:
        """
        Get the function call graph.
        
        Returns:
            Dictionary mapping function names to lists of called functions
        """
        return self.call_graph.copy()
    
    def get_control_flow_graph(self) -> Dict[str, Dict]:
        """
        Get the control flow graph.
        
        Returns:
            Dictionary representing control flow structure
        """
        return self.control_flow_graph.copy()
    
    def _extract_contract_name(self, code: str) -> str:
        """Extract contract name from code."""
        match = re.search(r'contract\s+(\w+)', code)
        return match.group(1) if match else "UnknownContract"
    
    def _parse_state_variables(self, code: str) -> List[ASTNode]:
        """Parse state variable declarations."""
        variables = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            # Match state variable patterns
            var_match = re.match(r'(uint\d*|int\d*|address|bool|string|bytes\d*)\s+(public|private|internal)?\s*(\w+)', line)
            if var_match and not line.startswith('function') and not line.startswith('//'):
                var_type, visibility, name = var_match.groups()
                
                node = ASTNode(
                    node_type=NodeType.VARIABLE,
                    name=name,
                    line_number=i + 1,
                    children=[],
                    attributes={
                        'type': var_type,
                        'visibility': visibility or 'internal',
                        'is_state_variable': True
                    }
                )
                variables.append(node)
        
        return variables
    
    def _parse_functions(self, code: str) -> List[ASTNode]:
        """Parse function definitions."""
        functions = []
        
        # Find all function definitions
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*([^{]*)\s*\{([^}]*)\}'
        matches = re.finditer(function_pattern, code, re.DOTALL)
        
        for match in matches:
            name = match.group(1)
            modifiers = match.group(2).strip()
            body = match.group(3)
            
            # Calculate line number
            line_number = code[:match.start()].count('\n') + 1
            
            # Parse function attributes
            attributes = self._parse_function_attributes(modifiers)
            
            # Parse function body
            body_nodes = self._parse_function_body(body, line_number)
            
            function_node = ASTNode(
                node_type=NodeType.FUNCTION,
                name=name,
                line_number=line_number,
                children=body_nodes,
                attributes=attributes
            )
            
            functions.append(function_node)
        
        return functions
    
    def _parse_modifiers(self, code: str) -> List[ASTNode]:
        """Parse modifier definitions."""
        modifiers = []
        
        modifier_pattern = r'modifier\s+(\w+)\s*\([^)]*\)\s*\{([^}]*)\}'
        matches = re.finditer(modifier_pattern, code, re.DOTALL)
        
        for match in matches:
            name = match.group(1)
            body = match.group(2)
            line_number = code[:match.start()].count('\n') + 1
            
            body_nodes = self._parse_function_body(body, line_number)
            
            modifier_node = ASTNode(
                node_type=NodeType.MODIFIER,
                name=name,
                line_number=line_number,
                children=body_nodes,
                attributes={'is_modifier': True}
            )
            
            modifiers.append(modifier_node)
        
        return modifiers
    
    def _parse_function_attributes(self, modifiers_str: str) -> Dict[str, Any]:
        """Parse function modifiers and attributes."""
        attributes = {
            'visibility': 'internal',
            'state_mutability': 'nonpayable',
            'modifiers': [],
            'is_payable': False,
            'is_view': False,
            'is_pure': False
        }
        
        # Parse visibility
        for visibility in ['public', 'private', 'external', 'internal']:
            if visibility in modifiers_str:
                attributes['visibility'] = visibility
                break
        
        # Parse state mutability
        if 'payable' in modifiers_str:
            attributes['state_mutability'] = 'payable'
            attributes['is_payable'] = True
        elif 'view' in modifiers_str:
            attributes['state_mutability'] = 'view'
            attributes['is_view'] = True
        elif 'pure' in modifiers_str:
            attributes['state_mutability'] = 'pure'
            attributes['is_pure'] = True
        
        # Parse custom modifiers
        modifier_matches = re.findall(r'\b(?!public|private|external|internal|payable|view|pure)\w+(?=\s*(?:\(|$))', modifiers_str)
        attributes['modifiers'] = modifier_matches
        
        return attributes
    
    def _parse_function_body(self, body: str, start_line: int) -> List[ASTNode]:
        """Parse function body statements."""
        nodes = []
        lines = body.split('\n')
        current_line = start_line
        
        for line in lines:
            line = line.strip()
            current_line += 1
            
            if not line or line.startswith('//'):
                continue
            
            # Parse different statement types
            if self._is_loop_statement(line):
                node = self._parse_loop_statement(line, current_line)
                nodes.append(node)
            elif self._is_conditional_statement(line):
                node = self._parse_conditional_statement(line, current_line)
                nodes.append(node)
            elif self._is_function_call(line):
                node = self._parse_function_call(line, current_line)
                nodes.append(node)
            else:
                node = ASTNode(
                    node_type=NodeType.STATEMENT,
                    name="statement",
                    line_number=current_line,
                    children=[],
                    attributes={'code': line}
                )
                nodes.append(node)
        
        return nodes
    
    def _is_loop_statement(self, line: str) -> bool:
        """Check if line is a loop statement."""
        return bool(re.match(r'\s*(for|while)\s*\(', line))
    
    def _is_conditional_statement(self, line: str) -> bool:
        """Check if line is a conditional statement."""
        return bool(re.match(r'\s*if\s*\(', line))
    
    def _is_function_call(self, line: str) -> bool:
        """Check if line contains a function call."""
        return bool(re.search(r'\w+\s*\(', line))
    
    def _parse_loop_statement(self, line: str, line_number: int) -> ASTNode:
        """Parse loop statement."""
        loop_type = 'for' if line.strip().startswith('for') else 'while'
        
        return ASTNode(
            node_type=NodeType.LOOP,
            name=f"{loop_type}_loop",
            line_number=line_number,
            children=[],
            attributes={
                'loop_type': loop_type,
                'code': line
            }
        )
    
    def _parse_conditional_statement(self, line: str, line_number: int) -> ASTNode:
        """Parse conditional statement."""
        return ASTNode(
            node_type=NodeType.CONDITIONAL,
            name="if_statement",
            line_number=line_number,
            children=[],
            attributes={'code': line}
        )
    
    def _parse_function_call(self, line: str, line_number: int) -> ASTNode:
        """Parse function call."""
        # Extract function name from call
        call_match = re.search(r'(\w+)\s*\(', line)
        function_name = call_match.group(1) if call_match else "unknown"
        
        # Determine call type
        call_type = "internal"
        if '.' in line:
            call_type = "external"
        elif any(keyword in line for keyword in ['call', 'delegatecall', 'send', 'transfer']):
            call_type = "low_level"
        
        return ASTNode(
            node_type=NodeType.CALL,
            name=function_name,
            line_number=line_number,
            children=[],
            attributes={
                'call_type': call_type,
                'code': line
            }
        )
    
    def _build_call_graph(self, root: ASTNode) -> Dict[str, List[str]]:
        """Build function call graph."""
        call_graph = {}
        
        # Find all functions
        functions = [node for node in root.children if node.node_type == NodeType.FUNCTION]
        
        for function in functions:
            calls = []
            # Find all function calls in this function
            for child in function.children:
                if child.node_type == NodeType.CALL:
                    calls.append(child.name)
            
            call_graph[function.name] = calls
        
        return call_graph
    
    def _build_control_flow_graph(self, root: ASTNode) -> Dict[str, Dict]:
        """Build control flow graph."""
        cfg = {}
        
        functions = [node for node in root.children if node.node_type == NodeType.FUNCTION]
        
        for function in functions:
            cfg[function.name] = {
                'entry': True,
                'exit': True,
                'loops': len([child for child in function.children if child.node_type == NodeType.LOOP]),
                'conditionals': len([child for child in function.children if child.node_type == NodeType.CONDITIONAL]),
                'calls': len([child for child in function.children if child.node_type == NodeType.CALL])
            }
        
        return cfg
    
    def _count_function_calls(self, root: ASTNode) -> int:
        """Count total function calls."""
        count = 0
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                count += len([child for child in node.children if child.node_type == NodeType.CALL])
        return count
    
    def _count_external_calls(self, root: ASTNode) -> int:
        """Count external function calls."""
        count = 0
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                for child in node.children:
                    if (child.node_type == NodeType.CALL and 
                        child.attributes.get('call_type') in ['external', 'low_level']):
                        count += 1
        return count
    
    def _count_internal_calls(self, root: ASTNode) -> int:
        """Count internal function calls."""
        count = 0
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                for child in node.children:
                    if (child.node_type == NodeType.CALL and 
                        child.attributes.get('call_type') == 'internal'):
                        count += 1
        return count
    
    def _count_recursive_calls(self, root: ASTNode) -> int:
        """Count recursive function calls."""
        count = 0
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                function_name = node.name
                for child in node.children:
                    if (child.node_type == NodeType.CALL and 
                        child.name == function_name):
                        count += 1
        return count
    
    def _calculate_cyclomatic_complexity(self, root: ASTNode) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                # Add complexity for each decision point
                for child in node.children:
                    if child.node_type in [NodeType.CONDITIONAL, NodeType.LOOP]:
                        complexity += 1
        
        return complexity
    
    def _get_max_function_complexity(self, root: ASTNode) -> int:
        """Get maximum function complexity."""
        max_complexity = 0
        
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                complexity = 1  # Base complexity
                for child in node.children:
                    if child.node_type in [NodeType.CONDITIONAL, NodeType.LOOP]:
                        complexity += 1
                max_complexity = max(max_complexity, complexity)
        
        return max_complexity
    
    def _get_avg_function_complexity(self, root: ASTNode) -> float:
        """Get average function complexity."""
        functions = [node for node in root.children if node.node_type == NodeType.FUNCTION]
        if not functions:
            return 0.0
        
        total_complexity = 0
        for function in functions:
            complexity = 1
            for child in function.children:
                if child.node_type in [NodeType.CONDITIONAL, NodeType.LOOP]:
                    complexity += 1
            total_complexity += complexity
        
        return total_complexity / len(functions)
    
    def _get_max_nested_loop_depth(self, root: ASTNode) -> int:
        """Get maximum nested loop depth."""
        # This is a simplified implementation
        # In practice, would need to parse nested structures more carefully
        max_depth = 0
        
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                loop_count = len([child for child in node.children if child.node_type == NodeType.LOOP])
                max_depth = max(max_depth, loop_count)
        
        return max_depth
    
    def _count_loops_with_breaks(self, root: ASTNode) -> int:
        """Count loops with break statements."""
        # Simplified implementation
        return 0
    
    def _detect_infinite_loop_risk(self, root: ASTNode) -> float:
        """Detect potential infinite loop risks."""
        # Simplified implementation
        return 0.0
    
    def _get_max_nested_conditional_depth(self, root: ASTNode) -> int:
        """Get maximum nested conditional depth."""
        # Simplified implementation
        max_depth = 0
        
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                conditional_count = len([child for child in node.children if child.node_type == NodeType.CONDITIONAL])
                max_depth = max(max_depth, conditional_count)
        
        return max_depth
    
    def _count_complex_conditionals(self, root: ASTNode) -> int:
        """Count complex conditional statements."""
        # Simplified implementation
        return 0
    
    def _get_function_dependency_depth(self, root: ASTNode) -> int:
        """Get function dependency depth."""
        # Simplified implementation using call graph
        if not self.call_graph:
            return 0
        
        max_depth = 0
        for function, calls in self.call_graph.items():
            depth = len(calls)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _detect_circular_dependencies(self, root: ASTNode) -> float:
        """Detect circular dependencies in call graph."""
        # Simplified cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.call_graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for function in self.call_graph:
            if function not in visited:
                if has_cycle(function):
                    return 1.0
        
        return 0.0
    
    def _calculate_max_fan_in(self) -> int:
        """Calculate maximum fan-in (functions calling this function)."""
        fan_in = {}
        
        for caller, callees in self.call_graph.items():
            for callee in callees:
                fan_in[callee] = fan_in.get(callee, 0) + 1
        
        return max(fan_in.values()) if fan_in else 0
    
    def _count_unreachable_functions(self, root: ASTNode) -> int:
        """Count unreachable functions."""
        # Simplified implementation
        all_functions = {node.name for node in root.children if node.node_type == NodeType.FUNCTION}
        called_functions = set()
        
        for calls in self.call_graph.values():
            called_functions.update(calls)
        
        # Functions that are never called (except public/external ones)
        unreachable = 0
        for node in root.children:
            if (node.node_type == NodeType.FUNCTION and 
                node.name not in called_functions and
                node.attributes.get('visibility') not in ['public', 'external']):
                unreachable += 1
        
        return unreachable
    
    def _count_dead_code_functions(self, root: ASTNode) -> int:
        """Count functions with dead code."""
        # Simplified implementation
        return 0
    
    def _detect_callback_patterns(self, root: ASTNode) -> float:
        """Detect callback patterns."""
        # Simplified implementation
        return 0.0
    
    def _detect_delegation_patterns(self, root: ASTNode) -> float:
        """Detect delegation patterns."""
        # Check for delegatecall usage
        for node in root.children:
            if node.node_type == NodeType.FUNCTION:
                for child in node.children:
                    if (child.node_type == NodeType.CALL and 
                        'delegatecall' in child.attributes.get('code', '')):
                        return 1.0
        return 0.0


def main():
    """Example usage of SolidityASTParser."""
    sample_contract = """
    pragma solidity ^0.8.0;
    
    contract ExampleContract {
        uint256 public balance;
        address public owner;
        
        constructor() {
            owner = msg.sender;
        }
        
        modifier onlyOwner() {
            require(msg.sender == owner);
            _;
        }
        
        function deposit() public payable {
            balance += msg.value;
        }
        
        function withdraw(uint256 amount) public onlyOwner {
            require(balance >= amount);
            
            if (amount > 0) {
                for (uint i = 0; i < 10; i++) {
                    // Some loop logic
                    processPayment(amount / 10);
                }
            }
            
            balance -= amount;
        }
        
        function processPayment(uint256 amount) internal {
            // Internal function
            balance -= amount;
        }
    }
    """
    
    parser = SolidityASTParser()
    
    # Parse the contract
    ast_root = parser.parse(sample_contract)
    
    print(f"Contract: {ast_root.name}")
    print(f"Total lines: {ast_root.attributes['total_lines']}")
    print(f"Child nodes: {len(ast_root.children)}")
    
    # Extract control flow features
    cf_features = parser.extract_control_flow_features(sample_contract)
    print("\nControl Flow Features:")
    for feature, value in cf_features.items():
        print(f"  {feature}: {value}")
    
    # Extract call graph features
    cg_features = parser.extract_call_graph_features(sample_contract)
    print("\nCall Graph Features:")
    for feature, value in cg_features.items():
        print(f"  {feature}: {value}")
    
    # Show call graph
    call_graph = parser.get_function_call_graph()
    print("\nFunction Call Graph:")
    for function, calls in call_graph.items():
        print(f"  {function} -> {calls}")


if __name__ == "__main__":
    main()