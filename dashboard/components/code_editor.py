"""
Code editor component for the Streamlit dashboard.
"""

import streamlit as st
from typing import Optional
import streamlit.components.v1 as components

def show_code_editor(initial_code: str = "", height: int = 400, key: str = "code_editor") -> str:
    """
    Display a code editor with Solidity syntax highlighting.
    
    Args:
        initial_code: Initial code to display in the editor
        height: Height of the editor in pixels
        key: Unique key for the component
        
    Returns:
        The code entered in the editor
    """
    # Use Streamlit's text_area with custom styling for Solidity
    code = st.text_area(
        "Smart Contract Code",
        value=initial_code,
        height=height,
        key=key,
        help="Paste your Solidity smart contract code here for analysis"
    )
    
    return code


def show_monaco_editor(initial_code: str = "", height: int = 400, key: str = "monaco_editor") -> str:
    """
    Display Monaco editor with advanced Solidity syntax highlighting.
    
    Args:
        initial_code: Initial code to display in the editor
        height: Height of the editor in pixels
        key: Unique key for the component
        
    Returns:
        The code entered in the editor
    """
    # Monaco Editor HTML template with Solidity syntax highlighting
    monaco_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            }}
            #container {{
                width: 100%;
                height: {height}px;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        
        <script src="https://unpkg.com/monaco-editor@0.34.1/min/vs/loader.js"></script>
        <script>
            require.config({{ paths: {{ vs: 'https://unpkg.com/monaco-editor@0.34.1/min/vs' }} }});
            
            require(['vs/editor/editor.main'], function() {{
                // Register Solidity language
                monaco.languages.register({{ id: 'solidity' }});
                
                // Define Solidity syntax highlighting
                monaco.languages.setMonarchTokensProvider('solidity', {{
                    tokenizer: {{
                        root: [
                            [/pragma\s+solidity/, 'keyword'],
                            [/contract|library|interface/, 'keyword'],
                            [/function|modifier|constructor|fallback|receive/, 'keyword'],
                            [/public|private|internal|external/, 'keyword'],
                            [/view|pure|payable|constant/, 'keyword'],
                            [/returns?|return/, 'keyword'],
                            [/if|else|for|while|do|break|continue/, 'keyword'],
                            [/require|assert|revert/, 'keyword'],
                            [/uint\d*|int\d*|address|bool|string|bytes\d*/, 'type'],
                            [/mapping|struct|enum|event/, 'type'],
                            [/msg\.sender|msg\.value|block\.timestamp|block\.number/, 'variable.predefined'],
                            [/\/\/.*$/, 'comment'],
                            [/\/\*[\s\S]*?\*\//, 'comment'],
                            [/"([^"\\\\]|\\\\.)*$/, 'string.invalid'],
                            [/"/, 'string', '@string'],
                            [/\d+/, 'number'],
                            [/[a-zA-Z_]\w*/, 'identifier']
                        ],
                        string: [
                            [/[^\\\\"]+/, 'string'],
                            [/\\\\./, 'string.escape'],
                            [/"/, 'string', '@pop']
                        ]
                    }}
                }});
                
                // Create editor
                const editor = monaco.editor.create(document.getElementById('container'), {{
                    value: `{initial_code.replace('`', '\\`')}`,
                    language: 'solidity',
                    theme: 'vs-dark',
                    automaticLayout: true,
                    minimap: {{ enabled: false }},
                    scrollBeyondLastLine: false,
                    fontSize: 14,
                    lineNumbers: 'on',
                    roundedSelection: false,
                    scrollbar: {{
                        vertical: 'visible',
                        horizontal: 'visible'
                    }},
                    wordWrap: 'on'
                }});
                
                // Send content changes to Streamlit
                editor.onDidChangeModelContent(function() {{
                    const code = editor.getValue();
                    window.parent.postMessage({{
                        type: 'streamlit:componentValue',
                        value: code
                    }}, '*');
                }});
                
                // Initial value
                window.parent.postMessage({{
                    type: 'streamlit:componentValue',
                    value: editor.getValue()
                }}, '*');
            }});
        </script>
    </body>
    </html>
    """
    
    # Use Streamlit components to embed Monaco editor
    code = components.html(monaco_html, height=height + 50, key=key)
    
    # Fallback to text area if Monaco doesn't work
    if code is None:
        code = st.text_area(
            "Smart Contract Code (Monaco Editor not available)",
            value=initial_code,
            height=height,
            key=f"{key}_fallback"
        )
    
    return code or initial_code


def show_code_with_highlighting(code: str, vulnerabilities: list = None) -> None:
    """
    Display code with vulnerability highlighting.
    
    Args:
        code: The source code to display
        vulnerabilities: List of vulnerability findings with line numbers
    """
    if not code:
        st.info("No code to display")
        return
    
    lines = code.split('\n')
    
    # Create vulnerability line mapping
    vuln_lines = set()
    if vulnerabilities:
        for vuln in vulnerabilities:
            if 'line_number' in vuln:
                vuln_lines.add(vuln['line_number'])
    
    # Display code with line numbers and highlighting
    st.markdown("### Code Analysis")
    
    code_html = "<div style='background-color: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px;'>"
    
    for i, line in enumerate(lines, 1):
        line_style = ""
        if i in vuln_lines:
            line_style = "background-color: #ff4444; color: white; font-weight: bold;"
        
        # Escape HTML characters
        escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        code_html += f"<div style='{line_style}'><span style='color: #666; margin-right: 10px;'>{i:3d}</span>{escaped_line}</div>"
    
    code_html += "</div>"
    
    st.markdown(code_html, unsafe_allow_html=True)
    
    # Show vulnerability details
    if vulnerabilities and vuln_lines:
        st.markdown("### Vulnerability Details")
        for vuln in vulnerabilities:
            if 'line_number' in vuln and vuln['line_number'] in vuln_lines:
                with st.expander(f"Line {vuln['line_number']}: {vuln.get('vulnerability_type', 'Unknown')}"):
                    st.error(f"**Severity:** {vuln.get('severity', 'Unknown')}")
                    st.write(f"**Description:** {vuln.get('description', 'No description')}")
                    if 'recommendation' in vuln:
                        st.info(f"**Recommendation:** {vuln['recommendation']}")
                    if 'code_snippet' in vuln:
                        st.code(vuln['code_snippet'], language='solidity')


def show_sample_contracts() -> dict:
    """
    Display sample contracts for users to try.
    
    Returns:
        Dictionary of sample contract names and code
    """
    samples = {
        "Safe Contract": """pragma solidity ^0.8.0;

contract SafeContract {
    address public owner;
    mapping(address => uint256) public balances;
    bool private locked;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier noReentrancy() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) external noReentrancy {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Safe: Update state before external call
        balances[msg.sender] -= amount;
        
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
    }
}""",
        
        "Reentrancy Vulnerable": """pragma solidity ^0.8.0;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // VULNERABLE: External call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        // State update after external call - allows reentrancy
        balances[msg.sender] -= amount;
    }
}""",
        
        "Access Control Bug": """pragma solidity ^0.8.0;

contract AccessControlBug {
    address public owner;
    mapping(address => uint256) public balances;
    
    constructor() {
        owner = msg.sender;
    }
    
    // VULNERABLE: Missing access control
    function transferOwnership(address newOwner) external {
        owner = newOwner;
    }
    
    // VULNERABLE: Using tx.origin
    function withdraw() external {
        require(tx.origin == owner, "Not owner");
        payable(tx.origin).transfer(address(this).balance);
    }
    
    // VULNERABLE: Unprotected selfdestruct
    function destroy() external {
        selfdestruct(payable(msg.sender));
    }
}""",
        
        "Bad Randomness": """pragma solidity ^0.8.0;

contract BadRandomness {
    mapping(address => uint256) public balances;
    
    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }
    
    // VULNERABLE: Using block.timestamp for randomness
    function lottery() external {
        require(balances[msg.sender] >= 0.1 ether, "Insufficient balance");
        
        balances[msg.sender] -= 0.1 ether;
        
        // Predictable randomness
        uint256 random = uint256(keccak256(abi.encodePacked(
            block.timestamp,
            block.number,
            msg.sender
        ))) % 100;
        
        if (random < 10) { // 10% chance
            balances[msg.sender] += 1 ether;
        }
    }
}"""
    }
    
    st.markdown("### Sample Contracts")
    st.write("Choose a sample contract to analyze:")
    
    selected_sample = st.selectbox(
        "Select Sample Contract",
        options=list(samples.keys()),
        key="sample_selector"
    )
    
    if selected_sample:
        with st.expander(f"View {selected_sample} Code"):
            st.code(samples[selected_sample], language='solidity')
    
    return samples


def create_code_input_section() -> str:
    """
    Create a complete code input section with editor and samples.
    
    Returns:
        The code entered by the user
    """
    st.markdown("## Smart Contract Code Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Paste Code", "Use Sample Contract", "Advanced Editor"],
        horizontal=True
    )
    
    code = ""
    
    if input_method == "Paste Code":
        code = show_code_editor(
            initial_code="// Paste your Solidity contract code here\npragma solidity ^0.8.0;\n\ncontract YourContract {\n    // Your code here\n}",
            height=300,
            key="main_code_editor"
        )
    
    elif input_method == "Use Sample Contract":
        samples = show_sample_contracts()
        selected_sample = st.selectbox(
            "Select Sample Contract",
            options=list(samples.keys()),
            key="sample_contract_selector"
        )
        
        if selected_sample:
            code = samples[selected_sample]
            st.code(code, language='solidity')
    
    elif input_method == "Advanced Editor":
        st.info("Advanced Monaco editor with syntax highlighting")
        code = show_monaco_editor(
            initial_code="pragma solidity ^0.8.0;\n\ncontract YourContract {\n    // Your code here\n}",
            height=400,
            key="monaco_code_editor"
        )
    
    return code