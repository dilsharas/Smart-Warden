"""
File upload widget component for the Streamlit dashboard.
"""

import streamlit as st
from typing import Optional, Tuple

def show_upload_widget() -> Tuple[Optional[str], Optional[str]]:
    """
    Display file upload widget with validation.
    
    Returns:
        Tuple of (contract_code, filename) or (None, None) if no file uploaded
    """
    uploaded_file = st.file_uploader(
        "Choose a Solidity file",
        type=['sol'],
        help="Upload a .sol file containing your smart contract code"
    )
    
    if uploaded_file is not None:
        try:
            # Read file content
            contract_code = uploaded_file.read().decode('utf-8')
            filename = uploaded_file.name
            
            # Basic validation
            if len(contract_code.strip()) == 0:
                st.error("The uploaded file is empty.")
                return None, None
            
            # Check file size (limit to 1MB)
            if len(contract_code) > 1000000:
                st.error("File is too large. Please upload files smaller than 1MB.")
                return None, None
            
            # Basic Solidity validation
            if 'pragma solidity' not in contract_code.lower():
                st.warning("⚠️ No 'pragma solidity' statement found. This may not be a valid Solidity file.")
            
            if 'contract ' not in contract_code.lower():
                st.warning("⚠️ No 'contract' definition found. This may not be a complete Solidity contract.")
            
            # Show file info
            lines = len(contract_code.split('\n'))
            chars = len(contract_code)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{chars:,} chars")
            with col2:
                st.metric("Lines", f"{lines:,}")
            with col3:
                st.metric("Functions", contract_code.count('function '))
            
            st.success(f"✅ File '{filename}' uploaded successfully!")
            
            return contract_code, filename
            
        except UnicodeDecodeError:
            st.error("❌ Unable to decode file. Please ensure it's a valid text file.")
            return None, None
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None, None
    
    return None, None


def show_drag_drop_area():
    """
    Display a drag-and-drop area for file uploads.
    Note: This is a visual placeholder as Streamlit doesn't support true drag-and-drop.
    """
    st.markdown("""
    <div style="
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
        margin: 1rem 0;
    ">
        <h4 style="color: #666666; margin-bottom: 1rem;">📁 Drag & Drop Solidity Files</h4>
        <p style="color: #888888; margin-bottom: 0;">
            Use the file uploader above to select your .sol files
        </p>
    </div>
    """, unsafe_allow_html=True)


def validate_solidity_file(contract_code: str) -> Tuple[bool, list]:
    """
    Validate Solidity contract code.
    
    Args:
        contract_code: The contract source code
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for pragma statement
    if 'pragma solidity' not in contract_code.lower():
        issues.append("Missing 'pragma solidity' statement")
    
    # Check for contract definition
    if 'contract ' not in contract_code.lower():
        issues.append("Missing contract definition")
    
    # Check for balanced braces
    open_braces = contract_code.count('{')
    close_braces = contract_code.count('}')
    if open_braces != close_braces:
        issues.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
    
    # Check for balanced parentheses
    open_parens = contract_code.count('(')
    close_parens = contract_code.count(')')
    if open_parens != close_parens:
        issues.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
    
    # Check for minimum content
    if len(contract_code.strip()) < 50:
        issues.append("Contract appears to be too short")
    
    # Check for common Solidity keywords
    solidity_keywords = ['function', 'modifier', 'event', 'struct', 'mapping']
    if not any(keyword in contract_code.lower() for keyword in solidity_keywords):
        issues.append("Contract doesn't contain common Solidity constructs")
    
    return len(issues) == 0, issues


def show_file_preview(contract_code: str, max_lines: int = 20):
    """
    Show a preview of the uploaded file.
    
    Args:
        contract_code: The contract source code
        max_lines: Maximum number of lines to show in preview
    """
    lines = contract_code.split('\n')
    
    if len(lines) <= max_lines:
        st.code(contract_code, language='solidity')
    else:
        preview_code = '\n'.join(lines[:max_lines])
        st.code(preview_code + f'\n... ({len(lines) - max_lines} more lines)', language='solidity')
        
        if st.button("Show Full Contract"):
            st.code(contract_code, language='solidity')


def show_upload_stats(contract_code: str):
    """
    Display statistics about the uploaded contract.
    
    Args:
        contract_code: The contract source code
    """
    lines = contract_code.split('\n')
    
    # Basic stats
    total_lines = len(lines)
    non_empty_lines = len([line for line in lines if line.strip()])
    comment_lines = len([line for line in lines if line.strip().startswith('//')])
    
    # Function analysis
    functions = contract_code.count('function ')
    public_functions = contract_code.count('function ') - contract_code.count('function ') + contract_code.count(' public')
    
    # Security-relevant patterns
    external_calls = contract_code.count('.call(') + contract_code.count('.send(') + contract_code.count('.transfer(')
    requires = contract_code.count('require(')
    modifiers = contract_code.count('modifier ')
    
    # Display stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Lines", total_lines)
        st.metric("Non-empty Lines", non_empty_lines)
    
    with col2:
        st.metric("Functions", functions)
        st.metric("Modifiers", modifiers)
    
    with col3:
        st.metric("External Calls", external_calls)
        st.metric("Require Statements", requires)
    
    with col4:
        st.metric("Comment Lines", comment_lines)
        comment_ratio = (comment_lines / total_lines * 100) if total_lines > 0 else 0
        st.metric("Comment Ratio", f"{comment_ratio:.1f}%")


def show_sample_contracts():
    """Display sample contracts that users can load."""
    st.markdown("### 📋 Sample Contracts")
    
    samples = {
        "Vulnerable Reentrancy": '''pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable to reentrancy
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
}''',
        
        "Access Control Issue": '''pragma solidity ^0.8.0;

contract VulnerableAccess {
    address public owner;
    uint256 public funds;
    
    constructor() {
        owner = msg.sender;
    }
    
    // Missing access control!
    function withdrawAll() public {
        payable(msg.sender).transfer(address(this).balance);
    }
    
    function deposit() public payable {
        funds += msg.value;
    }
}''',
        
        "Safe Contract": '''pragma solidity ^0.8.0;

contract SafeContract {
    address public owner;
    mapping(address => uint256) public balances;
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }
    
    function emergencyWithdraw() public onlyOwner {
        payable(owner).transfer(address(this).balance);
    }
}'''
    }
    
    selected_sample = st.selectbox("Choose a sample contract:", list(samples.keys()))
    
    if st.button("Load Sample Contract"):
        st.session_state.contract_code = samples[selected_sample]
        st.success(f"Loaded sample: {selected_sample}")
        st.rerun()
    
    # Show preview of selected sample
    with st.expander(f"Preview: {selected_sample}"):
        st.code(samples[selected_sample], language='solidity')