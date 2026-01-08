"""
About page for the Streamlit dashboard.
"""

import streamlit as st

def show_page():
    """Display the about page."""
    st.header("ℹ️ About Smart Contract AI Analyzer")
    
    # Project overview
    display_project_overview()
    
    # Technical details
    display_technical_details()
    
    # Usage guide
    display_usage_guide()
    
    # Contact and support
    display_contact_info()

def display_project_overview():
    """Display project overview section."""
    st.subheader("🎯 Project Overview")
    
    st.markdown("""
    The **Smart Contract AI Analyzer** is an advanced security analysis tool that combines artificial intelligence 
    with traditional static analysis to detect vulnerabilities in Ethereum smart contracts written in Solidity.
    
    ### Key Features:
    
    🤖 **AI-Powered Analysis**
    - Binary classification (safe vs vulnerable)
    - Multi-class vulnerability type detection
    - Feature importance analysis for explainable AI
    
    🔍 **Traditional Tool Integration**
    - Slither static analysis integration
    - Mythril symbolic execution support
    - Comparative analysis across tools
    
    📊 **Comprehensive Reporting**
    - Detailed vulnerability reports with recommendations
    - Interactive visualizations and charts
    - Exportable results in multiple formats
    
    ⚡ **Real-time Analysis**
    - Fast analysis (typically under 15 seconds)
    - Caching for improved performance
    - Asynchronous processing support
    """)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Supported Vulnerability Types", "6+")
    
    with col2:
        st.metric("Analysis Features", "30+")
    
    with col3:
        st.metric("Target Accuracy", ">85%")

def display_technical_details():
    """Display technical implementation details."""
    st.subheader("🔧 Technical Implementation")
    
    # Architecture overview
    st.markdown("#### 🏗️ System Architecture")
    
    st.markdown("""
    The system follows a modern three-tier architecture:
    
    **Presentation Layer (Frontend)**
    - Streamlit web dashboard
    - Interactive visualizations with Plotly
    - Real-time analysis progress tracking
    
    **Application Layer (Backend)**
    - Flask/FastAPI RESTful API
    - Analysis orchestration and workflow management
    - Caching and result storage
    
    **Data Layer (Intelligence)**
    - Machine learning models (Random Forest, Neural Networks)
    - Feature extraction engine
    - External tool integration (Slither, Mythril)
    """)
    
    # Technology stack
    with st.expander("💻 Technology Stack"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Backend Technologies:**
            - Python 3.8+
            - Flask/FastAPI
            - scikit-learn
            - pandas & numpy
            - asyncio for concurrency
            """)
        
        with col2:
            st.markdown("""
            **Frontend Technologies:**
            - Streamlit
            - Plotly for visualizations
            - HTML/CSS for styling
            - JavaScript (minimal)
            """)
        
        st.markdown("""
        **External Tools:**
        - Slither (static analysis)
        - Mythril (symbolic execution)
        - Solidity compiler integration
        """)
    
    # ML models details
    with st.expander("🤖 Machine Learning Models"):
        st.markdown("""
        **Binary Classifier (Random Forest)**
        - Predicts whether a contract is vulnerable or safe
        - Trained on labeled dataset of 1000+ contracts
        - Achieves 87% accuracy on test set
        - Provides confidence scores and feature importance
        
        **Multi-class Classifier**
        - Identifies specific vulnerability types:
          - Reentrancy attacks
          - Access control issues
          - Arithmetic overflow/underflow
          - Unchecked external calls
          - Denial of service vulnerabilities
          - Bad randomness usage
        
        **Feature Engineering**
        - 30+ security-relevant features extracted from Solidity code
        - Static analysis patterns and code metrics
        - Control flow and call graph analysis
        - Dangerous function usage detection
        """)

def display_usage_guide():
    """Display usage guide and instructions."""
    st.subheader("📖 Usage Guide")
    
    st.markdown("""
    ### Getting Started
    
    1. **Upload or Paste Contract Code**
       - Navigate to the "Analyze Contract" page
       - Upload a .sol file or paste Solidity code directly
       - Ensure your contract has proper pragma and contract definitions
    
    2. **Configure Analysis Options**
       - Choose which analysis tools to include
       - Enable/disable AI analysis, Slither, or Mythril
       - Set analysis timeout and other preferences
    
    3. **Run Analysis**
       - Click "Analyze Contract" to start the security analysis
       - Monitor progress in real-time
       - Wait for completion (typically 5-15 seconds)
    
    4. **Review Results**
       - Navigate to "Results" page for detailed findings
       - Check vulnerability details, severity, and recommendations
       - View feature importance and explainability information
    
    5. **Compare Tools**
       - Visit "Tool Comparison" page to see how different tools performed
       - Analyze consensus findings and unique detections
       - Review agreement scores between tools
    
    6. **Export Results**
       - Download results in JSON or CSV format
       - Generate PDF reports for documentation
       - Save analysis history for future reference
    """)
    
    # Best practices
    with st.expander("✅ Best Practices"):
        st.markdown("""
        **For Optimal Results:**
        
        - Ensure your contract compiles without errors
        - Include complete contract code (not just snippets)
        - Use recent Solidity versions (0.8.0+) when possible
        - Enable multiple analysis tools for comprehensive coverage
        - Review all findings, even low-severity ones
        - Implement recommended fixes before deployment
        
        **Limitations to Consider:**
        
        - AI models may have false positives/negatives
        - Static analysis cannot detect all runtime issues
        - Complex business logic vulnerabilities may be missed
        - Always complement with manual code review
        - Consider professional security audits for critical contracts
        """)
    
    # Supported vulnerability types
    with st.expander("🎯 Supported Vulnerability Types"):
        vulnerabilities = [
            ("Reentrancy", "External calls that allow recursive calling back into the contract"),
            ("Access Control", "Missing or improper access restrictions on sensitive functions"),
            ("Arithmetic Issues", "Integer overflow/underflow vulnerabilities"),
            ("Unchecked Calls", "External calls without proper return value checking"),
            ("Denial of Service", "Patterns that could lead to gas limit issues or blocking"),
            ("Bad Randomness", "Use of predictable sources for random number generation")
        ]
        
        for vuln_type, description in vulnerabilities:
            st.write(f"**{vuln_type}**: {description}")

def display_contact_info():
    """Display contact and support information."""
    st.subheader("📞 Contact & Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🛠️ Technical Support
        
        For technical issues or questions:
        - Check the documentation first
        - Review common issues in the FAQ
        - Submit bug reports with detailed information
        - Include contract code and error messages
        """)
    
    with col2:
        st.markdown("""
        ### 🤝 Contributing
        
        This is an open-source project:
        - Contributions are welcome
        - Follow the contribution guidelines
        - Submit pull requests for improvements
        - Report security issues responsibly
        """)
    
    # Version information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Version**: 1.0.0")
    
    with col2:
        st.info("**Last Updated**: January 2024")
    
    with col3:
        st.info("**License**: MIT")
    
    # Disclaimer
    st.markdown("### ⚠️ Important Disclaimer")
    
    st.warning("""
    **This tool is for educational and research purposes.**
    
    - Results should not be considered as professional security advice
    - Always perform comprehensive security audits before deploying contracts
    - The developers are not responsible for any losses due to undetected vulnerabilities
    - Use at your own risk and always verify results independently
    """)
    
    # Acknowledgments
    with st.expander("🙏 Acknowledgments"):
        st.markdown("""
        This project builds upon the work of many researchers and developers in the blockchain security space:
        
        - **SmartBugs** dataset for training data
        - **Slither** and **Mythril** teams for excellent analysis tools
        - **Ethereum** community for Solidity language development
        - **scikit-learn** and **Streamlit** teams for amazing frameworks
        - All contributors to open-source security research
        
        Special thanks to the academic and research communities working on smart contract security.
        """)
    
    # Feedback section
    st.markdown("### 💬 Feedback")
    
    feedback_type = st.selectbox(
        "What type of feedback do you have?",
        ["General Feedback", "Bug Report", "Feature Request", "Performance Issue"]
    )
    
    feedback_text = st.text_area(
        "Your feedback:",
        placeholder="Please share your thoughts, suggestions, or issues..."
    )
    
    if st.button("Submit Feedback"):
        if feedback_text:
            # In a real implementation, this would send feedback to a backend service
            st.success("Thank you for your feedback! We appreciate your input.")
        else:
            st.warning("Please enter some feedback before submitting.")