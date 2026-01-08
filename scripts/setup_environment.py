#!/usr/bin/env python3
"""
Environment setup script for the Smart Contract AI Analyzer.
This script sets up the development environment and downloads necessary datasets.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")


def install_requirements():
    """Install Python requirements."""
    logger.info("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw/smartbugs-curated",
        "data/processed",
        "models",
        "logs",
        "results/model_performance",
        "results/benchmarks",
        "results/test_cases",
        "results/reports",
        "notebooks",
        "dashboard/assets",
        "docs/thesis/figures"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_sample_datasets():
    """Download sample vulnerable contracts for testing."""
    logger.info("Creating sample vulnerable contracts...")
    
    # Sample reentrancy vulnerable contract
    reentrancy_contract = '''pragma solidity ^0.8.0;

contract ReentrancyVulnerable {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}'''
    
    # Sample access control vulnerable contract
    access_control_contract = '''pragma solidity ^0.8.0;

contract AccessControlVulnerable {
    address public owner;
    uint256 public totalSupply;
    
    constructor() {
        owner = msg.sender;
        totalSupply = 1000000;
    }
    
    // Vulnerable: missing access control
    function mint(address to, uint256 amount) public {
        totalSupply += amount;
        // Anyone can mint tokens!
    }
    
    // Vulnerable: weak access control
    function destroy() public {
        require(msg.sender == owner, "Only owner");
        selfdestruct(payable(owner));
    }
}'''
    
    # Sample safe contract
    safe_contract = '''pragma solidity ^0.8.0;

contract SafeContract {
    mapping(address => uint256) public balances;
    bool private locked;
    address public owner;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
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
    
    function withdraw(uint256 amount) public noReentrancy {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Safe: state update before external call
        balances[msg.sender] -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function emergencyStop() public onlyOwner {
        // Safe: proper access control
        selfdestruct(payable(owner));
    }
}'''
    
    # Create sample contract files
    contracts = [
        ("data/raw/reentrancy_vulnerable.sol", reentrancy_contract),
        ("data/raw/access_control_vulnerable.sol", access_control_contract),
        ("data/raw/safe_contract.sol", safe_contract)
    ]
    
    for filename, content in contracts:
        with open(filename, 'w') as f:
            f.write(content)
        logger.info(f"Created sample contract: {filename}")


def setup_git():
    """Initialize Git repository if not already initialized."""
    if not Path(".git").exists():
        logger.info("Initializing Git repository...")
        try:
            subprocess.check_call(["git", "init"])
            subprocess.check_call(["git", "add", "."])
            subprocess.check_call(["git", "commit", "-m", "Initial commit: Project structure setup"])
            logger.info("Git repository initialized")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git initialization failed: {e}")
    else:
        logger.info("Git repository already exists")


def check_external_tools():
    """Check if external security tools are available."""
    tools = ["solc", "slither", "myth"]
    
    for tool in tools:
        try:
            subprocess.check_call([tool, "--version"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            logger.info(f"✓ {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"✗ {tool} is not available - install it for full functionality")


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("Created .env file from template")
        else:
            logger.warning(".env.example not found")
    else:
        logger.info(".env file already exists")


def main():
    """Main setup function."""
    logger.info("Starting environment setup...")
    
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    
    # Run setup steps
    check_python_version()
    create_directories()
    install_requirements()
    download_sample_datasets()
    create_env_file()
    setup_git()
    check_external_tools()
    
    logger.info("Environment setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Install external tools: npm install -g solc && pip install slither-analyzer mythril")
    logger.info("2. Update .env file with your configuration")
    logger.info("3. Run tests: pytest tests/ -v")
    logger.info("4. Start development: python -m src.api.app")


if __name__ == "__main__":
    main()