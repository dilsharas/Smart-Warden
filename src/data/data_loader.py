"""
Data loader for SmartBugs dataset and smart contract files.
"""

import os
import pandas as pd
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContractInfo:
    """Information about a smart contract."""
    filename: str
    code: str
    vulnerability: str
    label: int  # 0 = safe, 1 = vulnerable
    file_path: str
    code_hash: str


class ContractDataLoader:
    """
    Loads smart contract data from various sources including SmartBugs dataset.
    
    Supports loading from:
    - SmartBugs curated dataset (organized by vulnerability type)
    - Individual .sol files
    - Directory of contracts
    """
    
    def __init__(self, dataset_path: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.vulnerability_types = [
            'reentrancy',
            'access_control', 
            'arithmetic',
            'unchecked_calls',
            'denial_of_service',
            'bad_randomness',
            'safe'  # For safe contracts
        ]
        self.contracts: List[ContractInfo] = []
        
    def load_contracts(self, include_safe: bool = True) -> pd.DataFrame:
        """
        Load all contracts from the dataset.
        
        Args:
            include_safe: Whether to include safe contracts
            
        Returns:
            DataFrame with columns: filename, code, vulnerability, label, file_path, code_hash
        """
        logger.info(f"Loading contracts from {self.dataset_path}")
        self.contracts = []
        
        # Load from SmartBugs structure if it exists
        smartbugs_path = self.dataset_path / "smartbugs-curated"
        if smartbugs_path.exists():
            self._load_smartbugs_dataset(smartbugs_path, include_safe)
        
        # Load individual .sol files from root directory
        self._load_individual_contracts(self.dataset_path)
        
        # Convert to DataFrame
        df = self._contracts_to_dataframe()
        
        logger.info(f"Loaded {len(df)} contracts")
        logger.info(f"Vulnerability distribution: {df['vulnerability'].value_counts().to_dict()}")
        
        return df
    
    def _load_smartbugs_dataset(self, smartbugs_path: Path, include_safe: bool):
        """Load contracts from SmartBugs curated dataset structure."""
        for vuln_type in self.vulnerability_types:
            if vuln_type == 'safe' and not include_safe:
                continue
                
            vuln_dir = smartbugs_path / vuln_type
            if vuln_dir.exists():
                self._load_contracts_from_directory(vuln_dir, vuln_type)
    
    def _load_individual_contracts(self, directory: Path):
        """Load individual .sol files from directory."""
        for sol_file in directory.glob("*.sol"):
            try:
                code = self._read_contract_file(sol_file)
                if code:
                    # Try to infer vulnerability type from filename
                    vuln_type = self._infer_vulnerability_type(sol_file.name)
                    
                    contract_info = ContractInfo(
                        filename=sol_file.name,
                        code=code,
                        vulnerability=vuln_type,
                        label=1 if vuln_type != 'safe' else 0,
                        file_path=str(sol_file),
                        code_hash=self._calculate_hash(code)
                    )
                    self.contracts.append(contract_info)
                    
            except Exception as e:
                logger.warning(f"Failed to load contract {sol_file}: {e}")
    
    def _load_contracts_from_directory(self, directory: Path, vulnerability_type: str):
        """Load all .sol files from a specific vulnerability directory."""
        logger.debug(f"Loading {vulnerability_type} contracts from {directory}")
        
        for sol_file in directory.glob("*.sol"):
            try:
                code = self._read_contract_file(sol_file)
                if code:
                    contract_info = ContractInfo(
                        filename=sol_file.name,
                        code=code,
                        vulnerability=vulnerability_type,
                        label=1 if vulnerability_type != 'safe' else 0,
                        file_path=str(sol_file),
                        code_hash=self._calculate_hash(code)
                    )
                    self.contracts.append(contract_info)
                    
            except Exception as e:
                logger.warning(f"Failed to load contract {sol_file}: {e}")
    
    def _read_contract_file(self, file_path: Path) -> Optional[str]:
        """
        Read contract code from file with encoding handling.
        
        Args:
            file_path: Path to the .sol file
            
        Returns:
            Contract code as string or None if failed
        """
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    code = f.read().strip()
                    if code:
                        return code
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        logger.error(f"Failed to read {file_path} with any encoding")
        return None
    
    def _infer_vulnerability_type(self, filename: str) -> str:
        """
        Infer vulnerability type from filename.
        
        Args:
            filename: Name of the contract file
            
        Returns:
            Inferred vulnerability type
        """
        filename_lower = filename.lower()
        
        # Check for vulnerability keywords in filename
        if any(keyword in filename_lower for keyword in ['reentrancy', 'reentrant']):
            return 'reentrancy'
        elif any(keyword in filename_lower for keyword in ['access', 'control', 'owner']):
            return 'access_control'
        elif any(keyword in filename_lower for keyword in ['overflow', 'underflow', 'arithmetic']):
            return 'arithmetic'
        elif any(keyword in filename_lower for keyword in ['unchecked', 'call']):
            return 'unchecked_calls'
        elif any(keyword in filename_lower for keyword in ['dos', 'denial', 'service']):
            return 'denial_of_service'
        elif any(keyword in filename_lower for keyword in ['random', 'timestamp', 'block']):
            return 'bad_randomness'
        elif any(keyword in filename_lower for keyword in ['safe', 'secure']):
            return 'safe'
        else:
            # Default to safe if can't determine
            return 'safe'
    
    def _calculate_hash(self, code: str) -> str:
        """Calculate SHA-256 hash of contract code."""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()
    
    def _contracts_to_dataframe(self) -> pd.DataFrame:
        """Convert loaded contracts to pandas DataFrame."""
        if not self.contracts:
            return pd.DataFrame(columns=['filename', 'code', 'vulnerability', 'label', 'file_path', 'code_hash'])
        
        data = []
        for contract in self.contracts:
            data.append({
                'filename': contract.filename,
                'code': contract.code,
                'vulnerability': contract.vulnerability,
                'label': contract.label,
                'file_path': contract.file_path,
                'code_hash': contract.code_hash
            })
        
        return pd.DataFrame(data)
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.contracts:
            return {}
        
        df = self._contracts_to_dataframe()
        
        stats = {
            'total_contracts': len(df),
            'vulnerable_contracts': len(df[df['label'] == 1]),
            'safe_contracts': len(df[df['label'] == 0]),
            'vulnerability_distribution': df['vulnerability'].value_counts().to_dict(),
            'average_code_length': df['code'].str.len().mean(),
            'median_code_length': df['code'].str.len().median(),
            'min_code_length': df['code'].str.len().min(),
            'max_code_length': df['code'].str.len().max()
        }
        
        return stats
    
    def save_to_csv(self, output_path: str, include_code: bool = True):
        """
        Save loaded contracts to CSV file.
        
        Args:
            output_path: Path to save the CSV file
            include_code: Whether to include the full contract code
        """
        df = self._contracts_to_dataframe()
        
        if not include_code:
            # Save without full code for smaller file size
            df = df.drop('code', axis=1)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} contracts to {output_path}")
    
    def load_single_contract(self, file_path: str) -> Optional[ContractInfo]:
        """
        Load a single contract file.
        
        Args:
            file_path: Path to the contract file
            
        Returns:
            ContractInfo object or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists() or file_path.suffix != '.sol':
            logger.error(f"Invalid contract file: {file_path}")
            return None
        
        code = self._read_contract_file(file_path)
        if not code:
            return None
        
        vuln_type = self._infer_vulnerability_type(file_path.name)
        
        return ContractInfo(
            filename=file_path.name,
            code=code,
            vulnerability=vuln_type,
            label=1 if vuln_type != 'safe' else 0,
            file_path=str(file_path),
            code_hash=self._calculate_hash(code)
        )
    
    def validate_dataset(self) -> Dict[str, List[str]]:
        """
        Validate the loaded dataset for common issues.
        
        Returns:
            Dictionary with validation results
        """
        issues = {
            'empty_contracts': [],
            'duplicate_hashes': [],
            'encoding_issues': [],
            'missing_pragma': []
        }
        
        seen_hashes = set()
        
        for contract in self.contracts:
            # Check for empty contracts
            if len(contract.code.strip()) < 50:  # Very short contracts
                issues['empty_contracts'].append(contract.filename)
            
            # Check for duplicates
            if contract.code_hash in seen_hashes:
                issues['duplicate_hashes'].append(contract.filename)
            seen_hashes.add(contract.code_hash)
            
            # Check for missing pragma
            if 'pragma solidity' not in contract.code.lower():
                issues['missing_pragma'].append(contract.filename)
        
        return issues


def main():
    """Example usage of ContractDataLoader."""
    # Initialize loader
    loader = ContractDataLoader("data/raw")
    
    # Load all contracts
    df = loader.load_contracts()
    print(f"Loaded {len(df)} contracts")
    
    # Get statistics
    stats = loader.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Validate dataset
    issues = loader.validate_dataset()
    print("\nValidation Issues:")
    for issue_type, files in issues.items():
        if files:
            print(f"  {issue_type}: {len(files)} files")
    
    # Save to CSV
    loader.save_to_csv("data/processed/all_contracts.csv")


if __name__ == "__main__":
    main()