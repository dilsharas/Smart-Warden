#!/usr/bin/env python3
"""
SmartBugs Wild Dataset Integration for Smart Warden.
Processes the SmartBugs Wild dataset for ML training.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import hashlib
import pickle

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from features.feature_extractor import SolidityFeatureExtractor

logger = logging.getLogger(__name__)

class SmartBugsWildProcessor:
    """
    Processes SmartBugs Wild dataset for Smart Warden training.
    """
    
    def __init__(self, smartbugs_path: str, output_dir: str = "data/processed"):
        """
        Initialize the processor.
        
        Args:
            smartbugs_path: Path to smartbugs-wild-master directory
            output_dir: Directory to save processed data
        """
        self.smartbugs_path = Path(smartbugs_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vulnerability type mapping
        self.vulnerability_mapping = {
            'reentrancy': 'reentrancy',
            'access_control': 'access_control', 
            'arithmetic': 'arithmetic',
            'unchecked_calls': 'unchecked_calls',
            'dos': 'dos',
            'bad_randomness': 'bad_randomness',
            'front_running': 'access_control',  # Map to access_control
            'time_manipulation': 'bad_randomness',  # Map to bad_randomness
            'short_addresses': 'unchecked_calls',  # Map to unchecked_calls
        }
        
        # Initialize feature extractor
        self.feature_extractor = SolidityFeatureExtractor()
        
    def discover_dataset_structure(self):
        """Discover and analyze the dataset structure."""
        print("🔍 Analyzing SmartBugs Wild dataset structure...")
        
        structure = {
            'contracts_dir': None,
            'results_dir': None,
            'metadata_files': [],
            'total_contracts': 0,
            'available_tools': []
        }
        
        # Look for contracts directory
        for subdir in ['contracts', 'dataset', 'source']:
            contracts_path = self.smartbugs_path / subdir
            if contracts_path.exists():
                structure['contracts_dir'] = contracts_path
                structure['total_contracts'] = len(list(contracts_path.glob('**/*.sol')))
                break
        
        # Look for results directory
        for subdir in ['results', 'analysis', 'output']:
            results_path = self.smartbugs_path / subdir
            if results_path.exists():
                structure['results_dir'] = results_path
                # Find available tools
                if results_path.exists():
                    structure['available_tools'] = [d.name for d in results_path.iterdir() if d.is_dir()]
                break
        
        # Look for metadata files
        for pattern in ['*.json', '*.csv', 'metadata/*']:
            metadata_files = list(self.smartbugs_path.glob(pattern))
            structure['metadata_files'].extend(metadata_files)
        
        return structure
    
    def load_vulnerability_labels(self) -> Dict[str, Dict]:
        """
        Load vulnerability labels from SmartBugs Wild results.
        
        Returns:
            Dictionary mapping contract names to vulnerability info
        """
        print("📊 Loading vulnerability labels...")
        
        structure = self.discover_dataset_structure()
        labels = {}
        
        # Try to load from results directory
        if structure['results_dir']:
            results_dir = structure['results_dir']
            
            # Process each tool's results
            for tool_name in structure['available_tools']:
                tool_dir = results_dir / tool_name
                if not tool_dir.exists():
                    continue
                    
                print(f"  Processing {tool_name} results...")
                
                # Look for JSON result files
                for result_file in tool_dir.glob('**/*.json'):
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        contract_name = result_file.stem
                        
                        if contract_name not in labels:
                            labels[contract_name] = {
                                'vulnerabilities': set(),
                                'tools': {},
                                'is_vulnerable': False
                            }
                        
                        # Extract vulnerabilities from result
                        vulns = self._extract_vulnerabilities_from_result(result_data, tool_name)
                        labels[contract_name]['vulnerabilities'].update(vulns)
                        labels[contract_name]['tools'][tool_name] = result_data
                        
                        if vulns:
                            labels[contract_name]['is_vulnerable'] = True
                            
                    except Exception as e:
                        logger.warning(f"Error processing {result_file}: {e}")
                        continue
        
        # Convert sets to lists for JSON serialization
        for contract_name in labels:
            labels[contract_name]['vulnerabilities'] = list(labels[contract_name]['vulnerabilities'])
        
        print(f"  ✅ Loaded labels for {len(labels)} contracts")
        return labels
    
    def _extract_vulnerabilities_from_result(self, result_data: Dict, tool_name: str) -> List[str]:
        """Extract vulnerability types from tool result."""
        vulnerabilities = []
        
        if tool_name.lower() == 'slither':
            # Slither result format
            if 'results' in result_data and 'detectors' in result_data['results']:
                for detector in result_data['results']['detectors']:
                    vuln_type = self._map_slither_detector(detector.get('check', ''))
                    if vuln_type:
                        vulnerabilities.append(vuln_type)
        
        elif tool_name.lower() == 'mythril':
            # Mythril result format
            if 'issues' in result_data:
                for issue in result_data['issues']:
                    vuln_type = self._map_mythril_issue(issue.get('title', ''))
                    if vuln_type:
                        vulnerabilities.append(vuln_type)
        
        elif tool_name.lower() == 'securify':
            # Securify result format
            if 'results' in result_data:
                for result in result_data['results']:
                    vuln_type = self._map_securify_result(result.get('name', ''))
                    if vuln_type:
                        vulnerabilities.append(vuln_type)
        
        return list(set(vulnerabilities))  # Remove duplicates
    
    def _map_slither_detector(self, detector_name: str) -> Optional[str]:
        """Map Slither detector names to our vulnerability types."""
        detector_mapping = {
            'reentrancy': 'reentrancy',
            'calls-loop': 'dos',
            'timestamp': 'bad_randomness',
            'block-timestamp': 'bad_randomness',
            'weak-prng': 'bad_randomness',
            'unchecked-lowlevel': 'unchecked_calls',
            'unchecked-send': 'unchecked_calls',
            'arbitrary-send': 'access_control',
            'suicidal': 'access_control',
            'unprotected-upgrade': 'access_control',
            'integer-overflow': 'arithmetic',
            'divide-before-multiply': 'arithmetic'
        }
        
        for pattern, vuln_type in detector_mapping.items():
            if pattern in detector_name.lower():
                return vuln_type
        
        return None
    
    def _map_mythril_issue(self, issue_title: str) -> Optional[str]:
        """Map Mythril issue titles to our vulnerability types."""
        title_lower = issue_title.lower()
        
        if 'reentrancy' in title_lower:
            return 'reentrancy'
        elif 'integer overflow' in title_lower or 'integer underflow' in title_lower:
            return 'arithmetic'
        elif 'unchecked call' in title_lower:
            return 'unchecked_calls'
        elif 'access control' in title_lower or 'authorization' in title_lower:
            return 'access_control'
        elif 'timestamp' in title_lower or 'block number' in title_lower:
            return 'bad_randomness'
        elif 'dos' in title_lower or 'denial of service' in title_lower:
            return 'dos'
        
        return None
    
    def _map_securify_result(self, result_name: str) -> Optional[str]:
        """Map Securify result names to our vulnerability types."""
        # Similar mapping for Securify
        name_lower = result_name.lower()
        
        if 'reentrancy' in name_lower:
            return 'reentrancy'
        elif 'overflow' in name_lower:
            return 'arithmetic'
        elif 'call' in name_lower:
            return 'unchecked_calls'
        
        return None
    
    def load_contracts(self, max_contracts: Optional[int] = None) -> List[Dict]:
        """
        Load Solidity contracts from the dataset.
        
        Args:
            max_contracts: Maximum number of contracts to load (None for all)
            
        Returns:
            List of contract dictionaries
        """
        print("📁 Loading Solidity contracts...")
        
        structure = self.discover_dataset_structure()
        contracts = []
        
        if not structure['contracts_dir']:
            raise ValueError("Contracts directory not found in SmartBugs Wild dataset")
        
        contracts_dir = structure['contracts_dir']
        sol_files = list(contracts_dir.glob('**/*.sol'))
        
        if max_contracts:
            sol_files = sol_files[:max_contracts]
        
        print(f"  Found {len(sol_files)} Solidity files")
        
        for sol_file in tqdm(sol_files, desc="Loading contracts"):
            try:
                with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                contracts.append({
                    'filename': sol_file.name,
                    'path': str(sol_file),
                    'code': code,
                    'size': len(code),
                    'hash': hashlib.md5(code.encode()).hexdigest()
                })
                
            except Exception as e:
                logger.warning(f"Error loading {sol_file}: {e}")
                continue
        
        print(f"  ✅ Loaded {len(contracts)} contracts successfully")
        return contracts
    
    def create_training_dataset(self, max_contracts: Optional[int] = None, 
                              save_intermediate: bool = True) -> pd.DataFrame:
        """
        Create complete training dataset with features and labels.
        
        Args:
            max_contracts: Maximum contracts to process
            save_intermediate: Save intermediate results
            
        Returns:
            DataFrame with features and labels
        """
        print("🏗️ Creating training dataset from SmartBugs Wild...")
        
        # Load contracts and labels
        contracts = self.load_contracts(max_contracts)
        labels = self.load_vulnerability_labels()
        
        # Process contracts
        dataset = []
        
        for contract in tqdm(contracts, desc="Extracting features"):
            try:
                # Extract features
                features = self.feature_extractor.extract_features(contract['code'])
                
                # Get labels for this contract
                contract_name = Path(contract['filename']).stem
                contract_labels = labels.get(contract_name, {
                    'vulnerabilities': [],
                    'is_vulnerable': False
                })
                
                # Create record
                record = {
                    'filename': contract['filename'],
                    'contract_hash': contract['hash'],
                    'is_vulnerable': contract_labels['is_vulnerable'],
                    'vulnerabilities': contract_labels['vulnerabilities'],
                    'vulnerability_count': len(contract_labels['vulnerabilities']),
                    **features
                }
                
                # Add binary label
                record['binary_label'] = 1 if contract_labels['is_vulnerable'] else 0
                
                # Add primary vulnerability type for multi-class
                if contract_labels['vulnerabilities']:
                    record['primary_vulnerability'] = contract_labels['vulnerabilities'][0]
                else:
                    record['primary_vulnerability'] = 'safe'
                
                dataset.append(record)
                
            except Exception as e:
                logger.warning(f"Error processing {contract['filename']}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(dataset)
        
        # Save intermediate results
        if save_intermediate:
            output_file = self.output_dir / 'smartbugs_wild_dataset.pkl'
            df.to_pickle(output_file)
            print(f"  💾 Saved dataset to {output_file}")
            
            # Save CSV for inspection
            csv_file = self.output_dir / 'smartbugs_wild_dataset.csv'
            df.to_csv(csv_file, index=False)
            print(f"  📊 Saved CSV to {csv_file}")
        
        print(f"  ✅ Created dataset with {len(df)} samples")
        print(f"  📈 Vulnerable contracts: {df['is_vulnerable'].sum()}")
        print(f"  📉 Safe contracts: {(~df['is_vulnerable']).sum()}")
        
        return df
    
    def analyze_dataset_statistics(self, df: pd.DataFrame):
        """Analyze and print dataset statistics."""
        print("\n📊 SmartBugs Wild Dataset Statistics")
        print("=" * 50)
        
        # Basic stats
        print(f"Total contracts: {len(df)}")
        print(f"Vulnerable contracts: {df['is_vulnerable'].sum()} ({df['is_vulnerable'].mean():.1%})")
        print(f"Safe contracts: {(~df['is_vulnerable']).sum()} ({(~df['is_vulnerable']).mean():.1%})")
        
        # Vulnerability type distribution
        print("\nVulnerability Type Distribution:")
        vuln_types = {}
        for vulns in df['vulnerabilities']:
            for vuln in vulns:
                vuln_types[vuln] = vuln_types.get(vuln, 0) + 1
        
        for vuln_type, count in sorted(vuln_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {vuln_type}: {count}")
        
        # Feature statistics
        feature_cols = [col for col in df.columns if col not in 
                       ['filename', 'contract_hash', 'is_vulnerable', 'vulnerabilities', 
                        'vulnerability_count', 'binary_label', 'primary_vulnerability']]
        
        print(f"\nFeatures extracted: {len(feature_cols)}")
        print(f"Average lines of code: {df['lines_of_code'].mean():.1f}")
        print(f"Average function count: {df['function_count'].mean():.1f}")
        
        return vuln_types


def main():
    """Main function to process SmartBugs Wild dataset."""
    # Configuration
    SMARTBUGS_PATH = r"E:\FreeLance-work\vinod_project\smartbugs-wild-master"
    MAX_CONTRACTS = 1000  # Start with subset for testing
    
    # Initialize processor
    processor = SmartBugsWildProcessor(SMARTBUGS_PATH)
    
    # Discover dataset structure
    structure = processor.discover_dataset_structure()
    print("Dataset Structure:")
    for key, value in structure.items():
        print(f"  {key}: {value}")
    
    # Create training dataset
    df = processor.create_training_dataset(max_contracts=MAX_CONTRACTS)
    
    # Analyze statistics
    processor.analyze_dataset_statistics(df)
    
    print("\n🎉 SmartBugs Wild integration complete!")
    print("Next steps:")
    print("1. Review the generated dataset")
    print("2. Update your training scripts to use this data")
    print("3. Retrain your models with the larger dataset")


if __name__ == "__main__":
    main()