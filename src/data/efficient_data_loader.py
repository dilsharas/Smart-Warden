#!/usr/bin/env python3
"""
Efficient Data Loading and Preprocessing Pipeline for SmartBugs Wild Dataset.
Implements parallel I/O, smart caching, and memory-efficient processing.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Union
import logging
from tqdm import tqdm
import hashlib
import pickle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass, asdict
import sqlite3
import joblib
from functools import lru_cache
import gc
import psutil

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

logger = logging.getLogger(__name__)

@dataclass
class ContractData:
    """Efficient contract data structure."""
    filename: str
    code: str
    hash: str
    size: int
    vulnerabilities: List[str]
    is_vulnerable: bool
    metadata: Dict

@dataclass
class ProcessingStats:
    """Statistics for processing performance."""
    total_contracts: int = 0
    processed_contracts: int = 0
    failed_contracts: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class SmartBugsDataLoader:
    """
    Efficient data loader for SmartBugs Wild dataset with parallel processing,
    smart caching, and memory optimization.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 cache_dir: str = "data/cache",
                 max_workers: int = None,
                 chunk_size: int = 1000,
                 enable_caching: bool = True):
        """
        Initialize the efficient data loader.
        
        Args:
            dataset_path: Path to SmartBugs Wild dataset
            cache_dir: Directory for caching processed data
            max_workers: Number of parallel workers (None for auto-detect)
            chunk_size: Size of processing chunks
            enable_caching: Enable smart caching
        """
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance settings
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        self.enable_caching = enable_caching
        
        # Cache database
        self.cache_db_path = self.cache_dir / "contract_cache.db"
        self._init_cache_db()
        
        # Statistics
        self.stats = ProcessingStats()
        
        # Dataset structure discovery
        self.dataset_structure = None
        
        logger.info(f"Initialized SmartBugsDataLoader with {self.max_workers} workers")
    
    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contract_cache (
                    hash TEXT PRIMARY KEY,
                    filename TEXT,
                    features BLOB,
                    labels BLOB,
                    metadata BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_cache (
                    dataset_hash TEXT PRIMARY KEY,
                    processed_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def discover_dataset_structure(self) -> Dict:
        """
        Efficiently discover and analyze dataset structure with caching.
        
        Returns:
            Dictionary containing dataset structure information
        """
        if self.dataset_structure is not None:
            return self.dataset_structure
        
        logger.info("🔍 Discovering SmartBugs Wild dataset structure...")
        
        structure = {
            'contracts_dir': None,
            'results_dir': None,
            'metadata_files': [],
            'total_contracts': 0,
            'available_tools': [],
            'vulnerability_types': set(),
            'dataset_size_mb': 0
        }
        
        # Parallel directory scanning
        def scan_directory(directory: Path) -> Dict:
            """Scan directory for relevant files."""
            local_structure = {
                'sol_files': [],
                'json_files': [],
                'subdirs': [],
                'size_mb': 0
            }
            
            try:
                for item in directory.rglob('*'):
                    if item.is_file():
                        size = item.stat().st_size
                        local_structure['size_mb'] += size / (1024 * 1024)
                        
                        if item.suffix == '.sol':
                            local_structure['sol_files'].append(item)
                        elif item.suffix == '.json':
                            local_structure['json_files'].append(item)
                    elif item.is_dir():
                        local_structure['subdirs'].append(item)
            except Exception as e:
                logger.warning(f"Error scanning {directory}: {e}")
            
            return local_structure
        
        # Find main directories
        for subdir_name in ['contracts', 'dataset', 'source', 'smartbugs-wild']:
            contracts_path = self.dataset_path / subdir_name
            if contracts_path.exists():
                scan_result = scan_directory(contracts_path)
                structure['contracts_dir'] = contracts_path
                structure['total_contracts'] = len(scan_result['sol_files'])
                structure['dataset_size_mb'] += scan_result['size_mb']
                break
        
        # Find results directory
        for subdir_name in ['results', 'analysis', 'output']:
            results_path = self.dataset_path / subdir_name
            if results_path.exists():
                structure['results_dir'] = results_path
                # Get available tools
                structure['available_tools'] = [
                    d.name for d in results_path.iterdir() 
                    if d.is_dir()
                ]
                break
        
        # Find metadata files
        for pattern in ['*.json', '*.csv', 'metadata/*']:
            metadata_files = list(self.dataset_path.glob(pattern))
            structure['metadata_files'].extend(metadata_files)
        
        self.dataset_structure = structure
        logger.info(f"✅ Dataset structure discovered: {structure['total_contracts']} contracts, "
                   f"{structure['dataset_size_mb']:.1f} MB")
        
        return structure
    
    def load_contracts_parallel(self, 
                              max_contracts: Optional[int] = None,
                              sample_ratio: float = 1.0) -> List[ContractData]:
        """
        Load contracts with parallel processing and smart sampling.
        
        Args:
            max_contracts: Maximum number of contracts to load
            sample_ratio: Ratio of contracts to sample (0.0-1.0)
            
        Returns:
            List of ContractData objects
        """
        logger.info("📁 Loading contracts with parallel processing...")
        start_time = time.time()
        
        structure = self.discover_dataset_structure()
        if not structure['contracts_dir']:
            raise ValueError("Contracts directory not found in dataset")
        
        # Get all contract files
        contracts_dir = structure['contracts_dir']
        sol_files = list(contracts_dir.rglob('*.sol'))
        
        # Apply sampling
        if sample_ratio < 1.0:
            import random
            random.shuffle(sol_files)
            sol_files = sol_files[:int(len(sol_files) * sample_ratio)]
        
        if max_contracts:
            sol_files = sol_files[:max_contracts]
        
        logger.info(f"Processing {len(sol_files)} contract files...")
        
        # Process in parallel chunks
        contracts = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunks for processing
            chunk_futures = []
            for i in range(0, len(sol_files), self.chunk_size):
                chunk = sol_files[i:i + self.chunk_size]
                future = executor.submit(self._process_contract_chunk, chunk)
                chunk_futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(as_completed(chunk_futures), 
                             total=len(chunk_futures), 
                             desc="Processing chunks"):
                try:
                    chunk_contracts = future.result()
                    contracts.extend(chunk_contracts)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    self.stats.failed_contracts += len(chunk)
        
        # Update statistics
        self.stats.total_contracts = len(sol_files)
        self.stats.processed_contracts = len(contracts)
        self.stats.processing_time = time.time() - start_time
        self.stats.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        logger.info(f"✅ Loaded {len(contracts)} contracts in {self.stats.processing_time:.2f}s")
        logger.info(f"📊 Success rate: {len(contracts)/len(sol_files)*100:.1f}%")
        
        return contracts
    
    def _process_contract_chunk(self, sol_files: List[Path]) -> List[ContractData]:
        """Process a chunk of contract files."""
        contracts = []
        
        for sol_file in sol_files:
            try:
                # Check cache first
                if self.enable_caching:
                    cached_contract = self._get_cached_contract(sol_file)
                    if cached_contract:
                        contracts.append(cached_contract)
                        self.stats.cache_hits += 1
                        continue
                
                # Load and process contract
                contract = self._load_single_contract(sol_file)
                if contract:
                    contracts.append(contract)
                    
                    # Cache the result
                    if self.enable_caching:
                        self._cache_contract(contract)
                        self.stats.cache_misses += 1
                
            except Exception as e:
                logger.warning(f"Error processing {sol_file}: {e}")
                continue
        
        return contracts
    
    def _load_single_contract(self, sol_file: Path) -> Optional[ContractData]:
        """Load a single contract file efficiently."""
        try:
            # Read file with encoding detection
            code = self._read_file_with_encoding(sol_file)
            if not code or len(code.strip()) < 10:
                return None
            
            # Calculate hash
            contract_hash = hashlib.md5(code.encode()).hexdigest()
            
            # Get vulnerability labels (placeholder - will be enhanced)
            vulnerabilities, is_vulnerable = self._get_contract_labels(sol_file)
            
            # Create contract data
            contract = ContractData(
                filename=sol_file.name,
                code=code,
                hash=contract_hash,
                size=len(code),
                vulnerabilities=vulnerabilities,
                is_vulnerable=is_vulnerable,
                metadata={
                    'path': str(sol_file),
                    'size_bytes': sol_file.stat().st_size,
                    'modified_time': sol_file.stat().st_mtime
                }
            )
            
            return contract
            
        except Exception as e:
            logger.warning(f"Error loading contract {sol_file}: {e}")
            return None
    
    def _read_file_with_encoding(self, file_path: Path) -> Optional[str]:
        """Read file with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        
        return None
    
    def _get_contract_labels(self, sol_file: Path) -> Tuple[List[str], bool]:
        """
        Get vulnerability labels for a contract.
        This is a placeholder - will be enhanced with actual label loading.
        """
        # For now, infer from filename/path
        path_str = str(sol_file).lower()
        vulnerabilities = []
        
        vuln_patterns = {
            'reentrancy': ['reentrancy', 'reentrant'],
            'access_control': ['access', 'control', 'owner'],
            'arithmetic': ['overflow', 'underflow', 'arithmetic'],
            'unchecked_calls': ['unchecked', 'call'],
            'dos': ['dos', 'denial'],
            'bad_randomness': ['random', 'timestamp', 'block']
        }
        
        for vuln_type, patterns in vuln_patterns.items():
            if any(pattern in path_str for pattern in patterns):
                vulnerabilities.append(vuln_type)
        
        is_vulnerable = len(vulnerabilities) > 0
        return vulnerabilities, is_vulnerable
    
    def _get_cached_contract(self, sol_file: Path) -> Optional[ContractData]:
        """Get contract from cache if available."""
        if not self.enable_caching:
            return None
        
        try:
            # Calculate file hash for cache key
            with open(sol_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute(
                    "SELECT filename, features, labels, metadata FROM contract_cache WHERE hash = ?",
                    (file_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Reconstruct contract from cache
                    filename, features_blob, labels_blob, metadata_blob = row
                    
                    # For now, return basic contract data
                    # Full feature caching will be implemented in feature extraction
                    return None  # Placeholder
        
        except Exception as e:
            logger.debug(f"Cache lookup failed for {sol_file}: {e}")
        
        return None
    
    def _cache_contract(self, contract: ContractData):
        """Cache contract data."""
        if not self.enable_caching:
            return
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO contract_cache 
                    (hash, filename, features, labels, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    contract.hash,
                    contract.filename,
                    pickle.dumps({}),  # Features placeholder
                    pickle.dumps({
                        'vulnerabilities': contract.vulnerabilities,
                        'is_vulnerable': contract.is_vulnerable
                    }),
                    pickle.dumps(contract.metadata)
                ))
                conn.commit()
        
        except Exception as e:
            logger.debug(f"Failed to cache contract {contract.filename}: {e}")
    
    def create_streaming_dataset(self, 
                                batch_size: int = 1000,
                                max_contracts: Optional[int] = None) -> Iterator[List[ContractData]]:
        """
        Create streaming dataset for memory-efficient processing.
        
        Args:
            batch_size: Size of each batch
            max_contracts: Maximum contracts to process
            
        Yields:
            Batches of ContractData objects
        """
        logger.info(f"🌊 Creating streaming dataset with batch size {batch_size}")
        
        structure = self.discover_dataset_structure()
        if not structure['contracts_dir']:
            raise ValueError("Contracts directory not found")
        
        contracts_dir = structure['contracts_dir']
        sol_files = list(contracts_dir.rglob('*.sol'))
        
        if max_contracts:
            sol_files = sol_files[:max_contracts]
        
        # Process in batches
        for i in range(0, len(sol_files), batch_size):
            batch_files = sol_files[i:i + batch_size]
            
            # Process batch
            batch_contracts = []
            for sol_file in batch_files:
                contract = self._load_single_contract(sol_file)
                if contract:
                    batch_contracts.append(contract)
            
            if batch_contracts:
                yield batch_contracts
            
            # Force garbage collection to manage memory
            gc.collect()
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        structure = self.discover_dataset_structure()
        
        stats = {
            'dataset_path': str(self.dataset_path),
            'total_contracts': structure['total_contracts'],
            'dataset_size_mb': structure['dataset_size_mb'],
            'available_tools': structure['available_tools'],
            'processing_stats': asdict(self.stats),
            'cache_efficiency': {
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'hit_rate': (self.stats.cache_hits / 
                           max(1, self.stats.cache_hits + self.stats.cache_misses))
            }
        }
        
        return stats
    
    def validate_dataset_quality(self, contracts: List[ContractData]) -> Dict:
        """
        Validate dataset quality and identify issues.
        
        Args:
            contracts: List of loaded contracts
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating dataset quality...")
        
        issues = {
            'empty_contracts': [],
            'duplicate_hashes': [],
            'encoding_issues': [],
            'missing_pragma': [],
            'very_large_contracts': [],
            'suspicious_contracts': []
        }
        
        seen_hashes = set()
        
        for contract in contracts:
            # Check for empty contracts
            if len(contract.code.strip()) < 50:
                issues['empty_contracts'].append(contract.filename)
            
            # Check for duplicates
            if contract.hash in seen_hashes:
                issues['duplicate_hashes'].append(contract.filename)
            seen_hashes.add(contract.hash)
            
            # Check for missing pragma
            if 'pragma solidity' not in contract.code.lower():
                issues['missing_pragma'].append(contract.filename)
            
            # Check for very large contracts (potential issues)
            if contract.size > 100000:  # 100KB
                issues['very_large_contracts'].append(contract.filename)
            
            # Check for suspicious patterns
            if any(pattern in contract.code.lower() for pattern in 
                  ['test', 'mock', 'example', 'demo']):
                issues['suspicious_contracts'].append(contract.filename)
        
        # Calculate quality metrics
        total_contracts = len(contracts)
        quality_score = 1.0
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                penalty = len(issue_list) / total_contracts * 0.1
                quality_score -= penalty
        
        quality_score = max(0.0, quality_score)
        
        validation_result = {
            'issues': issues,
            'quality_score': quality_score,
            'total_contracts': total_contracts,
            'valid_contracts': total_contracts - sum(len(v) for v in issues.values()),
            'recommendations': self._generate_quality_recommendations(issues)
        }
        
        logger.info(f"📊 Dataset quality score: {quality_score:.2f}")
        
        return validation_result
    
    def _generate_quality_recommendations(self, issues: Dict) -> List[str]:
        """Generate recommendations based on quality issues."""
        recommendations = []
        
        if issues['empty_contracts']:
            recommendations.append(
                f"Remove {len(issues['empty_contracts'])} empty contracts"
            )
        
        if issues['duplicate_hashes']:
            recommendations.append(
                f"Remove {len(issues['duplicate_hashes'])} duplicate contracts"
            )
        
        if issues['missing_pragma']:
            recommendations.append(
                f"Review {len(issues['missing_pragma'])} contracts without pragma"
            )
        
        if issues['very_large_contracts']:
            recommendations.append(
                f"Review {len(issues['very_large_contracts'])} very large contracts"
            )
        
        return recommendations
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("DELETE FROM contract_cache")
                conn.execute("DELETE FROM processing_cache")
                conn.commit()
            
            logger.info("🗑️ Cache cleared successfully")
        
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


def main():
    """Example usage of SmartBugsDataLoader."""
    # Configuration
    DATASET_PATH = "dataset/smartbugs-wild-master"
    
    # Initialize efficient loader
    loader = SmartBugsDataLoader(
        dataset_path=DATASET_PATH,
        max_workers=8,
        chunk_size=500,
        enable_caching=True
    )
    
    # Discover dataset structure
    structure = loader.discover_dataset_structure()
    print("📊 Dataset Structure:")
    for key, value in structure.items():
        print(f"  {key}: {value}")
    
    # Load contracts with parallel processing
    print("\n🚀 Loading contracts...")
    contracts = loader.load_contracts_parallel(max_contracts=1000)
    
    # Get statistics
    stats = loader.get_dataset_statistics()
    print("\n📈 Processing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Validate dataset quality
    validation = loader.validate_dataset_quality(contracts)
    print(f"\n✅ Dataset Quality Score: {validation['quality_score']:.2f}")
    
    # Example of streaming processing
    print("\n🌊 Streaming Dataset Example:")
    batch_count = 0
    for batch in loader.create_streaming_dataset(batch_size=100, max_contracts=500):
        batch_count += 1
        print(f"  Processed batch {batch_count} with {len(batch)} contracts")
        if batch_count >= 3:  # Limit for demo
            break
    
    print("\n🎉 Efficient data loading demonstration complete!")


if __name__ == "__main__":
    main()