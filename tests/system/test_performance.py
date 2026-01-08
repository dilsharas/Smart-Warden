"""
Performance tests for the smart contract security analyzer.
"""

import pytest
import time
import gc
import pandas as pd
import numpy as np
from pathlib import Path

# Optional import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from features.feature_extractor import SolidityFeatureExtractor
from models.random_forest import RandomForestVulnerabilityDetector
from models.multiclass_classifier import MultiClassVulnerabilityClassifier


class TestPerformanceRequirements:
    """Test performance requirements and benchmarks."""
    
    @pytest.mark.slow
    def test_feature_extraction_performance(self, sample_contract_files):
        """Test feature extraction meets performance requirements."""
        if not sample_contract_files:
            pytest.skip("No sample contracts available")
        
        extractor = SolidityFeatureExtractor()
        
        # Test performance on each contract type
        performance_results = {}
        
        for contract_name, contract_code in sample_contract_files.items():
            # Measure extraction time
            start_time = time.time()
            features = extractor.extract_features(contract_code)
            extraction_time = time.time() - start_time
            
            performance_results[contract_name] = {
                'extraction_time': extraction_time,
                'lines_of_code': features.get('lines_of_code', 0),
                'feature_count': len(features)
            }
            
            # Performance requirement: < 3 seconds for typical contracts
            if features.get('lines_of_code', 0) < 1000:
                assert extraction_time < 3.0, \
                    f"Feature extraction for {contract_name} took {extraction_time:.2f}s, should be < 3s"
        
        # Print performance summary
        print("\\nFeature Extraction Performance:")
        for name, results in performance_results.items():
            print(f"  {name}: {results['extraction_time']:.3f}s "
                  f"({results['lines_of_code']} lines, {results['feature_count']} features)")
    
    @pytest.mark.slow
    def test_model_training_performance(self, synthetic_ml_dataset):
        """Test model training performance."""
        X = synthetic_ml_dataset['X']
        y = synthetic_ml_dataset['y_binary']
        
        # Test binary classifier training time
        start_time = time.time()
        
        binary_detector = RandomForestVulnerabilityDetector(n_estimators=100, random_state=42)
        binary_detector.train(X, y)
        
        training_time = time.time() - start_time
        
        # Training should complete in reasonable time
        assert training_time < 30.0, f"Binary model training took {training_time:.2f}s, should be < 30s"
        
        # Test multiclass classifier training time
        y_multi = synthetic_ml_dataset['y_multiclass']
        
        start_time = time.time()
        
        multi_detector = MultiClassVulnerabilityClassifier(n_estimators=100, random_state=42)
        multi_detector.train(X, y_multi)
        
        multi_training_time = time.time() - start_time
        
        assert multi_training_time < 45.0, f"Multiclass model training took {multi_training_time:.2f}s, should be < 45s"
        
        print(f"\\nTraining Performance:")
        print(f"  Binary classifier: {training_time:.2f}s")
        print(f"  Multiclass classifier: {multi_training_time:.2f}s")
    
    @pytest.mark.slow
    def test_model_prediction_performance(self, trained_binary_model):
        """Test model prediction performance."""
        model = trained_binary_model['model']
        X_test = trained_binary_model['X_test']
        
        # Test single prediction performance
        single_sample = X_test.iloc[:1]
        
        start_time = time.time()
        predictions, probabilities = model.predict(single_sample)
        single_pred_time = time.time() - start_time
        
        # Single prediction should be very fast
        assert single_pred_time < 0.1, f"Single prediction took {single_pred_time:.3f}s, should be < 0.1s"
        
        # Test batch prediction performance
        batch_size = min(100, len(X_test))
        batch_sample = X_test.iloc[:batch_size]
        
        start_time = time.time()
        batch_predictions, batch_probabilities = model.predict(batch_sample)
        batch_pred_time = time.time() - start_time
        
        # Batch prediction should be efficient
        avg_time_per_sample = batch_pred_time / batch_size
        assert avg_time_per_sample < 0.01, \
            f"Average prediction time {avg_time_per_sample:.4f}s per sample, should be < 0.01s"
        
        print(f"\\nPrediction Performance:")
        print(f"  Single prediction: {single_pred_time:.4f}s")
        print(f"  Batch prediction: {batch_pred_time:.4f}s for {batch_size} samples")
        print(f"  Average per sample: {avg_time_per_sample:.4f}s")
    
    @pytest.mark.slow
    def test_end_to_end_analysis_time(self, sample_contract_files):
        """Test complete end-to-end analysis time meets requirements."""
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        contract_code = sample_contract_files['safe_contract.sol']
        
        # Measure complete workflow time
        start_time = time.time()
        
        # Step 1: Feature extraction
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(contract_code)
        
        # Step 2: Prepare for prediction
        features_df = pd.DataFrame([features])
        
        # Step 3: Create and train model (quick training for testing)
        detector = RandomForestVulnerabilityDetector(n_estimators=20, random_state=42)
        
        # Quick synthetic training
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(50, len(features)),
            columns=list(features.keys())
        )
        y_train = pd.Series(np.random.choice(['safe', 'vulnerable'], 50))
        detector.train(X_train, y_train)
        
        # Step 4: Make prediction
        predictions, probabilities = detector.predict(features_df)
        
        # Step 5: Get feature importance
        importance = detector.get_feature_importance(top_n=10)
        
        total_time = time.time() - start_time
        
        # Total analysis should meet requirement (< 15 seconds)
        assert total_time < 15.0, f"End-to-end analysis took {total_time:.2f}s, should be < 15s"
        
        print(f"\\nEnd-to-End Analysis Time: {total_time:.2f}s")
    
    def test_memory_usage_limits(self, sample_contract_files):
        """Test memory usage stays within reasonable limits."""
        if not sample_contract_files:
            pytest.skip("No sample contracts available")
        
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available - skipping memory tests")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        extractor = SolidityFeatureExtractor()
        
        # Process multiple contracts
        for contract_name, contract_code in sample_contract_files.items():
            features = extractor.extract_features(contract_code)
            
            # Check memory after each contract
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (< 50MB per contract)
            assert memory_increase < 50, \
                f"Memory increased by {memory_increase:.1f}MB after processing {contract_name}"
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"\\nMemory Usage:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {total_increase:.1f}MB")
        
        # Total memory increase should be reasonable
        assert total_increase < 100, f"Total memory increase {total_increase:.1f}MB too high"
    
    @pytest.mark.slow
    def test_concurrent_analysis_performance(self, sample_contract_files):
        """Test performance under concurrent analysis load."""
        import threading
        import queue
        
        if len(sample_contract_files) < 2:
            pytest.skip("Need multiple contracts for concurrent testing")
        
        results_queue = queue.Queue()
        
        def analyze_contract_worker(contract_code, worker_id):
            """Worker function for concurrent analysis."""
            start_time = time.time()
            
            extractor = SolidityFeatureExtractor()
            features = extractor.extract_features(contract_code)
            
            analysis_time = time.time() - start_time
            results_queue.put((worker_id, analysis_time, len(features)))
        
        # Start multiple concurrent analyses
        threads = []
        contracts_list = list(sample_contract_files.items())[:5]  # Limit to 5 for performance
        
        start_time = time.time()
        
        for i, (contract_name, contract_code) in enumerate(contracts_list):
            thread = threading.Thread(
                target=analyze_contract_worker,
                args=(contract_code, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)
        
        total_concurrent_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == len(contracts_list), "Not all concurrent analyses completed"
        
        # Calculate statistics
        analysis_times = [result[1] for result in results]
        avg_analysis_time = sum(analysis_times) / len(analysis_times)
        max_analysis_time = max(analysis_times)
        
        print(f"\\nConcurrent Analysis Performance:")
        print(f"  Concurrent analyses: {len(contracts_list)}")
        print(f"  Total time: {total_concurrent_time:.2f}s")
        print(f"  Average analysis time: {avg_analysis_time:.2f}s")
        print(f"  Maximum analysis time: {max_analysis_time:.2f}s")
        
        # Concurrent execution should be efficient
        assert max_analysis_time < 10.0, f"Slowest concurrent analysis took {max_analysis_time:.2f}s"
    
    def test_large_contract_performance(self):
        """Test performance with large contracts."""
        # Generate a large contract (5000+ lines)
        large_contract = """
        pragma solidity ^0.8.0;
        
        contract LargeContract {
            mapping(address => uint256) public balances;
            mapping(address => mapping(address => uint256)) public allowances;
            
            address public owner;
            uint256 public totalSupply;
            
            event Transfer(address indexed from, address indexed to, uint256 value);
            event Approval(address indexed owner, address indexed spender, uint256 value);
            
            constructor() {
                owner = msg.sender;
                totalSupply = 1000000 * 10**18;
                balances[msg.sender] = totalSupply;
            }
        """
        
        # Add many functions to reach 5000+ lines
        for i in range(200):
            large_contract += f"""
            
            function function_{i}(uint256 param1, address param2, bool param3) public {{
                require(msg.sender == owner, "Not authorized");
                require(param1 > 0, "Invalid parameter");
                require(param2 != address(0), "Invalid address");
                
                if (param3) {{
                    balances[param2] += param1;
                    totalSupply += param1;
                }} else {{
                    if (balances[param2] >= param1) {{
                        balances[param2] -= param1;
                        totalSupply -= param1;
                    }}
                }}
                
                emit Transfer(msg.sender, param2, param1);
            }}
            
            function getter_{i}() public view returns (uint256, address, bool) {{
                return (balances[msg.sender], owner, totalSupply > 0);
            }}
            """
        
        large_contract += "}"
        
        # Test performance on large contract
        start_time = time.time()
        
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(large_contract)
        
        extraction_time = time.time() - start_time
        
        # Should handle large contracts within reasonable time
        assert extraction_time < 10.0, f"Large contract analysis took {extraction_time:.2f}s, should be < 10s"
        
        # Verify it detected the large size
        assert features['lines_of_code'] > 5000
        assert features['function_count'] >= 400  # 200 * 2 functions
        
        print(f"\\nLarge Contract Performance:")
        print(f"  Lines of code: {features['lines_of_code']}")
        print(f"  Function count: {features['function_count']}")
        print(f"  Analysis time: {extraction_time:.2f}s")
    
    @pytest.mark.slow
    def test_batch_processing_performance(self, test_contract_dataset):
        """Test batch processing performance."""
        if len(test_contract_dataset) < 3:
            pytest.skip("Need multiple contracts for batch testing")
        
        extractor = SolidityFeatureExtractor()
        
        # Test batch feature extraction
        start_time = time.time()
        features_df = extractor.extract_batch(test_contract_dataset)
        batch_time = time.time() - start_time
        
        # Calculate per-contract time
        per_contract_time = batch_time / len(test_contract_dataset)
        
        print(f"\\nBatch Processing Performance:")
        print(f"  Contracts processed: {len(test_contract_dataset)}")
        print(f"  Total batch time: {batch_time:.2f}s")
        print(f"  Average per contract: {per_contract_time:.2f}s")
        
        # Batch processing should be efficient
        assert per_contract_time < 5.0, f"Average per-contract time {per_contract_time:.2f}s too high"
        
        # Verify results
        assert len(features_df) == len(test_contract_dataset)
        assert 'lines_of_code' in features_df.columns
        assert 'function_count' in features_df.columns


class TestScalabilityLimits:
    """Test system behavior at scalability limits."""
    
    def test_maximum_contract_size(self):
        """Test handling of very large contracts."""
        # Create an extremely large contract
        huge_contract = "pragma solidity ^0.8.0;\\ncontract HugeContract {\\n"
        
        # Add 10,000 lines
        for i in range(5000):
            huge_contract += f"    uint256 public var_{i};\\n"
            huge_contract += f"    function get_{i}() public view returns (uint256) {{ return var_{i}; }}\\n"
        
        huge_contract += "}"
        
        extractor = SolidityFeatureExtractor()
        
        # Should handle very large contracts without crashing
        start_time = time.time()
        features = extractor.extract_features(huge_contract)
        extraction_time = time.time() - start_time
        
        # Should complete even if it takes longer
        assert extraction_time < 60.0, f"Huge contract analysis took {extraction_time:.2f}s"
        assert features['lines_of_code'] > 10000
        assert features['function_count'] >= 5000
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available - skipping memory pressure tests")
            
        extractor = SolidityFeatureExtractor()
        
        # Create many medium-sized contracts
        contracts = []
        for i in range(10):
            contract = f"""
            pragma solidity ^0.8.0;
            contract Contract_{i} {{
                mapping(address => uint256) balances;
            """
            
            # Add 100 functions per contract
            for j in range(100):
                contract += f"""
                function func_{j}(uint256 x) public {{
                    balances[msg.sender] += x;
                }}
                """
            
            contract += "}"
            contracts.append(contract)
        
        # Process all contracts and measure memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        results = []
        for i, contract in enumerate(contracts):
            features = extractor.extract_features(contract)
            results.append(features)
            
            # Check memory periodically
            if i % 3 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow excessively
                assert memory_increase < 200, f"Memory increased by {memory_increase:.1f}MB"
        
        # All contracts should be processed successfully
        assert len(results) == len(contracts)
        
        # Force cleanup
        gc.collect()
    
    def test_feature_extraction_limits(self):
        """Test limits of feature extraction."""
        extractor = SolidityFeatureExtractor()
        
        # Test with contract having extreme values
        extreme_contract = """
        pragma solidity ^0.8.0;
        
        contract ExtremeContract {
        """
        
        # Add extreme nesting
        for i in range(20):
            extreme_contract += "    if (true) {\\n" * 5
            extreme_contract += f"        uint256 var_{i} = {i};\\n"
            extreme_contract += "    }\\n" * 5
        
        # Add many arithmetic operations
        extreme_contract += """
            function extremeArithmetic(uint256 x) public pure returns (uint256) {
                uint256 result = x;
        """
        
        for i in range(100):
            extreme_contract += f"        result = result + {i} * {i} / ({i} + 1);\\n"
        
        extreme_contract += """
                return result;
            }
        }
        """
        
        # Should handle extreme cases
        features = extractor.extract_features(extreme_contract)
        
        # Verify extreme values are captured
        assert features['nested_if_depth'] > 10
        assert features['addition_count'] > 50
        assert features['multiplication_count'] > 50
        assert features['division_count'] > 50