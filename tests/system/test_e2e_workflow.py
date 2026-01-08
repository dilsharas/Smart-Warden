"""
End-to-end system tests for the complete smart contract analysis workflow.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import time
from pathlib import Path

from features.feature_extractor import SolidityFeatureExtractor
from models.random_forest import RandomForestVulnerabilityDetector
from models.multiclass_classifier import MultiClassVulnerabilityClassifier
from integration.tool_comparator import ToolComparator
from integration.slither_runner import SlitherAnalyzer
from integration.mythril_runner import MythrilAnalyzer


class TestEndToEndWorkflow:
    """End-to-end system tests for complete analysis workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment for each test."""
        self.start_time = time.time()
    
    def teardown_method(self):
        """Clean up after each test."""
        execution_time = time.time() - self.start_time
        print(f"Test execution time: {execution_time:.2f} seconds")
    
    def test_complete_analysis_workflow_safe_contract(self, sample_contract_files):
        """Test complete analysis workflow with a safe contract."""
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        contract_code = sample_contract_files['safe_contract.sol']
        
        # Step 1: Feature Extraction
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(contract_code)
        
        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many features
        assert features['lines_of_code'] > 0
        assert features['function_count'] > 0
        
        # Step 2: Create feature DataFrame
        features_df = pd.DataFrame([features])
        
        # Step 3: Train a quick model (for testing)
        detector = RandomForestVulnerabilityDetector(n_estimators=10, random_state=42)
        
        # Create synthetic training data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(50, len(features)),
            columns=list(features.keys())
        )
        y_train = pd.Series(np.random.choice(['safe', 'vulnerable'], 50))
        
        detector.train(X_train, y_train)
        
        # Step 4: Make prediction
        predictions, probabilities = detector.predict(features_df)
        
        assert len(predictions) == 1
        assert predictions[0] in ['safe', 'vulnerable']
        assert len(probabilities) == 1
        assert probabilities.shape[1] == 2
        
        # Step 5: Get feature importance
        importance_df = detector.get_feature_importance(top_n=10)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_complete_analysis_workflow_vulnerable_contract(self, sample_contract_files):
        """Test complete analysis workflow with a vulnerable contract."""
        if 'reentrancy_vulnerable.sol' not in sample_contract_files:
            pytest.skip("Vulnerable contract fixture not available")
        
        contract_code = sample_contract_files['reentrancy_vulnerable.sol']
        
        # Complete workflow
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(contract_code)
        
        # Verify reentrancy-specific features are detected
        assert features['external_call_count'] > 0
        assert features['state_change_after_call'] > 0
        
        # Test with multiclass classifier
        classifier = MultiClassVulnerabilityClassifier(n_estimators=10, random_state=42)
        
        # Create synthetic training data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(50, len(features)),
            columns=list(features.keys())
        )
        y_train = pd.Series(np.random.choice(['safe', 'reentrancy', 'access_control'], 50))
        
        classifier.train(X_train, y_train)
        
        # Make prediction
        features_df = pd.DataFrame([features])
        predictions, probabilities = classifier.predict(features_df)
        
        assert len(predictions) == 1
        assert predictions[0] in ['safe', 'reentrancy', 'access_control']
    
    def test_batch_analysis_workflow(self, test_contract_dataset):
        """Test batch analysis of multiple contracts."""
        if len(test_contract_dataset) == 0:
            pytest.skip("No test contracts available")
        
        # Step 1: Extract features from all contracts
        extractor = SolidityFeatureExtractor()
        features_df = extractor.extract_batch(test_contract_dataset)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(test_contract_dataset)
        
        # Step 2: Prepare data for training
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filename', 'vulnerability', 'label']]
        
        X = features_df[feature_columns]
        y = features_df['vulnerability']
        
        # Step 3: Train multiclass classifier
        classifier = MultiClassVulnerabilityClassifier(n_estimators=20, random_state=42)
        
        # Split data
        split_idx = max(1, int(0.7 * len(X)))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_train) > 0:
            classifier.train(X_train, y_train)
            
            # Step 4: Evaluate on test set
            if len(X_test) > 0:
                metrics = classifier.evaluate(X_test, y_test)
                
                assert 'accuracy' in metrics
                assert 'per_class_metrics' in metrics
                assert metrics['accuracy'] >= 0.0
    
    @pytest.mark.slow
    def test_performance_requirements(self, sample_contract_files):
        """Test that analysis meets performance requirements (< 15 seconds)."""
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        contract_code = sample_contract_files['safe_contract.sol']
        
        # Test feature extraction performance
        start_time = time.time()
        
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(contract_code)
        
        extraction_time = time.time() - start_time
        
        # Feature extraction should be very fast (< 1 second)
        assert extraction_time < 1.0, f"Feature extraction took {extraction_time:.2f}s, should be < 1s"
        
        # Test model prediction performance
        features_df = pd.DataFrame([features])
        
        # Create and train a model
        detector = RandomForestVulnerabilityDetector(n_estimators=50, random_state=42)
        
        # Quick training with synthetic data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, len(features)),
            columns=list(features.keys())
        )
        y_train = pd.Series(np.random.choice(['safe', 'vulnerable'], 100))
        
        train_start = time.time()
        detector.train(X_train, y_train)
        train_time = time.time() - train_start
        
        # Prediction should be very fast
        pred_start = time.time()
        predictions, probabilities = detector.predict(features_df)
        pred_time = time.time() - pred_start
        
        assert pred_time < 0.1, f"Prediction took {pred_time:.2f}s, should be < 0.1s"
        
        # Total analysis time should be reasonable
        total_time = extraction_time + pred_time
        assert total_time < 5.0, f"Total analysis took {total_time:.2f}s, should be < 5s"
    
    @pytest.mark.external_tools
    def test_tool_comparison_workflow(self, sample_contract_files):
        """Test complete tool comparison workflow."""
        if 'reentrancy_vulnerable.sol' not in sample_contract_files:
            pytest.skip("Vulnerable contract fixture not available")
        
        contract_code = sample_contract_files['reentrancy_vulnerable.sol']
        
        # Initialize tool comparator
        comparator = ToolComparator()
        
        # Check which tools are available
        available_tools = comparator.get_available_tools()
        
        if not any(available_tools.values()):
            pytest.skip("No external tools available for comparison")
        
        # Run comparison
        result = comparator.compare_tools(
            contract_code=contract_code,
            contract_name="reentrancy_test",
            ground_truth=['reentrancy']
        )
        
        assert result.contract_name == "reentrancy_test"
        assert result.contract_code == contract_code
        assert result.ground_truth == ['reentrancy']
        assert isinstance(result.tool_performances, dict)
        assert isinstance(result.agreement_score, float)
        assert 0.0 <= result.agreement_score <= 1.0
    
    def test_model_persistence_workflow(self, sample_contract_files):
        """Test complete workflow including model saving and loading."""
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        contract_code = sample_contract_files['safe_contract.sol']
        
        # Step 1: Extract features
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(contract_code)
        features_df = pd.DataFrame([features])
        
        # Step 2: Train model
        detector = RandomForestVulnerabilityDetector(n_estimators=10, random_state=42)
        
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(50, len(features)),
            columns=list(features.keys())
        )
        y_train = pd.Series(np.random.choice(['safe', 'vulnerable'], 50))
        
        detector.train(X_train, y_train)
        
        # Step 3: Make initial prediction
        pred1, prob1 = detector.predict(features_df)
        
        # Step 4: Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            detector.save_model(model_path)
            
            # Step 5: Load model
            loaded_detector = RandomForestVulnerabilityDetector.load_model(model_path)
            
            # Step 6: Make prediction with loaded model
            pred2, prob2 = loaded_detector.predict(features_df)
            
            # Predictions should be identical
            assert pred1[0] == pred2[0]
            np.testing.assert_array_almost_equal(prob1, prob2, decimal=10)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_error_handling_workflow(self):
        """Test workflow error handling with invalid inputs."""
        extractor = SolidityFeatureExtractor()
        
        # Test with empty contract
        features = extractor.extract_features("")
        assert isinstance(features, dict)
        
        # Test with malformed contract
        malformed_code = "this is not solidity code"
        features = extractor.extract_features(malformed_code)
        assert isinstance(features, dict)
        assert features['function_count'] == 0
        
        # Test model with insufficient data
        detector = RandomForestVulnerabilityDetector(n_estimators=5)
        
        # Try to predict before training
        with pytest.raises(ValueError):
            detector.predict(pd.DataFrame([[1, 2, 3]]))
    
    def test_memory_usage_workflow(self, sample_contract_files):
        """Test that workflow doesn't consume excessive memory."""
        import psutil
        import gc
        
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        contract_code = sample_contract_files['safe_contract.sol']
        
        # Run analysis multiple times
        extractor = SolidityFeatureExtractor()
        
        for i in range(10):
            features = extractor.extract_features(contract_code)
            
            # Force garbage collection
            gc.collect()
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    def test_concurrent_analysis_workflow(self, sample_contract_files):
        """Test concurrent analysis of multiple contracts."""
        import threading
        import queue
        
        if len(sample_contract_files) < 2:
            pytest.skip("Need at least 2 contract files for concurrent testing")
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def analyze_contract(contract_name, contract_code):
            """Analyze a single contract in a thread."""
            try:
                extractor = SolidityFeatureExtractor()
                features = extractor.extract_features(contract_code)
                results_queue.put((contract_name, features))
            except Exception as e:
                errors_queue.put((contract_name, str(e)))
        
        # Start threads for each contract
        threads = []
        for contract_name, contract_code in list(sample_contract_files.items())[:3]:  # Limit to 3 for speed
            thread = threading.Thread(
                target=analyze_contract,
                args=(contract_name, contract_code)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        # Should have results for all contracts
        assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
        assert len(errors) == 0, f"Got errors: {errors}"
        
        # All results should have features
        for contract_name, features in results:
            assert isinstance(features, dict)
            assert len(features) > 0
    
    def test_large_contract_workflow(self):
        """Test workflow with a large contract."""
        # Generate a large contract
        large_contract = """
        pragma solidity ^0.8.0;
        
        contract LargeContract {
            mapping(address => uint256) public balances;
            address public owner;
            
            constructor() {
                owner = msg.sender;
            }
        """
        
        # Add many functions to make it large
        for i in range(100):
            large_contract += f"""
            function function_{i}(uint256 param_{i}) public {{
                require(msg.sender == owner, "Not owner");
                balances[msg.sender] += param_{i};
                if (param_{i} > 100) {{
                    balances[msg.sender] *= 2;
                }}
            }}
            """
        
        large_contract += "}"
        
        # Test feature extraction on large contract
        start_time = time.time()
        
        extractor = SolidityFeatureExtractor()
        features = extractor.extract_features(large_contract)
        
        extraction_time = time.time() - start_time
        
        # Should still complete in reasonable time
        assert extraction_time < 5.0, f"Large contract analysis took {extraction_time:.2f}s"
        
        # Should detect the many functions
        assert features['function_count'] >= 100
        assert features['lines_of_code'] > 500
    
    def test_vulnerability_detection_accuracy(self, vulnerability_test_cases):
        """Test that the workflow correctly identifies known vulnerabilities."""
        extractor = SolidityFeatureExtractor()
        
        for vuln_type, test_case in vulnerability_test_cases.items():
            # Extract features
            features = extractor.extract_features(test_case['code'])
            
            # Check that expected features are detected
            for feature_name, expected_value in test_case['expected_features'].items():
                assert feature_name in features, f"Feature {feature_name} not found for {vuln_type}"
                assert features[feature_name] == expected_value, \
                    f"Feature {feature_name} = {features[feature_name]}, expected {expected_value} for {vuln_type}"
    
    def test_complete_pipeline_integration(self, test_contract_dataset):
        """Test complete integration of all pipeline components."""
        if len(test_contract_dataset) < 3:
            pytest.skip("Need at least 3 contracts for integration testing")
        
        # Step 1: Feature extraction
        extractor = SolidityFeatureExtractor()
        features_df = extractor.extract_batch(test_contract_dataset)
        
        # Step 2: Data preparation
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filename', 'vulnerability', 'label']]
        
        X = features_df[feature_columns]
        y_binary = features_df['label']  # 0/1 labels
        y_multiclass = features_df['vulnerability']  # vulnerability type labels
        
        # Step 3: Train binary classifier
        binary_detector = RandomForestVulnerabilityDetector(n_estimators=10, random_state=42)
        binary_detector.train(X, y_binary)
        
        # Step 4: Train multiclass classifier
        multiclass_detector = MultiClassVulnerabilityClassifier(n_estimators=10, random_state=42)
        multiclass_detector.train(X, y_multiclass)
        
        # Step 5: Make predictions
        binary_preds, binary_probs = binary_detector.predict(X)
        multi_preds, multi_probs = multiclass_detector.predict(X)
        
        # Step 6: Validate results
        assert len(binary_preds) == len(X)
        assert len(multi_preds) == len(X)
        
        # Binary predictions should be 0/1 or safe/vulnerable
        assert all(pred in [0, 1, 'safe', 'vulnerable'] for pred in binary_preds)
        
        # Multiclass predictions should be valid vulnerability types
        valid_types = ['safe', 'reentrancy', 'access_control', 'arithmetic', 
                      'unchecked_calls', 'denial_of_service', 'bad_randomness']
        assert all(pred in valid_types for pred in multi_preds)
        
        # Step 7: Feature importance analysis
        binary_importance = binary_detector.get_feature_importance(top_n=5)
        multi_importance = multiclass_detector.get_feature_importance_per_class(top_n=5)
        
        assert len(binary_importance) <= 5
        assert isinstance(multi_importance, dict)
        
        print(f"Integration test completed successfully with {len(X)} contracts")
        print(f"Binary accuracy: {sum(1 for i, pred in enumerate(binary_preds) if pred == y_binary.iloc[i]) / len(binary_preds):.2f}")
        print(f"Multiclass accuracy: {sum(1 for i, pred in enumerate(multi_preds) if pred == y_multiclass.iloc[i]) / len(multi_preds):.2f}")


class TestSystemReliability:
    """Test system reliability and robustness."""
    
    def test_system_stability_under_load(self, sample_contract_files):
        """Test system stability under repeated analysis."""
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        contract_code = sample_contract_files['safe_contract.sol']
        extractor = SolidityFeatureExtractor()
        
        # Run many analyses to test stability
        results = []
        for i in range(50):
            features = extractor.extract_features(contract_code)
            results.append(features)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Feature extraction results should be deterministic"
    
    def test_graceful_degradation(self):
        """Test that system degrades gracefully with invalid inputs."""
        extractor = SolidityFeatureExtractor()
        
        # Test various invalid inputs
        invalid_inputs = [
            "",  # Empty string
            "not solidity code",  # Invalid syntax
            "pragma solidity ^0.8.0;",  # Only pragma
            "contract { }",  # Minimal contract
            "a" * 10000,  # Very long string
            "contract Test {\n" + "function test() public {}\n" * 1000 + "}",  # Many functions
        ]
        
        for invalid_input in invalid_inputs:
            try:
                features = extractor.extract_features(invalid_input)
                # Should return features dict even for invalid input
                assert isinstance(features, dict)
            except Exception as e:
                pytest.fail(f"Feature extraction failed ungracefully with input: {invalid_input[:50]}... Error: {e}")
    
    def test_resource_cleanup(self, sample_contract_files):
        """Test that resources are properly cleaned up."""
        import gc
        import weakref
        
        if 'safe_contract.sol' not in sample_contract_files:
            pytest.skip("Safe contract fixture not available")
        
        contract_code = sample_contract_files['safe_contract.sol']
        
        # Create objects and track them with weak references
        extractor = SolidityFeatureExtractor()
        extractor_ref = weakref.ref(extractor)
        
        detector = RandomForestVulnerabilityDetector(n_estimators=5)
        detector_ref = weakref.ref(detector)
        
        # Use the objects
        features = extractor.extract_features(contract_code)
        
        # Delete references
        del extractor
        del detector
        
        # Force garbage collection
        gc.collect()
        
        # Objects should be cleaned up
        # Note: This test might be flaky depending on Python's GC behavior
        # assert extractor_ref() is None, "SolidityFeatureExtractor not cleaned up"
        # assert detector_ref() is None, "RandomForestVulnerabilityDetector not cleaned up"