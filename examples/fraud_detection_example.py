"""
Blockchain Fraud Detection - Complete Example

This example demonstrates the full workflow of the fraud detection module:
1. Load transaction data
2. Preprocess the data
3. Extract features
4. Train the model
5. Evaluate performance
6. Generate visualizations
7. Export results
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fraud_detection import (
    TransactionDataLoader,
    DataPreprocessor,
    TransactionFeatureExtractor,
    FraudDetector,
    ModelEvaluator,
    VisualizationEngine,
    ReportGenerator,
)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample transaction data for demonstration.
    
    Args:
        n_samples: Number of transactions to generate
        
    Returns:
        DataFrame with sample transaction data
    """
    np.random.seed(42)
    
    # Generate addresses
    n_addresses = 100
    addresses = [f"0x{i:040x}" for i in range(n_addresses)]
    
    data = {
        'sender': np.random.choice(addresses, n_samples),
        'receiver': np.random.choice(addresses, n_samples),
        'value': np.random.exponential(scale=10, size=n_samples),
        'gas_used': np.random.normal(loc=21000, scale=5000, size=n_samples).clip(min=21000),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1S').astype(np.int64) // 10**9,
    }
    
    df = pd.DataFrame(data)
    
    # Add labels (fraud/legitimate) - 10% fraud
    df['label'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    return df


def main():
    """Run the complete fraud detection workflow."""
    
    print("=" * 70)
    print("BLOCKCHAIN FRAUD DETECTION - COMPLETE EXAMPLE")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path("results/fraud_detection_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 1: Generate Sample Data ==========
    print("Step 1: Generating sample transaction data...")
    df = generate_sample_data(n_samples=1000)
    print(f"  Generated {len(df)} transactions")
    print(f"  Columns: {list(df.columns)}")
    print()
    
    # ========== STEP 2: Load and Validate Data ==========
    print("Step 2: Loading and validating data...")
    loader = TransactionDataLoader()
    
    # Save sample data to CSV for demonstration
    sample_csv = output_dir / "sample_transactions.csv"
    df.to_csv(sample_csv, index=False)
    
    # Load the data
    loaded_df = loader.load_and_validate(str(sample_csv))
    print(f"  Loaded {len(loaded_df)} transactions")
    print(f"  Data summary: {loader.get_data_summary()['rows']} rows")
    print()
    
    # ========== STEP 3: Preprocess Data ==========
    print("Step 3: Preprocessing data...")
    preprocessor = DataPreprocessor(missing_strategy='mean')
    
    # Separate features and labels
    X = loaded_df.drop('label', axis=1)
    y = loaded_df['label'].values
    
    # Preprocess
    X_preprocessed = preprocessor.preprocess(X, fit=True)
    print(f"  Preprocessed shape: {X_preprocessed.shape}")
    print(f"  Preprocessing info: {preprocessor.get_preprocessing_info()}")
    print()
    
    # ========== STEP 4: Extract Features ==========
    print("Step 4: Extracting features...")
    extractor = TransactionFeatureExtractor()
    X_features = extractor.extract_features(X)
    print(f"  Extracted {X_features.shape[1]} features")
    print(f"  Feature names: {extractor.get_feature_names()}")
    print()
    
    # ========== STEP 5: Split Data ==========
    print("Step 5: Splitting data into train/test...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print()
    
    # ========== STEP 6: Train Model ==========
    print("Step 6: Training fraud detection model...")
    detector = FraudDetector(n_estimators=100, max_depth=20)
    detector.set_feature_names(extractor.get_feature_names())
    
    train_result = detector.train(X_train.values, y_train)
    print(f"  Training status: {train_result['status']}")
    print(f"  Training accuracy: {train_result['train_accuracy']:.4f}")
    print(f"  Classes: {train_result['classes']}")
    print()
    
    # ========== STEP 7: Evaluate Model ==========
    print("Step 7: Evaluating model performance...")
    evaluator = ModelEvaluator()
    
    # Create a simple model wrapper for evaluation
    class ModelWrapper:
        def __init__(self, detector):
            self.detector = detector
        
        def predict(self, X):
            return self.detector.predict(X)
        
        def predict_proba(self, X):
            return self.detector.predict_proba(X)
    
    model_wrapper = ModelWrapper(detector)
    eval_results = evaluator.evaluate(model_wrapper, X_test.values, y_test)
    
    print(f"  Accuracy: {eval_results['metrics']['accuracy']:.4f}")
    print(f"  Precision: {eval_results['metrics']['precision']:.4f}")
    print(f"  Recall: {eval_results['metrics']['recall']:.4f}")
    print(f"  F1-Score: {eval_results['metrics']['f1_score']:.4f}")
    print(f"  ROC-AUC: {eval_results['metrics']['roc_auc']:.4f}")
    print(f"  Mean Latency: {eval_results['latency']['mean_ms']:.2f} ms")
    print()
    
    # ========== STEP 8: Generate Visualizations ==========
    print("Step 8: Generating visualizations...")
    visualizer = VisualizationEngine()
    
    # Confusion matrix
    cm = eval_results['confusion_matrix']
    visualizer.plot_confusion_matrix(np.array(cm))
    
    # ROC curve
    y_proba = detector.predict_proba(X_test.values)
    visualizer.plot_roc_curve(y_test, y_proba)
    
    # Feature importance
    feature_importance = detector.get_feature_importance()
    importances = np.array(list(feature_importance.values()))
    visualizer.plot_feature_importance(extractor.get_feature_names(), importances)
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(eval_results['metrics'])
    
    # Latency distribution
    visualizer.plot_latency_distribution(evaluator.latency_measurements)
    
    # Save all figures
    figures_dir = output_dir / "figures"
    saved_figures = visualizer.save_all_figures(str(figures_dir))
    print(f"  Saved {len(saved_figures)} figures to {figures_dir}")
    print()
    
    # ========== STEP 9: Export Results ==========
    print("Step 9: Exporting results...")
    reporter = ReportGenerator()
    
    # Export in all formats
    exported_files = reporter.export_all_formats(
        eval_results,
        str(output_dir),
        chart_paths=list(saved_figures.values())
    )
    
    print(f"  Exported files:")
    for format_name, filepath in exported_files.items():
        print(f"    - {format_name}: {filepath}")
    print()
    
    # ========== STEP 10: Print Summary Report ==========
    print("Step 10: Summary Report")
    print("-" * 70)
    print(reporter.generate_summary_report(eval_results))
    print()
    
    # ========== STEP 11: Save Model ==========
    print("Step 11: Saving trained model...")
    model_path = output_dir / "fraud_detector_model.pkl"
    detector.save_model(str(model_path))
    print(f"  Model saved to: {model_path}")
    print()
    
    print("=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
