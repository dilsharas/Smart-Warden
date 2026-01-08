"""
Configuration constants for the Blockchain Fraud Detection Module
"""

# Data Loading Configuration
REQUIRED_FIELDS = ["sender", "receiver", "value", "gas_used", "timestamp"]
OPTIONAL_FIELDS = ["label", "transaction_hash"]

# Feature Extraction Configuration
FEATURE_COUNT = 15
FEATURE_NAMES = [
    "transaction_value",
    "value_mean",
    "value_std",
    "transaction_frequency",
    "avg_gas_used",
    "gas_anomaly_score",
    "time_interval_mean",
    "time_interval_std",
    "sender_activity_level",
    "receiver_activity_level",
    "value_to_gas_ratio",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "sender_receiver_interaction_count",
]

# Normalization Configuration
NORMALIZATION_RANGE = (0, 1)
MISSING_VALUE_STRATEGY = "mean"  # Options: "mean", "median", "drop"

# Model Training Configuration
MODEL_TYPE = "RandomForest"
N_ESTIMATORS = 100
MAX_DEPTH = 20
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42
MIN_ACCURACY = 0.85

# Evaluation Configuration
LATENCY_THRESHOLD_MS = 50
LATENCY_ITERATIONS = 100

# Visualization Configuration
FIGURE_DPI = 100
FIGURE_SIZE = (10, 6)
COLORMAP = "viridis"

# Export Configuration
EXPORT_FORMATS = ["json", "csv", "pdf"]
RESULTS_DIR = "results/fraud_detection"
MODELS_DIR = "models/fraud_detection"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/fraud_detection.log"
