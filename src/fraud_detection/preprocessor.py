"""
Data Preprocessing Module

Handles data cleaning, normalization, and encoding for transaction data.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from .config import NORMALIZATION_RANGE, MISSING_VALUE_STRATEGY

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses transaction data for machine learning.
    
    Handles missing values, normalization, and categorical encoding.
    """

    def __init__(self, missing_strategy: str = MISSING_VALUE_STRATEGY):
        """
        Initialize the preprocessor.
        
        Args:
            missing_strategy: Strategy for handling missing values ("mean", "median", "drop")
        """
        self.missing_strategy = missing_strategy
        self.scaler = MinMaxScaler(feature_range=NORMALIZATION_RANGE)
        self.label_encoders = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.fitted = False

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        initial_rows = len(df)
        
        if self.missing_strategy == "drop":
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        elif self.missing_strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with mean: {mean_val:.4f}")
        
        elif self.missing_strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val:.4f}")
        
        return df

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features to [0, 1] range.
        
        Args:
            df: DataFrame with numerical features
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            DataFrame with normalized features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_columns = numeric_cols
        
        if fit:
            df_normalized = df.copy()
            df_normalized[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info(f"Fitted and normalized {len(numeric_cols)} numerical features")
        else:
            df_normalized = df.copy()
            df_normalized[numeric_cols] = self.scaler.transform(df[numeric_cols])
            logger.info(f"Normalized {len(numeric_cols)} numerical features using fitted scaler")
        
        return df_normalized

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features to numerical values.
        
        Args:
            df: DataFrame with categorical features
            fit: Whether to fit the encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded categorical features
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_columns = categorical_cols
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
                logger.info(f"Encoded categorical feature {col} with {len(encoder.classes_)} classes")
            else:
                if col not in self.label_encoders:
                    raise ValueError(f"Encoder for column {col} not found. Fit the preprocessor first.")
                encoder = self.label_encoders[col]
                df_encoded[col] = encoder.transform(df[col].astype(str))
        
        return df_encoded

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the data.
        
        Args:
            df: Raw DataFrame
            fit: Whether to fit transformers (True for training, False for test)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Starting preprocessing with {len(df)} rows")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Encode categorical features
        df = self.encode_categorical(df, fit=fit)
        
        # Step 3: Normalize numerical features
        df = self.normalize_features(df, fit=fit)
        
        self.fitted = True
        logger.info(f"Preprocessing complete. Output shape: {df.shape}")
        
        return df

    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the preprocessor state for later use.
        
        Args:
            filepath: Path to save the preprocessor
        """
        import joblib
        
        state = {
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "missing_strategy": self.missing_strategy,
            "fitted": self.fitted,
        }
        
        joblib.dump(state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a previously saved preprocessor state.
        
        Args:
            filepath: Path to load the preprocessor from
        """
        import joblib
        
        state = joblib.load(filepath)
        self.scaler = state["scaler"]
        self.label_encoders = state["label_encoders"]
        self.numeric_columns = state["numeric_columns"]
        self.categorical_columns = state["categorical_columns"]
        self.missing_strategy = state["missing_strategy"]
        self.fitted = state["fitted"]
        
        logger.info(f"Preprocessor loaded from {filepath}")

    def get_preprocessing_info(self) -> dict:
        """
        Get information about the preprocessing configuration.
        
        Returns:
            Dictionary with preprocessing information
        """
        return {
            "fitted": self.fitted,
            "missing_strategy": self.missing_strategy,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "normalization_range": NORMALIZATION_RANGE,
            "num_label_encoders": len(self.label_encoders),
        }
