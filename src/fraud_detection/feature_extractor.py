"""
Feature Extraction Module

Extracts transaction-level features from raw blockchain transaction data.
"""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import FEATURE_NAMES, FEATURE_COUNT

logger = logging.getLogger(__name__)


class TransactionFeatureExtractor:
    """
    Extracts meaningful features from raw blockchain transaction data.
    
    Generates 15+ numerical features suitable for machine learning models.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = FEATURE_NAMES.copy()
        self.feature_count = FEATURE_COUNT
        self.fitted = False

    def extract_transaction_value_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transaction value statistics.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with value statistics features
        """
        features = pd.DataFrame(index=df.index)
        
        # Raw transaction value
        features['transaction_value'] = df['value'].astype(float)
        
        # Value statistics per sender
        sender_value_stats = df.groupby('sender')['value'].agg(['mean', 'std']).reset_index()
        sender_value_stats.columns = ['sender', 'value_mean', 'value_std']
        sender_value_stats['value_std'] = sender_value_stats['value_std'].fillna(0)
        
        features = features.merge(sender_value_stats, left_on=df['sender'], right_on='sender', how='left')
        features = features.drop('sender', axis=1)
        
        logger.info("Extracted transaction value statistics")
        return features

    def extract_transaction_frequency(self, df: pd.DataFrame, time_window_hours: int = 24) -> pd.DataFrame:
        """
        Extract transaction frequency per address.
        
        Args:
            df: DataFrame with transaction data
            time_window_hours: Time window for frequency calculation
            
        Returns:
            DataFrame with frequency features
        """
        features = pd.DataFrame(index=df.index)
        
        # Convert timestamp to datetime if needed
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate transaction frequency per sender
        sender_freq = df.groupby('sender').size().reset_index(name='transaction_frequency')
        features = features.merge(sender_freq, left_on=df['sender'], right_on='sender', how='left')
        features.drop('sender', axis=1, inplace=True)
        
        logger.info("Extracted transaction frequency features")
        return features

    def extract_gas_usage_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract gas usage pattern features.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with gas usage features
        """
        features = pd.DataFrame(index=df.index)
        
        # Average gas used per sender
        sender_gas_stats = df.groupby('sender')['gas_used'].agg(['mean']).reset_index()
        sender_gas_stats.columns = ['sender', 'avg_gas_used']
        
        features = features.merge(sender_gas_stats, left_on=df['sender'], right_on='sender', how='left')
        features.drop('sender', axis=1, inplace=True)
        
        # Gas anomaly score (deviation from mean)
        global_gas_mean = df['gas_used'].mean()
        features['gas_anomaly_score'] = np.abs(df['gas_used'].astype(float) - global_gas_mean) / (global_gas_mean + 1e-6)
        
        logger.info("Extracted gas usage pattern features")
        return features

    def extract_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time interval features between transactions.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with time interval features
        """
        features = pd.DataFrame(index=df.index)
        
        # Create a copy and convert timestamp to datetime
        df_copy = df.copy()
        
        # Convert timestamp to datetime if needed
        if df_copy['timestamp'].dtype != 'datetime64[ns]':
            try:
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='s')
            except:
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Sort by sender and timestamp
        df_sorted = df_copy.sort_values(['sender', 'timestamp']).reset_index(drop=True)
        
        # Calculate time intervals in seconds
        df_sorted['time_diff'] = df_sorted.groupby('sender')['timestamp'].diff().dt.total_seconds()
        
        # Time interval statistics per sender
        time_stats = df_sorted.groupby('sender')['time_diff'].agg(['mean', 'std']).reset_index()
        time_stats.columns = ['sender', 'time_interval_mean', 'time_interval_std']
        time_stats['time_interval_std'] = time_stats['time_interval_std'].fillna(0)
        
        # Merge with original dataframe
        df_with_sender = df.copy()
        df_with_sender['sender_key'] = df_with_sender['sender']
        features = features.merge(time_stats, left_on=df_with_sender['sender_key'], right_on='sender', how='left')
        features = features.drop('sender', axis=1)
        
        logger.info("Extracted time interval features")
        return features

    def extract_activity_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sender and receiver activity levels.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with activity level features
        """
        features = pd.DataFrame(index=df.index)
        
        # Sender activity level (number of transactions as sender)
        sender_activity = df.groupby('sender').size().reset_index(name='sender_activity_level')
        features = features.merge(sender_activity, left_on=df['sender'], right_on='sender', how='left')
        features.drop('sender', axis=1, inplace=True)
        
        # Receiver activity level (number of transactions as receiver)
        receiver_activity = df.groupby('receiver').size().reset_index(name='receiver_activity_level')
        features = features.merge(receiver_activity, left_on=df['receiver'], right_on='receiver', how='left')
        features.drop('receiver', axis=1, inplace=True)
        
        logger.info("Extracted activity level features")
        return features

    def extract_value_to_gas_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract value to gas ratio features.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with value-to-gas ratio features
        """
        features = pd.DataFrame(index=df.index)
        
        # Avoid division by zero
        features['value_to_gas_ratio'] = df['value'].astype(float) / (df['gas_used'].astype(float) + 1e-6)
        
        logger.info("Extracted value-to-gas ratio features")
        return features

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamps.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with temporal features
        """
        features = pd.DataFrame(index=df.index)
        
        # Convert timestamp to datetime if needed
        if df['timestamp'].dtype != 'datetime64[ns]':
            try:
                timestamps = pd.to_datetime(df['timestamp'], unit='s')
            except:
                timestamps = pd.to_datetime(df['timestamp'])
        else:
            timestamps = df['timestamp']
        
        # Hour of day
        features['hour_of_day'] = timestamps.dt.hour
        
        # Day of week
        features['day_of_week'] = timestamps.dt.dayofweek
        
        # Is weekend
        features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
        
        logger.info("Extracted temporal features")
        return features

    def extract_interaction_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sender-receiver interaction count features.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with interaction count features
        """
        features = pd.DataFrame(index=df.index)
        
        # Count prior interactions between sender and receiver
        interaction_counts = df.groupby(['sender', 'receiver']).cumcount().reset_index(drop=True)
        features['sender_receiver_interaction_count'] = interaction_counts
        
        logger.info("Extracted sender-receiver interaction count features")
        return features

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from transaction data.
        
        Args:
            df: DataFrame with raw transaction data
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Starting feature extraction for {len(df)} transactions")
        
        features = pd.DataFrame(index=df.index)
        
        # Extract all feature groups
        features = pd.concat([
            features,
            self.extract_transaction_value_stats(df),
            self.extract_transaction_frequency(df),
            self.extract_gas_usage_patterns(df),
            self.extract_time_intervals(df),
            self.extract_activity_levels(df),
            self.extract_value_to_gas_ratio(df),
            self.extract_temporal_features(df),
            self.extract_interaction_count(df),
        ], axis=1)
        
        # Ensure we have the expected number of features
        if len(features.columns) != self.feature_count:
            logger.warning(
                f"Expected {self.feature_count} features but got {len(features.columns)}. "
                f"Columns: {list(features.columns)}"
            )
        
        # Fill any remaining NaN values with 0
        features.fillna(0, inplace=True)
        
        logger.info(f"Feature extraction complete. Shape: {features.shape}")
        self.fitted = True
        
        return features

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()

    def get_feature_count(self) -> int:
        """
        Get the number of features.
        
        Returns:
            Number of features
        """
        return self.feature_count
