#!/usr/bin/env python3
"""
Automated Feature Selection and Dimensionality Reduction System.
Implements multiple feature selection methods with optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, mutual_info_classif, chi2
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""
    method: str = 'combined'  # 'univariate', 'rfe', 'importance', 'combined'
    k_features: int = 50
    percentile: float = 80
    use_pca: bool = False
    pca_components: int = 30
    cross_validation: bool = True
    cv_folds: int = 5

class AutomatedFeatureSelector:
    """
    Automated feature selection with multiple methods and optimization.
    """
    
    def __init__(self, config: FeatureSelectionConfig = None):
        self.config = config or FeatureSelectionConfig()
        self.selectors = {}
        self.selected_features = None
        self.feature_scores = None
        
    def select_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Select optimal features using configured method.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"🎯 Selecting features using {self.config.method} method...")
        
        if self.config.method == 'univariate':
            return self._univariate_selection(X, y)
        elif self.config.method == 'rfe':
            return self._rfe_selection(X, y)
        elif self.config.method == 'importance':
            return self._importance_selection(X, y)
        elif self.config.method == 'combined':
            return self._combined_selection(X, y)
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")
    
    def _univariate_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Univariate feature selection."""
        selector = SelectKBest(score_func=f_classif, k=self.config.k_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        self.selectors['univariate'] = selector
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index) 
   
    def _rfe_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Recursive feature elimination."""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        if self.config.cross_validation:
            selector = RFECV(estimator, cv=self.config.cv_folds, scoring='accuracy')
        else:
            selector = RFE(estimator, n_features_to_select=self.config.k_features)
        
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        self.selectors['rfe'] = selector
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def _importance_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Feature importance based selection."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top k features
        self.selected_features = feature_importance.head(self.config.k_features)['feature'].tolist()
        self.feature_scores = feature_importance
        
        return X[self.selected_features]
    
    def _combined_selection(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Combined feature selection using multiple methods."""
        # Method 1: Univariate selection
        univariate_selector = SelectKBest(score_func=f_classif, k=min(100, len(X.columns)))
        univariate_selector.fit(X, y)
        univariate_features = set(X.columns[univariate_selector.get_support()])
        
        # Method 2: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        rf_features = set(rf_importance.head(min(100, len(X.columns)))['feature'])
        
        # Method 3: Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        mi_features = set(mi_importance.head(min(100, len(X.columns)))['feature'])
        
        # Combine methods - features that appear in at least 2 methods
        feature_votes = {}
        for feature in X.columns:
            votes = 0
            if feature in univariate_features:
                votes += 1
            if feature in rf_features:
                votes += 1
            if feature in mi_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes, then top by RF importance
        candidate_features = [f for f, votes in feature_votes.items() if votes >= 2]
        
        if len(candidate_features) > self.config.k_features:
            # Rank by RF importance
            rf_ranking = {row['feature']: row['rf_importance'] 
                         for _, row in rf_importance.iterrows()}
            candidate_features.sort(key=lambda x: rf_ranking.get(x, 0), reverse=True)
            self.selected_features = candidate_features[:self.config.k_features]
        else:
            # If not enough candidates, add top RF features
            remaining_needed = self.config.k_features - len(candidate_features)
            additional_features = [f for f in rf_importance['feature'] 
                                 if f not in candidate_features][:remaining_needed]
            self.selected_features = candidate_features + additional_features
        
        # Store feature scores
        self.feature_scores = pd.DataFrame({
            'feature': self.selected_features,
            'rf_importance': [rf_ranking.get(f, 0) for f in self.selected_features],
            'votes': [feature_votes.get(f, 0) for f in self.selected_features]
        })
        
        return X[self.selected_features]
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """Get ranking of selected features."""
        if self.feature_scores is not None:
            return self.feature_scores
        else:
            return pd.DataFrame({'feature': self.selected_features or []})
    
    def save_selector(self, filepath: str):
        """Save the feature selector."""
        joblib.dump({
            'config': self.config,
            'selectors': self.selectors,
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores
        }, filepath)
    
    def load_selector(self, filepath: str):
        """Load a saved feature selector."""
        data = joblib.load(filepath)
        self.config = data['config']
        self.selectors = data['selectors']
        self.selected_features = data['selected_features']
        self.feature_scores = data['feature_scores']


def main():
    """Example usage."""
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 100), 
                     columns=[f'feature_{i}' for i in range(100)])
    y = np.random.randint(0, 2, 1000)
    
    # Configure and run feature selection
    config = FeatureSelectionConfig(method='combined', k_features=20)
    selector = AutomatedFeatureSelector(config)
    
    X_selected = selector.select_features(X, y)
    print(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
    
    # Get feature ranking
    ranking = selector.get_feature_ranking()
    print("\nTop 10 features:")
    print(ranking.head(10))


if __name__ == "__main__":
    main()