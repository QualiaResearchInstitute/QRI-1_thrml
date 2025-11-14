"""
CRV Diagnostic Classifier: Predicts step correctness from structural features.

Trains classifier on structural fingerprints extracted from attribution graphs.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GradientBoostingClassifier = None
    LogisticRegression = None
    RandomForestClassifier = None


class CRVDiagnosticClassifier:
    """
    Trains classifier to predict step correctness from structural features.
    
    Uses Gradient Boosting (like CRV paper) but supports multiple classifier types.
    """
    
    def __init__(
        self,
        classifier_type: str = "gradient_boosting",
        **classifier_kwargs,
    ):
        """
        Initialize diagnostic classifier.
        
        Args:
            classifier_type: Type of classifier ("gradient_boosting", "logistic_regression", "random_forest")
            **classifier_kwargs: Additional arguments for classifier
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for CRV classifier. Install with: pip install scikit-learn")
        
        self.classifier_type = classifier_type
        self.classifier_kwargs = classifier_kwargs
        self.clf = None
        self.feature_names: List[str] = []
        self.is_trained = False
    
    def train(
        self,
        features: List[Dict[str, float]],
        labels: List[bool],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train classifier on structural features.
        
        Args:
            features: List of feature dictionaries
            labels: List of correctness labels (True=correct, False=incorrect)
            feature_names: Optional list of feature names (for consistent ordering)
        
        Returns:
            Dictionary with training metrics (accuracy, etc.)
        """
        if len(features) == 0:
            raise ValueError("No features provided for training")
        
        if len(features) != len(labels):
            raise ValueError(f"Features ({len(features)}) and labels ({len(labels)}) must have same length")
        
        # Extract feature names from first feature dict
        if feature_names is None:
            feature_names = sorted(features[0].keys())
        
        self.feature_names = feature_names
        
        # Convert to numpy arrays
        X = np.array([[f.get(name, 0.0) for name in feature_names] for f in features])
        y = np.array([1 if label else 0 for label in labels])
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Initialize classifier
        if self.classifier_type == "gradient_boosting":
            self.clf = GradientBoostingClassifier(**self.classifier_kwargs)
        elif self.classifier_type == "logistic_regression":
            self.clf = LogisticRegression(max_iter=1000, **self.classifier_kwargs)
        elif self.classifier_type == "random_forest":
            self.clf = RandomForestClassifier(**self.classifier_kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Train
        self.clf.fit(X, y)
        self.is_trained = True
        
        # Compute training metrics
        y_pred = self.clf.predict(X)
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        
        accuracy = np.mean(y_pred == y)
        
        try:
            auc_roc = roc_auc_score(y, y_pred_proba)
        except ValueError:
            auc_roc = 0.5  # Only one class present
        
        try:
            auc_pr = average_precision_score(y, y_pred_proba)
        except ValueError:
            auc_pr = 0.0
        
        metrics = {
            "accuracy": float(accuracy),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "n_samples": len(features),
            "n_features": len(feature_names),
            "n_positive": int(np.sum(y)),
            "n_negative": int(len(y) - np.sum(y)),
        }
        
        return metrics
    
    def predict(
        self,
        features: Dict[str, float],
    ) -> Tuple[bool, float]:
        """
        Predict correctness and confidence.
        
        Args:
            features: Feature dictionary
        
        Returns:
            (is_correct: bool, confidence: float)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Convert to numpy array
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Predict
        y_pred = self.clf.predict(X)[0]
        y_pred_proba = self.clf.predict_proba(X)[0, 1]
        
        is_correct = bool(y_pred == 1)
        confidence = float(y_pred_proba)
        
        return is_correct, confidence
    
    def predict_batch(
        self,
        features_list: List[Dict[str, float]],
    ) -> List[Tuple[bool, float]]:
        """
        Predict correctness for a batch of features.
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of (is_correct, confidence) tuples
        """
        return [self.predict(f) for f in features_list]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        if not hasattr(self.clf, 'feature_importances_'):
            # Logistic regression: use absolute coefficients
            if hasattr(self.clf, 'coef_'):
                importances = np.abs(self.clf.coef_[0])
            else:
                return {name: 0.0 for name in self.feature_names}
        else:
            importances = self.clf.feature_importances_
        
        # Normalize to sum to 1
        importances = importances / (np.sum(importances) + 1e-10)
        
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
    
    def evaluate(
        self,
        features: List[Dict[str, float]],
        labels: List[bool],
    ) -> Dict[str, float]:
        """
        Evaluate classifier on test set.
        
        Args:
            features: List of feature dictionaries
            labels: List of correctness labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        predictions = self.predict_batch(features)
        y_pred = [p[0] for p in predictions]
        y_pred_proba = [p[1] for p in predictions]
        y_true = [1 if label else 0 for label in labels]
        
        accuracy = np.mean(np.array(y_pred) == np.array(y_true))
        
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            auc_pr = 0.0
        
        # Compute FPR@95 (false positive rate when 95% of positives are correctly identified)
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            # Find threshold where TPR >= 0.95
            tpr_95_idx = np.where(tpr >= 0.95)[0]
            if len(tpr_95_idx) > 0:
                fpr_95 = fpr[tpr_95_idx[0]]
            else:
                fpr_95 = 1.0
        except Exception:
            fpr_95 = 1.0
        
        return {
            "accuracy": float(accuracy),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "fpr_at_95": float(fpr_95),
            "n_samples": len(features),
        }
    
    def save(self, filepath: str) -> None:
        """
        Save classifier to file.
        
        Args:
            filepath: Path to save classifier (JSON format)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Cannot save untrained classifier.")
        
        # Save classifier using joblib (if available) or pickle
        try:
            import joblib
            joblib.dump(self.clf, filepath.replace('.json', '.pkl'))
        except ImportError:
            import pickle
            with open(filepath.replace('.json', '.pkl'), 'wb') as f:
                pickle.dump(self.clf, f)
        
        # Save metadata as JSON
        metadata = {
            "classifier_type": self.classifier_type,
            "classifier_kwargs": self.classifier_kwargs,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CRVDiagnosticClassifier':
        """
        Load classifier from file.
        
        Args:
            filepath: Path to classifier JSON file
        
        Returns:
            Loaded classifier instance
        """
        # Load metadata
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Create classifier instance
        classifier = cls(
            classifier_type=metadata['classifier_type'],
            **metadata.get('classifier_kwargs', {}),
        )
        
        classifier.feature_names = metadata['feature_names']
        
        # Load classifier model
        try:
            import joblib
            classifier.clf = joblib.load(filepath.replace('.json', '.pkl'))
        except ImportError:
            import pickle
            with open(filepath.replace('.json', '.pkl'), 'rb') as f:
                classifier.clf = pickle.load(f)
        
        classifier.is_trained = metadata.get('is_trained', True)
        
        return classifier


def load_crv_classifier(filepath: str) -> CRVDiagnosticClassifier:
    """
    Convenience function to load CRV classifier.
    
    Args:
        filepath: Path to classifier JSON file
    
    Returns:
        Loaded classifier instance
    """
    return CRVDiagnosticClassifier.load(filepath)

