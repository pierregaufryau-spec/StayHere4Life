"""ML model prediction module."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MLPredictor:
    """ML predictor for generating trading signals.
    
    Loads a trained model and applies it to new data to generate
    ML-enhanced trading signals.
    """
    
    def __init__(self, model_path: str):
        """Initialize ML predictor.
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.config = None
        self.metadata = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from file.
        
        Args:
            model_path: Path to model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Extract components
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_cols = model_package['feature_cols']
            self.config = model_package.get('config', {})
            
            # Store metadata
            self.metadata = {
                'timestamp': model_package.get('timestamp'),
                'version': model_package.get('version'),
                'metrics': model_package.get('metrics', {})
            }
            
            logger.info(f"Model loaded successfully")
            logger.info(f"  Version: {self.metadata['version']}")
            logger.info(f"  Training date: {self.metadata['timestamp']}")
            logger.info(f"  Features: {len(self.feature_cols)}")
            
            if 'metrics' in model_package and 'test_accuracy' in model_package['metrics']:
                logger.info(f"  Test accuracy: {model_package['metrics']['test_accuracy']:.4f}")
        
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': str(self.model_path),
            'feature_cols': self.feature_cols,
            'n_features': len(self.feature_cols) if self.feature_cols else 0,
            'config': self.config,
            'metadata': self.metadata
        }
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
        """Generate ML predictions for trading signals.
        
        Args:
            df: DataFrame with features
            threshold: Probability threshold for generating signals
                      (higher = more conservative)
            
        Returns:
            DataFrame with ML predictions added:
                - ml_proba_long: Probability of long signal
                - ml_proba_short: Probability of short signal
                - ml_signal: Final ML signal (1, -1, or 0)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        logger.info(f"Generating ML predictions with threshold={threshold}")
        
        df = df.copy()
        
        # Check for missing features
        missing_features = set(self.feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features
        X = df[self.feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Map predictions to signals
        # Assuming model predicts: 0 = bearish (-1), 1 = neutral (0), 2 = bullish (1)
        # We need to map model output to our signal convention
        
        # Get class probabilities
        n_classes = probabilities.shape[1]
        
        if n_classes == 3:
            # Three classes: bearish, neutral, bullish
            df['ml_proba_short'] = probabilities[:, 0]  # bearish
            df['ml_proba_neutral'] = probabilities[:, 1]  # neutral
            df['ml_proba_long'] = probabilities[:, 2]  # bullish
        elif n_classes == 2:
            # Binary classification
            df['ml_proba_short'] = probabilities[:, 0]
            df['ml_proba_long'] = probabilities[:, 1]
        else:
            raise ValueError(f"Unexpected number of classes: {n_classes}")
        
        # Generate signals based on threshold
        df['ml_signal'] = 0
        
        # Long signal: high probability of bullish outcome
        if 'ml_proba_long' in df.columns:
            df.loc[df['ml_proba_long'] > threshold, 'ml_signal'] = 1
        
        # Short signal: high probability of bearish outcome
        if 'ml_proba_short' in df.columns:
            df.loc[df['ml_proba_short'] > threshold, 'ml_signal'] = -1
        
        # Count signals
        signal_counts = df['ml_signal'].value_counts().sort_index()
        logger.info(f"ML predictions generated:")
        for signal, count in signal_counts.items():
            pct = count / len(df) * 100
            signal_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}.get(signal, 'Unknown')
            logger.info(f"  {signal_name}: {count} ({pct:.2f}%)")
        
        return df
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict signal for a single observation.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Check for missing features
        missing_features = set(self.feature_cols) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create feature vector
        X = [[features[col] for col in self.feature_cols]]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Map to signal
        signal_map = {0: -1, 1: 0, 2: 1}  # Assuming 3-class model
        signal = signal_map.get(prediction, 0)
        
        result = {
            'signal': signal,
            'prediction': int(prediction),
            'probabilities': probabilities.tolist()
        }
        
        return result
