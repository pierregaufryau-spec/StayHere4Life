"""ML model training module."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLTrainer:
    """Machine Learning trainer for trading signals.
    
    Handles feature selection, hyperparameter optimization,
    model training, and model persistence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ML trainer.
        
        Args:
            config: Configuration dictionary with ML parameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metrics = {}
        
        # Set random seed for reproducibility
        self.random_seed = config.get('random_seed', 42)
        np.random.seed(self.random_seed)
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str = 'importance',
        n_features: int = 20
    ) -> List[str]:
        """Select top features using specified method.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            method: Feature selection method ('importance', 'correlation', 'all')
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features: method={method}, n_features={n_features}")
        
        if method == 'all':
            logger.info(f"Using all {len(feature_names)} features")
            return feature_names
        
        if method == 'importance':
            # Train a simple model to get feature importance
            logger.debug("Training model for feature importance")
            
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_seed,
                verbose=-1
            )
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Select top n features
            top_indices = np.argsort(importance)[-n_features:]
            selected_features = [feature_names[i] for i in top_indices]
            
            logger.info(f"Selected {len(selected_features)} features by importance")
            
            # Log top features
            top_importance = sorted(
                [(feature_names[i], importance[i]) for i in top_indices],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            logger.debug(f"Top 10 features: {top_importance}")
            
            return selected_features
        
        elif method == 'correlation':
            # Select features with highest correlation to target
            logger.debug("Calculating feature correlations")
            
            import pandas as pd
            df_temp = pd.DataFrame(X, columns=feature_names)
            df_temp['target'] = y
            
            correlations = df_temp.corr()['target'].abs().sort_values(ascending=False)
            selected_features = correlations.iloc[1:n_features+1].index.tolist()
            
            logger.info(f"Selected {len(selected_features)} features by correlation")
            return selected_features
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Optimizing hyperparameters: n_trials={n_trials}")
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': self.random_seed,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val)
            accuracy = (val_pred == y_val).mean()
            
            return accuracy
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Optimization complete: best_score={best_score:.4f}")
        logger.debug(f"Best parameters: {best_params}")
        
        return best_params
    
    def train(
        self,
        df,
        feature_cols: List[str] = None,
        optimize: bool = True,
        n_trials: int = 30
    ) -> Tuple[Any, Any, Dict]:
        """Train ML model on data.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature columns to use
            optimize: Whether to optimize hyperparameters
            n_trials: Number of optimization trials if optimize=True
            
        Returns:
            Tuple of (model, scaler, metrics)
        """
        logger.info("Training ML model")
        
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        
        # Get feature columns if not provided
        if feature_cols is None:
            from ml_trading_strategy.features import get_feature_columns
            feature_cols = get_feature_columns(df)
        
        # Check for missing features
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in DataFrame: {missing_features}")
        
        self.feature_cols = feature_cols
        
        # Prepare data
        X = df[feature_cols].values
        y = df['label'].values
        
        logger.info(f"Training data: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Label distribution: {np.bincount(y + 1)}")  # +1 to handle -1, 0, 1 labels
        
        # Feature selection
        n_features = self.config.get('n_features', 20)
        feature_selection_method = self.config.get('feature_selection_method', 'importance')
        
        if len(feature_cols) > n_features:
            selected_features = self.select_features(
                X, y, feature_cols,
                method=feature_selection_method,
                n_features=n_features
            )
            X = df[selected_features].values
            self.feature_cols = selected_features
        
        # Split data
        test_size = self.config.get('test_size', 0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y
        )
        
        # Further split training into train and validation for optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.random_seed,
            stratify=y_train
        )
        
        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Scale features
        logger.debug("Scaling features")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters or use defaults
        if optimize:
            best_params = self.optimize_hyperparameters(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                n_trials=n_trials
            )
        else:
            logger.info("Using default hyperparameters")
            best_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': self.random_seed,
                'verbose': -1
            }
        
        # Train final model
        logger.info("Training final model")
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = (train_pred == y_train).mean()
        val_accuracy = (val_pred == y_val).mean()
        test_accuracy = (test_pred == y_test).mean()
        
        # Calculate probabilities for threshold analysis
        test_proba = self.model.predict_proba(X_test_scaled)
        
        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy),
            'test_accuracy': float(test_accuracy),
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'hyperparameters': best_params
        }
        
        logger.info(f"Model performance:")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Val accuracy: {val_accuracy:.4f}")
        logger.info(f"  Test accuracy: {test_accuracy:.4f}")
        
        return self.model, self.scaler, self.metrics
    
    def save_model(self, model_dir: str, model_name: str = None) -> str:
        """Save trained model with metadata.
        
        Args:
            model_dir: Directory to save model
            model_name: Model filename (without extension)
            
        Returns:
            Path to saved model file
        """
        if self.model is None or self.scaler is None:
            raise ValueError("No trained model to save. Train a model first.")
        
        # Create model directory
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f'model_{timestamp}'
        
        model_path = model_dir / f'{model_name}.pkl'
        
        # Create model package
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'config': self.config,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Also save metadata as JSON for easy inspection
        metadata_path = model_dir / f'{model_name}_metadata.json'
        metadata = {
            'feature_cols': self.feature_cols,
            'config': self.config,
            'metrics': self.metrics,
            'timestamp': model_package['timestamp'],
            'version': model_package['version']
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return str(model_path)
