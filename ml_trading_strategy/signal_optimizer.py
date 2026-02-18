#!/usr/bin/env python3
"""
Signal Optimizer for ML Trading Strategy
=========================================

Automated parameter optimization system for signal generation using Optuna.
Maximizes signal volume and quality for LightGBM models.

Author: ML Trading Strategy Team
License: MIT
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Configuration and Constants
# =============================================================================

DEFAULT_CONFIG = {
    'data': {
        'source': 'synthetic',  # 'csv', 'yfinance', or 'synthetic'
        'filepath': None,
        'symbol': 'BTC-USD',
        'period': '1y',
        'n_samples': 5000,
    },
    'optimization': {
        'n_trials': 100,
        'min_samples': 500,
        'target_samples': 2000,
        'min_accuracy': 0.55,
        'quality_weight': 0.6,
        'timeout': None,
    },
    'output': {
        'directory': './optimized_results',
        'generate_plots': True,
    }
}


# =============================================================================
# Class: ParameterSpace
# =============================================================================

@dataclass
class ParameterSpace:
    """
    Defines the parameter search space for optimization.
    
    All parameters have MIN and MAX bounds for Optuna to explore.
    """
    # RSI parameters
    rsi_period_min: int = 7
    rsi_period_max: int = 28
    rsi_oversold_min: float = 20.0
    rsi_oversold_max: float = 35.0
    rsi_overbought_min: float = 65.0
    rsi_overbought_max: float = 85.0
    
    # Moving Average parameters
    ma_fast_min: int = 5
    ma_fast_max: int = 25
    ma_slow_min: int = 30
    ma_slow_max: int = 100
    
    # Volume parameters
    volume_threshold_min: float = 1.1
    volume_threshold_max: float = 3.0
    
    # Volatility parameters
    volatility_window_min: int = 10
    volatility_window_max: int = 50
    volatility_threshold_min: float = 0.5
    volatility_threshold_max: float = 2.5
    
    # Bollinger Bands parameters
    bb_period_min: int = 15
    bb_period_max: int = 30
    bb_std_min: float = 1.5
    bb_std_max: float = 2.5


# =============================================================================
# Class: SignalQualityMetrics
# =============================================================================

class SignalQualityMetrics:
    """
    Calculates quality metrics for signal evaluation.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
        
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            if y_proba is not None and len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['roc_auc'] = 0.0
                
            # Class balance
            unique, counts = np.unique(y_true, return_counts=True)
            if len(unique) > 1:
                metrics['class_balance'] = min(counts) / max(counts)
            else:
                metrics['class_balance'] = 0.0
                
        except Exception as e:
            print(f"⚠️  Error calculating metrics: {e}")
            metrics = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'roc_auc': 0.0, 'class_balance': 0.0
            }
        
        return metrics


# =============================================================================
# Class: DataLoader
# =============================================================================

class DataLoader:
    """
    Handles data loading from various sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
    
    def load(self) -> pd.DataFrame:
        """
        Load data based on configuration.
        
        Returns:
            DataFrame with OHLCV data
        """
        source = self.data_config['source']
        
        print(f"📊 Loading data from source: {source}")
        
        if source == 'csv':
            return self._load_from_csv()
        elif source == 'yfinance':
            return self._load_from_yfinance()
        elif source == 'synthetic':
            return self._generate_synthetic_data()
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from CSV file."""
        filepath = self.data_config.get('filepath')
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} rows from CSV")
        return self._validate_and_prepare(df)
    
    def _load_from_yfinance(self) -> pd.DataFrame:
        """Load data from Yahoo Finance API."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")
        
        symbol = self.data_config.get('symbol', 'BTC-USD')
        period = self.data_config.get('period', '1y')
        
        df = yf.download(symbol, period=period, progress=False)
        df = df.reset_index()
        print(f"✅ Loaded {len(df)} rows from yfinance ({symbol})")
        return self._validate_and_prepare(df)
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        n_samples = self.data_config.get('n_samples', 5000)
        
        np.random.seed(42)
        
        # Generate base price with trend and noise
        trend = np.linspace(100, 150, n_samples)
        noise = np.cumsum(np.random.randn(n_samples) * 2)
        close = trend + noise
        
        # Generate OHLC
        high = close * (1 + np.abs(np.random.randn(n_samples) * 0.02))
        low = close * (1 - np.abs(np.random.randn(n_samples) * 0.02))
        open_price = close + np.random.randn(n_samples) * 1
        
        # Generate volume
        volume = np.random.lognormal(10, 1, n_samples)
        
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=n_samples, freq='h'),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        
        print(f"✅ Generated {len(df)} synthetic samples")
        return df
    
    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare data."""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Handle different column name formats
        df.columns = [col.capitalize() for col in df.columns]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove NaN values
        df = df.dropna()
        
        # Ensure proper data types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError("Insufficient data after cleaning (need at least 100 samples)")
        
        return df


# =============================================================================
# Class: SignalGeneratorWrapper
# =============================================================================

class SignalGeneratorWrapper:
    """
    Wrapper for signal generation with fallback implementation.
    
    Tries to use existing signal_generator.py if available,
    otherwise uses built-in fallback implementation.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.use_custom_generator = self._check_custom_generator()
    
    def _check_custom_generator(self) -> bool:
        """Check if custom signal_generator.py exists."""
        signal_gen_path = Path(__file__).parent / 'signal_generator.py'
        return signal_gen_path.exists()
    
    def generate_signals(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals with given parameters.
        
        Args:
            params: Dictionary of signal generation parameters
        
        Returns:
            DataFrame with signals and features
        """
        if self.use_custom_generator:
            return self._generate_with_custom(params)
        else:
            return self._generate_fallback(params)
    
    def _generate_with_custom(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Use custom signal_generator.py if available."""
        try:
            from . import signal_generator
            # Assuming signal_generator has a generate_signals function
            return signal_generator.generate_signals(self.df, params)
        except Exception as e:
            print(f"⚠️  Custom generator failed: {e}. Using fallback.")
            return self._generate_fallback(params)
    
    def _generate_fallback(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Fallback signal generation implementation.
        
        Generates technical indicators and trading signals.
        """
        df = self.df.copy()
        
        # Extract parameters
        rsi_period = int(params.get('rsi_period', 14))
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        ma_fast = int(params.get('ma_fast', 10))
        ma_slow = int(params.get('ma_slow', 50))
        volume_threshold = params.get('volume_threshold', 1.5)
        volatility_window = int(params.get('volatility_window', 20))
        volatility_threshold = params.get('volatility_threshold', 1.5)
        bb_period = int(params.get('bb_period', 20))
        bb_std = params.get('bb_std', 2.0)
        
        # Calculate RSI
        df['RSI'] = self._calculate_rsi(df['Close'], rsi_period)
        
        # Calculate Moving Averages
        df['MA_Fast'] = df['Close'].rolling(window=ma_fast).mean()
        df['MA_Slow'] = df['Close'].rolling(window=ma_slow).mean()
        df['MA_Cross'] = (df['MA_Fast'] > df['MA_Slow']).astype(int)
        
        # Volume indicator
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['High_Volume'] = (df['Volume_Ratio'] > volume_threshold).astype(int)
        
        # Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=volatility_window).std()
        df['Volatility_MA'] = df['Volatility'].rolling(window=20).mean()
        df['High_Volatility'] = (df['Volatility'] > df['Volatility_MA'] * volatility_threshold).astype(int)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        df['BB_Std'] = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * df['BB_Std'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Trend Strength
        df['Trend_Strength'] = abs(df['MA_Fast'] - df['MA_Slow']) / df['Close']
        
        # Generate signals
        df['Signal'] = 0  # 0 = no signal, 1 = buy signal
        
        # Buy signal conditions
        buy_conditions = (
            (df['RSI'] < rsi_oversold) &
            (df['MA_Cross'] == 1) &
            (df['High_Volume'] == 1) &
            (df['BB_Position'] < 0.3)
        )
        
        df.loc[buy_conditions, 'Signal'] = 1
        
        # Signal confidence
        df['Signal_Confidence'] = (
            ((100 - df['RSI']) / 100) * 0.3 +
            df['Trend_Strength'] * 0.3 +
            df['Volume_Ratio'].clip(0, 3) / 3 * 0.2 +
            (1 - df['BB_Position']).clip(0, 1) * 0.2
        )
        
        # Drop rows with NaN
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# =============================================================================
# Class: LightGBMEvaluator
# =============================================================================

class LightGBMEvaluator:
    """
    Evaluates signal quality using LightGBM with time series cross-validation.
    """
    
    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits
        self.model = None
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate signal quality with LightGBM.
        
        Args:
            df: DataFrame with signals and features
        
        Returns:
            Dictionary with evaluation results
        """
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['Date', 'Signal', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        if len(feature_cols) == 0:
            return {
                'metrics': SignalQualityMetrics.calculate_all_metrics(
                    np.zeros(10), np.zeros(10)
                ),
                'n_signals': 0,
                'success': False
            }
        
        X = df[feature_cols].values
        y = df['Signal'].values
        
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            return {
                'metrics': SignalQualityMetrics.calculate_all_metrics(y, y),
                'n_signals': int(np.sum(y)),
                'success': False
            }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Skip if test set doesn't have both classes
            if len(np.unique(y_test)) < 2:
                continue
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            # Predict
            y_proba = model.predict(X_test)
            y_pred = (y_proba > 0.5).astype(int)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
        
        if len(all_y_true) == 0:
            return {
                'metrics': SignalQualityMetrics.calculate_all_metrics(y, y),
                'n_signals': int(np.sum(y)),
                'success': False
            }
        
        # Calculate metrics
        metrics = SignalQualityMetrics.calculate_all_metrics(
            np.array(all_y_true),
            np.array(all_y_pred),
            np.array(all_y_proba)
        )
        
        return {
            'metrics': metrics,
            'n_signals': int(np.sum(y)),
            'success': True
        }


# =============================================================================
# Class: OptimizationObjective
# =============================================================================

class OptimizationObjective:
    """
    Optuna objective function for parameter optimization.
    """
    
    def __init__(self, data_loader: DataLoader, config: Dict[str, Any],
                 parameter_space: ParameterSpace):
        self.data_loader = data_loader
        self.config = config
        self.parameter_space = parameter_space
        self.opt_config = config['optimization']
        
        # Load data once
        self.df = data_loader.load()
        self.evaluator = LightGBMEvaluator(n_splits=3)
        
        # Get configuration values
        self.min_samples = self.opt_config.get('min_samples', 500)
        self.target_samples = self.opt_config.get('target_samples', 2000)
        self.min_accuracy = self.opt_config.get('min_accuracy', 0.55)
        self.quality_weight = self.opt_config.get('quality_weight', 0.6)
        self.volume_weight = 1.0 - self.quality_weight
        
        print(f"🎯 Optimization target: {self.target_samples} signals, "
              f"{self.min_accuracy*100:.1f}% accuracy")
        print(f"⚖️  Score weights: {self.quality_weight*100:.0f}% quality, "
              f"{self.volume_weight*100:.0f}% volume")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Composite score (higher is better)
        """
        # Sample parameters from search space
        params = self._suggest_parameters(trial)
        
        # Generate signals
        signal_gen = SignalGeneratorWrapper(self.df)
        df_signals = signal_gen.generate_signals(params)
        
        # Evaluate with LightGBM
        eval_results = self.evaluator.evaluate(df_signals)
        
        if not eval_results['success']:
            return 0.0
        
        metrics = eval_results['metrics']
        n_signals = eval_results['n_signals']
        
        # Calculate scores
        score = self._calculate_composite_score(metrics, n_signals)
        
        # Store metrics in trial user attributes
        trial.set_user_attr('n_signals', n_signals)
        trial.set_user_attr('accuracy', metrics['accuracy'])
        trial.set_user_attr('f1_score', metrics['f1_score'])
        trial.set_user_attr('precision', metrics['precision'])
        trial.set_user_attr('recall', metrics['recall'])
        
        return score
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameter values for trial."""
        ps = self.parameter_space
        
        return {
            'rsi_period': trial.suggest_int('rsi_period', ps.rsi_period_min, ps.rsi_period_max),
            'rsi_oversold': trial.suggest_float('rsi_oversold', ps.rsi_oversold_min, ps.rsi_oversold_max),
            'rsi_overbought': trial.suggest_float('rsi_overbought', ps.rsi_overbought_min, ps.rsi_overbought_max),
            'ma_fast': trial.suggest_int('ma_fast', ps.ma_fast_min, ps.ma_fast_max),
            'ma_slow': trial.suggest_int('ma_slow', ps.ma_slow_min, ps.ma_slow_max),
            'volume_threshold': trial.suggest_float('volume_threshold', ps.volume_threshold_min, ps.volume_threshold_max),
            'volatility_window': trial.suggest_int('volatility_window', ps.volatility_window_min, ps.volatility_window_max),
            'volatility_threshold': trial.suggest_float('volatility_threshold', ps.volatility_threshold_min, ps.volatility_threshold_max),
            'bb_period': trial.suggest_int('bb_period', ps.bb_period_min, ps.bb_period_max),
            'bb_std': trial.suggest_float('bb_std', ps.bb_std_min, ps.bb_std_max),
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, float], 
                                   n_signals: int) -> float:
        """
        Calculate composite score balancing quality and volume.
        
        Score formula:
        - Volume score: min(n_signals / target_samples, 1.0) * 100
        - Quality score: (0.7 * f1_score + 0.3 * accuracy) * 100
        - Composite: quality_weight * quality + volume_weight * volume - penalties
        """
        # Volume score
        volume_score = min(n_signals / self.target_samples, 1.0) * 100
        
        # Quality score
        f1 = metrics.get('f1_score', 0.0)
        accuracy = metrics.get('accuracy', 0.0)
        quality_score = (0.7 * f1 + 0.3 * accuracy) * 100
        
        # Composite score
        composite_score = (
            self.quality_weight * quality_score +
            self.volume_weight * volume_score
        )
        
        # Apply penalties
        penalties = 0.0
        
        # Penalty for insufficient signals
        if n_signals < self.min_samples:
            penalties += (self.min_samples - n_signals) / self.min_samples * 20
        
        # Penalty for low accuracy
        if accuracy < self.min_accuracy:
            penalties += (self.min_accuracy - accuracy) * 100
        
        # Penalty for imbalanced classes
        class_balance = metrics.get('class_balance', 0.0)
        if class_balance < 0.2:
            penalties += (0.2 - class_balance) * 50
        
        final_score = max(0.0, composite_score - penalties)
        
        return final_score


# =============================================================================
# Class: SignalOptimizerMain
# =============================================================================

class SignalOptimizerMain:
    """
    Main orchestrator for signal optimization process.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output']['directory'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.study = None
        self.best_params = None
        self.best_metrics = None
        
    def optimize(self) -> optuna.Study:
        """
        Run optimization process.
        
        Returns:
            Completed Optuna study
        """
        print("\n" + "="*70)
        print("🚀 Signal Parameter Optimization Started")
        print("="*70 + "\n")
        
        # Initialize components
        data_loader = DataLoader(self.config)
        parameter_space = ParameterSpace()
        objective = OptimizationObjective(data_loader, self.config, parameter_space)
        
        # Create Optuna study
        n_trials = self.config['optimization'].get('n_trials', 100)
        timeout = self.config['optimization'].get('timeout')
        
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='signal_optimization'
        )
        
        print(f"🔬 Starting optimization with {n_trials} trials...")
        print(f"⏱️  Algorithm: Tree-structured Parzen Estimator (TPE)\n")
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._trial_callback]
        )
        
        print("\n" + "="*70)
        print("✅ Optimization Completed!")
        print("="*70 + "\n")
        
        # Extract best results
        self._extract_best_results()
        
        return self.study
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback after each trial."""
        if trial.number % 10 == 0:
            best_score = study.best_value
            n_signals = trial.user_attrs.get('n_signals', 0)
            accuracy = trial.user_attrs.get('accuracy', 0)
            
            print(f"Trial {trial.number}: Score={trial.value:.2f}, "
                  f"Signals={n_signals}, Accuracy={accuracy:.3f}")
    
    def _extract_best_results(self):
        """Extract best parameters and metrics."""
        self.best_params = self.study.best_params
        best_trial = self.study.best_trial
        
        self.best_metrics = {
            'composite_score': best_trial.value,
            'n_signals': best_trial.user_attrs.get('n_signals', 0),
            'accuracy': best_trial.user_attrs.get('accuracy', 0.0),
            'f1_score': best_trial.user_attrs.get('f1_score', 0.0),
            'precision': best_trial.user_attrs.get('precision', 0.0),
            'recall': best_trial.user_attrs.get('recall', 0.0),
        }
        
        print(f"🏆 Best Score: {self.best_metrics['composite_score']:.2f}")
        print(f"📊 Signals Generated: {self.best_metrics['n_signals']}")
        print(f"🎯 Accuracy: {self.best_metrics['accuracy']:.1%}")
        print(f"📈 F1-Score: {self.best_metrics['f1_score']:.1%}\n")
    
    def export_results(self):
        """Export optimization results to multiple formats."""
        print("💾 Exporting results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export JSON
        self._export_json(timestamp)
        
        # Export Python dict
        self._export_python(timestamp)
        
        # Export CSV trials
        self._export_trials_csv(timestamp)
        
        # Export text report
        self._export_text_report(timestamp)
        
        print("✅ All exports completed!\n")
    
    def _export_json(self, timestamp: str):
        """Export as JSON."""
        output_data = {
            'timestamp': timestamp,
            'best_parameters': self.best_params,
            'best_metrics': self.best_metrics,
            'configuration': self.config
        }
        
        filepath = self.output_dir / f"optimized_params_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"   📄 JSON: {filepath}")
    
    def _export_python(self, timestamp: str):
        """Export as Python dict."""
        filepath = self.output_dir / f"optimized_params_{timestamp}.py"
        
        with open(filepath, 'w') as f:
            f.write(f"# Optimized Parameters - {timestamp}\n\n")
            f.write("OPTIMIZED_PARAMS = ")
            f.write(repr(self.best_params))
            f.write("\n\nBEST_METRICS = ")
            f.write(repr(self.best_metrics))
            f.write("\n")
        
        print(f"   🐍 Python: {filepath}")
    
    def _export_trials_csv(self, timestamp: str):
        """Export all trials to CSV."""
        trials_df = self.study.trials_dataframe()
        filepath = self.output_dir / f"optimization_trials_{timestamp}.csv"
        trials_df.to_csv(filepath, index=False)
        
        print(f"   📊 CSV Trials: {filepath}")
    
    def _export_text_report(self, timestamp: str):
        """Export detailed text report."""
        filepath = self.output_dir / f"optimization_report_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SIGNAL OPTIMIZATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BEST PARAMETERS:\n")
            f.write("-"*70 + "\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\nBEST METRICS:\n")
            f.write("-"*70 + "\n")
            for metric, value in self.best_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            
            f.write("\nOPTIMIZATION CONFIGURATION:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Trials: {len(self.study.trials)}\n")
            f.write(f"  Data Source: {self.config['data']['source']}\n")
            f.write(f"  Min Samples: {self.config['optimization']['min_samples']}\n")
            f.write(f"  Target Samples: {self.config['optimization']['target_samples']}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"   📝 Report: {filepath}")
    
    def generate_plots(self):
        """Generate optimization visualizations."""
        if not self.config['output'].get('generate_plots', True):
            return
        
        print("📊 Generating plots...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Try plotly first (with kaleido)
            import plotly
            import kaleido
            
            self._generate_plotly_plots(timestamp)
            print("   ✅ Plotly plots generated")
        except ImportError:
            print("   ⚠️  Plotly/Kaleido not available, using matplotlib")
            self._generate_matplotlib_plots(timestamp)
    
    def _generate_plotly_plots(self, timestamp: str):
        """Generate plots with Plotly."""
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances
        )
        
        # Optimization history
        fig = plot_optimization_history(self.study)
        filepath = self.output_dir / f"optimization_history_{timestamp}.png"
        fig.write_image(str(filepath))
        
        # Parameter importances
        fig = plot_param_importances(self.study)
        filepath = self.output_dir / f"param_importances_{timestamp}.png"
        fig.write_image(str(filepath))
    
    def _generate_matplotlib_plots(self, timestamp: str):
        """Generate plots with Matplotlib."""
        trials_df = self.study.trials_dataframe()
        
        # Optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(trials_df['number'], trials_df['value'], 'b-', alpha=0.6)
        plt.xlabel('Trial Number')
        plt.ylabel('Composite Score')
        plt.title('Optimization History')
        plt.grid(True, alpha=0.3)
        
        filepath = self.output_dir / f"optimization_history_{timestamp}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 History plot: {filepath}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main entry point for signal optimizer.
    """
    print("\n" + "="*70)
    print("🎯 Signal Parameter Optimizer")
    print("="*70 + "\n")
    
    # Use default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Interactive configuration (optional)
    print("Configuration:")
    print(f"  Data Source: {config['data']['source']}")
    print(f"  Optimization Trials: {config['optimization']['n_trials']}")
    print(f"  Target Signals: {config['optimization']['target_samples']}")
    print(f"  Min Accuracy: {config['optimization']['min_accuracy']*100}%")
    print()
    
    # Create optimizer
    optimizer = SignalOptimizerMain(config)
    
    # Run optimization
    try:
        study = optimizer.optimize()
        
        # Export results
        optimizer.export_results()
        
        # Generate plots
        optimizer.generate_plots()
        
        print("\n" + "="*70)
        print("🎉 Optimization Complete!")
        print("="*70)
        print(f"\n📁 Results saved to: {optimizer.output_dir}")
        print("\nNext steps:")
        print("  1. Review the optimization report")
        print("  2. Test parameters with verify_overfitting.py")
        print("  3. Integrate best parameters into your trading strategy")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
