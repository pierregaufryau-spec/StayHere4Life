#!/usr/bin/env python3
"""Script to optimize trading strategy hyperparameters.

This script:
1. Loads configuration
2. Prepares data
3. Runs Optuna optimization
4. Tests different signal and ML parameters
5. Saves best parameters
6. Generates optimization report
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from ml_trading_strategy.config import load_config
from ml_trading_strategy.data import prepare_data
from ml_trading_strategy.indicators import compute_indicators
from ml_trading_strategy.signals import generate_raw_signals
from ml_trading_strategy.features import create_features, create_labels
from ml_trading_strategy.models import MLTrainer
from ml_trading_strategy.backtest import backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for best parameters (default: results/best_params.json)'
    )
    parser.add_argument(
        '--optimize-signals',
        action='store_true',
        help='Optimize signal parameters (default: optimize ML parameters)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = load_config(None)
    
    # Load and prepare data once
    logger.info("Loading and preparing data")
    csv_path = config.get('data.csv_path')
    generate_sample = config.get('data.generate_sample')
    
    df = prepare_data(csv_path, generate_if_missing=generate_sample)
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Compute indicators
    logger.info("Computing technical indicators")
    df = compute_indicators(df)
    
    if args.optimize_signals:
        logger.info("Optimizing signal generation parameters")
        
        def objective_signals(trial):
            """Objective function for signal optimization."""
            # Suggest parameters
            min_ema_distance = trial.suggest_float('min_ema_distance', 0.01, 0.15)
            adx_threshold = trial.suggest_int('adx_threshold', 15, 35)
            volume_multiplier = trial.suggest_float('volume_multiplier', 1.0, 2.0)
            
            # Generate signals with these parameters
            df_trial = generate_raw_signals(
                df.copy(),
                min_ema_distance=min_ema_distance,
                adx_threshold=adx_threshold,
                volume_multiplier=volume_multiplier
            )
            
            # Run backtest
            trading_config = config.to_dict()['trading']
            _, _, metrics = backtest(df_trial, 'raw_signal', trading_config)
            
            # Optimize for Sharpe ratio
            return metrics['sharpe_ratio']
        
        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_signals, n_trials=args.trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best Sharpe ratio: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Update config section
        param_section = 'signals'
        
    else:
        logger.info("Optimizing ML parameters")
        
        # Create features and labels once
        logger.info("Creating features and labels")
        df = generate_raw_signals(
            df,
            min_ema_distance=config.get('signals.min_ema_distance'),
            adx_threshold=config.get('signals.adx_threshold'),
            volume_multiplier=config.get('signals.volume_multiplier')
        )
        df = create_features(df)
        df = create_labels(
            df,
            forward_bars=config.get('ml.forward_bars'),
            threshold=config.get('ml.label_threshold')
        )
        
        def objective_ml(trial):
            """Objective function for ML optimization."""
            # Suggest parameters
            forward_bars = trial.suggest_int('forward_bars', 3, 10)
            label_threshold = trial.suggest_float('label_threshold', 0.001, 0.01)
            ml_threshold = trial.suggest_float('ml_threshold', 0.25, 0.50)
            n_features = trial.suggest_int('n_features', 10, 50)
            
            # Recreate labels with new parameters
            df_trial = create_labels(
                df.copy(),
                forward_bars=forward_bars,
                threshold=label_threshold
            )
            
            # Train model with these parameters
            ml_config = {
                'forward_bars': forward_bars,
                'label_threshold': label_threshold,
                'n_features': n_features,
                'feature_selection_method': 'importance',
                'optimize': False,  # Don't optimize hyperparameters within optimization
                'test_size': 0.3,
                'random_seed': 42
            }
            
            trainer = MLTrainer(ml_config)
            
            try:
                model, scaler, metrics = trainer.train(df_trial, optimize=False)
                
                # Use validation accuracy as objective
                return metrics['val_accuracy']
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_ml, n_trials=args.trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best validation accuracy: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Update config section
        param_section = 'ml'
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = config.get('paths.results_dir')
        output_path = Path(results_dir) / 'best_params.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimization_type': 'signals' if args.optimize_signals else 'ml',
        'n_trials': args.trials,
        'best_value': best_value,
        'best_params': best_params,
        'config_section': param_section
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Generate optimization report
    logger.info("=" * 80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Optimization type: {results['optimization_type']}")
    logger.info(f"Number of trials: {args.trials}")
    logger.info(f"Best value: {best_value:.4f}")
    logger.info("")
    logger.info("Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("")
    logger.info("To use these parameters, update your config.yaml file:")
    logger.info(f"{param_section}:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
