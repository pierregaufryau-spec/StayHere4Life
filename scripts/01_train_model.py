#!/usr/bin/env python3
"""Script to train ML model for trading strategy.

This script:
1. Loads and prepares data
2. Computes technical indicators
3. Generates raw trading signals
4. Creates features and labels
5. Trains ML model
6. Saves trained model with metadata
7. Generates training report
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_trading_strategy.config import load_config
from ml_trading_strategy.data import prepare_data
from ml_trading_strategy.indicators import compute_indicators
from ml_trading_strategy.signals import generate_raw_signals
from ml_trading_strategy.features import create_features, create_labels
from ml_trading_strategy.models import MLTrainer
from ml_trading_strategy.visualization import plot_feature_importance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for training ML model."""
    parser = argparse.ArgumentParser(description='Train ML model for trading strategy')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name for saved model (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip hyperparameter optimization'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=None,
        help='Number of optimization trials (overrides config)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ML MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = load_config(None)
    
    # Load and prepare data
    logger.info("Loading and preparing data")
    csv_path = config.get('data.csv_path')
    generate_sample = config.get('data.generate_sample')
    
    df = prepare_data(csv_path, generate_if_missing=generate_sample)
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Compute technical indicators
    logger.info("Computing technical indicators")
    df = compute_indicators(df)
    logger.info(f"Data shape after indicators: {df.shape}")
    
    # Generate raw signals
    logger.info("Generating raw trading signals")
    df = generate_raw_signals(
        df,
        min_ema_distance=config.get('signals.min_ema_distance'),
        adx_threshold=config.get('signals.adx_threshold'),
        volume_multiplier=config.get('signals.volume_multiplier')
    )
    
    # Create features
    logger.info("Creating ML features")
    df = create_features(df)
    logger.info(f"Data shape after feature engineering: {df.shape}")
    
    # Create labels
    logger.info("Creating labels for supervised learning")
    df = create_labels(
        df,
        forward_bars=config.get('ml.forward_bars'),
        threshold=config.get('ml.label_threshold')
    )
    logger.info(f"Final data shape: {df.shape}")
    
    # Train model
    logger.info("Training ML model")
    
    # Override config with command line arguments
    ml_config = config.to_dict()['ml']
    if args.no_optimize:
        ml_config['optimize'] = False
    if args.trials is not None:
        ml_config['n_trials'] = args.trials
    
    trainer = MLTrainer(ml_config)
    
    model, scaler, metrics = trainer.train(
        df,
        optimize=ml_config['optimize'],
        n_trials=ml_config['n_trials']
    )
    
    # Save model
    logger.info("Saving trained model")
    models_dir = config.get('paths.models_dir')
    model_path = trainer.save_model(models_dir, model_name=args.model_name)
    
    # Create symbolic link to latest model
    latest_link = Path(models_dir) / 'model_latest.pkl'
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(Path(model_path).name)
    logger.info(f"Updated latest model link: {latest_link}")
    
    # Plot feature importance
    logger.info("Generating feature importance plot")
    feature_importance = dict(zip(
        trainer.feature_cols,
        model.feature_importances_
    ))
    
    plots_dir = config.get('paths.plots_dir')
    importance_path = Path(plots_dir) / 'feature_importance.png'
    plot_feature_importance(feature_importance, save_path=str(importance_path))
    
    # Print summary
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Training accuracy: {metrics['train_accuracy']:.4f}")
    logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Number of features: {metrics['n_features']}")
    logger.info(f"Training samples: {metrics['train_size']}")
    logger.info(f"Test samples: {metrics['test_size']}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
