#!/usr/bin/env python3
"""Main orchestration script for ML trading strategy.

This script runs the complete pipeline:
1. Load configuration
2. Train model (if needed or --retrain flag)
3. Run backtest
4. Generate reports and visualizations
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ml_trading_strategy.config import load_config
from ml_trading_strategy.data import prepare_data
from ml_trading_strategy.indicators import compute_indicators
from ml_trading_strategy.signals import generate_raw_signals
from ml_trading_strategy.features import create_features, create_labels
from ml_trading_strategy.models import MLTrainer, MLPredictor
from ml_trading_strategy.backtest import backtest
from ml_trading_strategy.visualization import (
    plot_comparison,
    plot_feature_importance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(config, force=False):
    """Train ML model if needed.
    
    Args:
        config: Configuration object
        force: Force retraining even if model exists
        
    Returns:
        Path to trained model
    """
    models_dir = Path(config.get('paths.models_dir'))
    latest_model = models_dir / 'model_latest.pkl'
    
    if latest_model.exists() and not force:
        logger.info(f"Using existing model: {latest_model}")
        return latest_model
    
    logger.info("Training new ML model")
    
    # Load and prepare data
    csv_path = config.get('data.csv_path')
    generate_sample = config.get('data.generate_sample')
    df = prepare_data(csv_path, generate_if_missing=generate_sample)
    
    # Compute indicators
    df = compute_indicators(df)
    
    # Generate signals
    df = generate_raw_signals(
        df,
        min_ema_distance=config.get('signals.min_ema_distance'),
        adx_threshold=config.get('signals.adx_threshold'),
        volume_multiplier=config.get('signals.volume_multiplier')
    )
    
    # Create features and labels
    df = create_features(df)
    df = create_labels(
        df,
        forward_bars=config.get('ml.forward_bars'),
        threshold=config.get('ml.label_threshold')
    )
    
    # Train model
    ml_config = config.to_dict()['ml']
    trainer = MLTrainer(ml_config)
    model, scaler, metrics = trainer.train(
        df,
        optimize=ml_config['optimize'],
        n_trials=ml_config['n_trials']
    )
    
    # Save model
    model_path = trainer.save_model(models_dir)
    
    # Create symbolic link
    if latest_model.exists() or latest_model.is_symlink():
        latest_model.unlink()
    latest_model.symlink_to(Path(model_path).name)
    
    # Plot feature importance
    feature_importance = dict(zip(
        trainer.feature_cols,
        model.feature_importances_
    ))
    plots_dir = config.get('paths.plots_dir')
    importance_path = Path(plots_dir) / 'feature_importance.png'
    plot_feature_importance(feature_importance, save_path=str(importance_path))
    
    logger.info(f"Model training complete: {model_path}")
    logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    
    return latest_model


def run_backtest(config, model_path):
    """Run backtest with trained model.
    
    Args:
        config: Configuration object
        model_path: Path to trained model
        
    Returns:
        Dictionary with backtest results
    """
    logger.info("Running backtest")
    
    # Load model
    predictor = MLPredictor(str(model_path))
    
    # Load and prepare data
    csv_path = config.get('data.csv_path')
    generate_sample = config.get('data.generate_sample')
    df = prepare_data(csv_path, generate_if_missing=generate_sample)
    
    # Compute indicators
    df = compute_indicators(df)
    
    # Generate signals
    df = generate_raw_signals(
        df,
        min_ema_distance=config.get('signals.min_ema_distance'),
        adx_threshold=config.get('signals.adx_threshold'),
        volume_multiplier=config.get('signals.volume_multiplier')
    )
    
    # Create features
    df = create_features(df)
    
    # Apply ML predictions
    df = predictor.predict(df, threshold=config.get('ml.ml_threshold'))
    
    # Run backtests
    trading_config = config.to_dict()['trading']
    
    logger.info("Backtesting raw strategy")
    df_raw, trades_raw, metrics_raw = backtest(df, 'raw_signal', trading_config)
    
    logger.info("Backtesting ML-enhanced strategy")
    df_ml, trades_ml, metrics_ml = backtest(df, 'ml_signal', trading_config)
    
    return {
        'df_raw': df_raw,
        'df_ml': df_ml,
        'trades_raw': trades_raw,
        'trades_ml': trades_ml,
        'metrics_raw': metrics_raw,
        'metrics_ml': metrics_ml
    }


def generate_reports(config, results):
    """Generate reports and visualizations.
    
    Args:
        config: Configuration object
        results: Backtest results dictionary
    """
    logger.info("Generating reports and visualizations")
    
    plots_dir = Path(config.get('paths.plots_dir'))
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plot
    plot_comparison(
        results['df_raw'],
        results['df_ml'],
        results['metrics_raw'],
        results['metrics_ml'],
        results['trades_raw'],
        results['trades_ml'],
        save_path=str(plots_dir / 'strategy_comparison.png')
    )
    
    logger.info(f"Reports saved to {plots_dir}")


def print_summary(results):
    """Print summary of results.
    
    Args:
        results: Backtest results dictionary
    """
    metrics_raw = results['metrics_raw']
    metrics_ml = results['metrics_ml']
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE - RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("RAW STRATEGY:")
    logger.info(f"  Total Return: {metrics_raw['total_return']:.2f}%")
    logger.info(f"  Total Trades: {metrics_raw['total_trades']}")
    logger.info(f"  Win Rate: {metrics_raw['win_rate']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics_raw['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics_raw['max_drawdown']:.2f}%")
    logger.info("")
    logger.info("ML-ENHANCED STRATEGY:")
    logger.info(f"  Total Return: {metrics_ml['total_return']:.2f}%")
    logger.info(f"  Total Trades: {metrics_ml['total_trades']}")
    logger.info(f"  Win Rate: {metrics_ml['win_rate']:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics_ml['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics_ml['max_drawdown']:.2f}%")
    logger.info("")
    logger.info("IMPROVEMENT:")
    logger.info(f"  Return: {metrics_ml['total_return'] - metrics_raw['total_return']:+.2f}%")
    logger.info(f"  Win Rate: {metrics_ml['win_rate'] - metrics_raw['win_rate']:+.2f}%")
    logger.info(f"  Sharpe: {metrics_ml['sharpe_ratio'] - metrics_raw['sharpe_ratio']:+.2f}")
    logger.info("=" * 80)


def main():
    """Main function for complete pipeline."""
    parser = argparse.ArgumentParser(
        description='ML Trading Strategy - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python main.py
  
  # Run with custom config
  python main.py --config my_config.yaml
  
  # Force retrain model
  python main.py --retrain
  
  # Run in verbose mode
  python main.py --verbose
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retraining of ML model'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("ML TRADING STRATEGY - FULL PIPELINE")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = load_config(None)
    
    try:
        # Step 1: Train model (if needed)
        model_path = train_model(config, force=args.retrain)
        
        # Step 2: Run backtest
        results = run_backtest(config, model_path)
        
        # Step 3: Generate reports
        generate_reports(config, results)
        
        # Step 4: Print summary
        print_summary(results)
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
