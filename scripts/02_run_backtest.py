#!/usr/bin/env python3
"""Script to run backtest with trained ML model.

This script:
1. Loads trained ML model
2. Loads and prepares data
3. Computes indicators
4. Generates raw signals
5. Applies ML predictions
6. Runs backtest for both raw and ML strategies
7. Compares results
8. Generates plots and reports
"""

import argparse
import json
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
from ml_trading_strategy.features import create_features
from ml_trading_strategy.models import MLPredictor
from ml_trading_strategy.backtest import backtest
from ml_trading_strategy.visualization import (
    plot_comparison,
    plot_equity_curve,
    plot_drawdown,
    plot_trade_distribution
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for running backtest."""
    parser = argparse.ArgumentParser(description='Run backtest with trained ML model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: saved_models/model_latest.pkl)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='ML prediction threshold (default: from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: from config)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("BACKTEST WITH ML MODEL")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = load_config(None)
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        models_dir = config.get('paths.models_dir')
        model_path = Path(models_dir) / 'model_latest.pkl'
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train a model first using scripts/01_train_model.py")
        return 1
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    predictor = MLPredictor(str(model_path))
    
    model_info = predictor.get_model_info()
    logger.info(f"Model loaded: {model_info['n_features']} features")
    
    # Load and prepare data
    logger.info("Loading and preparing data")
    csv_path = config.get('data.csv_path')
    generate_sample = config.get('data.generate_sample')
    
    df = prepare_data(csv_path, generate_if_missing=generate_sample)
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Compute technical indicators
    logger.info("Computing technical indicators")
    df = compute_indicators(df)
    
    # Generate raw signals
    logger.info("Generating raw trading signals")
    df = generate_raw_signals(
        df,
        min_ema_distance=config.get('signals.min_ema_distance'),
        adx_threshold=config.get('signals.adx_threshold'),
        volume_multiplier=config.get('signals.volume_multiplier')
    )
    
    # Create features (needed for ML predictions)
    logger.info("Creating features for ML predictions")
    df = create_features(df)
    
    # Apply ML predictions
    logger.info("Generating ML predictions")
    threshold = args.threshold if args.threshold else config.get('ml.ml_threshold')
    df = predictor.predict(df, threshold=threshold)
    
    # Run backtest for raw strategy
    logger.info("Running backtest: Raw Strategy")
    trading_config = config.to_dict()['trading']
    df_raw, trades_raw, metrics_raw = backtest(df, 'raw_signal', trading_config)
    
    # Run backtest for ML strategy
    logger.info("Running backtest: ML-Enhanced Strategy")
    df_ml, trades_ml, metrics_ml = backtest(df, 'ml_signal', trading_config)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        results_dir = config.get('paths.results_dir')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(results_dir) / f'backtest_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")
    
    # Save metrics as JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'threshold': threshold,
        'raw_strategy': metrics_raw,
        'ml_strategy': metrics_ml,
        'comparison': {
            'return_improvement': metrics_ml['total_return'] - metrics_raw['total_return'],
            'trades_reduction': metrics_raw['total_trades'] - metrics_ml['total_trades'],
            'win_rate_improvement': metrics_ml['win_rate'] - metrics_raw['win_rate'],
            'sharpe_improvement': metrics_ml['sharpe_ratio'] - metrics_raw['sharpe_ratio']
        }
    }
    
    results_path = output_dir / 'backtest_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Generate plots
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    logger.info("Generating comparison plot")
    plot_comparison(
        df_raw, df_ml,
        metrics_raw, metrics_ml,
        trades_raw, trades_ml,
        save_path=str(plots_dir / 'comparison.png')
    )
    
    logger.info("Generating equity curves")
    plot_equity_curve(df_raw, save_path=str(plots_dir / 'equity_raw.png'), title='Raw Strategy Equity')
    plot_equity_curve(df_ml, save_path=str(plots_dir / 'equity_ml.png'), title='ML-Enhanced Strategy Equity')
    
    logger.info("Generating drawdown plots")
    plot_drawdown(df_raw, save_path=str(plots_dir / 'drawdown_raw.png'), title='Raw Strategy Drawdown')
    plot_drawdown(df_ml, save_path=str(plots_dir / 'drawdown_ml.png'), title='ML-Enhanced Strategy Drawdown')
    
    logger.info("Generating trade distribution plots")
    if trades_raw:
        plot_trade_distribution(trades_raw, save_path=str(plots_dir / 'trades_raw.png'), title='Raw Strategy Trades')
    if trades_ml:
        plot_trade_distribution(trades_ml, save_path=str(plots_dir / 'trades_ml.png'), title='ML-Enhanced Strategy Trades')
    
    # Print summary
    logger.info("=" * 80)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("RAW STRATEGY:")
    logger.info(f"  Total Return: {metrics_raw['total_return']:.2f}%")
    logger.info(f"  Total Trades: {metrics_raw['total_trades']}")
    logger.info(f"  Win Rate: {metrics_raw['win_rate']:.2f}%")
    logger.info(f"  Profit Factor: {metrics_raw['profit_factor']:.2f}")
    logger.info(f"  Sharpe Ratio: {metrics_raw['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics_raw['max_drawdown']:.2f}%")
    logger.info("")
    logger.info("ML-ENHANCED STRATEGY:")
    logger.info(f"  Total Return: {metrics_ml['total_return']:.2f}%")
    logger.info(f"  Total Trades: {metrics_ml['total_trades']}")
    logger.info(f"  Win Rate: {metrics_ml['win_rate']:.2f}%")
    logger.info(f"  Profit Factor: {metrics_ml['profit_factor']:.2f}")
    logger.info(f"  Sharpe Ratio: {metrics_ml['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics_ml['max_drawdown']:.2f}%")
    logger.info("")
    logger.info("IMPROVEMENT:")
    logger.info(f"  Return: {results['comparison']['return_improvement']:+.2f}%")
    logger.info(f"  Trades: {results['comparison']['trades_reduction']:+d}")
    logger.info(f"  Win Rate: {results['comparison']['win_rate_improvement']:+.2f}%")
    logger.info(f"  Sharpe: {results['comparison']['sharpe_improvement']:+.2f}")
    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
