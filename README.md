# ML Trading Strategy

A modular, production-ready trading strategy framework with machine learning enhancement. This project transforms traditional technical analysis signals into ML-enhanced trading decisions through a clean, maintainable architecture.

## Features

- **Modular Architecture**: Clean separation of concerns with independent, reusable modules
- **Comprehensive Technical Analysis**: 70+ technical indicators including EMAs, RSI, MACD, ADX, Bollinger Bands, and more
- **Machine Learning Enhancement**: LightGBM classifier with feature selection and hyperparameter optimization
- **Professional Backtesting**: Full backtest engine with stop-loss, take-profit, and position sizing
- **Risk Management**: ATR-based position sizing and risk control
- **Visualization**: Comprehensive charts and comparison plots
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Standalone Scripts**: Independent scripts for training, backtesting, and optimization

## Project Structure

```
ml_trading_strategy/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── config.py          # Config loading and validation
├── data/                  # Data handling
│   ├── __init__.py
│   └── data_loader.py     # Data loading and validation
├── indicators/            # Technical indicators
│   ├── __init__.py
│   └── technical_indicators.py
├── signals/               # Signal generation
│   ├── __init__.py
│   └── signal_generator.py
├── features/              # Feature engineering
│   ├── __init__.py
│   └── feature_engineering.py
├── models/                # ML models
│   ├── __init__.py
│   ├── ml_trainer.py      # Model training
│   └── ml_predictor.py    # Model prediction
├── backtest/              # Backtesting engine
│   ├── __init__.py
│   ├── backtest_engine.py # Backtest execution
│   └── metrics.py         # Performance metrics
└── visualization/         # Plotting and visualization
    ├── __init__.py
    └── plots.py           # Chart generation

scripts/                   # Standalone scripts
├── 01_train_model.py     # Train ML model
├── 02_run_backtest.py    # Run backtest with model
└── 03_optimize.py        # Hyperparameter optimization

config.yaml               # Configuration file
requirements.txt          # Python dependencies
main.py                   # Main orchestration script
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pierregaufryau-spec/StayHere4Life.git
cd StayHere4Life
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run Complete Pipeline

The easiest way to get started is to run the complete pipeline:

```bash
python main.py
```

This will:
1. Load configuration from `config.yaml`
2. Generate sample data (or load from CSV if available)
3. Train an ML model
4. Run backtests for both raw and ML-enhanced strategies
5. Generate comparison plots and reports

### Run Individual Steps

#### 1. Train ML Model

```bash
python scripts/01_train_model.py
```

Options:
- `--config CONFIG`: Path to config file (default: config.yaml)
- `--model-name NAME`: Custom name for saved model
- `--no-optimize`: Skip hyperparameter optimization
- `--trials N`: Number of optimization trials

Example:
```bash
python scripts/01_train_model.py --trials 50
```

#### 2. Run Backtest

```bash
python scripts/02_run_backtest.py
```

Options:
- `--config CONFIG`: Path to config file
- `--model MODEL`: Path to trained model (default: saved_models/model_latest.pkl)
- `--threshold THRESHOLD`: ML prediction threshold
- `--output DIR`: Output directory for results

Example:
```bash
python scripts/02_run_backtest.py --threshold 0.4
```

#### 3. Optimize Parameters

```bash
python scripts/03_optimize.py --trials 100
```

Options:
- `--config CONFIG`: Path to config file
- `--trials N`: Number of optimization trials
- `--output FILE`: Output file for best parameters
- `--optimize-signals`: Optimize signal parameters instead of ML

Example:
```bash
# Optimize ML parameters
python scripts/03_optimize.py --trials 50

# Optimize signal parameters
python scripts/03_optimize.py --trials 50 --optimize-signals
```

## Configuration

Edit `config.yaml` to customize the strategy:

### Data Configuration
```yaml
data:
  csv_path: btcusd_4h.csv      # Path to your data file
  generate_sample: true         # Generate sample data if file missing
  sample_size: 3000            # Number of bars for sample data
```

### Trading Parameters
```yaml
trading:
  initial_capital: 100000      # Starting capital
  risk_per_trade: 0.01         # Risk 1% per trade
  atr_sl_multiplier: 1.5       # Stop loss = 1.5 × ATR
  atr_tp_multiplier: 3.0       # Take profit = 3.0 × ATR
  commission: 0.001            # 0.1% commission per trade
```

### Signal Parameters
```yaml
signals:
  min_ema_distance: 0.05       # Minimum 5% EMA separation
  adx_threshold: 25            # ADX must be > 25
  volume_multiplier: 1.1       # Volume must be 1.1× average
```

### ML Parameters
```yaml
ml:
  forward_bars: 5              # Look ahead 5 bars for labels
  label_threshold: 0.003       # 0.3% price change threshold
  n_features: 20               # Select top 20 features
  feature_selection_method: importance
  optimize: true               # Use Optuna optimization
  n_trials: 30                 # Number of optimization trials
  ml_threshold: 0.35           # ML prediction threshold
  test_size: 0.3               # 30% test split
  random_seed: 42              # For reproducibility
```

## Using Your Own Data

To use your own data instead of generated sample data:

1. Prepare a CSV file with these columns:
   - `timestamp`: Date/time of the bar
   - `open`: Opening price
   - `high`: Highest price
   - `low`: Lowest price
   - `close`: Closing price
   - `volume`: Trading volume

2. Update `config.yaml`:
```yaml
data:
  csv_path: path/to/your/data.csv
  generate_sample: false
```

3. Run the pipeline as normal

## Module Documentation

### Config Module

Handles configuration loading, validation, and management.

```python
from ml_trading_strategy.config import load_config

# Load from file
config = load_config('config.yaml')

# Get values using dot notation
csv_path = config.get('data.csv_path')
initial_capital = config.get('trading.initial_capital')

# Set values
config.set('ml.n_features', 30)

# Save configuration
config.save('new_config.yaml')
```

### Data Module

Loads and validates OHLCV data.

```python
from ml_trading_strategy.data import prepare_data, load_data, generate_sample_data

# Prepare data (load or generate if missing)
df = prepare_data('btcusd_4h.csv', generate_if_missing=True)

# Load from CSV
df = load_data('data.csv')

# Generate sample data
df = generate_sample_data(n=3000)
```

### Indicators Module

Computes technical indicators.

```python
from ml_trading_strategy.indicators import compute_indicators

# Compute all indicators
df = compute_indicators(df)

# Result includes: EMAs, RSI, MACD, Bollinger Bands, ADX, ATR, etc.
```

### Signals Module

Generates trading signals based on technical conditions.

```python
from ml_trading_strategy.signals import generate_raw_signals

# Generate signals
df = generate_raw_signals(
    df,
    min_ema_distance=0.05,
    adx_threshold=25,
    volume_multiplier=1.1
)

# Result includes: raw_signal, long_signal, short_signal columns
```

### Features Module

Creates features and labels for ML.

```python
from ml_trading_strategy.features import create_features, create_labels, get_feature_columns

# Create features
df = create_features(df)

# Create labels
df = create_labels(df, forward_bars=5, threshold=0.003)

# Get feature columns
feature_cols = get_feature_columns(df)
```

### Models Module

Train and use ML models.

```python
from ml_trading_strategy.models import MLTrainer, MLPredictor

# Train model
trainer = MLTrainer(config.to_dict()['ml'])
model, scaler, metrics = trainer.train(df)
model_path = trainer.save_model('saved_models')

# Load and predict
predictor = MLPredictor(model_path)
df = predictor.predict(df, threshold=0.35)
```

### Backtest Module

Run backtests and compute metrics.

```python
from ml_trading_strategy.backtest import backtest, compute_metrics

# Run backtest
df_result, trades, metrics = backtest(
    df,
    signal_col='ml_signal',
    config=trading_config
)

# Metrics include: total_return, win_rate, sharpe_ratio, max_drawdown, etc.
```

### Visualization Module

Generate charts and plots.

```python
from ml_trading_strategy.visualization import (
    plot_comparison,
    plot_equity_curve,
    plot_drawdown,
    plot_feature_importance
)

# Plot strategy comparison
plot_comparison(df_raw, df_ml, metrics_raw, metrics_ml, 
                trades_raw, trades_ml, save_path='comparison.png')

# Plot equity curve
plot_equity_curve(df, save_path='equity.png')

# Plot drawdown
plot_drawdown(df, save_path='drawdown.png')

# Plot feature importance
plot_feature_importance(feature_dict, save_path='importance.png')
```

## Strategy Logic

### Signal Generation

The strategy generates signals based on multiple technical conditions:

**Long Signals** (Buy):
- EMA alignment: 9 > 21 > 50
- Minimum EMA distance (avoid choppy markets)
- ADX > threshold (strong trend)
- Volume > average × multiplier
- RSI < 70 (not overbought)
- MACD bullish
- Price above EMA 50

**Short Signals** (Sell):
- EMA alignment: 9 < 21 < 50
- Minimum EMA distance
- ADX > threshold (strong trend)
- Volume > average × multiplier
- RSI > 30 (not oversold)
- MACD bearish
- Price below EMA 50

### ML Enhancement

The ML model learns from historical patterns to filter signals:

1. **Feature Engineering**: Creates 70+ features from technical indicators
2. **Label Creation**: Labels based on forward price movement
3. **Feature Selection**: Selects top N most important features
4. **Model Training**: LightGBM classifier with hyperparameter optimization
5. **Signal Filtering**: Only takes trades with high ML confidence

### Risk Management

- **Position Sizing**: ATR-based risk per trade (default: 1% of equity)
- **Stop Loss**: Dynamic stop loss at entry ± (ATR × multiplier)
- **Take Profit**: Dynamic take profit at entry ± (ATR × multiplier × TP ratio)
- **Commission**: Realistic commission costs included

## Performance Metrics

The framework calculates comprehensive metrics:

- **Return Metrics**: Total return, PnL
- **Trade Metrics**: Win rate, profit factor, average trade
- **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown
- **Trade Analysis**: Long/short breakdown, holding periods

## Output Files

After running the pipeline, you'll find:

```
saved_models/
├── model_latest.pkl           # Latest trained model
├── model_YYYYMMDD_HHMMSS.pkl # Timestamped models
└── *_metadata.json            # Model metadata

results/
└── backtest_YYYYMMDD_HHMMSS/
    ├── backtest_results.json  # Detailed metrics
    └── plots/
        ├── comparison.png     # Strategy comparison
        ├── equity_raw.png     # Raw strategy equity
        ├── equity_ml.png      # ML strategy equity
        ├── drawdown_raw.png   # Raw strategy drawdown
        ├── drawdown_ml.png    # ML strategy drawdown
        ├── trades_raw.png     # Raw trades distribution
        └── trades_ml.png      # ML trades distribution

plots/
├── feature_importance.png     # Feature importance chart
└── strategy_comparison.png    # Latest comparison
```

## Tips and Best Practices

1. **Start with defaults**: Run the pipeline with default settings first
2. **Optimize gradually**: Use `03_optimize.py` to find better parameters
3. **Use your own data**: Replace sample data with real market data
4. **Monitor overfitting**: Check train vs test metrics for overfitting
5. **Adjust thresholds**: Lower ML threshold for more trades, higher for quality
6. **Backtest properly**: Use out-of-sample data for realistic performance
7. **Consider slippage**: Real trading has more slippage than backtests

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running scripts from the project root:
```bash
cd /path/to/StayHere4Life
python main.py
```

### Missing Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

### No Model Found

Train a model first:
```bash
python scripts/01_train_model.py
```

### Poor Performance

Try:
1. Optimize parameters: `python scripts/03_optimize.py --trials 100`
2. Adjust ML threshold in config.yaml
3. Use real market data instead of synthetic data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Disclaimer

This software is for educational purposes only. Do not use it for real trading without proper testing and risk management. Past performance does not guarantee future results. Trading involves substantial risk of loss.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Technical indicators powered by [ta](https://github.com/bukosabino/ta) library
- Machine learning with [LightGBM](https://github.com/microsoft/LightGBM)
- Hyperparameter optimization with [Optuna](https://optuna.org/)
