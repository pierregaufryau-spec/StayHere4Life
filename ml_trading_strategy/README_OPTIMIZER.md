# Signal Parameter Optimizer - Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [Parameter Explanations](#parameter-explanations)
6. [Understanding Results](#understanding-results)
7. [Integration Guide](#integration-guide)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Advanced Configuration](#advanced-configuration)

---

## 🎯 Overview

The Signal Parameter Optimizer is an automated system that uses **Optuna** with the **Tree-structured Parzen Estimator (TPE)** algorithm to find optimal parameters for signal generation in trading strategies. It maximizes both signal volume and quality for LightGBM-based models.

### Key Features

- ✅ **Intelligent Optimization**: Uses Optuna's TPE algorithm for efficient parameter search
- ✅ **Composite Scoring**: Balances signal volume and quality (accuracy, F1-score)
- ✅ **Time Series Validation**: Uses TimeSeriesSplit to prevent look-ahead bias
- ✅ **Multiple Data Sources**: Supports CSV files, Yahoo Finance API, or synthetic data
- ✅ **Comprehensive Exports**: JSON, Python dict, CSV trials, and text reports
- ✅ **Visualization**: Optimization history and parameter importance plots
- ✅ **Overfitting Detection**: Built-in validation tool
- ✅ **Fallback Implementation**: Works with or without custom signal generator

### Architecture

```
signal_optimizer.py
├── ParameterSpace         # Defines search space boundaries
├── SignalQualityMetrics   # Calculates performance metrics
├── DataLoader             # Loads data from various sources
├── SignalGeneratorWrapper # Generates signals (custom or fallback)
├── LightGBMEvaluator      # Evaluates with time series CV
├── OptimizationObjective  # Optuna objective function
└── SignalOptimizerMain    # Orchestrates entire process
```

---

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements_optimizer.txt
```

The required packages are:
- numpy >= 1.21.0
- pandas >= 1.3.0
- optuna >= 3.0.0
- lightgbm >= 3.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- plotly >= 5.0.0
- kaleido >= 0.2.0 (optional, for Plotly export)
- yfinance >= 0.2.0 (optional, for Yahoo Finance data)

### 2. Verify Installation

```python
python -c "import optuna, lightgbm, pandas; print('✅ All dependencies installed')"
```

### 3. Environment Compatibility

- **Python**: 3.8+
- **Operating Systems**: Windows, Linux, macOS
- **IDEs**: Spyder, Jupyter, VSCode, PyCharm
- **Memory**: Minimum 4GB RAM recommended

---

## 🚀 Quick Start

### Basic Usage

Run with default settings (synthetic data, 100 trials):

```bash
cd ml_trading_strategy
python signal_optimizer.py
```

This will:
1. Generate 5,000 synthetic data samples
2. Run 100 optimization trials
3. Export results to `./optimized_results/`
4. Generate visualization plots

### Expected Output

```
=====================================
🚀 Signal Parameter Optimization Started
=====================================

📊 Loading data from source: synthetic
✅ Generated 5000 synthetic samples

🎯 Optimization target: 2000 signals, 55.0% accuracy
⚖️  Score weights: 60% quality, 40% volume

🔬 Starting optimization with 100 trials...
⏱️  Algorithm: Tree-structured Parzen Estimator (TPE)

Trial 0: Score=45.23, Signals=1234, Accuracy=0.587
Trial 10: Score=52.18, Signals=1678, Accuracy=0.623
...

✅ Optimization Completed!

🏆 Best Score: 67.45
📊 Signals Generated: 1845
🎯 Accuracy: 68.2%
📈 F1-Score: 65.4%

💾 Exporting results...
   📄 JSON: ./optimized_results/optimized_params_20240218_143022.json
   🐍 Python: ./optimized_results/optimized_params_20240218_143022.py
   📊 CSV Trials: ./optimized_results/optimization_trials_20240218_143022.csv
   📝 Report: ./optimized_results/optimization_report_20240218_143022.txt

📊 Generating plots...
   ✅ Plotly plots generated

🎉 Optimization Complete!
```

---

## 📖 Detailed Usage

### Using Your Own Data

#### Option 1: CSV File

```python
from signal_optimizer import SignalOptimizerMain, DEFAULT_CONFIG

# Configure for CSV data
config = DEFAULT_CONFIG.copy()
config['data']['source'] = 'csv'
config['data']['filepath'] = '/path/to/your/data.csv'

# Your CSV must have these columns:
# Date, Open, High, Low, Close, Volume

optimizer = SignalOptimizerMain(config)
optimizer.optimize()
optimizer.export_results()
optimizer.generate_plots()
```

#### Option 2: Yahoo Finance

```python
config = DEFAULT_CONFIG.copy()
config['data']['source'] = 'yfinance'
config['data']['symbol'] = 'BTC-USD'  # Any valid Yahoo Finance ticker
config['data']['period'] = '2y'       # '1d', '5d', '1mo', '1y', '2y', '5y', 'max'

optimizer = SignalOptimizerMain(config)
optimizer.optimize()
optimizer.export_results()
optimizer.generate_plots()
```

### Advanced Configuration

```python
config = {
    'data': {
        'source': 'csv',
        'filepath': '../data/BTCUSD_1h.csv',
    },
    'optimization': {
        'n_trials': 200,           # Number of optimization trials
        'min_samples': 800,        # Minimum acceptable signals
        'target_samples': 3000,    # Target number of signals
        'min_accuracy': 0.58,      # Minimum accuracy (58%)
        'quality_weight': 0.65,    # 65% quality, 35% volume
        'timeout': 3600,           # Optional timeout in seconds
    },
    'output': {
        'directory': './my_results',
        'generate_plots': True,
    }
}

optimizer = SignalOptimizerMain(config)
study = optimizer.optimize()
```

---

## 🔧 Parameter Explanations

### Optimized Parameters

The optimizer searches for optimal values in these ranges:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **rsi_period** | 7-28 | RSI calculation window |
| **rsi_oversold** | 20-35 | RSI oversold threshold |
| **rsi_overbought** | 65-85 | RSI overbought threshold |
| **ma_fast** | 5-25 | Fast moving average period |
| **ma_slow** | 30-100 | Slow moving average period |
| **volume_threshold** | 1.1-3.0 | Volume spike multiplier |
| **volatility_window** | 10-50 | Volatility calculation window |
| **volatility_threshold** | 0.5-2.5 | High volatility multiplier |
| **bb_period** | 15-30 | Bollinger Bands period |
| **bb_std** | 1.5-2.5 | Bollinger Bands standard deviations |

### Technical Indicators Generated

The fallback signal generator creates these indicators:

1. **RSI (Relative Strength Index)**: Momentum oscillator
2. **Moving Averages**: Fast/slow MA and crossover signals
3. **Volume Ratio**: Volume compared to moving average
4. **Volatility**: Rolling standard deviation of returns
5. **Bollinger Bands**: Price envelope indicators
6. **MACD**: Moving Average Convergence Divergence
7. **Trend Strength**: Normalized MA separation
8. **Signal Confidence**: Composite confidence score

---

## 📊 Understanding Results

### Composite Score Formula

```python
volume_score = min(n_signals / target_samples, 1.0) * 100
quality_score = (0.7 * f1_score + 0.3 * accuracy) * 100
composite_score = (quality_weight * quality_score + 
                   volume_weight * volume_score - penalties)
```

### Penalties Applied

- **Insufficient signals**: `(min_samples - n_signals) / min_samples * 20`
- **Low accuracy**: `(min_accuracy - accuracy) * 100`
- **Imbalanced classes**: `(0.2 - class_balance) * 50` if balance < 20%

### Interpreting Scores

| Composite Score | Interpretation |
|----------------|----------------|
| 0-30 | Poor - parameters not useful |
| 30-50 | Fair - may need refinement |
| 50-70 | Good - acceptable for testing |
| 70-85 | Very good - ready for production |
| 85+ | Excellent - exceptional performance |

### Quality Metrics

- **Accuracy**: Overall correctness (target: >55%)
- **Precision**: Positive prediction accuracy (minimize false positives)
- **Recall**: Signal detection rate (minimize false negatives)
- **F1-Score**: Harmonic mean of precision and recall (target: >60%)
- **ROC-AUC**: Area under ROC curve (discrimination ability)

---

## 🔌 Integration Guide

### Using Optimized Parameters

After optimization, integrate the results into your trading strategy:

#### Method 1: Import Python File

```python
# Import optimized parameters
from optimized_results.optimized_params_20240218_143022 import OPTIMIZED_PARAMS

# Use in your signal generator
signal_generator = SignalGenerator(OPTIMIZED_PARAMS)
signals = signal_generator.generate(data)
```

#### Method 2: Load from JSON

```python
import json

# Load parameters
with open('optimized_results/optimized_params_20240218_143022.json', 'r') as f:
    config = json.load(f)

params = config['best_parameters']

# Apply to your strategy
strategy = TradingStrategy(params)
```

#### Method 3: Direct Integration

```python
from signal_optimizer import SignalGeneratorWrapper

# Load your market data
df = pd.read_csv('market_data.csv')

# Generate signals with optimized parameters
signal_gen = SignalGeneratorWrapper(df)
df_signals = signal_gen.generate_signals(OPTIMIZED_PARAMS)

# Use signals in your strategy
buy_signals = df_signals[df_signals['Signal'] == 1]
```

---

## 🔍 Overfitting Verification

### Why Verify?

Optimized parameters may overfit to training data. Always validate on unseen data.

### Usage

```bash
# Verify with your test data
python verify_overfitting.py \
    optimized_results/optimized_params_20240218_143022.json \
    --test-data ../data/test_market_data.csv
```

### Interpreting Results

| Severity | Action |
|----------|--------|
| **NONE** (Δ<5%) | ✅ Safe for production |
| **LOW** (Δ<10%) | ⚠️ Acceptable with monitoring |
| **MODERATE** (Δ<15%) | ⚠️ Use with caution, consider re-optimization |
| **HIGH** (Δ>15%) | ❌ DO NOT use in production |

Δ = Difference between train and test accuracy/F1

---

## 📈 Visualization

### Using Plotly (Preferred)

If kaleido is installed, plots are generated automatically:

```python
optimizer.generate_plots()
```

Generates:
- `optimization_history_*.png`: Score progression over trials
- `param_importances_*.png`: Which parameters matter most

### Using Matplotlib Alternative

If kaleido is not available:

```bash
python plot_results_alternative.py \
    optimized_results/optimization_trials_20240218_143022.csv
```

Generates:
- `optimization_history_matplotlib.png`: Trial progression
- `score_distribution_matplotlib.png`: Score statistics
- `parameter_correlations_matplotlib.png`: Parameter importance

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Module not found" Error

```bash
# Install missing dependencies
pip install -r requirements_optimizer.txt
```

#### 2. Insufficient Signals Generated

**Problem**: Optimizer generates too few signals

**Solutions**:
- Lower `min_samples` threshold
- Expand parameter search space
- Use more diverse data
- Relax signal generation conditions

#### 3. Low Quality Scores

**Problem**: Accuracy/F1 below targets

**Solutions**:
- Increase `quality_weight` in config
- Add more training data
- Use feature engineering
- Try different parameter ranges

#### 4. Optimization Takes Too Long

**Problem**: 100 trials take hours

**Solutions**:
```python
config['optimization']['n_trials'] = 50  # Reduce trials
config['optimization']['timeout'] = 1800  # 30-minute timeout
```

#### 5. Kaleido Import Error

**Problem**: Cannot generate Plotly plots

**Solutions**:
```bash
# Use matplotlib alternative
python plot_results_alternative.py trials.csv

# Or install kaleido
pip install kaleido==0.2.1
```

#### 6. Memory Issues

**Problem**: Out of memory during optimization

**Solutions**:
- Reduce data samples
- Use smaller `n_splits` in cross-validation
- Process data in chunks

---

## 🏆 Best Practices

### 1. Data Preparation

✅ **DO:**
- Use at least 2,000+ samples
- Ensure clean, complete OHLCV data
- Include multiple market conditions (bull, bear, sideways)
- Verify data quality before optimization

❌ **DON'T:**
- Use data with excessive gaps
- Mix different timeframes
- Include forward-filled missing values

### 2. Optimization Strategy

✅ **DO:**
- Start with 50-100 trials for exploration
- Use 200+ trials for production parameters
- Set realistic targets for your market
- Balance quality vs volume based on strategy

❌ **DON'T:**
- Over-optimize on limited data
- Ignore the composite score
- Skip overfitting verification
- Use synthetic data for final validation

### 3. Parameter Usage

✅ **DO:**
- Test on out-of-sample data first
- Monitor performance over time
- Re-optimize periodically (monthly/quarterly)
- Keep track of parameter versions

❌ **DON'T:**
- Deploy without backtesting
- Use parameters indefinitely
- Ignore changing market conditions
- Skip A/B testing

### 4. Production Deployment

✅ **DO:**
- Verify on multiple time periods
- Implement gradual rollout
- Set up monitoring and alerts
- Keep fallback parameters ready

❌ **DON'T:**
- Deploy high-overfitting parameters
- Trust single-period validation
- Ignore real-time performance drift
- Skip proper risk management

---

## ⚙️ Advanced Configuration

### Custom Parameter Space

Modify `ParameterSpace` class to add/change parameters:

```python
from dataclasses import dataclass

@dataclass
class CustomParameterSpace:
    # Add your custom parameter ranges
    atr_period_min: int = 10
    atr_period_max: int = 30
    atr_multiplier_min: float = 1.5
    atr_multiplier_max: float = 3.5
    
    # Include standard parameters
    rsi_period_min: int = 7
    rsi_period_max: int = 28
    # ... etc
```

### Custom Objective Function

Create specialized scoring:

```python
class CustomObjective(OptimizationObjective):
    def _calculate_composite_score(self, metrics, n_signals):
        # Your custom scoring logic
        sharpe_ratio = self._calculate_sharpe(metrics)
        profit_factor = self._calculate_profit_factor(metrics)
        
        return sharpe_ratio * 40 + profit_factor * 60
```

### Multi-Objective Optimization

Optimize for multiple goals:

```python
from optuna.multi_objective import MultiObjectiveStudy

# Define multiple objectives
def multi_objective(trial):
    params = self._suggest_parameters(trial)
    results = self._evaluate(params)
    
    return results['accuracy'], results['n_signals']  # Returns tuple

# Create multi-objective study
study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=100)
```

### Distributed Optimization

Scale with parallel execution:

```python
# Use Optuna's distributed optimization
import optuna

# Create persistent storage
storage = "sqlite:///optimization.db"

study = optuna.create_study(
    study_name='distributed_optimization',
    storage=storage,
    load_if_exists=True,
    sampler=TPESampler(seed=42)
)

# Run on multiple machines/cores
study.optimize(objective, n_trials=100)
```

---

## 📚 References

### Documentation

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [LightGBM Guide](https://lightgbm.readthedocs.io/)
- [scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

### Related Papers

- Tree-structured Parzen Estimator: [Bergstra et al., 2011]
- Time Series Cross-Validation: [Hyndman & Athanasopoulos, 2018]
- Trading Signal Generation: [Pardo, 2008]

---

## 📝 Changelog

### Version 1.0.0 (2024-02-18)

**Initial Release**
- ✅ Optuna-based optimization with TPE
- ✅ Composite scoring system
- ✅ Time series cross-validation
- ✅ Multiple data source support
- ✅ Comprehensive export formats
- ✅ Overfitting verification tool
- ✅ Matplotlib visualization alternative
- ✅ Full documentation

---

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue or submit a pull request.

## 📄 License

MIT License - feel free to use in your trading strategies!

---

## ⚠️ Disclaimer

**IMPORTANT**: This tool is for educational and research purposes. Past performance does not guarantee future results. Always:

- Backtest thoroughly before live trading
- Use proper risk management
- Never risk more than you can afford to lose
- Consult with financial professionals
- Test in paper trading first

**Trading involves substantial risk of loss and is not suitable for every investor.**

---

## 🆘 Support

Need help?

1. Check this README first
2. Review the troubleshooting section
3. Examine the example outputs in `optimized_results/`
4. Open an issue with:
   - Your configuration
   - Error messages
   - Data characteristics
   - Environment details

Happy optimizing! 🚀📈
