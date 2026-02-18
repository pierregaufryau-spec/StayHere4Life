# Quick Reference Guide

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements_optimizer.txt

# Run with defaults (synthetic data, 100 trials)
python signal_optimizer.py

# Run examples
python example_usage.py
```

## 📊 Basic Usage Pattern

```python
from signal_optimizer import SignalOptimizerMain, DEFAULT_CONFIG

# 1. Configure
config = DEFAULT_CONFIG.copy()
config['optimization']['n_trials'] = 100

# 2. Optimize
optimizer = SignalOptimizerMain(config)
study = optimizer.optimize()

# 3. Export
optimizer.export_results()
optimizer.generate_plots()
```

## 🔧 Common Configuration Options

### Data Sources
```python
# CSV file
config['data']['source'] = 'csv'
config['data']['filepath'] = 'data/BTCUSD.csv'

# Yahoo Finance
config['data']['source'] = 'yfinance'
config['data']['symbol'] = 'BTC-USD'
config['data']['period'] = '1y'

# Synthetic (for testing)
config['data']['source'] = 'synthetic'
config['data']['n_samples'] = 5000
```

### Optimization Settings
```python
config['optimization'] = {
    'n_trials': 100,          # Number of optimization trials
    'min_samples': 500,       # Minimum acceptable signals
    'target_samples': 2000,   # Optimal signal count
    'min_accuracy': 0.55,     # Minimum accuracy (55%)
    'quality_weight': 0.6,    # 60% quality, 40% volume
}
```

### Quality vs Volume Trade-off
```python
# High quality (fewer but better signals)
config['optimization']['quality_weight'] = 0.7
config['optimization']['min_accuracy'] = 0.60
config['optimization']['target_samples'] = 1500

# High volume (more signals, slightly lower quality)
config['optimization']['quality_weight'] = 0.4
config['optimization']['min_accuracy'] = 0.52
config['optimization']['target_samples'] = 3000

# Balanced (default)
config['optimization']['quality_weight'] = 0.6
config['optimization']['min_accuracy'] = 0.55
config['optimization']['target_samples'] = 2000
```

## 📈 Using Optimized Parameters

### Method 1: Import Python File
```python
from optimized_results.optimized_params_TIMESTAMP import OPTIMIZED_PARAMS

# Use parameters
signal_generator = SignalGenerator(OPTIMIZED_PARAMS)
```

### Method 2: Load JSON
```python
import json

with open('optimized_results/optimized_params_TIMESTAMP.json') as f:
    config = json.load(f)
    params = config['best_parameters']
```

### Method 3: Direct Usage
```python
from signal_optimizer import SignalGeneratorWrapper

df = pd.read_csv('market_data.csv')
signal_gen = SignalGeneratorWrapper(df)
df_signals = signal_gen.generate_signals(OPTIMIZED_PARAMS)
```

## 🔍 Verification Commands

```bash
# Verify with synthetic test data
python verify_overfitting.py optimized_results/optimized_params_TIMESTAMP.json

# Verify with real test data
python verify_overfitting.py optimized_results/optimized_params_TIMESTAMP.json \
    --test-data data/test_market.csv

# Generate alternative plots
python plot_results_alternative.py optimized_results/optimization_trials_TIMESTAMP.csv
```

## 📊 Interpreting Results

### Composite Score Ranges
- **0-30**: Poor, not useful
- **30-50**: Fair, needs refinement
- **50-70**: Good, ready for testing
- **70-85**: Very good, production ready
- **85+**: Excellent performance

### Quality Metrics Targets
- **Accuracy**: >55% (ideally 60-75%)
- **F1-Score**: >50% (ideally 60-75%)
- **Class Balance**: >20% each class
- **Signal Count**: 500-3000 (depends on strategy)

### Overfitting Severity
- **NONE** (Δ<5%): ✅ Safe for production
- **LOW** (Δ<10%): ⚠️ Acceptable with monitoring
- **MODERATE** (Δ<15%): ⚠️ Use with caution
- **HIGH** (Δ>15%): ❌ Do not use

*Δ = Difference between train/test accuracy or F1*

## 🎯 Parameter Ranges (Search Space)

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| rsi_period | 7 | 28 | RSI window |
| rsi_oversold | 20 | 35 | Buy threshold |
| rsi_overbought | 65 | 85 | Sell threshold |
| ma_fast | 5 | 25 | Fast MA period |
| ma_slow | 30 | 100 | Slow MA period |
| volume_threshold | 1.1 | 3.0 | Volume spike |
| volatility_window | 10 | 50 | Vol window |
| volatility_threshold | 0.5 | 2.5 | High vol |
| bb_period | 15 | 30 | BB period |
| bb_std | 1.5 | 2.5 | BB std devs |

## 🐛 Common Issues

### "Module not found"
```bash
pip install -r requirements_optimizer.txt
```

### Too few signals
- Lower `min_samples` threshold
- Expand parameter search space
- Relax signal conditions

### Low quality scores
- Increase `quality_weight`
- Add more training data
- Adjust parameter ranges

### Optimization too slow
```python
config['optimization']['n_trials'] = 50  # Reduce
config['optimization']['timeout'] = 1800  # 30 min limit
```

### Kaleido not available
```bash
# Use matplotlib alternative
python plot_results_alternative.py trials.csv
```

## 📁 Output Files

All results saved to `./optimized_results/`:
- `optimized_params_*.json` - Parameters + metrics
- `optimized_params_*.py` - Python importable dict
- `optimization_trials_*.csv` - All trial results
- `optimization_report_*.txt` - Human-readable report
- `optimization_history_*.png` - Score progression
- `param_importances_*.png` - Parameter importance
- `overfitting_report_*.txt` - Validation results

## 🔗 Useful Links

- Full Documentation: [README_OPTIMIZER.md](README_OPTIMIZER.md)
- Examples: Run `python example_usage.py`
- Optuna Docs: https://optuna.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/

## ⚡ Performance Tips

1. Start with 50-100 trials for exploration
2. Use 200+ trials for production parameters
3. Include multiple market conditions in data
4. Re-optimize monthly or quarterly
5. Always verify on out-of-sample data
6. Monitor live performance continuously

## ⚠️ Important Notes

- Always backtest before live trading
- Past performance ≠ future results
- Use proper risk management
- Start with paper trading
- Monitor for performance drift

---

**Need more help?** See [README_OPTIMIZER.md](README_OPTIMIZER.md) for complete documentation.
