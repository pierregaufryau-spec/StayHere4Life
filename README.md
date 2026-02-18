# StayHere4Life
ML-Enhanced Trading Strategy with modular architecture

## 📦 Components

### Signal Parameter Optimizer
An automated parameter optimization system using Optuna and LightGBM to find optimal trading signal parameters.

**Quick Start:**
```bash
cd ml_trading_strategy
pip install -r ../requirements_optimizer.txt
python signal_optimizer.py
```

**Documentation:**
- [Full Guide](ml_trading_strategy/README_OPTIMIZER.md) - Complete documentation
- [Quick Reference](ml_trading_strategy/QUICK_REFERENCE.md) - Common commands and patterns
- [Examples](ml_trading_strategy/example_usage.py) - Interactive examples

**Features:**
- ✅ Intelligent parameter search with Optuna TPE algorithm
- ✅ Composite scoring (quality + volume)
- ✅ Time series cross-validation
- ✅ Multiple data sources (CSV, Yahoo Finance, synthetic)
- ✅ Overfitting detection
- ✅ Comprehensive exports and visualizations
