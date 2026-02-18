"""
ML Trading Strategy - Signal Optimizer Module
==============================================

Automated parameter optimization system for trading signal generation.

Main Components:
- signal_optimizer.py: Core optimization engine with Optuna
- verify_overfitting.py: Validation tool for parameter robustness
- plot_results_alternative.py: Matplotlib-based visualization
- README_OPTIMIZER.md: Complete documentation

Quick Start:
    >>> python signal_optimizer.py

For detailed usage, see README_OPTIMIZER.md
"""

__version__ = "1.0.0"
__author__ = "ML Trading Strategy Team"
__license__ = "MIT"

# Make key classes available at package level
try:
    from .signal_optimizer import (
        SignalOptimizerMain,
        SignalGeneratorWrapper,
        LightGBMEvaluator,
        ParameterSpace,
        DEFAULT_CONFIG,
    )
    
    __all__ = [
        'SignalOptimizerMain',
        'SignalGeneratorWrapper',
        'LightGBMEvaluator',
        'ParameterSpace',
        'DEFAULT_CONFIG',
    ]
except ImportError:
    # Allow import of package even if dependencies aren't installed
    pass
