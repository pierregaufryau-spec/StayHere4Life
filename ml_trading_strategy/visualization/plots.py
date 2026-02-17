"""Visualization module for trading strategy results."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_equity_curve(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Equity Curve"
) -> None:
    """Plot equity curve over time.
    
    Args:
        df: DataFrame with equity column
        save_path: Path to save plot (if None, displays plot)
        title: Plot title
    """
    logger.info(f"Plotting equity curve: {title}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'equity' not in df.columns:
        raise ValueError("DataFrame must contain 'equity' column")
    
    ax.plot(df.index, df['equity'], linewidth=2, label='Equity')
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('Equity ($)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Equity curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_drawdown(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Drawdown"
) -> None:
    """Plot drawdown over time.
    
    Args:
        df: DataFrame with equity column
        save_path: Path to save plot
        title: Plot title
    """
    logger.info(f"Plotting drawdown: {title}")
    
    if 'equity' not in df.columns:
        raise ValueError("DataFrame must contain 'equity' column")
    
    # Calculate drawdown
    cummax = df['equity'].cummax()
    drawdown = (df['equity'] - cummax) / cummax * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(df.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    ax.plot(df.index, drawdown, linewidth=1, color='red')
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Drawdown plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trade_distribution(
    trades: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Trade Distribution"
) -> None:
    """Plot distribution of trade returns.
    
    Args:
        trades: List of trade dictionaries
        save_path: Path to save plot
        title: Plot title
    """
    logger.info(f"Plotting trade distribution: {title}")
    
    if not trades:
        logger.warning("No trades to plot")
        return
    
    pnls = [trade['pnl'] for trade in trades]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(pnls, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.set_xlabel('PnL ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trade PnL Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cumulative PnL
    cumulative_pnl = pd.Series(pnls).cumsum()
    ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='green')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative PnL ($)')
    ax2.set_title('Cumulative PnL by Trade')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trade distribution saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    feature_importance: Dict[str, float],
    save_path: Optional[str] = None,
    top_n: int = 20,
    title: str = "Feature Importance"
) -> None:
    """Plot feature importance from ML model.
    
    Args:
        feature_importance: Dictionary of feature names and importance values
        save_path: Path to save plot
        top_n: Number of top features to display
        title: Plot title
    """
    logger.info(f"Plotting feature importance: {title}")
    
    if not feature_importance:
        logger.warning("No feature importance data to plot")
        return
    
    # Sort by importance and take top N
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importance = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = range(len(features))
    ax.barh(y_pos, importance, alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    df_raw: pd.DataFrame,
    df_ml: pd.DataFrame,
    metrics_raw: Dict,
    metrics_ml: Dict,
    trades_raw: List[Dict],
    trades_ml: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """Plot comparison between raw and ML-enhanced strategies.
    
    Args:
        df_raw: DataFrame with raw strategy results
        df_ml: DataFrame with ML strategy results
        metrics_raw: Metrics for raw strategy
        metrics_ml: Metrics for ML strategy
        trades_raw: Trades from raw strategy
        trades_ml: Trades from ML strategy
        save_path: Path to save plot
    """
    logger.info("Plotting strategy comparison")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Equity curves comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_raw.index, df_raw['equity'], linewidth=2, label='Raw Strategy', alpha=0.8)
    ax1.plot(df_ml.index, df_ml['equity'], linewidth=2, label='ML-Enhanced Strategy', alpha=0.8)
    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curve Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Drawdown comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    cummax_raw = df_raw['equity'].cummax()
    drawdown_raw = (df_raw['equity'] - cummax_raw) / cummax_raw * 100
    
    cummax_ml = df_ml['equity'].cummax()
    drawdown_ml = (df_ml['equity'] - cummax_ml) / cummax_ml * 100
    
    ax2.fill_between(df_raw.index, drawdown_raw, 0, alpha=0.3, color='red', label='Raw')
    ax2.fill_between(df_ml.index, drawdown_ml, 0, alpha=0.3, color='blue', label='ML-Enhanced')
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Trade count comparison
    ax3 = fig.add_subplot(gs[1, 1])
    
    strategies = ['Raw', 'ML-Enhanced']
    trade_counts = [len(trades_raw), len(trades_ml)]
    win_counts = [
        sum(1 for t in trades_raw if t['pnl'] > 0),
        sum(1 for t in trades_ml if t['pnl'] > 0)
    ]
    loss_counts = [
        sum(1 for t in trades_raw if t['pnl'] <= 0),
        sum(1 for t in trades_ml if t['pnl'] <= 0)
    ]
    
    x = range(len(strategies))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], win_counts, width, label='Winning Trades', color='green', alpha=0.7)
    ax3.bar([i + width/2 for i in x], loss_counts, width, label='Losing Trades', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.set_ylabel('Number of Trades')
    ax3.set_title('Trade Count Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Metrics comparison table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    metrics_comparison = [
        ['Metric', 'Raw Strategy', 'ML-Enhanced Strategy'],
        ['Total Return (%)', f"{metrics_raw['total_return']:.2f}", f"{metrics_ml['total_return']:.2f}"],
        ['Total Trades', f"{metrics_raw['total_trades']}", f"{metrics_ml['total_trades']}"],
        ['Win Rate (%)', f"{metrics_raw['win_rate']:.2f}", f"{metrics_ml['win_rate']:.2f}"],
        ['Profit Factor', f"{metrics_raw['profit_factor']:.2f}", f"{metrics_ml['profit_factor']:.2f}"],
        ['Sharpe Ratio', f"{metrics_raw['sharpe_ratio']:.2f}", f"{metrics_ml['sharpe_ratio']:.2f}"],
        ['Max Drawdown (%)', f"{metrics_raw['max_drawdown']:.2f}", f"{metrics_ml['max_drawdown']:.2f}"],
    ]
    
    table = ax4.table(cellText=metrics_comparison, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Raw vs ML-Enhanced Strategy Comparison', fontsize=16, fontweight='bold')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
