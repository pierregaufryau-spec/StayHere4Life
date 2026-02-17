"""Metrics calculation module for backtesting."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_drawdown(equity_series: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve.
    
    Drawdown represents the peak-to-trough decline in equity.
    
    Args:
        equity_series: Series of equity values over time
        
    Returns:
        Series of drawdown values (as negative percentages)
    """
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    return drawdown


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: int = 2190) -> float:
    """Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (default: 2190 for 4h bars)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return float(sharpe)


def compute_sortino_ratio(returns: pd.Series, periods_per_year: int = 2190) -> float:
    """Calculate annualized Sortino ratio.
    
    Similar to Sharpe but only considers downside volatility.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = returns.mean()
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    sortino = (mean_return / downside_std) * np.sqrt(periods_per_year)
    
    return float(sortino)


def compute_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate from trades.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Win rate as percentage (0-100)
    """
    if len(trades) == 0:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
    win_rate = (winning_trades / len(trades)) * 100
    
    return float(win_rate)


def compute_profit_factor(trades: List[Dict]) -> float:
    """Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Profit factor
    """
    if len(trades) == 0:
        return 0.0
    
    gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
    gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    
    return float(profit_factor)


def compute_average_trade(trades: List[Dict]) -> Dict[str, float]:
    """Calculate average trade statistics.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with average trade statistics
    """
    if len(trades) == 0:
        return {
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_bars': 0.0
        }
    
    pnls = [trade['pnl'] for trade in trades]
    winning_pnls = [pnl for pnl in pnls if pnl > 0]
    losing_pnls = [pnl for pnl in pnls if pnl < 0]
    
    avg_trade = np.mean(pnls)
    avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
    avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
    
    # Calculate average bars held
    bars_held = [trade['exit_idx'] - trade['entry_idx'] for trade in trades]
    avg_bars = np.mean(bars_held) if bars_held else 0.0
    
    return {
        'avg_trade': float(avg_trade),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'avg_bars': float(avg_bars)
    }


def compute_metrics(
    df: pd.DataFrame,
    trades: List[Dict],
    initial_capital: float
) -> Dict[str, float]:
    """Compute comprehensive trading metrics.
    
    Args:
        df: DataFrame with backtest results including equity curve
        trades: List of executed trades
        initial_capital: Starting capital
        
    Returns:
        Dictionary of performance metrics
    """
    logger.info("Computing performance metrics")
    
    metrics = {}
    
    # Basic metrics
    final_equity = df['equity'].iloc[-1] if len(df) > 0 else initial_capital
    metrics['initial_capital'] = float(initial_capital)
    metrics['final_equity'] = float(final_equity)
    metrics['total_return'] = float((final_equity - initial_capital) / initial_capital * 100)
    metrics['total_pnl'] = float(final_equity - initial_capital)
    
    # Trade statistics
    metrics['total_trades'] = len(trades)
    
    if len(trades) > 0:
        metrics['win_rate'] = compute_win_rate(trades)
        metrics['profit_factor'] = compute_profit_factor(trades)
        
        avg_stats = compute_average_trade(trades)
        metrics.update(avg_stats)
        
        # Trade breakdown
        long_trades = [t for t in trades if t['direction'] == 1]
        short_trades = [t for t in trades if t['direction'] == -1]
        
        metrics['long_trades'] = len(long_trades)
        metrics['short_trades'] = len(short_trades)
        
        if long_trades:
            metrics['long_win_rate'] = compute_win_rate(long_trades)
        else:
            metrics['long_win_rate'] = 0.0
            
        if short_trades:
            metrics['short_win_rate'] = compute_win_rate(short_trades)
        else:
            metrics['short_win_rate'] = 0.0
    else:
        metrics['win_rate'] = 0.0
        metrics['profit_factor'] = 0.0
        metrics['avg_trade'] = 0.0
        metrics['avg_win'] = 0.0
        metrics['avg_loss'] = 0.0
        metrics['avg_bars'] = 0.0
        metrics['long_trades'] = 0
        metrics['short_trades'] = 0
        metrics['long_win_rate'] = 0.0
        metrics['short_win_rate'] = 0.0
    
    # Equity curve metrics
    if 'equity' in df.columns and len(df) > 0:
        # Returns
        equity_returns = df['equity'].pct_change().dropna()
        
        if len(equity_returns) > 0:
            metrics['sharpe_ratio'] = compute_sharpe_ratio(equity_returns)
            metrics['sortino_ratio'] = compute_sortino_ratio(equity_returns)
        else:
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
        
        # Drawdown
        drawdown = compute_drawdown(df['equity'])
        metrics['max_drawdown'] = float(drawdown.min() * 100)
        
        # Find max drawdown duration
        is_drawdown = drawdown < 0
        if is_drawdown.any():
            # Find continuous drawdown periods
            drawdown_periods = (is_drawdown != is_drawdown.shift()).cumsum()
            drawdown_lengths = drawdown_periods[is_drawdown].value_counts()
            metrics['max_drawdown_duration'] = int(drawdown_lengths.max())
        else:
            metrics['max_drawdown_duration'] = 0
    else:
        metrics['sharpe_ratio'] = 0.0
        metrics['sortino_ratio'] = 0.0
        metrics['max_drawdown'] = 0.0
        metrics['max_drawdown_duration'] = 0
    
    # Risk metrics
    if metrics['total_trades'] > 0:
        # Expectancy
        win_rate_decimal = metrics['win_rate'] / 100
        metrics['expectancy'] = (
            win_rate_decimal * metrics['avg_win'] +
            (1 - win_rate_decimal) * metrics['avg_loss']
        )
    else:
        metrics['expectancy'] = 0.0
    
    # Log summary
    logger.info(f"Metrics computed:")
    logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
    logger.info(f"  Total Trades: {metrics['total_trades']}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    return metrics
