"""Backtesting engine for trading strategy."""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for trading strategies.
    
    Handles position management, stop loss/take profit execution,
    and tracks equity over time.
    """
    
    def __init__(self, config: Dict):
        """Initialize backtest engine.
        
        Args:
            config: Configuration dictionary with trading parameters
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.5)
        self.atr_tp_multiplier = config.get('atr_tp_multiplier', 3.0)
        self.commission = config.get('commission', 0.001)
        
        # State variables
        self.equity = self.initial_capital
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0.0
        self.entry_idx = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0.0
        
        self.trades = []
        self.equity_curve = []
    
    def calculate_position_size(self, price: float, atr: float, direction: int) -> float:
        """Calculate position size based on risk management.
        
        Uses ATR-based risk management to size positions.
        
        Args:
            price: Entry price
            atr: Average True Range value
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            Position size in units
        """
        risk_amount = self.equity * self.risk_per_trade
        stop_distance = atr * self.atr_sl_multiplier
        
        if stop_distance == 0:
            return 0.0
        
        position_size = risk_amount / stop_distance
        
        # Limit position size to available capital
        max_size = self.equity / price
        position_size = min(position_size, max_size * 0.95)  # Keep 5% buffer
        
        return position_size
    
    def enter_position(self, idx: int, price: float, atr: float, direction: int) -> None:
        """Enter a new position.
        
        Args:
            idx: Index in the dataframe
            price: Entry price
            atr: ATR value for stop loss calculation
            direction: 1 for long, -1 for short
        """
        if self.position != 0:
            logger.warning(f"Attempting to enter position while already in position at idx {idx}")
            return
        
        # Calculate position size
        position_size = self.calculate_position_size(price, atr, direction)
        
        if position_size <= 0:
            logger.warning(f"Invalid position size at idx {idx}")
            return
        
        # Calculate stop loss and take profit
        stop_distance = atr * self.atr_sl_multiplier
        tp_distance = atr * self.atr_tp_multiplier
        
        if direction == 1:  # Long
            self.stop_loss = price - stop_distance
            self.take_profit = price + tp_distance
        else:  # Short
            self.stop_loss = price + stop_distance
            self.take_profit = price - tp_distance
        
        # Apply commission
        commission_cost = position_size * price * self.commission
        self.equity -= commission_cost
        
        # Set position state
        self.position = direction
        self.entry_price = price
        self.entry_idx = idx
        self.position_size = position_size
        
        logger.debug(f"Entered {('long' if direction == 1 else 'short')} position at idx {idx}: "
                    f"price={price:.2f}, size={position_size:.4f}, SL={self.stop_loss:.2f}, "
                    f"TP={self.take_profit:.2f}")
    
    def exit_position(self, idx: int, price: float, reason: str) -> None:
        """Exit current position.
        
        Args:
            idx: Index in the dataframe
            price: Exit price
            reason: Reason for exit ('signal', 'stop_loss', 'take_profit')
        """
        if self.position == 0:
            return
        
        # Calculate PnL
        if self.position == 1:  # Long
            pnl = (price - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - price) * self.position_size
        
        # Apply commission
        commission_cost = self.position_size * price * self.commission
        pnl -= commission_cost
        
        # Update equity
        self.equity += pnl
        
        # Record trade
        trade = {
            'entry_idx': self.entry_idx,
            'exit_idx': idx,
            'direction': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'position_size': self.position_size,
            'pnl': pnl,
            'return_pct': (pnl / (self.entry_price * self.position_size)) * 100,
            'reason': reason,
            'bars_held': idx - self.entry_idx
        }
        self.trades.append(trade)
        
        logger.debug(f"Exited {('long' if self.position == 1 else 'short')} position at idx {idx}: "
                    f"price={price:.2f}, pnl={pnl:.2f}, reason={reason}")
        
        # Reset position state
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0.0
    
    def check_exit_conditions(self, idx: int, high: float, low: float) -> Tuple[bool, str]:
        """Check if position should be exited based on stop loss or take profit.
        
        Args:
            idx: Current index
            high: High price of current bar
            low: Low price of current bar
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.position == 0:
            return False, ''
        
        if self.position == 1:  # Long position
            # Check stop loss
            if low <= self.stop_loss:
                return True, 'stop_loss'
            # Check take profit
            if high >= self.take_profit:
                return True, 'take_profit'
        
        elif self.position == -1:  # Short position
            # Check stop loss
            if high >= self.stop_loss:
                return True, 'stop_loss'
            # Check take profit
            if low <= self.take_profit:
                return True, 'take_profit'
        
        return False, ''
    
    def process_bar(self, idx: int, row: pd.Series, signal: int) -> None:
        """Process a single bar of data.
        
        Args:
            idx: Index in the dataframe
            row: Series with OHLC data and indicators
            signal: Trading signal (1 for long, -1 for short, 0 for none)
        """
        close = row['close']
        high = row['high']
        low = row['low']
        atr = row.get('atr_14', row['close'] * 0.02)  # Default to 2% if ATR not available
        
        # Check exit conditions first
        if self.position != 0:
            should_exit, reason = self.check_exit_conditions(idx, high, low)
            
            if should_exit:
                # Use stop loss or take profit price as exit
                if reason == 'stop_loss':
                    exit_price = self.stop_loss
                else:  # take_profit
                    exit_price = self.take_profit
                
                self.exit_position(idx, exit_price, reason)
            
            # Check for opposite signal (exit on signal reversal)
            elif signal != 0 and signal != self.position:
                self.exit_position(idx, close, 'signal')
        
        # Check for new entry signal
        if self.position == 0 and signal != 0:
            self.enter_position(idx, close, atr, signal)
        
        # Record equity
        self.equity_curve.append(self.equity)


def backtest(
    df: pd.DataFrame,
    signal_col: str,
    config: Dict
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Run backtest on data with given signals.
    
    Args:
        df: DataFrame with OHLC data, indicators, and signals
        signal_col: Name of column containing signals
        config: Configuration dictionary with trading parameters
        
    Returns:
        Tuple of (df_with_results, trades, metrics)
    """
    logger.info(f"Running backtest with signal column: {signal_col}")
    
    # Validate inputs
    if signal_col not in df.columns:
        raise ValueError(f"Signal column '{signal_col}' not found in DataFrame")
    
    required_cols = ['close', 'high', 'low']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize engine
    engine = BacktestEngine(config)
    
    # Reset index to ensure integer indexing
    df = df.reset_index(drop=True)
    df_result = df.copy()
    
    # Process each bar
    for idx, row in df.iterrows():
        signal = row[signal_col]
        engine.process_bar(idx, row, signal)
    
    # Close any open position at the end
    if engine.position != 0:
        last_row = df.iloc[-1]
        engine.exit_position(len(df) - 1, last_row['close'], 'end_of_data')
    
    # Add equity curve to dataframe
    df_result['equity'] = engine.equity_curve
    df_result['returns'] = df_result['equity'].pct_change().fillna(0)
    df_result['cumulative_returns'] = (1 + df_result['returns']).cumprod() - 1
    
    # Compute metrics
    metrics = compute_metrics(df_result, engine.trades, engine.initial_capital)
    
    logger.info(f"Backtest complete: {len(engine.trades)} trades executed")
    
    return df_result, engine.trades, metrics
