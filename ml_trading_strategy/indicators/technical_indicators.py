"""Technical indicators calculation module."""

import logging
from typing import Optional

import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Compute comprehensive technical indicators for trading strategy.
    
    Calculates a wide range of technical indicators including:
    - Moving averages (EMA, SMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators
    - Trend indicators (ADX, Aroon)
    
    Args:
        df: DataFrame with OHLCV data
        validate: If True, validate data before computing indicators
        
    Returns:
        DataFrame with all technical indicators added
    """
    logger.info("Computing technical indicators")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    if validate:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Price data
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # === Moving Averages ===
    logger.debug("Computing moving averages")
    df['ema_9'] = ta.trend.ema_indicator(close, window=9)
    df['ema_21'] = ta.trend.ema_indicator(close, window=21)
    df['ema_50'] = ta.trend.ema_indicator(close, window=50)
    df['ema_100'] = ta.trend.ema_indicator(close, window=100)
    df['ema_200'] = ta.trend.ema_indicator(close, window=200)
    
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)
    
    # === RSI ===
    logger.debug("Computing RSI")
    df['rsi_14'] = ta.momentum.rsi(close, window=14)
    df['rsi_7'] = ta.momentum.rsi(close, window=7)
    df['rsi_21'] = ta.momentum.rsi(close, window=21)
    
    # === MACD ===
    logger.debug("Computing MACD")
    macd = ta.trend.MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # === Bollinger Bands ===
    logger.debug("Computing Bollinger Bands")
    bollinger = ta.volatility.BollingerBands(close)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (close - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # === ATR (Average True Range) ===
    logger.debug("Computing ATR")
    df['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['atr_7'] = ta.volatility.average_true_range(high, low, close, window=7)
    
    # === ADX (Average Directional Index) ===
    logger.debug("Computing ADX")
    df['adx_14'] = ta.trend.adx(high, low, close, window=14)
    df['adx_pos'] = ta.trend.adx_pos(high, low, close, window=14)
    df['adx_neg'] = ta.trend.adx_neg(high, low, close, window=14)
    
    # === Stochastic Oscillator ===
    logger.debug("Computing Stochastic")
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # === CCI (Commodity Channel Index) ===
    logger.debug("Computing CCI")
    df['cci_20'] = ta.trend.cci(high, low, close, window=20)
    
    # === Williams %R ===
    logger.debug("Computing Williams %R")
    df['williams_r'] = ta.momentum.williams_r(high, low, close)
    
    # === Volume Indicators ===
    logger.debug("Computing volume indicators")
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # On-Balance Volume
    df['obv'] = ta.volume.on_balance_volume(close, volume)
    
    # Volume Weighted Average Price (approximation)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # === Momentum ===
    logger.debug("Computing momentum indicators")
    df['momentum_10'] = close - close.shift(10)
    df['momentum_20'] = close - close.shift(20)
    df['rate_of_change'] = ta.momentum.roc(close, window=10)
    
    # === Aroon ===
    logger.debug("Computing Aroon")
    aroon = ta.trend.AroonIndicator(close)
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    
    # === Ichimoku ===
    logger.debug("Computing Ichimoku")
    ichimoku = ta.trend.IchimokuIndicator(high, low)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    
    # === Price-based features ===
    logger.debug("Computing price-based features")
    df['price_change'] = close.pct_change()
    df['price_range'] = (high - low) / close
    df['close_to_high'] = (high - close) / close
    df['close_to_low'] = (close - low) / close
    
    # Moving average convergence
    df['ema_9_21_distance'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
    df['ema_21_50_distance'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
    df['ema_50_200_distance'] = (df['ema_50'] - df['ema_200']) / df['ema_200']
    
    # Price position relative to EMAs
    df['price_to_ema_9'] = (close - df['ema_9']) / df['ema_9']
    df['price_to_ema_21'] = (close - df['ema_21']) / df['ema_21']
    df['price_to_ema_50'] = (close - df['ema_50']) / df['ema_50']
    
    # Volatility features
    df['volatility_20'] = close.pct_change().rolling(window=20).std()
    df['volatility_50'] = close.pct_change().rolling(window=50).std()
    
    # Drop rows with NaN values created by indicators
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with NaN values after indicator calculation")
    
    logger.info(f"Computed {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")
    
    return df
