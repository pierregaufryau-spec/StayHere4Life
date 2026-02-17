"""Signal generation module for trading strategy."""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def get_long_conditions(df: pd.DataFrame, params: Dict) -> pd.Series:
    """Generate long (buy) signal conditions.
    
    Long conditions based on multiple technical factors:
    - EMA alignment (9 > 21 > 50)
    - Minimum EMA distance
    - ADX above threshold (trending market)
    - Volume above average
    - RSI not overbought
    
    Args:
        df: DataFrame with technical indicators
        params: Dictionary with signal parameters:
            - min_ema_distance: Minimum distance between EMAs
            - adx_threshold: Minimum ADX for trend strength
            - volume_multiplier: Volume must be above SMA * multiplier
            
    Returns:
        Boolean Series indicating long signals
    """
    min_ema_distance = params.get('min_ema_distance', 0.05)
    adx_threshold = params.get('adx_threshold', 25)
    volume_multiplier = params.get('volume_multiplier', 1.1)
    
    # EMA alignment: 9 > 21 > 50
    ema_aligned = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
    
    # Minimum EMA distance (avoid choppy markets)
    ema_distance = (df['ema_9'] - df['ema_21']) / df['ema_21']
    sufficient_distance = ema_distance > min_ema_distance
    
    # Strong trend (ADX)
    strong_trend = df['adx_14'] > adx_threshold
    
    # Volume confirmation
    volume_confirm = df['volume'] > (df['volume_sma_20'] * volume_multiplier)
    
    # RSI not overbought
    rsi_ok = df['rsi_14'] < 70
    
    # MACD bullish
    macd_bullish = df['macd'] > df['macd_signal']
    
    # Price above EMA 50
    price_above_ema = df['close'] > df['ema_50']
    
    # Combine all conditions
    long_signal = (
        ema_aligned &
        sufficient_distance &
        strong_trend &
        volume_confirm &
        rsi_ok &
        macd_bullish &
        price_above_ema
    )
    
    return long_signal


def get_short_conditions(df: pd.DataFrame, params: Dict) -> pd.Series:
    """Generate short (sell) signal conditions.
    
    Short conditions based on multiple technical factors:
    - EMA alignment (9 < 21 < 50)
    - Minimum EMA distance
    - ADX above threshold (trending market)
    - Volume above average
    - RSI not oversold
    
    Args:
        df: DataFrame with technical indicators
        params: Dictionary with signal parameters:
            - min_ema_distance: Minimum distance between EMAs
            - adx_threshold: Minimum ADX for trend strength
            - volume_multiplier: Volume must be above SMA * multiplier
            
    Returns:
        Boolean Series indicating short signals
    """
    min_ema_distance = params.get('min_ema_distance', 0.05)
    adx_threshold = params.get('adx_threshold', 25)
    volume_multiplier = params.get('volume_multiplier', 1.1)
    
    # EMA alignment: 9 < 21 < 50
    ema_aligned = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
    
    # Minimum EMA distance (avoid choppy markets)
    ema_distance = (df['ema_21'] - df['ema_9']) / df['ema_21']
    sufficient_distance = ema_distance > min_ema_distance
    
    # Strong trend (ADX)
    strong_trend = df['adx_14'] > adx_threshold
    
    # Volume confirmation
    volume_confirm = df['volume'] > (df['volume_sma_20'] * volume_multiplier)
    
    # RSI not oversold
    rsi_ok = df['rsi_14'] > 30
    
    # MACD bearish
    macd_bearish = df['macd'] < df['macd_signal']
    
    # Price below EMA 50
    price_below_ema = df['close'] < df['ema_50']
    
    # Combine all conditions
    short_signal = (
        ema_aligned &
        sufficient_distance &
        strong_trend &
        volume_confirm &
        rsi_ok &
        macd_bearish &
        price_below_ema
    )
    
    return short_signal


def generate_raw_signals(
    df: pd.DataFrame,
    min_ema_distance: float = 0.05,
    adx_threshold: float = 25,
    volume_multiplier: float = 1.1
) -> pd.DataFrame:
    """Generate raw trading signals based on technical indicators.
    
    Creates both long and short signals based on technical analysis.
    The signals represent trading opportunities before ML filtering.
    
    Args:
        df: DataFrame with technical indicators
        min_ema_distance: Minimum distance between EMAs for valid signal
        adx_threshold: Minimum ADX value for trend strength
        volume_multiplier: Volume must be above average * multiplier
        
    Returns:
        DataFrame with added signal columns:
            - raw_signal: 1 for long, -1 for short, 0 for no signal
            - long_signal: Boolean for long conditions
            - short_signal: Boolean for short conditions
    """
    logger.info("Generating raw trading signals")
    
    df = df.copy()
    
    params = {
        'min_ema_distance': min_ema_distance,
        'adx_threshold': adx_threshold,
        'volume_multiplier': volume_multiplier
    }
    
    # Generate long and short signals
    df['long_signal'] = get_long_conditions(df, params)
    df['short_signal'] = get_short_conditions(df, params)
    
    # Create combined signal column
    df['raw_signal'] = 0
    df.loc[df['long_signal'], 'raw_signal'] = 1
    df.loc[df['short_signal'], 'raw_signal'] = -1
    
    # Statistics
    n_long = df['long_signal'].sum()
    n_short = df['short_signal'].sum()
    n_total = len(df)
    
    logger.info(f"Generated signals: {n_long} long ({n_long/n_total*100:.2f}%), "
                f"{n_short} short ({n_short/n_total*100:.2f}%)")
    
    # Check for conflicting signals (should not happen with current logic)
    conflicts = (df['long_signal'] & df['short_signal']).sum()
    if conflicts > 0:
        logger.warning(f"Found {conflicts} conflicting signals (both long and short)")
    
    return df
