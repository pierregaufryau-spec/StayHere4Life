"""Feature engineering module for ML model."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Get list of feature columns from DataFrame.
    
    Identifies all columns that can be used as features for ML,
    excluding timestamp, OHLCV, and label columns.
    
    Args:
        df: DataFrame with all columns
        exclude_cols: Additional columns to exclude
        
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Columns to exclude by default
    default_exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'label', 'forward_return', 'forward_high', 'forward_low',
        'raw_signal', 'long_signal', 'short_signal', 'ml_signal',
        'position', 'equity', 'returns', 'cumulative_returns',
        'drawdown', 'entry_price', 'stop_loss', 'take_profit'
    ]
    
    exclude_set = set(default_exclude + exclude_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_set]
    
    logger.debug(f"Identified {len(feature_cols)} feature columns")
    
    return feature_cols


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for ML model.
    
    Generates derived features from existing technical indicators:
    - Lag features (previous values)
    - Rolling statistics (mean, std)
    - Ratios and interactions
    - Time-based features
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with additional features
    """
    logger.info("Creating ML features")
    
    df = df.copy()
    
    # === Lag Features ===
    logger.debug("Creating lag features")
    lag_features = ['rsi_14', 'macd', 'adx_14', 'cci_20', 'stoch_k']
    for feature in lag_features:
        if feature in df.columns:
            df[f'{feature}_lag1'] = df[feature].shift(1)
            df[f'{feature}_lag2'] = df[feature].shift(2)
    
    # === Rolling Statistics ===
    logger.debug("Creating rolling statistics")
    rolling_features = ['rsi_14', 'adx_14', 'volume_ratio']
    for feature in rolling_features:
        if feature in df.columns:
            df[f'{feature}_ma5'] = df[feature].rolling(window=5).mean()
            df[f'{feature}_std5'] = df[feature].rolling(window=5).std()
    
    # === Momentum Changes ===
    logger.debug("Creating momentum change features")
    if 'rsi_14' in df.columns:
        df['rsi_change'] = df['rsi_14'].diff()
        df['rsi_acceleration'] = df['rsi_change'].diff()
    
    if 'macd_diff' in df.columns:
        df['macd_diff_change'] = df['macd_diff'].diff()
    
    # === Volatility Ratios ===
    logger.debug("Creating volatility features")
    if 'atr_14' in df.columns and 'close' in df.columns:
        df['atr_ratio'] = df['atr_14'] / df['close']
    
    if 'volatility_20' in df.columns and 'volatility_50' in df.columns:
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
    
    # === Trend Strength ===
    logger.debug("Creating trend strength features")
    if all(col in df.columns for col in ['adx_pos', 'adx_neg']):
        df['directional_strength'] = df['adx_pos'] - df['adx_neg']
    
    if all(col in df.columns for col in ['aroon_up', 'aroon_down']):
        df['aroon_difference'] = df['aroon_up'] - df['aroon_down']
    
    # === Volume Features ===
    logger.debug("Creating volume features")
    if 'volume_ratio' in df.columns:
        df['volume_ratio_ma10'] = df['volume_ratio'].rolling(window=10).mean()
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    
    # === Price Momentum ===
    logger.debug("Creating price momentum features")
    if 'close' in df.columns:
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)
    
    # === Cross Features (Interactions) ===
    logger.debug("Creating interaction features")
    if 'rsi_14' in df.columns and 'adx_14' in df.columns:
        df['rsi_adx_product'] = df['rsi_14'] * df['adx_14']
    
    if 'bb_position' in df.columns and 'rsi_14' in df.columns:
        df['bb_rsi_product'] = df['bb_position'] * df['rsi_14']
    
    # === EMA Crosses ===
    logger.debug("Creating EMA cross features")
    if all(col in df.columns for col in ['ema_9', 'ema_21']):
        df['ema_9_above_21'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['ema_9_21_cross_up'] = ((df['ema_9'] > df['ema_21']) & 
                                    (df['ema_9'].shift(1) <= df['ema_21'].shift(1))).astype(int)
        df['ema_9_21_cross_down'] = ((df['ema_9'] < df['ema_21']) & 
                                      (df['ema_9'].shift(1) >= df['ema_21'].shift(1))).astype(int)
    
    # === Support/Resistance Proximity ===
    logger.debug("Creating support/resistance features")
    if 'close' in df.columns:
        # Distance from recent high/low
        df['distance_from_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
        df['distance_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
    
    # Drop rows with NaN from new features
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with NaN values after feature creation")
    
    # Count total features
    feature_cols = get_feature_columns(df)
    logger.info(f"Created total of {len(feature_cols)} features for ML")
    
    return df


def create_labels(df: pd.DataFrame, forward_bars: int = 5, threshold: float = 0.003) -> pd.DataFrame:
    """Create labels for supervised learning.
    
    Labels are created based on forward price movement:
    - Label 1: Price increases by threshold% within forward_bars
    - Label -1: Price decreases by threshold% within forward_bars
    - Label 0: Price stays within threshold range
    
    Args:
        df: DataFrame with price data
        forward_bars: Number of bars to look ahead
        threshold: Minimum price change threshold for label
        
    Returns:
        DataFrame with label column added
    """
    logger.info(f"Creating labels: forward_bars={forward_bars}, threshold={threshold}")
    
    df = df.copy()
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    # Calculate forward returns
    df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
    
    # Calculate forward high and low for more accurate labeling
    df['forward_high'] = df['high'].rolling(window=forward_bars).max().shift(-forward_bars)
    df['forward_low'] = df['low'].rolling(window=forward_bars).min().shift(-forward_bars)
    
    # Calculate max potential gain and loss
    df['potential_gain'] = df['forward_high'] / df['close'] - 1
    df['potential_loss'] = df['forward_low'] / df['close'] - 1
    
    # Create labels based on potential moves
    df['label'] = 0
    
    # Bullish: potential gain exceeds threshold
    df.loc[df['potential_gain'] > threshold, 'label'] = 1
    
    # Bearish: potential loss exceeds threshold (negative)
    df.loc[df['potential_loss'] < -threshold, 'label'] = -1
    
    # Drop temporary columns
    df = df.drop(columns=['potential_gain', 'potential_loss'])
    
    # Drop rows where we can't calculate forward return (end of dataset)
    df = df.dropna(subset=['forward_return', 'forward_high', 'forward_low'])
    
    # Statistics
    label_counts = df['label'].value_counts().sort_index()
    logger.info(f"Label distribution:")
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        label_name = {-1: 'Bearish', 0: 'Neutral', 1: 'Bullish'}.get(label, 'Unknown')
        logger.info(f"  {label_name} ({label}): {count} ({pct:.2f}%)")
    
    return df
