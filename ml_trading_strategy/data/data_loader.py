"""Data loading and preparation module."""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_sample_data(n: int = 3000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing.
    
    Creates synthetic price data with realistic patterns including
    trends, volatility, and volume variations.
    
    Args:
        n: Number of bars to generate
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    logger.info(f"Generating {n} bars of sample data")
    
    np.random.seed(42)
    
    # Generate base price with trend and random walk
    base_price = 50000
    trend = np.linspace(0, 5000, n)
    random_walk = np.cumsum(np.random.randn(n) * 500)
    close_prices = base_price + trend + random_walk
    
    # Ensure prices stay positive
    close_prices = np.maximum(close_prices, 1000)
    
    # Generate OHLC with realistic spreads
    open_prices = close_prices + np.random.randn(n) * 50
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 100)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 100)
    
    # Generate volume with trend and randomness
    base_volume = 1000000
    volume = base_volume + np.abs(np.random.randn(n) * 500000)
    
    # Create timestamps (4-hour intervals)
    start_time = pd.Timestamp('2020-01-01')
    timestamps = [start_time + pd.Timedelta(hours=4*i) for i in range(n)]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    logger.info(f"Generated sample data: {len(df)} rows, "
                f"price range [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate OHLCV data format and quality.
    
    Checks for required columns, data types, missing values,
    and basic data quality issues.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
        
    Raises:
        ValueError: If data validation fails with specific error message
    """
    # Check required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Check for missing values
    if df[required_columns].isnull().any().any():
        null_counts = df[required_columns].isnull().sum()
        null_cols = null_counts[null_counts > 0]
        raise ValueError(f"Missing values found in columns: {null_cols.to_dict()}")
    
    # Check price columns are numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")
    
    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values")
    
    # Check OHLC consistency
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    if invalid_ohlc.any():
        raise ValueError(f"OHLC consistency violated in {invalid_ohlc.sum()} rows")
    
    # Check for negative volume
    if (df['volume'] < 0).any():
        raise ValueError("Volume contains negative values")
    
    logger.info(f"Data validation passed: {len(df)} rows")
    return True


def load_data(csv_path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If data validation fails
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    logger.info(f"Loading data from {csv_path}")
    
    # Try to load with common CSV formats
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Convert timestamp column if present
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logger.warning(f"Failed to parse timestamp column: {e}")
    elif 'date' in df.columns:
        # Handle alternative column name
        df['timestamp'] = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    
    # Validate the loaded data
    validate_data(df)
    
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def prepare_data(csv_path: str, generate_if_missing: bool = True) -> pd.DataFrame:
    """Prepare data for analysis, loading from file or generating sample data.
    
    This is the main entry point for data preparation. It will either load
    data from a CSV file or generate sample data if the file doesn't exist.
    
    Args:
        csv_path: Path to CSV file
        generate_if_missing: If True, generate sample data when file is missing.
                           If False, raise error when file is missing.
        
    Returns:
        DataFrame with prepared OHLCV data
        
    Raises:
        FileNotFoundError: If file doesn't exist and generate_if_missing is False
    """
    if os.path.exists(csv_path):
        logger.info(f"Loading data from existing file: {csv_path}")
        return load_data(csv_path)
    
    if generate_if_missing:
        logger.warning(f"Data file not found: {csv_path}")
        logger.info("Generating sample data instead")
        return generate_sample_data()
    
    raise FileNotFoundError(f"Data file not found: {csv_path}")
