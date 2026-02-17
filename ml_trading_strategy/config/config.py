"""Configuration management for ML trading strategy."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'data': {
        'csv_path': 'btcusd_4h.csv',
        'generate_sample': True,
        'sample_size': 3000
    },
    'trading': {
        'initial_capital': 100000,
        'risk_per_trade': 0.01,
        'atr_sl_multiplier': 1.5,
        'atr_tp_multiplier': 3.0,
        'commission': 0.001
    },
    'signals': {
        'min_ema_distance': 0.05,
        'adx_threshold': 25,
        'volume_multiplier': 1.1
    },
    'ml': {
        'forward_bars': 5,
        'label_threshold': 0.003,
        'n_features': 20,
        'feature_selection_method': 'importance',
        'optimize': True,
        'n_trials': 30,
        'ml_threshold': 0.35,
        'test_size': 0.3,
        'random_seed': 42
    },
    'paths': {
        'models_dir': 'saved_models',
        'results_dir': 'results',
        'plots_dir': 'plots'
    }
}


class Config:
    """Configuration manager for ML trading strategy.
    
    Handles loading configuration from YAML files, validation,
    and provides easy access to configuration parameters.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary. If None, uses DEFAULT_CONFIG.
        """
        if config_dict is None:
            self.config = DEFAULT_CONFIG.copy()
        else:
            self.config = self._merge_configs(DEFAULT_CONFIG, config_dict)
        
        self._validate_config()
        self._create_directories()
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override config into default config.
        
        Args:
            default: Default configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['data', 'trading', 'signals', 'ml', 'paths']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")
        
        # Validate data config
        if self.config['data']['sample_size'] <= 0:
            raise ValueError("sample_size must be positive")
        
        # Validate trading config
        if self.config['trading']['initial_capital'] <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.config['trading']['risk_per_trade'] <= 1:
            raise ValueError("risk_per_trade must be between 0 and 1")
        
        # Validate ML config
        if self.config['ml']['forward_bars'] <= 0:
            raise ValueError("forward_bars must be positive")
        if not 0 < self.config['ml']['test_size'] < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        logger.info("Configuration validated successfully")
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_key in ['models_dir', 'results_dir', 'plots_dir']:
            dir_path = Path(self.config['paths'][dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Path to configuration key (e.g., 'data.csv_path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('data.csv_path')
            'btcusd_4h.csv'
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key_path: Path to configuration key (e.g., 'data.csv_path')
            value: Value to set
            
        Example:
            >>> config.set('data.csv_path', 'new_data.csv')
        """
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
        logger.debug(f"Set config: {key_path} = {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
    """
    if config_path is None:
        logger.info("Using default configuration")
        return Config()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return Config(config_dict)
