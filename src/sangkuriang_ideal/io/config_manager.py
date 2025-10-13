# io/config_manager.py
"""Configuration file parser for KdV simulations."""

from pathlib import Path


class ConfigManager:
    """Parse configuration files for KdV simulations."""
    
    @staticmethod
    def load(config_path: str) -> dict:
        """
        Load configuration from file.
        
        File format:
            # Comments
            key = value
        
        Supported types: bool, int, float, str
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = {}
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty/comment lines
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                # Parse key = value
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove inline comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Parse type
                config[key] = ConfigManager._parse_value(value)
        
        return config
    
    @staticmethod
    def _parse_value(value: str):
        """Parse string to appropriate Python type."""
        # Boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Numeric
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
