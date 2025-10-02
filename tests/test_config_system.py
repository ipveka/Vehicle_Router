"""Unit tests for Configuration System"""

import unittest
import sys
import os
from unittest.mock import patch

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import (
    OPTIMIZATION_METHOD, UI_CONFIG, OPTIMIZATION_DEFAULTS,
    METHOD_PARAMS, DISTANCE_CONFIG, DEPOT_CONFIG,
    validate_config, get_config_summary, get_method_params, get_method_display_name
)


class TestConfigurationSystem(unittest.TestCase):
    """Test cases for configuration system"""
    
    def test_configuration_constants(self):
        """Test configuration constants"""
        # Test optimization method
        self.assertIsInstance(OPTIMIZATION_METHOD, str)
        self.assertIn(OPTIMIZATION_METHOD, ['standard', 'genetic', 'enhanced'])
        
        # Test UI config
        self.assertIsInstance(UI_CONFIG, dict)
        self.assertIn('page_title', UI_CONFIG)
        
        # Test optimization defaults
        self.assertIsInstance(OPTIMIZATION_DEFAULTS, dict)
        self.assertIn('max_orders_per_truck', OPTIMIZATION_DEFAULTS)
        
        # Test method params
        self.assertIsInstance(METHOD_PARAMS, dict)
        for method in ['standard', 'enhanced', 'genetic']:
            self.assertIn(method, METHOD_PARAMS)
        
        # Test distance config
        self.assertIsInstance(DISTANCE_CONFIG, dict)
        self.assertIn('default_method', DISTANCE_CONFIG)
        
        # Test depot config
        self.assertIsInstance(DEPOT_CONFIG, dict)
        self.assertIn('default_location', DEPOT_CONFIG)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        issues = validate_config()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        self.assertIn('info', issues)
        
        # Test invalid method
        with patch('app.config.OPTIMIZATION_METHOD', 'invalid_method'):
            issues = validate_config()
            self.assertGreater(len(issues['errors']), 0)
    
    def test_configuration_helpers(self):
        """Test configuration helper functions"""
        # Test get_config_summary
        summary = get_config_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn('Default Algorithm', summary)
        
        # Test get_method_display_name
        display_name = get_method_display_name()
        self.assertIsInstance(display_name, str)
        self.assertGreater(len(display_name), 0)
        
        # Test get_method_params
        params = get_method_params()
        self.assertIsInstance(params, dict)
        self.assertIn('solver_timeout', params)


if __name__ == '__main__':
    unittest.main()