"""Unit tests for Streamlit App Components"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock streamlit before importing app modules
sys.modules['streamlit'] = MagicMock()

from app.config import (
    DEFAULT_ALGORITHM, AVAILABLE_MODELS, OPTIMIZATION_DEFAULTS,
    METHOD_DEFAULTS, validate_config, get_config_summary,
    get_enabled_models, get_model_display_name, is_model_enabled,
    get_method_defaults
)
from app.streamlit_app import VehicleRouterApp
from vehicle_router.data_generator import DataGenerator


class TestStreamlitApp(unittest.TestCase):
    """Test cases for Streamlit app components"""
    
    def test_configuration_system(self):
        """Test configuration system"""
        # Test default algorithm
        self.assertIn(DEFAULT_ALGORITHM, ['standard', 'genetic', 'enhanced'])
        self.assertTrue(is_model_enabled(DEFAULT_ALGORITHM))
        
        # Test available models
        for key in ['genetic', 'standard', 'enhanced']:
            self.assertIn(key, AVAILABLE_MODELS)
            self.assertIn('name', AVAILABLE_MODELS[key])
            self.assertIn('enabled', AVAILABLE_MODELS[key])
        
        # Test optimization defaults
        required_keys = [
            'max_orders_per_truck', 'depot_return', 'enable_greedy_routes',
            'use_real_distances', 'validation_enabled', 'solver_timeout'
        ]
        for key in required_keys:
            self.assertIn(key, OPTIMIZATION_DEFAULTS)
        
        # Test method defaults
        for method in ['standard', 'enhanced', 'genetic']:
            self.assertIn(method, METHOD_DEFAULTS)
            method_config = METHOD_DEFAULTS[method]
            self.assertIn('solver_timeout', method_config)
        
        # Test configuration validation
        issues = validate_config()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        
        # Test configuration summary
        summary = get_config_summary()
        self.assertIsInstance(summary, str)
        self.assertIn('Default Algorithm', summary)
        
        # Test helper functions
        enabled_models = get_enabled_models()
        self.assertIsInstance(enabled_models, list)
        
        display_name = get_model_display_name('standard')
        self.assertIsInstance(display_name, str)
        
        defaults = get_method_defaults('standard')
        self.assertIsInstance(defaults, dict)
        self.assertIn('solver_timeout', defaults)
    
    def test_vehicle_router_app(self):
        """Test VehicleRouterApp class"""
        # Test app initialization
        app = VehicleRouterApp()
        self.assertIsNotNone(app)
        
        # Test session state initialization
        app.initialize_session_state()
        # Just test that the method runs without error
        self.assertIsNotNone(app)
        
        # Test that app has expected attributes
        self.assertTrue(hasattr(app, 'logger'))
        self.assertTrue(hasattr(app, 'initialize_session_state'))
    
    def test_app_integration(self):
        """Test app integration workflows"""
        # Test data loading workflow
        app = VehicleRouterApp()
        app.initialize_session_state()
        
        # Test that app methods exist
        self.assertTrue(hasattr(app, 'initialize_session_state'))
        self.assertTrue(hasattr(app, 'logger'))


if __name__ == '__main__':
    unittest.main()