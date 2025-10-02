"""
Unit tests for Configuration System

This module contains comprehensive unit tests for the Vehicle Router configuration system,
testing all configuration modules, validation functions, and helper utilities.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import (
    DEFAULT_ALGORITHM, AVAILABLE_MODELS, UI_CONFIG, OPTIMIZATION_DEFAULTS,
    METHOD_DEFAULTS, DISTANCE_CONFIG, DEPOT_CONFIG, EXPORT_CONFIG,
    LOGGING_CONFIG, VALIDATION_CONFIG, ADVANCED_CONFIG,
    validate_config, get_config_summary, get_enabled_models,
    get_model_display_name, is_model_enabled, get_method_defaults
)


class TestConfigurationConstants(unittest.TestCase):
    """Test cases for configuration constants"""
    
    def test_default_algorithm(self):
        """Test DEFAULT_ALGORITHM constant"""
        self.assertIsInstance(DEFAULT_ALGORITHM, str)
        self.assertIn(DEFAULT_ALGORITHM, ['standard', 'genetic', 'enhanced'])
    
    def test_available_models_structure(self):
        """Test AVAILABLE_MODELS structure and content"""
        self.assertIsInstance(AVAILABLE_MODELS, dict)
        
        # Check that all required models are present
        required_models = ['genetic', 'standard', 'enhanced']
        for model in required_models:
            self.assertIn(model, AVAILABLE_MODELS)
            
            model_config = AVAILABLE_MODELS[model]
            self.assertIsInstance(model_config, dict)
            
            # Check required keys
            required_keys = ['name', 'help', 'enabled', 'description']
            for key in required_keys:
                self.assertIn(key, model_config)
                self.assertIsInstance(model_config[key], (str, bool))
    
    def test_ui_config_structure(self):
        """Test UI_CONFIG structure"""
        self.assertIsInstance(UI_CONFIG, dict)
        
        required_keys = [
            'page_title', 'page_icon', 'layout', 'initial_sidebar_state',
            'show_logs_by_default', 'enable_progress_bars', 'enable_animations'
        ]
        
        for key in required_keys:
            self.assertIn(key, UI_CONFIG)
            self.assertIsInstance(UI_CONFIG[key], (str, bool))
    
    def test_optimization_defaults_structure(self):
        """Test OPTIMIZATION_DEFAULTS structure"""
        self.assertIsInstance(OPTIMIZATION_DEFAULTS, dict)
        
        required_keys = [
            'max_orders_per_truck', 'depot_return', 'enable_greedy_routes',
            'use_real_distances', 'validation_enabled', 'solver_timeout'
        ]
        
        for key in required_keys:
            self.assertIn(key, OPTIMIZATION_DEFAULTS)
            self.assertIsInstance(OPTIMIZATION_DEFAULTS[key], (int, bool))
    
    def test_method_defaults_structure(self):
        """Test METHOD_DEFAULTS structure"""
        self.assertIsInstance(METHOD_DEFAULTS, dict)
        
        for method in ['standard', 'enhanced', 'genetic']:
            self.assertIn(method, METHOD_DEFAULTS)
            method_config = METHOD_DEFAULTS[method]
            self.assertIsInstance(method_config, dict)
            
            # Check required keys for each method
            self.assertIn('solver_timeout', method_config)
            self.assertIn('cost_weight', method_config)
            self.assertIn('distance_weight', method_config)
            
            # Check genetic algorithm specific keys
            if method == 'genetic':
                genetic_keys = ['population_size', 'max_generations', 'mutation_rate']
                for key in genetic_keys:
                    self.assertIn(key, method_config)
                    self.assertIsInstance(method_config[key], (int, float))
    
    def test_distance_config_structure(self):
        """Test DISTANCE_CONFIG structure"""
        self.assertIsInstance(DISTANCE_CONFIG, dict)
        
        required_keys = [
            'default_method', 'country_code', 'rate_limit_delay',
            'geocoding_timeout', 'enable_caching', 'cache_duration'
        ]
        
        for key in required_keys:
            self.assertIn(key, DISTANCE_CONFIG)
            self.assertIsInstance(DISTANCE_CONFIG[key], (str, int, bool, float))
    
    def test_depot_config_structure(self):
        """Test DEPOT_CONFIG structure"""
        self.assertIsInstance(DEPOT_CONFIG, dict)
        
        required_keys = ['default_location', 'allow_custom_depot', 'show_depot_selector']
        
        for key in required_keys:
            self.assertIn(key, DEPOT_CONFIG)
            self.assertIsInstance(DEPOT_CONFIG[key], (str, bool))
    
    def test_export_config_structure(self):
        """Test EXPORT_CONFIG structure"""
        self.assertIsInstance(EXPORT_CONFIG, dict)
        
        required_keys = [
            'enable_excel_export', 'enable_csv_export', 'enable_detailed_reports',
            'default_export_format', 'include_visualizations', 'save_plots_to_disk'
        ]
        
        for key in required_keys:
            self.assertIn(key, EXPORT_CONFIG)
            self.assertIsInstance(EXPORT_CONFIG[key], (str, bool))
    
    def test_logging_config_structure(self):
        """Test LOGGING_CONFIG structure"""
        self.assertIsInstance(LOGGING_CONFIG, dict)
        
        required_keys = [
            'log_level', 'enable_file_logging', 'enable_console_logging',
            'enable_performance_tracking', 'log_rotation', 'max_log_files',
            'log_directory'
        ]
        
        for key in required_keys:
            self.assertIn(key, LOGGING_CONFIG)
            self.assertIsInstance(LOGGING_CONFIG[key], (str, int, bool))
    
    def test_validation_config_structure(self):
        """Test VALIDATION_CONFIG structure"""
        self.assertIsInstance(VALIDATION_CONFIG, dict)
        
        required_keys = [
            'enable_solution_validation', 'validate_capacity_constraints',
            'validate_order_assignment', 'validate_route_feasibility',
            'strict_validation'
        ]
        
        for key in required_keys:
            self.assertIn(key, VALIDATION_CONFIG)
            self.assertIsInstance(VALIDATION_CONFIG[key], bool)
    
    def test_advanced_config_structure(self):
        """Test ADVANCED_CONFIG structure"""
        self.assertIsInstance(ADVANCED_CONFIG, dict)
        
        required_keys = [
            'enable_debug_mode', 'show_technical_details',
            'enable_experimental_features', 'memory_usage_warnings',
            'performance_monitoring'
        ]
        
        for key in required_keys:
            self.assertIn(key, ADVANCED_CONFIG)
            self.assertIsInstance(ADVANCED_CONFIG[key], bool)


class TestConfigurationValidation(unittest.TestCase):
    """Test cases for configuration validation functions"""
    
    def test_validate_config_valid_configuration(self):
        """Test validate_config with valid configuration"""
        issues = validate_config()
        
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        self.assertIn('info', issues)
        
        # With default configuration, there should be no errors
        self.assertEqual(len(issues['errors']), 0)
    
    def test_validate_config_invalid_algorithm(self):
        """Test validate_config with invalid default algorithm"""
        # Temporarily modify DEFAULT_ALGORITHM
        original_algorithm = DEFAULT_ALGORITHM
        
        # Mock the DEFAULT_ALGORITHM to be invalid
        with patch('app.config.DEFAULT_ALGORITHM', 'invalid_algorithm'):
            issues = validate_config()
            self.assertGreater(len(issues['errors']), 0)
            self.assertIn('invalid_algorithm', str(issues['errors']))
    
    def test_validate_config_disabled_algorithm(self):
        """Test validate_config with disabled default algorithm"""
        # Mock AVAILABLE_MODELS to have disabled default algorithm
        with patch('app.config.AVAILABLE_MODELS', {
            'genetic': {'enabled': False},
            'standard': {'enabled': True},
            'enhanced': {'enabled': True}
        }):
            with patch('app.config.DEFAULT_ALGORITHM', 'genetic'):
                issues = validate_config()
                self.assertGreater(len(issues['warnings']), 0)
                self.assertIn('disabled', str(issues['warnings']))
    
    def test_validate_config_no_enabled_models(self):
        """Test validate_config with no enabled models"""
        # Mock AVAILABLE_MODELS to have all models disabled
        with patch('app.config.AVAILABLE_MODELS', {
            'genetic': {'enabled': False},
            'standard': {'enabled': False},
            'enhanced': {'enabled': False}
        }):
            issues = validate_config()
            self.assertGreater(len(issues['errors']), 0)
            self.assertIn('No optimization models are enabled', str(issues['errors']))
    
    def test_validate_config_invalid_distance_method(self):
        """Test validate_config with invalid distance method"""
        # Mock DISTANCE_CONFIG to have invalid method
        with patch('app.config.DISTANCE_CONFIG', {
            'default_method': 'invalid_method',
            'country_code': 'ES',
            'rate_limit_delay': 0.5,
            'geocoding_timeout': 10,
            'enable_caching': True,
            'cache_duration': 3600
        }):
            issues = validate_config()
            self.assertGreater(len(issues['errors']), 0)
            self.assertIn('Distance method must be', str(issues['errors']))


class TestConfigurationHelpers(unittest.TestCase):
    """Test cases for configuration helper functions"""
    
    def test_get_enabled_models(self):
        """Test get_enabled_models function"""
        enabled_models = get_enabled_models()
        
        self.assertIsInstance(enabled_models, list)
        self.assertGreater(len(enabled_models), 0)
        
        # Check that all returned models are actually enabled
        for model in enabled_models:
            self.assertTrue(is_model_enabled(model))
    
    def test_get_model_display_name(self):
        """Test get_model_display_name function"""
        for model_key in AVAILABLE_MODELS.keys():
            display_name = get_model_display_name(model_key)
            
            self.assertIsInstance(display_name, str)
            self.assertGreater(len(display_name), 0)
            self.assertEqual(display_name, AVAILABLE_MODELS[model_key]['name'])
        
        # Test with invalid model key
        invalid_name = get_model_display_name('invalid_model')
        self.assertEqual(invalid_name, 'invalid_model')
    
    def test_is_model_enabled(self):
        """Test is_model_enabled function"""
        for model_key, model_config in AVAILABLE_MODELS.items():
            expected_enabled = model_config['enabled']
            actual_enabled = is_model_enabled(model_key)
            self.assertEqual(actual_enabled, expected_enabled)
        
        # Test with invalid model key
        self.assertFalse(is_model_enabled('invalid_model'))
    
    def test_get_method_defaults(self):
        """Test get_method_defaults function"""
        for method in ['standard', 'enhanced', 'genetic']:
            defaults = get_method_defaults(method)
            
            self.assertIsInstance(defaults, dict)
            self.assertIn('solver_timeout', defaults)
            self.assertIn('cost_weight', defaults)
            self.assertIn('distance_weight', defaults)
        
        # Test with invalid method
        invalid_defaults = get_method_defaults('invalid_method')
        self.assertEqual(invalid_defaults, {})
    
    def test_get_config_summary(self):
        """Test get_config_summary function"""
        summary = get_config_summary()
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        
        # Check that summary contains expected information
        expected_sections = [
            'Default Algorithm',
            'Enabled Models',
            'Distance Method',
            'Max Orders per Truck',
            'Real Distances',
            'Validation',
            'Logging Level'
        ]
        
        for section in expected_sections:
            self.assertIn(section, summary)
        
        # Check that actual values are included
        self.assertIn(DEFAULT_ALGORITHM, summary)
        self.assertIn(str(OPTIMIZATION_DEFAULTS['max_orders_per_truck']), summary)
        self.assertIn(str(OPTIMIZATION_DEFAULTS['use_real_distances']), summary)


class TestConfigurationEdgeCases(unittest.TestCase):
    """Test cases for configuration edge cases and error handling"""
    
    def test_configuration_consistency(self):
        """Test that configuration values are consistent"""
        # Check that default algorithm is enabled
        self.assertTrue(is_model_enabled(DEFAULT_ALGORITHM))
        
        # Check that at least one model is enabled
        enabled_models = get_enabled_models()
        self.assertGreater(len(enabled_models), 0)
        
        # Check that method defaults exist for all enabled models
        for model in enabled_models:
            if model in ['standard', 'enhanced', 'genetic']:
                defaults = get_method_defaults(model)
                self.assertIsInstance(defaults, dict)
                self.assertGreater(len(defaults), 0)
    
    def test_configuration_value_ranges(self):
        """Test that configuration values are within valid ranges"""
        # Check solver timeouts are positive
        for method, defaults in METHOD_DEFAULTS.items():
            self.assertGreater(defaults['solver_timeout'], 0)
        
        # Check weights are between 0 and 1
        for method, defaults in METHOD_DEFAULTS.items():
            self.assertGreaterEqual(defaults['cost_weight'], 0.0)
            self.assertLessEqual(defaults['cost_weight'], 1.0)
            self.assertGreaterEqual(defaults['distance_weight'], 0.0)
            self.assertLessEqual(defaults['distance_weight'], 1.0)
        
        # Check genetic algorithm parameters
        genetic_defaults = METHOD_DEFAULTS['genetic']
        self.assertGreater(genetic_defaults['population_size'], 0)
        self.assertGreater(genetic_defaults['max_generations'], 0)
        self.assertGreaterEqual(genetic_defaults['mutation_rate'], 0.0)
        self.assertLessEqual(genetic_defaults['mutation_rate'], 1.0)
        
        # Check optimization defaults
        self.assertGreater(OPTIMIZATION_DEFAULTS['max_orders_per_truck'], 0)
        self.assertGreater(OPTIMIZATION_DEFAULTS['solver_timeout'], 0)
    
    def test_configuration_immutability(self):
        """Test that configuration constants are not accidentally modified"""
        # Test that we can't modify the configuration through helper functions
        original_enabled = get_enabled_models()
        
        # These should not modify the original configuration
        get_model_display_name('standard')
        is_model_enabled('genetic')
        get_method_defaults('enhanced')
        
        # Configuration should remain unchanged
        current_enabled = get_enabled_models()
        self.assertEqual(original_enabled, current_enabled)


if __name__ == '__main__':
    unittest.main()
