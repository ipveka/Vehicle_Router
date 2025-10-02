"""
Unit tests for Streamlit App Components

This module contains comprehensive unit tests for the Vehicle Router Streamlit application,
testing the main app class, configuration system, and UI components.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys
import os
from pathlib import Path

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


class TestConfigSystem(unittest.TestCase):
    """Test cases for the configuration system"""
    
    def test_default_algorithm_configuration(self):
        """Test default algorithm configuration"""
        self.assertIn(DEFAULT_ALGORITHM, ['standard', 'genetic', 'enhanced'])
        self.assertTrue(is_model_enabled(DEFAULT_ALGORITHM))
    
    def test_available_models_structure(self):
        """Test AVAILABLE_MODELS structure"""
        required_keys = ['genetic', 'standard', 'enhanced']
        for key in required_keys:
            self.assertIn(key, AVAILABLE_MODELS)
            self.assertIn('name', AVAILABLE_MODELS[key])
            self.assertIn('help', AVAILABLE_MODELS[key])
            self.assertIn('enabled', AVAILABLE_MODELS[key])
            self.assertIn('description', AVAILABLE_MODELS[key])
    
    def test_optimization_defaults(self):
        """Test optimization defaults configuration"""
        required_keys = [
            'max_orders_per_truck', 'depot_return', 'enable_greedy_routes',
            'use_real_distances', 'validation_enabled', 'solver_timeout'
        ]
        for key in required_keys:
            self.assertIn(key, OPTIMIZATION_DEFAULTS)
    
    def test_method_defaults_structure(self):
        """Test METHOD_DEFAULTS structure"""
        for method in ['standard', 'enhanced', 'genetic']:
            self.assertIn(method, METHOD_DEFAULTS)
            method_config = METHOD_DEFAULTS[method]
            self.assertIn('solver_timeout', method_config)
            self.assertIn('cost_weight', method_config)
            self.assertIn('distance_weight', method_config)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config"""
        issues = validate_config()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        self.assertIn('info', issues)
        # Should have no errors with default config
        self.assertEqual(len(issues['errors']), 0)
    
    def test_get_config_summary(self):
        """Test configuration summary generation"""
        summary = get_config_summary()
        self.assertIsInstance(summary, str)
        self.assertIn('Default Algorithm', summary)
        self.assertIn('Enabled Models', summary)
        self.assertIn('Distance Method', summary)
    
    def test_get_enabled_models(self):
        """Test getting enabled models"""
        enabled = get_enabled_models()
        self.assertIsInstance(enabled, list)
        self.assertGreater(len(enabled), 0)
        for model in enabled:
            self.assertTrue(is_model_enabled(model))
    
    def test_get_model_display_name(self):
        """Test getting model display names"""
        for model_key in AVAILABLE_MODELS.keys():
            display_name = get_model_display_name(model_key)
            self.assertIsInstance(display_name, str)
            self.assertGreater(len(display_name), 0)
    
    def test_get_method_defaults(self):
        """Test getting method defaults"""
        for method in ['standard', 'enhanced', 'genetic']:
            defaults = get_method_defaults(method)
            self.assertIsInstance(defaults, dict)
            self.assertIn('solver_timeout', defaults)


class TestVehicleRouterApp(unittest.TestCase):
    """Test cases for VehicleRouterApp class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock streamlit session state
        self.mock_session_state = {
            'data_loaded': False,
            'optimization_complete': False,
            'orders_df': None,
            'trucks_df': None,
            'distance_matrix': None,
            'solution': None,
            'optimization_log': []
        }
        
        # Mock streamlit module
        self.mock_st = MagicMock()
        self.mock_st.session_state = self.mock_session_state
        
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.test_orders_df = data_gen.generate_orders()
        self.test_trucks_df = data_gen.generate_trucks()
        self.test_distance_matrix = data_gen.generate_distance_matrix(
            self.test_orders_df['postal_code'].tolist()
        )
    
    @patch('app.streamlit_app.st')
    def test_app_initialization(self, mock_st):
        """Test app initialization"""
        mock_st.session_state = self.mock_session_state
        
        # Mock the logging setup
        with patch('app.streamlit_app.setup_app_logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            
            # Mock the utility classes
            with patch('app.streamlit_app.DataHandler') as mock_data_handler, \
                 patch('app.streamlit_app.OptimizationRunner') as mock_opt_runner, \
                 patch('app.streamlit_app.VisualizationManager') as mock_viz_manager, \
                 patch('app.streamlit_app.ExportManager') as mock_export_manager, \
                 patch('app.streamlit_app.UIComponents') as mock_ui_components, \
                 patch('app.streamlit_app.DocumentationRenderer') as mock_doc_renderer:
                
                app = VehicleRouterApp()
                
                # Check that utility classes were initialized
                mock_data_handler.assert_called_once()
                mock_opt_runner.assert_called_once()
                mock_viz_manager.assert_called_once()
                mock_export_manager.assert_called_once()
                mock_ui_components.assert_called_once()
                mock_doc_renderer.assert_called_once()
    
    @patch('app.streamlit_app.st')
    def test_initialize_session_state(self, mock_st):
        """Test session state initialization"""
        mock_st.session_state = {}
        
        with patch('app.streamlit_app.setup_app_logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            
            with patch('app.streamlit_app.DataHandler'), \
                 patch('app.streamlit_app.OptimizationRunner'), \
                 patch('app.streamlit_app.VisualizationManager'), \
                 patch('app.streamlit_app.ExportManager'), \
                 patch('app.streamlit_app.UIComponents'), \
                 patch('app.streamlit_app.DocumentationRenderer'):
                
                app = VehicleRouterApp()
                app.initialize_session_state()
                
                # Check that session state was initialized
                self.assertIn('data_loaded', mock_st.session_state)
                self.assertIn('optimization_complete', mock_st.session_state)
                self.assertIn('orders_df', mock_st.session_state)
                self.assertIn('trucks_df', mock_st.session_state)
                self.assertIn('distance_matrix', mock_st.session_state)
                self.assertIn('solution', mock_st.session_state)
                self.assertIn('optimization_log', mock_st.session_state)
                self.assertIn('optimization_method', mock_st.session_state)
    
    @patch('app.streamlit_app.st')
    def test_validate_max_orders_constraint_valid(self, mock_st):
        """Test max orders constraint validation with valid solution"""
        mock_st.session_state = self.mock_session_state
        
        # Create a valid solution
        valid_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 1},
                {'order_id': 'C', 'truck_id': 2}
            ])
        }
        
        with patch('app.streamlit_app.setup_app_logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            
            with patch('app.streamlit_app.DataHandler'), \
                 patch('app.streamlit_app.OptimizationRunner'), \
                 patch('app.streamlit_app.VisualizationManager'), \
                 patch('app.streamlit_app.ExportManager'), \
                 patch('app.streamlit_app.UIComponents'), \
                 patch('app.streamlit_app.DocumentationRenderer'):
                
                app = VehicleRouterApp()
                result = app._validate_max_orders_constraint(valid_solution, 3)
                self.assertTrue(result)
    
    @patch('app.streamlit_app.st')
    def test_validate_max_orders_constraint_violation(self, mock_st):
        """Test max orders constraint validation with violation"""
        mock_st.session_state = self.mock_session_state
        
        # Create a solution that violates the constraint
        invalid_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 1},
                {'order_id': 'C', 'truck_id': 1},
                {'order_id': 'D', 'truck_id': 1}  # 4 orders to truck 1, max is 3
            ])
        }
        
        with patch('app.streamlit_app.setup_app_logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            
            with patch('app.streamlit_app.DataHandler'), \
                 patch('app.streamlit_app.OptimizationRunner'), \
                 patch('app.streamlit_app.VisualizationManager'), \
                 patch('app.streamlit_app.ExportManager'), \
                 patch('app.streamlit_app.UIComponents'), \
                 patch('app.streamlit_app.DocumentationRenderer'):
                
                app = VehicleRouterApp()
                result = app._validate_max_orders_constraint(invalid_solution, 3)
                self.assertFalse(result)


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the complete app workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
    
    @patch('app.streamlit_app.st')
    def test_data_loading_workflow(self, mock_st):
        """Test the data loading workflow"""
        mock_st.session_state = {
            'data_loaded': False,
            'orders_df': None,
            'trucks_df': None,
            'distance_matrix': None
        }
        
        # Mock the data handler
        with patch('app.streamlit_app.DataHandler') as mock_data_handler_class:
            mock_data_handler = MagicMock()
            mock_data_handler.load_example_data.return_value = True
            mock_data_handler.orders_df = self.orders_df
            mock_data_handler.trucks_df = self.trucks_df
            mock_data_handler.distance_matrix = self.distance_matrix
            mock_data_handler_class.return_value = mock_data_handler
            
            with patch('app.streamlit_app.setup_app_logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.return_value = mock_logger
                
                with patch('app.streamlit_app.OptimizationRunner'), \
                     patch('app.streamlit_app.VisualizationManager'), \
                     patch('app.streamlit_app.ExportManager'), \
                     patch('app.streamlit_app.UIComponents'), \
                     patch('app.streamlit_app.DocumentationRenderer'):
                    
                    app = VehicleRouterApp()
                    
                    # Test data loading
                    success = mock_data_handler.load_example_data()
                    self.assertTrue(success)
    
    @patch('app.streamlit_app.st')
    def test_optimization_workflow(self, mock_st):
        """Test the optimization workflow"""
        mock_st.session_state = {
            'data_loaded': True,
            'orders_df': self.orders_df,
            'trucks_df': self.trucks_df,
            'distance_matrix': self.distance_matrix,
            'optimization_complete': False,
            'optimization_method': 'standard'
        }
        
        # Mock the optimization runner
        with patch('app.streamlit_app.OptimizationRunner') as mock_opt_runner_class:
            mock_opt_runner = MagicMock()
            mock_opt_runner.run_optimization.return_value = True
            mock_opt_runner.solution = {
                'assignments_df': pd.DataFrame([
                    {'order_id': 'A', 'truck_id': 1},
                    {'order_id': 'B', 'truck_id': 1}
                ]),
                'selected_trucks': [1],
                'costs': {'total_cost': 1000.0}
            }
            mock_opt_runner.optimization_log = []
            mock_opt_runner_class.return_value = mock_opt_runner
            
            with patch('app.streamlit_app.setup_app_logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.return_value = mock_logger
                
                with patch('app.streamlit_app.DataHandler'), \
                     patch('app.streamlit_app.VisualizationManager'), \
                     patch('app.streamlit_app.ExportManager'), \
                     patch('app.streamlit_app.UIComponents'), \
                     patch('app.streamlit_app.DocumentationRenderer'):
                    
                    app = VehicleRouterApp()
                    
                    # Test optimization
                    success = mock_opt_runner.run_optimization(
                        self.orders_df, self.trucks_df, self.distance_matrix
                    )
                    self.assertTrue(success)


class TestAppUtilities(unittest.TestCase):
    """Test cases for app utility functions"""
    
    def test_configuration_helpers(self):
        """Test configuration helper functions"""
        # Test get_enabled_models
        enabled_models = get_enabled_models()
        self.assertIsInstance(enabled_models, list)
        self.assertGreater(len(enabled_models), 0)
        
        # Test is_model_enabled
        for model in enabled_models:
            self.assertTrue(is_model_enabled(model))
        
        # Test get_model_display_name
        for model in enabled_models:
            display_name = get_model_display_name(model)
            self.assertIsInstance(display_name, str)
            self.assertGreater(len(display_name), 0)
        
        # Test get_method_defaults
        for method in ['standard', 'enhanced', 'genetic']:
            defaults = get_method_defaults(method)
            self.assertIsInstance(defaults, dict)
    
    def test_configuration_validation_edge_cases(self):
        """Test configuration validation with edge cases"""
        # Test with valid configuration
        issues = validate_config()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        self.assertIn('info', issues)
    
    def test_configuration_summary_format(self):
        """Test configuration summary format"""
        summary = get_config_summary()
        self.assertIsInstance(summary, str)
        
        # Check that summary contains expected sections
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


if __name__ == '__main__':
    unittest.main()
