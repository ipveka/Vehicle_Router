"""
Unit tests for App Utilities

This module contains comprehensive unit tests for the Vehicle Router app utilities,
testing data handling, optimization running, visualization, export, and UI components.
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

from app_utils.data_handler import DataHandler
from app_utils.optimization_runner import OptimizationRunner
from app_utils.visualization_manager import VisualizationManager
from app_utils.export_manager import ExportManager
from app_utils.ui_components import UIComponents
from app_utils.documentation import DocumentationRenderer
from vehicle_router.data_generator import DataGenerator


class TestDataHandler(unittest.TestCase):
    """Test cases for DataHandler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_handler = DataHandler()
        
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
    
    def test_data_handler_initialization(self):
        """Test DataHandler initialization"""
        self.assertIsNone(self.data_handler.orders_df)
        self.assertIsNone(self.data_handler.trucks_df)
        self.assertIsNone(self.data_handler.distance_matrix)
    
    @patch('app_utils.data_handler.DataGenerator')
    def test_load_example_data(self, mock_data_gen_class):
        """Test loading example data"""
        # Mock the DataGenerator
        mock_data_gen = MagicMock()
        mock_data_gen.generate_orders.return_value = self.orders_df
        mock_data_gen.generate_trucks.return_value = self.trucks_df
        mock_data_gen.generate_distance_matrix.return_value = self.distance_matrix
        mock_data_gen_class.return_value = mock_data_gen
        
        # Test loading example data
        success = self.data_handler.load_example_data()
        
        self.assertTrue(success)
        self.assertIsNotNone(self.data_handler.orders_df)
        self.assertIsNotNone(self.data_handler.trucks_df)
        self.assertIsNotNone(self.data_handler.distance_matrix)
    
    def test_load_custom_data(self):
        """Test loading custom data from file objects"""
        # Create mock file objects
        mock_orders_file = MagicMock()
        mock_orders_file.read.return_value = b"order_id,volume,postal_code\nA,25.0,08027\nB,50.0,08028"
        
        mock_trucks_file = MagicMock()
        mock_trucks_file.read.return_value = b"truck_id,capacity,cost\n1,100.0,1000.0\n2,50.0,500.0"
        
        # Test loading custom data
        success = self.data_handler.load_custom_data(mock_orders_file, mock_trucks_file)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.data_handler.orders_df)
        self.assertIsNotNone(self.data_handler.trucks_df)
        self.assertIsNotNone(self.data_handler.distance_matrix)
    
    def test_reload_distance_matrix(self):
        """Test reloading distance matrix"""
        # First load some data
        self.data_handler.orders_df = self.orders_df
        self.data_handler.trucks_df = self.trucks_df
        
        # Test reloading distance matrix
        success = self.data_handler.reload_distance_matrix()
        
        self.assertTrue(success)
        self.assertIsNotNone(self.data_handler.distance_matrix)


class TestOptimizationRunner(unittest.TestCase):
    """Test cases for OptimizationRunner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_runner = OptimizationRunner()
        
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
    
    def test_optimization_runner_initialization(self):
        """Test OptimizationRunner initialization"""
        self.assertIsNone(self.optimization_runner.solution)
        self.assertEqual(self.optimization_runner.optimization_log, [])
    
    @patch('app_utils.optimization_runner.VrpOptimizer')
    def test_run_optimization_standard(self, mock_optimizer_class):
        """Test running standard optimization"""
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.solve.return_value = True
        mock_optimizer.get_solution.return_value = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 1}
            ]),
            'selected_trucks': [1],
            'costs': {'total_cost': 1000.0}
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Test running optimization
        success = self.optimization_runner.run_optimization(
            self.orders_df, self.trucks_df, self.distance_matrix,
            optimization_method='standard'
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(self.optimization_runner.solution)
    
    @patch('app_utils.optimization_runner.GeneticVrpOptimizer')
    def test_run_optimization_genetic(self, mock_optimizer_class):
        """Test running genetic optimization"""
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.solve.return_value = True
        mock_optimizer.get_solution.return_value = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 2}
            ]),
            'selected_trucks': [1, 2],
            'costs': {'total_cost': 1500.0}
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Test running optimization
        success = self.optimization_runner.run_optimization(
            self.orders_df, self.trucks_df, self.distance_matrix,
            optimization_method='genetic'
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(self.optimization_runner.solution)
    
    @patch('app_utils.optimization_runner.EnhancedVrpOptimizer')
    def test_run_optimization_enhanced(self, mock_optimizer_class):
        """Test running enhanced optimization"""
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.solve.return_value = True
        mock_optimizer.get_solution.return_value = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 1}
            ]),
            'selected_trucks': [1],
            'costs': {'total_cost': 1200.0}
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Test running optimization
        success = self.optimization_runner.run_optimization(
            self.orders_df, self.trucks_df, self.distance_matrix,
            optimization_method='enhanced'
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(self.optimization_runner.solution)
    
    def test_run_optimization_invalid_method(self):
        """Test running optimization with invalid method"""
        # Test with invalid optimization method
        success = self.optimization_runner.run_optimization(
            self.orders_df, self.trucks_df, self.distance_matrix,
            optimization_method='invalid_method'
        )
        
        self.assertFalse(success)
        self.assertIsNone(self.optimization_runner.solution)


class TestVisualizationManager(unittest.TestCase):
    """Test cases for VisualizationManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualization_manager = VisualizationManager()
        
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
        
        # Create test solution
        self.solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 1},
                {'order_id': 'C', 'truck_id': 2}
            ]),
            'selected_trucks': [1, 2],
            'costs': {'total_cost': 2000.0},
            'routes_df': pd.DataFrame([
                {'truck_id': 1, 'route_sequence': ['A', 'B'], 'route_distance': 10.0},
                {'truck_id': 2, 'route_sequence': ['C'], 'route_distance': 5.0}
            ]),
            'utilization': {
                1: {'volume_used': 75.0, 'capacity': 100.0},
                2: {'volume_used': 25.0, 'capacity': 50.0}
            }
        }
    
    def test_visualization_manager_initialization(self):
        """Test VisualizationManager initialization"""
        self.assertIsNotNone(self.visualization_manager)
    
    def test_create_orders_volume_chart(self):
        """Test creating orders volume chart"""
        fig = self.visualization_manager.create_orders_volume_chart(self.orders_df)
        self.assertIsNotNone(fig)
    
    def test_create_orders_distribution_chart(self):
        """Test creating orders distribution chart"""
        fig = self.visualization_manager.create_orders_distribution_chart(self.orders_df)
        self.assertIsNotNone(fig)
    
    def test_create_trucks_capacity_chart(self):
        """Test creating trucks capacity chart"""
        fig = self.visualization_manager.create_trucks_capacity_chart(self.trucks_df)
        self.assertIsNotNone(fig)
    
    def test_create_trucks_cost_efficiency_chart(self):
        """Test creating trucks cost efficiency chart"""
        fig = self.visualization_manager.create_trucks_cost_efficiency_chart(self.trucks_df)
        self.assertIsNotNone(fig)
    
    def test_create_route_visualization(self):
        """Test creating route visualization"""
        fig = self.visualization_manager.create_route_visualization(
            self.solution['routes_df'], self.orders_df, self.trucks_df
        )
        self.assertIsNotNone(fig)
    
    def test_create_cost_breakdown_chart(self):
        """Test creating cost breakdown chart"""
        fig = self.visualization_manager.create_cost_breakdown_chart(
            self.solution['costs'], self.trucks_df
        )
        self.assertIsNotNone(fig)
    
    def test_create_volume_usage_chart(self):
        """Test creating volume usage chart"""
        fig = self.visualization_manager.create_volume_usage_chart(
            self.solution['utilization'], self.trucks_df
        )
        self.assertIsNotNone(fig)


class TestExportManager(unittest.TestCase):
    """Test cases for ExportManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.export_manager = ExportManager()
        
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        
        # Create test solution
        self.solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1},
                {'order_id': 'B', 'truck_id': 1},
                {'order_id': 'C', 'truck_id': 2}
            ]),
            'selected_trucks': [1, 2],
            'costs': {'total_cost': 2000.0},
            'routes_df': pd.DataFrame([
                {'truck_id': 1, 'route_sequence': ['A', 'B'], 'route_distance': 10.0},
                {'truck_id': 2, 'route_sequence': ['C'], 'route_distance': 5.0}
            ])
        }
    
    def test_export_manager_initialization(self):
        """Test ExportManager initialization"""
        self.assertIsNotNone(self.export_manager)
    
    def test_create_excel_report(self):
        """Test creating Excel report"""
        excel_buffer = self.export_manager.create_excel_report(
            self.orders_df, self.trucks_df, self.solution
        )
        
        self.assertIsNotNone(excel_buffer)
        self.assertGreater(len(excel_buffer.getvalue()), 0)
    
    def test_create_summary_csv(self):
        """Test creating summary CSV"""
        csv_buffer = self.export_manager.create_summary_csv(self.solution)
        
        self.assertIsNotNone(csv_buffer)
        self.assertGreater(len(csv_buffer.getvalue()), 0)
    
    def test_create_detailed_report(self):
        """Test creating detailed report"""
        report_buffer = self.export_manager.create_detailed_report(
            self.orders_df, self.trucks_df, self.solution
        )
        
        self.assertIsNotNone(report_buffer)
        self.assertGreater(len(report_buffer.getvalue()), 0)


class TestUIComponents(unittest.TestCase):
    """Test cases for UIComponents class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ui_components = UIComponents()
    
    def test_ui_components_initialization(self):
        """Test UIComponents initialization"""
        self.assertIsNotNone(self.ui_components)


class TestDocumentationRenderer(unittest.TestCase):
    """Test cases for DocumentationRenderer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.documentation_renderer = DocumentationRenderer()
    
    def test_documentation_renderer_initialization(self):
        """Test DocumentationRenderer initialization"""
        self.assertIsNotNone(self.documentation_renderer)
    
    def test_render_introduction(self):
        """Test rendering introduction"""
        # This would normally render to streamlit, but we can test the method exists
        self.assertTrue(hasattr(self.documentation_renderer, 'render_introduction'))
    
    def test_render_methodology(self):
        """Test rendering methodology"""
        # This would normally render to streamlit, but we can test the method exists
        self.assertTrue(hasattr(self.documentation_renderer, 'render_methodology'))


class TestAppUtilsIntegration(unittest.TestCase):
    """Integration tests for app utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
    
    def test_data_handler_and_optimization_runner_integration(self):
        """Test integration between DataHandler and OptimizationRunner"""
        # Initialize components
        data_handler = DataHandler()
        optimization_runner = OptimizationRunner()
        
        # Load data
        data_handler.orders_df = self.orders_df
        data_handler.trucks_df = self.trucks_df
        data_handler.distance_matrix = self.distance_matrix
        
        # Mock the optimizer
        with patch('app_utils.optimization_runner.VrpOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.solve.return_value = True
            mock_optimizer.get_solution.return_value = {
                'assignments_df': pd.DataFrame([
                    {'order_id': 'A', 'truck_id': 1}
                ]),
                'selected_trucks': [1],
                'costs': {'total_cost': 1000.0}
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            # Run optimization
            success = optimization_runner.run_optimization(
                data_handler.orders_df, data_handler.trucks_df, data_handler.distance_matrix
            )
            
            self.assertTrue(success)
            self.assertIsNotNone(optimization_runner.solution)
    
    def test_optimization_runner_and_visualization_manager_integration(self):
        """Test integration between OptimizationRunner and VisualizationManager"""
        # Initialize components
        optimization_runner = OptimizationRunner()
        visualization_manager = VisualizationManager()
        
        # Mock the optimizer
        with patch('app_utils.optimization_runner.VrpOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.solve.return_value = True
            mock_optimizer.get_solution.return_value = {
                'assignments_df': pd.DataFrame([
                    {'order_id': 'A', 'truck_id': 1}
                ]),
                'selected_trucks': [1],
                'costs': {'total_cost': 1000.0},
                'routes_df': pd.DataFrame([
                    {'truck_id': 1, 'route_sequence': ['A'], 'route_distance': 5.0}
                ]),
                'utilization': {1: {'volume_used': 25.0, 'capacity': 100.0}}
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            # Run optimization
            success = optimization_runner.run_optimization(
                self.orders_df, self.trucks_df, self.distance_matrix
            )
            
            self.assertTrue(success)
            
            # Test visualization
            if optimization_runner.solution:
                fig = visualization_manager.create_route_visualization(
                    optimization_runner.solution['routes_df'],
                    self.orders_df, self.trucks_df
                )
                self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
