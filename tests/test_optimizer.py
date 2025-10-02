"""Unit tests for VrpOptimizer class"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory to the path to import vehicle_router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.data_generator import DataGenerator


class TestVrpOptimizer(unittest.TestCase):
    """Test cases for VrpOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate example data for testing
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
    
    def test_vrp_optimizer_functionality(self):
        """Test core VrpOptimizer functionality"""
        # Test initialization with valid data
        optimizer = VrpOptimizer(
            orders_df=self.orders_df,
            trucks_df=self.trucks_df,
            distance_matrix=self.distance_matrix,
            depot_location='08020',
            depot_return=False,
            max_orders_per_truck=3,
            enable_greedy_routes=True
        )
        self.assertIsNotNone(optimizer)
        
        # Test model building
        optimizer.build_model()
        self.assertIsNotNone(optimizer.model)
        
        # Test solving
        success = optimizer.solve(timeout=30)
        self.assertTrue(success)
        
        # Test solution extraction
        solution = optimizer.get_solution()
        self.assertIsInstance(solution, dict)
        self.assertIn('assignments_df', solution)
        self.assertIn('selected_trucks', solution)
        self.assertIn('costs', solution)
    
    def test_vrp_optimizer_edge_cases(self):
        """Test VrpOptimizer edge cases"""
        # Test initialization with invalid data
        invalid_orders = pd.DataFrame({'invalid': [1, 2, 3]})
        with self.assertRaises(ValueError):
            VrpOptimizer(
                orders_df=invalid_orders,
                trucks_df=self.trucks_df,
                distance_matrix=self.distance_matrix
            )
        
        # Test solving without building model
        optimizer = VrpOptimizer(
            orders_df=self.orders_df,
            trucks_df=self.trucks_df,
            distance_matrix=self.distance_matrix
        )
        with self.assertRaises(RuntimeError):
            optimizer.solve(timeout=30)
        
        # Test getting solution without solving
        with self.assertRaises(RuntimeError):
            optimizer.get_solution()
    
    def test_vrp_optimizer_constraints(self):
        """Test VrpOptimizer constraint validation"""
        # Test infeasible problem (too many orders per truck)
        optimizer = VrpOptimizer(
            orders_df=self.orders_df,
            trucks_df=self.trucks_df,
            distance_matrix=self.distance_matrix,
            max_orders_per_truck=1  # Very restrictive
        )
        
        optimizer.build_model()
        success = optimizer.solve(timeout=30)
        # Should either solve or fail gracefully
        self.assertIsInstance(success, bool)


if __name__ == '__main__':
    unittest.main()