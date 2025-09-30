"""
Integration tests for Vehicle Router App

This module contains integration tests that verify the complete app workflow
from data loading through optimization to solution generation.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock streamlit before importing app modules
sys.modules['streamlit'] = MagicMock()

from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.genetic_optimizer import GeneticVrpOptimizer
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer
from vehicle_router.validation import SolutionValidator
from app.config import validate_config, get_config_summary


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
    
    def test_complete_standard_optimization_workflow(self):
        """Test complete workflow with standard MILP optimization"""
        # Step 1: Initialize optimizer
        optimizer = VrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        
        # Step 2: Build and solve model
        optimizer.build_model()
        success = optimizer.solve()
        
        self.assertTrue(success)
        
        # Step 3: Get solution
        solution = optimizer.get_solution()
        self.assertIsNotNone(solution)
        self.assertIn('assignments_df', solution)
        self.assertIn('selected_trucks', solution)
        self.assertIn('costs', solution)
        
        # Step 4: Validate solution
        validator = SolutionValidator(
            solution, self.orders_df, self.trucks_df
        )
        validation_result = validator.validate_solution()
        
        self.assertTrue(validation_result['is_valid'])
    
    def test_complete_genetic_optimization_workflow(self):
        """Test complete workflow with genetic algorithm optimization"""
        # Step 1: Initialize optimizer
        optimizer = GeneticVrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        
        # Step 2: Set parameters
        optimizer.set_parameters(
            population_size=20,  # Smaller for testing
            max_generations=10,  # Fewer for testing
            mutation_rate=0.1
        )
        
        # Step 3: Solve
        success = optimizer.solve(timeout=30)  # Short timeout for testing
        
        self.assertTrue(success)
        
        # Step 4: Get solution
        solution = optimizer.get_solution()
        self.assertIsNotNone(solution)
        self.assertIn('assignments_df', solution)
        self.assertIn('selected_trucks', solution)
        self.assertIn('costs', solution)
        
        # Step 5: Validate solution
        validator = SolutionValidator(
            solution, self.orders_df, self.trucks_df
        )
        validation_result = validator.validate_solution()
        
        self.assertTrue(validation_result['is_valid'])
    
    def test_complete_enhanced_optimization_workflow(self):
        """Test complete workflow with enhanced MILP optimization"""
        # Step 1: Initialize optimizer
        optimizer = EnhancedVrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        
        # Step 2: Set objective weights
        optimizer.set_objective_weights(cost_weight=0.6, distance_weight=0.4)
        
        # Step 3: Build and solve model
        optimizer.build_model()
        success = optimizer.solve(timeout=30)  # Short timeout for testing
        
        self.assertTrue(success)
        
        # Step 4: Get solution
        solution = optimizer.get_solution()
        self.assertIsNotNone(solution)
        self.assertIn('assignments_df', solution)
        self.assertIn('selected_trucks', solution)
        self.assertIn('costs', solution)
        
        # Step 5: Validate solution
        validator = SolutionValidator(
            solution, self.orders_df, self.trucks_df
        )
        validation_result = validator.validate_solution()
        
        self.assertTrue(validation_result['is_valid'])
    
    def test_data_generation_and_validation(self):
        """Test data generation and validation workflow"""
        # Step 1: Generate data
        data_gen = DataGenerator(use_example_data=True)
        orders_df = data_gen.generate_orders()
        trucks_df = data_gen.generate_trucks()
        distance_matrix = data_gen.generate_distance_matrix(
            orders_df['postal_code'].tolist()
        )
        
        # Step 2: Validate data structure
        self.assertIsInstance(orders_df, pd.DataFrame)
        self.assertIsInstance(trucks_df, pd.DataFrame)
        self.assertIsInstance(distance_matrix, pd.DataFrame)
        
        # Check required columns
        self.assertIn('order_id', orders_df.columns)
        self.assertIn('volume', orders_df.columns)
        self.assertIn('postal_code', orders_df.columns)
        
        self.assertIn('truck_id', trucks_df.columns)
        self.assertIn('capacity', trucks_df.columns)
        self.assertIn('cost', trucks_df.columns)
        
        # Check data validity
        self.assertTrue((orders_df['volume'] > 0).all())
        self.assertTrue((trucks_df['capacity'] > 0).all())
        self.assertTrue((trucks_df['cost'] > 0).all())
        
        # Check distance matrix
        self.assertEqual(distance_matrix.shape[0], distance_matrix.shape[1])
        self.assertTrue((distance_matrix >= 0).all().all())
    
    def test_configuration_system_integration(self):
        """Test configuration system integration"""
        # Test configuration validation
        issues = validate_config()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        self.assertIn('info', issues)
        
        # Test configuration summary
        summary = get_config_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_optimization_methods_comparison(self):
        """Test comparison between different optimization methods"""
        results = {}
        
        # Test Standard MILP
        standard_optimizer = VrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        standard_optimizer.build_model()
        standard_success = standard_optimizer.solve()
        
        if standard_success:
            standard_solution = standard_optimizer.get_solution()
            results['standard'] = {
                'success': True,
                'cost': standard_solution['costs']['total_cost'],
                'trucks': len(standard_solution['selected_trucks'])
            }
        else:
            results['standard'] = {'success': False}
        
        # Test Genetic Algorithm
        genetic_optimizer = GeneticVrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        genetic_optimizer.set_parameters(population_size=20, max_generations=10)
        genetic_success = genetic_optimizer.solve(timeout=30)
        
        if genetic_success:
            genetic_solution = genetic_optimizer.get_solution()
            results['genetic'] = {
                'success': True,
                'cost': genetic_solution['costs']['total_cost'],
                'trucks': len(genetic_solution['selected_trucks'])
            }
        else:
            results['genetic'] = {'success': False}
        
        # Verify both methods succeeded
        self.assertTrue(results['standard']['success'])
        self.assertTrue(results['genetic']['success'])
        
        # Both should use reasonable number of trucks
        self.assertGreater(results['standard']['trucks'], 0)
        self.assertGreater(results['genetic']['trucks'], 0)
        self.assertLessEqual(results['standard']['trucks'], len(self.trucks_df))
        self.assertLessEqual(results['genetic']['trucks'], len(self.trucks_df))
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # Test with invalid data
        invalid_orders = pd.DataFrame([
            {'order_id': 'A', 'volume': -10.0, 'postal_code': '08027'}  # Negative volume
        ])
        
        with self.assertRaises(ValueError):
            VrpOptimizer(invalid_orders, self.trucks_df, self.distance_matrix)
        
        # Test with insufficient capacity - this should raise an error during initialization
        small_trucks = pd.DataFrame([
            {'truck_id': 1, 'capacity': 10.0, 'cost': 500.0}  # Too small capacity
        ])
        
        with self.assertRaises(ValueError):
            VrpOptimizer(self.orders_df, small_trucks, self.distance_matrix)
    
    def test_solution_validation_comprehensive(self):
        """Test comprehensive solution validation"""
        # Generate a solution
        optimizer = VrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        optimizer.build_model()
        success = optimizer.solve()
        
        self.assertTrue(success)
        
        solution = optimizer.get_solution()
        
        # Test comprehensive validation
        validator = SolutionValidator(solution, self.orders_df, self.trucks_df)
        validation_result = validator.validate_solution()
        
        self.assertTrue(validation_result['is_valid'])
        self.assertIn('capacity_check', validation_result)
        self.assertIn('delivery_check', validation_result)  # Updated key name
        self.assertIn('route_check', validation_result)  # Updated key name
        
        # All checks should pass
        self.assertTrue(validation_result['capacity_check']['is_valid'])
        self.assertTrue(validation_result['delivery_check']['is_valid'])
        self.assertTrue(validation_result['route_check']['is_valid'])


class TestAppPerformance(unittest.TestCase):
    """Performance tests for the app"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate test data
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        self.distance_matrix = data_gen.generate_distance_matrix(
            self.orders_df['postal_code'].tolist()
        )
    
    def test_standard_optimization_performance(self):
        """Test standard optimization performance"""
        import time
        
        optimizer = VrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        
        start_time = time.time()
        optimizer.build_model()
        success = optimizer.solve()
        end_time = time.time()
        
        self.assertTrue(success)
        self.assertLess(end_time - start_time, 10.0)  # Should complete within 10 seconds
    
    def test_genetic_optimization_performance(self):
        """Test genetic optimization performance"""
        import time
        
        optimizer = GeneticVrpOptimizer(
            self.orders_df, self.trucks_df, self.distance_matrix,
            max_orders_per_truck=3
        )
        optimizer.set_parameters(population_size=20, max_generations=10)
        
        start_time = time.time()
        success = optimizer.solve(timeout=30)
        end_time = time.time()
        
        self.assertTrue(success)
        self.assertLess(end_time - start_time, 30.0)  # Should complete within 30 seconds


if __name__ == '__main__':
    unittest.main()
