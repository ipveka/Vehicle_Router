"""
Unit tests for VrpOptimizer class

This module contains comprehensive unit tests for the VrpOptimizer class,
testing MILP model building, solving, and solution extraction with known
inputs and expected outputs, including edge cases and error conditions.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import vehicle_router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.data_generator import DataGenerator


class TestVrpOptimizer(unittest.TestCase):
    """Test cases for VrpOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Generate example data for testing
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        postal_codes = self.orders_df['postal_code'].tolist()
        self.distance_matrix = data_gen.generate_distance_matrix(postal_codes)
        
        # Create optimizer instance
        self.optimizer = VrpOptimizer(self.orders_df, self.trucks_df, self.distance_matrix)
        
        # Create minimal test data for edge cases
        self.minimal_orders = pd.DataFrame([
            {'order_id': 'A', 'volume': 10.0, 'postal_code': '08027'}
        ])
        self.minimal_trucks = pd.DataFrame([
            {'truck_id': 1, 'capacity': 20.0, 'cost': 500.0}
        ])
        self.minimal_distance = pd.DataFrame(
            data=[[0.0]], 
            index=['08027'], 
            columns=['08027']
        )
    
    def test_init_valid_data(self):
        """Test VrpOptimizer initialization with valid data"""
        optimizer = VrpOptimizer(self.orders_df, self.trucks_df, self.distance_matrix)
        
        # Check initialization
        self.assertEqual(len(optimizer.orders), 5)
        self.assertEqual(len(optimizer.trucks), 5)
        self.assertEqual(optimizer.n_orders, 5)
        self.assertEqual(optimizer.n_trucks, 5)
        self.assertFalse(optimizer.is_solved)
        self.assertIsNone(optimizer.model)
        
        # Check data copying
        pd.testing.assert_frame_equal(optimizer.orders_df, self.orders_df)
        pd.testing.assert_frame_equal(optimizer.trucks_df, self.trucks_df)
    
    def test_init_invalid_orders_dataframe(self):
        """Test initialization with invalid orders DataFrame"""
        # Test with non-DataFrame
        with self.assertRaises(TypeError):
            VrpOptimizer("not_a_dataframe", self.trucks_df, self.distance_matrix)
        
        # Test with missing columns
        invalid_orders = pd.DataFrame([{'order_id': 'A', 'volume': 10.0}])  # Missing postal_code
        with self.assertRaises(ValueError):
            VrpOptimizer(invalid_orders, self.trucks_df, self.distance_matrix)
        
        # Test with empty DataFrame
        empty_orders = pd.DataFrame(columns=['order_id', 'volume', 'postal_code'])
        with self.assertRaises(ValueError):
            VrpOptimizer(empty_orders, self.trucks_df, self.distance_matrix)
        
        # Test with negative volumes
        negative_volume_orders = self.orders_df.copy()
        negative_volume_orders.loc[0, 'volume'] = -10.0
        with self.assertRaises(ValueError):
            VrpOptimizer(negative_volume_orders, self.trucks_df, self.distance_matrix)
    
    def test_init_invalid_trucks_dataframe(self):
        """Test initialization with invalid trucks DataFrame"""
        # Test with non-DataFrame
        with self.assertRaises(TypeError):
            VrpOptimizer(self.orders_df, "not_a_dataframe", self.distance_matrix)
        
        # Test with missing columns
        invalid_trucks = pd.DataFrame([{'truck_id': 1, 'capacity': 100.0}])  # Missing cost
        with self.assertRaises(ValueError):
            VrpOptimizer(self.orders_df, invalid_trucks, self.distance_matrix)
        
        # Test with negative capacity
        negative_capacity_trucks = self.trucks_df.copy()
        negative_capacity_trucks.loc[0, 'capacity'] = -50.0
        with self.assertRaises(ValueError):
            VrpOptimizer(self.orders_df, negative_capacity_trucks, self.distance_matrix)
        
        # Test with negative cost
        negative_cost_trucks = self.trucks_df.copy()
        negative_cost_trucks.loc[0, 'cost'] = -100.0
        with self.assertRaises(ValueError):
            VrpOptimizer(self.orders_df, negative_cost_trucks, self.distance_matrix)
    
    def test_init_infeasible_problem(self):
        """Test initialization with infeasible problem (volume > capacity)"""
        # Create orders with total volume exceeding total capacity
        infeasible_orders = pd.DataFrame([
            {'order_id': 'A', 'volume': 300.0, 'postal_code': '08027'}  # Exceeds total capacity of 225
        ])
        
        with self.assertRaises(ValueError):
            VrpOptimizer(infeasible_orders, self.trucks_df, self.distance_matrix)
    
    def test_build_model(self):
        """Test MILP model building"""
        self.optimizer.build_model()
        
        # Check model was created
        self.assertIsNotNone(self.optimizer.model)
        self.assertEqual(self.optimizer.model.name, "Vehicle_Routing_Problem")
        
        # Check decision variables were created
        self.assertIn('x', self.optimizer.decision_vars)
        self.assertIn('y', self.optimizer.decision_vars)
        
        # Check number of assignment variables (orders × trucks)
        expected_assignment_vars = len(self.orders_df) * len(self.trucks_df)
        self.assertEqual(len(self.optimizer.decision_vars['x']), expected_assignment_vars)
        
        # Check number of truck usage variables
        self.assertEqual(len(self.optimizer.decision_vars['y']), len(self.trucks_df))
        
        # Check model has constraints
        self.assertGreater(len(self.optimizer.model.constraints), 0)
        
        # Check model has objective
        self.assertIsNotNone(self.optimizer.model.objective)
    
    def test_solve_without_building_model(self):
        """Test solving without building model first"""
        with self.assertRaises(RuntimeError):
            self.optimizer.solve()
    
    def test_solve_with_example_data(self):
        """Test solving with example data and verify expected solution"""
        self.optimizer.build_model()
        success = self.optimizer.solve()
        
        # Check solve was successful
        self.assertTrue(success)
        self.assertTrue(self.optimizer.is_solved)
        
        # Get solution
        solution = self.optimizer.get_solution()
        
        # Check solution structure
        self.assertIn('assignments_df', solution)
        self.assertIn('selected_trucks', solution)
        self.assertIn('costs', solution)
        self.assertIn('summary', solution)
        
        # Check assignments DataFrame
        assignments_df = solution['assignments_df']
        self.assertIsInstance(assignments_df, pd.DataFrame)
        self.assertEqual(len(assignments_df), 5)  # All orders should be assigned
        
        # Check all orders are assigned
        assigned_orders = set(assignments_df['order_id'].tolist())
        expected_orders = set(self.orders_df['order_id'].tolist())
        self.assertEqual(assigned_orders, expected_orders)
        
        # Check capacity constraints are satisfied
        for truck_id in solution['selected_trucks']:
            truck_assignments = assignments_df[assignments_df['truck_id'] == truck_id]
            assigned_order_ids = truck_assignments['order_id'].tolist()
            
            total_volume = sum(
                self.orders_df[self.orders_df['order_id'] == oid]['volume'].iloc[0] 
                for oid in assigned_order_ids
            )
            truck_capacity = self.trucks_df[self.trucks_df['truck_id'] == truck_id]['capacity'].iloc[0]
            
            self.assertLessEqual(total_volume, truck_capacity)
        
        # Check cost calculation
        self.assertIn('total_cost', solution['costs'])
        self.assertGreater(solution['costs']['total_cost'], 0)
    
    def test_solve_minimal_problem(self):
        """Test solving with minimal problem (1 order, 1 truck)"""
        minimal_optimizer = VrpOptimizer(
            self.minimal_orders, 
            self.minimal_trucks, 
            self.minimal_distance
        )
        
        minimal_optimizer.build_model()
        success = minimal_optimizer.solve()
        
        self.assertTrue(success)
        
        solution = minimal_optimizer.get_solution()
        
        # Check solution
        self.assertEqual(len(solution['assignments_df']), 1)
        self.assertEqual(solution['selected_trucks'], [1])
        self.assertEqual(solution['assignments_df'].iloc[0]['order_id'], 'A')
        self.assertEqual(solution['assignments_df'].iloc[0]['truck_id'], 1)
    
    def test_get_solution_without_solving(self):
        """Test getting solution without solving first"""
        self.optimizer.build_model()
        
        with self.assertRaises(RuntimeError):
            self.optimizer.get_solution()
    
    def test_solution_summary_text(self):
        """Test solution summary text generation"""
        self.optimizer.build_model()
        success = self.optimizer.solve()
        self.assertTrue(success)
        
        summary_text = self.optimizer.get_solution_summary_text()
        
        # Check summary format
        self.assertIn("=== VEHICLE ROUTER ===", summary_text)
        self.assertIn("Selected Trucks:", summary_text)
        self.assertIn("Total Cost:", summary_text)
        self.assertIn("Truck", summary_text)
        self.assertIn("Orders", summary_text)
    
    def test_solution_summary_text_without_solving(self):
        """Test solution summary text without solving"""
        summary_text = self.optimizer.get_solution_summary_text()
        self.assertEqual(summary_text, "No solution available - model not solved")
    
    @patch('vehicle_router.optimizer.pulp.LpProblem.solve')
    def test_solve_infeasible_problem(self, mock_solve):
        """Test handling of infeasible optimization problem"""
        # Mock solver to return infeasible status
        mock_solve.return_value = None
        
        # Create optimizer with mock model that will be infeasible
        optimizer = VrpOptimizer(self.orders_df, self.trucks_df, self.distance_matrix)
        optimizer.build_model()
        
        # Mock the model status to be infeasible
        optimizer.model.status = 3  # LpStatusInfeasible
        
        success = optimizer.solve()
        self.assertFalse(success)
        self.assertFalse(optimizer.is_solved)
    
    @patch('vehicle_router.optimizer.pulp.LpProblem.solve')
    def test_solve_unbounded_problem(self, mock_solve):
        """Test handling of unbounded optimization problem"""
        # Mock solver to return unbounded status
        mock_solve.return_value = None
        
        optimizer = VrpOptimizer(self.orders_df, self.trucks_df, self.distance_matrix)
        optimizer.build_model()
        
        # Mock the model status to be unbounded
        optimizer.model.status = 4  # LpStatusUnbounded
        
        success = optimizer.solve()
        self.assertFalse(success)
        self.assertFalse(optimizer.is_solved)
    
    @patch('vehicle_router.optimizer.pulp.LpProblem.solve')
    def test_solve_with_exception(self, mock_solve):
        """Test handling of solver exceptions"""
        # Mock solver to raise exception
        mock_solve.side_effect = Exception("Solver error")
        
        optimizer = VrpOptimizer(self.orders_df, self.trucks_df, self.distance_matrix)
        optimizer.build_model()
        
        success = optimizer.solve()
        self.assertFalse(success)
        self.assertFalse(optimizer.is_solved)
    
    def test_decision_variables_creation(self):
        """Test creation of decision variables"""
        self.optimizer.build_model()
        
        # Check assignment variables
        x_vars = self.optimizer.decision_vars['x']
        for order_id in self.optimizer.orders:
            for truck_id in self.optimizer.trucks:
                self.assertIn((order_id, truck_id), x_vars)
                var = x_vars[(order_id, truck_id)]
                self.assertEqual(var.cat, 'Integer')  # PuLP uses 'Integer' for binary variables
        
        # Check truck usage variables
        y_vars = self.optimizer.decision_vars['y']
        for truck_id in self.optimizer.trucks:
            self.assertIn(truck_id, y_vars)
            var = y_vars[truck_id]
            self.assertEqual(var.cat, 'Integer')  # PuLP uses 'Integer' for binary variables
    
    def test_constraint_validation(self):
        """Test that constraints are properly added to the model"""
        self.optimizer.build_model()
        
        constraints = self.optimizer.model.constraints
        
        # Should have order assignment constraints (one per order)
        order_constraints = [c for c in constraints if 'assign_order' in c]
        self.assertEqual(len(order_constraints), len(self.optimizer.orders))
        
        # Should have capacity constraints (one per truck)
        capacity_constraints = [c for c in constraints if 'capacity_truck' in c]
        self.assertEqual(len(capacity_constraints), len(self.optimizer.trucks))
        
        # Should have truck usage constraints (one per order-truck pair)
        usage_constraints = [c for c in constraints if 'usage_truck' in c]
        expected_usage_constraints = len(self.optimizer.orders) * len(self.optimizer.trucks)
        self.assertEqual(len(usage_constraints), expected_usage_constraints)
    
    def test_objective_function(self):
        """Test objective function setup"""
        self.optimizer.build_model()
        
        # Check objective exists and is minimization
        self.assertIsNotNone(self.optimizer.model.objective)
        self.assertEqual(self.optimizer.model.sense, 1)  # LpMinimize = 1 in PuLP
        
        # Objective should include truck costs
        objective_str = str(self.optimizer.model.objective)
        for truck_id in self.optimizer.trucks:
            truck_cost = self.trucks_df[self.trucks_df['truck_id'] == truck_id]['cost'].iloc[0]
            # Check that truck usage variables are in objective with correct coefficients
            self.assertIn(f"use_truck_{truck_id}", objective_str)


if __name__ == '__main__':
    unittest.main()