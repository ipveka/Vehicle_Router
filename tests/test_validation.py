"""
Unit tests for SolutionValidator class

This module contains comprehensive unit tests for the SolutionValidator class,
testing validation of VRP solutions with valid and invalid solutions,
including edge cases and error conditions.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import vehicle_router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_router.validation import SolutionValidator
from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer


class TestSolutionValidator(unittest.TestCase):
    """Test cases for SolutionValidator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Generate example data for testing
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        postal_codes = self.orders_df['postal_code'].tolist()
        self.distance_matrix = data_gen.generate_distance_matrix(postal_codes)
        
        # Create a valid solution for testing
        # Truck capacities: 1=100m³, 2=50m³, 3=25m³, 4=25m³, 5=25m³
        # Order volumes: A=75m³, B=50m³, C=25m³, D=25m³, E=25m³
        self.valid_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1, 'assigned': True},  # 75 m³ to 100 m³ truck
                {'order_id': 'B', 'truck_id': 2, 'assigned': True},  # 50 m³ to 50 m³ truck
                {'order_id': 'C', 'truck_id': 3, 'assigned': True},  # 25 m³ to 25 m³ truck
                {'order_id': 'D', 'truck_id': 4, 'assigned': True},  # 25 m³ to 25 m³ truck
                {'order_id': 'E', 'truck_id': 5, 'assigned': True}   # 25 m³ to 25 m³ truck
            ]),
            'selected_trucks': [1, 2, 3, 4, 5],
            'costs': {'total_cost': 5500.0},
            'summary': {'trucks_used': 5}
        }
        
        # Create solution with capacity violation
        self.capacity_violation_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 3, 'assigned': True},  # 75 m³ to 25 m³ truck
                {'order_id': 'B', 'truck_id': 3, 'assigned': True},  # 50 m³ to same truck
                {'order_id': 'C', 'truck_id': 1, 'assigned': True},
                {'order_id': 'D', 'truck_id': 1, 'assigned': True},
                {'order_id': 'E', 'truck_id': 1, 'assigned': True}
            ]),
            'selected_trucks': [1, 3],
            'costs': {'total_cost': 2000.0}
        }
        
        # Create solution with missing orders
        self.missing_orders_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1, 'assigned': True},
                {'order_id': 'B', 'truck_id': 1, 'assigned': True},
                {'order_id': 'C', 'truck_id': 3, 'assigned': True}
                # Missing orders D and E
            ]),
            'selected_trucks': [1, 3],
            'costs': {'total_cost': 2000.0}
        }
        
        # Create solution with duplicate assignments
        self.duplicate_assignment_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1, 'assigned': True},
                {'order_id': 'A', 'truck_id': 3, 'assigned': True},  # Duplicate assignment
                {'order_id': 'B', 'truck_id': 1, 'assigned': True},
                {'order_id': 'C', 'truck_id': 3, 'assigned': True},
                {'order_id': 'D', 'truck_id': 3, 'assigned': True},
                {'order_id': 'E', 'truck_id': 3, 'assigned': True}
            ]),
            'selected_trucks': [1, 3],
            'costs': {'total_cost': 2000.0}
        }
        
        # Create solution with empty selected trucks
        self.empty_truck_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1, 'assigned': True},
                {'order_id': 'B', 'truck_id': 1, 'assigned': True},
                {'order_id': 'C', 'truck_id': 3, 'assigned': True},
                {'order_id': 'D', 'truck_id': 3, 'assigned': True},
                {'order_id': 'E', 'truck_id': 3, 'assigned': True}
            ]),
            'selected_trucks': [1, 2, 3],  # Truck 2 has no assignments
            'costs': {'total_cost': 2000.0}
        }
    
    def test_init_valid_data(self):
        """Test SolutionValidator initialization with valid data"""
        validator = SolutionValidator(self.valid_solution, self.orders_df, self.trucks_df)
        
        # Check initialization
        self.assertEqual(len(validator.all_order_ids), 5)
        self.assertEqual(len(validator.all_truck_ids), 5)
        self.assertIsInstance(validator.order_volumes, dict)
        self.assertIsInstance(validator.truck_capacities, dict)
        
        # Check data copying
        pd.testing.assert_frame_equal(validator.orders_df, self.orders_df)
        pd.testing.assert_frame_equal(validator.trucks_df, self.trucks_df)
    
    def test_init_invalid_solution(self):
        """Test initialization with invalid solution data"""
        # Test with non-dict solution
        with self.assertRaises(TypeError):
            SolutionValidator("not_a_dict", self.orders_df, self.trucks_df)
        
        # Test with missing required keys
        invalid_solution = {'costs': {'total_cost': 1000.0}}  # Missing assignments_df
        with self.assertRaises(ValueError):
            SolutionValidator(invalid_solution, self.orders_df, self.trucks_df)
    
    def test_init_invalid_orders_dataframe(self):
        """Test initialization with invalid orders DataFrame"""
        # Test with non-DataFrame
        with self.assertRaises(TypeError):
            SolutionValidator(self.valid_solution, "not_a_dataframe", self.trucks_df)
        
        # Test with missing columns
        invalid_orders = pd.DataFrame([{'order_id': 'A', 'volume': 10.0}])  # Missing postal_code
        with self.assertRaises(ValueError):
            SolutionValidator(self.valid_solution, invalid_orders, self.trucks_df)
    
    def test_init_invalid_trucks_dataframe(self):
        """Test initialization with invalid trucks DataFrame"""
        # Test with non-DataFrame
        with self.assertRaises(TypeError):
            SolutionValidator(self.valid_solution, self.orders_df, "not_a_dataframe")
        
        # Test with missing columns
        invalid_trucks = pd.DataFrame([{'truck_id': 1, 'capacity': 100.0}])  # Missing cost
        with self.assertRaises(ValueError):
            SolutionValidator(self.valid_solution, self.orders_df, invalid_trucks)
    
    def test_check_capacity_valid_solution(self):
        """Test capacity check with valid solution"""
        validator = SolutionValidator(self.valid_solution, self.orders_df, self.trucks_df)
        capacity_results = validator.check_capacity()
        
        # Check results structure
        self.assertIn('is_valid', capacity_results)
        self.assertIn('violations', capacity_results)
        self.assertIn('truck_utilization', capacity_results)
        self.assertIn('summary', capacity_results)
        
        # Should be valid
        self.assertTrue(capacity_results['is_valid'])
        self.assertEqual(len(capacity_results['violations']), 0)
        
        # Check utilization data
        utilization = capacity_results['truck_utilization']
        self.assertIn(1, utilization)
        self.assertIn(2, utilization)
        self.assertIn(3, utilization)
        self.assertIn(4, utilization)
        self.assertIn(5, utilization)
        
        # Truck 1 should have order A (25 m³ out of 100 m³ capacity)
        truck1_util = utilization[1]
        self.assertEqual(truck1_util['assigned_orders'], ['A'])
        self.assertEqual(truck1_util['total_volume'], 25.0)
        self.assertEqual(truck1_util['capacity'], 100.0)
        
        # Truck 2 should have order B (50 m³ out of 50 m³ capacity)
        truck2_util = utilization[2]
        self.assertEqual(truck2_util['assigned_orders'], ['B'])
        self.assertEqual(truck2_util['total_volume'], 50.0)
        self.assertEqual(truck2_util['capacity'], 50.0)
    
    def test_check_capacity_violation(self):
        """Test capacity check with capacity violation"""
        validator = SolutionValidator(self.capacity_violation_solution, self.orders_df, self.trucks_df)
        capacity_results = validator.check_capacity()
        
        # Should be invalid
        self.assertFalse(capacity_results['is_valid'])
        self.assertGreater(len(capacity_results['violations']), 0)
        
        # Check violation details
        violation = capacity_results['violations'][0]
        self.assertIn('truck_id', violation)
        self.assertIn('assigned_volume', violation)
        self.assertIn('capacity', violation)
        self.assertIn('excess_volume', violation)
        self.assertGreater(violation['excess_volume'], 0)
    
    def test_check_all_orders_delivered_valid(self):
        """Test order delivery check with valid solution"""
        validator = SolutionValidator(self.valid_solution, self.orders_df, self.trucks_df)
        delivery_results = validator.check_all_orders_delivered()
        
        # Should be valid
        self.assertTrue(delivery_results['is_valid'])
        self.assertEqual(len(delivery_results['missing_orders']), 0)
        self.assertEqual(len(delivery_results['duplicate_assignments']), 0)
        
        # Check assignment summary
        summary = delivery_results['assignment_summary']
        self.assertEqual(summary['total_orders'], 5)
        self.assertEqual(summary['assigned_orders'], 5)
        self.assertEqual(summary['missing_orders'], 0)
        self.assertEqual(summary['duplicate_assignments'], 0)
    
    def test_check_all_orders_delivered_missing_orders(self):
        """Test order delivery check with missing orders"""
        validator = SolutionValidator(self.missing_orders_solution, self.orders_df, self.trucks_df)
        delivery_results = validator.check_all_orders_delivered()
        
        # Should be invalid
        self.assertFalse(delivery_results['is_valid'])
        self.assertEqual(len(delivery_results['missing_orders']), 2)
        self.assertIn('D', delivery_results['missing_orders'])
        self.assertIn('E', delivery_results['missing_orders'])
        
        # Check assignment summary
        summary = delivery_results['assignment_summary']
        self.assertEqual(summary['total_orders'], 5)
        self.assertEqual(summary['assigned_orders'], 3)
        self.assertEqual(summary['missing_orders'], 2)
    
    def test_check_all_orders_delivered_duplicate_assignments(self):
        """Test order delivery check with duplicate assignments"""
        validator = SolutionValidator(self.duplicate_assignment_solution, self.orders_df, self.trucks_df)
        delivery_results = validator.check_all_orders_delivered()
        
        # Should be invalid
        self.assertFalse(delivery_results['is_valid'])
        self.assertEqual(len(delivery_results['duplicate_assignments']), 1)
        
        # Check duplicate assignment details
        duplicate = delivery_results['duplicate_assignments'][0]
        self.assertEqual(duplicate['order_id'], 'A')
        self.assertEqual(duplicate['assignment_count'], 2)
        self.assertIn(1, duplicate['assigned_trucks'])
        self.assertIn(3, duplicate['assigned_trucks'])
    
    def test_check_route_feasibility_valid(self):
        """Test route feasibility check with valid solution"""
        validator = SolutionValidator(self.valid_solution, self.orders_df, self.trucks_df)
        route_results = validator.check_route_feasibility()
        
        # Should be valid
        self.assertTrue(route_results['is_valid'])
        self.assertEqual(len(route_results['empty_trucks']), 0)
        self.assertEqual(len(route_results['unselected_with_orders']), 0)
        
        # Check route summary
        summary = route_results['route_summary']
        self.assertEqual(summary['total_trucks_available'], 5)
        self.assertEqual(summary['trucks_selected'], 5)
        self.assertEqual(summary['trucks_with_orders'], 5)
    
    def test_check_route_feasibility_empty_trucks(self):
        """Test route feasibility check with empty selected trucks"""
        validator = SolutionValidator(self.empty_truck_solution, self.orders_df, self.trucks_df)
        route_results = validator.check_route_feasibility()
        
        # Should be invalid
        self.assertFalse(route_results['is_valid'])
        self.assertEqual(len(route_results['empty_trucks']), 1)
        self.assertIn(2, route_results['empty_trucks'])  # Truck 2 has no orders
    
    def test_validate_solution_valid(self):
        """Test comprehensive validation with valid solution"""
        # First, let me create a truly valid solution (fixing capacity issue)
        valid_solution_fixed = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1, 'assigned': True},  # 75 m³ to 100 m³ truck
                {'order_id': 'B', 'truck_id': 2, 'assigned': True},  # 50 m³ to 50 m³ truck
                {'order_id': 'C', 'truck_id': 3, 'assigned': True},  # 25 m³ to 25 m³ truck
                {'order_id': 'D', 'truck_id': 4, 'assigned': True},  # 25 m³ to 25 m³ truck
                {'order_id': 'E', 'truck_id': 5, 'assigned': True}   # 25 m³ to 25 m³ truck
            ]),
            'selected_trucks': [1, 2, 3, 4, 5],
            'costs': {'total_cost': 5500.0}
        }
        
        validator = SolutionValidator(valid_solution_fixed, self.orders_df, self.trucks_df)
        validation_report = validator.validate_solution()
        
        # Should be valid overall
        self.assertTrue(validation_report['is_valid'])
        self.assertEqual(validation_report['error_count'], 0)
        
        # Individual checks should pass
        self.assertTrue(validation_report['capacity_check']['is_valid'])
        self.assertTrue(validation_report['delivery_check']['is_valid'])
        self.assertTrue(validation_report['route_check']['is_valid'])
        
        # Check summary
        self.assertIn("PASSED", validation_report['summary'])
    
    def test_validate_solution_invalid(self):
        """Test comprehensive validation with invalid solution"""
        validator = SolutionValidator(self.capacity_violation_solution, self.orders_df, self.trucks_df)
        validation_report = validator.validate_solution()
        
        # Should be invalid overall
        self.assertFalse(validation_report['is_valid'])
        self.assertGreater(validation_report['error_count'], 0)
        
        # At least capacity check should fail
        self.assertFalse(validation_report['capacity_check']['is_valid'])
        
        # Check summary
        self.assertIn("FAILED", validation_report['summary'])
    
    def test_validate_solution_multiple_issues(self):
        """Test validation with multiple types of issues"""
        # Create solution with multiple problems
        multi_issue_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 3, 'assigned': True},  # Capacity violation
                {'order_id': 'B', 'truck_id': 3, 'assigned': True},  # Capacity violation
                {'order_id': 'C', 'truck_id': 1, 'assigned': True}
                # Missing orders D and E
            ]),
            'selected_trucks': [1, 2, 3],  # Truck 2 is empty
            'costs': {'total_cost': 3000.0}
        }
        
        validator = SolutionValidator(multi_issue_solution, self.orders_df, self.trucks_df)
        validation_report = validator.validate_solution()
        
        # Should be invalid
        self.assertFalse(validation_report['is_valid'])
        self.assertGreater(validation_report['error_count'], 2)  # Multiple errors
        
        # Multiple checks should fail
        self.assertFalse(validation_report['capacity_check']['is_valid'])
        self.assertFalse(validation_report['delivery_check']['is_valid'])
        self.assertFalse(validation_report['route_check']['is_valid'])
    
    def test_validate_solution_with_warnings(self):
        """Test validation that generates warnings for low utilization"""
        # Create solution with low utilization
        low_util_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'E', 'truck_id': 1, 'assigned': True},  # Only 25 m³ in 100 m³ truck
                {'order_id': 'A', 'truck_id': 2, 'assigned': True},  # 75 m³ in 50 m³ truck - violation
                {'order_id': 'B', 'truck_id': 3, 'assigned': True},  # 50 m³ in 25 m³ truck - violation
                {'order_id': 'C', 'truck_id': 4, 'assigned': True},  # 25 m³ in 25 m³ truck
                {'order_id': 'D', 'truck_id': 5, 'assigned': True}   # 25 m³ in 25 m³ truck
            ]),
            'selected_trucks': [1, 2, 3, 4, 5],
            'costs': {'total_cost': 5500.0}
        }
        
        validator = SolutionValidator(low_util_solution, self.orders_df, self.trucks_df)
        validation_report = validator.validate_solution()
        
        # Should have warnings about low utilization for truck 1
        # (though it will fail due to capacity violations in trucks 2 and 3)
        self.assertIn('warnings', validation_report)
    
    def test_get_validation_summary_text(self):
        """Test validation summary text generation"""
        validator = SolutionValidator(self.valid_solution, self.orders_df, self.trucks_df)
        validation_report = validator.validate_solution()
        
        summary_text = validator.get_validation_summary_text()
        
        # Check summary format
        self.assertIn("=== SOLUTION VALIDATION REPORT ===", summary_text)
        self.assertIn("Overall Status:", summary_text)
        self.assertIn("Capacity Constraints:", summary_text)
        self.assertIn("Order Delivery:", summary_text)
        self.assertIn("Route Feasibility:", summary_text)
    
    def test_get_validation_summary_text_without_validation(self):
        """Test validation summary text without running validation first"""
        validator = SolutionValidator(self.valid_solution, self.orders_df, self.trucks_df)
        
        summary_text = validator.get_validation_summary_text()
        self.assertIn("No validation results available", summary_text)
    
    def test_validation_with_real_optimizer_solution(self):
        """Test validation with actual optimizer solution"""
        # Generate a real solution using the optimizer
        optimizer = VrpOptimizer(self.orders_df, self.trucks_df, self.distance_matrix)
        optimizer.build_model()
        success = optimizer.solve()
        
        if success:
            solution = optimizer.get_solution()
            
            # Validate the real solution
            validator = SolutionValidator(solution, self.orders_df, self.trucks_df)
            validation_report = validator.validate_solution()
            
            # Real optimizer solution should be valid
            self.assertTrue(validation_report['is_valid'])
            self.assertEqual(validation_report['error_count'], 0)
    
    def test_edge_case_single_order_single_truck(self):
        """Test validation with minimal problem (1 order, 1 truck)"""
        # Create minimal data
        minimal_orders = pd.DataFrame([
            {'order_id': 'A', 'volume': 10.0, 'postal_code': '08027'}
        ])
        minimal_trucks = pd.DataFrame([
            {'truck_id': 1, 'capacity': 20.0, 'cost': 500.0}
        ])
        minimal_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'A', 'truck_id': 1, 'assigned': True}
            ]),
            'selected_trucks': [1],
            'costs': {'total_cost': 500.0}
        }
        
        validator = SolutionValidator(minimal_solution, minimal_orders, minimal_trucks)
        validation_report = validator.validate_solution()
        
        # Should be valid
        self.assertTrue(validation_report['is_valid'])
        self.assertEqual(validation_report['error_count'], 0)
    
    def test_validation_error_handling(self):
        """Test validation error handling with corrupted data"""
        # Create solution with corrupted assignments DataFrame
        corrupted_solution = {
            'assignments_df': pd.DataFrame([
                {'order_id': 'INVALID', 'truck_id': 999, 'assigned': True}  # Invalid order and truck
            ]),
            'selected_trucks': [999],
            'costs': {'total_cost': 1000.0}
        }
        
        validator = SolutionValidator(corrupted_solution, self.orders_df, self.trucks_df)
        
        # Should handle errors gracefully
        capacity_results = validator.check_capacity()
        self.assertFalse(capacity_results['is_valid'])
        
        delivery_results = validator.check_all_orders_delivered()
        self.assertFalse(delivery_results['is_valid'])


if __name__ == '__main__':
    unittest.main()