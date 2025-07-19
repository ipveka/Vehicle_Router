"""
Unit tests for utility functions

This module contains comprehensive unit tests for the utility functions
in the vehicle_router.utils module, testing all helper functions with
various inputs and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import vehicle_router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_router.utils import (
    calculate_distance_matrix,
    format_solution,
    validate_postal_codes,
    calculate_route_distance,
    format_currency,
    get_solution_summary
)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.sample_postal_codes = ['08027', '08028', '08029', '08030', '08031']
        self.sample_orders_df = pd.DataFrame([
            {'order_id': 'A', 'volume': 75.0, 'postal_code': '08031'},
            {'order_id': 'B', 'volume': 50.0, 'postal_code': '08030'},
            {'order_id': 'C', 'volume': 25.0, 'postal_code': '08029'}
        ])
        self.sample_trucks_df = pd.DataFrame([
            {'truck_id': 1, 'capacity': 100.0, 'cost': 1500.0},
            {'truck_id': 2, 'capacity': 50.0, 'cost': 1000.0},
            {'truck_id': 3, 'capacity': 25.0, 'cost': 500.0}
        ])
    
    def test_calculate_distance_matrix_valid_input(self):
        """Test calculate_distance_matrix with valid postal codes"""
        distance_matrix = calculate_distance_matrix(self.sample_postal_codes)
        
        # Check structure
        self.assertIsInstance(distance_matrix, pd.DataFrame)
        self.assertEqual(distance_matrix.shape, (5, 5))
        self.assertListEqual(list(distance_matrix.index), self.sample_postal_codes)
        self.assertListEqual(list(distance_matrix.columns), self.sample_postal_codes)
        
        # Check diagonal is zero
        for code in self.sample_postal_codes:
            self.assertEqual(distance_matrix.loc[code, code], 0.0)
        
        # Check symmetry
        for i, code1 in enumerate(self.sample_postal_codes):
            for j, code2 in enumerate(self.sample_postal_codes):
                self.assertEqual(distance_matrix.loc[code1, code2], 
                               distance_matrix.loc[code2, code1])
        
        # Check specific distances
        self.assertEqual(distance_matrix.loc['08027', '08028'], 1.0)
        self.assertEqual(distance_matrix.loc['08027', '08031'], 4.0)
    
    def test_calculate_distance_matrix_custom_distance_per_unit(self):
        """Test calculate_distance_matrix with custom distance per unit"""
        distance_matrix = calculate_distance_matrix(self.sample_postal_codes, distance_per_unit=2.0)
        
        # Check that distances are scaled
        self.assertEqual(distance_matrix.loc['08027', '08028'], 2.0)
        self.assertEqual(distance_matrix.loc['08027', '08031'], 8.0)
    
    def test_calculate_distance_matrix_invalid_input(self):
        """Test calculate_distance_matrix with invalid inputs"""
        # Test with non-list input
        with self.assertRaises(TypeError):
            calculate_distance_matrix("not_a_list")
        
        # Test with empty list
        with self.assertRaises(ValueError):
            calculate_distance_matrix([])
        
        # Test with non-string postal codes
        with self.assertRaises(TypeError):
            calculate_distance_matrix([8027, 8028, 8029])
        
        # Test with invalid postal code format
        with self.assertRaises(ValueError):
            calculate_distance_matrix(['08027', 'invalid', '08029'])
    
    def test_calculate_distance_matrix_duplicates(self):
        """Test calculate_distance_matrix with duplicate postal codes"""
        postal_codes_with_duplicates = ['08027', '08028', '08027', '08029']
        distance_matrix = calculate_distance_matrix(postal_codes_with_duplicates)
        
        # Should handle duplicates by removing them
        unique_codes = ['08027', '08028', '08029']
        self.assertEqual(distance_matrix.shape, (3, 3))
        self.assertListEqual(sorted(distance_matrix.index.tolist()), unique_codes)
    
    def test_format_solution_valid_input(self):
        """Test format_solution with valid raw solution"""
        raw_solution = {
            'assignments': [
                {'order_id': 'A', 'truck_id': 1, 'value': 1.0},
                {'order_id': 'B', 'truck_id': 2, 'value': 1.0},
                {'order_id': 'C', 'truck_id': 3, 'value': 0.0}  # Not assigned
            ]
        }
        
        formatted = format_solution(raw_solution, self.sample_orders_df, self.sample_trucks_df)
        
        # Check structure
        self.assertIn('selected_trucks', formatted)
        self.assertIn('costs', formatted)
        self.assertIn('summary', formatted)
        
        # Check selected trucks
        self.assertEqual(formatted['selected_trucks'], [1, 2])
        
        # Check costs
        expected_truck_cost = 1500.0 + 1000.0  # Trucks 1 and 2
        self.assertEqual(formatted['costs']['total_truck_cost'], expected_truck_cost)
        self.assertEqual(formatted['costs']['travel_cost'], 100.0)  # 2 trucks * 50
        self.assertEqual(formatted['costs']['total_cost'], expected_truck_cost + 100.0)
    
    def test_format_solution_empty_assignments(self):
        """Test format_solution with no assignments"""
        raw_solution = {'assignments': []}
        
        formatted = format_solution(raw_solution, self.sample_orders_df, self.sample_trucks_df)
        
        # Should handle empty assignments gracefully
        self.assertEqual(formatted['selected_trucks'], [])
        self.assertEqual(formatted['costs']['total_truck_cost'], 0.0)
        self.assertEqual(formatted['costs']['travel_cost'], 0.0)
        self.assertEqual(formatted['costs']['total_cost'], 0.0)
    
    def test_format_solution_missing_assignments_key(self):
        """Test format_solution with missing assignments key"""
        raw_solution = {'other_data': 'value'}
        
        formatted = format_solution(raw_solution, self.sample_orders_df, self.sample_trucks_df)
        
        # Should handle missing key gracefully
        self.assertEqual(formatted['selected_trucks'], [])
    
    def test_format_solution_error_handling(self):
        """Test format_solution error handling with invalid data"""
        raw_solution = {
            'assignments': [
                {'order_id': 'A', 'truck_id': 999, 'value': 1.0}  # Invalid truck ID
            ]
        }
        
        formatted = format_solution(raw_solution, self.sample_orders_df, self.sample_trucks_df)
        
        # Should handle errors and include error message
        self.assertIn('summary', formatted)
        if 'error' in formatted['summary']:
            self.assertIsInstance(formatted['summary']['error'], str)
    
    def test_validate_postal_codes_valid(self):
        """Test validate_postal_codes with valid postal codes"""
        is_valid, invalid_codes = validate_postal_codes(self.sample_postal_codes)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(invalid_codes), 0)
    
    def test_validate_postal_codes_invalid(self):
        """Test validate_postal_codes with invalid postal codes"""
        invalid_postal_codes = ['08027', 'invalid', '08029', '123']
        is_valid, invalid_codes = validate_postal_codes(invalid_postal_codes)
        
        self.assertFalse(is_valid)
        self.assertIn('invalid', invalid_codes)
        self.assertIn('123', invalid_codes)
        self.assertEqual(len(invalid_codes), 2)
    
    def test_validate_postal_codes_mixed(self):
        """Test validate_postal_codes with mixed valid and invalid codes"""
        mixed_codes = ['08027', '08028', 'bad_code', '08030']
        is_valid, invalid_codes = validate_postal_codes(mixed_codes)
        
        self.assertFalse(is_valid)
        self.assertEqual(len(invalid_codes), 1)
        self.assertIn('bad_code', invalid_codes)
    
    def test_validate_postal_codes_empty_list(self):
        """Test validate_postal_codes with empty list"""
        is_valid, invalid_codes = validate_postal_codes([])
        
        self.assertTrue(is_valid)
        self.assertEqual(len(invalid_codes), 0)
    
    def test_calculate_route_distance_valid_route(self):
        """Test calculate_route_distance with valid route"""
        distance_matrix = calculate_distance_matrix(self.sample_postal_codes)
        route = ['08027', '08028', '08030', '08031']
        
        total_distance = calculate_route_distance(route, distance_matrix)
        
        # Expected: 1 + 2 + 1 = 4 km
        expected_distance = 1.0 + 2.0 + 1.0
        self.assertEqual(total_distance, expected_distance)
    
    def test_calculate_route_distance_single_location(self):
        """Test calculate_route_distance with single location"""
        distance_matrix = calculate_distance_matrix(self.sample_postal_codes)
        route = ['08027']
        
        total_distance = calculate_route_distance(route, distance_matrix)
        
        # Single location should have zero distance
        self.assertEqual(total_distance, 0.0)
    
    def test_calculate_route_distance_empty_route(self):
        """Test calculate_route_distance with empty route"""
        distance_matrix = calculate_distance_matrix(self.sample_postal_codes)
        route = []
        
        total_distance = calculate_route_distance(route, distance_matrix)
        
        # Empty route should have zero distance
        self.assertEqual(total_distance, 0.0)
    
    def test_calculate_route_distance_missing_codes(self):
        """Test calculate_route_distance with codes not in distance matrix"""
        distance_matrix = calculate_distance_matrix(['08027', '08028'])
        route = ['08027', '08029', '08030']  # 08029 and 08030 not in matrix
        
        # Should handle missing codes gracefully (with warnings)
        total_distance = calculate_route_distance(route, distance_matrix)
        
        # Should return 0 since no valid segments found
        self.assertEqual(total_distance, 0.0)
    
    def test_format_currency_default(self):
        """Test format_currency with default currency symbol"""
        formatted = format_currency(1500.0)
        self.assertEqual(formatted, "€1,500")
        
        formatted = format_currency(1234.56)
        self.assertEqual(formatted, "€1,235")  # Rounded to nearest integer
    
    def test_format_currency_custom_symbol(self):
        """Test format_currency with custom currency symbol"""
        formatted = format_currency(1500.0, currency="$")
        self.assertEqual(formatted, "$1,500")
        
        formatted = format_currency(2000.0, currency="USD ")
        self.assertEqual(formatted, "USD 2,000")
    
    def test_format_currency_zero_and_negative(self):
        """Test format_currency with zero and negative amounts"""
        self.assertEqual(format_currency(0.0), "€0")
        self.assertEqual(format_currency(-500.0), "€-500")
    
    def test_format_currency_large_numbers(self):
        """Test format_currency with large numbers"""
        self.assertEqual(format_currency(1000000.0), "€1,000,000")
        self.assertEqual(format_currency(1234567.89), "€1,234,568")
    
    def test_get_solution_summary_valid_solution(self):
        """Test get_solution_summary with valid formatted solution"""
        formatted_solution = {
            'selected_trucks': [1, 3, 5],
            'costs': {'total_cost': 3000.0}
        }
        
        summary = get_solution_summary(formatted_solution)
        
        # Check summary format
        self.assertIn("=== VEHICLE ROUTER ===", summary)
        self.assertIn("Selected Trucks: [1, 3, 5]", summary)
        self.assertIn("Total Cost: €3,000", summary)
    
    def test_get_solution_summary_empty_solution(self):
        """Test get_solution_summary with empty solution"""
        formatted_solution = {}
        
        summary = get_solution_summary(formatted_solution)
        
        # Should handle empty solution gracefully
        self.assertIn("=== VEHICLE ROUTER ===", summary)
        self.assertIn("Selected Trucks: []", summary)
        self.assertIn("Total Cost: €0", summary)
    
    def test_get_solution_summary_missing_data(self):
        """Test get_solution_summary with missing data"""
        formatted_solution = {
            'selected_trucks': [1, 2]
            # Missing costs
        }
        
        summary = get_solution_summary(formatted_solution)
        
        # Should handle missing data gracefully
        self.assertIn("=== VEHICLE ROUTER ===", summary)
        self.assertIn("Selected Trucks: [1, 2]", summary)
        self.assertIn("Total Cost: €0", summary)
    
    def test_get_solution_summary_error_handling(self):
        """Test get_solution_summary error handling"""
        # Create a solution that will cause an error by having invalid cost structure
        class BadDict(dict):
            def get(self, key, default=None):
                if key == 'costs':
                    raise ValueError("Simulated error")
                return super().get(key, default)
        
        formatted_solution = BadDict({
            'selected_trucks': [1, 2],
            'costs': {'total_cost': 1000.0}
        })
        
        summary = get_solution_summary(formatted_solution)
        
        # Should handle errors gracefully
        self.assertIn("=== VEHICLE ROUTER ===", summary)
        # Should contain error message
        self.assertIn("Error:", summary)


if __name__ == '__main__':
    unittest.main()