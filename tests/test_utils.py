"""Unit tests for utility functions"""

import unittest
import pandas as pd
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
    
    def test_calculate_distance_matrix(self):
        """Test distance matrix calculation"""
        postal_codes = ['08020', '08021', '08022']
        
        # Test valid input
        matrix = calculate_distance_matrix(postal_codes)
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(matrix.shape, (3, 3))
        
        # Test invalid input
        with self.assertRaises(ValueError):
            calculate_distance_matrix([])
    
    def test_format_currency(self):
        """Test currency formatting"""
        # Test default formatting
        result = format_currency(1234.56)
        self.assertEqual(result, "€1,235")
        
        # Test zero and negative
        self.assertEqual(format_currency(0), "€0")
        self.assertEqual(format_currency(-100), "€-100")
    
    def test_validate_postal_codes(self):
        """Test postal code validation"""
        # Test valid codes
        valid_codes = ['08020', '08021', '08022']
        result = validate_postal_codes(valid_codes)
        self.assertTrue(result)
        
        # Test invalid codes
        invalid_codes = ['invalid', '123', '']
        result, invalid_list = validate_postal_codes(invalid_codes)
        self.assertFalse(result)
        self.assertEqual(len(invalid_list), 3)
    
    def test_calculate_route_distance(self):
        """Test route distance calculation"""
        distance_matrix = pd.DataFrame({
            '08020': [0, 1, 2],
            '08021': [1, 0, 1],
            '08022': [2, 1, 0]
        }, index=['08020', '08021', '08022'])
        
        # Test valid route
        route = ['08020', '08021', '08022']
        distance = calculate_route_distance(route, distance_matrix)
        self.assertEqual(distance, 2.0)
        
        # Test single location
        distance = calculate_route_distance(['08020'], distance_matrix)
        self.assertEqual(distance, 0.0)
    
    def test_format_solution(self):
        """Test solution formatting"""
        solution = {
            'assignments': [
                {'order_id': 'O1', 'truck_id': 'T1'},
                {'order_id': 'O2', 'truck_id': 'T1'}
            ]
        }
        
        orders_df = pd.DataFrame({'order_id': ['O1', 'O2']})
        trucks_df = pd.DataFrame({'truck_id': ['T1']})
        
        result = format_solution(solution, orders_df, trucks_df)
        self.assertIsInstance(result, dict)
        self.assertIn('assignments', result)
        self.assertIn('costs', result)
    
    def test_get_solution_summary(self):
        """Test solution summary generation"""
        solution = {
            'costs': {'total_cost': 1000},
            'selected_trucks': ['T1', 'T2'],
            'utilization': {'T1': {'utilization_percent': 80}}
        }
        
        summary = get_solution_summary(solution)
        self.assertIsInstance(summary, str)
        self.assertIn('1,000', summary)
        self.assertIn('T1', summary)


if __name__ == '__main__':
    unittest.main()