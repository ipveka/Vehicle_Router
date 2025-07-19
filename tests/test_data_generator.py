"""
Unit tests for DataGenerator class

This module contains comprehensive unit tests for the DataGenerator class,
testing all methods with both example data and random data generation,
including edge cases and error conditions.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to import vehicle_router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_router.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    """Test cases for DataGenerator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.example_data_gen = DataGenerator(use_example_data=True)
        self.random_data_gen = DataGenerator(use_example_data=False, random_seed=42)
    
    def test_init_example_data(self):
        """Test DataGenerator initialization with example data"""
        data_gen = DataGenerator(use_example_data=True)
        self.assertTrue(data_gen.use_example_data)
        self.assertIsNone(data_gen.random_seed)
    
    def test_init_random_data(self):
        """Test DataGenerator initialization with random data"""
        data_gen = DataGenerator(use_example_data=False, random_seed=123)
        self.assertFalse(data_gen.use_example_data)
        self.assertEqual(data_gen.random_seed, 123)
    
    def test_generate_orders_example_data(self):
        """Test generate_orders method with example data"""
        orders_df = self.example_data_gen.generate_orders()
        
        # Check DataFrame structure
        self.assertIsInstance(orders_df, pd.DataFrame)
        self.assertEqual(len(orders_df), 5)
        self.assertListEqual(list(orders_df.columns), ['order_id', 'volume', 'postal_code'])
        
        # Check specific example data
        expected_orders = [
            {'order_id': 'A', 'volume': 25.0, 'postal_code': '08031'},
            {'order_id': 'B', 'volume': 50.0, 'postal_code': '08030'},
            {'order_id': 'C', 'volume': 25.0, 'postal_code': '08029'},
            {'order_id': 'D', 'volume': 25.0, 'postal_code': '08028'},
            {'order_id': 'E', 'volume': 25.0, 'postal_code': '08027'}
        ]
        
        for i, expected in enumerate(expected_orders):
            self.assertEqual(orders_df.iloc[i]['order_id'], expected['order_id'])
            self.assertEqual(orders_df.iloc[i]['volume'], expected['volume'])
            self.assertEqual(orders_df.iloc[i]['postal_code'], expected['postal_code'])
        
        # Check total volume
        self.assertEqual(orders_df['volume'].sum(), 150.0)
        
        # Check data types
        self.assertTrue(orders_df['order_id'].dtype == 'object')
        self.assertTrue(pd.api.types.is_numeric_dtype(orders_df['volume']))
        self.assertTrue(orders_df['postal_code'].dtype == 'object')
    
    def test_generate_orders_random_data(self):
        """Test generate_orders method with random data"""
        orders_df = self.random_data_gen.generate_orders()
        
        # Check DataFrame structure
        self.assertIsInstance(orders_df, pd.DataFrame)
        self.assertGreaterEqual(len(orders_df), 3)
        self.assertLessEqual(len(orders_df), 7)
        self.assertListEqual(list(orders_df.columns), ['order_id', 'volume', 'postal_code'])
        
        # Check data validity
        self.assertTrue(all(orders_df['volume'] >= 10.0))
        self.assertTrue(all(orders_df['volume'] <= 80.0))
        self.assertTrue(all(orders_df['postal_code'].str.match(r'^\d{5}$')))
        
        # Check order IDs are sequential letters
        expected_ids = [chr(65 + i) for i in range(len(orders_df))]
        self.assertListEqual(orders_df['order_id'].tolist(), expected_ids)
    
    def test_generate_trucks_example_data(self):
        """Test generate_trucks method with example data"""
        trucks_df = self.example_data_gen.generate_trucks()
        
        # Check DataFrame structure
        self.assertIsInstance(trucks_df, pd.DataFrame)
        self.assertEqual(len(trucks_df), 5)
        self.assertListEqual(list(trucks_df.columns), ['truck_id', 'capacity', 'cost'])
        
        # Check specific example data
        expected_trucks = [
            {'truck_id': 1, 'capacity': 100.0, 'cost': 1500.0},
            {'truck_id': 2, 'capacity': 50.0, 'cost': 1000.0},
            {'truck_id': 3, 'capacity': 25.0, 'cost': 500.0},
            {'truck_id': 4, 'capacity': 25.0, 'cost': 1500.0},
            {'truck_id': 5, 'capacity': 25.0, 'cost': 1000.0}
        ]
        
        for i, expected in enumerate(expected_trucks):
            self.assertEqual(trucks_df.iloc[i]['truck_id'], expected['truck_id'])
            self.assertEqual(trucks_df.iloc[i]['capacity'], expected['capacity'])
            self.assertEqual(trucks_df.iloc[i]['cost'], expected['cost'])
        
        # Check total capacity
        self.assertEqual(trucks_df['capacity'].sum(), 225.0)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(trucks_df['truck_id']))
        self.assertTrue(pd.api.types.is_numeric_dtype(trucks_df['capacity']))
        self.assertTrue(pd.api.types.is_numeric_dtype(trucks_df['cost']))
    
    def test_generate_trucks_random_data(self):
        """Test generate_trucks method with random data"""
        trucks_df = self.random_data_gen.generate_trucks()
        
        # Check DataFrame structure
        self.assertIsInstance(trucks_df, pd.DataFrame)
        self.assertGreaterEqual(len(trucks_df), 3)
        self.assertLessEqual(len(trucks_df), 6)
        self.assertListEqual(list(trucks_df.columns), ['truck_id', 'capacity', 'cost'])
        
        # Check data validity
        self.assertTrue(all(trucks_df['capacity'] > 0))
        self.assertTrue(all(trucks_df['cost'] >= 500.0))
        self.assertTrue(all(trucks_df['cost'] <= 2000.0))
        
        # Check truck IDs are sequential
        expected_ids = list(range(1, len(trucks_df) + 1))
        self.assertListEqual(trucks_df['truck_id'].tolist(), expected_ids)
    
    def test_generate_distance_matrix_example_data(self):
        """Test generate_distance_matrix method with example postal codes"""
        postal_codes = ['08027', '08028', '08029', '08030', '08031']
        distance_matrix = self.example_data_gen.generate_distance_matrix(postal_codes)
        
        # Check DataFrame structure
        self.assertIsInstance(distance_matrix, pd.DataFrame)
        self.assertEqual(distance_matrix.shape, (5, 5))
        self.assertListEqual(list(distance_matrix.index), postal_codes)
        self.assertListEqual(list(distance_matrix.columns), postal_codes)
        
        # Check diagonal is zero
        for code in postal_codes:
            self.assertEqual(distance_matrix.loc[code, code], 0.0)
        
        # Check symmetry
        for i, code1 in enumerate(postal_codes):
            for j, code2 in enumerate(postal_codes):
                self.assertEqual(distance_matrix.loc[code1, code2], 
                               distance_matrix.loc[code2, code1])
        
        # Check specific distances (1km per unit difference)
        self.assertEqual(distance_matrix.loc['08027', '08028'], 1.0)
        self.assertEqual(distance_matrix.loc['08027', '08031'], 4.0)
        self.assertEqual(distance_matrix.loc['08029', '08030'], 1.0)
    
    def test_generate_distance_matrix_empty_list(self):
        """Test generate_distance_matrix with empty postal codes list"""
        with self.assertRaises(ValueError):
            self.example_data_gen.generate_distance_matrix([])
    
    def test_generate_distance_matrix_invalid_codes(self):
        """Test generate_distance_matrix with invalid postal codes"""
        invalid_codes = ['08027', 'invalid', '08029']
        with self.assertRaises(ValueError):
            self.example_data_gen.generate_distance_matrix(invalid_codes)
    
    def test_generate_distance_matrix_non_string_codes(self):
        """Test generate_distance_matrix with non-string postal codes"""
        with self.assertRaises(TypeError):
            self.example_data_gen.generate_distance_matrix([8027, 8028, 8029])
    
    def test_generate_distance_matrix_duplicate_codes(self):
        """Test generate_distance_matrix with duplicate postal codes"""
        postal_codes = ['08027', '08028', '08027', '08029']
        distance_matrix = self.example_data_gen.generate_distance_matrix(postal_codes)
        
        # Should handle duplicates by removing them
        unique_codes = ['08027', '08028', '08029']
        self.assertEqual(distance_matrix.shape, (3, 3))
        self.assertListEqual(sorted(distance_matrix.index.tolist()), unique_codes)
    
    def test_get_data_summary(self):
        """Test get_data_summary method"""
        orders_df = self.example_data_gen.generate_orders()
        trucks_df = self.example_data_gen.generate_trucks()
        postal_codes = orders_df['postal_code'].tolist()
        distance_matrix = self.example_data_gen.generate_distance_matrix(postal_codes)
        
        summary = self.example_data_gen.get_data_summary(orders_df, trucks_df, distance_matrix)
        
        # Check summary structure
        self.assertIn('orders', summary)
        self.assertIn('trucks', summary)
        self.assertIn('distances', summary)
        self.assertIn('feasibility', summary)
        
        # Check orders summary
        orders_summary = summary['orders']
        self.assertEqual(orders_summary['count'], 5)
        self.assertEqual(orders_summary['total_volume'], 150.0)
        self.assertEqual(orders_summary['min_volume'], 25.0)
        self.assertEqual(orders_summary['max_volume'], 50.0)
        
        # Check trucks summary
        trucks_summary = summary['trucks']
        self.assertEqual(trucks_summary['count'], 5)
        self.assertEqual(trucks_summary['total_capacity'], 225.0)
        self.assertEqual(trucks_summary['min_capacity'], 25.0)
        self.assertEqual(trucks_summary['max_capacity'], 100.0)
        
        # Check feasibility
        feasibility = summary['feasibility']
        self.assertTrue(feasibility['capacity_sufficient'])
        self.assertAlmostEqual(feasibility['capacity_utilization'], 66.67, places=1)
    
    def test_reproducibility_with_seed(self):
        """Test that random data generation is reproducible with same seed"""
        data_gen1 = DataGenerator(use_example_data=False, random_seed=42)
        data_gen2 = DataGenerator(use_example_data=False, random_seed=42)
        
        orders1 = data_gen1.generate_orders()
        orders2 = data_gen2.generate_orders()
        
        # Should generate identical data with same seed
        pd.testing.assert_frame_equal(orders1, orders2)
        
        trucks1 = data_gen1.generate_trucks()
        trucks2 = data_gen2.generate_trucks()
        
        pd.testing.assert_frame_equal(trucks1, trucks2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different random data"""
        data_gen1 = DataGenerator(use_example_data=False, random_seed=42)
        data_gen2 = DataGenerator(use_example_data=False, random_seed=123)
        
        orders1 = data_gen1.generate_orders()
        orders2 = data_gen2.generate_orders()
        
        # Should generate different data with different seeds
        # (Note: there's a small chance they could be identical, but very unlikely)
        self.assertFalse(orders1.equals(orders2))


if __name__ == '__main__':
    unittest.main()