"""Unit tests for DataGenerator class"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory to the path to import vehicle_router
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle_router.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    """Test cases for DataGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.example_data_gen = DataGenerator(use_example_data=True)
        self.random_data_gen = DataGenerator(use_example_data=False, random_seed=42)
    
    def test_data_generator_functionality(self):
        """Test core DataGenerator functionality"""
        # Test initialization
        self.assertIsNotNone(self.example_data_gen)
        self.assertIsNotNone(self.random_data_gen)
        
        # Test orders generation
        orders_df = self.example_data_gen.generate_orders()
        self.assertIsInstance(orders_df, pd.DataFrame)
        self.assertGreater(len(orders_df), 0)
        self.assertIn('order_id', orders_df.columns)
        self.assertIn('postal_code', orders_df.columns)
        self.assertIn('volume', orders_df.columns)
        
        # Test trucks generation
        trucks_df = self.example_data_gen.generate_trucks()
        self.assertIsInstance(trucks_df, pd.DataFrame)
        self.assertGreater(len(trucks_df), 0)
        self.assertIn('truck_id', trucks_df.columns)
        self.assertIn('capacity', trucks_df.columns)
        self.assertIn('cost', trucks_df.columns)
        
        # Test distance matrix generation
        postal_codes = orders_df['postal_code'].tolist()
        distance_matrix = self.example_data_gen.generate_distance_matrix(postal_codes)
        self.assertIsInstance(distance_matrix, pd.DataFrame)
        self.assertEqual(distance_matrix.shape[0], len(postal_codes))
        self.assertEqual(distance_matrix.shape[1], len(postal_codes))
    
    def test_data_generator_edge_cases(self):
        """Test DataGenerator edge cases"""
        # Test empty postal codes list
        with self.assertRaises(ValueError):
            self.example_data_gen.generate_distance_matrix([])
        
        # Test invalid postal codes
        with self.assertRaises(ValueError):
            self.example_data_gen.generate_distance_matrix(['invalid', 'codes'])
        
        # Test reproducibility with seed
        gen1 = DataGenerator(use_example_data=False, random_seed=42)
        gen2 = DataGenerator(use_example_data=False, random_seed=42)
        
        orders1 = gen1.generate_orders()
        orders2 = gen2.generate_orders()
        
        # Should produce same data with same seed
        pd.testing.assert_frame_equal(orders1, orders2)
    
    def test_data_summary(self):
        """Test data summary generation"""
        orders_df = self.example_data_gen.generate_orders()
        trucks_df = self.example_data_gen.generate_trucks()
        postal_codes = orders_df['postal_code'].tolist()
        distance_matrix = self.example_data_gen.generate_distance_matrix(postal_codes)
        
        summary = self.example_data_gen.get_data_summary(orders_df, trucks_df, distance_matrix)
        self.assertIsInstance(summary, dict)
        self.assertIn('orders', summary)
        self.assertIn('trucks', summary)
        self.assertIn('feasibility', summary)


if __name__ == '__main__':
    unittest.main()