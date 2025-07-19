"""
Data Handler Module

This module provides the DataHandler class for loading and managing data
in the Streamlit application. It handles both example data loading and
custom data upload functionality.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Tuple
import io

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vehicle_router.data_generator import DataGenerator


class DataHandler:
    """
    Data Handler for Streamlit Application
    
    This class manages data loading, validation, and preparation for the
    Vehicle Router Streamlit application. It supports both example data
    and custom data upload functionality.
    """
    
    def __init__(self):
        """Initialize the DataHandler"""
        self.orders_df = None
        self.trucks_df = None
        self.distance_matrix = None
        self.data_generator = None
    
    def load_example_data(self) -> bool:
        """
        Load the example data using DataGenerator
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            # Initialize data generator with example data
            self.data_generator = DataGenerator(use_example_data=True)
            
            # Generate example data
            self.orders_df = self.data_generator.generate_orders()
            self.trucks_df = self.data_generator.generate_trucks()
            
            # Generate distance matrix
            postal_codes = self.orders_df['postal_code'].tolist()
            self.distance_matrix = self.data_generator.generate_distance_matrix(postal_codes)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")
            return False
    
    def load_custom_data(self, orders_file, trucks_file) -> bool:
        """
        Load custom data from uploaded CSV files
        
        Args:
            orders_file: Uploaded orders CSV file
            trucks_file: Uploaded trucks CSV file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            # Read orders CSV
            orders_content = orders_file.read()
            self.orders_df = pd.read_csv(io.StringIO(orders_content.decode('utf-8')))
            
            # Read trucks CSV
            trucks_content = trucks_file.read()
            self.trucks_df = pd.read_csv(io.StringIO(trucks_content.decode('utf-8')))
            
            # Validate data format
            if not self._validate_orders_data():
                return False
            
            if not self._validate_trucks_data():
                return False
            
            # Generate distance matrix
            postal_codes = self.orders_df['postal_code'].tolist()
            self.data_generator = DataGenerator(use_example_data=False)
            self.distance_matrix = self.data_generator.generate_distance_matrix(postal_codes)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading custom data: {str(e)}")
            return False
    
    def _validate_orders_data(self) -> bool:
        """
        Validate orders data format and content
        
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['order_id', 'volume', 'postal_code']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in self.orders_df.columns]
        if missing_columns:
            st.error(f"Orders CSV missing required columns: {missing_columns}")
            return False
        
        # Check data types and values
        if not pd.api.types.is_numeric_dtype(self.orders_df['volume']):
            st.error("Volume column must contain numeric values")
            return False
        
        if self.orders_df['volume'].min() <= 0:
            st.error("All order volumes must be positive")
            return False
        
        if self.orders_df['order_id'].duplicated().any():
            st.error("Order IDs must be unique")
            return False
        
        return True
    
    def _validate_trucks_data(self) -> bool:
        """
        Validate trucks data format and content
        
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['truck_id', 'capacity', 'cost']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in self.trucks_df.columns]
        if missing_columns:
            st.error(f"Trucks CSV missing required columns: {missing_columns}")
            return False
        
        # Check data types and values
        if not pd.api.types.is_numeric_dtype(self.trucks_df['capacity']):
            st.error("Capacity column must contain numeric values")
            return False
        
        if not pd.api.types.is_numeric_dtype(self.trucks_df['cost']):
            st.error("Cost column must contain numeric values")
            return False
        
        if self.trucks_df['capacity'].min() <= 0:
            st.error("All truck capacities must be positive")
            return False
        
        if self.trucks_df['cost'].min() < 0:
            st.error("All truck costs must be non-negative")
            return False
        
        if self.trucks_df['truck_id'].duplicated().any():
            st.error("Truck IDs must be unique")
            return False
        
        return True
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            dict: Summary statistics
        """
        if self.orders_df is None or self.trucks_df is None:
            return {}
        
        return {
            'orders': {
                'count': len(self.orders_df),
                'total_volume': self.orders_df['volume'].sum(),
                'avg_volume': self.orders_df['volume'].mean(),
                'min_volume': self.orders_df['volume'].min(),
                'max_volume': self.orders_df['volume'].max()
            },
            'trucks': {
                'count': len(self.trucks_df),
                'total_capacity': self.trucks_df['capacity'].sum(),
                'avg_capacity': self.trucks_df['capacity'].mean(),
                'min_capacity': self.trucks_df['capacity'].min(),
                'max_capacity': self.trucks_df['capacity'].max(),
                'total_cost': self.trucks_df['cost'].sum(),
                'avg_cost': self.trucks_df['cost'].mean()
            },
            'feasibility': {
                'capacity_sufficient': self.trucks_df['capacity'].sum() >= self.orders_df['volume'].sum(),
                'capacity_utilization': (self.orders_df['volume'].sum() / self.trucks_df['capacity'].sum()) * 100
            }
        }