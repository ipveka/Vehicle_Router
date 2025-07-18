"""
Utility Functions Module

This module provides helper functions for the Vehicle Routing Problem optimization.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def calculate_distance_matrix(postal_codes: List[str], distance_per_unit: float = 1.0) -> pd.DataFrame:
    """Calculate distance matrix between postal code locations"""
    logger.info(f"Calculating distance matrix for {len(postal_codes)} postal codes...")
    
    # Input validation
    if not isinstance(postal_codes, list):
        raise TypeError("postal_codes must be a list")
    
    if not postal_codes:
        raise ValueError("postal_codes cannot be empty")
    
    if not all(isinstance(code, str) for code in postal_codes):
        raise TypeError("All postal codes must be strings")
    
    # Validate postal code format
    postal_pattern = re.compile(r'^\d{5}$')
    invalid_codes = [code for code in postal_codes if not postal_pattern.match(code)]
    if invalid_codes:
        raise ValueError(f"Invalid postal codes found: {invalid_codes}")
    
    # Remove duplicates and sort
    unique_codes = sorted(list(set(postal_codes)))
    n_codes = len(unique_codes)
    
    logger.info(f"Processing {n_codes} unique postal codes: {unique_codes}")
    
    # Initialize distance matrix
    distance_matrix = pd.DataFrame(
        data=0.0,
        index=unique_codes,
        columns=unique_codes,
        dtype=float
    )
    
    # Calculate distances
    for i, code1 in enumerate(unique_codes):
        for j, code2 in enumerate(unique_codes):
            if i != j:
                code1_num = int(code1)
                code2_num = int(code2)
                distance = abs(code1_num - code2_num) * distance_per_unit
                distance_matrix.loc[code1, code2] = distance
    
    logger.info(f"Distance matrix generated successfully")
    return distance_matrix


def format_solution(raw_solution: Dict[str, Any], orders_df: pd.DataFrame, 
                   trucks_df: pd.DataFrame) -> Dict[str, Any]:
    """Format optimization results into structured, readable format"""
    logger.info("Formatting optimization solution...")
    
    # Initialize formatted solution structure
    formatted_solution = {
        'assignments': pd.DataFrame(),
        'routes': pd.DataFrame(),
        'costs': {},
        'summary': {},
        'utilization': {},
        'selected_trucks': []
    }
    
    try:
        # Extract assignment information
        assignments_data = []
        
        if 'assignments' in raw_solution:
            for assignment in raw_solution['assignments']:
                if assignment.get('value', 0) > 0.5:
                    assignments_data.append({
                        'order_id': assignment['order_id'],
                        'truck_id': assignment['truck_id'],
                        'assigned': True
                    })
        
        # Determine selected trucks
        if assignments_data:
            selected_trucks = sorted(list(set([a['truck_id'] for a in assignments_data])))
        else:
            selected_trucks = []
        
        formatted_solution['selected_trucks'] = selected_trucks
        
        # Calculate costs
        total_truck_cost = 0.0
        for truck_id in selected_trucks:
            truck_cost = trucks_df[trucks_df['truck_id'] == truck_id]['cost'].iloc[0]
            total_truck_cost += truck_cost
        
        formatted_solution['costs'] = {
            'total_truck_cost': total_truck_cost,
            'travel_cost': len(selected_trucks) * 50.0,
            'total_cost': total_truck_cost + len(selected_trucks) * 50.0
        }
        
        logger.info("Solution formatting completed successfully")
        
    except Exception as e:
        logger.error(f"Error formatting solution: {str(e)}")
        formatted_solution['summary'] = {'error': str(e)}
    
    return formatted_solution


def validate_postal_codes(postal_codes: List[str]) -> Tuple[bool, List[str]]:
    """Validate postal code format and consistency"""
    postal_pattern = re.compile(r'^\d{5}$')
    invalid_codes = [code for code in postal_codes if not postal_pattern.match(str(code))]
    is_valid = len(invalid_codes) == 0
    return is_valid, invalid_codes


def calculate_route_distance(route_postal_codes: List[str], distance_matrix: pd.DataFrame) -> float:
    """Calculate total distance for a given route"""
    if len(route_postal_codes) <= 1:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(route_postal_codes) - 1):
        from_code = route_postal_codes[i]
        to_code = route_postal_codes[i + 1]
        
        if from_code in distance_matrix.index and to_code in distance_matrix.columns:
            distance = distance_matrix.loc[from_code, to_code]
            total_distance += distance
        else:
            logger.warning(f"Distance not found for route segment {from_code} -> {to_code}")
    
    return total_distance


def format_currency(amount: float, currency: str = "€") -> str:
    """Format monetary values for display"""
    return f"{currency}{amount:,.0f}"


def get_solution_summary(formatted_solution: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the solution"""
    summary_lines = ["=== VEHICLE ROUTER ==="]
    
    try:
        selected_trucks = formatted_solution.get('selected_trucks', [])
        summary_lines.append(f"Selected Trucks: {selected_trucks}")
        
        total_cost = formatted_solution.get('costs', {}).get('total_cost', 0)
        summary_lines.append(f"Total Cost: {format_currency(total_cost)}")
    
    except Exception as e:
        logger.error(f"Error generating solution summary: {str(e)}")
        summary_lines.append(f"Error: {str(e)}")
    
    return "\n".join(summary_lines)