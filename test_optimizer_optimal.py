#!/usr/bin/env python3
"""
Test script to verify optimizer finds optimal solution
"""

import sys
import os
sys.path.append('.')

from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer

def test_optimal_solution():
    """Test that optimizer finds the expected optimal solution"""
    print("Testing optimal solution finding...")
    
    # Generate example data
    data_gen = DataGenerator(use_example_data=True)
    orders_df = data_gen.generate_orders()
    trucks_df = data_gen.generate_trucks()
    
    # Get postal codes from orders
    postal_codes = orders_df['postal_code'].tolist()
    distance_matrix = data_gen.generate_distance_matrix(postal_codes)
    
    print("Expected optimal solution analysis:")
    print("Order A: 75 m³ - needs truck with capacity >= 75")
    print("Order B: 50 m³ - needs truck with capacity >= 50") 
    print("Orders C,D,E: 25 m³ each - can fit in trucks with capacity >= 25")
    print()
    print("Available trucks:")
    print("Truck 1: 100 m³, €1500 - can take A (75) + E (25) = 100 m³")
    print("Truck 2: 50 m³, €1000 - can take B (50)")
    print("Truck 3: 25 m³, €500 - can take C (25)")
    print("Truck 4: 25 m³, €1500 - can take D (25)")
    print("Truck 5: 25 m³, €1000 - can take D (25)")
    print()
    print("Optimal should be: Truck 1 (A+E), Truck 2 (B), Truck 3 (C), Truck 5 (D)")
    print("Cost: €1500 + €1000 + €500 + €1000 = €4000")
    print("Alternative: Truck 1 (A+E), Truck 2 (B), Truck 3 (C), Truck 4 (D)")
    print("Cost: €1500 + €1000 + €500 + €1500 = €4500")
    print()
    
    # Create and run optimizer
    optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
    optimizer.build_model()
    success = optimizer.solve()
    
    if success:
        solution = optimizer.get_solution()
        print("ACTUAL SOLUTION:")
        print(optimizer.get_solution_summary_text())
        
        # Verify optimality
        expected_cost = 4000  # Minimum possible cost
        actual_cost = solution['costs']['total_cost']
        
        print(f"\nOptimality check:")
        print(f"Expected minimum cost: €{expected_cost}")
        print(f"Actual cost: €{actual_cost}")
        
        if actual_cost == expected_cost:
            print("✓ Optimal solution found!")
        else:
            print("✗ Solution may not be optimal")
            
        return actual_cost == expected_cost
    else:
        print("Optimization failed!")
        return False

if __name__ == "__main__":
    test_optimal_solution()