#!/usr/bin/env python3
"""
Test script for the Enhanced VRP Optimizer with distance minimization

This script tests the enhanced model to ensure it works correctly with
the example data and produces valid solutions.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vehicle_router.data_generator import DataGenerator
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer


def test_enhanced_optimizer():
    """Test the enhanced optimizer with example data"""
    print("=== Testing Enhanced VRP Optimizer ===")
    
    # Generate example data
    print("\n1. Generating example data...")
    data_gen = DataGenerator(use_example_data=True)
    orders_df = data_gen.generate_orders()
    trucks_df = data_gen.generate_trucks()
    
    # Get unique postal codes and generate distance matrix
    postal_codes = orders_df['postal_code'].unique().tolist()
    distance_matrix = data_gen.generate_distance_matrix(postal_codes)
    
    print(f"   Orders: {len(orders_df)}")
    print(f"   Trucks: {len(trucks_df)}")
    print(f"   Locations: {len(postal_codes)}")
    
    # Initialize enhanced optimizer
    print("\n2. Initializing enhanced optimizer...")
    optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix)
    
    # Set objective weights
    optimizer.set_objective_weights(cost_weight=0.6, distance_weight=0.4)
    
    # Build model
    print("\n3. Building enhanced MILP model...")
    optimizer.build_model()
    
    # Solve optimization (with shorter timeout for testing)
    print("\n4. Solving optimization...")
    success = optimizer.solve(timeout=60)
    
    if success:
        print("✅ Optimization successful!")
        
        # Get solution
        solution = optimizer.get_solution()
        
        # Display results
        print(f"\n=== ENHANCED SOLUTION RESULTS ===")
        print(f"Selected Trucks: {solution['selected_trucks']}")
        print(f"Total Truck Cost: €{solution['costs']['total_truck_cost']:.0f}")
        print(f"Total Distance: {solution['costs']['total_distance']:.1f} km")
        print(f"Orders Delivered: {len(solution['assignments_df'])}")
        
        # Show truck assignments and routes
        for _, route in solution['routes_df'].iterrows():
            truck_id = route['truck_id']
            assigned_orders = route['assigned_orders']
            route_distance = route['route_distance']
            route_sequence = route['route_sequence']
            
            print(f"\nTruck {truck_id}:")
            print(f"  Orders: {assigned_orders}")
            print(f"  Route: {' → '.join(route_sequence)}")
            print(f"  Distance: {route_distance:.1f} km")
        
        # Show utilization
        print(f"\n=== UTILIZATION ANALYSIS ===")
        for truck_id, util in solution['utilization'].items():
            print(f"Truck {truck_id}: {util['used_volume']:.1f}/{util['capacity']:.1f} m³ ({util['utilization_percent']:.1f}%)")
        
        return True
    else:
        print("❌ Optimization failed!")
        return False


if __name__ == "__main__":
    try:
        success = test_enhanced_optimizer()
        if success:
            print("\n🎉 Enhanced optimizer test completed successfully!")
        else:
            print("\n💥 Enhanced optimizer test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test error: {str(e)}")
        sys.exit(1)