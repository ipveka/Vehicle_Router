"""
Vehicle Router Package

A production-ready Python package for solving Vehicle Routing Problems (VRP)
with order assignment optimization using Mixed Integer Linear Programming (MILP).

This package provides:
- Data generation for orders, trucks, and distance matrices
- MILP-based optimization using PuLP
- Solution validation and constraint checking
- Visualization and plotting capabilities
- Comprehensive logging and error handling

Main Components:
- DataGenerator: Generate and manage input data
- VrpOptimizer: MILP optimization engine
- SolutionValidator: Validate optimization results
- Plotting utilities: Visualize routes and costs
- Utility functions: Helper functions for calculations

Example Usage:
    from vehicle_router import DataGenerator, VrpOptimizer
    
    # Generate data
    data_gen = DataGenerator(use_example_data=True)
    orders_df = data_gen.generate_orders()
    trucks_df = data_gen.generate_trucks()
    
    # Optimize
    optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
    optimizer.solve()
    solution = optimizer.get_solution()
"""

__version__ = "1.0.0"
__author__ = "Vehicle Router Team"
__email__ = "team@vehiclerouter.com"

# Import main classes for easy access
from .data_generator import DataGenerator

# Other imports will be added as modules are implemented
# from .optimizer import VrpOptimizer
# from .validation import SolutionValidator

__all__ = [
    "DataGenerator",
    # "VrpOptimizer", 
    # "SolutionValidator",
    "__version__"
]