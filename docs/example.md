# Vehicle Router - Usage Examples

This guide provides simple examples of how to use the Vehicle Router optimizers.

## Quick Start Example

### Using the CLI
```bash
# Run with default settings (Standard MILP + Greedy)
python src/main.py

# Run with specific method
python src/main.py --method enhanced

# Run with real-world distances
python src/main.py --real-distances
```

### Using the Web Interface
```bash
# Launch Streamlit app
streamlit run app/streamlit_app.py
```

## Programmatic Examples

### Example 1: Basic Optimization
```python
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.data_generator import DataGenerator

# Generate example data
generator = DataGenerator(use_example_data=True)
orders_df = generator.generate_orders()
trucks_df = generator.generate_trucks()
distance_matrix = generator.generate_distance_matrix(['08020', '08027', '08030'])

# Create and run optimizer
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
optimizer.build_model()
success = optimizer.solve(timeout=300)

if success:
    solution = optimizer.get_solution()
    print(f"Total cost: €{solution['costs']['total_cost']}")
    print(f"Selected trucks: {solution['selected_trucks']}")
```

### Example 2: Enhanced MILP Optimization
```python
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer

# Create enhanced optimizer
optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix)

# Set cost and distance weights
optimizer.set_weights(cost_weight=0.7, distance_weight=0.3)

# Build and solve
optimizer.build_model()
success = optimizer.solve(timeout=300)

if success:
    solution = optimizer.get_solution()
    print(f"Enhanced solution cost: €{solution['costs']['total_cost']}")
```

### Example 3: Genetic Algorithm Optimization
```python
from vehicle_router.genetic_optimizer import GeneticVrpOptimizer

# Create genetic optimizer
optimizer = GeneticVrpOptimizer(orders_df, trucks_df, distance_matrix)

# Set genetic algorithm parameters
optimizer.set_parameters(
    population_size=100,
    max_generations=50,
    mutation_rate=0.1
)

# Run optimization
solution = optimizer.optimize()

print(f"Genetic solution cost: €{solution['costs']['total_cost']}")
print(f"Generations run: {optimizer.generation}")
```

### Example 4: Custom Data
```python
import pandas as pd
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.utils import calculate_distance_matrix

# Create custom orders
orders_data = {
    'order_id': ['O1', 'O2', 'O3', 'O4'],
    'volume': [2.5, 1.8, 3.2, 2.1],
    'postal_code': ['08020', '08027', '08030', '08035']
}
orders_df = pd.DataFrame(orders_data)

# Create custom trucks
trucks_data = {
    'truck_id': ['T1', 'T2', 'T3'],
    'capacity': [10.0, 8.0, 12.0],
    'cost': [500, 400, 600]
}
trucks_df = pd.DataFrame(trucks_data)

# Calculate distance matrix
postal_codes = orders_df['postal_code'].unique().tolist()
distance_matrix = calculate_distance_matrix(postal_codes)

# Optimize
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
optimizer.build_model()
success = optimizer.solve(timeout=300)

if success:
    solution = optimizer.get_solution()
    print("Optimization successful!")
    print(f"Total cost: €{solution['costs']['total_cost']}")
    print(f"Trucks used: {len(solution['selected_trucks'])}")
```

### Example 5: Using the App Utilities
```python
from app_utils.optimization_runner import OptimizationRunner

# Create optimization runner
runner = OptimizationRunner()

# Run optimization with specific parameters
success = runner.run_optimization(
    orders_df=orders_df,
    trucks_df=trucks_df,
    distance_matrix=distance_matrix,
    optimization_method='enhanced',
    solver_timeout=300,
    cost_weight=0.6,
    distance_weight=0.4
)

if success:
    solution = runner.solution
    print(f"Solution found with cost: €{solution['costs']['total_cost']}")
```

## Configuration Examples

### Method Selection
```python
# Standard MILP + Greedy (default)
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)

# Enhanced MILP
optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix)

# Genetic Algorithm
optimizer = GeneticVrpOptimizer(orders_df, trucks_df, distance_matrix)
```

### Parameter Configuration
```python
# Standard MILP parameters
optimizer = VrpOptimizer(
    orders_df, trucks_df, distance_matrix,
    depot_location='08020',
    depot_return=False,
    max_orders_per_truck=3,
    enable_greedy_routes=True
)

# Enhanced MILP weights
enhanced_optimizer.set_weights(cost_weight=0.7, distance_weight=0.3)

# Genetic algorithm parameters
genetic_optimizer.set_parameters(
    population_size=100,
    max_generations=50,
    mutation_rate=0.1
)
```

## Output Analysis

### Solution Structure
```python
solution = optimizer.get_solution()

# Access key information
total_cost = solution['costs']['total_cost']
selected_trucks = solution['selected_trucks']
assignments = solution['assignments']
utilization = solution['utilization']

# Print summary
print(f"Total cost: €{total_cost}")
print(f"Trucks used: {len(selected_trucks)}")
print(f"Orders delivered: {len(assignments)}")
```

### Route Information
```python
routes_df = solution['routes_df']
for _, route in routes_df.iterrows():
    print(f"Truck {route['truck_id']}: {route['route_sequence']}")
    print(f"Distance: {route['route_distance']} km")
    print(f"Orders: {route['assigned_orders']}")
```

## Error Handling

### Common Error Handling
```python
try:
    optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
    optimizer.build_model()
    success = optimizer.solve(timeout=300)
    
    if not success:
        print("Optimization failed - check constraints")
    else:
        solution = optimizer.get_solution()
        print("Optimization successful!")
        
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Start with Standard MILP + Greedy** for quick results
2. **Use Enhanced MILP** for optimal solutions
3. **Use Genetic Algorithm** for large-scale problems
4. **Use simulated distances** for development and testing
5. **Use real-world distances** for production deployments
6. **Adjust solver timeout** based on problem size
7. **Monitor memory usage** for large problems
