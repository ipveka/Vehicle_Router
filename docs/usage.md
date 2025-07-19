# Usage Guide

This comprehensive guide explains how to use the Vehicle Router application, customize it for your specific needs, and integrate it into your workflows.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Command Line Interface](#command-line-interface)
3. [Programmatic Usage](#programmatic-usage)
4. [Customizing Data](#customizing-data)
5. [Configuration Options](#configuration-options)
6. [Output Formats](#output-formats)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Basic Usage

### Quick Start

The simplest way to run the Vehicle Router is with the built-in example data:

```bash
python src/main.py
```

This will:
1. Generate the standard example dataset (5 orders, 5 trucks)
2. Solve the optimization problem
3. Validate the solution
4. Create visualizations and save results in the `output/` directory
5. Display results in the console

### Expected Output

```
=== VEHICLE ROUTER ===
Selected Trucks: [1, 2, 3, 5]
Truck 1 -> Orders ['A', 'E']
Truck 2 -> Orders ['B']
Truck 3 -> Orders ['C']
Truck 5 -> Orders ['D']
Total Cost: €4000
```

## Command Line Interface

The application supports various command-line options for different use cases:

### Basic Options

```bash
# Use random data instead of example data
python src/main.py --random-data

# Set random seed for reproducible results
python src/main.py --random-data --seed 123

# Skip plot generation (faster execution)
python src/main.py --no-plots

# Display plots interactively
python src/main.py --show-plots

# Skip solution validation
python src/main.py --no-validation

# Reduce output verbosity
python src/main.py --quiet
```

### Logging Options

```bash
# Set logging level
python src/main.py --log-level DEBUG
python src/main.py --log-level WARNING
python src/main.py --log-level ERROR
```

### Combined Options

```bash
# Generate random data with specific seed, show plots, and use debug logging
python src/main.py --random-data --seed 42 --show-plots --log-level DEBUG
```

## Programmatic Usage

### Basic Python Integration

```python
from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.validation import SolutionValidator

# Generate data
data_gen = DataGenerator(use_example_data=True)
orders_df = data_gen.generate_orders()
trucks_df = data_gen.generate_trucks()
distance_matrix = data_gen.generate_distance_matrix(orders_df['postal_code'].tolist())

# Optimize
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
optimizer.build_model()
success = optimizer.solve()

if success:
    solution = optimizer.get_solution()
    print(f"Total cost: €{solution['costs']['total_cost']}")
    
    # Validate solution
    validator = SolutionValidator(solution, orders_df, trucks_df)
    validation_report = validator.validate_solution()
    print(f"Solution valid: {validation_report['is_valid']}")
```

### Using the Main Application Class

```python
from src.main import VehicleRouterApp

# Create application with custom configuration
config = {
    'use_example_data': False,
    'random_seed': 123,
    'save_plots': True,
    'show_plots': False,
    'validation_enabled': True,
    'verbose_output': True
}

app = VehicleRouterApp(config)
success = app.run()

if success:
    print("Optimization completed successfully!")
```

## Customizing Data

### Custom Orders

Create your own order dataset:

```python
import pandas as pd

# Define custom orders
orders_data = [
    {'order_id': 'Order_1', 'volume': 45.0, 'postal_code': '12345'},
    {'order_id': 'Order_2', 'volume': 30.0, 'postal_code': '12346'},
    {'order_id': 'Order_3', 'volume': 60.0, 'postal_code': '12347'},
    {'order_id': 'Order_4', 'volume': 25.0, 'postal_code': '12348'},
]

orders_df = pd.DataFrame(orders_data)
```

**Required columns:**
- `order_id`: Unique identifier (string)
- `volume`: Order volume in m³ (float, positive)
- `postal_code`: Delivery location (string, 5 digits)

### Custom Trucks

Define your truck fleet:

```python
# Define custom trucks
trucks_data = [
    {'truck_id': 1, 'capacity': 80.0, 'cost': 1200.0},
    {'truck_id': 2, 'capacity': 60.0, 'cost': 900.0},
    {'truck_id': 3, 'capacity': 40.0, 'cost': 600.0},
]

trucks_df = pd.DataFrame(trucks_data)
```

**Required columns:**
- `truck_id`: Unique identifier (integer)
- `capacity`: Maximum capacity in m³ (float, positive)
- `cost`: Operational cost in euros (float, non-negative)

### Custom Distance Matrix

For custom locations, you can provide your own distance matrix:

```python
import numpy as np

postal_codes = ['12345', '12346', '12347', '12348']
n_locations = len(postal_codes)

# Create symmetric distance matrix
distances = np.zeros((n_locations, n_locations))

# Fill with actual distances (example: random distances)
for i in range(n_locations):
    for j in range(i+1, n_locations):
        distance = np.random.uniform(1.0, 10.0)  # 1-10 km
        distances[i, j] = distance
        distances[j, i] = distance  # Symmetric

distance_matrix = pd.DataFrame(
    distances, 
    index=postal_codes, 
    columns=postal_codes
)
```

### Complete Custom Example

```python
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.validation import SolutionValidator
import pandas as pd

# Custom orders
orders_df = pd.DataFrame([
    {'order_id': 'A', 'volume': 35.0, 'postal_code': '10001'},
    {'order_id': 'B', 'volume': 45.0, 'postal_code': '10002'},
    {'order_id': 'C', 'volume': 25.0, 'postal_code': '10003'},
])

# Custom trucks
trucks_df = pd.DataFrame([
    {'truck_id': 1, 'capacity': 70.0, 'cost': 1000.0},
    {'truck_id': 2, 'capacity': 50.0, 'cost': 800.0},
])

# Simple distance matrix (1km between consecutive postal codes)
postal_codes = orders_df['postal_code'].tolist()
distance_matrix = pd.DataFrame(
    [[abs(int(code1) - int(code2)) for code2 in postal_codes] 
     for code1 in postal_codes],
    index=postal_codes,
    columns=postal_codes
)

# Solve
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
optimizer.build_model()
success = optimizer.solve()

if success:
    solution = optimizer.get_solution()
    print(optimizer.get_solution_summary_text())
```

## Configuration Options

### Application Configuration

The `VehicleRouterApp` class accepts a configuration dictionary:

```python
config = {
    # Data generation
    'use_example_data': True,      # Use built-in example data
    'random_seed': 42,             # Seed for random data generation
    
    # Visualization
    'save_plots': True,            # Save plots to files
    'plot_directory': 'output',    # Directory for saved plots
    'show_plots': False,           # Display plots interactively
    
    # Validation
    'validation_enabled': True,    # Enable solution validation
    
    # Output
    'verbose_output': True,        # Show detailed analysis
}
```

### Logging Configuration

Control logging behavior:

```python
from src.main import setup_logging

# Configure logging
setup_logging(
    log_level="INFO",                    # DEBUG, INFO, WARNING, ERROR
    log_file="logs/custom_run.log"       # Custom log file path
)
```

### Data Generator Configuration

Customize data generation:

```python
from vehicle_router.data_generator import DataGenerator

# Example data
data_gen = DataGenerator(use_example_data=True)

# Random data with seed
data_gen = DataGenerator(use_example_data=False, random_seed=123)
```

## Output Formats

### Console Output

The standard console output format:

```
=== VEHICLE ROUTER ===
Selected Trucks: [1, 3, 5]
Truck 1 -> Orders ['A', 'B']
Truck 3 -> Orders ['C']
Truck 5 -> Orders ['D', 'E']
Total Cost: €3000
```

### Detailed Analysis

When `verbose_output=True`, additional details are shown:

```
Truck Utilization Details:
  Truck 1:
    Capacity: 100.0 m³
    Used: 125.0 m³
    Utilization: 125.0%
    Orders: ['A', 'B']

Cost Breakdown:
  Truck 1: €1500
  Truck 3: €500
  Truck 5: €1000
  Total: €3000

Efficiency Metrics:
  Cost per order: €600
  Cost per m³: €15
  Average utilization: 95.0%
```

### Solution Data Structure

The `get_solution()` method returns a structured dictionary:

```python
solution = {
    'assignments_df': pd.DataFrame,     # Order-to-truck assignments
    'routes_df': pd.DataFrame,          # Route information
    'costs': {                          # Cost breakdown
        'truck_costs': {1: 1500, 3: 500, 5: 1000},
        'total_cost': 3000
    },
    'utilization': {                    # Utilization metrics
        1: {'used_volume': 125.0, 'capacity': 100.0, 'utilization_percent': 125.0},
        # ...
    },
    'selected_trucks': [1, 3, 5],      # Selected truck IDs
    'summary': {                        # High-level summary
        'trucks_used': 3,
        'orders_delivered': 5,
        'total_volume_delivered': 200.0
    }
}
```

### Visualization Outputs

Three types of plots are automatically generated:

1. **Route Visualization** (`output/routes.png`)
   - 2D map showing truck routes and order locations
   - Different colors for each truck
   - Order locations marked with volumes

2. **Cost Analysis** (`output/costs.png`)
   - Bar chart showing cost contribution by truck
   - Total cost displayed
   - Cost-effectiveness metrics

3. **Utilization Analysis** (`output/utilization.png`)
   - Capacity utilization rates for each selected truck
   - Efficiency indicators
   - Unused capacity visualization

## Advanced Features

### Custom Validation Rules

Extend the validation system:

```python
from vehicle_router.validation import SolutionValidator

class CustomValidator(SolutionValidator):
    def check_custom_constraint(self):
        """Add your custom validation logic"""
        # Example: Check maximum orders per truck
        max_orders_per_truck = 3
        violations = []
        
        for truck_id in self.solution['selected_trucks']:
            assigned_orders = [a['order_id'] for a in self.solution['assignments'] 
                             if a['truck_id'] == truck_id]
            if len(assigned_orders) > max_orders_per_truck:
                violations.append(f"Truck {truck_id} has too many orders: {len(assigned_orders)}")
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations
        }
```

### Custom Plotting

Create custom visualizations:

```python
from vehicle_router.plotting import plot_routes
import matplotlib.pyplot as plt

# Generate custom plot
fig, ax = plt.subplots(figsize=(12, 8))
plot_routes(solution['routes_df'], orders_df, trucks_df, distance_matrix, ax=ax)

# Add custom annotations
ax.set_title("Custom Route Analysis")
ax.text(0.02, 0.98, f"Total Cost: €{solution['costs']['total_cost']}", 
        transform=ax.transAxes, verticalalignment='top')

plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
```

### Batch Processing

Process multiple scenarios:

```python
import itertools
from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer

# Define parameter ranges
seeds = [42, 123, 456]
use_example = [True, False]

results = []

for seed, example in itertools.product(seeds, use_example):
    # Generate data
    data_gen = DataGenerator(use_example_data=example, random_seed=seed)
    orders_df = data_gen.generate_orders()
    trucks_df = data_gen.generate_trucks()
    distance_matrix = data_gen.generate_distance_matrix(orders_df['postal_code'].tolist())
    
    # Optimize
    optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
    optimizer.build_model()
    success = optimizer.solve()
    
    if success:
        solution = optimizer.get_solution()
        results.append({
            'seed': seed,
            'example_data': example,
            'total_cost': solution['costs']['total_cost'],
            'trucks_used': len(solution['selected_trucks']),
            'avg_utilization': solution['summary']['average_utilization']
        })

# Analyze results
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.groupby('example_data').agg({
    'total_cost': ['mean', 'std'],
    'trucks_used': ['mean', 'std'],
    'avg_utilization': ['mean', 'std']
}))
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'vehicle_router'
```

**Solution**: Ensure you're running from the project root directory and have installed dependencies:
```bash
cd vehicle-router
pip install -r requirements.txt
python src/main.py
```

#### 2. Infeasible Problems
```
[ERROR] Problem is infeasible - no solution exists
```

**Solution**: Check that total truck capacity exceeds total order volume:
```python
total_volume = orders_df['volume'].sum()
total_capacity = trucks_df['capacity'].sum()
print(f"Volume: {total_volume}, Capacity: {total_capacity}")
```

#### 3. Solver Issues
```
[ERROR] Solver failed with status: Undefined
```

**Solution**: Try installing additional solvers:
```bash
# Install GLPK
pip install glpk

# Or use different solver
import pulp
model.solve(pulp.GLPK_CMD())
```

#### 4. Memory Issues
```
MemoryError: Unable to allocate array
```

**Solution**: Reduce problem size or increase system memory:
```python
# Limit problem size
max_orders = 50
max_trucks = 15
```

### Performance Optimization

#### 1. Reduce Problem Size
```python
# Filter trucks by efficiency
trucks_df = trucks_df.nsmallest(10, 'cost')  # Keep 10 cheapest trucks

# Group similar orders
orders_df = orders_df.groupby('postal_code').agg({
    'volume': 'sum',
    'order_id': lambda x: '_'.join(x)
}).reset_index()
```

#### 2. Solver Configuration
```python
# Set solver timeout
model.solve(pulp.PULP_CBC_CMD(timeLimit=300))  # 5 minutes

# Use parallel processing
model.solve(pulp.PULP_CBC_CMD(threads=4))
```

#### 3. Preprocessing
```python
# Remove dominated trucks (higher cost, lower capacity)
def remove_dominated_trucks(trucks_df):
    trucks_sorted = trucks_df.sort_values(['cost', 'capacity'])
    keep = []
    
    for i, truck in trucks_sorted.iterrows():
        dominated = False
        for j, other in trucks_sorted.iterrows():
            if (other['cost'] <= truck['cost'] and 
                other['capacity'] >= truck['capacity'] and
                (other['cost'] < truck['cost'] or other['capacity'] > truck['capacity'])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    
    return trucks_df.loc[keep]
```

### Debugging Tips

#### 1. Enable Debug Logging
```bash
python src/main.py --log-level DEBUG
```

#### 2. Validate Input Data
```python
# Check data consistency
print("Orders:")
print(orders_df.info())
print(orders_df.describe())

print("Trucks:")
print(trucks_df.info())
print(trucks_df.describe())

print("Distance Matrix:")
print(distance_matrix.shape)
print(distance_matrix.min().min(), distance_matrix.max().max())
```

#### 3. Inspect Optimization Model
```python
# Print model statistics
print(f"Variables: {len(optimizer.model.variables())}")
print(f"Constraints: {len(optimizer.model.constraints)}")

# Print variable values after solving
for var in optimizer.model.variables():
    if var.varValue > 0:
        print(f"{var.name} = {var.varValue}")
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Review `logs/vehicle_router.log` for detailed error messages
2. **Run tests**: Execute `pytest` to verify installation
3. **Minimal example**: Create a minimal reproduction case
4. **GitHub Issues**: Report bugs with complete error messages and system information
5. **Documentation**: Review the API documentation in the source code docstrings

---

*This usage guide covers the most common scenarios and customization options. For advanced use cases or integration questions, please refer to the source code documentation or contact the development team.*