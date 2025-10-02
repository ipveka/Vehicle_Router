# Vehicle Router Usage Guide

## Quick Start

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run with example data
python src/main.py

# Launch web interface
streamlit run app/streamlit_app.py

# Compare all methods
python src/comparison.py
```

## Streamlit Web Application

### Launch
```bash
streamlit run app/streamlit_app.py
```

### Features
- Interactive data loading
- Method selection and parameter configuration
- Real-time optimization
- Interactive visualizations
- Excel report export

## Command Line Interface

### Basic Usage
```bash
python src/main.py
```

### Options
- `--method`: Optimization method (standard, enhanced, genetic)
- `--depot`: Depot location (default: 08020)
- `--real-distances`: Use real-world distances
- `--timeout`: Solver timeout in seconds

## Programmatic Usage

### Basic Example
```python
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.data_generator import DataGenerator

# Generate data
generator = DataGenerator(use_example_data=True)
orders_df = generator.generate_orders()
trucks_df = generator.generate_trucks()
distance_matrix = generator.generate_distance_matrix(['08020', '08027', '08030'])

# Optimize
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
optimizer.build_model()
optimizer.solve(timeout=300)
solution = optimizer.get_solution()
```

## Data Formats

### Orders DataFrame
| Column | Type | Description |
|--------|------|-------------|
| order_id | str | Unique order identifier |
| volume | float | Order volume in m³ |
| postal_code | str | Delivery postal code |

### Trucks DataFrame
| Column | Type | Description |
|--------|------|-------------|
| truck_id | str | Unique truck identifier |
| capacity | float | Truck capacity in m³ |
| cost | float | Truck cost in € |

## Configuration

### Method Parameters
- **Standard MILP + Greedy**: Solver timeout, greedy routes
- **Enhanced MILP**: Cost weight, distance weight, solver timeout
- **Genetic Algorithm**: Population size, max generations, mutation rate

### Distance Methods
- **Simulated**: Mathematical approximation
- **Real-world**: OpenStreetMap geocoding

## Output Analysis

### Solution Structure
```python
{
    'assignments': [...],      # Order-to-truck assignments
    'selected_trucks': [...],  # Selected truck IDs
    'costs': {...},            # Cost breakdown
    'utilization': {...},      # Truck utilization
    'routes_df': pd.DataFrame  # Route details
}
```

### Excel Reports
- Summary sheet with key metrics
- Orders sheet with assignments
- Trucks sheet with utilization
- Routes sheet with detailed paths

## Troubleshooting

### Common Issues
- **Import errors**: Ensure all dependencies are installed
- **Solver timeout**: Increase timeout or reduce problem size
- **Memory issues**: Use smaller population size for genetic algorithm
- **Network errors**: Check internet connection for real-world distances