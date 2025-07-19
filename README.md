# Vehicle Router

A production-ready Python application for solving Vehicle Routing Problems (VRP) with order assignment optimization using Mixed Integer Linear Programming (MILP). Includes both a command-line interface and an interactive Streamlit web application.

## Problem Description

The Vehicle Router solves the classic Vehicle Routing Problem with capacity constraints, where the goal is to optimally assign orders to trucks and determine delivery routes while minimizing total operational costs. The system considers:

- **Order Requirements**: Each order has a specific volume and delivery location (postal code)
- **Truck Constraints**: Each truck has a maximum capacity and associated operational cost
- **Optimization Goal**: Minimize total cost while ensuring all orders are delivered within capacity limits

### Example Problem

The system includes a built-in example with:
- **5 Orders**: A (25 m³), B (50 m³), C (25 m³), D (25 m³), E (25 m³) - Total: 150 m³
- **5 Trucks**: Capacities of 100, 50, 25, 25, 25 m³ with costs €1500, €1000, €500, €1500, €1000
- **Locations**: Postal codes 08027-08031 with 1km spacing between consecutive codes

## Repository Structure

```
vehicle_router/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation configuration
├── LICENSE                      # MIT License
│
├── docs/                        # Comprehensive documentation
│   ├── index.md                # Project overview and getting started
│   ├── model_description.md    # MILP formulation details
│   └── usage.md                # Usage instructions and examples
│
├── src/                         # Main application entry point
│   ├── __init__.py
│   └── main.py                 # Main workflow orchestration
│
├── app/                         # Streamlit web application
│   └── streamlit_app.py        # Interactive web interface
│
├── app_utils/                   # Streamlit application utilities
│   ├── __init__.py
│   ├── data_handler.py         # Data loading and management
│   ├── optimization_runner.py  # Optimization execution
│   ├── visualization_manager.py # Interactive visualizations
│   ├── export_manager.py       # Data export functionality
│   └── ui_components.py        # Reusable UI components
│
├── vehicle_router/              # Core optimization package
│   ├── __init__.py
│   ├── data_generator.py       # Data generation and management
│   ├── optimizer.py            # MILP optimization engine
│   ├── plotting.py             # Visualization and plotting
│   ├── validation.py           # Solution validation
│   └── utils.py                # Utility functions
│
├── tests/                       # Comprehensive test suite
│   ├── test_data_generator.py
│   ├── test_optimizer.py
│   ├── test_utils.py
│   └── test_validation.py
│
├── output/                      # Generated outputs (CSV files and plots)
│   ├── orders.csv              # Input orders data
│   ├── trucks.csv              # Input trucks data
│   ├── distance_matrix.csv     # Distance matrix
│   ├── solution_assignments.csv # Order-to-truck assignments
│   ├── solution_routes.csv     # Route information
│   ├── truck_utilization.csv  # Capacity utilization data
│   ├── cost_breakdown.csv      # Cost analysis data
│   ├── solution_summary.csv    # Summary statistics
│   ├── routes.png              # Route visualization
│   ├── costs.png               # Cost analysis chart
│   └── utilization.png         # Capacity utilization chart
│
└── logs/                        # Application logs
    └── vehicle_router.log
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ipveka/vehicle-router.git
   cd vehicle-router
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

### Dependencies

The application requires the following Python packages:
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization
- `PuLP>=2.7.0` - Linear programming optimization
- `pytest>=7.0.0` - Testing framework
- `streamlit>=1.28.0` - Web application framework
- `plotly>=5.15.0` - Interactive visualizations
- `openpyxl>=3.1.0` - Excel file support
- `scipy>=1.9.0` - Scientific computing utilities

## Usage

### Streamlit Web Application

Launch the interactive web application:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit application provides:
- **Interactive Data Loading**: Upload custom CSV files or use example data
- **Real-time Optimization**: Run optimization with configurable parameters
- **Interactive Visualizations**: Explore results with Plotly charts
- **Data Export**: Download results in Excel, CSV, or text formats
- **Comprehensive Analysis**: Detailed solution analysis and methodology explanation

#### Streamlit App Features

1. **Introduction Section**: Overview of the application and optimization objectives
2. **Data Exploration**: Interactive tables and visualizations of input data
3. **Solution Analysis**: Detailed results with export capabilities
4. **Visualization**: Interactive charts for routes, costs, and utilization
5. **Methodology**: In-depth explanation of the MILP formulation and algorithms

### Command Line Interface

Run the optimization with the built-in example data:

```bash
python src/main.py
```

### Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  --random-data        Use random data instead of example data
  --seed INTEGER       Random seed for data generation (default: 42)
  --no-plots          Skip plot generation
  --show-plots        Display plots interactively
  --no-validation     Skip solution validation
  --quiet             Reduce output verbosity
  --log-level LEVEL   Set logging level (DEBUG, INFO, WARNING, ERROR)
```

### Example Output

```
=== VEHICLE ROUTER ===
Selected Trucks: [1, 2, 3, 5]
Truck 1 -> Orders ['A', 'E']
Truck 2 -> Orders ['B']
Truck 3 -> Orders ['C']
Truck 5 -> Orders ['D']
Total Cost: €4000
```

### Programmatic Usage

```python
from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer

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
```

## Features

### Core Functionality
- **MILP Optimization**: Uses PuLP library for robust mathematical optimization
- **Data Generation**: Built-in example data plus random data generation for testing
- **Solution Validation**: Comprehensive constraint checking and feasibility verification
- **Visualization**: Automatic generation of route maps, cost analysis, and utilization charts
- **Comprehensive Logging**: Detailed progress tracking and debugging information

### Advanced Features
- **Flexible Data Input**: Support for custom orders, trucks, and distance matrices
- **Extensible Architecture**: Modular design for easy customization and extension
- **Production Ready**: Comprehensive error handling, logging, and validation
- **Testing Suite**: Full unit test coverage with pytest
- **Documentation**: Extensive documentation with examples and API reference

## Visualization

The application automatically generates three types of visualizations saved to the `output/` directory:

### 1. Route Visualization (`output/routes.png`)
- 2D map showing truck routes and order locations
- Different colors for each selected truck
- Order locations marked with postal codes and volumes
- Visual representation of the optimal delivery routes

### 2. Cost Analysis (`output/costs.png`)
- Bar chart showing cost contribution by each selected truck
- Total cost displayed prominently
- Cost-effectiveness comparison between trucks
- Helps identify the most economical truck selections

### 3. Utilization Analysis (`output/utilization.png`)
- Capacity utilization rates for each selected truck
- Shows used vs. available capacity
- Efficiency indicators for fleet optimization
- Identifies underutilized or fully utilized trucks

### Sample Output Visualization
When you run the application with the example data, you'll see output like:

```
=== VEHICLE ROUTER ===
Selected Trucks: [1, 2, 3, 5]
Truck 1 -> Orders ['A', 'E']
Truck 2 -> Orders ['B']
Truck 3 -> Orders ['C']
Truck 5 -> Orders ['D']
Total Cost: €4000

Efficiency Metrics:
  Cost per order: €800
  Cost per m³: €20
  Average utilization: 100.0%
```

The visualizations provide immediate insights into:
- Which trucks were selected and why
- How efficiently capacity is being used
- The cost structure of the optimal solution
- Geographic distribution of deliveries

## Algorithm Details

The optimization uses a Mixed Integer Linear Programming (MILP) formulation:

- **Decision Variables**: Binary variables for order-to-truck assignments and truck usage
- **Objective Function**: Minimize total operational costs (truck selection costs)
- **Constraints**: 
  - Each order assigned to exactly one truck
  - Truck capacity limits respected
  - Truck usage properly linked to assignments

For detailed mathematical formulation, see [docs/model_description.md](docs/model_description.md).

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vehicle_router

# Run specific test file
pytest tests/test_optimizer.py -v
```

## Performance

The application is designed for efficiency:
- **Small Problems** (5-10 orders, 3-5 trucks): < 1 second
- **Medium Problems** (20-50 orders, 10-15 trucks): 1-10 seconds
- **Large Problems** (100+ orders, 20+ trucks): May require several minutes

Performance depends on problem complexity and available system resources.

## Customization

### Custom Data

Replace the example data by modifying the `DataGenerator` class or providing your own DataFrames:

```python
# Custom orders
orders_df = pd.DataFrame({
    'order_id': ['X', 'Y', 'Z'],
    'volume': [30.0, 45.0, 20.0],
    'postal_code': ['12345', '12346', '12347']
})

# Custom trucks
trucks_df = pd.DataFrame({
    'truck_id': [1, 2],
    'capacity': [80.0, 60.0],
    'cost': [1200.0, 900.0]
})
```

### Configuration

Modify application behavior through configuration parameters:

```python
config = {
    'use_example_data': False,
    'random_seed': 123,
    'save_plots': True,
    'show_plots': True,
    'validation_enabled': True,
    'verbose_output': True
}

app = VehicleRouterApp(config)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/ipveka/vehicle-router/issues)
- **Documentation**: [docs/](docs/)
- **Email**: team@vehiclerouter.com

## Acknowledgments

- Built with [PuLP](https://github.com/coin-or/pulp) for linear programming optimization
- Uses [pandas](https://pandas.pydata.org/) for data manipulation
- Visualization powered by [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)