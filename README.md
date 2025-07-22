# Vehicle Router

A Python application for solving Vehicle Routing Problems (VRP) with **multiple optimization approaches**. This project implements multi-objective optimization using Mixed Integer Linear Programming (MILP) and Genetic Algorithms. Includes both command-line interface and interactive Streamlit web application with comparison tools.

**📱 Streamlit App**: Features a simplified interface with **two main optimization methods** (Standard MILP + Greedy and Genetic Algorithm) for an intuitive user experience.

## Problem Description

The Vehicle Router solves the Capacitated Vehicle Routing Problem (CVRP) with **three distinct optimization methodologies**, each designed for different use cases and computational requirements:

### 📊 **Method 1: Standard MILP + Greedy** ⭐ (Default)
**Best for: Balanced performance and solution quality**

- **Approach**: Hybrid MILP-Greedy optimization for cost-efficient routes
- **Objective**: Minimize truck costs → optimize route sequences
- **Algorithm**: Phase 1: MILP cost optimization → Phase 2: Greedy route optimization  
- **Performance**: < 5 seconds solve time, excellent scalability
- **Features**:
  - **Cost-optimal truck selection** via MILP solver
  - **Route-optimal sequences** via exhaustive permutation testing
  - **Detailed logging** with performance metrics and improvements
  - **Efficient scaling** handles 8! = 40,320 permutations per truck
  - **Configurable depot return** and location settings
- **Use Case**: Situations needing fast, reliable optimization
- **Best For**: Most real-world logistics applications (recommended starting point)

### 🚀 **Method 2: Enhanced MILP**
**Best for: Highest solution quality with moderate complexity**

- **Approach**: Enhanced MILP with integrated cost-distance optimization
- **Objective**: Multi-objective weighted optimization (α×cost + β×distance)
- **Algorithm**: Single-phase MILP with routing variables and flow conservation
- **Performance**: 5-60 seconds solve time, globally optimal solutions
- **Features**: 
  - **True multi-objective optimization** with configurable weights
  - **Integrated routing variables** z_{k,l,j} for truck movements
  - **Flow conservation constraints** ensuring valid route continuity
  - **Subtour elimination** preventing disconnected routes
  - **Route reconstruction** from MILP solution
- **Use Case**: Cases where you want optimal multi-objective solutions
- **Best For**: Small-medium instances (≤50 orders), quality-critical scenarios

### 🧬 **Method 3: Genetic Algorithm** 
**Best for: Large-scale and complex multi-objective problems**

- **Approach**: Evolutionary optimization with population-based search
- **Objective**: Multi-objective fitness with configurable cost-distance weights
- **Algorithm**: Population evolution with selection, crossover, and mutation operators
- **Performance**: Configurable 30s-10min solve time, excellent scalability
- **Features**:
  - **Population-based search** with 50-200 individuals
  - **Genetic operators** (tournament selection, order crossover, adaptive mutation)
  - **Constraint repair mechanisms** for capacity violations
  - **Route optimization** via nearest neighbor heuristics
  - **Convergence detection** with diversity preservation
  - **Configurable parameters** (population size, generations, mutation rate)
- **Use Case**: Large-scale problems where MILP approaches struggle
- **Best For**: 50+ orders, complex constraints, research applications

## 🔧 **Enhanced Comparison Tools**

The system includes powerful comparison capabilities for evaluating all three optimization methods:

- **`src/comparison.py`**: Comparison script that runs all three methods and provides detailed comparative analysis
- **Detailed route information**: Shows exact truck routes, distances, and sequences for each method
- **Performance metrics**: Execution time, solution quality, and efficiency comparisons
- **CSV export capabilities**: Save comparison results for further analysis
- **Detailed reporting**: Generate performance analysis with method rankings

## 📊 **Method Comparison**

| Aspect | Standard MILP + Greedy | Enhanced MILP | Genetic Algorithm |
|--------|----------------------|---------------|-------------------|
| **Primary Focus** | Cost + Route efficiency | Integrated optimization | Evolutionary multi-objective |
| **Algorithm Type** | Hybrid MILP-Heuristic | Enhanced MILP | Metaheuristic |
| **Route Quality** | Optimized (permutations) | Optimal (integrated) | Near-optimal (evolved) |
| **Solve Time** | Fast (< 5s) | Moderate (< 60s) | Configurable (30s-10m) |
| **Memory Usage** | Low (< 100MB) | Moderate (< 500MB) | Variable (100MB-2GB) |
| **Scalability** | Excellent | Good | Excellent |
| **Solution Guarantee** | Cost-optimal + Route-heuristic | Multi-objective optimal | Near-optimal |
| **Global Search** | Limited | No | Yes |
| **Parameter Tuning** | Minimal | Minimal | Extensive |

## 🔧 **Key Features**

### Core Problem Elements
- **Order Requirements**: Each order has volume and delivery location (postal code)
- **Truck Constraints**: Each truck has capacity limits and operational costs
- **Distance Matrix**: Travel distances between all postal code locations
- **Capacity Limits**: Ensuring orders fit within truck capacity constraints

### Key Features
- **Depot Configuration**: Configurable depot location for truck start/end points
- **Multi-Objective Optimization**: Balanced cost and distance optimization
- **Route Optimization**: Actual route sequences with distance calculations
- **Interactive Visualization**: Detailed route plots with depot-to-customer paths
- **Detailed Logging**: Performance metrics and optimization progress
- **Flexible Parameters**: Configurable weights, timeouts, and algorithm settings

### Example Problem

The system includes a built-in example with:
- **5 Orders**: A (25 m³), B (50 m³), C (25 m³), D (25 m³), E (25 m³) - Total: 150 m³
- **5 Trucks**: Capacities of 100, 50, 25, 25, 25 m³ with costs €1500, €1000, €500, €1500, €1000
- **Locations**: Postal codes 08027-08031 with 1km spacing between consecutive codes

## Repository Structure

```
vehicle_router/
│
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
│
├── docs/
│   ├── methods.md
│   └── usage.md
│
├── src/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── app_utils/
│   ├── __init__.py
│   ├── data_handler.py
│   ├── optimization_runner.py
│   ├── visualization_manager.py
│   ├── export_manager.py
│   └── ui_components.py
│
├── vehicle_router/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── optimizer.py
│   ├── enhanced_optimizer.py
│   ├── plotting.py
│   ├── validation.py
│   └── utils.py
│
├── tests/
│   ├── test_data_generator.py
│   ├── test_optimizer.py
│   ├── test_utils.py
│   └── test_validation.py
│
├── output/
│   ├── orders.csv
│   ├── trucks.csv
│   ├── distance_matrix.csv
│   ├── solution_assignments.csv
│   ├── solution_routes.csv
│   ├── truck_utilization.csv
│   ├── cost_breakdown.csv
│   ├── solution_summary.csv
│   ├── routes.png
│   ├── costs.png
│   └── utilization.png
│
└── logs/
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

#### Quick Start (Recommended)

Use the automated run script that installs requirements and launches the app:

```bash
# Cross-platform Python script
python run_app.py

# Or use platform-specific scripts:
./run_app.sh        # Unix/Linux/macOS
run_app.bat         # Windows
```

#### Manual Launch

Alternatively, launch the interactive web application manually:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit application provides:
- **Three Optimization Models**: Standard MILP + Greedy, Enhanced MILP, and Genetic Algorithm
- **Multi-Objective Optimization**: Configure weights for cost vs. distance optimization
- **Interactive Data Loading**: Upload custom CSV files or use example data
- **Real-time Optimization**: Run optimization with configurable parameters and timeouts
- **Method-Specific Documentation**: Detailed technical documentation for selected optimization method
- **Route Visualization**: View actual route sequences with distances and depot locations
- **Interactive Visualizations**: Explore results with Plotly charts including route maps and analysis
- **Data Export**: Download results in Excel, CSV, or text formats
- **Detailed Analysis**: Solution analysis with performance metrics

#### Streamlit App Features

1. **Introduction Section**: Overview of the application and three optimization approaches
2. **Data Exploration**: Interactive tables and visualizations of input data
3. **Configuration Options**: 
   - **Model Selection**: Choose between available optimization methods (Standard MILP + Greedy and Genetic Algorithm by default)
   - **Method-Specific Parameters**: Algorithm-specific settings (GA population, generations, mutation rate, Enhanced MILP weights if enabled)
   - **Depot Configuration**: Customizable depot location and return requirements
   - **Optimization Parameters**: Solver timeout, validation options, objective weights
4. **Solution Analysis**: Detailed results with route sequences, distances, and export capabilities
5. **Visualization**: Interactive route maps with optimized sequences and performance metrics
6. **Method Documentation**: In-depth technical explanation of the selected optimization method

### 🎛️ App Configuration

The Streamlit application can be configured to show different optimization methods based on your needs:

#### **Default Configuration (Simplified Interface)**
- **📊 Standard MILP + Greedy**: Fast, balanced optimization (enabled by default)
- **🧬 Genetic Algorithm**: Evolutionary approach with fixed 50/50 cost-distance weighting (enabled by default)
- **🚀 Enhanced MILP**: Hidden by default for simplified user experience

#### **Customizing Available Models**
To enable/disable models, edit the `AVAILABLE_MODELS` configuration in `app/streamlit_app.py`:

```python
AVAILABLE_MODELS = {
    'standard': {'enabled': True},   # Standard MILP + Greedy
    'enhanced': {'enabled': False},  # Enhanced MILP (hidden by default)
    'genetic': {'enabled': True}     # Genetic Algorithm
}
```

#### **Key App Simplifications**
- **Genetic Algorithm**: Uses fixed 50/50 cost-distance weights (no user configuration needed)
- **Model Selection**: Only shows enabled models in the sidebar
- **Default Method**: Automatically selects first enabled model (Standard MILP + Greedy)

### Command Line Interface

#### Single Method Optimization

Run optimization using any of the three methods:

```bash
# Standard MILP + Greedy (default)
python src/main.py --optimizer standard

# Enhanced MILP
python src/main.py --optimizer enhanced

# Genetic Algorithm
python src/main.py --optimizer genetic
```

#### Method Comparison

Run all three methods and compare results:

```bash
# Basic comparison
python src/comparison.py

# Comparison with custom parameters
python src/comparison.py --timeout 60 --ga-population 100 --ga-generations 50

# With depot return and result saving
python src/comparison.py --depot-return --save-results

# Different objective weights
python src/comparison.py --cost-weight 0.7 --distance-weight 0.3
```

### Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  --optimizer {standard,enhanced,genetic}  Optimizer type (default: standard)
                                          standard: MILP cost optimization + greedy route optimization
                                          enhanced: Enhanced MILP with integrated cost-distance optimization
                                          genetic: Evolutionary algorithm for multi-objective optimization
  --depot POSTAL_CODE                     Depot postal code (default: 08020)
  --quiet                                 Reduce output verbosity

Examples:
  python src/main.py                          # Standard MILP + Greedy (recommended)
  python src/main.py --optimizer enhanced     # Enhanced MILP with multi-objective optimization
  python src/main.py --optimizer genetic      # Genetic Algorithm for large-scale problems
  python src/main.py --depot 08031           # Standard model with custom depot location
  python src/main.py --quiet                 # Minimal output
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

#### Standard MILP + Greedy (Recommended)
```python
from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer

# Generate data
data_gen = DataGenerator(use_example_data=True)
orders_df = data_gen.generate_orders()
trucks_df = data_gen.generate_trucks()
distance_matrix = data_gen.generate_distance_matrix(orders_df['postal_code'].tolist())

# Optimize with hybrid MILP-Greedy approach
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix, 
                        enable_greedy_routes=True, depot_return=False)
optimizer.build_model()
success = optimizer.solve()
if success:
    solution = optimizer.get_solution()
```

#### Enhanced MILP (Multi-Objective)
```python
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer

# Multi-objective optimization
enhanced_optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix,
                                         depot_location='08020', depot_return=True)
enhanced_optimizer.set_objective_weights(cost_weight=0.6, distance_weight=0.4)
enhanced_optimizer.build_model()
success = enhanced_optimizer.solve(timeout=120)
```

#### Genetic Algorithm (Large-Scale)
```python
from vehicle_router.genetic_optimizer import GeneticVrpOptimizer

# Evolutionary optimization
genetic_optimizer = GeneticVrpOptimizer(orders_df, trucks_df, distance_matrix,
                                       depot_location='08020', depot_return=True)
genetic_optimizer.set_parameters(population_size=50, max_generations=100, mutation_rate=0.1)
genetic_optimizer.set_objective_weights(cost_weight=0.6, distance_weight=0.4)
success = genetic_optimizer.solve(timeout=300)

if success:
    solution = optimizer.get_solution()
    print(f"Total cost: €{solution['costs']['total_cost']}")
```

## Features

### Core Functionality
- **MILP Optimization**: Uses PuLP library for robust mathematical optimization
- **Data Generation**: Built-in example data plus random data generation for testing
- **Solution Validation**: Constraint checking and feasibility verification
- **Visualization**: Automatic generation of route maps, cost analysis, and utilization charts
- **Detailed Logging**: Progress tracking and debugging information

### Additional Features
- **Flexible Data Input**: Support for custom orders, trucks, and distance matrices
- **Extensible Architecture**: Modular design for easy customization and extension
- **Ready to Use**: Error handling, logging, and validation
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

## Documentation

For complete documentation, see:

- **[Optimization Methods](docs/methods.md)** - Complete guide to all three optimization methods
- **[Usage Guide](docs/usage.md)** - Usage examples, API reference, and configuration options

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