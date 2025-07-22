# Vehicle Router Optimization System

A comprehensive Python application for solving Vehicle Routing Problems (VRP) using multiple advanced optimization approaches. The system includes both command-line tools and an interactive Streamlit web application for route optimization with real-world geographic distances.

## 🚀 Key Features

- **Multiple Optimization Methods**: Standard MILP + Greedy, Enhanced MILP, and Genetic Algorithm
- **Real-World Distance Integration**: OpenStreetMap geocoding with Haversine calculations for accurate geographic routing
- **Interactive Web Interface**: Streamlit app with real-time distance matrix updates and visual analysis
- **Command-Line Tools**: CLI scripts for automation, batch processing, and method comparison
- **Comprehensive Analysis**: Solution validation, performance metrics, and multi-method comparison
- **Flexible Configuration**: Customizable models, parameters, and distance calculation methods
- **Advanced Visualizations**: Route maps, cost breakdowns, capacity utilization, and distance heatmaps

## 📊 Optimization Methods

### 1. **Standard MILP + Greedy** *(Default)*
- **Approach**: Two-phase hybrid optimization combining cost-optimal truck selection with route sequence optimization
- **Phase 1**: Mixed Integer Linear Programming for minimum-cost truck selection
- **Phase 2**: Exhaustive permutation testing to find optimal route sequences (tests all route combinations)
- **Best for**: Balanced cost-distance optimization, daily operations, quick results
- **Performance**: Fast execution (< 5s), cost-optimal truck selection with distance-optimized routes
- **Scalability**: Excellent for typical distributions (≤8 orders per truck, handles 40,320 permutations efficiently)

### 2. **Enhanced MILP** *(Advanced)*
- **Approach**: Simultaneous multi-objective optimization balancing cost and distance in a single mathematical model
- **Method**: Extended MILP formulation with routing variables and flow conservation constraints
- **Optimization**: Weighted objective function combining normalized truck costs and travel distances
- **Best for**: High-quality routes requiring optimal cost-distance balance
- **Performance**: Medium execution time (1-30s), globally optimal solutions for multi-objective function
- **Scalability**: Good for small-medium problems (≤50 orders, ≤15 trucks)

### 3. **Genetic Algorithm** *(Evolutionary)*
- **Approach**: Population-based evolutionary optimization using genetic operators for multi-objective VRP
- **Method**: Tournament selection, Order Crossover (OX), adaptive mutation with constraint repair
- **Optimization**: Fixed 50/50 cost-distance weighting for balanced multi-objective optimization
- **Best for**: Large problems, solution diversity exploration, complex routing scenarios
- **Performance**: Fast convergence (often < 20 generations), generally good solutions with variety
- **Scalability**: Excellent for large problems (100+ orders, 20+ trucks)

## 🌍 Distance Calculation Methods

### **Simulated Distances** *(Default)*
- **Method**: Mathematical calculation based on postal code unit differences (1km per unit)
- **Advantages**: Instant calculation, no network dependencies, consistent results
- **Use Cases**: Quick testing, development environments, offline scenarios
- **Performance**: Immediate results, no API rate limiting

### **Real-World Distances** *(Advanced)*
- **Method**: OpenStreetMap Nominatim geocoding + Haversine great-circle distance calculation
- **Process**: 
  1. **Geocoding**: Postal codes → latitude/longitude coordinates via OpenStreetMap API
  2. **Distance Calculation**: Haversine formula for accurate geographic distances
  3. **Fallback Mechanism**: Static calculation if geocoding fails
- **Advantages**: Accurate geographic distances, realistic route planning based on actual coordinates
- **Considerations**: Requires internet connection, ~0.5s per postal code (respectful rate limiting)
- **Use Cases**: Production routing, accurate distance estimation, geographic analysis
- **Coverage**: Global coverage through OpenStreetMap, works with any country code

### **Distance Accuracy Comparison**
```
Example (Barcelona postal codes):
Route               Simulated    Real-World    Improvement
08020 → 08027       7.0 km       1.3 km        -81% (more accurate)
08020 → 08028       8.0 km       7.7 km        -4% (slight improvement)  
08027 → 08028       1.0 km       6.8 km        +580% (captures actual geography)

Real distances reflect actual urban geography, road networks, and geographic barriers.
```

## 🎛️ Application Configuration

### **Streamlit App Model Selection**
The web application supports configurable optimization method availability:

**Default Configuration:**
- **Standard MILP + Greedy**: ✅ Enabled (fast, balanced optimization)
- **Enhanced MILP**: ❌ Hidden (advanced users only)
- **Genetic Algorithm**: ✅ Enabled (evolutionary approach)

**Customization:**
Modify `AVAILABLE_MODELS` in `app/streamlit_app.py` or use `.streamlit/config.toml`:

```toml
[app.models.standard]
name = "📊 Standard MILP + Greedy"
enabled = true

[app.models.enhanced] 
name = "🚀 Enhanced MILP"
enabled = false

[app.models.genetic]
name = "🧬 Genetic Algorithm" 
enabled = true
```

### **Distance Calculation Configuration**
- **Web App**: Toggle "🌍 Use Real-World Distances" checkbox in sidebar
- **CLI**: Use `--real-distances` flag for main.py and comparison.py
- **Programmatic**: Set `use_real_distances=True` in data generation

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+ with pip package manager
- Internet connection (for real-world distances feature)
- ~500MB disk space for dependencies

### **Installation Steps**
```bash
# 1. Clone repository
git clone <repository-url>
cd Vehicle_Router

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Install package in development mode
pip install -e .

# 4. Verify installation
python src/main.py --help
```

### **Required Dependencies**
```
Core Libraries:
- pandas >= 1.3.0          # Data manipulation and analysis
- numpy >= 1.21.0          # Numerical computing
- pulp >= 2.6.0           # Linear programming optimization
- streamlit >= 1.25.0     # Interactive web applications
- plotly >= 5.0.0         # Interactive visualizations
- requests >= 2.25.0      # HTTP library for OpenStreetMap API
- openpyxl >= 3.0.0       # Excel file support

Optimization:
- CBC Solver              # Open-source MILP solver (auto-installed with PuLP)
```

## 🚀 Usage Guide

### **Streamlit Web Application**
```bash
# Launch interactive web interface
streamlit run app/streamlit_app.py

# Or use run script
python run_app.py
```

**Web App Workflow:**
1. **Load Data**: Click "Load Example Data" or upload custom CSV files
2. **Configure Distance Method**: Toggle "🌍 Use Real-World Distances" (auto-updates distance matrix)
3. **Select Optimization Method**: Choose from available models via buttons
4. **Set Parameters**: Adjust method-specific parameters (GA population, generations, etc.)
5. **Run Optimization**: Click "🚀 Run Optimization" and monitor progress
6. **Analyze Results**: Review solution analysis, visualizations, and documentation
7. **Export Data**: Download Excel reports and solution summaries

**Key Features:**
- **Real-time Distance Matrix Updates**: Automatically reloads when switching distance methods
- **Progress Tracking**: Live optimization progress with status indicators
- **Interactive Visualizations**: Route maps, cost analysis, distance heatmaps
- **Method-Specific Documentation**: Detailed explanations shown only after optimization
- **Solution Comparison**: Built-in tools to compare different methods

### **Command-Line Interface**

**Main Optimization Script:**
```bash
# Quick optimization with default settings
python src/main.py

# Genetic algorithm with real-world distances
python src/main.py --optimizer genetic --real-distances

# Enhanced MILP with custom depot location
python src/main.py --optimizer enhanced --depot 08025 --real-distances

# Available options:
# --optimizer: standard, enhanced, genetic (default: standard)
# --depot: depot postal code (default: 08020) 
# --real-distances: use OpenStreetMap distances (default: simulated)
# --quiet: reduce output verbosity
```

**Method Comparison Script:**
```bash
# Compare all methods with simulated distances
python src/comparison.py

# Compare with real-world distances and timeout
python src/comparison.py --real-distances --timeout 60

# Quick comparison with minimal output
python src/comparison.py --real-distances --timeout 30 --quiet

# Additional options:
# --ga-population: genetic algorithm population size (default: 50)
# --ga-generations: maximum generations (default: 100)
# --ga-mutation: mutation rate (default: 0.1)
# --depot-return: force trucks to return to depot
```

**Expected Performance:**
```
Standard MILP + Greedy:   < 5 seconds    (cost-optimal + route-optimized)
Enhanced MILP:           1-30 seconds    (globally optimal multi-objective)
Genetic Algorithm:       5-60 seconds    (evolutionary with high diversity)

Real-world distance overhead: +3-10 seconds (geocoding time)
```

### **Programmatic Integration**
```python
from vehicle_router import VrpOptimizer, DataGenerator, GeneticVrpOptimizer

# Generate test data
data_gen = DataGenerator(use_example_data=True)
orders_df = data_gen.generate_orders()
trucks_df = data_gen.generate_trucks()

# Option 1: Simulated distances (fast)
distance_matrix = data_gen.generate_distance_matrix(
    orders_df['postal_code'].tolist()
)

# Option 2: Real-world distances (accurate)
real_distance_matrix = data_gen.generate_distance_matrix(
    orders_df['postal_code'].tolist(),
    use_real_distances=True
)

# Standard MILP + Greedy optimization
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
success = optimizer.solve()

if success:
    solution = optimizer.get_solution()
    print(f"Total cost: €{solution['total_cost']}")
    print(f"Total distance: {solution['total_distance']:.1f} km")
    print(f"Selected trucks: {solution['selected_trucks']}")

# Genetic Algorithm optimization
ga_optimizer = GeneticVrpOptimizer(orders_df, trucks_df, real_distance_matrix)
ga_optimizer.set_parameters(population_size=50, max_generations=100, mutation_rate=0.1)
ga_success = ga_optimizer.solve(timeout=120)

if ga_success:
    ga_solution = ga_optimizer.get_solution()
    print(f"GA total cost: €{ga_solution['total_cost']}")
    print(f"GA total distance: {ga_solution['total_distance']:.1f} km")
```

## 📁 Project Architecture

```
Vehicle_Router/
├── app/                              # Streamlit web application
│   └── streamlit_app.py             # Main app with configurable models & real-time updates
├── app_utils/                        # App support modules
│   ├── data_handler.py              # Data loading with distance matrix reload
│   ├── optimization_runner.py       # Multi-method optimization orchestration
│   ├── visualization_manager.py     # Interactive charts with distance heatmaps
│   ├── export_manager.py            # Excel reports and solution exports
│   └── documentation.py             # Dynamic method-specific documentation
├── src/                             # Command-line tools
│   ├── main.py                      # CLI optimization with --real-distances flag
│   ├── main_utils.py               # CLI utilities with enhanced output formatting
│   └── comparison.py               # Multi-method comparison with recommendations
├── vehicle_router/                  # Core optimization library
│   ├── optimizer.py                # Standard MILP + Greedy (two-phase hybrid)
│   ├── enhanced_optimizer.py       # Enhanced MILP (multi-objective simultaneous)
│   ├── genetic_optimizer.py        # Genetic Algorithm (evolutionary metaheuristic)
│   ├── distance_calculator.py      # OpenStreetMap integration + fallback
│   ├── data_generator.py           # Test data with configurable distance methods
│   ├── validation.py               # Comprehensive solution validation
│   └── plotting.py                 # Visualization utilities
├── docs/                           # Comprehensive documentation
│   ├── methods.md                  # Detailed methodology explanations
│   └── usage.md                    # Complete usage guide with examples
├── .streamlit/                     # Streamlit configuration
│   └── config.toml                # App settings and model configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📊 Performance Benchmarks

### **Method Comparison** *(5 orders, Barcelona postal codes)*

| Method | Cost | Distance (Simulated) | Distance (Real-World) | Time | Best Use Case |
|--------|------|---------------------|----------------------|------|---------------|
| **Standard MILP + Greedy** | €2500 | 21.0 km | 12.2 km | 0.08s | Daily operations |
| **Enhanced MILP** | €2500 | 22.8 km | 13.9 km | 0.09s | Balanced optimization |
| **Genetic Algorithm** | €2500 | 19.0 km | 12.2 km | 0.32s | Large problems |

### **Real-World Distance Impact:**
- **Accuracy Improvement**: 15-40% more realistic distance estimates
- **Route Quality**: Better optimization due to actual geographic constraints
- **Performance Overhead**: +3-10 seconds for geocoding (cached during session)

### **Scalability Analysis:**
```
Problem Size     Standard MILP    Enhanced MILP    Genetic Algorithm
≤10 orders       < 1s             < 5s             < 10s
11-25 orders     < 5s             5-30s            10-30s  
26-50 orders     5-15s            30-120s          20-60s
51-100 orders    10-60s           120-300s*        30-120s
100+ orders      30-300s*         Not recommended  60-300s

* May require solver parameter tuning or additional computational resources
```

## 🔧 Advanced Configuration

### **Real-World Distance Calculation**
The system uses OpenStreetMap's Nominatim service with intelligent fallback:

- **Rate Limiting**: 0.5 seconds between API calls (respectful to free service)
- **Caching**: Coordinates cached in memory during execution
- **Fallback Hierarchy**: OpenStreetMap → Static calculation → Default (10km)
- **Error Handling**: Graceful degradation with informative logging

**Configuration Options:**
```python
# In vehicle_router/distance_calculator.py
calculator = DistanceCalculator(country_code="ES")  # Spain
calculator = DistanceCalculator(country_code="FR")  # France
calculator = DistanceCalculator(country_code="DE")  # Germany
```

### **Optimization Parameters**

**Standard MILP + Greedy:**
- `depot_return`: Whether trucks return to depot (default: False)
- `enable_greedy_routes`: Enable route sequence optimization (default: True)
- `solver_timeout`: CBC solver timeout in seconds (default: 60)

**Enhanced MILP:**
- `cost_weight`: Weight for truck costs (0-1, default: 0.6)
- `distance_weight`: Weight for travel distances (0-1, default: 0.4)
- `solver_timeout`: Extended timeout for complex model (default: 120)

**Genetic Algorithm:**
- `population_size`: Number of solutions per generation (default: 50)
- `max_generations`: Maximum evolution iterations (default: 100)
- `mutation_rate`: Probability of solution mutation (default: 0.1)
- `cost_weight` / `distance_weight`: Fixed at 0.5/0.5 for balanced optimization

## 🧪 Testing & Validation

```bash
# Run comprehensive test suite
python -m pytest tests/

# Test specific components
python -m pytest tests/test_optimizer.py
python -m pytest tests/test_data_generator.py
python -m pytest tests/test_validation.py

# Test real distance calculation
python vehicle_router/distance_calculator.py

# Integration testing
python src/comparison.py --real-distances --timeout 30 --quiet
```

## 📚 Documentation

- **[Methodology Guide](docs/methods.md)**: Detailed explanations of all three optimization approaches
- **[Usage Examples](docs/usage.md)**: Comprehensive usage guide with code examples
- **[API Documentation](vehicle_router/)**: Core module and class documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Implement changes with tests
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/enhancement`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenStreetMap**: Free geographic data and Nominatim geocoding service
- **PuLP**: Mixed Integer Linear Programming optimization library
- **Streamlit**: Interactive web application framework
- **CBC Solver**: Open-source MILP solver for optimization
- **Plotly**: Interactive visualization library
- **Vehicle Routing Research Community**: Mathematical foundations and algorithmic insights