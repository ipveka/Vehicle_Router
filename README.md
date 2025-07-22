# Vehicle Router Optimization System

A Python application for solving Vehicle Routing Problems (VRP) using multiple optimization approaches. The system includes both command-line tools and an interactive Streamlit web application for route optimization with real-world geographic distances.

## 🚀 Key Features

- **Multiple Optimization Methods**: Standard MILP + Greedy, Enhanced MILP, and Genetic Algorithm
- **Real-World Distances**: OpenStreetMap integration for accurate geographic routing
- **Interactive Web Interface**: Streamlit app with visual analysis and export capabilities  
- **Command-Line Tools**: CLI scripts for automation and batch processing
- **Comprehensive Analysis**: Solution comparison, validation, and performance metrics
- **Flexible Configuration**: Customizable models, parameters, and distance calculation methods

## 📊 Optimization Methods

### 1. **Standard MILP + Greedy** *(Default)*
- **Approach**: Cost-focused MILP followed by route sequence optimization
- **Best for**: Daily operations, cost minimization priority
- **Performance**: Fast execution (< 0.1s), cost-optimal truck selection

### 2. **Enhanced MILP** 
- **Approach**: Multi-objective optimization balancing cost and distance
- **Best for**: High-quality routes, balanced optimization
- **Performance**: Medium execution time, globally optimal solutions

### 3. **Genetic Algorithm**
- **Approach**: Evolutionary algorithm with balanced 50/50 cost-distance optimization
- **Best for**: Large problems, diverse solution exploration  
- **Performance**: Longer execution time, generally good solutions with variety

## 🌍 Distance Calculation Methods

### **Simulated Distances** *(Default)*
- **Method**: 1km per postal code unit difference
- **Pros**: Instant calculation, no network dependencies
- **Use case**: Quick testing, development, offline environments

### **Real-World Distances** *(New)*
- **Method**: OpenStreetMap geocoding + Haversine distance calculation
- **Pros**: Accurate geographic distances, realistic route planning  
- **Cons**: Requires internet connection, ~1 second per postal code geocoding
- **Use case**: Production routing, accurate distance estimation
- **Fallback**: Automatically falls back to static calculation if geocoding fails

## 🎛️ App Configuration

The Streamlit application supports configurable model selection. By default, two main optimization methods are enabled:

- **Standard MILP + Greedy**: Fast, cost-focused optimization
- **Genetic Algorithm**: Balanced multi-objective optimization  
- **Enhanced MILP**: Hidden by default (can be enabled)

To customize available models, modify the `AVAILABLE_MODELS` dictionary in `app/streamlit_app.py`:

```python
AVAILABLE_MODELS = {
    'standard': {'name': '📊 Standard MILP + Greedy', 'enabled': True},
    'enhanced': {'name': '🚀 Enhanced MILP', 'enabled': False},  # Hidden
    'genetic': {'name': '🧬 Genetic Algorithm', 'enabled': True}
}
```

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- pip package manager
- Internet connection (for real-world distances)

### **Setup**
```bash
# Clone the repository
git clone <repository-url>
cd Vehicle_Router

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### **Required Dependencies**
```
pandas >= 1.3.0
numpy >= 1.21.0
streamlit >= 1.25.0
plotly >= 5.0.0
pulp >= 2.6.0
requests >= 2.25.0  # For OpenStreetMap integration
```

## 🚀 Quick Start

### **Streamlit Web Application**
```bash
# Launch the interactive app
streamlit run app/streamlit_app.py

# Or use the run script
python run_app.py
```

**Streamlit App Features:**
- **Configurable Models**: Select from available optimization methods
- **Real Distance Toggle**: Switch between simulated and OpenStreetMap distances (updates automatically)
- **Method-Specific Parameters**: Genetic algorithm population, generations, mutation rate
- **Interactive Analysis**: Route visualization, cost breakdown, performance metrics  
- **Data Management**: Upload custom CSV files or use example data
- **Export Capabilities**: Download results and visualizations

### **Command-Line Interface**
```bash
# Run optimization with default settings
python src/main.py

# Use genetic algorithm with real distances
python src/main.py --optimizer genetic
# (Then set use_real_distances=True in config)

# Compare all methods with real distances
python src/comparison.py --real-distances --timeout 60

# Available optimizers: standard, enhanced, genetic
# Available options: --depot, --timeout, --real-distances, --quiet
```

### **Programmatic Usage**
```python
from vehicle_router import VrpOptimizer, DataGenerator

# Generate test data
data_gen = DataGenerator(use_example_data=True)
orders_df = data_gen.generate_orders()
trucks_df = data_gen.generate_trucks()

# Option 1: Simulated distances (fast)
distance_matrix = data_gen.generate_distance_matrix(
    orders_df['postal_code'].tolist()
)

# Option 2: Real-world distances (OpenStreetMap)
real_distance_matrix = data_gen.generate_distance_matrix(
    orders_df['postal_code'].tolist(),
    use_real_distances=True
)

# Solve with Standard MILP + Greedy
optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
success = optimizer.solve()

if success:
    solution = optimizer.get_solution()
    print(f"Total cost: €{solution['total_cost']}")
    print(f"Total distance: {solution['total_distance']:.1f} km")
```

## 📁 Project Structure

```
Vehicle_Router/
├── app/                          # Streamlit web application
│   └── streamlit_app.py         # Main app interface
├── app_utils/                    # App support modules
│   ├── data_handler.py          # Data loading and validation
│   ├── optimization_runner.py   # Optimization orchestration
│   ├── visualization_manager.py # Plot generation
│   └── documentation.py         # In-app documentation
├── src/                         # Command-line tools
│   ├── main.py                  # CLI optimization runner
│   ├── main_utils.py           # CLI utilities and managers
│   └── comparison.py           # Multi-method comparison tool
├── vehicle_router/              # Core optimization library
│   ├── optimizer.py            # Standard MILP + Greedy
│   ├── enhanced_optimizer.py   # Enhanced MILP
│   ├── genetic_optimizer.py    # Genetic Algorithm
│   ├── distance_calculator.py  # OpenStreetMap integration + static fallback
│   ├── data_generator.py       # Test data generation
│   ├── validation.py           # Solution validation
│   └── plotting.py             # Visualization utilities
├── docs/                       # Documentation
│   ├── methods.md              # Optimization methods guide
│   └── usage.md                # Usage examples
├── tests/                      # Test suite
├── .streamlit/                 # Streamlit configuration
│   └── config.toml            # App and theme settings
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Performance Comparison

**Example Results** *(5 orders, Barcelona postal codes)*:

| Method | Cost | Distance | Time | Best Use Case |
|--------|------|----------|------|---------------|
| **Standard MILP + Greedy** | €2500 | 12.2 km | 0.08s | Daily operations |
| **Enhanced MILP** | €2500 | 13.9 km | 0.09s | Balanced optimization |
| **Genetic Algorithm** | €2500 | 12.2 km | 0.32s | Large problems |

**Distance Accuracy** *(Real vs Simulated)*:
- **Real-world distances**: Based on actual geographic coordinates
- **Typical improvement**: 20-40% more accurate route distances
- **Geocoding time**: ~1 second per postal code (cached for subsequent runs)

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/

# Test specific components
python -m pytest tests/test_optimizer.py
python -m pytest tests/test_data_generator.py

# Test real distance calculation
python vehicle_router/simple_distance_calculator.py
```

## 🔧 Configuration

### **Real-World Distances**
Real distances use OpenStreetMap's Nominatim geocoding service:
- **Rate limiting**: 1 request per second (respectful to free service)
- **Caching**: Coordinates cached in memory during execution
- **Fallback**: Falls back to static calculation if geocoding fails
- **Coverage**: Global coverage, works for any country code
- **Auto-reload**: Distance matrix updates automatically when toggled in app

### **Distance Methods Comparison**
```python
# Simulated distances (current default)
distance_matrix = data_gen.generate_distance_matrix(postal_codes)

# Real-world distances (OpenStreetMap)
real_matrix = data_gen.generate_distance_matrix(
    postal_codes, use_real_distances=True
)
```

## 📚 Documentation

- **[Optimization Methods](docs/methods.md)**: Detailed guide to all three optimization approaches
- **[Usage Examples](docs/usage.md)**: Code examples and integration patterns
- **[API Reference](vehicle_router/)**: Core module documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)  
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenStreetMap**: Free geographic data and geocoding services
- **PuLP**: Linear programming optimization library
- **Streamlit**: Interactive web application framework
- **CBC Solver**: Open-source MILP solver