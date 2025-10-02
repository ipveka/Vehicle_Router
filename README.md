# Vehicle Router Optimization System

A comprehensive Python application for solving Vehicle Routing Problems (VRP) using multiple optimization approaches. Features both command-line tools and an interactive Streamlit web application with real-world geographic distance integration.

## 🚀 Key Features

- **Multiple Optimization Methods**: Standard MILP + Greedy, Enhanced MILP, and Genetic Algorithm
- **Real-World Distance Integration**: OpenStreetMap geocoding with Haversine calculations
- **Interactive Web Interface**: Streamlit app with real-time updates and visualizations
- **Command-Line Tools**: CLI scripts for automation and batch processing
- **Solution Analysis**: Comprehensive validation, performance metrics, and method comparison
- **Flexible Configuration**: Customizable parameters and optimization constraints

## 📊 Optimization Methods

### 1. **Standard MILP + Greedy** *(Fast & Balanced)*
- **Approach**: Two-phase hybrid optimization
- **Phase 1**: MILP for cost-optimal truck selection
- **Phase 2**: Greedy route optimization for distance minimization
- **Best for**: Daily operations, quick results (< 5s)
- **Performance**: Excellent for typical distributions

### 2. **Enhanced MILP** *(Advanced)*
- **Approach**: Simultaneous multi-objective optimization
- **Method**: Extended MILP with routing variables and flow constraints
- **Best for**: High-quality routes requiring optimal cost-distance balance
- **Performance**: Medium execution time (1-30s), globally optimal solutions

### 3. **Genetic Algorithm** *(Evolutionary)*
- **Approach**: Population-based evolutionary optimization
- **Method**: Tournament selection, crossover, mutation with constraint repair
- **Best for**: Large problems, solution diversity exploration
- **Performance**: Excellent scalability for complex scenarios

## 🌍 Distance Calculation

### **Simulated Distances** *(Simple)*
- Mathematical calculation based on postal code differences (1km per unit)
- Instant results, no network dependencies
- Perfect for testing and development

### **Real-World Distances** *(Advanced)*
- OpenStreetMap geocoding + Haversine distance calculation
- Accurate geographic distances reflecting actual geography
- Requires internet connection, ~0.5s per postal code
- Ideal for production routing

## 🌐 Streamlit Web Application

### **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
python run_app.py
```

The web app will open at `http://localhost:8501` with:
- **Interactive Data Loading**: Upload CSV files or use example data
- **Real-Time Optimization**: Live progress tracking and status updates
- **Route Visualization**: Interactive maps showing delivery routes
- **Excel Export**: Comprehensive reports with multiple sheets
- **Distance Method Selection**: Switch between simulated and real-world distances

## 💻 Command-Line Interface

### **Basic Usage**
```bash
# Run optimization with default settings
python src/main.py

# Run with custom parameters
python src/main.py --real-distances --timeout 300 --max-orders 4
```

### **Available Options**
- `--real-distances`: Use OpenStreetMap for geographic distances
- `--timeout SECONDS`: Solver timeout (default: 300)
- `--max-orders N`: Maximum orders per truck (default: 3)
- `--depot POSTAL_CODE`: Depot location (default: '08020')
- `--depot-return`: Enable depot return (default: False)

## 📁 Project Structure

```
Vehicle_Router/
├── app/                          # Streamlit web application
│   ├── config.py                 # App configuration
│   └── streamlit_app.py          # Main Streamlit app
├── app_utils/                    # Web app utilities
│   ├── data_handler.py           # Data loading and validation
│   ├── documentation.py          # App documentation
│   ├── export_manager.py         # Excel export functionality
│   ├── optimization_runner.py    # Optimization workflow
│   ├── ui_components.py          # UI components
│   └── visualization_manager.py # Interactive visualizations
├── src/                          # Command-line tools
│   ├── main.py                   # Main CLI application
│   └── main_utils.py             # CLI utilities
├── vehicle_router/               # Core optimization engine
│   ├── data_generator.py         # Test data generation
│   ├── distance_calculator.py    # Distance calculations
│   ├── logger_config.py          # Logging configuration
│   ├── optimizer.py              # Standard MILP + Greedy optimizer
│   ├── plotting.py               # Static plot generation
│   ├── utils.py                  # Utility functions
│   └── validation.py            # Solution validation
├── tests/                        # Test suite
├── output/                       # Generated files (plots, Excel reports)
├── logs/                         # Application logs
└── requirements.txt              # Python dependencies
```

## 🔧 Installation

### **Requirements**
- Python 3.8+
- Required packages listed in `requirements.txt`

### **Setup**
```bash
# Clone the repository
git clone <repository-url>
cd Vehicle_Router

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python run_tests.py
```

## 📊 Output Files

### **CLI Output** (`output/` directory)
- `vehicle_router_results.xlsx`: Comprehensive Excel report with multiple sheets
- `routes.png`: Route visualization plot
- `costs.png`: Cost breakdown chart
- `utilization.png`: Capacity utilization analysis

### **Excel Report Sheets**
- **Summary**: Key metrics and performance indicators
- **Orders**: Complete order data
- **Trucks**: Truck specifications and costs
- **Assignments**: Order-to-truck assignments
- **Routes**: Detailed route sequences
- **Utilization**: Capacity utilization analysis
- **Cost Breakdown**: Detailed cost analysis

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test modules
python -m pytest tests/test_streamlit_app.py -v
python -m pytest tests/test_config_system.py -v
```

## 📈 Performance Characteristics

- **Execution Time**: Typically < 5 seconds for standard problems
- **Scalability**: Handles up to 8 orders per truck efficiently
- **Memory Usage**: Low resource requirements (< 100MB)
- **Solution Quality**: Cost-optimal selection with distance-optimized routes

## 🌍 Real-World Distance Integration

The system integrates with OpenStreetMap for accurate geographic distances:

- **Geocoding**: Postal codes → GPS coordinates
- **Distance Calculation**: Haversine formula for great-circle distances
- **Caching**: Coordinate caching to minimize API calls
- **Rate Limiting**: Built-in delays to respect service limits
- **Fallback**: Graceful degradation if services unavailable

## 📝 Configuration

### **App Configuration** (`app/config.py`)
- `OPTIMIZATION_METHOD`: Currently set to 'standard'
- `OPTIMIZATION_DEFAULTS`: Default parameters for optimization
- `DISTANCE_CONFIG`: Distance calculation settings
- `DEPOT_CONFIG`: Default depot location

### **CLI Configuration** (`src/main.py`)
- Configurable via command-line arguments
- Default settings optimized for typical use cases
- Comprehensive logging and error handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions:
- Check the test suite for usage examples
- Review the configuration files for customization options
- Examine the optimization logs for debugging information