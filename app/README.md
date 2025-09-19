# Vehicle Router - Streamlit Web Application

## 🌐 Overview

The Vehicle Router Streamlit application provides an interactive web interface for solving Vehicle Routing Problems (VRP) with multiple optimization approaches. The app features real-time optimization, interactive visualizations, and comprehensive solution analysis.

## 🚀 Quick Start

```bash
# Launch the web application
streamlit run app/streamlit_app.py

# Or use the app runner script
python run_app.py
```

Access the application at: `http://localhost:8501`

## 📊 Data Structure Requirements

### Orders Data (CSV Format)

**Required Columns:**
- `order_id` (string): Unique identifier for each order (e.g., "A", "B", "C", "Order_001")
- `volume` (numeric): Order volume in cubic meters (m³), must be positive
- `postal_code` (string): Delivery location postal code (e.g., "08027", "08030")

**Example Orders CSV:**
```csv
order_id,volume,postal_code
A,25.0,08031
B,50.0,08030
C,25.0,08029
D,25.0,08028
E,25.0,08027
```

**Validation Rules:**
- All order IDs must be unique
- Volume values must be positive numbers
- Postal codes should be valid (5-digit format recommended)
- No missing values allowed in required columns

### Trucks Data (CSV Format)

**Required Columns:**
- `truck_id` (integer): Unique identifier for each truck (e.g., 1, 2, 3)
- `capacity` (numeric): Truck capacity in cubic meters (m³), must be positive
- `cost` (numeric): Truck usage cost in euros (€), must be non-negative

**Example Trucks CSV:**
```csv
truck_id,capacity,cost
1,100.0,1500.0
2,50.0,1000.0
3,25.0,500.0
4,25.0,1500.0
5,25.0,1000.0
```

**Validation Rules:**
- All truck IDs must be unique
- Capacity values must be positive numbers
- Cost values must be non-negative numbers
- Total truck capacity should be sufficient for all orders
- No missing values allowed in required columns

## 🎛️ Application Features

### Data Management
- **Example Data Loading**: Pre-configured Barcelona postal codes dataset
- **Custom Data Upload**: Support for CSV file uploads with validation
- **Data Validation**: Automatic format and content validation
- **Data Summary**: Real-time statistics and feasibility analysis

### Distance Calculation Methods
- **🌍 Real-World Distances**: OpenStreetMap geocoding with Haversine calculations
- **📐 Simulated Distances**: Mathematical approximation (1km per postal code unit)
- **Real-Time Switching**: Toggle between methods with automatic distance matrix reload

### Optimization Methods

#### 1. 📊 Standard MILP + Greedy
- **Approach**: Two-phase hybrid optimization
- **Performance**: < 5 seconds execution time
- **Best for**: Daily operations, balanced cost-distance optimization

#### 2. 🧬 Genetic Algorithm
- **Approach**: Evolutionary multi-objective optimization
- **Performance**: 5-60 seconds execution time
- **Best for**: Large problems, solution diversity exploration

#### 3. 🚀 Enhanced MILP (Advanced)
- **Approach**: Simultaneous multi-objective optimization
- **Performance**: 1-30 seconds execution time
- **Best for**: High-quality routes requiring optimal balance
- **Note**: Hidden by default, can be enabled in configuration

### Interactive Features
- **Real-Time Progress**: Live optimization status with progress indicators
- **Interactive Visualizations**: Route maps, cost analysis, distance heatmaps
- **Solution Comparison**: Side-by-side method comparison tools
- **Export Capabilities**: Excel reports and CSV downloads

## 🔧 Configuration Options

### Model Availability
Configure which optimization methods are available in the UI by modifying `AVAILABLE_MODELS` in `streamlit_app.py`:

```python
AVAILABLE_MODELS = {
    'standard': {
        'name': '📊 Standard MILP + Greedy',
        'enabled': True
    },
    'genetic': {
        'name': '🧬 Genetic Algorithm',
        'enabled': True
    },
    'enhanced': {
        'name': '🚀 Enhanced MILP',
        'enabled': False  # Hidden by default
    }
}
```

### Distance Settings
- **Default Depot**: 08020 (Barcelona, configurable)
- **Real-World Distances**: Toggle in sidebar
- **Rate Limiting**: 0.5 seconds between OpenStreetMap API calls
- **Fallback**: Automatic fallback to simulated distances if API fails

### Optimization Parameters
- **Max Orders per Truck**: Default 3 (production constraint)
- **Genetic Algorithm**: Population size, generations, mutation rate
- **Solver Timeouts**: Configurable per optimization method

## 📈 Application Workflow

### 1. Data Loading
- Click "Load Example Data" for Barcelona dataset
- Or upload custom CSV files using file uploaders
- View data summary and validation results

### 2. Distance Configuration
- Toggle "🌍 Use Real-World Distances" in sidebar
- Distance matrix automatically reloads when toggled
- View distance calculation progress

### 3. Optimization Setup
- Select optimization method using buttons
- Configure method-specific parameters
- Set maximum orders per truck constraint

### 4. Run Optimization
- Click "🚀 Run Optimization" button
- Monitor real-time progress indicators
- View optimization status and timing

### 5. Results Analysis
- Explore interactive route visualizations
- Review cost breakdown and performance metrics
- Compare different optimization methods
- Export results to Excel or CSV

## 🎨 User Interface Components

### Sidebar
- Distance calculation toggle
- Data loading controls
- Method selection buttons
- Parameter configuration

### Main Panel
- Data summary cards
- Optimization controls
- Progress indicators
- Results visualization
- Export options

### Visualization Types
- **Route Maps**: Interactive geographic route display
- **Cost Analysis**: Breakdown of truck costs and distances
- **Distance Heatmaps**: Visual distance matrix representation
- **Performance Charts**: Execution time and quality metrics

## 🔍 Data Validation & Error Handling

### Automatic Validation
- Column presence and naming
- Data type verification
- Value range checking
- Uniqueness constraints
- Feasibility analysis

### Error Messages
- Clear, actionable error descriptions
- Specific column and row references
- Suggested corrections
- Validation status indicators

### Fallback Mechanisms
- Distance calculation fallbacks
- Default parameter values
- Graceful error recovery
- Session state preservation

## 📊 Performance Expectations

### Data Size Limits
- **Small**: ≤10 orders, ≤5 trucks (< 1 second)
- **Medium**: 11-25 orders, ≤10 trucks (< 30 seconds)
- **Large**: 26-50 orders, ≤15 trucks (< 2 minutes)

### Distance Calculation Overhead
- **Simulated**: Instant calculation
- **Real-World**: +3-10 seconds (first time, then cached)

### Memory Usage
- Typical session: 50-100MB
- Large datasets: 200-500MB
- Distance matrices: Scales quadratically with locations

## 🛠️ Troubleshooting

### Common Issues
1. **CSV Upload Errors**: Check column names and data types
2. **Distance Calculation Slow**: Use simulated distances for testing
3. **Optimization Timeout**: Reduce problem size or increase timeout
4. **Memory Issues**: Restart application for large datasets

### Debug Information
- Check browser console for JavaScript errors
- View Streamlit logs for Python exceptions
- Use "Show Data" toggles for data inspection
- Monitor system resources during optimization

## 📁 Application Architecture

```
app/
├── streamlit_app.py          # Main application entry point
└── README.md                 # This documentation

app_utils/                    # Supporting modules
├── data_handler.py          # Data loading and validation
├── optimization_runner.py   # Optimization orchestration
├── visualization_manager.py # Interactive charts and maps
├── export_manager.py        # Excel and CSV export
├── ui_components.py         # Reusable UI elements
└── documentation.py         # Dynamic help content
```

## 🔗 Related Documentation

- [Main README](../README.md) - Complete project overview
- [Setup Guide](../setup.md) - Installation instructions
- [CLI Documentation](../src/) - Command-line tools
- [Core Library](../vehicle_router/) - Optimization algorithms
