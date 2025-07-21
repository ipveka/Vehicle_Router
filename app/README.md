# Vehicle Router Streamlit Application

A comprehensive web-based interface for solving Vehicle Routing Problems (VRP) with order assignment optimization using Mixed Integer Linear Programming (MILP).

## 🚀 Quick Start

### Running the Application

From the project root directory:

```bash
# Option 1: Use the automated runner (recommended)
python run_app.py

# Option 2: Run directly with Streamlit
streamlit run app/streamlit_app.py

# Option 3: Use the shell script (Unix/macOS)
./run_app.sh
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 📋 Application Overview

### What It Does

The Vehicle Router Streamlit app provides an interactive web interface for:

- **Data Management**: Load example data or upload custom CSV files
- **Problem Solving**: Run MILP optimization to find optimal vehicle assignments
- **Solution Analysis**: Explore detailed results with metrics and validation
- **Visualization**: Interactive charts, route maps, and cost analysis
- **Export**: Download results in Excel, CSV, or detailed report formats

### Key Features

#### 🔧 **Control Panel (Sidebar)**
- **Data Loading**: Choose between example data or custom CSV uploads
- **Optimization Parameters**: Configure solver timeout and validation settings
- **Real-time Status**: Monitor data loading and optimization progress

#### 📊 **Data Exploration**
- **Overview Metrics**: Total orders, volume, trucks, and capacity
- **Feasibility Check**: Automatic validation of problem constraints
- **Interactive Tables**: Sortable data views for orders, trucks, and distances
- **Data Visualizations**: Charts showing volume distribution, capacity analysis, and distance heatmaps

#### 🎯 **Solution Analysis**
- **Cost Breakdown**: Detailed analysis of optimization results
- **Truck Utilization**: Capacity usage and efficiency metrics
- **Assignment Details**: Complete order-to-truck mapping
- **Export Options**: Multiple formats for result sharing

#### 📈 **Advanced Visualizations**
- **Route Maps**: Interactive delivery route visualization
- **Cost Analysis**: Breakdown of operational costs and efficiency
- **Utilization Charts**: Capacity usage across selected trucks

## 📁 Data Requirements

### Input Data Format

#### Orders CSV
Required columns:
- `order_id`: Unique identifier for each order
- `volume`: Order volume in cubic meters (m³)
- `postal_code`: Delivery location postal code

Example:
```csv
order_id,volume,postal_code
ORD001,2.5,12345
ORD002,1.8,67890
ORD003,3.2,54321
```

#### Trucks CSV
Required columns:
- `truck_id`: Unique identifier for each truck
- `capacity`: Maximum capacity in cubic meters (m³)
- `cost`: Operational cost in euros (€)

Example:
```csv
truck_id,capacity,cost
1,10.0,150
2,15.0,200
3,8.0,120
```

### Data Validation

The application automatically validates:
- **File Format**: Ensures CSV files have required columns
- **Data Types**: Validates numeric values for volume, capacity, and cost
- **Feasibility**: Checks if total capacity meets total volume requirements
- **Completeness**: Verifies no missing or invalid data

## 🎯 Using the Application

### Step-by-Step Guide

#### 1. **Load Data**
   - **Option A**: Click "Load Example Data" for a quick start
   - **Option B**: Upload your own CSV files using the file uploaders
   - Wait for the success confirmation message

#### 2. **Configure Optimization**
   - Set solver timeout (10-300 seconds)
   - Enable/disable solution validation
   - Review the problem feasibility status

#### 3. **Run Optimization**
   - Click "Run Optimization" button
   - Monitor progress in the sidebar
   - Wait for completion confirmation

#### 4. **Analyze Results**
   - Review solution metrics and cost breakdown
   - Examine truck assignments and utilization
   - Validate constraint compliance

#### 5. **Explore Visualizations**
   - View interactive route maps
   - Analyze cost efficiency charts
   - Study capacity utilization patterns

#### 6. **Export Results**
   - Download Excel reports for detailed analysis
   - Export CSV summaries for data processing
   - Generate text reports for documentation

### Navigation Tips

- **Sidebar**: All controls and status information
- **Main Area**: Organized in expandable sections
- **Tabs**: Switch between different data views and visualizations
- **Metrics**: Key performance indicators displayed prominently
- **Interactive Elements**: Click, hover, and zoom on charts

## 🔧 Technical Details

### Architecture

The application is built using:

- **Frontend**: Streamlit for web interface
- **Backend**: Python with PuLP for optimization
- **Solver**: CBC (Coin-or Branch and Cut) for MILP
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and NumPy

### Application Structure

```
app/
├── streamlit_app.py          # Main application entry point
└── README.md                 # This documentation

app_utils/                    # Supporting modules
├── data_handler.py           # Data loading and validation
├── optimization_runner.py    # MILP solver interface
├── visualization_manager.py  # Chart and plot generation
├── export_manager.py         # Result export functionality
└── ui_components.py          # Reusable UI elements
```

### Performance Considerations

- **Solver Timeout**: Adjust based on problem size (larger problems need more time)
- **Data Size**: Application tested with up to 100 orders and 20 trucks
- **Memory Usage**: Efficient data structures minimize memory footprint
- **Response Time**: Most operations complete within seconds

## 🎨 User Interface Features

### Styling and Design

- **Professional Theme**: Clean, modern interface design
- **Color Coding**: Consistent color scheme for different data types
- **Responsive Layout**: Adapts to different screen sizes
- **Interactive Elements**: Hover effects and clickable components

### Accessibility

- **Clear Navigation**: Logical flow and intuitive controls
- **Status Indicators**: Visual feedback for all operations
- **Error Handling**: Informative error messages and recovery suggestions
- **Help Text**: Tooltips and explanations for complex features

## 🔍 Troubleshooting

### Common Issues

#### Data Loading Problems
- **File Format**: Ensure CSV files have correct column names
- **Data Types**: Check that numeric columns contain valid numbers
- **File Size**: Large files may take longer to process

#### Optimization Failures
- **Infeasible Problem**: Total truck capacity must exceed total order volume
- **Solver Timeout**: Increase timeout for complex problems
- **Memory Issues**: Reduce problem size or restart application

#### Visualization Issues
- **Chart Loading**: Refresh the page if charts don't appear
- **Interactive Features**: Ensure JavaScript is enabled in browser
- **Export Problems**: Check browser download settings

### Getting Help

1. **Check Status Messages**: Look for error messages in the sidebar
2. **Review Data**: Ensure input data meets format requirements
3. **Adjust Parameters**: Try different solver settings
4. **Restart Application**: Close and reopen if issues persist

## 📊 Example Use Cases

### Logistics Company
- **Scenario**: Daily delivery route optimization
- **Data**: 50 orders across city postal codes, 8 available trucks
- **Goal**: Minimize operational costs while meeting all deliveries

### E-commerce Fulfillment
- **Scenario**: Warehouse order distribution
- **Data**: Variable order volumes, mixed truck capacities
- **Goal**: Optimize truck utilization and reduce shipping costs

### Supply Chain Management
- **Scenario**: Multi-location distribution planning
- **Data**: Regional orders, specialized vehicle fleet
- **Goal**: Balance cost efficiency with service requirements

## 🚀 Advanced Features

### Custom Optimization Parameters
- Solver timeout adjustment for complex problems
- Solution validation toggle for performance optimization
- Real-time progress monitoring during optimization

### Professional Reporting
- **Excel Reports**: Comprehensive analysis with multiple worksheets
- **CSV Exports**: Machine-readable data for further processing
- **Text Reports**: Human-readable summaries for documentation

### Interactive Visualizations
- **Zoomable Charts**: Detailed exploration of data patterns
- **Hover Information**: Contextual data on chart elements
- **Downloadable Plots**: Save visualizations for presentations

---

## 📞 Support

For technical support or feature requests, please refer to the main project documentation or contact the development team.

**Happy Optimizing! 🚛📈**