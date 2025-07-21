# Vehicle Router Documentation

Welcome to the comprehensive documentation for the Vehicle Router project - a production-ready Python application for solving Vehicle Routing Problems (VRP) with order assignment optimization.

## Project Overview

The Vehicle Router is designed to solve complex logistics optimization problems where you need to:
- Assign orders to available trucks based on capacity constraints
- Minimize total operational costs
- Ensure all orders are delivered efficiently
- Generate detailed reports and visualizations

The system uses Mixed Integer Linear Programming (MILP) to find optimal solutions, making it suitable for both academic research and real-world logistics applications.

## Key Features

### 🚀 **Production Ready**
- Comprehensive error handling and validation
- Detailed logging and progress tracking
- Robust optimization engine using PuLP
- Full test suite with pytest

### 📊 **Data Management**
- Built-in example datasets for quick testing
- Random data generation for experimentation
- Support for custom data inputs
- Pandas-based data structures for easy manipulation

### 🎯 **Dual Optimization Engines**
- **Standard Model**: MILP cost optimization + greedy route optimization
- **Enhanced Model**: Integrated MILP with simultaneous cost and distance optimization
- **Configurable Depot Options**: Customizable depot location and return requirements
- **Advanced Route Optimization**: Multiple algorithms for different use cases
- Scalable to medium-large problem instances

### 📈 **Visualization & Analysis**
- Automatic route visualization
- Cost breakdown analysis
- Capacity utilization charts
- Export-ready plots and reports

### 🌐 **Interactive Web Application**
- Streamlit-based web interface
- Interactive data exploration and visualization
- Real-time optimization execution
- Excel/CSV export functionality

## Getting Started

### Quick Start (5 minutes)

1. **Install the application**:
   ```bash
   git clone https://github.com/your-org/vehicle-router.git
   cd vehicle-router
   pip install -r requirements.txt
   ```

2. **Run the web application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   
   Or run the command-line version:
   ```bash
   python src/main.py
   ```

3. **View results**:
   - **Web App**: Interactive interface with data exploration, optimization, and visualization
   - **Command Line**: Console output shows optimal truck selection and assignments
   - Check `output/` directory for generated visualizations and CSV files
   - Review `logs/vehicle_router.log` for detailed execution logs

### Example Output

When you run the application with the built-in example data, you'll see:

```
=== VEHICLE ROUTER ===
Selected Trucks: [1, 2, 3, 5]
Truck 1 -> Orders ['A', 'E']
Truck 2 -> Orders ['B']
Truck 3 -> Orders ['C']
Truck 5 -> Orders ['D']
Total Cost: €4000
```

This means the optimizer found that using trucks 1, 2, 3, and 5 provides the most cost-effective solution for delivering all orders.

## Architecture Overview

The Vehicle Router follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Generator │    │  MILP Optimizer │    │ Solution        │
│                 │───▶│                 │───▶│ Validator       │
│ • Orders        │    │ • PuLP Model    │    │ • Constraints   │
│ • Trucks        │    │ • Variables     │    │ • Feasibility   │
│ • Distances     │    │ • Constraints   │    │ • Reports       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Main Application│    │   Visualization │    │    Utilities    │
│                 │    │                 │    │                 │
│ • Workflow      │    │ • Route Maps    │    │ • Distance Calc │
│ • Logging       │    │ • Cost Charts   │    │ • Formatting    │
│ • Error Handling│    │ • Utilization   │    │ • Helpers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. DataGenerator (`vehicle_router/data_generator.py`)
Handles all data generation and management:
- **Example Data**: Replicates the standard test case with 5 orders and 5 trucks
- **Random Data**: Generates test scenarios with configurable parameters
- **Validation**: Ensures data consistency and feasibility
- **Distance Matrices**: Calculates distances between postal code locations

### 2. VrpOptimizer (`vehicle_router/optimizer.py`)
The heart of the optimization engine:
- **MILP Formulation**: Builds mathematical model with decision variables and constraints
- **PuLP Integration**: Uses industry-standard linear programming solver
- **Solution Extraction**: Converts solver output to structured results
- **Performance Monitoring**: Tracks solving time and convergence

### 3. SolutionValidator (`vehicle_router/validation.py`)
Ensures solution correctness:
- **Capacity Validation**: Verifies no truck exceeds its capacity limit
- **Assignment Validation**: Confirms all orders are delivered exactly once
- **Route Feasibility**: Checks logical consistency of delivery routes
- **Comprehensive Reports**: Provides detailed validation summaries

### 4. Visualization (`vehicle_router/plotting.py`)
Creates publication-ready visualizations:
- **Route Maps**: 2D visualization of truck routes and order locations
- **Cost Analysis**: Bar charts showing cost breakdown by truck
- **Utilization Charts**: Capacity utilization rates and efficiency metrics
- **Export Options**: Save plots in multiple formats (PNG, PDF, SVG)

## Problem Formulation

The Vehicle Router solves a variant of the Capacitated Vehicle Routing Problem (CVRP) with the following characteristics:

### Input Data
- **Orders**: Set of delivery orders, each with volume and location
- **Trucks**: Fleet of vehicles with capacity and cost parameters
- **Distances**: Travel distances between all location pairs

### Optimization Objective
Minimize total operational cost while satisfying all constraints:
- Each order must be delivered exactly once
- No truck can exceed its capacity limit
- All selected trucks incur their associated costs

### Mathematical Model
The problem is formulated as a Mixed Integer Linear Program (MILP):
- **Binary Variables**: Order-to-truck assignments and truck usage indicators
- **Linear Constraints**: Capacity limits and assignment requirements
- **Objective Function**: Sum of costs for selected trucks

For detailed mathematical formulation, see [Model Description](model_description.md).

## Use Cases

### 1. **Logistics Planning**
- Optimize daily delivery routes for courier services
- Plan truck assignments for distribution centers
- Minimize transportation costs in supply chain operations

### 2. **Academic Research**
- Study vehicle routing algorithms and heuristics
- Compare optimization approaches and solution quality
- Generate test instances for algorithm development

### 3. **Proof of Concept**
- Demonstrate optimization capabilities to stakeholders
- Prototype logistics solutions before full implementation
- Validate business cases for route optimization investments

## Performance Characteristics

The Vehicle Router is designed for efficiency across different problem sizes:

| Problem Size | Orders | Trucks | Typical Solve Time | Memory Usage |
|--------------|--------|--------|--------------------|--------------|
| Small        | 5-10   | 3-5    | < 1 second        | < 50 MB      |
| Medium       | 20-50  | 10-15  | 1-10 seconds      | < 200 MB     |
| Large        | 100+   | 20+    | 1-10 minutes      | < 1 GB       |

Performance depends on:
- Problem complexity (number of feasible solutions)
- Hardware specifications (CPU, memory)
- Solver configuration and parameters

## Next Steps

### For New Users
1. **Read the [Usage Guide](usage.md)** - Learn how to customize the application for your needs
2. **Explore the [Model Description](model_description.md)** - Understand the mathematical formulation
3. **Run the Examples** - Try different scenarios and configurations

### For Developers
1. **Review the Source Code** - Understand the implementation details
2. **Run the Test Suite** - Ensure everything works correctly (`pytest`)
3. **Extend the Functionality** - Add new features or optimization approaches

### For Researchers
1. **Study the MILP Formulation** - Analyze the mathematical model
2. **Generate Test Instances** - Create custom datasets for experimentation
3. **Compare with Other Approaches** - Benchmark against heuristic methods

## Support and Resources

- **Documentation**: Complete API reference and examples in this docs/ directory
- **Source Code**: Well-commented Python code with extensive docstrings
- **Test Suite**: Comprehensive tests demonstrating usage patterns
- **Issue Tracking**: GitHub Issues for bug reports and feature requests
- **Community**: Active development and user community

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing use cases, your input helps make the Vehicle Router better for everyone.

See the main [README.md](../README.md) for contribution guidelines and development setup instructions.

---

*This documentation is maintained alongside the codebase to ensure accuracy and completeness. Last updated: January 2025*