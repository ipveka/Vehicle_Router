# Vehicle Router - Quick Setup Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+ with pip
- Internet connection (for real-world distances)
- ~500MB disk space

### Installation
```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd Vehicle_Router

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package in development mode
pip install -e .

# 4. Verify installation
python src/main.py --help
```

## 🌐 Run Web Application
```bash
# Launch Streamlit web interface
streamlit run app/streamlit_app.py

# Access at: http://localhost:8501
```

**Web App Workflow:**
1. Click "Load Example Data" or upload CSV files
2. Toggle "🌍 Use Real-World Distances" for accuracy
3. Select optimization method (Standard MILP, Genetic Algorithm)
4. Click "🚀 Run Optimization"
5. Analyze results and export data

## 💻 Command Line Usage
```bash
# Quick start with default settings
python src/main.py

# Advanced optimization with real-world distances
python src/main.py --optimizer enhanced --real-distances

# Compare all methods
python src/comparison.py --real-distances

# Available optimizers: standard, enhanced, genetic
# Use --help for full options
```

## 🔧 Key Features
- **3 Optimization Methods**: Standard MILP + Greedy, Enhanced MILP, Genetic Algorithm
- **Real-World Distances**: OpenStreetMap integration with Haversine calculations
- **Production Constraint**: Max 3 orders per truck (configurable)
- **Interactive Web UI**: Real-time optimization with visualizations
- **CLI Tools**: Automation and batch processing support

## 📊 Expected Performance
- **Standard MILP + Greedy**: < 5 seconds (daily operations)
- **Enhanced MILP**: 1-30 seconds (balanced optimization)
- **Genetic Algorithm**: 5-60 seconds (large problems)

## 🧪 Test Installation
```bash
# Run test suite
python -m pytest tests/

# Quick functionality test
python src/main.py --quiet
```

## 📁 Project Structure
```
Vehicle_Router/
├── app/streamlit_app.py      # Web application
├── src/main.py               # CLI optimization
├── src/comparison.py         # Method comparison
├── vehicle_router/           # Core optimization library
├── requirements.txt          # Dependencies
└── README.md                 # Detailed documentation
```

## 🆘 Troubleshooting
- **Import errors**: Ensure `pip install -e .` was run
- **Streamlit issues**: Try `pip install --upgrade streamlit`
- **Solver errors**: PuLP's CBC solver installs automatically
- **Distance calculation**: Requires internet for real-world distances

For detailed documentation, see [README.md](README.md)
