#!/bin/bash

# Vehicle Router - App Runner Script (Unix/Linux/macOS)
# This script automatically installs requirements and runs the Streamlit application.

echo "============================================================"
echo "🚛 VEHICLE ROUTER OPTIMIZER"
echo "============================================================"
echo "Setting up and launching the Streamlit web application..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Error: Python is not installed or not in PATH"
        echo "   Please install Python 3.8+ and try again."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "🔍 Using Python: $PYTHON_CMD"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    echo "   Make sure you're running this script from the project root directory."
    exit 1
fi

# Install requirements
echo ""
echo "📦 Installing required packages..."
$PYTHON_CMD -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error installing packages"
    echo "💡 Try running manually: pip install -r requirements.txt"
    exit 1
fi

echo "✅ All packages installed successfully"

# Check if Streamlit app exists
if [ ! -f "app/streamlit_app.py" ]; then
    echo "❌ Error: Streamlit app not found at app/streamlit_app.py"
    echo "   Make sure the application files are in the correct location."
    exit 1
fi

echo "✅ Streamlit application found"
echo ""

# Launch the app
echo "🚀 Launching Vehicle Router Streamlit App..."
echo ""
echo "📋 Instructions:"
echo "   1. The app will open in your default web browser"
echo "   2. Click 'Load Example Data' in the sidebar to get started"
echo "   3. Run optimization and explore the results"
echo "   4. Press Ctrl+C in this terminal to stop the app"
echo ""
echo "🌐 Starting Streamlit server..."
echo "----------------------------------------"

$PYTHON_CMD -m streamlit run app/streamlit_app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false

echo ""
echo "🛑 Application stopped"
echo "Thank you for using Vehicle Router Optimizer!"