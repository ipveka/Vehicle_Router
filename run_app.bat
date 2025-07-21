@echo off
REM Vehicle Router - App Runner Script (Windows)
REM This script automatically installs requirements and runs the Streamlit application.

echo ============================================================
echo 🚛 VEHICLE ROUTER OPTIMIZER
echo ============================================================
echo Setting up and launching the Streamlit web application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo    Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo 🔍 Checking Python installation...
python -c "import sys; print(f'✅ Python version: {sys.version.split()[0]}')"

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ❌ Error: requirements.txt not found
    echo    Make sure you're running this script from the project root directory.
    pause
    exit /b 1
)

REM Install requirements
echo.
echo 📦 Installing required packages...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Error installing packages
    echo 💡 Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ All packages installed successfully

REM Check if Streamlit app exists
if not exist "app\streamlit_app.py" (
    echo ❌ Error: Streamlit app not found at app\streamlit_app.py
    echo    Make sure the application files are in the correct location.
    pause
    exit /b 1
)

echo ✅ Streamlit application found
echo.

REM Launch the app
echo 🚀 Launching Vehicle Router Streamlit App...
echo.
echo 📋 Instructions:
echo    1. The app will open in your default web browser
echo    2. Click 'Load Example Data' in the sidebar to get started
echo    3. Run optimization and explore the results
echo    4. Press Ctrl+C in this terminal to stop the app
echo.
echo 🌐 Starting Streamlit server...
echo ----------------------------------------

python -m streamlit run app/streamlit_app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false

echo.
echo 🛑 Application stopped
echo Thank you for using Vehicle Router Optimizer!
pause