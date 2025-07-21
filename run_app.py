#!/usr/bin/env python3
"""
Vehicle Router - App Runner Script

This script automatically installs requirements and runs the Streamlit application.
It handles the complete setup and launch process for the Vehicle Router web app.

Usage:
    python run_app.py
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🚛 VEHICLE ROUTER OPTIMIZER")
    print("=" * 60)
    print("Setting up and launching the Streamlit web application...")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    print()

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        print("   Make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ All packages installed successfully")
        print()
        
    except subprocess.CalledProcessError as e:
        print("❌ Error installing packages:")
        print(f"   {e.stderr}")
        print("\n💡 Try running manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def check_streamlit_app():
    """Check if the Streamlit app file exists"""
    print("🔍 Checking Streamlit application...")
    
    app_file = Path("app/streamlit_app.py")
    
    if not app_file.exists():
        print("❌ Error: Streamlit app not found at app/streamlit_app.py")
        print("   Make sure the application files are in the correct location.")
        sys.exit(1)
    
    print("✅ Streamlit application found")
    print()

def run_streamlit_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Vehicle Router Streamlit App...")
    print()
    print("📋 Instructions:")
    print("   1. The app will open in your default web browser")
    print("   2. Click 'Load Example Data' in the sidebar to get started")
    print("   3. Run optimization and explore the results")
    print("   4. Press Ctrl+C in this terminal to stop the app")
    print()
    print("🌐 Starting Streamlit server...")
    print("-" * 40)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        print("Thank you for using Vehicle Router Optimizer!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit app: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run app/streamlit_app.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main execution function"""
    try:
        # Print banner
        print_banner()
        
        # Check Python version
        check_python_version()
        
        # Install requirements
        install_requirements()
        
        # Check app exists
        check_streamlit_app()
        
        # Run the app
        run_streamlit_app()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()