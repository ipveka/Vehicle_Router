"""
Vehicle Router Source Package

This package contains the main application entry point for the Vehicle Routing
Problem optimization system. It provides the main workflow orchestration and
command-line interface for the vehicle router application.

Modules:
    main: Main application workflow and entry point

Usage:
    python -m src.main
    or
    python src/main.py
"""

__version__ = "1.0.0"
__author__ = "Vehicle Router Team"

# Import main application class for easy access
from .main import VehicleRouterApp, main

__all__ = [
    "VehicleRouterApp",
    "main",
    "__version__"
]