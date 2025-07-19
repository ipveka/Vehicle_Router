"""
App Utils Package

This package contains utility modules for the Vehicle Router Streamlit application.
It provides modular components for data handling, optimization, visualization,
export functionality, and UI components.

Modules:
    data_handler: Data loading and management utilities
    optimization_runner: Optimization execution and management
    visualization_manager: Chart and plot creation utilities
    export_manager: Data export and report generation
    ui_components: Reusable UI components and widgets
"""

__version__ = "1.0.0"
__author__ = "Vehicle Router Team"

# Import main classes for easy access
from .data_handler import DataHandler
from .optimization_runner import OptimizationRunner
from .visualization_manager import VisualizationManager
from .export_manager import ExportManager
from .ui_components import UIComponents

__all__ = [
    "DataHandler",
    "OptimizationRunner", 
    "VisualizationManager",
    "ExportManager",
    "UIComponents"
]