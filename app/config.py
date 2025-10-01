"""
Vehicle Router App Configuration

This module contains all configurable parameters for the Vehicle Router Streamlit application.
Modify these settings to customize the app behavior without changing the main code.

Configuration Categories:
- Default Algorithm Selection
- Available Models and Features
- UI Settings and Labels
- Optimization Parameters
- Distance Calculation Settings
"""

from typing import Dict, Any, List

# =============================================================================
# DEFAULT ALGORITHM CONFIGURATION
# =============================================================================

# Default optimization method when the app starts
# Options: 'standard', 'genetic', 'enhanced'
DEFAULT_ALGORITHM = 'genetic'

# =============================================================================
# AVAILABLE MODELS CONFIGURATION
# =============================================================================

# Control which optimization models are available in the UI
# Set 'enabled': False to hide a model from the interface
AVAILABLE_MODELS = {
    'genetic': {
        'name': '🧬 Genetic Algorithm',
        'help': 'Evolutionary multi-objective optimization',
        'enabled': True,
        'description': 'Best for large problems and solution diversity exploration'
    },
    'standard': {
        'name': '📊 Standard MILP + Greedy',
        'help': 'Cost optimization with route enhancement',
        'enabled': True,
        'description': 'Fast and balanced optimization for daily operations'
    },
    'enhanced': {
        'name': '🚀 Enhanced MILP',
        'help': 'Advanced MILP with integrated routing',
        'enabled': False,  # Hidden by default (advanced users only)
        'description': 'Globally optimal multi-objective optimization'
    }
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# App appearance and behavior settings
UI_CONFIG = {
    'page_title': 'Vehicle Router Optimizer',
    'page_icon': '🚛',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'show_logs_by_default': False,
    'enable_progress_bars': True,
    'enable_animations': True
}

# =============================================================================
# OPTIMIZATION DEFAULTS
# =============================================================================

# Default optimization parameters
OPTIMIZATION_DEFAULTS = {
    'max_orders_per_truck': 3,
    'depot_return': False,
    'enable_greedy_routes': True,
    'use_real_distances': True,
    'validation_enabled': True,
    'solver_timeout': 300,
}

# Method-specific default parameters
METHOD_DEFAULTS = {
    'standard': {
        'solver_timeout': 60,
        'cost_weight': 1.0,
        'distance_weight': 0.0,
    },
    'enhanced': {
        'solver_timeout': 300,
        'cost_weight': 0.6,
        'distance_weight': 0.4,
    },
    'genetic': {
        'solver_timeout': 300,
        'cost_weight': 0.5,
        'distance_weight': 0.5,
        'population_size': 50,
        'max_generations': 100,
        'mutation_rate': 0.1,
    }
}

# =============================================================================
# DISTANCE CALCULATION CONFIGURATION
# =============================================================================

# Distance calculation settings
DISTANCE_CONFIG = {
    'default_method': 'real',  # 'real' or 'simulated'
    'country_code': 'ES',  # ISO country code for geocoding
    'rate_limit_delay': 0.5,  # Seconds between API calls
    'geocoding_timeout': 10,  # Seconds timeout for geocoding requests
    'enable_caching': True,
    'cache_duration': 3600,  # Cache duration in seconds (1 hour)
}

# =============================================================================
# DEPOT CONFIGURATION
# =============================================================================

# Default depot settings
DEPOT_CONFIG = {
    'default_location': '08020',
    'allow_custom_depot': True,
    'show_depot_selector': True,
}

# =============================================================================
# EXPORT AND VISUALIZATION CONFIGURATION
# =============================================================================

# Export and visualization settings
EXPORT_CONFIG = {
    'enable_excel_export': True,
    'enable_csv_export': True,
    'enable_detailed_reports': True,
    'default_export_format': 'excel',
    'include_visualizations': True,
    'save_plots_to_disk': False,
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging and monitoring settings
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'enable_file_logging': True,
    'enable_console_logging': False,
    'enable_performance_tracking': True,
    'log_rotation': True,
    'max_log_files': 10,
    'log_directory': 'logs/app',
}

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# Solution validation settings
VALIDATION_CONFIG = {
    'enable_solution_validation': True,
    'validate_capacity_constraints': True,
    'validate_order_assignment': True,
    'validate_route_feasibility': True,
    'strict_validation': False,
}

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Advanced configuration options
ADVANCED_CONFIG = {
    'enable_debug_mode': False,
    'show_technical_details': False,
    'enable_experimental_features': False,
    'memory_usage_warnings': True,
    'performance_monitoring': True,
}

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """
    Validate the configuration and return any issues found.
    
    Returns:
        Dict containing validation results and any warnings/errors
    """
    issues = {
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Validate default algorithm
    if DEFAULT_ALGORITHM not in AVAILABLE_MODELS:
        issues['errors'].append(f"Default algorithm '{DEFAULT_ALGORITHM}' not found in AVAILABLE_MODELS")
    elif not AVAILABLE_MODELS[DEFAULT_ALGORITHM]['enabled']:
        issues['warnings'].append(f"Default algorithm '{DEFAULT_ALGORITHM}' is disabled")
    
    # Validate that at least one model is enabled
    enabled_models = [k for k, v in AVAILABLE_MODELS.items() if v['enabled']]
    if not enabled_models:
        issues['errors'].append("No optimization models are enabled")
    
    # Validate method defaults
    for method in AVAILABLE_MODELS.keys():
        if method not in METHOD_DEFAULTS:
            issues['warnings'].append(f"No default parameters defined for method '{method}'")
    
    # Validate distance config
    if DISTANCE_CONFIG['default_method'] not in ['real', 'simulated']:
        issues['errors'].append("Distance method must be 'real' or 'simulated'")
    
    return issues

def get_config_summary() -> str:
    """
    Get a summary of the current configuration.
    
    Returns:
        String summary of key configuration settings
    """
    enabled_models = [k for k, v in AVAILABLE_MODELS.items() if v['enabled']]
    
    summary = f"""
Vehicle Router App Configuration Summary:
========================================

Default Algorithm: {DEFAULT_ALGORITHM}
Enabled Models: {', '.join(enabled_models)}
Distance Method: {DISTANCE_CONFIG['default_method']}
Max Orders per Truck: {OPTIMIZATION_DEFAULTS['max_orders_per_truck']}
Real Distances: {OPTIMIZATION_DEFAULTS['use_real_distances']}
Validation: {VALIDATION_CONFIG['enable_solution_validation']}
Logging Level: {LOGGING_CONFIG['log_level']}
    """
    
    return summary.strip()

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_enabled_models() -> List[str]:
    """Get list of enabled model keys."""
    return [k for k, v in AVAILABLE_MODELS.items() if v['enabled']]

def get_model_display_name(model_key: str) -> str:
    """Get display name for a model key."""
    return AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)

def is_model_enabled(model_key: str) -> bool:
    """Check if a model is enabled."""
    return AVAILABLE_MODELS.get(model_key, {}).get('enabled', False)

def get_method_defaults(method: str) -> Dict[str, Any]:
    """Get default parameters for a specific method."""
    return METHOD_DEFAULTS.get(method, {})

# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

# Example configurations for different use cases:

# For development/testing (fast, simple)
DEVELOPMENT_CONFIG = {
    'DEFAULT_ALGORITHM': 'standard',
    'AVAILABLE_MODELS': {
        'standard': {'enabled': True},
        'genetic': {'enabled': True},
        'enhanced': {'enabled': False}
    },
    'OPTIMIZATION_DEFAULTS': {
        'use_real_distances': False,  # Use simulated distances for speed
        'max_orders_per_truck': 5
    }
}

# For production (comprehensive, robust)
PRODUCTION_CONFIG = {
    'DEFAULT_ALGORITHM': 'genetic',
    'AVAILABLE_MODELS': {
        'standard': {'enabled': True},
        'genetic': {'enabled': True},
        'enhanced': {'enabled': True}
    },
    'OPTIMIZATION_DEFAULTS': {
        'use_real_distances': True,  # Use real-world distances
        'max_orders_per_truck': 3
    },
    'VALIDATION_CONFIG': {
        'strict_validation': True
    }
}

# For demonstration (show all features)
DEMO_CONFIG = {
    'DEFAULT_ALGORITHM': 'genetic',
    'AVAILABLE_MODELS': {
        'standard': {'enabled': True},
        'genetic': {'enabled': True},
        'enhanced': {'enabled': True}
    },
    'UI_CONFIG': {
        'show_logs_by_default': True
    }
}
