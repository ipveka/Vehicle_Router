"""
Vehicle Router App Configuration

Simple configuration for the Vehicle Router Streamlit application.
"""

from typing import Dict, Any

# =============================================================================
# OPTIMIZATION METHOD SELECTION
# =============================================================================

# The optimization method to use in the app
OPTIMIZATION_METHOD = 'standard'

# For backward compatibility with tests
DEFAULT_ALGORITHM = OPTIMIZATION_METHOD

# =============================================================================
# UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    'page_title': 'Vehicle Router Optimizer',
    'page_icon': '🚛',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

OPTIMIZATION_DEFAULTS = {
    'max_orders_per_truck': 3,
    'depot_return': False,
    'enable_greedy_routes': True,
    'use_real_distances': True,
    'validation_enabled': True,
    'solver_timeout': 300,
}

METHOD_PARAMS = {
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

DISTANCE_CONFIG = {
    'default_method': 'simulated',
    'country_code': 'ES',
    'geocoding_timeout': 5.0,
    'rate_limit_delay': 0.1,
    'cache_coordinates': True
}

# =============================================================================
# DEPOT CONFIGURATION
# =============================================================================

DEPOT_CONFIG = {
    'default_location': '08020',
}

# =============================================================================
# AVAILABLE MODELS CONFIGURATION
# =============================================================================

AVAILABLE_MODELS = {
    'standard': {
        'name': '📊 Standard MILP + Greedy',
        'help': 'Two-phase hybrid optimization: cost-optimal truck selection + route optimization',
        'enabled': True,
        'description': 'Uses mixed-integer linear programming with greedy route construction'
    },
    'enhanced': {
        'name': '🚀 Enhanced MILP',
        'help': 'Advanced MILP with multi-objective optimization',
        'enabled': True,
        'description': 'Enhanced MILP with cost and distance weighting'
    },
    'genetic': {
        'name': '🧬 Genetic Algorithm',
        'help': 'Evolutionary optimization approach',
        'enabled': True,
        'description': 'Genetic algorithm for complex optimization problems'
    }
}

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_enabled_models() -> list:
    """Get only enabled models"""
    return [k for k, v in AVAILABLE_MODELS.items() if v.get('enabled', True)]

def is_model_enabled(model_key: str) -> bool:
    """Check if a model is enabled"""
    return AVAILABLE_MODELS.get(model_key, {}).get('enabled', True)

def get_model_display_name(model_key: str) -> str:
    """Get display name for a model"""
    return AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)

def get_method_display_name(method: str = None) -> str:
    """Get display name for a method (defaults to current method)"""
    if method is None:
        method = OPTIMIZATION_METHOD
    return AVAILABLE_MODELS.get(method, {}).get('name', method)

def get_method_defaults(method: str) -> Dict[str, Any]:
    """Get default parameters for a method"""
    return METHOD_PARAMS.get(method, {})

def get_method_params(method: str = None) -> Dict[str, Any]:
    """Get parameters for a method (defaults to current method)"""
    if method is None:
        method = OPTIMIZATION_METHOD
    return METHOD_PARAMS.get(method, {})

# Method defaults for backward compatibility
METHOD_DEFAULTS = METHOD_PARAMS

def validate_config() -> Dict[str, Any]:
    """Validate the configuration"""
    issues = {'errors': [], 'warnings': [], 'info': []}
    
    # Validate optimization method
    if OPTIMIZATION_METHOD not in AVAILABLE_MODELS:
        issues['errors'].append(f"OPTIMIZATION_METHOD '{OPTIMIZATION_METHOD}' must be one of: {list(AVAILABLE_MODELS.keys())}")
    
    # Validate method parameters exist
    if OPTIMIZATION_METHOD in METHOD_PARAMS:
        method_params = METHOD_PARAMS[OPTIMIZATION_METHOD]
        
        # Validate genetic algorithm parameters
        if OPTIMIZATION_METHOD == 'genetic':
            if method_params.get('population_size', 0) < 10:
                issues['warnings'].append("Genetic Algorithm population_size should be 10 or more")
            if method_params.get('max_generations', 0) < 10:
                issues['warnings'].append("Genetic Algorithm max_generations should be 10 or more")
            mutation_rate = method_params.get('mutation_rate', 0)
            if not (0 <= mutation_rate <= 1):
                issues['errors'].append("Genetic Algorithm mutation_rate must be between 0 and 1")
        
        # Validate enhanced MILP weights
        elif OPTIMIZATION_METHOD == 'enhanced':
            cost_w = method_params.get('cost_weight', 0)
            dist_w = method_params.get('distance_weight', 0)
            if not (0 <= cost_w <= 1 and 0 <= dist_w <= 1):
                issues['errors'].append("Enhanced MILP cost_weight and distance_weight must be between 0 and 1")
            if abs(cost_w + dist_w - 1.0) > 1e-6 and (cost_w > 0 or dist_w > 0):
                issues['warnings'].append("Enhanced MILP cost_weight and distance_weight do not sum to 1. They will be normalized.")
    
    return issues

def get_config_summary() -> str:
    """Get a summary of the current configuration"""
    method_name = get_model_display_name(OPTIMIZATION_METHOD)
    defaults = OPTIMIZATION_DEFAULTS
    enabled_models = get_enabled_models()
    
    summary_parts = [
        f"Default Algorithm: {OPTIMIZATION_METHOD}",
        f"Method: {method_name}",
        f"Enabled Models: {', '.join(enabled_models)}",
        f"Distance Method: {DISTANCE_CONFIG['default_method']}",
        f"Max Orders per Truck: {defaults['max_orders_per_truck']}",
        f"Real Distances: {defaults['use_real_distances']}",
        f"Validation: {'Enabled' if defaults['validation_enabled'] else 'Disabled'}",
        f"Logging Level: INFO"
    ]
    
    return " | ".join(summary_parts)