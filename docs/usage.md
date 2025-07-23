# Vehicle Router - Complete Usage Guide

This comprehensive guide covers all aspects of using the Vehicle Router application, from basic usage to advanced configuration and integration patterns.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Distance Calculation Methods](#distance-calculation-methods)
3. [Streamlit Web Application](#streamlit-web-application)
4. [Command Line Interface](#command-line-interface)
5. [Programmatic Usage](#programmatic-usage)
6. [Data Formats](#data-formats)
7. [Configuration Options](#configuration-options)
8. [Output Analysis](#output-analysis)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Basic Usage (30 seconds)

```bash
# 1. Install and setup
pip install -r requirements.txt
pip install -e .

# 2. Run with example data (simulated distances)
python src/main.py

# 3. Launch web interface
streamlit run app/streamlit_app.py

# 4. Compare all methods
python src/comparison.py
```

### Quick Results Overview:
```
Standard MILP + Greedy: 2 trucks, €2500, 21.0 km (0.08s)
Enhanced MILP:          2 trucks, €2500, 22.8 km (0.09s) 
Genetic Algorithm:      2 trucks, €2500, 19.0 km (0.32s)
```

## 🌍 Distance Calculation Methods

The Vehicle Router supports two distance calculation approaches with seamless switching between them.

### Simulated Distances (Default)

**When to Use:**
- Development and testing
- Offline environments
- Quick algorithm comparisons
- Resource-constrained environments

**Characteristics:**
- **Speed**: Instantaneous
- **Accuracy**: Mathematical approximation
- **Dependencies**: None
- **Formula**: `|postal_code_1 - postal_code_2| * 1.0 km`

### Real-World Distances (Advanced)

**When to Use:**
- Production routing applications
- Geographic accuracy requirements
- Realistic distance estimation
- Route quality analysis

**Characteristics:**
- **Speed**: ~0.5s per postal code (with geocoding)
- **Accuracy**: Geographic coordinates + great-circle distance
- **Dependencies**: Internet connection
- **Data Source**: OpenStreetMap Nominatim API

**Technical Details:**
```python
# Process: Postal Code → Coordinates → Distance
"08020" → (41.4206, 2.2016) → Haversine calculation → 1.3 km to "08027"
"08027" → (41.4216, 2.1859) → Haversine calculation
```

**Distance Accuracy Examples (Barcelona):**
```
Route               Simulated    Real-World    Improvement
08020 → 08027       7.0 km       1.3 km        81% more accurate
08020 → 08028       8.0 km       7.7 km        4% more accurate
08027 → 08028       1.0 km       6.8 km        Captures actual geography
```

## 🖥️ Streamlit Web Application

### Launch and Access
```bash
# Standard launch
streamlit run app/streamlit_app.py

# Custom port and configuration
streamlit run app/streamlit_app.py --server.port 8502

# Background launch
nohup streamlit run app/streamlit_app.py &
```

### Application Workflow

#### 1. Data Loading
**Example Data (Recommended for Testing):**
- Click "Load Example Data" in sidebar
- Automatically generates 5 orders and 5 trucks
- Creates distance matrix based on current distance setting

**Custom Data Upload:**
- Upload orders CSV: `order_id`, `postal_code`, `volume`
- Upload trucks CSV: `truck_id`, `capacity`, `cost`  
- Validates data format and generates compatible distance matrix

#### 2. Distance Method Configuration
```
Toggle: 🌍 Use Real-World Distances
- OFF: Simulated distances (instant calculation)
- ON: OpenStreetMap geocoding + Haversine distances
- Auto-reload: Distance matrix updates automatically when toggled
```

**Real-Time Update Process:**
1. User toggles distance method
2. App detects change and shows progress indicator
3. System reloads distance matrix with new method
4. Data exploration section updates automatically
5. Previous optimization results are cleared (re-run required)

#### 3. Method Selection
**Available Methods (configurable):**
- **📊 Standard MILP + Greedy**: Fast, cost-optimal with route optimization
- **🚀 Enhanced MILP**: Multi-objective optimization (hidden by default)
- **🧬 Genetic Algorithm**: Evolutionary approach for large problems

**Method-Specific Parameters:**

*Enhanced MILP:*
- Cost Weight: 0.0-1.0 (default: 0.6)
- Distance Weight: 0.0-1.0 (default: 0.4)
- Auto-normalized to sum = 1.0

*Genetic Algorithm:*
- Population Size: 20-100 (default: 50)
- Max Generations: 50-300 (default: 100)
- Mutation Rate: 5-30% (default: 10%)
- Fixed 50/50 cost-distance weighting

#### 4. Optimization Execution
```
Click: 🚀 Run Optimization
- Progress tracking with status updates
- Real-time log display during execution
- Method-specific progress indicators (GA generations, MILP solver status)
- Automatic validation of solution upon completion
```

#### 5. Results Analysis

**Solution Summary:**
- Key metrics: Total Cost, Trucks Used (X/Y), Orders Assigned (X/Y), Total KM
- Detailed route information with sequences and distances
- Truck utilization percentages and capacity analysis

**Interactive Visualizations:**
- **Route Maps**: Individual truck routes with postal code labels
- **Distance Heatmap**: Interactive matrix showing all pairwise distances
- **Cost Breakdown**: Truck-wise cost analysis with utilization
- **Performance Charts**: Method comparison and efficiency metrics

**Documentation Section:**
- Method-specific technical documentation
- Only appears after optimization completion
- Dynamic content based on selected method
- Mathematical foundations and algorithmic details

#### 6. Data Export
- **Excel Reports**: Comprehensive solution data with multiple sheets
- **CSV Files**: Orders, trucks, assignments, routes, and utilization
- **Solution Summary**: Formatted text report with key insights

### Configuration Management

**Model Availability Configuration:**
```python
# In app/streamlit_app.py
AVAILABLE_MODELS = {
    'standard': {'name': '📊 Standard MILP + Greedy', 'enabled': True},
    'enhanced': {'name': '🚀 Enhanced MILP', 'enabled': False},  # Hidden
    'genetic': {'name': '🧬 Genetic Algorithm', 'enabled': True}
}
```

**Alternative: .streamlit/config.toml**
```toml
[app.models.standard]
name = "📊 Standard MILP + Greedy"
enabled = true

[app.models.enhanced]
name = "🚀 Enhanced MILP" 
enabled = false

[app.models.genetic]
name = "🧬 Genetic Algorithm"
enabled = true
```

## 💻 Command Line Interface

### Main Optimization Script (src/main.py)

**Basic Usage:**
```bash
# Default: Standard MILP + Greedy with simulated distances
python src/main.py

# All optimization methods with real distances
python src/main.py --optimizer standard --real-distances
python src/main.py --optimizer enhanced --real-distances  
python src/main.py --optimizer genetic --real-distances

# Custom depot location
python src/main.py --depot 08025 --real-distances

# Quiet mode (reduced output)
python src/main.py --optimizer genetic --real-distances --quiet
```

**Available Options:**
```
--optimizer {standard,enhanced,genetic}    Optimization method (default: standard)
--depot POSTAL_CODE                       Depot location (default: 08020)
--real-distances                          Use OpenStreetMap distances (default: simulated)
--quiet                                   Reduce output verbosity
--help                                    Show help message
```

**Execution Flow:**
1. **Data Generation**: Creates example orders, trucks, and distance matrix
2. **Optimization**: Runs selected method with progress tracking
3. **Validation**: Comprehensive solution validation
4. **Visualization**: Creates route maps, cost analysis, and utilization charts
5. **Export**: Saves results to CSV files and generates plots
6. **Summary**: Displays detailed results with method-specific metrics

**Example Output (Genetic Algorithm with Real Distances):**
```
🏆 VEHICLE ROUTER OPTIMIZATION RESULT
============================================================

📊 CONFIGURATION:
  Method: Genetic Algorithm
  Depot Location: 08020
  Execution Time: 0.35s

🎯 SOLUTION SUMMARY:
✅ SUCCESS: 2 trucks, €2500, 12.2 km

📈 KEY METRICS:
💰 Total Cost: €2500
📏 Total Distance: 12.2 km  
🚚 Trucks Used: 2
📦 Orders Delivered: 5
📊 Average Utilization: 100.0%
🧬 GA Generations: 19
🏅 GA Final Fitness: 0.301277

🔬 DETAILED RESULTS:
🚛 Truck 1: 4 orders → ['C', 'D', 'A', 'E']
   📍 Route: 08020 → 08027 → 08031 → 08029 → 08028
   📏 Distance: 10.4 km
   📦 Utilization: 100.0/100.0 m³ (100.0%)

🚛 Truck 2: 1 orders → ['B']
   📍 Route: 08020 → 08030
   📏 Distance: 1.8 km  
   📦 Utilization: 50.0/50.0 m³ (100.0%)
==================================================
```

### Comparison Script (src/comparison.py)

**Purpose**: Compare all three optimization methods with comprehensive analysis and recommendations.

**Basic Usage:**
```bash
# Compare all methods with simulated distances
python src/comparison.py

# Compare with real-world distances
python src/comparison.py --real-distances

# Quick comparison with timeout and minimal output
python src/comparison.py --real-distances --timeout 30 --quiet

# Extended analysis with custom GA parameters
python src/comparison.py --real-distances --ga-population 100 --ga-generations 200
```

**Available Options:**
```
--timeout SECONDS              Solver timeout per method (default: 60)
--depot-return                 Force trucks to return to depot
--ga-population SIZE           GA population size (default: 50)
--ga-generations COUNT         GA max generations (default: 100)  
--ga-mutation RATE            GA mutation rate (default: 0.1)
--real-distances              Use OpenStreetMap distances
--quiet                       Reduce output verbosity
--help                        Show help message
```

**Comprehensive Output:**
```
🏆 VEHICLE ROUTER OPTIMIZATION METHODS COMPARISON
======================================================================

📊 TEST CONFIGURATION:
  Problem Size: 5 orders, 5 trucks
  Depot Location: 08020  
  Depot Return: False
  Solver Timeout: 30s
  Distance Method: Real-World (OpenStreetMap)

🎯 OPTIMIZATION RESULTS:
----------------------------------------------------------------------
METHOD                    TRUCKS   COST       DISTANCE     TIME    
----------------------------------------------------------------------
Standard MILP + Greedy    [1, 2]   €2500      12.2 km      0.08s   
Enhanced MILP             [1, 2]   €2500      13.9 km      0.09s   
Genetic Algorithm         [1, 2]   €2500      12.2 km      0.32s   

🎯 RECOMMENDATIONS:
==================================================
💰 Best for Cost Optimization: Standard MILP + Greedy
📏 Best for Distance Optimization: Standard MILP + Greedy  
⚡ Best for Speed: Enhanced MILP
⚖️ Best Overall Balance: Standard MILP + Greedy

💡 SPECIFIC RECOMMENDATIONS:
📏 For minimum distance: Use Standard MILP + Greedy (saves 1.7 km, 12.2% reduction)
⚡ For fastest results: Use Enhanced MILP (0.23s faster)

🏆 OVERALL WINNER: Standard MILP + Greedy (best balanced performance)
```

**Detailed Route Analysis:**
```
🔬 DETAILED METHOD ANALYSIS:

Standard MILP + Greedy:
  Status: ✅ SUCCESS
  Selected Trucks: [1, 2]
  Total Cost: €2500
  Total Distance: 12.2 km
  📋 Route Details:
    🚛 Truck 1: 4 orders → ['A', 'C', 'D', 'E']
       Route: 08020 → 08027 → 08031 → 08029 → 08028 (10.4 km)
    🚛 Truck 2: 1 orders → ['B']  
       Route: 08020 → 08030 (1.8 km)
```

## 🔧 Programmatic Usage

### Basic Integration

```python
from vehicle_router import VrpOptimizer, EnhancedVrpOptimizer, GeneticVrpOptimizer
from vehicle_router.data_generator import DataGenerator

# Generate test data
data_gen = DataGenerator(use_example_data=True)
orders_df = data_gen.generate_orders()
trucks_df = data_gen.generate_trucks()

# Distance matrix options
simulated_distances = data_gen.generate_distance_matrix(
    orders_df['postal_code'].tolist()
)

real_distances = data_gen.generate_distance_matrix(
    orders_df['postal_code'].tolist(),
    use_real_distances=True
)
```

### Method-Specific Usage

#### Standard MILP + Greedy
```python
# Initialize with greedy route optimization enabled
optimizer = VrpOptimizer(
    orders_df=orders_df,
    trucks_df=trucks_df, 
    distance_matrix=real_distances,
    depot_location='08020',
    depot_return=False,
    enable_greedy_routes=True
)

# Build and solve
optimizer.build_model()
success = optimizer.solve()

if success:
    solution = optimizer.get_solution()
    print(f"Total cost: €{solution['total_cost']}")
    print(f"Total distance: {solution['total_distance']:.1f} km")
    print(f"Selected trucks: {solution['selected_trucks']}")
    
    # Access detailed routes
    routes_df = solution['routes_df']
    for _, route in routes_df.iterrows():
        truck_id = route['truck_id']
        sequence = route['route_sequence']
        distance = route['route_distance']
        print(f"Truck {truck_id}: {' → '.join(sequence)} ({distance:.1f} km)")
```

#### Enhanced MILP
```python
# Initialize with multi-objective optimization
enhanced = EnhancedVrpOptimizer(
    orders_df=orders_df,
    trucks_df=trucks_df,
    distance_matrix=real_distances,
    depot_location='08020',
    depot_return=False
)

# Configure objective weights
enhanced.set_objective_weights(cost_weight=0.7, distance_weight=0.3)

# Solve with timeout
enhanced.build_model()
success = enhanced.solve(timeout=120)

if success:
    solution = enhanced.get_solution()
    print(f"Multi-objective value: {solution['objective_value']:.6f}")
    print(f"Cost component: €{solution['total_cost']}")
    print(f"Distance component: {solution['total_distance']:.1f} km")
```

#### Genetic Algorithm
```python
# Initialize evolutionary optimizer
genetic = GeneticVrpOptimizer(
    orders_df=orders_df,
    trucks_df=trucks_df,
    distance_matrix=real_distances,
    depot_location='08020',
    depot_return=False
)

# Configure algorithm parameters
genetic.set_parameters(
    population_size=100,
    max_generations=200, 
    mutation_rate=0.15
)

# Fixed balanced objective weights (automatic)
print("Objective weights: 50% cost, 50% distance (fixed)")

# Solve with progress tracking
success = genetic.solve(timeout=300)

if success:
    solution = genetic.get_solution()
    print(f"GA generations: {solution['ga_generations']}")
    print(f"Final fitness: {solution['ga_fitness']:.6f}")
    print(f"Solution diversity: {len(set(solution['selected_trucks']))} truck combinations explored")
```

### Advanced Usage Patterns

#### Batch Processing
```python
def batch_optimize(problem_list, method='standard'):
    """Process multiple routing problems"""
    results = []
    
    for problem_data in problem_list:
        orders_df, trucks_df = problem_data
        
        # Generate real-world distances
        postal_codes = orders_df['postal_code'].tolist()
        data_gen = DataGenerator(use_example_data=False)
        distance_matrix = data_gen.generate_distance_matrix(
            postal_codes, use_real_distances=True
        )
        
        # Optimize based on method
        if method == 'standard':
            optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
        elif method == 'enhanced':
            optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix)
            optimizer.set_objective_weights(0.6, 0.4)
        elif method == 'genetic':
            optimizer = GeneticVrpOptimizer(orders_df, trucks_df, distance_matrix)
            optimizer.set_parameters(population_size=50, max_generations=100)
        
        optimizer.build_model()
        success = optimizer.solve(timeout=120)
        
        if success:
            solution = optimizer.get_solution()
            results.append({
                'method': method,
                'cost': solution['total_cost'],
                'distance': solution['total_distance'],
                'trucks': len(solution['selected_trucks'])
            })
    
    return results
```

#### Method Comparison
```python
def compare_methods(orders_df, trucks_df, distance_matrix):
    """Compare all three methods on same problem"""
    methods = {
        'standard': VrpOptimizer,
        'enhanced': EnhancedVrpOptimizer, 
        'genetic': GeneticVrpOptimizer
    }
    
    results = {}
    
    for name, OptimizerClass in methods.items():
        optimizer = OptimizerClass(orders_df, trucks_df, distance_matrix)
        
        # Method-specific configuration
        if name == 'enhanced':
            optimizer.set_objective_weights(0.6, 0.4)
        elif name == 'genetic':
            optimizer.set_parameters(population_size=50, max_generations=100)
        
        optimizer.build_model()
        start_time = time.time()
        success = optimizer.solve(timeout=60)
        execution_time = time.time() - start_time
        
        if success:
            solution = optimizer.get_solution()
            results[name] = {
                'cost': solution['total_cost'],
                'distance': solution['total_distance'],
                'time': execution_time,
                'trucks': solution['selected_trucks']
            }
        else:
            results[name] = {'status': 'failed'}
    
    return results
```

## 📊 Data Formats

### Orders Data Format (CSV)
```csv
order_id,postal_code,volume
A,08031,25.0
B,08030,50.0
C,08029,25.0
D,08028,25.0
E,08027,25.0
```

**Requirements:**
- `order_id`: Unique identifier (string)
- `postal_code`: Valid postal code (string/numeric)
- `volume`: Order volume in m³ (positive float)

### Trucks Data Format (CSV)
```csv
truck_id,capacity,cost
1,100.0,1500.0
2,50.0,1000.0
3,25.0,500.0
4,25.0,1500.0
5,25.0,1000.0
```

**Requirements:**
- `truck_id`: Unique identifier (string/numeric)
- `capacity`: Vehicle capacity in m³ (positive float)
- `cost`: Operational cost in currency units (positive float)

### Solution Output Format

**Solution Dictionary Structure:**
```python
solution = {
    'total_cost': 2500.0,                    # Total operational cost
    'total_distance': 12.2,                  # Total travel distance (km)
    'selected_trucks': [1, 2],               # List of used truck IDs
    'assignments_df': pd.DataFrame,          # Order-truck assignments
    'routes_df': pd.DataFrame,               # Detailed route information
    'costs_df': pd.DataFrame,                # Cost breakdown by truck
    'utilization_df': pd.DataFrame,          # Capacity utilization
    'objective_value': 0.301277,             # Method-specific objective
    'solve_time': 0.35,                      # Execution time (seconds)
    'method_specific': {                     # Method-dependent metrics
        'ga_generations': 19,                # Genetic algorithm only
        'ga_fitness': 0.301277,              # Genetic algorithm only
        'milp_variables': 180,               # Enhanced MILP only
        'milp_constraints': 90               # Enhanced MILP only
    }
}
```

**Routes DataFrame Structure:**
```python
routes_df = pd.DataFrame({
    'truck_id': [1, 2],
    'assigned_orders': [['A', 'C', 'D', 'E'], ['B']],
    'route_sequence': [['08020', '08027', '08031', '08029', '08028'], ['08020', '08030']],
    'route_distance': [10.4, 1.8],
    'truck_cost': [1500.0, 1000.0],
    'capacity_used': [100.0, 50.0],
    'utilization_pct': [100.0, 100.0]
})
```

## ⚙️ Configuration Options

### Global Configuration
```python
# src/main.py configuration
config = {
    'optimizer_type': 'genetic',
    'depot_location': '08020',
    'depot_return': False,
    'use_real_distances': True,
    'solver_timeout': 120,
    'save_plots': True,
    'plot_directory': 'output',
    'validation_enabled': True
}
```

### Method-Specific Parameters

**Standard MILP + Greedy:**
```python
VrpOptimizer(
    depot_return=False,              # Trucks return to depot
    enable_greedy_routes=True,       # Enable route optimization
    solver_timeout=60                # CBC solver timeout (seconds)
)
```

**Enhanced MILP:**
```python
EnhancedVrpOptimizer(
    depot_return=False               # Trucks return to depot
)
optimizer.set_objective_weights(
    cost_weight=0.6,                 # Weight for truck costs (0-1)
    distance_weight=0.4              # Weight for distances (0-1)
)
```

**Genetic Algorithm:**
```python
GeneticVrpOptimizer(
    depot_return=False               # Trucks return to depot
)
optimizer.set_parameters(
    population_size=50,              # Population size (20-100)
    max_generations=100,             # Max generations (50-300)
    mutation_rate=0.1,               # Mutation rate (0.05-0.3)
    elite_size=5                     # Elite preservation (2-10)
)
```

### Distance Calculation Configuration
```python
# Real-world distances with country specification
from vehicle_router.distance_calculator import DistanceCalculator

calculator = DistanceCalculator(country_code="ES")  # Spain
calculator = DistanceCalculator(country_code="FR")  # France
calculator = DistanceCalculator(country_code="DE")  # Germany

distance_matrix = calculator.calculate_distance_matrix(postal_codes)
```

## 📈 Output Analysis

### Performance Metrics

**Key Indicators:**
- **Total Cost**: Sum of operational costs for selected trucks
- **Total Distance**: Sum of travel distances for all routes
- **Truck Utilization**: Percentage of capacity used per truck
- **Orders Coverage**: Percentage of orders successfully assigned
- **Execution Time**: Algorithm runtime including distance calculation

**Efficiency Ratios:**
```python
cost_per_km = total_cost / total_distance
utilization_avg = sum(utilizations) / len(selected_trucks)  
distance_efficiency = theoretical_min_distance / actual_distance
```

### Solution Quality Assessment

**Cost Optimality:**
- Standard MILP: Guaranteed optimal for truck selection phase
- Enhanced MILP: Globally optimal for weighted multi-objective
- Genetic Algorithm: Near-optimal with solution diversity

**Route Quality:**
- Standard MILP + Greedy: Optimal sequences for assigned orders
- Enhanced MILP: Simultaneous optimization may find better overall routes
- Genetic Algorithm: Good routes with exploration of alternatives

**Robustness Indicators:**
- Solution diversity (different truck combinations)
- Sensitivity to parameter changes
- Consistency across multiple runs (for Genetic Algorithm)

### Comparative Analysis

**Multi-Method Results Interpretation:**
```python
def analyze_results(results_dict):
    """Analyze and compare optimization results"""
    best_cost = min(r['cost'] for r in results_dict.values())
    best_distance = min(r['distance'] for r in results_dict.values())
    fastest_time = min(r['time'] for r in results_dict.values())
    
    recommendations = {
        'cost_leader': min(results_dict.items(), key=lambda x: x[1]['cost']),
        'distance_leader': min(results_dict.items(), key=lambda x: x[1]['distance']),
        'speed_leader': min(results_dict.items(), key=lambda x: x[1]['time']),
        'balanced_score': {}
    }
    
    # Calculate balanced scores
    for method, result in results_dict.items():
        cost_ratio = result['cost'] / best_cost
        distance_ratio = result['distance'] / best_distance  
        time_ratio = result['time'] / fastest_time
        
        balanced_score = (cost_ratio + distance_ratio + time_ratio) / 3
        recommendations['balanced_score'][method] = balanced_score
    
    return recommendations
```

## 🚀 Performance Optimization

### For Large Problems (50+ orders)

**Method Selection:**
- Use Genetic Algorithm for scalability
- Avoid Enhanced MILP (computational limits)
- Consider Standard MILP with custom heuristics

**Parameter Tuning:**
```python
# Genetic Algorithm for large problems
genetic.set_parameters(
    population_size=100,         # Larger population for diversity
    max_generations=300,         # More generations for convergence
    mutation_rate=0.2,           # Higher mutation for exploration
    tournament_size=5            # Increased selection pressure
)
```

### For Real-Time Applications

**Speed Optimization:**
- Use Standard MILP + Greedy (fastest method)
- Use simulated distances (no network delay)
- Reduce solver timeout for faster results
- Cache distance matrices when possible

**Resource Management:**
```python
# Quick optimization configuration
optimizer = VrpOptimizer(
    enable_greedy_routes=False,  # Skip route optimization for speed
    solver_timeout=10            # Reduced timeout
)
```

### For High-Quality Routes

**Quality Optimization:**
- Use Enhanced MILP for globally optimal routes
- Use real-world distances for accuracy
- Increase solver timeout for complex problems
- Fine-tune objective weights

```python
# Quality-focused configuration
enhanced = EnhancedVrpOptimizer(...)
enhanced.set_objective_weights(cost_weight=0.3, distance_weight=0.7)  # Distance priority
success = enhanced.solve(timeout=300)  # Extended timeout
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Distance Calculation Issues

**Problem**: "Depot location not found in distance matrix"
```python
# Solution: Ensure depot is included in postal codes list
postal_codes = orders_df['postal_code'].tolist()
if depot_location not in postal_codes:
    postal_codes.append(depot_location)
distance_matrix = data_gen.generate_distance_matrix(postal_codes)
```

**Problem**: Real-world distances fail due to network issues
```python
# Solution: Enable fallback mechanism
try:
    distance_matrix = data_gen.generate_distance_matrix(
        postal_codes, use_real_distances=True
    )
except Exception as e:
    print(f"Real distances failed: {e}")
    print("Falling back to simulated distances")
    distance_matrix = data_gen.generate_distance_matrix(
        postal_codes, use_real_distances=False
    )
```

#### Optimization Failures

**Problem**: MILP solver timeout on large problems
```python
# Solution: Increase timeout or use alternative method
optimizer.solve(timeout=300)  # Increase timeout

# Or switch to Genetic Algorithm for large problems
genetic = GeneticVrpOptimizer(orders_df, trucks_df, distance_matrix)
genetic.solve(timeout=120)
```

**Problem**: No feasible solution found
```python
# Solution: Check capacity constraints
total_volume = orders_df['volume'].sum()
total_capacity = trucks_df['capacity'].sum()

if total_volume > total_capacity:
    print("Infeasible: Total order volume exceeds truck capacity")
    print(f"Orders need: {total_volume:.1f} m³")
    print(f"Trucks provide: {total_capacity:.1f} m³")
```

#### Performance Issues

**Problem**: Slow execution with real distances
```python
# Solution: Use simulated distances for development
distance_matrix = data_gen.generate_distance_matrix(
    postal_codes, use_real_distances=False  # Faster for testing
)

# Or cache real distances for repeated use
cached_matrix = distance_calculator.calculate_distance_matrix(postal_codes)
pickle.dump(cached_matrix, open('distance_cache.pkl', 'wb'))
```

**Problem**: Memory issues with Enhanced MILP
```python
# Solution: Reduce problem size or use Standard MILP
if len(orders_df) > 25:
    print("Large problem detected, using Standard MILP + Greedy")
    optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
else:
    optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix)
```

#### Data Format Issues

**Problem**: Invalid CSV format
```python
# Solution: Validate data before optimization
def validate_orders_data(orders_df):
    required_columns = ['order_id', 'postal_code', 'volume']
    if not all(col in orders_df.columns for col in required_columns):
        raise ValueError(f"Orders data must contain: {required_columns}")
    
    if orders_df['volume'].min() <= 0:
        raise ValueError("All order volumes must be positive")
    
    if orders_df['order_id'].duplicated().any():
        raise ValueError("Order IDs must be unique")

validate_orders_data(orders_df)
```

### Debug Mode

**Enable Detailed Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python src/main.py --optimizer genetic --real-distances  # No --quiet flag
```

**Solution Validation:**
```python
from vehicle_router.validation import SolutionValidator

validator = SolutionValidator(orders_df, trucks_df, distance_matrix)
is_valid, validation_report = validator.validate_solution(solution)

if not is_valid:
    print("Validation failed:")
    print(validation_report)
```

This comprehensive usage guide provides all the information needed to effectively utilize the Vehicle Router system across different use cases, from basic operations to advanced integration and troubleshooting.