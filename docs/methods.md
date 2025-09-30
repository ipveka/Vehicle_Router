# Vehicle Router Optimization Methods Guide

This comprehensive guide explains the three optimization methods and distance calculation approaches available in the Vehicle Router system, including mathematical foundations, performance characteristics, and practical applications.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Distance Calculation Methods](#distance-calculation-methods)
3. [Optimization Methods](#optimization-methods)
4. [Performance Comparison](#performance-comparison)
5. [Selection Guidelines](#selection-guidelines)

## Overview

The Vehicle Router implements three distinct optimization approaches for solving the Capacitated Vehicle Routing Problem (CVRP), each designed for different scenarios and requirements:

- **Standard MILP + Greedy**: Two-phase hybrid optimization for balanced cost-distance optimization
- **Enhanced MILP**: Single-phase multi-objective optimization for globally optimal solutions
- **Genetic Algorithm**: Evolutionary metaheuristic for large-scale problems and solution diversity

Additionally, the system supports two distance calculation methods:

- **Simulated Distances**: Mathematical approximation for rapid testing
- **Real-World Distances**: OpenStreetMap geocoding for geographic accuracy

## 🌍 Distance Calculation Methods

### Simulated Distances (Default)

**Method**: Mathematical calculation based on postal code numerical differences
**Formula**: `distance = |postal_code_1 - postal_code_2| * 1.0 km`

**Characteristics:**
- **Speed**: Instantaneous calculation
- **Dependencies**: None (works offline)
- **Accuracy**: Approximation suitable for testing and development
- **Use Cases**: Rapid prototyping, algorithm development, offline environments

**Example**:
```
08020 → 08027: |08020 - 08027| = 7 units = 7.0 km
08027 → 08030: |08027 - 08030| = 3 units = 3.0 km
```

### Real-World Distances (Advanced)

**Method**: OpenStreetMap Nominatim geocoding + Haversine great-circle distance calculation

**Process Flow:**
1. **Geocoding**: Postal codes → latitude/longitude coordinates via OpenStreetMap API
2. **Distance Calculation**: Haversine formula for accurate spherical distances
3. **Caching**: In-memory coordinate storage for efficiency
4. **Fallback**: Static calculation if geocoding fails

**Mathematical Foundation - Haversine Formula:**
```
a = sin²(Δφ/2) + cos φ₁ ⋅ cos φ₂ ⋅ sin²(Δλ/2)
c = 2 ⋅ atan2(√a, √(1−a))
d = R ⋅ c

Where:
φ = latitude, λ = longitude, R = Earth's radius (6,371 km)
Δφ = φ₂ - φ₁, Δλ = λ₂ - λ₁
```

**Technical Specifications:**
- **API Provider**: OpenStreetMap Nominatim (free service)
- **Rate Limiting**: 0.5 seconds between requests (respectful usage)
- **Timeout**: 10 seconds per geocoding request
- **Coverage**: Global (works with any country code)
- **Accuracy**: Great-circle distance (direct spherical distance)

**Performance Characteristics:**
- **Geocoding Time**: ~0.5-2 seconds per postal code
- **Network Dependency**: Requires internet connection
- **Caching**: Coordinates cached during session
- **Error Handling**: Graceful fallback to static calculation

**Distance Accuracy Comparison (Barcelona Example):**
```
Route               Simulated    Real-World    Geographic Insight
08020 → 08027       7.0 km       1.3 km        Close neighborhoods
08020 → 08028       8.0 km       7.7 km        Similar distance
08027 → 08028       1.0 km       6.8 km        Actual urban layout
08029 → 08031       2.0 km       5.2 km        Geographic barriers
```

## 📊 Optimization Methods

### 1. Standard MILP + Greedy (Hybrid Approach)

**Overview**: Two-phase optimization combining mathematical programming for cost optimization with exhaustive route sequence optimization.

#### Phase 1: Mixed Integer Linear Programming (MILP)

**Objective**: Find minimum-cost combination of trucks that can deliver all orders

**Mathematical Formulation:**
```
Minimize: Σⱼ (costⱼ × yⱼ)

Subject to:
Σⱼ xᵢⱼ = 1                    ∀i ∈ Orders     (each order assigned once)
Σᵢ (volumeᵢ × xᵢⱼ) ≤ capacityⱼ × yⱼ  ∀j ∈ Trucks  (capacity constraints)
xᵢⱼ ≤ yⱼ                      ∀i,j           (truck usage logic)
xᵢⱼ ∈ {0,1}, yⱼ ∈ {0,1}      ∀i,j           (binary variables)

Where:
xᵢⱼ = 1 if order i assigned to truck j, 0 otherwise
yⱼ = 1 if truck j is used, 0 otherwise
```

**Solver**: CBC (Coin-OR Branch and Cut) with 60-second timeout

#### Phase 2: Greedy Route Optimization

**Objective**: Find optimal delivery sequence for each truck to minimize travel distance

**Algorithm**: Exhaustive permutation testing
```python
For each truck with assigned orders:
    For each permutation of order locations:
        Calculate route distance: depot → order₁ → order₂ → ... → orderₙ
        Track minimum distance route
    Select optimal permutation
```

**Computational Complexity**:
- 2 orders: 2! = 2 permutations (<0.001s)
- 4 orders: 4! = 24 permutations (<0.01s)
- 6 orders: 6! = 720 permutations (<0.1s)
- 8 orders: 8! = 40,320 permutations (<5s)

**Performance Characteristics:**
- **Speed**: Fastest method (typically <5 seconds total)
- **Memory**: Low usage (<100MB)
- **Scalability**: Excellent for ≤8 orders per truck
- **Solution Quality**: Cost-optimal truck selection + distance-optimized routes
- **Deterministic**: Always produces same result for same input

**Use Cases:**
- Daily logistics operations requiring fast decisions
- Cost-sensitive applications with secondary distance optimization
- Moderate problem sizes with balanced objectives

### 2. Enhanced MILP (Multi-Objective Approach)

**Overview**: Simultaneous optimization of truck costs and travel distances in a single mathematical model with routing variables.

#### Mathematical Formulation

**Extended Variable Set:**
```
xᵢⱼ = 1 if order i assigned to truck j, 0 otherwise
yⱼ = 1 if truck j is used, 0 otherwise  
zₖₗⱼ = 1 if truck j travels from location k to location l, 0 otherwise
```

**Multi-Objective Function:**
```
Minimize: α × (Σⱼ costⱼ × yⱼ / max_cost) + β × (Σⱼₖₗ distₖₗ × zₖₗⱼ / max_distance)

Where: α + β = 1 (default: α=0.6, β=0.4)
```

**Enhanced Constraints:**
```
Flow Conservation:
Σₗ zₖₗⱼ = Σₗ zₗₖⱼ     ∀k,j    (what comes in, goes out)

Depot Constraints:
Σₗ z₍depot,l₎ⱼ = yⱼ    ∀j      (trucks start from depot)

Order Routing:
Σₖₗ zₖₗⱼ ≥ xᵢⱼ        ∀i,j    (assigned orders must be routed)

Subtour Elimination:
MTZ or flow-based constraints to prevent invalid subtours
```

**Computational Complexity**: O(|orders| × |trucks| + |locations|² × |trucks|) variables

**Performance Characteristics:**
- **Speed**: Medium execution time (1-30 seconds)
- **Memory**: Higher usage (100-500MB for medium instances)
- **Scalability**: Good for small-medium problems (≤50 orders, ≤15 trucks)
- **Solution Quality**: Globally optimal for multi-objective function
- **Flexibility**: Configurable cost-distance weight balance

**Use Cases:**
- Applications requiring optimal cost-distance balance
- Scenarios where route quality is critical
- Medium computational budgets with quality requirements

### 3. Genetic Algorithm (Evolutionary Approach)

**Overview**: Population-based metaheuristic using evolutionary principles to solve multi-objective VRP with solution diversity.

#### Algorithm Components

**Representation**: Each individual represents a complete solution as truck-order assignments
```
Individual = [truck_assignments, fitness_score]
truck_assignments = {truck_1: [order_A, order_C], truck_2: [order_B], ...}
```

**Population Initialization**: Generate diverse random feasible solutions
```python
For i in range(population_size):
    solution = randomly_assign_orders_to_trucks()
    repair_constraints(solution)  # Ensure capacity constraints
    population.append(solution)
```

**Fitness Function**: Multi-objective evaluation with fixed balanced weighting
```
fitness = 0.5 × (normalized_cost) + 0.5 × (normalized_distance)

normalized_cost = current_cost / max_possible_cost
normalized_distance = current_distance / max_possible_distance
```

**Genetic Operators:**

1. **Selection**: Tournament selection with configurable tournament size
```python
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x.fitness)
```

2. **Crossover**: Order Crossover (OX) preserving assignment structure
```python
def order_crossover(parent1, parent2):
    # Preserve order relationships while combining assignments
    offspring = combine_truck_assignments(parent1, parent2)
    return repair_constraints(offspring)
```

3. **Mutation**: Adaptive assignment mutation exploring neighborhoods
```python
def adaptive_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        # Move random order to different truck
        reassign_random_order(individual)
        repair_constraints(individual)
```

**Evolution Process:**
1. Initialize population with diverse solutions
2. Evaluate fitness for all individuals
3. Select parents using tournament selection
4. Create offspring through crossover
5. Apply mutation to introduce variation
6. Repair constraints to maintain feasibility
7. Replace population using elitist strategy
8. Repeat until convergence or time limit

**Algorithm Parameters:**
- **Population Size**: 20-100 individuals (default: 50)
- **Max Generations**: 50-300 iterations (default: 100)
- **Mutation Rate**: 5-30% probability (default: 10%)
- **Elite Size**: Top solutions preserved each generation (default: 5)
- **Tournament Size**: Selection pressure parameter (default: 3)

**Convergence Criteria:**
- No improvement for consecutive generations
- Maximum generations reached
- Time limit exceeded
- Target fitness achieved

**Performance Characteristics:**
- **Speed**: Fast initial convergence (often <20 generations, <10 seconds)
- **Memory**: Moderate usage scaling with population size
- **Scalability**: Excellent for large problems (100+ orders, 20+ trucks)
- **Solution Quality**: Generally good solutions with high diversity
- **Stochastic**: Different runs may produce different solutions

**Use Cases:**
- Large-scale routing problems beyond MILP capacity
- Applications requiring solution diversity and multiple alternatives
- Scenarios where near-optimal solutions are acceptable
- Complex problem variants not easily modeled in MILP

## 📈 Performance Comparison

### Computational Performance (5 orders, Barcelona postal codes)

| Method | Execution Time | Memory Usage | Solution Quality | Deterministic |
|--------|---------------|--------------|------------------|---------------|
| **Standard MILP + Greedy** | 0.08s | <50MB | Cost-optimal + route-optimized | Yes |
| **Enhanced MILP** | 0.09s | 100-200MB | Globally optimal multi-objective | Yes |
| **Genetic Algorithm** | 0.32s | 50-100MB | Generally good with diversity | No |

### Solution Quality Analysis

**Cost Optimization:**
- All methods achieve €2500 (optimal cost for this instance)
- Standard MILP guarantees cost optimality
- Enhanced MILP balances cost with distance
- Genetic Algorithm explores cost-distance trade-offs

**Distance Optimization:**
```
Method                    Simulated    Real-World    Improvement
Standard MILP + Greedy    21.0 km      12.2 km       42% better
Enhanced MILP             22.8 km      13.9 km       39% better  
Genetic Algorithm         19.0 km      12.2 km       36% better
```

**Real-World Distance Impact:**
- Standard MILP + Greedy: 8.8 km improvement (42% reduction)
- Enhanced MILP: 8.9 km improvement (39% reduction)
- Genetic Algorithm: 6.8 km improvement (36% reduction)

### Scalability Analysis

| Problem Size | Standard MILP | Enhanced MILP | Genetic Algorithm |
|-------------|---------------|---------------|-------------------|
| ≤10 orders | <1s (optimal) | <5s (optimal) | <10s (near-optimal) |
| 11-25 orders | <5s (optimal) | 5-30s (optimal) | 10-30s (good) |
| 26-50 orders | 5-15s (may timeout) | 30-120s (may timeout) | 20-60s (good) |
| 51-100 orders | 10-60s (heuristic) | Not recommended | 30-120s (good) |
| 100+ orders | Custom heuristics | Not feasible | 60-300s (acceptable) |

## 🎯 Selection Guidelines

### Choose Standard MILP + Greedy When:
- **Priority**: Balanced cost-distance optimization
- **Timeline**: Quick results needed (<5 seconds)
- **Problem Size**: Small to medium (≤50 orders)
- **Requirements**: Cost optimality guaranteed
- **Resources**: Limited computational resources
- **Application**: Daily operational routing

### Choose Enhanced MILP When:
- **Priority**: Globally optimal multi-objective solution
- **Timeline**: Medium computation budget (30-120 seconds)
- **Problem Size**: Small to medium (≤25 orders for reliability)
- **Requirements**: Optimal cost-distance balance
- **Resources**: Adequate computational resources
- **Application**: High-quality route planning

### Choose Genetic Algorithm When:
- **Priority**: Solution diversity and exploration
- **Timeline**: Flexible computation budget (60-300 seconds)
- **Problem Size**: Large problems (50+ orders)
- **Requirements**: Good solutions acceptable
- **Resources**: Scalable computational resources
- **Application**: Large-scale logistics, complex variants

### Distance Method Selection:

**Use Simulated Distances When:**
- Development and testing phases
- Offline environments without internet
- Rapid prototyping and algorithm comparison
- Computational resources are extremely limited

**Use Real-World Distances When:**
- Production routing applications
- Geographic accuracy is important
- Realistic distance estimation required
- Internet connectivity is available

## 🔧 Configuration Examples

### CLI Usage with Real Distances:
```bash
# Standard MILP + Greedy with real distances
python src/main.py --real-distances

# Genetic algorithm comparison with all methods
python src/comparison.py --real-distances --timeout 60

# Enhanced MILP with custom depot
python src/main.py --optimizer enhanced --depot 08025 --real-distances
```

### Programmatic Configuration:
```python
# Real-world distance matrix
real_distances = data_gen.generate_distance_matrix(
    postal_codes, use_real_distances=True
)

# Standard MILP + Greedy
optimizer = VrpOptimizer(orders_df, trucks_df, real_distances, enable_greedy_routes=True)

# Enhanced MILP with custom weights
enhanced = EnhancedVrpOptimizer(orders_df, trucks_df, real_distances)
enhanced.set_objective_weights(cost_weight=0.7, distance_weight=0.3)

# Genetic Algorithm with custom parameters
genetic = GeneticVrpOptimizer(orders_df, trucks_df, real_distances)
genetic.set_parameters(population_size=100, max_generations=200, mutation_rate=0.15)
```

This comprehensive methodology guide provides the foundation for understanding and effectively utilizing the Vehicle Router optimization system across different problem types and requirements. 