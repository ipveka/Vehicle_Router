# Vehicle Router Optimization Methods

This document explains the three optimization methods available in the Vehicle Router.

## Overview

The Vehicle Router includes three different optimization approaches for solving the Vehicle Routing Problem (VRP), each with different trade-offs between speed and solution quality.

## 📊 Method 1: Standard MILP + Greedy (Default)

### Summary
Two-phase hybrid optimization combining cost-optimal truck selection with distance-optimal route sequences.

### How It Works
1. **Phase 1 (MILP)**: Finds the minimum-cost combination of trucks that can deliver all orders
2. **Phase 2 (Greedy)**: Tests all possible route permutations to minimize travel distances for each truck

### Mathematical Foundation
- **Decision Variables**: Binary variables for truck selection and order assignment
- **Objective Function**: Minimize total truck operational costs
- **Constraints**: Order assignment uniqueness, capacity limits, truck usage logic
- **Solver**: CBC with 60-second timeout

### Performance Characteristics
- **Speed**: Fastest method (typically < 5 seconds total)
- **Memory**: Low usage (< 100MB)
- **Scalability**: Excellent for typical distributions (≤8 orders per truck)
- **Solution Quality**: Cost-optimal with distance-optimized routes

### Route Optimization
Tests all permutations for each truck's assigned orders:
- 2-4 orders: < 0.01s (2-24 permutations)
- 5-6 orders: < 0.2s (120-720 permutations)
- 7-8 orders: < 5s (5,040-40,320 permutations)

### Use Cases
- Balanced cost-distance optimization
- Real-world logistics applications
- Moderate computational resources
- Quick results needed

## 🚀 Method 2: Enhanced MILP

### Summary
Multi-objective optimization that tries to balance both cost and distance in a single optimization step.

### How It Works
1. **Multi-Objective Formulation**: Combines truck costs and travel distances in weighted objective function
2. **Routing Variables**: Creates binary variables for every possible route segment between locations
3. **Flow Conservation**: Ensures trucks follow valid, continuous routes from depot through all assigned orders
4. **Simultaneous Optimization**: Optimizes truck selection, order assignment, and route planning in single model

### Mathematical Foundation
- **Variables**: Assignment variables, truck usage variables, route segment variables
- **Objective**: α × costs + β × distances (where α + β = 1)
- **Constraints**: Flow conservation, depot routing, capacity limits, subtour elimination
- **Complexity**: O(|orders| × |trucks| + |locations|² × |trucks|) variables

### Multi-Objective Configuration
- **Default Weights**: 60% cost focus + 40% distance focus
- **Customizable**: User can adjust weight balance via sliders
- **Normalization**: Scales objectives to ensure meaningful contribution

### Performance Characteristics
- **Speed**: Moderate (1-30 seconds depending on problem complexity)
- **Memory**: Higher usage (100-500MB for medium instances)
- **Scalability**: Good for small-medium problems (≤50 orders, ≤15 trucks)
- **Solution Quality**: Globally optimal for multi-objective function

### Use Cases
- High-quality solutions required
- Cost and distance equally important
- Computational resources available
- Small-medium problem size

## 🧬 Method 3: Genetic Algorithm

### Summary
Population-based optimization that evolves solutions over multiple generations to find good trade-offs between cost and distance.

### How It Works
1. **Population Initialization**: Creates diverse set of random feasible solutions
2. **Fitness Evaluation**: Scores each solution using weighted cost-distance objective function
3. **Selection Mechanism**: Tournament selection chooses parents based on fitness ranking
4. **Crossover Operation**: Order Crossover creates offspring by combining parent solutions
5. **Mutation Process**: Adaptive assignment mutation explores new solution neighborhoods
6. **Constraint Repair**: Ensures all generated solutions remain feasible
7. **Evolution Loop**: Iterates through generations until convergence or time limit

### Genetic Operators
- **Selection**: Tournament selection with configurable tournament size
- **Crossover**: Order Crossover preserving route structure
- **Mutation**: Adaptive assignment mutation with configurable rate
- **Replacement**: Elitist strategy maintaining best solutions

### Algorithm Parameters
- **Population Size**: 20-100 individuals (default: 50)
- **Generations**: 50-300 iterations (default: 100)
- **Mutation Rate**: 5-30% probability (default: 10%)
- **Elite Size**: Top solutions preserved each generation

### Performance Characteristics
- **Speed**: Fast initial convergence (often < 10 seconds)
- **Memory**: Moderate usage scales with population size
- **Scalability**: Excellent for large problems (100+ orders)
- **Solution Quality**: Generally good solutions with variety

### Multi-Objective Optimization
- Fitness combines normalized cost and distance with equal importance
- Fixed weighting: 50% cost + 50% distance for balanced optimization
- Automatically balances operational costs with travel efficiency

### Use Cases
- Complex large-scale problems
- Solution diversity required
- Traditional methods struggle with computational complexity
- Robust global optimization needed

## Method Comparison

| Aspect | Standard MILP + Greedy | Enhanced MILP | Genetic Algorithm |
|--------|----------------------|---------------|-------------------|
| **Solution Quality** | Cost-optimal + route-optimized | Optimal | Good |
| **Scalability** | Excellent | Limited | Excellent |
| **Multi-objective** | Sequential | Simultaneous | Simultaneous |
| **Solve Time** | Fast (< 5s) | Moderate (< 30s) | Configurable |
| **Memory Usage** | Low | Moderate | Moderate |
| **Global Search** | Limited | No | Yes |

## Selection Guidelines

### Choose Standard MILP + Greedy when:
- Both cost and distance matter
- Need quick results (< 5 seconds)
- Typical problem size (≤8 orders per truck)
- Most real-world logistics applications

### Choose Enhanced MILP when:
- You want the best possible solution quality
- Cost and distance are equally important
- Small-medium problems (≤50 orders)
- You don't mind waiting a bit longer

### Choose Genetic Algorithm when:
- Large problems (50+ orders)
- You want to explore different solutions
- Traditional methods are too slow
- You're okay with good (not perfect) solutions

## Configuration

### Standard MILP + Greedy
- Greedy route optimization: Always enabled
- Solver timeout: 60 seconds
- Depot return: Configurable (default: False)

### Enhanced MILP
- Cost weight: 0.6 (configurable via slider)
- Distance weight: 0.4 (configurable via slider)
- Solver timeout: 300 seconds
- Depot return: Configurable (default: False)

### Genetic Algorithm
- Population size: 50 (configurable: 20-100)
- Max generations: 100 (configurable: 50-300)
- Mutation rate: 0.1 (configurable: 0.05-0.3)
- Cost weight: 0.5 (fixed - balanced optimization)
- Distance weight: 0.5 (fixed - balanced optimization)
- Depot return: Configurable (default: False) 