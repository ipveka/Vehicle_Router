# Vehicle Router Optimization Methods

The Vehicle Router implements three optimization approaches for solving the Vehicle Routing Problem:

- **Standard MILP + Greedy**: Two-phase hybrid optimization
- **Enhanced MILP**: Multi-objective optimization  
- **Genetic Algorithm**: Evolutionary metaheuristic

## Distance Calculation Methods

### Simulated Distances (Default)
Mathematical calculation: `distance = |postal_code_1 - postal_code_2| * 1.0 km`

- **Speed**: Instantaneous
- **Use Cases**: Testing, development, offline environments

### Real-World Distances (Advanced)
OpenStreetMap geocoding + Haversine distance calculation

- **Speed**: ~2-5 seconds per postal code (first time), instant (cached)
- **Use Cases**: Production deployments, geographic accuracy

## Optimization Methods

### Standard MILP + Greedy
Two-phase approach combining MILP optimization with greedy route construction.

**Process:**
1. MILP optimization for truck selection and order assignment
2. Greedy algorithm for route optimization

**Best for:** Balanced cost-distance optimization, medium-scale problems

### Enhanced MILP
Single-phase multi-objective optimization with cost and distance weighting.

**Features:**
- Cost weight: Controls cost optimization importance
- Distance weight: Controls distance optimization importance
- Global optimization: Single-phase approach

**Best for:** Globally optimal solutions, cost-distance trade-offs

### Genetic Algorithm
Evolutionary metaheuristic using population-based optimization.

**Parameters:**
- Population size: Number of solutions per generation
- Max generations: Maximum iterations
- Mutation rate: Probability of random changes

**Best for:** Large-scale problems, solution diversity, complex constraints

## Performance Comparison

| Method | Speed | Optimality | Scalability | Use Case |
|--------|-------|------------|--------------|----------|
| Standard MILP + Greedy | Fast | Good | Medium | Balanced optimization |
| Enhanced MILP | Medium | Optimal | Medium | Global optimization |
| Genetic Algorithm | Slow | Good | High | Large-scale problems |

## Selection Guidelines

- **Development/Testing**: Standard MILP + Greedy
- **Production (Small-Medium)**: Enhanced MILP
- **Production (Large Scale)**: Genetic Algorithm
- **Offline Environment**: Simulated distances
- **Geographic Accuracy**: Real-world distances