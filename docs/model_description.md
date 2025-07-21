# Mathematical Model Description

This document provides a comprehensive description of the Mixed Integer Linear Programming (MILP) formulation used by the Vehicle Router to solve the Vehicle Routing Problem with order assignment optimization.

## Problem Definition

The Vehicle Router solves the Capacitated Vehicle Routing Problem (CVRP) with two distinct optimization approaches:

### Standard Model
- **Objective**: Minimize total truck operational costs
- **Focus**: Optimal truck selection and order assignment
- **Complexity**: Simplified model suitable for cost-focused optimization

### Enhanced Model (NEW)
- **Objective**: Minimize weighted combination of truck costs and travel distances
- **Focus**: Comprehensive route optimization with distance minimization
- **Features**: Multi-objective optimization, route sequencing, depot constraints
- **Complexity**: Advanced model with routing variables and subtour elimination

### Problem Characteristics

- **Orders**: Each order has a specific volume requirement and delivery location
- **Trucks**: Each truck has a maximum capacity and associated operational cost
- **Distance Matrix**: Travel distances between all postal code locations
- **Depot**: Central location where trucks start and end their routes
- **Objectives**: Minimize costs and/or travel distances while satisfying all constraints
- **Constraints**: Capacity limits, order assignment requirements, route continuity, depot constraints

## Mathematical Formulation

### Sets and Indices

| Symbol | Description |
|--------|-------------|
| `I` | Set of orders (indexed by `i`) |
| `J` | Set of trucks (indexed by `j`) |
| `K` | Set of locations/postal codes (indexed by `k`) |

### Parameters

| Symbol | Description | Units |
|--------|-------------|-------|
| `v_i` | Volume of order `i` | m³ |
| `c_j` | Operational cost of truck `j` | € |
| `cap_j` | Capacity of truck `j` | m³ |
| `d_{k,l}` | Distance between locations `k` and `l` | km |
| `loc_i` | Location (postal code) of order `i` | - |

### Decision Variables

#### Standard Model
The standard MILP formulation uses two types of binary decision variables:

**Order Assignment Variables:**
```
x_{i,j} ∈ {0, 1}  ∀i ∈ I, ∀j ∈ J
```
- `x_{i,j} = 1` if order `i` is assigned to truck `j`
- `x_{i,j} = 0` otherwise

**Truck Usage Variables:**
```
y_j ∈ {0, 1}  ∀j ∈ J
```
- `y_j = 1` if truck `j` is used in the solution
- `y_j = 0` otherwise

#### Enhanced Model
The enhanced MILP formulation includes additional routing variables:

**Routing Variables:**
```
z_{j,k,l} ∈ {0, 1}  ∀j ∈ J, ∀k,l ∈ K, k ≠ l
```
- `z_{j,k,l} = 1` if truck `j` travels directly from location `k` to location `l`
- `z_{j,k,l} = 0` otherwise

### Objective Function

#### Standard Model
The standard model minimizes total truck operational costs:

```
minimize: ∑_{j ∈ J} c_j × y_j
```

This formulation focuses on truck selection costs and is suitable when fixed costs dominate.

#### Enhanced Model
The enhanced model minimizes a weighted combination of truck costs and travel distances:

```
minimize: α × (∑_{j ∈ J} c_j × y_j) + β × (∑_{j ∈ J} ∑_{k ∈ K} ∑_{l ∈ K} d_{k,l} × z_{j,k,l})
```

Where:
- `α` = cost weight (typically 0.6)
- `β` = distance weight (typically 0.4)
- `α + β = 1` for proper scaling
- `z_{j,k,l}` = routing variables indicating truck `j` travels from location `k` to `l`

This multi-objective approach balances cost efficiency with route optimization.

### Constraints

#### Standard Model Constraints

**1. Order Assignment Constraints**
Each order must be assigned to exactly one truck:

```
∑_{j ∈ J} x_{i,j} = 1  ∀i ∈ I
```

**Interpretation**: Every order is delivered exactly once.

**2. Capacity Constraints**
The total volume of orders assigned to each truck cannot exceed its capacity:

```
∑_{i ∈ I} v_i × x_{i,j} ≤ cap_j  ∀j ∈ J
```

**Interpretation**: No truck exceeds its capacity limit.

**3. Truck Usage Constraints**
If any order is assigned to a truck, the truck must be marked as used:

```
y_j ≥ x_{i,j}  ∀i ∈ I, ∀j ∈ J
```

**Interpretation**: Truck usage variables are properly linked to order assignments.

#### Enhanced Model Additional Constraints

**4. Route Continuity Constraints**
For each truck and location, flow conservation must be maintained:

```
∑_{l ∈ K, l ≠ k} z_{j,l,k} = ∑_{l ∈ K, l ≠ k} z_{j,k,l} = ∑_{i ∈ I: loc_i = k} x_{i,j}  ∀j ∈ J, ∀k ∈ K
```

**Interpretation**: If a truck visits a location, it must arrive and depart, serving all assigned orders.

**5. Depot Constraints**
Each used truck must start and end at the depot:

```
∑_{k ∈ K, k ≠ depot} z_{j,depot,k} = y_j  ∀j ∈ J
∑_{k ∈ K, k ≠ depot} z_{j,k,depot} = y_j  ∀j ∈ J
```

**Interpretation**: Used trucks leave and return to the depot exactly once.

### Complete MILP Formulation

```
minimize: ∑_{j ∈ J} c_j × y_j

subject to:
    ∑_{j ∈ J} x_{i,j} = 1                    ∀i ∈ I     (Order assignment)
    ∑_{i ∈ I} v_i × x_{i,j} ≤ cap_j         ∀j ∈ J     (Capacity limits)
    y_j ≥ x_{i,j}                           ∀i ∈ I, ∀j ∈ J  (Truck usage)
    x_{i,j} ∈ {0, 1}                        ∀i ∈ I, ∀j ∈ J  (Binary variables)
    y_j ∈ {0, 1}                            ∀j ∈ J     (Binary variables)
```

## Example Application

### Problem Instance
Consider the built-in example with:
- **Orders**: A(75m³), B(50m³), C(25m³), D(25m³), E(25m³)
- **Trucks**: T1(100m³,€1500), T2(50m³,€1000), T3(25m³,€500), T4(25m³,€1500), T5(25m³,€1000)

### Variable Instantiation
- **Order assignment variables**: `x_{A,1}, x_{A,2}, ..., x_{E,5}` (25 variables)
- **Truck usage variables**: `y_1, y_2, y_3, y_4, y_5` (5 variables)
- **Total variables**: 30 binary variables

### Constraint Instantiation
- **Order assignment**: 5 constraints (one per order)
- **Capacity**: 5 constraints (one per truck)
- **Truck usage**: 25 constraints (one per order-truck pair)
- **Total constraints**: 35 constraints

### Optimal Solution
The solver finds:
- **Selected trucks**: [1, 2, 3, 5]
- **Assignments**: A→T1, E→T1, B→T2, C→T3, D→T5
- **Total cost**: €4000 (1500+1000+500+1000)

## Model Properties

### Complexity Analysis
- **Variables**: O(|I| × |J|) binary variables
- **Constraints**: O(|I| × |J|) constraints
- **Problem class**: NP-hard (variant of bin packing)

### Solution Quality
- **Optimality**: MILP guarantees optimal solutions for small-medium instances
- **Scalability**: Suitable for problems with up to ~100 orders and ~20 trucks
- **Solver performance**: Typically solves in seconds for practical instances

### Model Extensions

#### 1. Distance-Based Routing
Add route variables and distance costs:
```
z_{j,k,l} ∈ {0, 1}  ∀j ∈ J, ∀k,l ∈ K
```

#### 2. Time Windows
Add time constraints for order deliveries:
```
t_i^{start} ≤ t_i ≤ t_i^{end}  ∀i ∈ I
```

#### 3. Multiple Depots
Extend to multiple starting locations:
```
∑_{d ∈ D} w_{j,d} = y_j  ∀j ∈ J
```

#### 4. Heterogeneous Fleet
Different truck types with varying capabilities:
```
∑_{t ∈ T} u_{j,t} = 1  ∀j ∈ J
```

## Implementation Details

### PuLP Integration
The Vehicle Router uses the PuLP library to implement this MILP formulation:

```python
# Create decision variables
x = {}
for i in orders:
    for j in trucks:
        x[i,j] = pulp.LpVariable(f"assign_{i}_to_{j}", cat='Binary')

y = {}
for j in trucks:
    y[j] = pulp.LpVariable(f"use_truck_{j}", cat='Binary')

# Set objective
model += pulp.lpSum([cost[j] * y[j] for j in trucks])

# Add constraints
for i in orders:
    model += pulp.lpSum([x[i,j] for j in trucks]) == 1

for j in trucks:
    model += pulp.lpSum([volume[i] * x[i,j] for i in orders]) <= capacity[j]
    
for i in orders:
    for j in trucks:
        model += y[j] >= x[i,j]
```

### Solver Configuration
- **Default solver**: CBC (Coin-or Branch and Cut)
- **Alternative solvers**: CPLEX, Gurobi, GLPK
- **Solver parameters**: Configurable timeout, gap tolerance, threads

### Performance Optimization
- **Preprocessing**: Remove dominated trucks, infeasible assignments
- **Variable fixing**: Fix variables based on problem structure
- **Cut generation**: Add valid inequalities to strengthen formulation
- **Heuristic initialization**: Provide good starting solutions

## Validation and Testing

### Solution Verification
The model includes comprehensive validation:

1. **Feasibility Check**: Verify all constraints are satisfied
2. **Optimality Check**: Confirm objective value matches solver output
3. **Consistency Check**: Ensure variable values are logically consistent

### Test Cases
- **Small instances**: 3-10 orders, 2-5 trucks
- **Medium instances**: 20-50 orders, 5-15 trucks
- **Edge cases**: Tight capacity, high cost variation, infeasible instances

### Benchmarking
Compare against:
- **Heuristic methods**: Greedy assignment, local search
- **Metaheuristics**: Genetic algorithms, simulated annealing
- **Commercial solvers**: CPLEX, Gurobi performance comparison

## Limitations and Assumptions

### Current Limitations
1. **Route optimization**: Simplified distance modeling
2. **Time constraints**: No delivery time windows
3. **Dynamic updates**: Static problem formulation
4. **Stochastic elements**: Deterministic demand and travel times

### Key Assumptions
1. **Truck availability**: All trucks are available when needed
2. **Order splitting**: Orders cannot be split between trucks
3. **Capacity homogeneity**: Single capacity constraint per truck
4. **Cost structure**: Fixed costs only (no variable distance costs)

### Future Enhancements
1. **Rich routing**: Full TSP/VRP integration with route optimization
2. **Multi-period**: Planning over multiple time periods
3. **Uncertainty**: Stochastic demand and travel times
4. **Real-time**: Dynamic re-optimization capabilities

---

*This mathematical model provides the foundation for the Vehicle Router's optimization capabilities. The formulation balances solution quality with computational efficiency, making it suitable for practical logistics applications.*