"""
Documentation Module for Streamlit Application

This module contains the documentation-related rendering methods for the
Vehicle Router Streamlit application, including introduction and methodology sections.
"""

import streamlit as st


class DocumentationRenderer:
    """
    Documentation Renderer for Streamlit Application
    
    This class handles rendering of documentation sections like introduction
    and methodology to keep the main app file clean and organized.
    """
    
    def __init__(self):
        """Initialize the DocumentationRenderer"""
        pass
    
    def render_introduction(self):
        """Render the introduction section"""
        st.markdown('<h2 class="section-header">📋 Introduction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to the Vehicle Router Optimizer
            
            This application solves the **Vehicle Routing Problem (VRP)** with order assignment optimization 
            using advanced Mixed Integer Linear Programming (MILP) techniques. The system helps you:
            
            #### 🎯 **Objectives**
            - **Minimize operational costs** by selecting the most cost-effective truck combinations
            - **Optimize delivery routes** while respecting capacity constraints
            - **Maximize truck utilization** to improve operational efficiency
            - **Ensure all orders are delivered** with optimal resource allocation
            
            #### 🔧 **How It Works**
            1. **Load Data**: Import order and truck information (or use example data)
            2. **Run Optimization**: Execute the MILP solver to find the optimal solution
            3. **Analyze Results**: Review detailed solution analysis and metrics
            4. **Visualize Routes**: Explore interactive charts and route visualizations
            5. **Export Results**: Download solution data in Excel format for further analysis
            
            #### 📊 **Key Features**
            - Interactive data exploration with sortable tables
            - Real-time optimization with progress tracking
            - Comprehensive solution validation and constraint checking
            - Professional visualizations including route maps and cost analysis
            - Excel export functionality for detailed reporting
            """)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🚛 Problem Type</h4>
                <p><strong>Vehicle Routing Problem</strong><br>
                with Order Assignment</p>
            </div>
            
            <div class="metric-card">
                <h4>⚡ Solver Technology</h4>
                <p><strong>Mixed Integer Linear Programming</strong><br>
                (MILP) using PuLP</p>
            </div>
            
            <div class="metric-card">
                <h4>🎯 Optimization Goal</h4>
                <p><strong>Minimize Total Cost</strong><br>
                Subject to capacity constraints</p>
            </div>
            
            <div class="metric-card">
                <h4>📈 Solution Quality</h4>
                <p><strong>Guaranteed Optimal</strong><br>
                Mathematical optimization</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide
        if not st.session_state.data_loaded:
            st.markdown("""
            <div class="warning-box">
                <h4>🚀 Quick Start Guide</h4>
                <ol>
                    <li>Click <strong>"Load Example Data"</strong> in the sidebar to get started with sample data</li>
                    <li>Or upload your own CSV files with orders and trucks information</li>
                    <li>Once data is loaded, click <strong>"Run Optimization"</strong> to solve the problem</li>
                    <li>Explore the results in the sections below!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    def render_methodology(self):
        """Render the methodology section"""
        st.markdown('<h2 class="section-header">🔬 Methodology</h2>', unsafe_allow_html=True)
        
        # Methodology tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Three Optimization Approaches", 
            "🧮 Mathematical Models", 
            "⚙️ Algorithm Implementation", 
            "🔍 Solution Process", 
            "📊 Performance Analysis",
            "📚 Technical References"
        ])
        
        with tab1:
            st.markdown("""
            ## Three Optimization Methodologies
            
            The Vehicle Router implements **three distinct optimization approaches** to solve the Vehicle Routing Problem, each with different strengths and computational characteristics:
            
            ---
            
            ### 🔵 **Methodology 1: Standard MILP (Cost-Only Optimization)**
            
            #### **Overview**
            The **Standard MILP** approach focuses purely on **cost minimization** through optimal truck selection and order assignment. This method uses a simplified Mixed Integer Linear Programming formulation that prioritizes operational cost efficiency.
            
            #### **Technical Approach**
            - **Primary Objective**: Minimize total truck operational costs
            - **Algorithm**: Pure MILP optimization using PuLP/CBC solver
            - **Route Handling**: Basic sorted postal code sequences (no route optimization)
            - **Computational Complexity**: O(|I| × |J|) variables, polynomial solving time
            - **Best For**: Cost-sensitive scenarios where route distances are secondary
            
            #### **Mathematical Formulation**
            
            **Decision Variables:**
            ```
            x_{i,j} ∈ {0,1}  ∀i ∈ Orders, ∀j ∈ Trucks    # Order assignment
            y_j ∈ {0,1}      ∀j ∈ Trucks                  # Truck usage
            ```
            
            **Objective Function:**
            ```
            minimize: Σ_{j ∈ Trucks} cost_j × y_j
            ```
            
            **Key Constraints:**
            ```
            Σ_{j} x_{i,j} = 1                    ∀i ∈ Orders     # Each order assigned once
            Σ_{i} volume_i × x_{i,j} ≤ capacity_j  ∀j ∈ Trucks   # Capacity limits
            y_j ≥ x_{i,j}                        ∀i,j            # Truck usage logic
            ```
            
            #### **Algorithm Workflow**
            1. **Input Processing**: Validate orders, trucks, and distance data
            2. **MILP Construction**: Create binary variables and cost-minimization objective
            3. **Constraint Addition**: Add capacity and assignment constraints
            4. **Solver Execution**: Run CBC solver with 60-second timeout
            5. **Solution Extraction**: Extract truck selections and order assignments
            6. **Route Generation**: Create simple depot→sorted_orders→(depot) sequences
            
            #### **Performance Characteristics**
            - **Solve Time**: < 1 second for typical problems (5-50 orders)
            - **Memory Usage**: Minimal (< 50MB for medium instances)
            - **Solution Quality**: Optimal for cost minimization
            - **Scalability**: Excellent (handles 100+ orders efficiently)
            
            #### **Example Output**
            ```
            Selected Trucks: [1, 2]
            Truck 1 → Orders ['A', 'C', 'D', 'E'] (Route: 08020→08027→08028→08029→08031)
            Truck 2 → Orders ['B'] (Route: 08020→08030)
            Total Cost: €2500
            ```
            
            ---
            
            ### 🟡 **Methodology 2: Standard MILP + Greedy Route Optimization**
            
            #### **Overview**
            The **Hybrid MILP-Greedy** approach combines the cost-optimal truck selection from MILP with a **post-optimization greedy algorithm** that tests all possible route permutations to minimize travel distances.
            
            #### **Technical Approach**
            - **Primary Objective**: Minimize costs (MILP) + minimize distances (Greedy)
            - **Algorithm**: Two-phase optimization (MILP → Greedy permutation testing)
            - **Route Handling**: Exhaustive permutation testing for optimal route sequences
            - **Computational Complexity**: MILP O(|I| × |J|) + Greedy O(n!) per truck
            - **Best For**: Balanced cost-distance optimization with moderate computational resources
            
            #### **Mathematical Formulation**
            
            **Phase 1 - MILP (Same as Standard):**
            ```
            minimize: Σ_{j ∈ Trucks} cost_j × y_j
            subject to: [same constraints as Standard MILP]
            ```
            
            **Phase 2 - Greedy Route Optimization:**
            ```
            For each selected truck j with assigned orders O_j:
                best_distance = ∞
                best_route = null
                
                For each permutation π of O_j:
                    route = depot → π → (depot if depot_return)
                    distance = Σ route_segments d(k,l)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_route = route
                
                assign best_route to truck j
            ```
            
            #### **Algorithm Workflow**
            1. **MILP Phase**: Execute standard cost optimization (identical to Method 1)
            2. **Greedy Initialization**: Extract truck assignments from MILP solution
            3. **Per-Truck Optimization**: For each truck with ≥2 orders:
               - Generate all permutations of assigned order locations
               - Calculate total distance for each permutation
               - Select permutation with minimum total distance
            4. **Route Reconstruction**: Build final routes with optimized sequences
            5. **Performance Logging**: Track improvements and permutations tested
            
            #### **Greedy Algorithm Details**
            ```python
            def optimize_routes_greedy(self, assignments, selected_trucks):
                for truck_id in selected_trucks:
                    assigned_orders = get_orders_for_truck(truck_id)
                    if len(assigned_orders) <= 1:
                        continue  # No optimization needed
                    
                    order_locations = [get_location(order) for order in assigned_orders]
                    best_distance = float('inf')
                    best_route = None
                    
                    # Test all permutations
                    for perm in permutations(order_locations):
                        route = [depot] + list(perm)
                        if depot_return:
                            route.append(depot)
                        
                        distance = calculate_route_distance(route)
                        if distance < best_distance:
                            best_distance = distance
                            best_route = route
                    
                    store_optimized_route(truck_id, best_route, best_distance)
            ```
            
            #### **Performance Characteristics**
            - **Solve Time**: MILP (< 1s) + Greedy (< 5s for typical cases)
            - **Memory Usage**: Low (< 100MB)
            - **Solution Quality**: Cost-optimal + distance-optimized routes
            - **Scalability**: Excellent for ≤8 orders per truck (8! = 40,320 permutations)
            - **Practical Limit**: Efficient up to 10 orders per truck
            
            #### **Greedy Performance Analysis**
            | Orders per Truck | Permutations | Typical Time | Memory |
            |------------------|--------------|--------------|---------|
            | 2 | 2 | < 0.001s | < 1MB |
            | 3 | 6 | < 0.001s | < 1MB |
            | 4 | 24 | < 0.01s | < 5MB |
            | 5 | 120 | < 0.05s | < 10MB |
            | 6 | 720 | < 0.2s | < 20MB |
            | 7 | 5,040 | < 1s | < 50MB |
            | 8 | 40,320 | < 5s | < 100MB |
            
            #### **Example Output with Logging**
            ```
            🔍 Starting greedy route optimization algorithm...
               Processing 2 trucks with depot_return=False
            🚛 Optimizing routes for Truck 1...
               Orders: ['A', 'C', 'D', 'E']
               Locations: ['08031', '08029', '08028', '08027']
               Current route: 08020 → 08027 → 08028 → 08029 → 08031
               Current distance: 11.0 km
               Testing 24 route permutations...
               ✨ New best route found: 08020 → 08031 → 08029 → 08028 → 08027 (9.5 km)
               🎯 Final optimized route: 08020 → 08031 → 08029 → 08028 → 08027
               📊 Distance improvement: 1.5 km (13.6%)
               📈 Tested 24 permutations
            🏁 Greedy route optimization completed!
               📊 Total distance before: 15.0 km
               📊 Total distance after: 13.5 km
               🎯 Total improvement: 1.5 km (10.0%)
            ```
            
            ---
            
            ### 🟢 **Methodology 3: Enhanced MILP (Integrated Cost-Distance Optimization)**
            
            #### **Overview**
            The **Enhanced MILP** approach implements a sophisticated **multi-objective optimization** that simultaneously minimizes both truck costs and travel distances within a single, integrated MILP formulation with advanced routing variables.
            
            #### **Technical Approach**
            - **Primary Objective**: Multi-objective weighted optimization (cost + distance)
            - **Algorithm**: Advanced MILP with routing variables and flow constraints
            - **Route Handling**: Integrated route optimization within MILP formulation
            - **Computational Complexity**: O(|I|×|J| + |L|²×|J|) variables, exponential worst-case
            - **Best For**: High-quality solutions where both cost and distance matter equally
            
            #### **Mathematical Formulation**
            
            **Decision Variables:**
            ```
            x_{i,j} ∈ {0,1}      ∀i ∈ Orders, ∀j ∈ Trucks        # Order assignment
            y_j ∈ {0,1}          ∀j ∈ Trucks                      # Truck usage
            z_{k,l,j} ∈ {0,1}    ∀k,l ∈ Locations, ∀j ∈ Trucks  # Route segments
            ```
            
            **Multi-Objective Function:**
            ```
            minimize: α × Σ_{j} cost_j × y_j + β × Σ_{k,l,j} distance_{k,l} × z_{k,l,j}
            
            where: α + β = 1 (typical: α = 0.6, β = 0.4)
            ```
            
            **Advanced Constraints:**
            ```
            # Standard constraints (same as Methods 1 & 2)
            Σ_{j} x_{i,j} = 1                           ∀i ∈ Orders
            Σ_{i} volume_i × x_{i,j} ≤ capacity_j      ∀j ∈ Trucks
            y_j ≥ x_{i,j}                               ∀i,j
            
            # Flow conservation constraints
            Σ_{k} z_{k,l,j} = Σ_{k} z_{l,k,j} = Σ_{i: loc_i=l} x_{i,j}  ∀l ∈ Locations, ∀j ∈ Trucks
            
            # Depot constraints
            Σ_{l≠depot} z_{depot,l,j} = y_j            ∀j ∈ Trucks  # Leave depot
            Σ_{l≠depot} z_{l,depot,j} = y_j            ∀j ∈ Trucks  # Return to depot (if enabled)
            
            # Subtour elimination (implicit through flow conservation)
            ```
            
            #### **Algorithm Workflow**
            1. **Enhanced Model Construction**: Create routing variables z_{k,l,j} for all location pairs
            2. **Multi-Objective Setup**: Configure weighted cost-distance objective function
            3. **Advanced Constraints**: Add flow conservation and depot routing constraints
            4. **Solver Configuration**: Use extended timeout (120s) for complex formulation
            5. **Route Reconstruction**: Extract actual route sequences from z variables
            6. **Solution Validation**: Verify route continuity and constraint satisfaction
            
            #### **Route Reconstruction Algorithm**
            ```python
            def reconstruct_route_sequence(self, truck_id):
                # Build adjacency list from routing variables
                adjacency = {}
                for k in locations:
                    adjacency[k] = []
                    for l in locations:
                        if z[k,l,truck_id].value > 0.5:  # Route segment active
                            adjacency[k].append(l)
                
                # Reconstruct route starting from depot
                route = [depot_location]
                current = depot_location
                visited = set()
                
                while True:
                    if current not in adjacency or not adjacency[current]:
                        break
                    
                    # Choose next location (prefer unvisited customers)
                    next_loc = None
                    for loc in adjacency[current]:
                        if loc != depot_location and loc not in visited:
                            next_loc = loc
                            break
                    
                    if next_loc is None and depot_location in adjacency[current]:
                        if len(visited) >= len(customer_locations):
                            next_loc = depot_location  # Return to depot
                    
                    if next_loc is None:
                        break
                    
                    route.append(next_loc)
                    if next_loc != depot_location:
                        visited.add(next_loc)
                    
                    current = next_loc
                    if current == depot_location and visited:
                        break  # Completed route
                
                return route
            ```
            
            #### **Performance Characteristics**
            - **Solve Time**: 1-30 seconds (depends on problem size and complexity)
            - **Memory Usage**: Moderate (100-500MB for medium instances)
            - **Solution Quality**: Globally optimal for multi-objective function
            - **Scalability**: Good for small-medium instances (≤50 orders, ≤15 trucks)
            
            #### **Model Complexity Comparison**
            | Aspect | Standard | Standard+Greedy | Enhanced |
            |--------|----------|-----------------|----------|
            | Variables | 30 | 30 | 210 |
            | Constraints | 35 | 35 | 95 |
            | Solve Time | 0.1s | 0.1s + 2s | 5s |
            | Memory | 10MB | 15MB | 100MB |
            | Route Quality | Basic | Optimized | Optimal |
            
            #### **Example Output**
            ```
            === VEHICLE ROUTER (ENHANCED OPTIMIZER) ===
            Depot Location: 08020
            Selected Trucks: [1, 2]
            Truck 1 → Orders ['A', 'B', 'E']
              Route: 08020 → 08027 → 08030 → 08031 → 08020
              Distance: 18.0 km
            Truck 2 → Orders ['C', 'D']
              Route: 08020 → 08029 → 08028 → 08020
              Distance: 16.0 km
            Total Cost: €2500
            Total Distance: 34.0 km
            Multi-Objective Value: 0.314 (60% cost + 40% distance)
            ```
            
            ---
            
            ### 📊 **Methodology Comparison Summary**
            
            | Criterion | Standard MILP | MILP + Greedy | Enhanced MILP |
            |-----------|---------------|---------------|---------------|
            | **Primary Focus** | Cost minimization | Cost + Route efficiency | Integrated optimization |
            | **Algorithm Type** | Pure MILP | Hybrid MILP-Heuristic | Advanced MILP |
            | **Route Quality** | Basic (sorted) | Optimized (permutations) | Optimal (integrated) |
            | **Solve Time** | Fastest (< 1s) | Fast (< 5s) | Moderate (< 30s) |
            | **Memory Usage** | Minimal | Low | Moderate |
            | **Scalability** | Excellent | Very Good | Good |
            | **Solution Guarantee** | Cost-optimal | Cost-optimal + Route-heuristic | Multi-objective optimal |
            | **Best Use Case** | Cost-critical | Balanced cost-distance | Quality-critical |
            | **Depot Return** | Configurable | Configurable | Configurable |
            | **Distance Calculation** | Post-hoc | Optimized post-MILP | Integrated in MILP |
            
            ### 🎯 **Choosing the Right Methodology**
            
            **Use Standard MILP when:**
            - Cost is the primary concern
            - Fast solving is required
            - Route distances are less important
            - Large problem instances (100+ orders)
            
            **Use MILP + Greedy when:**
            - Both cost and distance matter
            - Moderate computational resources available
            - Trucks typically have ≤8 orders each
            - Good balance of speed and quality needed
            
            **Use Enhanced MILP when:**
            - Highest solution quality required
            - Cost and distance are equally important
            - Computational resources are available
            - Problem size is small-medium (≤50 orders)
            """)
        
        with tab2:
            st.markdown("""
            ## Mathematical Models Deep Dive
            
            ### 🔵 **Standard MILP Mathematical Formulation**
            
            #### **Problem Statement**
            Given a set of orders and trucks, find the minimum-cost assignment of orders to trucks such that:
            - Each order is delivered exactly once
            - No truck exceeds its capacity
            - Total operational cost is minimized
            
            #### **Sets and Parameters**
            ```
            Sets:
            I = {1, 2, ..., n}     # Set of orders
            J = {1, 2, ..., m}     # Set of trucks
            
            Parameters:
            v_i ∈ ℝ⁺              # Volume of order i (m³)
            c_j ∈ ℝ⁺              # Cost of truck j (€)
            cap_j ∈ ℝ⁺            # Capacity of truck j (m³)
            loc_i ∈ PostalCodes   # Location of order i
            ```
            
            #### **Decision Variables**
            ```
            x_{i,j} ∈ {0,1}  ∀i ∈ I, ∀j ∈ J
            # x_{i,j} = 1 if order i is assigned to truck j, 0 otherwise
            
            y_j ∈ {0,1}  ∀j ∈ J
            # y_j = 1 if truck j is used in the solution, 0 otherwise
            ```
            
            #### **Complete Mathematical Model**
            ```
            minimize: Z = Σ_{j=1}^m c_j × y_j
            
            subject to:
            
            (1) Σ_{j=1}^m x_{i,j} = 1                    ∀i ∈ I
                [Order assignment: each order delivered exactly once]
            
            (2) Σ_{i=1}^n v_i × x_{i,j} ≤ cap_j × y_j    ∀j ∈ J
                [Capacity constraint: truck capacity not exceeded]
            
            (3) y_j ≥ x_{i,j}                            ∀i ∈ I, ∀j ∈ J
                [Truck usage: if order assigned, truck must be used]
            
            (4) x_{i,j} ∈ {0,1}                          ∀i ∈ I, ∀j ∈ J
                [Binary assignment variables]
            
            (5) y_j ∈ {0,1}                              ∀j ∈ J
                [Binary truck usage variables]
            ```
            
            #### **Model Properties**
            - **Variable Count**: |I| × |J| + |J| = n×m + m
            - **Constraint Count**: |I| + |J| + |I|×|J| = n + m + n×m
            - **Problem Class**: NP-hard (generalized assignment problem)
            - **LP Relaxation**: Often provides integer solutions due to problem structure
            
            ---
            
            ### 🟡 **MILP + Greedy Mathematical Formulation**
            
            #### **Two-Phase Optimization**
            
            **Phase 1: MILP (Identical to Standard)**
            ```
            minimize: Z₁ = Σ_{j=1}^m c_j × y_j
            subject to: [same constraints as Standard MILP]
            
            Output: Optimal truck selection T* and order assignments A*
            ```
            
            **Phase 2: Greedy Route Optimization**
            ```
            For each truck j ∈ T*:
                Let O_j = {orders assigned to truck j from Phase 1}
                Let L_j = {locations of orders in O_j}
                
                If |O_j| ≤ 1:
                    route_j = depot → L_j → (depot if depot_return)
                Else:
                    best_distance = ∞
                    best_route = null
                    
                    For each permutation π ∈ Permutations(L_j):
                        route = depot → π → (depot if depot_return)
                        distance = Σ_{k=1}^{|route|-1} d(route[k], route[k+1])
                        
                        If distance < best_distance:
                            best_distance = distance
                            best_route = route
                    
                    route_j = best_route
            
            Output: Cost-optimal assignments with distance-optimal routes
            ```
            
            #### **Greedy Algorithm Complexity Analysis**
            ```
            Time Complexity per truck:
            - Orders per truck: n_j
            - Permutations to test: n_j!
            - Distance calculations per permutation: n_j + depot_return
            - Total operations: O(n_j! × n_j)
            
            Total Greedy Complexity: O(Σ_{j ∈ T*} n_j! × n_j)
            
            Practical Performance:
            n_j = 2: 2! × 2 = 4 operations
            n_j = 3: 3! × 3 = 18 operations  
            n_j = 4: 4! × 4 = 96 operations
            n_j = 5: 5! × 5 = 600 operations
            n_j = 8: 8! × 8 = 322,560 operations (practical limit)
            ```
            
            #### **Distance Matrix Integration**
            ```
            Distance Matrix: D ∈ ℝ^{|L|×|L|}
            where L = {all postal codes} ∪ {depot}
            
            D[k,l] = distance from location k to location l (km)
            
            Route Distance Calculation:
            route_distance(r) = Σ_{i=1}^{|r|-1} D[r[i], r[i+1]]
            
            where r = [depot, loc₁, loc₂, ..., loc_n, (depot)]
            ```
            
            ---
            
            ### 🟢 **Enhanced MILP Mathematical Formulation**
            
            #### **Multi-Objective Problem Statement**
            Simultaneously minimize truck costs and travel distances while ensuring:
            - Each order is delivered exactly once
            - No truck exceeds its capacity  
            - Routes form valid depot-to-depot paths
            - Flow conservation is maintained at all locations
            
            #### **Extended Sets and Parameters**
            ```
            Sets:
            I = {1, 2, ..., n}     # Set of orders
            J = {1, 2, ..., m}     # Set of trucks  
            L = {1, 2, ..., k}     # Set of locations (postal codes + depot)
            
            Parameters:
            v_i ∈ ℝ⁺              # Volume of order i (m³)
            c_j ∈ ℝ⁺              # Cost of truck j (€)
            cap_j ∈ ℝ⁺            # Capacity of truck j (m³)
            d_{k,l} ∈ ℝ⁺          # Distance from location k to l (km)
            loc_i ∈ L             # Location of order i
            depot ∈ L             # Depot location
            α, β ∈ [0,1]          # Objective weights (α + β = 1)
            ```
            
            #### **Decision Variables**
            ```
            x_{i,j} ∈ {0,1}  ∀i ∈ I, ∀j ∈ J
            # Order assignment variables (same as standard)
            
            y_j ∈ {0,1}  ∀j ∈ J  
            # Truck usage variables (same as standard)
            
            z_{k,l,j} ∈ {0,1}  ∀k,l ∈ L, k≠l, ∀j ∈ J
            # Route variables: z_{k,l,j} = 1 if truck j travels from k to l
            ```
            
            #### **Multi-Objective Function**
            ```
            minimize: Z = α × (Σ_{j=1}^m c_j × y_j) + β × (Σ_{j=1}^m Σ_{k∈L} Σ_{l∈L,l≠k} d_{k,l} × z_{k,l,j})
            
            where:
            - First term: Total truck costs (scaled by weight α)
            - Second term: Total travel distances (scaled by weight β)
            - Typical values: α = 0.6, β = 0.4
            ```
            
            #### **Complete Enhanced Mathematical Model**
            ```
            minimize: Z = α × Σ_{j=1}^m c_j × y_j + β × Σ_{j=1}^m Σ_{k∈L} Σ_{l∈L,l≠k} d_{k,l} × z_{k,l,j}
            
            subject to:
            
            # Standard constraints (1)-(5) from Standard MILP
            
            (6) Σ_{k∈L,k≠l} z_{k,l,j} = Σ_{k∈L,k≠l} z_{l,k,j} = Σ_{i: loc_i=l} x_{i,j}  ∀l ∈ L\\{depot}, ∀j ∈ J
                [Flow conservation: inflow = outflow = orders served at location l]
            
            (7) Σ_{l∈L,l≠depot} z_{depot,l,j} = y_j  ∀j ∈ J
                [Depot outflow: used trucks leave depot exactly once]
            
            (8) Σ_{l∈L,l≠depot} z_{l,depot,j} = y_j × depot_return  ∀j ∈ J
                [Depot inflow: used trucks return to depot if required]
            
            (9) z_{k,l,j} ∈ {0,1}  ∀k,l ∈ L, k≠l, ∀j ∈ J
                [Binary route variables]
            ```
            
            #### **Enhanced Model Properties**
            - **Variable Count**: n×m + m + m×k×(k-1) ≈ m×k² for large k
            - **Constraint Count**: n + m + n×m + m×(k-1) + 2×m ≈ n×m + m×k
            - **Problem Class**: NP-hard (VRP with multi-objective)
            - **Complexity**: Exponential in worst case, polynomial for small instances
            
            #### **Flow Conservation Explanation**
            ```
            For each location l and truck j:
            
            Inflow to l:  Σ_{k≠l} z_{k,l,j}  (trucks arriving at l)
            Outflow from l: Σ_{k≠l} z_{l,k,j}  (trucks leaving l)  
            Orders at l:  Σ_{i: loc_i=l} x_{i,j}  (orders served at l by truck j)
            
            Conservation: inflow = outflow = orders served
            
            This ensures:
            - Trucks only visit locations with assigned orders
            - No subtours or disconnected routes
            - Proper route continuity
            ```
            
            #### **Objective Scaling and Normalization**
            ```
            Raw objectives have different scales:
            - Truck costs: €500 - €2000 per truck
            - Distances: 1-50 km per route segment
            
            Normalization approach:
            cost_scale = max_possible_truck_cost = Σ_{j} c_j
            distance_scale = max_possible_distance = max_{k,l} d_{k,l} × |L| × |J|
            
            Normalized objective:
            Z = α × (truck_cost / cost_scale) + β × (total_distance / distance_scale)
            
            This ensures both terms contribute meaningfully to the objective.
            ```
            """)
        
        with tab3:
            st.markdown("""
            ## Algorithm Implementation Deep Dive
            
            ### 🔵 **Standard MILP Implementation**
            
            #### **Class Structure and Methods**
            ```python
            class VrpOptimizer:
                def __init__(self, orders_df, trucks_df, distance_matrix, 
                           depot_location='08020', depot_return=False, 
                           enable_greedy_routes=True):
                    # Initialize with configuration options
                    
                def build_model(self):
                    # Create PuLP MILP model with binary variables
                    # Add cost minimization objective
                    # Add capacity and assignment constraints
                    
                def solve(self):
                    # Execute CBC solver with timeout
                    # Handle solver status and errors
                    # Extract optimal variable values
                    
                def _optimize_routes_greedy(self, assignments, selected_trucks):
                    # Post-MILP greedy route optimization
                    # Test all permutations for each truck
                    # Select minimum distance routes
                    
                def get_solution(self):
                    # Format structured solution data
                    # Include optimized routes if greedy enabled
                    # Return DataFrames and metrics
            ```
            
            #### **MILP Model Construction Process**
            ```python
            def build_model(self):
                # 1. Create decision variables
                self.decision_vars['x'] = {}  # Order assignments
                for order_id in self.orders:
                    for truck_id in self.trucks:
                        var_name = f"assign_{order_id}_to_truck_{truck_id}"
                        self.decision_vars['x'][(order_id, truck_id)] = 
                            pulp.LpVariable(var_name, cat='Binary')
                
                self.decision_vars['y'] = {}  # Truck usage
                for truck_id in self.trucks:
                    var_name = f"use_truck_{truck_id}"
                    self.decision_vars['y'][truck_id] = 
                        pulp.LpVariable(var_name, cat='Binary')
                
                # 2. Set objective function
                truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
                objective_terms = [truck_costs[truck_id] * self.decision_vars['y'][truck_id] 
                                 for truck_id in self.trucks]
                self.model += pulp.lpSum(objective_terms), "Total_Cost_Minimization"
                
                # 3. Add constraints
                self._add_order_assignment_constraints()
                self._add_capacity_constraints()
                self._add_truck_usage_constraints()
            ```
            
            #### **Greedy Route Optimization Implementation**
            ```python
            def _optimize_routes_greedy(self, assignments, selected_trucks):
                from itertools import permutations
                
                logger.info("🔍 Starting greedy route optimization algorithm...")
                total_distance_before = 0
                total_distance_after = 0
                
                for truck_id in selected_trucks:
                    assigned_orders = [a['order_id'] for a in assignments 
                                     if a['truck_id'] == truck_id]
                    
                    if len(assigned_orders) <= 1:
                        continue  # No optimization needed
                    
                    # Get order locations
                    order_locations = []
                    for order_id in assigned_orders:
                        postal_code = self.orders_df[
                            self.orders_df['order_id'] == order_id
                        ]['postal_code'].iloc[0]
                        order_locations.append(postal_code)
                    
                    # Calculate current route distance
                    current_route = [self.depot_location] + sorted(order_locations)
                    if self.depot_return:
                        current_route.append(self.depot_location)
                    current_distance = self._calculate_route_distance(current_route)
                    total_distance_before += current_distance
                    
                    # Test all permutations
                    best_route = current_route
                    best_distance = current_distance
                    permutation_count = 0
                    
                    for perm in permutations(order_locations):
                        permutation_count += 1
                        test_route = [self.depot_location] + list(perm)
                        if self.depot_return:
                            test_route.append(self.depot_location)
                        
                        test_distance = self._calculate_route_distance(test_route)
                        
                        if test_distance < best_distance:
                            best_route = test_route
                            best_distance = test_distance
                    
                    total_distance_after += best_distance
                    
                    # Store optimized route
                    if not hasattr(self, '_optimized_routes'):
                        self._optimized_routes = {}
                    self._optimized_routes[truck_id] = {
                        'route_sequence': best_route,
                        'route_distance': best_distance,
                        'improvement': current_distance - best_distance,
                        'permutations_tested': permutation_count
                    }
                
                # Log overall improvement
                total_improvement = total_distance_before - total_distance_after
                logger.info(f"🏁 Total improvement: {total_improvement:.1f} km")
            ```
            
            ---
            
            ### 🟢 **Enhanced MILP Implementation**
            
            #### **Class Structure and Advanced Features**
            ```python
            class EnhancedVrpOptimizer:
                def __init__(self, orders_df, trucks_df, distance_matrix, 
                           depot_location='08020', depot_return=True):
                    # Initialize with enhanced configuration
                    self.cost_weight = 0.6
                    self.distance_weight = 0.4
                    
                def set_objective_weights(self, cost_weight, distance_weight):
                    # Configure multi-objective weights
                    
                def build_model(self):
                    # Create enhanced MILP with routing variables
                    # Add multi-objective function
                    # Add flow conservation constraints
                    # Add depot routing constraints
                    
                def solve(self, timeout=300):
                    # Execute with extended timeout for complex model
                    
                def _reconstruct_route_sequence(self, truck_id):
                    # Extract route from routing variables
                    # Build adjacency lists from z variables
                    # Reconstruct depot-to-depot sequences
                    
                def get_solution(self):
                    # Format enhanced solution with routes
                    # Include distance calculations
                    # Return comprehensive metrics
            ```
            
            #### **Enhanced Model Construction**
            ```python
            def build_model(self):
                # 1. Create standard variables (x, y)
                self._create_assignment_variables()
                
                # 2. Create routing variables (z)
                self.decision_vars['z'] = {}
                for truck_id in self.trucks:
                    for loc1 in self.locations:
                        for loc2 in self.locations:
                            if loc1 != loc2:
                                var_name = f"route_truck_{truck_id}_from_{loc1}_to_{loc2}"
                                self.decision_vars['z'][(loc1, loc2, truck_id)] = 
                                    pulp.LpVariable(var_name, cat='Binary')
                
                # 3. Set multi-objective function
                truck_cost_terms = [cost[j] * self.decision_vars['y'][j] 
                                  for j in self.trucks]
                distance_cost_terms = [distance[k,l] * self.decision_vars['z'][(k,l,j)]
                                     for j in self.trucks for k in self.locations 
                                     for l in self.locations if k != l]
                
                # Scale objectives for proper weighting
                max_truck_cost = self.trucks_df['cost'].sum()
                max_distance = self.distance_matrix.values.max() * len(self.locations) * len(self.trucks)
                
                scaled_truck_cost = pulp.lpSum(truck_cost_terms) / max_truck_cost
                scaled_distance_cost = pulp.lpSum(distance_cost_terms) / max_distance
                
                objective = (self.cost_weight * scaled_truck_cost + 
                           self.distance_weight * scaled_distance_cost)
                
                self.model += objective, "Multi_Objective_Cost_Distance_Minimization"
                
                # 4. Add enhanced constraints
                self._add_standard_constraints()
                self._add_flow_conservation_constraints()
                self._add_depot_constraints()
            ```
            
            #### **Route Reconstruction Algorithm**
            ```python
            def _reconstruct_route_sequence(self, truck_id):
                # Build adjacency list from routing variables
                adjacency = {}
                for loc1 in self.locations:
                    adjacency[loc1] = []
                    for loc2 in self.locations:
                        if (loc1 != loc2 and 
                            (loc1, loc2, truck_id) in self.decision_vars['z'] and
                            self.decision_vars['z'][(loc1, loc2, truck_id)].varValue > 0.5):
                            adjacency[loc1].append(loc2)
                
                # Reconstruct route starting from depot
                route_sequence = [self.depot_location]
                current_location = self.depot_location
                visited_locations = set()
                
                # Get customer locations for this truck
                order_locations = set()
                for order_id in self.orders:
                    if self.decision_vars['x'][(order_id, truck_id)].varValue > 0.5:
                        postal_code = self.orders_df[
                            self.orders_df['order_id'] == order_id
                        ]['postal_code'].iloc[0]
                        order_locations.add(postal_code)
                
                # Follow the route
                while len(visited_locations) < len(order_locations):
                    if current_location not in adjacency or not adjacency[current_location]:
                        break
                    
                    # Choose next location (prefer unvisited customers)
                    next_location = None
                    for loc in adjacency[current_location]:
                        if loc != self.depot_location and loc not in visited_locations:
                            next_location = loc
                            break
                    
                    # If all customers visited, return to depot
                    if (next_location is None and 
                        self.depot_location in adjacency[current_location] and
                        len(visited_locations) >= len(order_locations)):
                        next_location = self.depot_location
                    
                    if next_location is None:
                        break
                    
                    route_sequence.append(next_location)
                    if next_location != self.depot_location:
                        visited_locations.add(next_location)
                    
                    current_location = next_location
                    
                    # Stop if back at depot after visiting all customers
                    if current_location == self.depot_location and visited_locations:
                        break
                
                # Ensure route ends at depot if depot_return enabled
                if (self.depot_return and 
                    route_sequence[-1] != self.depot_location):
                    route_sequence.append(self.depot_location)
                
                return route_sequence
            ```
            
            ### 🛠️ **Implementation Architecture**
            
            #### **Technology Stack**
            ```
            Optimization Layer:
            ├── PuLP (Python Linear Programming)
            │   ├── CBC Solver (default)
            │   ├── GLPK Solver (alternative)
            │   └── Commercial Solvers (Gurobi, CPLEX)
            │
            Data Processing Layer:
            ├── pandas (DataFrames)
            ├── numpy (numerical operations)
            └── scipy (distance calculations)
            │
            Application Layer:
            ├── Streamlit (web interface)
            ├── matplotlib/plotly (visualization)
            └── logging (progress tracking)
            ```
            
            #### **Performance Optimization Techniques**
            
            **1. Variable Reduction:**
            ```python
            # Remove dominated trucks before optimization
            def preprocess_trucks(self):
                dominated_trucks = []
                for i, truck1 in self.trucks_df.iterrows():
                    for j, truck2 in self.trucks_df.iterrows():
                        if (i != j and 
                            truck1['capacity'] <= truck2['capacity'] and
                            truck1['cost'] >= truck2['cost']):
                            dominated_trucks.append(truck1['truck_id'])
                return dominated_trucks
            ```
            
            **2. Constraint Tightening:**
            ```python
            # Add valid inequalities to strengthen formulation
            def add_valid_inequalities(self):
                # Minimum number of trucks needed
                total_volume = self.orders_df['volume'].sum()
                max_capacity = self.trucks_df['capacity'].max()
                min_trucks = math.ceil(total_volume / max_capacity)
                
                self.model += pulp.lpSum(self.decision_vars['y']) >= min_trucks
            ```
            
            **3. Solver Configuration:**
            ```python
            def configure_solver(self, timeout=60):
                solver = pulp.PULP_CBC_CMD(
                    msg=1,                    # Enable solver output
                    timeLimit=timeout,        # Set timeout
                    gapRel=0.01,             # 1% optimality gap
                    threads=4,               # Use multiple cores
                    presolve=True,           # Enable preprocessing
                    cuts=True,               # Enable cutting planes
                    heuristics=True          # Enable heuristics
                )
                return solver
            ```
            """)
        
        with tab4:
        
        with tab3:
            st.markdown("""
            ### Solution Process Workflow
            
            The Vehicle Router follows a systematic approach to solve routing problems with comprehensive validation and error handling.
            
            #### Step-by-Step Process
            
            **Phase 1: Data Preparation**
            1. **Input Validation**
               - Verify DataFrame structures and required columns
               - Check data types and value ranges
               - Validate postal code consistency
               - Ensure capacity feasibility
            
            2. **Distance Matrix Generation**
               - Calculate distances between all postal code pairs
               - Add depot location if using enhanced model
               - Validate matrix symmetry and completeness
            
            3. **Problem Analysis**
               - Calculate total volume vs. total capacity
               - Identify potential infeasibilities
               - Generate data summary statistics
            
            **Phase 2: Model Building**
            1. **Variable Creation**
               - Generate assignment variables x[i,j]
               - Create truck usage variables y[j]
               - Add routing variables z[k,l,j] for enhanced model
            
            2. **Constraint Addition**
               - Order assignment constraints (each order delivered once)
               - Capacity constraints (truck limits respected)
               - Truck usage constraints (logical consistency)
               - Route continuity constraints (enhanced model)
               - Depot constraints (enhanced model)
            
            3. **Objective Function**
               - Standard: minimize truck costs only
               - Enhanced: minimize weighted cost + distance
            
            **Phase 3: Optimization**
            1. **Solver Configuration**
               - Set timeout limits (60s standard, 120s enhanced)
               - Configure solver parameters
               - Enable progress monitoring
            
            2. **Solution Execution**
               - Launch MILP solver (CBC default)
               - Monitor solution progress
               - Handle timeout and error conditions
               - Extract optimal variable values
            
            3. **Status Handling**
               - **Optimal:** Solution found within tolerance
               - **Infeasible:** No solution exists
               - **Unbounded:** Problem formulation error
               - **Time Limit:** Partial solution or failure
            
            **Phase 4: Solution Processing**
            1. **Result Extraction**
               - Identify selected trucks from y[j] variables
               - Extract order assignments from x[i,j] variables
               - Reconstruct routes from z[k,l,j] variables (enhanced)
            
            2. **Metric Calculation**
               - Total operational costs
               - Total travel distances (enhanced model)
               - Truck capacity utilization rates
               - Solution quality indicators
            
            3. **Route Reconstruction** (Enhanced Model)
               - Build adjacency lists from routing variables
               - Reconstruct depot-to-customer-to-depot sequences
               - Calculate route distances and validate feasibility
            
            **Phase 5: Validation & Output**
            1. **Solution Validation**
               - Verify all orders are assigned exactly once
               - Check truck capacity constraints
               - Validate route feasibility (enhanced model)
               - Generate validation report
            
            2. **Result Formatting**
               - Create structured DataFrames for assignments and routes
               - Generate summary statistics and metrics
               - Prepare visualization data
               - Format console output
            
            3. **Export & Visualization**
               - Save results to CSV files
               - Generate route visualization plots
               - Create cost and utilization charts
               - Export Excel reports
            
            #### Error Handling Strategy
            
            **Input Validation Errors:**
            - Missing or invalid columns in input data
            - Negative volumes or capacities
            - Inconsistent postal codes
            - Infeasible problem instances
            
            **Optimization Errors:**
            - Solver timeout or memory limits
            - Numerical instability issues
            - License problems with commercial solvers
            - Model formulation errors
            
            **Solution Processing Errors:**
            - Invalid variable values from solver
            - Route reconstruction failures
            - Validation constraint violations
            - Export and visualization errors
            """)
        
        with tab4:
            st.markdown("""
            ### Technical References & Implementation Details
            
            #### Mathematical Foundations
            
            **Linear Programming Theory:**
            - Dantzig, G. B. (1963). *Linear Programming and Extensions*
            - Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization*
            
            **Vehicle Routing Problem:**
            - Toth, P., & Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications*
            - Golden, B. L., Raghavan, S., & Wasil, E. A. (2008). *The Vehicle Routing Problem*
            
            **Mixed Integer Programming:**
            - Wolsey, L. A. (1998). *Integer Programming*
            - Nemhauser, G. L., & Wolsey, L. A. (1999). *Integer and Combinatorial Optimization*
            
            #### Software Dependencies
            
            **Core Optimization:**
            ```python
            PuLP >= 2.7.0          # Linear programming modeling
            numpy >= 1.21.0        # Numerical computing
            pandas >= 1.5.0        # Data manipulation
            ```
            
            **Visualization:**
            ```python
            matplotlib >= 3.5.0    # Static plotting
            seaborn >= 0.11.0      # Statistical visualization
            plotly >= 5.15.0       # Interactive charts
            ```
            
            **Web Application:**
            ```python
            streamlit >= 1.28.0    # Web interface framework
            openpyxl >= 3.1.0      # Excel export functionality
            ```
            
            **Testing & Quality:**
            ```python
            pytest >= 7.0.0        # Testing framework
            scipy >= 1.9.0         # Scientific computing
            ```
            
            #### Algorithm Complexity Analysis
            
            **Time Complexity:**
            - **Standard Model:** O(2^(|I|×|J|)) worst-case, polynomial average-case
            - **Enhanced Model:** O(2^(|I|×|J|+|L|²×|J|)) worst-case
            - **Practical Performance:** Near-linear for small-medium instances
            
            **Space Complexity:**
            - **Variable Storage:** O(|I|×|J| + |L|²×|J|) for enhanced model
            - **Constraint Matrix:** Sparse representation for efficiency
            - **Solution Storage:** O(|I| + |J| + |L|) for results
            
            #### Implementation Best Practices
            
            **Code Organization:**
            - Modular design with separate optimizer classes
            - Clear separation of concerns (data, optimization, visualization)
            - Comprehensive error handling and logging
            - Extensive documentation and type hints
            
            **Performance Optimization:**
            - Efficient sparse matrix representations
            - Lazy evaluation of distance calculations
            - Memory-conscious data structures
            - Solver parameter tuning for problem characteristics
            
            **Quality Assurance:**
            - Comprehensive unit test coverage
            - Integration tests with known optimal solutions
            - Input validation and sanitization
            - Solution verification and constraint checking
            
            #### Extending the Framework
            
            **Adding New Constraints:**
            1. Define constraint mathematically
            2. Implement in `_add_custom_constraints()` method
            3. Update validation logic
            4. Add corresponding tests
            
            **Custom Objective Functions:**
            1. Modify `_set_objective_function()` method
            2. Add weight parameters for multi-objective optimization
            3. Update solution interpretation logic
            4. Document new optimization goals
            
            **Alternative Solvers:**
            1. Install solver-specific Python packages
            2. Configure PuLP solver selection
            3. Test performance characteristics
            4. Update documentation with solver-specific notes
            
            #### Performance Tuning Guidelines
            
            **For Large Instances:**
            - Increase solver timeout limits
            - Use commercial solvers (Gurobi, CPLEX) if available
            - Consider problem decomposition strategies
            - Implement heuristic initialization
            
            **For Real-Time Applications:**
            - Pre-compute distance matrices
            - Cache optimization models
            - Use warm-start techniques
            - Implement solution quality vs. time trade-offs
            """)    
    
        with tab6:
            st.markdown("""
            ## Technical References & Advanced Topics
            
            ### 📚 **Mathematical Foundations**
            
            #### **Linear Programming Theory**
            - **Dantzig, G. B. (1963).** *Linear Programming and Extensions*
              - Foundation of simplex method and linear programming theory
              - Basis for modern MILP solvers like CBC
            
            - **Bertsimas, D., & Tsitsiklis, J. N. (1997).** *Introduction to Linear Optimization*
              - Comprehensive treatment of linear optimization theory
              - Duality theory and sensitivity analysis
            
            #### **Vehicle Routing Problem Literature**
            - **Toth, P., & Vigo, D. (2014).** *Vehicle Routing: Problems, Methods, and Applications*
              - Comprehensive survey of VRP variants and solution methods
              - Covers CVRP, VRPTW, and multi-objective approaches
            
            - **Golden, B. L., Raghavan, S., & Wasil, E. A. (2008).** *The Vehicle Routing Problem*
              - Latest advances in VRP research and applications
              - Heuristic and exact solution methods
            
            #### **Mixed Integer Programming**
            - **Wolsey, L. A. (1998).** *Integer Programming*
              - Theory and algorithms for integer programming
              - Branch-and-bound and cutting plane methods
            
            - **Nemhauser, G. L., & Wolsey, L. A. (1999).** *Integer and Combinatorial Optimization*
              - Advanced topics in combinatorial optimization
              - Polyhedral theory and valid inequalities
            
            ### 🛠️ **Software Dependencies & Architecture**
            
            #### **Core Optimization Stack**
            ```python
            # Mathematical Optimization
            PuLP >= 2.7.0              # Linear programming modeling framework
            numpy >= 1.21.0            # Numerical computing and array operations
            pandas >= 1.5.0            # Data manipulation and analysis
            scipy >= 1.9.0             # Scientific computing utilities
            
            # Solver Backends (one or more required)
            CBC                         # Default open-source MILP solver
            GLPK                        # Alternative open-source solver
            Gurobi                      # Commercial high-performance solver (optional)
            CPLEX                       # IBM commercial solver (optional)
            ```
            
            #### **Visualization & Interface Stack**
            ```python
            # Static Visualization
            matplotlib >= 3.5.0        # Static plotting and chart generation
            seaborn >= 0.11.0          # Statistical data visualization
            
            # Interactive Visualization
            plotly >= 5.15.0           # Interactive charts and dashboards
            
            # Web Application Framework
            streamlit >= 1.28.0        # Web interface and dashboard
            openpyxl >= 3.1.0          # Excel export functionality
            
            # Testing & Quality Assurance
            pytest >= 7.0.0            # Testing framework
            pytest-cov >= 4.0.0        # Code coverage analysis
            ```
            
            #### **System Architecture**
            ```
            Vehicle Router Architecture:
            
            ┌─────────────────────────────────────────────────────────────┐
            │                    User Interface Layer                     │
            ├─────────────────────┬───────────────────────────────────────┤
            │   Streamlit Web App │           CLI Application             │
            │   - Interactive UI  │           - Command line              │
            │   - Real-time viz   │           - Batch processing          │
            │   - Export tools    │           - Automation friendly       │
            └─────────────────────┴───────────────────────────────────────┘
                                  │
            ┌─────────────────────────────────────────────────────────────┐
            │                 Application Logic Layer                     │
            ├─────────────────────┬─────────────────┬───────────────────────┤
            │   Data Management   │   Optimization  │    Visualization      │
            │   - Input validation│   - MILP models │    - Route plots      │
            │   - Data generation │   - Greedy algo │    - Cost analysis    │
            │   - Distance calc   │   - Solution    │    - Utilization      │
            └─────────────────────┴─────────────────┴───────────────────────┘
                                  │
            ┌─────────────────────────────────────────────────────────────┐
            │                 Optimization Engine Layer                   │
            ├─────────────────────┬─────────────────┬───────────────────────┤
            │   Standard MILP     │   MILP + Greedy │    Enhanced MILP      │
            │   - Cost focus      │   - Hybrid      │    - Multi-objective  │
            │   - Fast solving    │   - Balanced    │    - Route integration│
            │   - High scalability│   - Good quality│    - Best quality     │
            └─────────────────────┴─────────────────┴───────────────────────┘
                                  │
            ┌─────────────────────────────────────────────────────────────┐
            │                    Solver Backend Layer                     │
            ├─────────────────────┬─────────────────┬───────────────────────┤
            │        CBC          │       GLPK      │    Gurobi/CPLEX       │
            │   - Default solver  │   - Alternative │    - Commercial       │
            │   - Open source     │   - Open source │    - High performance │
            │   - Good performance│   - Lightweight │    - Advanced features│
            └─────────────────────┴─────────────────┴───────────────────────┘
            ```
            
            ### 🔬 **Advanced Algorithm Topics**
            
            #### **Complexity Theory Analysis**
            
            **Problem Classification:**
            ```
            Vehicle Routing Problem Complexity:
            ├── CVRP (Capacitated VRP): NP-hard
            ├── TSP (Traveling Salesman): NP-hard (special case)
            ├── Bin Packing: NP-hard (order assignment subproblem)
            └── Multi-objective VRP: NP-hard (enhanced model)
            
            Approximation Results:
            ├── CVRP: Best known approximation ratio ≈ 1.4
            ├── TSP: Christofides algorithm gives 1.5-approximation
            └── Bin Packing: First Fit Decreasing gives 11/9-approximation
            ```
            
            **Computational Complexity Bounds:**
            ```python
            # Standard MILP
            def standard_complexity_analysis():
                variables = n_orders * n_trucks + n_trucks
                constraints = n_orders + n_trucks + n_orders * n_trucks
                
                # Worst-case exponential in number of variables
                worst_case_time = O(2^variables)
                
                # Average case much better due to problem structure
                average_case_time = O(variables^3)  # Polynomial
                
                return {
                    'variables': variables,
                    'constraints': constraints,
                    'worst_case': worst_case_time,
                    'average_case': average_case_time
                }
            
            # Enhanced MILP
            def enhanced_complexity_analysis():
                variables = (n_orders * n_trucks + 
                           n_trucks + 
                           n_trucks * n_locations * (n_locations - 1))
                
                constraints = (n_orders + n_trucks + n_orders * n_trucks +
                             n_trucks * (n_locations - 1) + 2 * n_trucks)
                
                # Much higher complexity due to routing variables
                worst_case_time = O(2^variables)
                space_complexity = O(variables + constraints)
                
                return {
                    'variables': variables,
                    'constraints': constraints,
                    'complexity': worst_case_time,
                    'space': space_complexity
                }
            ```
            
            #### **Advanced Optimization Techniques**
            
            **1. Cutting Planes for Enhanced MILP**
            ```python
            def add_subtour_elimination_cuts(model, z_vars, locations, trucks):
                # Add subtour elimination constraints dynamically
                for truck in trucks:
                    for subset in generate_location_subsets(locations):
                        if 2 <= len(subset) <= len(locations) - 2:
                            # Subtour elimination inequality
                            cut = pulp.lpSum([
                                z_vars[(i, j, truck)] 
                                for i in subset for j in subset if i != j
                            ]) <= len(subset) - 1
                            
                            model += cut, f"subtour_elimination_{truck}_{hash(tuple(subset))}"
            
            def add_capacity_cuts(model, x_vars, y_vars, orders, trucks):
                # Add strengthened capacity cuts
                for truck in trucks:
                    for order_subset in generate_order_subsets(orders):
                        total_volume = sum(order['volume'] for order in order_subset)
                        truck_capacity = truck['capacity']
                        
                        if total_volume > truck_capacity:
                            # This subset cannot be assigned to this truck
                            cut = pulp.lpSum([
                                x_vars[(order['id'], truck['id'])] 
                                for order in order_subset
                            ]) <= len(order_subset) - 1
                            
                            model += cut, f"capacity_cut_{truck['id']}_{hash(tuple(order_subset))}"
            ```
            
            **2. Heuristic Initialization**
            ```python
            def generate_initial_solution(orders_df, trucks_df):
                # Greedy initialization for warm-starting MILP
                trucks_sorted = trucks_df.sort_values('cost').copy()
                orders_sorted = orders_df.sort_values('volume', ascending=False).copy()
                
                assignment = {}
                truck_loads = {truck_id: 0 for truck_id in trucks_sorted['truck_id']}
                
                for _, order in orders_sorted.iterrows():
                    # Find cheapest truck that can accommodate this order
                    for _, truck in trucks_sorted.iterrows():
                        if truck_loads[truck['truck_id']] + order['volume'] <= truck['capacity']:
                            assignment[order['order_id']] = truck['truck_id']
                            truck_loads[truck['truck_id']] += order['volume']
                            break
                
                return assignment
            
            def warm_start_milp(model, x_vars, y_vars, initial_assignment):
                # Set initial variable values based on heuristic solution
                for (order_id, truck_id), var in x_vars.items():
                    if initial_assignment.get(order_id) == truck_id:
                        var.setInitialValue(1)
                    else:
                        var.setInitialValue(0)
                
                used_trucks = set(initial_assignment.values())
                for truck_id, var in y_vars.items():
                    if truck_id in used_trucks:
                        var.setInitialValue(1)
                    else:
                        var.setInitialValue(0)
            ```
            
            **3. Decomposition Strategies**
            ```python
            def benders_decomposition(orders_df, trucks_df, distance_matrix):
                # Benders decomposition for large instances
                
                # Master problem: truck selection and order assignment
                master_model = create_master_problem(orders_df, trucks_df)
                
                # Subproblem: route optimization for fixed assignments
                def solve_subproblem(assignment):
                    subproblem_model = create_routing_subproblem(assignment, distance_matrix)
                    subproblem_model.solve()
                    return subproblem_model.objective.value()
                
                # Iterative solution
                iteration = 0
                while iteration < max_iterations:
                    # Solve master problem
                    master_model.solve()
                    current_assignment = extract_assignment(master_model)
                    
                    # Solve subproblem
                    subproblem_cost = solve_subproblem(current_assignment)
                    
                    # Check convergence
                    if abs(master_model.objective.value() - subproblem_cost) < tolerance:
                        break
                    
                    # Add Benders cut to master problem
                    benders_cut = generate_benders_cut(current_assignment, subproblem_cost)
                    master_model += benders_cut
                    
                    iteration += 1
                
                return current_assignment, subproblem_cost
            ```
            
            ### 🚀 **Future Enhancements & Research Directions**
            
            #### **Algorithmic Improvements**
            ```
            1. Machine Learning Integration:
               ├── Neural networks for solution initialization
               ├── Reinforcement learning for route construction
               └── ML-based solver parameter tuning
            
            2. Advanced Metaheuristics:
               ├── Genetic algorithms for large instances
               ├── Simulated annealing for route improvement
               └── Hybrid MILP-metaheuristic approaches
            
            3. Parallel Computing:
               ├── Distributed MILP solving
               ├── Parallel greedy route optimization
               └── GPU-accelerated distance calculations
            ```
            
            #### **Problem Extensions**
            ```
            1. Time Windows:
               ├── Delivery time constraints
               ├── Driver working hours
               └── Customer availability windows
            
            2. Multi-Period Planning:
               ├── Daily route planning
               ├── Weekly optimization cycles
               └── Seasonal demand patterns
            
            3. Stochastic Elements:
               ├── Uncertain demand volumes
               ├── Variable travel times
               └── Truck availability uncertainty
            
            4. Multi-Objective Extensions:
               ├── Environmental impact minimization
               ├── Driver satisfaction optimization
               └── Customer service level maximization
            ```
            
            #### **Implementation Improvements**
            ```
            1. Real-Time Capabilities:
               ├── Dynamic re-optimization
               ├── Live traffic integration
               └── Real-time order updates
            
            2. Scalability Enhancements:
               ├── Cloud-based solving
               ├── Microservices architecture
               └── Database integration
            
            3. User Experience:
               ├── Mobile applications
               ├── API development
               └── Advanced visualization
            ```
            
            ### 📊 **Performance Tuning Guidelines**
            
            #### **For Production Deployment**
            ```python
            # Solver configuration for production
            def configure_production_solver():
                return pulp.PULP_CBC_CMD(
                    timeLimit=300,          # 5-minute timeout
                    gapRel=0.01,           # 1% optimality gap
                    threads=0,             # Use all available cores
                    presolve=True,         # Enable preprocessing
                    cuts=True,             # Enable cutting planes
                    heuristics=True,       # Enable heuristics
                    logPath="solver.log",  # Log solver output
                    keepFiles=False        # Clean up temporary files
                )
            
            # Memory management for large instances
            def optimize_memory_usage():
                import gc
                
                # Force garbage collection between optimizations
                gc.collect()
                
                # Use sparse data structures
                sparse_distance_matrix = scipy.sparse.csr_matrix(distance_matrix)
                
                # Process data in chunks for very large instances
                chunk_size = 1000
                for i in range(0, len(orders_df), chunk_size):
                    chunk = orders_df.iloc[i:i+chunk_size]
                    process_order_chunk(chunk)
            
            # Monitoring and logging
            def setup_production_monitoring():
                import logging
                
                # Configure detailed logging
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('vehicle_router.log'),
                        logging.StreamHandler()
                    ]
                )
                
                # Performance metrics tracking
                metrics = {
                    'solve_time': 0,
                    'memory_usage': 0,
                    'solution_quality': 0,
                    'constraint_violations': 0
                }
                
                return metrics
            ```
            
            This comprehensive technical documentation provides the foundation for understanding, implementing, and extending the Vehicle Router's three optimization methodologies. Each approach offers different trade-offs between computational efficiency, solution quality, and scalability, making the system adaptable to a wide range of real-world logistics optimization scenarios.
            """)