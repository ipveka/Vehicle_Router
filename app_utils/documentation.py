"""
Documentation Module for Streamlit Application

This module contains the documentation-related rendering methods for the
Vehicle Router Streamlit application, including introduction and methodology sections.
"""

import streamlit as st


class DocumentationRenderer:
    """
    Documentation Renderer for Vehicle Router Streamlit Application
    
    This class provides documentation rendering for the Vehicle Router application
    for the Vehicle Router optimization application.
    """
    
    def __init__(self):
        """Initialize the DocumentationRenderer"""
        pass
    
    def render_introduction(self):
        """Render the introduction section"""
        st.title("🚛 Vehicle Router Optimizer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to the Vehicle Router Optimizer
            
            This application solves the **Vehicle Routing Problem (VRP)** using **three different optimization approaches**. 
            Choose from Mathematical Programming, Hybrid Optimization, or Evolutionary Algorithms to find good solutions for your routing needs.
            
            #### 🎯 **Three Optimization Methods**
            - **📊 Standard MILP + Greedy**: Cost-optimal truck selection with route distance optimization
            - **🚀 Enhanced MILP**: Multi-objective optimization balancing cost and distance  
            - **🧬 Genetic Algorithm**: Evolutionary approach for larger routing problems
            
            #### 🔧 **How It Works**
            1. **Load Data**: Import order and truck information (or use example data)
            2. **Select Method**: Choose optimization approach based on your priorities
            3. **Configure Parameters**: Adjust method-specific settings (weights, GA parameters, etc.)
            4. **Run Optimization**: Execute solver to find optimal solution
            5. **Analyze Results**: Review solution analysis with metrics
            6. **Visualize Routes**: View route maps and performance charts
            7. **Export & Compare**: Download results and compare methods using built-in tools
            
            #### 📊 **Key Features**
            - **Three Optimization Methods**: MILP, Hybrid, and Genetic Algorithm approaches
            - **Interactive Interface**: Optimization with progress tracking
            - **Visualizations**: Route maps, cost analysis, and performance metrics
            - **Method Comparison**: Tools to compare different optimization approaches
            - **Export Options**: Excel reports and solution summaries
            - **Easy to Use**: Works with example data or your own files
            """)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🚛 Problem Type</h4>
                <p><strong>Vehicle Routing</strong><br>
                with Multi-Objective Optimization</p>
            </div>
            
            <div class="metric-card">
                <h4>⚡ Algorithm Technologies</h4>
                <p><strong>MILP + Genetic Algorithms</strong><br>
                Mathematical & Evolutionary Methods</p>
            </div>
            
            <div class="metric-card">
                <h4>🎯 Optimization Goals</h4>
                <p><strong>Cost + Distance Optimization</strong><br>
                Flexible multi-objective balancing</p>
            </div>
            
            <div class="metric-card">
                <h4>📈 Solution Quality</h4>
                <p><strong>Good to Optimal</strong><br>
                Depends on method chosen</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide
        if not st.session_state.data_loaded:
            st.markdown("""
            <div class="warning-box">
                <h4>🚀 Quick Start Guide</h4>
                <ol>
                    <li><strong>Load Data:</strong> Click "Load Example Data" in the sidebar or upload your own CSV files</li>
                    <li><strong>Choose Method:</strong> Select from Standard MILP + Greedy (default), Enhanced MILP, or Genetic Algorithm</li>
                    <li><strong>Configure:</strong> Adjust method-specific parameters (optional - defaults work well)</li>
                    <li><strong>Optimize:</strong> Click "🚀 Run Optimization" to solve the problem</li>
                    <li><strong>Analyze:</strong> Review results in Solution Analysis, Visualization, and Documentation sections</li>
                    <li><strong>Compare:</strong> Try different methods to find the best approach for your needs</li>
                </ol>
                <p><strong>💡 Tip:</strong> Start with Standard MILP + Greedy for fastest results, then try Enhanced MILP for better route quality!</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_methodology(self, selected_method: str = "standard"):
        """Render the methodology section for the selected optimization method"""
        # Show documentation only for the selected method (no header - handled by main app)
        if selected_method == "standard":
            self._render_standard_milp_method()
        elif selected_method == "hybrid":
            self._render_standard_milp_greedy_method()
        elif selected_method == "enhanced":
            self._render_enhanced_milp_method()
        elif selected_method == "genetic":
            self._render_genetic_algorithm_method()
        else:
            # Default to hybrid if unknown method
            self._render_standard_milp_greedy_method()
    
    # Method-specific documentation with improved naming and enhanced content
    def _render_standard_milp_method(self):
        """Render documentation for Standard MILP method"""
        st.markdown("""
        ### 📊 Standard MILP
        
        **Cost-focused optimization** using Mixed Integer Linear Programming with simple route generation.
        
        **Detailed Process:**
        1. **Cost Minimization:** Uses MILP to find the minimum-cost combination of trucks that can deliver all orders
        2. **Order Assignment:** Assigns each order to exactly one truck while respecting capacity constraints
        3. **Route Generation:** Creates simple depot→orders→depot routes using postal code sorting
        4. **Validation:** Ensures all constraints are satisfied and solution is feasible
        
        **Mathematical Foundation:**
        - **Decision Variables:** Binary variables for truck selection and order assignment
        - **Objective Function:** Minimize Σ(truck_cost × truck_usage)
        - **Constraints:** Order assignment uniqueness, capacity limits, truck usage logic
        - **Solver:** CBC (Coin-OR Branch and Cut) with 60-second timeout
        
        **Performance Characteristics:**
        - **Speed:** Fastest method (typically < 1 second)
        - **Memory:** Minimal usage (< 50MB for medium problems)
        - **Scalability:** Works well (handles 100+ orders efficiently)
        - **Solution Quality:** Finds cost-optimal truck selection
        
        **Example Results:**
        - Selected Trucks: 1, 2 (€2500 total cost)
        - Truck 1 delivers Orders A, C, D, E (100% capacity utilization)
        - Truck 2 delivers Order B (100% capacity utilization)
        """)
    
    def _render_standard_milp_greedy_method(self):
        """Render documentation for Standard MILP + Greedy method"""
        st.markdown("""
        ### 🔄 Standard MILP + Greedy
        
        **Two-step approach** that first finds cost-optimal trucks, then improves routes by testing different sequences.
        
        **Detailed Process:**
        1. **Phase 1 - MILP Optimization:** Executes standard MILP to find cost-optimal truck selection and order assignments
        2. **Phase 2 - Greedy Route Optimization:** For each selected truck, tests all possible route permutations to minimize travel distance
        3. **Permutation Testing:** Evaluates factorial combinations (n!) of delivery sequences per truck
        4. **Best Route Selection:** Chooses the route permutation with minimum total distance for each truck
        5. **Final Solution:** Combines cost-optimal assignments with distance-optimal routes
        
        **Algorithm Details:**
        - **MILP Component:** Identical to Standard MILP (cost-optimal truck selection)
        - **Greedy Component:** Exhaustive permutation testing with distance matrix calculations
        - **Complexity:** O(n × m) for MILP + O(k!) per truck for route optimization
        - **Practical Limit:** Efficient up to 8 orders per truck (40,320 permutations)
        
        **Performance Characteristics:**
        - **Speed:** Fast (< 5 seconds total: MILP + greedy optimization)
        - **Memory:** Low usage (< 100MB)
        - **Scalability:** Works well for typical distributions (≤8 orders per truck)
        - **Solution Quality:** Cost-optimal + improved routes
        
        **Route Optimization Process:**
        - Tests all possible route permutations for each truck
        - Evaluates distance for each route combination  
        - Selects optimal route: 08020 → 08027 → 08028 → 08029 → 08031
        - Achieves distance improvement: 1.5 km (13.6% better)
        """)
    
    def _render_enhanced_milp_method(self):
        """Render documentation for Enhanced MILP method"""
        st.markdown("""
        ### 🚀 Enhanced MILP
        
        **Multi-objective approach** that tries to balance cost and distance in a single optimization step.
        
        **Detailed Process:**
        1. **Multi-Objective Formulation:** Combines truck costs and travel distances in weighted objective function
        2. **Routing Variables:** Creates binary variables for every possible route segment between locations
        3. **Flow Conservation:** Ensures trucks follow valid, continuous routes from depot through all assigned orders
        4. **Simultaneous Optimization:** Optimizes truck selection, order assignment, and route planning in single model
        5. **Solution Reconstruction:** Extracts actual route sequences from optimal routing variable values
        
        **Mathematical Model:**
        - **Variables:** Assignment (x_{i,j}), truck usage (y_j), route segments (z_{k,l,j})
        - **Objective:** α × Σ(costs) + β × Σ(distances) where α + β = 1
        - **Constraints:** Flow conservation, depot routing, capacity limits, subtour elimination
        - **Complexity:** O(|orders| × |trucks| + |locations|² × |trucks|) variables
        
        **Multi-Objective Configuration:**
        - **Default Weights:** 60% cost focus + 40% distance focus
        - **Customizable:** User can adjust weight balance via sliders
        - **Normalization:** Scales objectives to ensure meaningful contribution
        - **Trade-off Analysis:** Explores Pareto-efficient solutions
        
        **Performance Characteristics:**
        - **Speed:** Moderate (1-30 seconds depending on problem complexity)
        - **Memory:** Higher usage (100-500MB for medium instances)
        - **Scalability:** Good for small-medium problems (≤50 orders, ≤15 trucks)
        - **Solution Quality:** Finds optimal balance between cost and distance
        """)
    
    def _render_genetic_algorithm_method(self):
        """Render documentation for Genetic Algorithm method"""
        st.markdown("""
        ### 🧬 Genetic Algorithm
        
        **Evolutionary metaheuristic optimization** using population-based search with genetic operators for multi-objective VRP solving.
        
        **Detailed Process:**
        1. **Population Initialization:** Creates diverse set of random feasible solutions (truck-order assignments)
        2. **Fitness Evaluation:** Scores each solution using weighted cost-distance objective function
        3. **Selection Mechanism:** Tournament selection chooses parents based on fitness ranking
        4. **Crossover Operation:** Order Crossover (OX) creates offspring by combining parent solutions
        5. **Mutation Process:** Adaptive assignment mutation explores new solution neighborhoods
        6. **Constraint Repair:** Ensures all generated solutions remain feasible (capacity, assignment)
        7. **Evolution Loop:** Iterates through generations until convergence or time limit
        
        **Genetic Operators:**
        - **Selection:** Tournament selection with configurable tournament size
        - **Crossover:** Order Crossover (OX) preserving route structure
        - **Mutation:** Adaptive assignment mutation with configurable rate
        - **Replacement:** Elitist strategy maintaining best solutions
        
        **Algorithm Parameters:**
        - **Population Size:** 20-100 individuals (default: 50)
        - **Generations:** 50-300 iterations (default: 100)
        - **Mutation Rate:** 5-30% probability (default: 10%)
        - **Elite Size:** Top solutions preserved each generation
        
        **Performance Characteristics:**
        - **Speed:** Fast initial convergence (often < 10 seconds)
        - **Memory:** Moderate usage scales with population size
        - **Scalability:** Works well for large problems (100+ orders)
        - **Solution Quality:** Generally good solutions with variety
        
        **Multi-Objective Optimization:**
        - Fitness combines normalized cost and distance with equal importance
        - Fixed weighting: 50% cost + 50% distance for balanced optimization
        - Automatically balances operational costs with travel efficiency
        """) 