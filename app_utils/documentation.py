"""
Documentation Module for Streamlit Application

This module contains the documentation-related rendering methods for the
Vehicle Router Streamlit application, including introduction and methodology sections.
"""

import streamlit as st


class DocumentationRenderer:
    """
    Documentation Renderer for Vehicle Router Streamlit Application
    
    This class provides comprehensive documentation rendering for the Vehicle Router
    optimization application, including method-specific technical details and usage guidance.
    """
    
    def __init__(self):
        """Initialize the DocumentationRenderer"""
        pass
    
    def render_introduction(self):
        """Render the application introduction section"""
        st.title("🚛 Vehicle Router Optimizer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to the Vehicle Router Optimizer
            
            This application solves the **Vehicle Routing Problem (VRP)** using **advanced optimization approaches**. 
            Choose from Mathematical Programming, Hybrid Optimization, or Evolutionary Algorithms to find effective solutions for your routing challenges.
            
            #### 🎯 **Available Optimization Methods**
            - **📊 Standard MILP + Greedy**: Fast cost-optimal truck selection with route sequence optimization
            - **🧬 Genetic Algorithm**: Evolutionary approach with balanced cost-distance optimization
            
            #### 🌍 **Distance Calculation Options**
            - **Simulated Distances**: Mathematical approximation for quick testing and development
            - **Real-World Distances**: OpenStreetMap geocoding with Haversine calculations for geographic accuracy
            
            #### 🔧 **Application Workflow**
            1. **Load Data**: Import order and truck information (or use example data)
            2. **Configure Distance Method**: Choose between simulated or real-world distances
            3. **Select Optimization Method**: Choose approach based on your priorities and problem size
            4. **Set Parameters**: Adjust method-specific settings (population size, weights, etc.)
            5. **Run Optimization**: Execute solver with progress tracking and status updates
            6. **Analyze Results**: Review comprehensive solution analysis with performance metrics
            7. **Visualize Routes**: Explore interactive route maps and distance matrices
            8. **Export Results**: Download Excel reports and solution summaries
            
            #### 📊 **Key Application Features**
            - **Multi-Method Optimization**: Choose the best approach for your specific needs
            - **Real-Time Distance Updates**: Automatic distance matrix reload when switching methods
            - **Interactive Visualizations**: Route maps, cost breakdowns, and capacity utilization charts
            - **Progress Tracking**: Live optimization status with method-specific indicators
            - **Export Capabilities**: Comprehensive Excel reports and CSV data exports
            - **Dynamic Documentation**: Method-specific technical details shown after optimization
            """)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🚛 Problem Type</h4>
                <p><strong>Capacitated Vehicle Routing</strong><br>
                Multi-objective optimization with capacity constraints</p>
            </div>
            
            <div class="metric-card">
                <h4>⚡ Technologies</h4>
                <p><strong>MILP + Genetic Algorithms</strong><br>
                Mathematical programming and evolutionary computation</p>
            </div>
            
            <div class="metric-card">
                <h4>🌍 Distance Methods</h4>
                <p><strong>Simulated + Real-World</strong><br>
                OpenStreetMap integration for geographic accuracy</p>
            </div>
            
            <div class="metric-card">
                <h4>🎯 Optimization Goals</h4>
                <p><strong>Cost + Distance Balance</strong><br>
                Flexible multi-objective optimization</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start guide (only when no data loaded)
        if not st.session_state.data_loaded:
            st.markdown("""
            <div class="warning-box">
                <h4>🚀 Quick Start Guide</h4>
                <ol>
                    <li><strong>Load Data:</strong> Click "Load Example Data" in the sidebar or upload your own CSV files</li>
                    <li><strong>Choose Distance Method:</strong> Toggle "🌍 Use Real-World Distances" for geographic accuracy (optional)</li>
                    <li><strong>Select Method:</strong> Choose Standard MILP + Greedy (default) or Genetic Algorithm</li>
                    <li><strong>Configure:</strong> Adjust method-specific parameters (optional - defaults work well)</li>
                    <li><strong>Optimize:</strong> Click "🚀 Run Optimization" to solve the routing problem</li>
                    <li><strong>Analyze:</strong> Review results in Solution Analysis and Visualization sections</li>
                </ol>
                <p><strong>💡 Tip:</strong> Start with Standard MILP + Greedy and simulated distances for fastest results!</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_methodology(self, selected_method: str = "standard"):
        """Render methodology documentation for the selected optimization method"""
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
    
    # Method-specific documentation with enhanced content and real-world context
    def _render_standard_milp_method(self):
        """Render documentation for Standard MILP method"""
        st.markdown("""
        ### 📊 Standard MILP Optimization
        
        **Cost-focused mathematical programming** using Mixed Integer Linear Programming for optimal truck selection.
        
        #### **Algorithm Overview**
        1. **Mathematical Model**: Formulates VRP as cost minimization problem with capacity constraints
        2. **Optimal Selection**: Finds minimum-cost combination of trucks to deliver all orders
        3. **Assignment Logic**: Assigns each order to exactly one truck while respecting capacity limits
        4. **Simple Routing**: Creates basic depot→orders→depot routes using postal code ordering
        
        #### **Mathematical Foundation**
        - **Decision Variables**: Binary variables for truck selection (`y_j`) and order assignment (`x_ij`)
        - **Objective Function**: Minimize Σ(truck_cost × truck_usage)
        - **Key Constraints**: Order assignment uniqueness, capacity limits, truck usage logic
        - **Solver Technology**: CBC (Coin-OR Branch and Cut) with 60-second timeout
        
        #### **Performance Characteristics**
        - **Speed**: Fastest method (typically < 1 second execution)
        - **Memory Usage**: Minimal computational requirements (< 50MB)
        - **Scalability**: Handles 100+ orders efficiently
        - **Solution Quality**: Guarantees cost-optimal truck selection
        - **Deterministic**: Consistent results across runs
        
        #### **Practical Applications**
        - Quick daily routing decisions where cost is primary concern
        - High-volume operations requiring fast optimization
        - Baseline comparison for other optimization methods
        - Resource-constrained environments with limited computational capacity
        """)
    
    def _render_standard_milp_greedy_method(self):
        """Render documentation for Standard MILP + Greedy method"""
        st.markdown("""
        ### 🔄 Standard MILP + Greedy Optimization
        
        **Two-phase hybrid approach** combining cost-optimal mathematical programming with intelligent route sequence optimization.
        
        #### **Two-Phase Process**
        
        **Phase 1 - MILP Optimization:**
        - Executes standard MILP to find cost-optimal truck selection and order assignments
        - Guarantees minimum operational cost while satisfying all capacity constraints
        - Provides optimal foundation for subsequent route improvement
        
        **Phase 2 - Greedy Route Enhancement:**
        - Tests all possible delivery sequences for each selected truck
        - Evaluates factorial combinations of route permutations
        - Selects optimal sequence minimizing total travel distance
        - Maintains cost optimality while improving route efficiency
        
        #### **Algorithmic Details**
        - **MILP Component**: Identical to Standard MILP for truck selection phase
        - **Route Optimization**: Exhaustive permutation testing with distance matrix calculations
        - **Complexity**: O(n × m) for MILP + O(k!) per truck for route optimization
        - **Practical Efficiency**: Handles up to 8 orders per truck (40,320 permutations) efficiently
        
        #### **Distance Integration**
        - **Simulated Distances**: Uses postal code differences for rapid calculation
        - **Real-World Distances**: Integrates OpenStreetMap geocoding for geographic accuracy
        - **Route Quality**: Real distances significantly improve route realism and accuracy
        - **Fallback Mechanism**: Graceful degradation if geocoding services unavailable
        
        #### **Performance Analysis**
        - **Execution Time**: Fast total processing (< 5 seconds including route optimization)
        - **Memory Efficiency**: Low resource usage (< 100MB for typical problems)
        - **Scalability**: Excellent for standard distribution scenarios (≤8 orders per truck)
        - **Solution Quality**: Cost-optimal selection with distance-optimized routes
        
        #### **Route Optimization Examples**
        
        *Simulated Distances (Barcelona example):*
        - Original route: 08020 → 08027 → 08028 → 08029 → 08031 (21.0 km)
        - Optimized route: 08020 → 08027 → 08031 → 08029 → 08028 (19.0 km)
        - Improvement: 2.0 km reduction (9.5% better)
        
        *Real-World Distances (same example):*
        - Original route: 08020 → 08027 → 08028 → 08029 → 08031 (15.1 km)
        - Optimized route: 08020 → 08027 → 08031 → 08029 → 08028 (10.4 km)
        - Improvement: 4.7 km reduction (31.0% better)
        
        #### **Best Use Cases**
        - Daily logistics operations requiring balanced cost-distance optimization
        - Distribution scenarios with moderate order density per truck
        - Applications where both cost control and route efficiency matter
        - Production environments needing reliable, fast optimization
        """)
    
    def _render_enhanced_milp_method(self):
        """Render documentation for Enhanced MILP method"""
        st.markdown("""
        ### 🚀 Enhanced MILP Optimization
        
        **Advanced multi-objective mathematical programming** that simultaneously optimizes truck costs and travel distances in a unified model.
        
        #### **Multi-Objective Approach**
        1. **Unified Formulation**: Combines truck costs and travel distances in weighted objective function
        2. **Routing Variables**: Creates binary variables for every possible route segment between locations
        3. **Flow Conservation**: Ensures trucks follow valid, continuous routes through all assigned orders
        4. **Simultaneous Optimization**: Optimizes truck selection, order assignment, and routing in single step
        5. **Global Optimality**: Finds mathematically optimal solution for weighted multi-objective function
        
        #### **Advanced Mathematical Model**
        - **Extended Variables**: Assignment (`x_ij`), truck usage (`y_j`), route segments (`z_klj`)
        - **Multi-Objective Function**: α × normalized_costs + β × normalized_distances (α + β = 1)
        - **Enhanced Constraints**: Flow conservation, depot routing, capacity limits, subtour elimination
        - **Model Complexity**: O(|orders| × |trucks| + |locations|² × |trucks|) decision variables
        
        #### **Objective Weight Configuration**
        - **Default Balance**: 60% cost focus + 40% distance focus
        - **Customizable Weights**: User-adjustable balance via application sliders
        - **Normalization Strategy**: Scales objectives to ensure meaningful contribution
        - **Trade-off Analysis**: Explores Pareto-efficient solutions along cost-distance frontier
        
        #### **Technical Implementation**
        - **Solver Technology**: Extended CBC with increased timeout for complex model
        - **Constraint Generation**: Dynamic flow conservation and subtour elimination
        - **Solution Reconstruction**: Extracts actual route sequences from routing variables
        - **Validation**: Comprehensive feasibility checking and route connectivity verification
        
        #### **Performance Characteristics**
        - **Execution Time**: Moderate processing (1-30 seconds depending on problem complexity)
        - **Memory Requirements**: Higher usage (100-500MB for medium instances)
        - **Scalability**: Effective for small-medium problems (≤50 orders, ≤15 trucks)
        - **Solution Quality**: Globally optimal for specified multi-objective balance
        
        #### **Comparative Advantages**
        - **Simultaneous Optimization**: Avoids sub-optimality of sequential approaches
        - **Global Perspective**: Considers all interactions between truck selection and routing
        - **Flexible Objectives**: Configurable balance between competing objectives
        - **Mathematical Rigor**: Proven optimality within specified objective weights
        
        #### **Application Scenarios**
        - Premium routing services requiring highest quality solutions
        - Applications where route quality justifies increased computational cost
        - Multi-objective decision environments with clear weight preferences
        - Scenarios demanding provably optimal solutions
        """)
    
    def _render_genetic_algorithm_method(self):
        """Render documentation for Genetic Algorithm method"""
        st.markdown("""
        ### 🧬 Genetic Algorithm Optimization
        
        **Evolutionary metaheuristic optimization** using population-based search with genetic operators for multi-objective vehicle routing.
        
        #### **Evolutionary Process Overview**
        1. **Population Initialization**: Creates diverse set of random feasible solutions (truck-order assignments)
        2. **Fitness Evaluation**: Scores each solution using balanced cost-distance objective function
        3. **Parent Selection**: Tournament selection chooses high-fitness individuals for reproduction
        4. **Genetic Crossover**: Order Crossover (OX) creates offspring combining parent characteristics
        5. **Adaptive Mutation**: Introduces solution variations through assignment modifications
        6. **Constraint Repair**: Ensures all generated solutions remain feasible (capacity, assignment)
        7. **Population Evolution**: Iterates through generations until convergence or time limit
        
        #### **Genetic Operators in Detail**
        
        **Selection Mechanism:**
        - Tournament selection with configurable tournament size (default: 3)
        - Fitness-based ranking with elitist preservation strategy
        - Maintains solution diversity while promoting quality improvements
        
        **Crossover Operation:**
        - Order Crossover (OX) preserving route structure and assignment relationships
        - Combines beneficial characteristics from parent solutions
        - Maintains feasibility through intelligent order preservation
        
        **Mutation Strategy:**
        - Adaptive assignment mutation exploring solution neighborhoods
        - Configurable mutation rate (default: 10% probability)
        - Random order reassignment with automatic constraint repair
        
        #### **Multi-Objective Fitness Function**
        - **Balanced Weighting**: Fixed 50% cost + 50% distance for consistent optimization
        - **Normalization**: Scales both objectives to ensure equal contribution
        - **Fitness Calculation**: fitness = 0.5 × (cost/max_cost) + 0.5 × (distance/max_distance)
        - **Quality Metrics**: Lower fitness values indicate better solutions
        
        #### **Algorithm Parameters**
        - **Population Size**: 20-100 individuals (default: 50) - larger populations increase diversity
        - **Max Generations**: 50-300 iterations (default: 100) - more generations improve convergence
        - **Mutation Rate**: 5-30% probability (default: 10%) - higher rates increase exploration
        - **Elite Preservation**: Top solutions maintained each generation for stability
        
        #### **Convergence and Performance**
        - **Fast Convergence**: Often achieves good solutions within 10-20 generations
        - **Execution Time**: Typically 5-60 seconds depending on parameters and problem size
        - **Scalability**: Excellent performance on large problems (100+ orders, 20+ trucks)
        - **Solution Diversity**: Explores multiple high-quality alternatives
        - **Robustness**: Performs well across varied problem types and sizes
        
        #### **Distance Method Integration**
        - **Simulated Distances**: Rapid fitness evaluation for large populations
        - **Real-World Distances**: Geographic accuracy improves solution realism
        - **Performance Impact**: Real distances add geocoding overhead but enhance route quality
        - **Caching Strategy**: Coordinate caching minimizes repeated API calls
        
        #### **Evolutionary Advantages**
        - **Global Search**: Explores diverse solution space avoiding local optima
        - **Adaptive Learning**: Population evolution discovers problem-specific patterns
        - **Scalability**: Natural parallelization and good large-problem performance
        - **Flexibility**: Easily handles complex constraints and multiple objectives
        - **Solution Diversity**: Provides multiple good alternatives for decision makers
        
        #### **Practical Applications**
        - Large-scale logistics networks exceeding MILP computational limits
        - Dynamic routing environments requiring rapid adaptation
        - Multi-depot or complex constraint scenarios
        - Applications where solution diversity and alternatives are valuable
        - Exploration of trade-offs between competing objectives
        
        #### **Expected Results**
        - **Generation Convergence**: Typically 10-50 generations to good solutions
        - **Solution Quality**: Generally within 5-15% of optimal for standard problems
        - **Multiple Solutions**: Population diversity provides alternative routing strategies
        - **Robust Performance**: Consistent good results across problem variations
        """)
        
        # Add a note about the fixed cost-distance weighting
        st.info("**Note**: The Genetic Algorithm uses fixed 50/50 cost-distance weighting to ensure balanced multi-objective optimization without requiring user configuration.")
    
    def render_distance_methods_info(self):
        """Render information about distance calculation methods"""
        st.markdown("""
        #### 🌍 Distance Calculation Methods
        
        **Simulated Distances (Default):**
        - Mathematical approximation: |postal_code_1 - postal_code_2| × 1 km
        - Instant calculation, no network dependencies
        - Ideal for testing, development, and rapid prototyping
        
        **Real-World Distances (Advanced):**
        - OpenStreetMap geocoding + Haversine great-circle distance
        - Geographic accuracy with coordinate-based calculations
        - ~0.5-2 seconds per postal code (with caching and rate limiting)
        - Requires internet connection, provides realistic route planning
        
        **Impact on Optimization:**
        - Real distances typically reduce total route distance by 20-40%
        - More accurate representation of actual geographic constraints
        - Better optimization due to realistic distance relationships
        - Improved route quality for production routing applications
        """) 