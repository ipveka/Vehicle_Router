"""
Documentation Module for Streamlit Application

This module contains the documentation rendering methods for the Vehicle Router Streamlit application.
"""

import streamlit as st


class DocumentationRenderer:
    """Documentation Renderer for Vehicle Router Streamlit Application"""
    
    def __init__(self):
        """Initialize the DocumentationRenderer"""
        pass
    
    def render_introduction(self):
        """Render the application introduction section"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to the Vehicle Router Optimizer
            
            This application solves the **Vehicle Routing Problem (VRP)** using **Standard MILP + Greedy optimization**. 
            This hybrid approach combines cost-optimal mathematical programming with intelligent route sequence optimization.
            
            #### 🎯 **Optimization Method**
            - **📊 Standard MILP + Greedy**: Fast cost-optimal truck selection with route sequence optimization
            
            #### 🌍 **Distance Calculation**
            - **Simulated Distances**: Mathematical approximation for quick testing
            - **Real-World Distances**: OpenStreetMap geocoding for geographic accuracy
            
            #### 🔧 **How to Use**
            1. **Load Data**: Import orders and trucks (or use example data)
            2. **Choose Distance Method**: Simulated or real-world distances
            3. **Set Parameters**: Adjust max orders per truck, solver timeout
            4. **Run Optimization**: Execute solver with progress tracking
            5. **View Results**: Analyze solution with visualizations and exports
            """)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🚛 Problem Type</h4>
                <p><strong>Capacitated Vehicle Routing</strong><br>
                Cost optimization with capacity constraints</p>
            </div>
            
            <div class="metric-card">
                <h4>⚡ Technology</h4>
                <p><strong>MILP + Greedy</strong><br>
                Mathematical programming with route optimization</p>
            </div>
            
            <div class="metric-card">
                <h4>🌍 Distance Methods</h4>
                <p><strong>Simulated + Real-World</strong><br>
                OpenStreetMap integration</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_methodology(self, selected_method: str = "standard"):
        """Render methodology documentation for the selected optimization method"""
        if selected_method == "standard":
            self._render_standard_milp_greedy_method()
        else:
            # Default to standard method
            self._render_standard_milp_greedy_method()
    
    def _render_standard_milp_greedy_method(self):
        """Render documentation for Standard MILP + Greedy method"""
        st.markdown("""
        ### 📊 Standard MILP + Greedy Optimization
        
        **Two-phase hybrid approach** combining cost-optimal mathematical programming with intelligent route sequence optimization.
        
        #### **How It Works**
        
        **Phase 1 - MILP Optimization:**
        - Finds cost-optimal truck selection and order assignments
        - Guarantees minimum operational cost while respecting capacity constraints
        
        **Phase 2 - Greedy Route Enhancement:**
        - Tests all possible delivery sequences for each selected truck
        - Selects optimal sequence minimizing total travel distance
        - Maintains cost optimality while improving route efficiency
        
        #### **Performance**
        - **Speed**: Fast processing (< 5 seconds including route optimization)
        - **Scalability**: Excellent for standard distribution scenarios (≤8 orders per truck)
        - **Solution Quality**: Cost-optimal selection with distance-optimized routes
        
        #### **Distance Integration**
        - **Simulated Distances**: Uses postal code differences for rapid calculation
        - **Real-World Distances**: Integrates OpenStreetMap geocoding for geographic accuracy
        - **Route Quality**: Real distances significantly improve route realism
        
        #### **Best Use Cases**
        - Daily logistics operations requiring balanced cost-distance optimization
        - Distribution scenarios with moderate order density per truck
        - Applications where both cost control and route efficiency matter
        """)
    
    def render_distance_methods_info(self):
        """Render information about distance calculation methods"""
        st.markdown("""
        #### 🌍 Distance Calculation Methods
        
        **Simulated Distances (Default):**
        - Mathematical approximation: |postal_code_1 - postal_code_2| × 1 km
        - Instant calculation, no network dependencies
        - Ideal for testing and development
        
        **Real-World Distances (Advanced):**
        - OpenStreetMap geocoding + Haversine great-circle distance
        - Geographic accuracy with coordinate-based calculations
        - ~0.5-2 seconds per postal code (with caching and rate limiting)
        - Requires internet connection, provides realistic route planning
        
        **Impact on Optimization:**
        - Real distances typically reduce total route distance by 20-40%
        - More accurate representation of actual geographic constraints
        - Better optimization due to realistic distance relationships
        """)