#!/usr/bin/env python3
"""
Vehicle Router Streamlit Application

This Streamlit application provides an interactive web interface for the Vehicle Routing
Problem optimization system. It allows users to explore data, run optimizations, 
analyze solutions, and visualize results through a user-friendly web interface.

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import application utilities
from app_utils.data_handler import DataHandler
from app_utils.optimization_runner import OptimizationRunner
from app_utils.visualization_manager import VisualizationManager
from app_utils.export_manager import ExportManager
from app_utils.ui_components import UIComponents

# Configure Streamlit page
st.set_page_config(
    page_title="Vehicle Router Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class VehicleRouterApp:
    """Main Streamlit Application Class for Vehicle Router"""
    
    def __init__(self):
        """Initialize the Streamlit application"""
        self.initialize_session_state()
        self.data_handler = DataHandler()
        self.optimization_runner = OptimizationRunner()
        self.visualization_manager = VisualizationManager()
        self.export_manager = ExportManager()
        self.ui_components = UIComponents()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'optimization_complete' not in st.session_state:
            st.session_state.optimization_complete = False
        if 'orders_df' not in st.session_state:
            st.session_state.orders_df = None
        if 'trucks_df' not in st.session_state:
            st.session_state.trucks_df = None
        if 'distance_matrix' not in st.session_state:
            st.session_state.distance_matrix = None
        if 'solution' not in st.session_state:
            st.session_state.solution = None
        if 'optimization_log' not in st.session_state:
            st.session_state.optimization_log = []
    
    def run(self):
        """Run the complete Streamlit application"""
        # Main header
        st.markdown('<h1 class="main-header">🚛 Vehicle Router Optimizer</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content sections
        self.render_introduction()
        
        if st.session_state.data_loaded:
            self.render_data_exploration()
            
            if st.session_state.optimization_complete:
                self.render_solution_analysis()
                self.render_visualization()
        
        self.render_methodology()
    
    def render_sidebar(self):
        """Render the sidebar with data loading and optimization controls"""
        st.sidebar.markdown("## 🔧 Control Panel")
        
        # Data Loading Section
        st.sidebar.markdown("### 📊 Data Management")
        
        data_source = st.sidebar.radio(
            "Select Data Source:",
            ["Example Data", "Upload Custom Data"],
            help="Choose between predefined example data or upload your own CSV files"
        )
        
        if data_source == "Example Data":
            if st.sidebar.button("🔄 Load Example Data", type="primary"):
                with st.spinner("Loading example data..."):
                    success = self.data_handler.load_example_data()
                    if success:
                        st.session_state.orders_df = self.data_handler.orders_df
                        st.session_state.trucks_df = self.data_handler.trucks_df
                        st.session_state.distance_matrix = self.data_handler.distance_matrix
                        st.session_state.data_loaded = True
                        st.sidebar.success("✅ Example data loaded successfully!")
                    else:
                        st.sidebar.error("❌ Failed to load example data")
        
        else:
            st.sidebar.markdown("Upload your CSV files:")
            
            orders_file = st.sidebar.file_uploader(
                "Orders CSV", type=['csv'],
                help="CSV with columns: order_id, volume, postal_code"
            )
            trucks_file = st.sidebar.file_uploader(
                "Trucks CSV", type=['csv'],
                help="CSV with columns: truck_id, capacity, cost"
            )
            
            if orders_file and trucks_file:
                if st.sidebar.button("📤 Load Custom Data", type="primary"):
                    with st.spinner("Loading custom data..."):
                        success = self.data_handler.load_custom_data(orders_file, trucks_file)
                        if success:
                            st.session_state.orders_df = self.data_handler.orders_df
                            st.session_state.trucks_df = self.data_handler.trucks_df
                            st.session_state.distance_matrix = self.data_handler.distance_matrix
                            st.session_state.data_loaded = True
                            st.sidebar.success("✅ Custom data loaded successfully!")
                        else:
                            st.sidebar.error("❌ Failed to load custom data")
        
        # Optimization Section
        if st.session_state.data_loaded:
            st.sidebar.markdown("### 🎯 Optimization")
            
            # Model selection
            st.sidebar.markdown("**Model Configuration:**")
            use_enhanced_model = st.sidebar.checkbox(
                "🚀 Enhanced Model with Distance Optimization", 
                value=True,
                help="Use advanced model that minimizes both truck costs and travel distances"
            )
            
            # Objective weights (only for enhanced model)
            if use_enhanced_model:
                st.sidebar.markdown("**Objective Weights:**")
                cost_weight = st.sidebar.slider(
                    "Cost Weight", 
                    min_value=0.0, max_value=1.0, value=0.6, step=0.1,
                    help="Weight for truck costs in objective function"
                )
                distance_weight = st.sidebar.slider(
                    "Distance Weight", 
                    min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                    help="Weight for travel distances in objective function"
                )
                
                # Normalize weights to sum to 1
                total_weight = cost_weight + distance_weight
                if total_weight > 0:
                    cost_weight = cost_weight / total_weight
                    distance_weight = distance_weight / total_weight
                    st.sidebar.info(f"Normalized weights: Cost={cost_weight:.2f}, Distance={distance_weight:.2f}")
            
            # Solver parameters
            st.sidebar.markdown("**Solver Parameters:**")
            solver_timeout = st.sidebar.slider(
                "Solver Timeout (seconds)", 
                min_value=30, max_value=600, value=120 if use_enhanced_model else 60,
                help="Maximum time allowed for optimization (enhanced model needs more time)"
            )
            
            enable_validation = st.sidebar.checkbox(
                "Enable Solution Validation", 
                value=True,
                help="Validate the solution for constraint compliance"
            )
            
            # Run optimization button
            if st.sidebar.button("🚀 Run Optimization", type="primary"):
                with st.spinner("Running optimization..."):
                    if use_enhanced_model:
                        success = self.optimization_runner.run_optimization(
                            st.session_state.orders_df,
                            st.session_state.trucks_df,
                            st.session_state.distance_matrix,
                            solver_timeout=solver_timeout,
                            enable_validation=enable_validation,
                            use_enhanced_model=True,
                            cost_weight=cost_weight,
                            distance_weight=distance_weight
                        )
                    else:
                        success = self.optimization_runner.run_optimization(
                            st.session_state.orders_df,
                            st.session_state.trucks_df,
                            st.session_state.distance_matrix,
                            solver_timeout=solver_timeout,
                            enable_validation=enable_validation,
                            use_enhanced_model=False
                        )
                    
                    if success:
                        st.session_state.solution = self.optimization_runner.solution
                        st.session_state.optimization_log = self.optimization_runner.optimization_log
                        st.session_state.optimization_complete = True
                        st.sidebar.success("✅ Optimization completed successfully!")
                        
                        # Show enhanced metrics if available
                        if use_enhanced_model and 'costs' in st.session_state.solution:
                            if 'total_distance' in st.session_state.solution['costs']:
                                total_distance = st.session_state.solution['costs']['total_distance']
                                st.sidebar.info(f"📏 Total Distance: {total_distance:.1f} km")
                    else:
                        st.sidebar.error("❌ Optimization failed")
        
        # Data Status
        st.sidebar.markdown("### 📈 Status")
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data Loaded")
            st.sidebar.info(f"Orders: {len(st.session_state.orders_df)}")
            st.sidebar.info(f"Trucks: {len(st.session_state.trucks_df)}")
        else:
            st.sidebar.warning("⚠️ No Data Loaded")
        
        if st.session_state.optimization_complete:
            st.sidebar.success("✅ Optimization Complete")
        else:
            st.sidebar.info("⏳ Optimization Pending")
    
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
    
    def render_data_exploration(self):
        """Render the data exploration section"""
        st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
        
        # Data overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_orders = len(st.session_state.orders_df)
        total_volume = st.session_state.orders_df['volume'].sum()
        total_trucks = len(st.session_state.trucks_df)
        total_capacity = st.session_state.trucks_df['capacity'].sum()
        
        with col1:
            st.metric("Total Orders", total_orders)
        with col2:
            st.metric("Total Volume", f"{total_volume:.1f} m³")
        with col3:
            st.metric("Available Trucks", total_trucks)
        with col4:
            st.metric("Total Capacity", f"{total_capacity:.1f} m³")
        
        # Feasibility check
        feasible = total_capacity >= total_volume
        utilization = (total_volume / total_capacity) * 100 if total_capacity > 0 else 0
        
        if feasible:
            st.markdown(f"""
            <div class="success-box">
                ✅ <strong>Problem is Feasible</strong><br>
                Total capacity ({total_capacity:.1f} m³) exceeds total volume ({total_volume:.1f} m³)<br>
                Expected capacity utilization: {utilization:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <strong>Problem may be Infeasible</strong><br>
                Total volume ({total_volume:.1f} m³) exceeds total capacity ({total_capacity:.1f} m³)<br>
                Additional trucks or capacity may be required.
            </div>
            """, unsafe_allow_html=True)
        
        # Data tables
        tab1, tab2, tab3 = st.tabs(["📦 Orders Data", "🚛 Trucks Data", "📏 Distance Matrix"])
        
        with tab1:
            st.markdown("### Orders Information")
            st.dataframe(
                st.session_state.orders_df,
                use_container_width=True,
                column_config={
                    "order_id": st.column_config.TextColumn("Order ID", width="small"),
                    "volume": st.column_config.NumberColumn("Volume (m³)", format="%.1f"),
                    "postal_code": st.column_config.TextColumn("Postal Code", width="medium")
                }
            )
            
            # Orders visualization
            col1, col2 = st.columns(2)
            with col1:
                fig = self.visualization_manager.create_orders_volume_chart(st.session_state.orders_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = self.visualization_manager.create_orders_distribution_chart(st.session_state.orders_df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Trucks Information")
            st.dataframe(
                st.session_state.trucks_df,
                use_container_width=True,
                column_config={
                    "truck_id": st.column_config.NumberColumn("Truck ID", width="small"),
                    "capacity": st.column_config.NumberColumn("Capacity (m³)", format="%.1f"),
                    "cost": st.column_config.NumberColumn("Cost (€)", format="%.0f")
                }
            )
            
            # Trucks visualization
            col1, col2 = st.columns(2)
            with col1:
                fig = self.visualization_manager.create_trucks_capacity_chart(st.session_state.trucks_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = self.visualization_manager.create_trucks_cost_efficiency_chart(st.session_state.trucks_df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Distance Matrix")
            st.dataframe(
                st.session_state.distance_matrix,
                use_container_width=True
            )
    
    def render_solution_analysis(self):
        """Render the solution analysis section"""
        st.markdown('<h2 class="section-header">🎯 Solution Analysis</h2>', unsafe_allow_html=True)
        
        solution = st.session_state.solution
        
        # Solution overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cost", f"€{solution['costs']['total_cost']:.0f}")
        with col2:
            st.metric("Trucks Used", f"{len(solution['selected_trucks'])}/{len(st.session_state.trucks_df)}")
        with col3:
            st.metric("Orders Delivered", f"{len(solution['assignments_df'])}/{len(st.session_state.orders_df)}")
        with col4:
            avg_util = np.mean([u['utilization_percent'] for u in solution['utilization'].values()])
            st.metric("Avg Utilization", f"{avg_util:.1f}%")
        
        # Solution summary
        st.markdown("### 📋 Solution Summary")
        
        summary_text = f"""
        **Selected Trucks:** {solution['selected_trucks']}
        """
        
        for truck_id in solution['selected_trucks']:
            assigned_orders = [a['order_id'] for a in solution['assignments_df'].to_dict('records') if a['truck_id'] == truck_id]
            summary_text += f"\n- Truck {truck_id} → Orders {assigned_orders}"
        
        summary_text += f"\n\n**Total Cost:** €{solution['costs']['total_cost']:.0f}"
        
        st.markdown(summary_text)
        
        # Export section
        st.markdown("### 📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to Excel", type="primary"):
                excel_buffer = self.export_manager.create_excel_report(
                    st.session_state.orders_df,
                    st.session_state.trucks_df,
                    solution
                )
                
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_buffer,
                    file_name=f"vehicle_router_solution_{int(time.time())}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📋 Export Summary CSV"):
                csv_buffer = self.export_manager.create_summary_csv(solution)
                st.download_button(
                    label="⬇️ Download Summary CSV",
                    data=csv_buffer,
                    file_name=f"solution_summary_{int(time.time())}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📄 Export Detailed Report"):
                report_buffer = self.export_manager.create_detailed_report(
                    st.session_state.orders_df,
                    st.session_state.trucks_df,
                    solution
                )
                st.download_button(
                    label="⬇️ Download Report",
                    data=report_buffer,
                    file_name=f"detailed_report_{int(time.time())}.txt",
                    mime="text/plain"
                )
    
    def render_visualization(self):
        """Render the visualization section"""
        st.markdown('<h2 class="section-header">📈 Visualization</h2>', unsafe_allow_html=True)
        
        solution = st.session_state.solution
        
        # Visualization tabs
        tab1, tab2 = st.tabs(["🗺️ Route Visualization", "📊 Solution Analysis"])
        
        with tab1:
            st.markdown("### Delivery Routes Visualization")
            fig = self.visualization_manager.create_route_visualization(
                solution['routes_df'],
                st.session_state.orders_df,
                st.session_state.trucks_df
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Solution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cost Breakdown by Truck")
                fig = self.visualization_manager.create_cost_breakdown_chart(
                    solution['costs'],
                    st.session_state.trucks_df
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Volume Usage by Truck")
                fig = self.visualization_manager.create_volume_usage_chart(
                    solution['utilization'],
                    st.session_state.trucks_df
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_methodology(self):
        """Render the methodology section"""
        st.markdown('<h2 class="section-header">🔬 Methodology</h2>', unsafe_allow_html=True)
        
        # Methodology tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🧮 Mathematical Model", "⚙️ Algorithm Implementation", "🔍 Solution Process", "📚 Technical References"])
        
        with tab1:
            st.markdown("""
            ### Mixed Integer Linear Programming (MILP) Formulation
            
            The Vehicle Router implements a **Capacitated Vehicle Routing Problem (CVRP)** variant focused on order assignment optimization using Mixed Integer Linear Programming.
            
            #### Problem Definition
            
            **Given:**
            - Set of orders `I = {1, 2, ..., n}` with volumes `v_i` and locations `loc_i`
            - Set of trucks `J = {1, 2, ..., m}` with capacities `cap_j` and costs `c_j`
            - Distance matrix `d_{k,l}` between postal code locations
            
            **Objective:** Minimize total operational costs while satisfying all delivery requirements
            
            #### Decision Variables
            
            The MILP formulation uses **two types of binary decision variables**:
            
            **Order Assignment Variables:**
            ```
            x_{i,j} ∈ {0, 1}  ∀i ∈ I, ∀j ∈ J
            ```
            - `x_{i,j} = 1` if order `i` is assigned to truck `j`
            - `x_{i,j} = 0` otherwise
            - **Total variables:** `|I| × |J|` (e.g., 5 orders × 5 trucks = 25 variables)
            
            **Truck Usage Variables:**
            ```
            y_j ∈ {0, 1}  ∀j ∈ J
            ```
            - `y_j = 1` if truck `j` is selected for use in the solution
            - `y_j = 0` otherwise
            - **Total variables:** `|J|` (e.g., 5 trucks = 5 variables)
            
            #### Objective Function
            
            **Minimize total truck operational costs:**
            ```
            minimize: Σ_{j ∈ J} c_j × y_j
            ```
            
            This cost structure focuses on **truck selection costs** rather than distance-based routing costs, making it suitable for scenarios where:
            - Fixed truck costs dominate variable costs
            - Route optimization is secondary to capacity optimization
            - Simplified distance modeling is acceptable
            
            #### Mathematical Constraints
            
            **1. Order Assignment Constraints (Demand Satisfaction)**
            ```
            Σ_{j ∈ J} x_{i,j} = 1  ∀i ∈ I
            ```
            - **Meaning:** Each order must be assigned to exactly one truck
            - **Count:** `|I|` constraints (one per order)
            - **Purpose:** Ensures all orders are delivered
            
            **2. Capacity Constraints (Resource Limits)**
            ```
            Σ_{i ∈ I} v_i × x_{i,j} ≤ cap_j  ∀j ∈ J
            ```
            - **Meaning:** Total volume assigned to each truck cannot exceed its capacity
            - **Count:** `|J|` constraints (one per truck)
            - **Purpose:** Prevents overloading of trucks
            
            **3. Truck Usage Constraints (Logical Consistency)**
            ```
            y_j ≥ x_{i,j}  ∀i ∈ I, ∀j ∈ J
            ```
            - **Meaning:** If any order is assigned to a truck, the truck must be marked as used
            - **Count:** `|I| × |J|` constraints
            - **Purpose:** Links assignment variables to truck selection variables
            
            #### Complete MILP Formulation
            
            ```
            minimize: Σ_{j ∈ J} c_j × y_j
            
            subject to:
                Σ_{j ∈ J} x_{i,j} = 1                    ∀i ∈ I     (Order assignment)
                Σ_{i ∈ I} v_i × x_{i,j} ≤ cap_j         ∀j ∈ J     (Capacity limits)
                y_j ≥ x_{i,j}                           ∀i ∈ I, ∀j ∈ J  (Truck usage)
                x_{i,j} ∈ {0, 1}                        ∀i ∈ I, ∀j ∈ J  (Binary variables)
                y_j ∈ {0, 1}                            ∀j ∈ J     (Binary variables)
            ```
            
            #### Model Complexity Analysis
            
            **Problem Size:**
            - **Variables:** `|I| × |J| + |J|` binary variables
            - **Constraints:** `|I| + |J| + |I| × |J|` constraints
            - **Example:** 5 orders, 5 trucks → 30 variables, 35 constraints
            
            **Computational Complexity:**
            - **Problem Class:** NP-hard (variant of bin packing problem)
            - **Scalability:** Suitable for up to ~100 orders and ~20 trucks
            - **Solution Quality:** Guarantees optimal solutions for practical instances
            """)
        
        with tab2:
            st.markdown("""
            ### Algorithm Implementation Details
            
            The Vehicle Router uses the **VrpOptimizer class** implemented in Python with comprehensive validation, logging, and error handling.
            
            #### Core Architecture
            
            **Class Structure:**
            ```python
            class VrpOptimizer:
                def __init__(orders_df, trucks_df, distance_matrix)
                def build_model()           # Construct MILP formulation
                def solve()                 # Execute optimization
                def get_solution()          # Extract structured results
            ```
            
            #### Solver Technology Stack
            
            **Optimization Engine:** PuLP (Python Linear Programming)
            - Open-source modeling library for linear programming
            - Provides high-level abstraction over multiple solvers
            - Supports MILP, LP, and other optimization problem types
            - Automatic solver detection and configuration
            
            **Backend Solver:** CBC (Coin-or Branch and Cut)
            - Open-source mixed integer programming solver
            - Part of the COIN-OR optimization suite
            - Implements advanced branch-and-cut algorithms
            - High performance for medium to large-scale problems
            - Default solver with `pulp.PULP_CBC_CMD(msg=0)`
            
            #### Implementation Steps
            
            **1. Data Validation and Preprocessing**
            ```python
            def _validate_input_data():
                # Validate DataFrame structures and required columns
                # Check data types and value ranges
                # Verify problem feasibility (total capacity ≥ total volume)
                # Log validation results and warnings
            ```
            
            **2. Decision Variable Creation**
            ```python
            def _create_decision_variables():
                # Order assignment variables: x[order_id, truck_id]
                # Truck usage variables: y[truck_id]
                # All variables are binary (cat='Binary')
                # Variable naming: "assign_A_to_truck_1", "use_truck_1"
            ```
            
            **3. Objective Function Setup**
            ```python
            def _set_objective_function():
                # Extract truck costs from DataFrame
                # Create objective terms: cost[j] * y[j]
                # Set minimization objective: pulp.lpSum(objective_terms)
            ```
            
            **4. Constraint Addition**
            ```python
            def _add_order_assignment_constraints():
                # For each order: sum(x[i,j] for all j) == 1
                
            def _add_capacity_constraints():
                # For each truck: sum(volume[i] * x[i,j] for all i) <= capacity[j]
                
            def _add_truck_usage_constraints():
                # For each order-truck pair: y[j] >= x[i,j]
            ```
            
            **5. Model Statistics and Logging**
            - **Variable count:** Tracks total decision variables created
            - **Constraint count:** Monitors constraint addition process
            - **Problem size:** Reports model dimensions for performance analysis
            - **Validation status:** Confirms model construction success
            
            #### Solver Configuration
            
            **Default Parameters:**
            - **Solver:** CBC with silent mode (`msg=0`)
            - **Timeout:** Configurable (default: 60 seconds)
            - **Gap tolerance:** Default CBC settings
            - **Threads:** Automatic detection
            
            **Alternative Solvers:**
            - CPLEX: `pulp.CPLEX_CMD()`
            - Gurobi: `pulp.GUROBI_CMD()`
            - GLPK: `pulp.GLPK_CMD()`
            
            #### Error Handling and Robustness
            
            **Input Validation:**
            - DataFrame structure validation
            - Required column presence checks
            - Data type and range validation
            - Feasibility pre-checks
            
            **Solver Error Handling:**
            - Solution status checking (`LpStatusOptimal`, `LpStatusInfeasible`)
            - Timeout handling and graceful degradation
            - Exception catching with informative error messages
            - Logging of solver performance metrics
            
            **Solution Validation:**
            - Constraint satisfaction verification
            - Variable value consistency checks
            - Objective value validation
            - Utilization metric calculations
            """)
        
        with tab3:
            st.markdown("""
            ### Solution Process and Optimization Flow
            
            #### Complete Optimization Workflow
            
            **Phase 1: Initialization and Validation**
            ```
            1. Load input data (orders_df, trucks_df, distance_matrix)
            2. Validate data structure and completeness
            3. Check problem feasibility (capacity vs. demand)
            4. Initialize optimizer with validated data
            5. Log problem dimensions and characteristics
            ```
            
            **Phase 2: Model Construction**
            ```
            1. Create PuLP LpProblem instance with minimization objective
            2. Generate binary decision variables:
               - x[order_id, truck_id] for assignments
               - y[truck_id] for truck usage
            3. Set objective function: minimize Σ(cost[j] × y[j])
            4. Add constraints:
               - Order assignment: each order to exactly one truck
               - Capacity limits: respect truck capacity constraints
               - Truck usage: link assignments to truck selection
            5. Log model statistics (variables, constraints)
            ```
            
            **Phase 3: Optimization Execution**
            ```
            1. Initialize CBC solver with configuration
            2. Start optimization timer
            3. Execute branch-and-bound algorithm:
               - Solve LP relaxation at each node
               - Branch on fractional binary variables
               - Apply cutting planes to tighten bounds
               - Prune infeasible/suboptimal branches
            4. Monitor solution status and termination criteria
            5. Log solving time and final status
            ```
            
            **Phase 4: Solution Extraction and Processing**
            ```
            1. Check optimization status (optimal/infeasible/timeout)
            2. Extract variable values from solved model
            3. Process assignments: identify order-to-truck mappings
            4. Calculate solution metrics:
               - Selected trucks and total cost
               - Truck utilization percentages
               - Volume distribution analysis
            5. Structure results for output and visualization
            ```
            
            #### Branch-and-Bound Algorithm Details
            
            **Tree Search Process:**
            1. **Root Node:** Solve LP relaxation (continuous variables)
            2. **Branching:** Select fractional binary variable, create two branches
            3. **Bounding:** Calculate lower bounds using LP relaxation
            4. **Pruning:** Eliminate nodes that cannot improve best solution
            5. **Termination:** Continue until optimal solution found or timeout
            
            **Cutting Planes Enhancement:**
            - **Valid inequalities:** Add constraints that tighten LP relaxation
            - **Gomory cuts:** General-purpose cuts for integer programs
            - **Problem-specific cuts:** Leverage VRP structure for efficiency
            
            #### Solution Quality Metrics
            
            **Optimality Verification:**
            - **Objective value:** Confirmed optimal cost minimization
            - **Constraint satisfaction:** All constraints verified as satisfied
            - **Variable consistency:** Binary variables have integer values
            - **Feasibility check:** Solution respects all problem requirements
            
            **Performance Indicators:**
            - **Truck utilization:** Percentage of capacity used per selected truck
            - **Cost efficiency:** Cost per unit volume delivered
            - **Resource allocation:** Distribution of orders across trucks
            - **Solution robustness:** Sensitivity to parameter changes
            
            #### Real-World Application Considerations
            
            **Scalability Limits:**
            - **Small instances:** 3-10 orders, 2-5 trucks (seconds)
            - **Medium instances:** 20-50 orders, 5-15 trucks (minutes)
            - **Large instances:** 100+ orders, 20+ trucks (may require heuristics)
            
            **Solution Interpretation:**
            - **Selected trucks:** Optimal subset of available fleet
            - **Order assignments:** Complete delivery plan
            - **Cost breakdown:** Detailed cost analysis by truck
            - **Utilization analysis:** Efficiency metrics for resource usage
            
            **Practical Extensions:**
            - **Time windows:** Add delivery time constraints
            - **Route optimization:** Include distance-based costs
            - **Multi-period:** Plan over multiple time horizons
            - **Stochastic elements:** Handle uncertain demand/travel times
            """)
        
        with tab4:
            st.markdown("""
            ### Technical References and Implementation Details
            
            #### Academic Background
            
            **Vehicle Routing Problem (VRP)**
            - **Origin:** Introduced by Dantzig and Ramser (1959)
            - **Problem class:** NP-hard combinatorial optimization
            - **Variants:** CVRP, VRPTW, MDVRP, HFVRP
            - **Applications:** Logistics, transportation, supply chain management
            - **Literature:** Extensive research with thousands of publications
            
            **Mixed Integer Linear Programming (MILP)**
            - **Foundation:** Linear programming with integer constraints
            - **Solution methods:** Branch-and-bound, cutting planes, branch-and-cut
            - **Optimality:** Guarantees global optimal solutions
            - **Complexity:** Exponential worst-case, polynomial for many practical instances
            - **Software:** Commercial (CPLEX, Gurobi) and open-source (CBC, GLPK) solvers
            
            #### Software Architecture
            
            **PuLP (Python Linear Programming)**
            - **Version:** Compatible with PuLP 2.0+
            - **License:** MIT License (open source)
            - **Features:** High-level modeling, multiple solver interfaces
            - **Documentation:** https://coin-or.github.io/pulp/
            - **Installation:** `pip install pulp`
            
            **CBC (Coin-or Branch and Cut)**
            - **Version:** CBC 2.10+ (bundled with PuLP)
            - **License:** Eclipse Public License
            - **Performance:** Competitive with commercial solvers for many problems
            - **Features:** Parallel processing, advanced cutting planes
            - **Configuration:** Extensive parameter tuning options
            
            #### Implementation Quality Assurance
            
            **Code Structure:**
            - **Modular design:** Separate classes for optimization, validation, visualization
            - **Error handling:** Comprehensive exception management
            - **Logging:** Detailed execution tracking and debugging
            - **Documentation:** Extensive docstrings and comments
            - **Testing:** Unit tests for all major components
            
            **Performance Optimization:**
            - **Data structures:** Efficient pandas DataFrame operations
            - **Memory management:** Minimal memory footprint for large problems
            - **Solver tuning:** Optimized parameters for typical VRP instances
            - **Preprocessing:** Problem reduction and variable fixing
            
            #### Validation and Testing Framework
            
            **Test Coverage:**
            - **Unit tests:** Individual component testing
            - **Integration tests:** End-to-end workflow validation
            - **Performance tests:** Scalability and timing benchmarks
            - **Edge cases:** Boundary conditions and error scenarios
            
            **Benchmarking:**
            - **Standard instances:** Comparison with literature results
            - **Random instances:** Statistical performance analysis
            - **Real-world data:** Validation with actual logistics problems
            - **Solver comparison:** CBC vs. commercial solvers
            
            #### Future Development Roadmap
            
            **Short-term Enhancements:**
            - **Route optimization:** Full TSP integration for distance minimization
            - **Time windows:** Delivery time constraint support
            - **Multi-objective:** Balance cost, time, and service quality
            - **Heuristic methods:** Fast approximate solutions for large instances
            
            **Long-term Vision:**
            - **Dynamic optimization:** Real-time re-optimization capabilities
            - **Machine learning:** Demand prediction and pattern recognition
            - **Cloud deployment:** Scalable web service architecture
            - **Industry integration:** ERP and logistics system connectivity
            
            #### Related Resources
            
            **Academic Papers:**
            - Toth, P., & Vigo, D. (2014). Vehicle Routing: Problems, Methods, and Applications
            - Laporte, G. (2009). Fifty years of vehicle routing. Transportation Science
            - Baldacci, R., et al. (2012). Recent exact algorithms for solving the vehicle routing problem
            
            **Software Tools:**
            - **OR-Tools:** Google's optimization suite
            - **VRPH:** VRP heuristic library
            - **CVRPLIB:** Standard benchmark instances
            - **NetworkX:** Graph algorithms for routing problems
            
            **Online Communities:**
            - **INFORMS:** Institute for Operations Research and Management Sciences
            - **COIN-OR:** Computational Infrastructure for Operations Research
            - **Stack Overflow:** Programming and implementation questions
            - **GitHub:** Open-source VRP implementations and datasets
            """)


def main():
    """Main entry point for the Streamlit application"""
    app = VehicleRouterApp()
    app.run()


if __name__ == "__main__":
    main()