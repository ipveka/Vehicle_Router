#!/usr/bin/env python3
"""
Vehicle Router Streamlit Application

This Streamlit application provides an interactive web interface for the Vehicle Routing
Problem optimization system. It allows users to explore data, run optimizations, 
analyze solutions, and visualize results through a user-friendly web interface.

Usage:
    streamlit run app/streamlit_app.py

Author: Vehicle Router Team
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
            
            # Optimization parameters
            st.sidebar.markdown("**Solver Parameters:**")
            solver_timeout = st.sidebar.slider(
                "Solver Timeout (seconds)", 
                min_value=10, max_value=300, value=60,
                help="Maximum time allowed for optimization"
            )
            
            enable_validation = st.sidebar.checkbox(
                "Enable Solution Validation", 
                value=True,
                help="Validate the solution for constraint compliance"
            )
            
            # Run optimization button
            if st.sidebar.button("🚀 Run Optimization", type="primary"):
                with st.spinner("Running optimization..."):
                    success = self.optimization_runner.run_optimization(
                        st.session_state.orders_df,
                        st.session_state.trucks_df,
                        st.session_state.distance_matrix,
                        solver_timeout=solver_timeout,
                        enable_validation=enable_validation
                    )
                    
                    if success:
                        st.session_state.solution = self.optimization_runner.solution
                        st.session_state.optimization_log = self.optimization_runner.optimization_log
                        st.session_state.optimization_complete = True
                        st.sidebar.success("✅ Optimization completed successfully!")
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
                fig = self.visualization_manager.create_orders_location_chart(st.session_state.orders_df)
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
            
            # Distance matrix heatmap
            fig = self.visualization_manager.create_distance_heatmap(st.session_state.distance_matrix)
            st.plotly_chart(fig, use_container_width=True)
    
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
        
        **Truck Assignments:**
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
        tab1, tab2, tab3 = st.tabs(["🗺️ Route Visualization", "💰 Cost Analysis", "📊 Utilization Analysis"])
        
        with tab1:
            st.markdown("### Delivery Routes Visualization")
            fig = self.visualization_manager.create_route_visualization(
                solution['routes_df'],
                st.session_state.orders_df,
                st.session_state.trucks_df
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Cost Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = self.visualization_manager.create_cost_breakdown_chart(
                    solution['costs'],
                    st.session_state.trucks_df
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = self.visualization_manager.create_cost_efficiency_chart(
                    solution['costs'],
                    solution['utilization'],
                    st.session_state.trucks_df
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Capacity Utilization Analysis")
            
            fig = self.visualization_manager.create_utilization_chart(
                solution['utilization'],
                st.session_state.trucks_df
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_methodology(self):
        """Render the methodology section"""
        st.markdown('<h2 class="section-header">🔬 Methodology</h2>', unsafe_allow_html=True)
        
        # Methodology tabs
        tab1, tab2, tab3 = st.tabs(["🧮 Mathematical Model", "⚙️ Algorithm Details", "📚 Technical References"])
        
        with tab1:
            st.markdown("""
            ### Mixed Integer Linear Programming (MILP) Formulation
            
            The Vehicle Routing Problem with order assignment is formulated as a Mixed Integer Linear Programming problem:
            
            #### Decision Variables
            - **x[i,j]**: Binary variable = 1 if order i is assigned to truck j, 0 otherwise
            - **y[j]**: Binary variable = 1 if truck j is used in the solution, 0 otherwise
            
            #### Objective Function
            ```
            Minimize: Σ(cost[j] × y[j])
            ```
            Where cost[j] is the operational cost of truck j.
            
            #### Constraints
            
            **1. Order Assignment Constraint**
            ```
            Σ(x[i,j]) = 1  ∀ orders i
            ```
            Each order must be assigned to exactly one truck.
            
            **2. Capacity Constraint**
            ```
            Σ(volume[i] × x[i,j]) ≤ capacity[j] × y[j]  ∀ trucks j
            ```
            Total volume assigned to each truck cannot exceed its capacity.
            
            **3. Truck Usage Constraint**
            ```
            y[j] ≥ x[i,j]  ∀ orders i, trucks j
            ```
            If any order is assigned to a truck, the truck must be marked as used.
            """)
        
        with tab2:
            st.markdown("""
            ### Algorithm Implementation Details
            
            #### Solver Technology
            - **Optimization Engine**: PuLP (Python Linear Programming)
            - **Backend Solver**: CBC (Coin-or Branch and Cut)
            - **Problem Type**: Mixed Integer Linear Programming (MILP)
            - **Solution Method**: Branch-and-bound with cutting planes
            
            #### Algorithm Steps
            
            1. **Problem Preprocessing**
               - Validate input data for consistency and feasibility
               - Check that total truck capacity ≥ total order volume
               - Generate distance matrix from postal codes
            
            2. **Model Construction**
               - Create binary decision variables for assignments and truck usage
               - Define objective function (minimize total truck costs)
               - Add all constraint equations to the model
            
            3. **Optimization Execution**
               - Initialize the CBC solver with default parameters
               - Apply branch-and-bound algorithm to explore solution space
               - Use cutting planes to tighten linear relaxation bounds
               - Continue until optimal solution found or timeout reached
            
            4. **Solution Extraction**
               - Extract variable values from solved model
               - Construct assignment matrix and truck selection
               - Calculate utilization metrics and cost breakdown
            
            5. **Solution Validation**
               - Verify all constraints are satisfied
               - Check capacity limits and order assignments
               - Validate solution feasibility and optimality
            """)
        
        with tab3:
            st.markdown("""
            ### Technical References and Resources
            
            #### Academic Background
            
            **Vehicle Routing Problem (VRP)**
            - Classic combinatorial optimization problem in operations research
            - First introduced by Dantzig and Ramser (1959)
            - Extensive literature on variants and solution methods
            - Applications in logistics, transportation, and supply chain management
            
            **Mixed Integer Linear Programming**
            - Mathematical optimization technique for discrete decision problems
            - Combines continuous linear programming with integer constraints
            - Solved using branch-and-bound and cutting plane methods
            - Guarantees optimal solutions for well-formulated problems
            
            #### Software Libraries and Tools
            
            **PuLP (Python Linear Programming)**
            - Open-source linear programming modeler for Python
            - Provides high-level interface to multiple solvers
            - Supports MILP, LP, and other optimization problem types
            - Documentation: https://coin-or.github.io/pulp/
            
            **CBC (Coin-or Branch and Cut)**
            - Open-source mixed integer programming solver
            - Part of the COIN-OR optimization suite
            - Implements advanced branch-and-cut algorithms
            - High performance for medium to large-scale problems
            """)


def main():
    """Main entry point for the Streamlit application"""
    app = VehicleRouterApp()
    app.run()


if __name__ == "__main__":
    main()