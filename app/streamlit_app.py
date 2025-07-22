#!/usr/bin/env python3
"""
Vehicle Router Streamlit Application

This Streamlit application provides an interactive web interface for the Vehicle Routing
Problem optimization system. It allows users to explore data, run optimizations, 
analyze solutions, and visualize results through a user-friendly web interface.

Usage:
    streamlit run app/streamlit_app.py0
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
from app_utils.documentation import DocumentationRenderer

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
    
    # App Configuration - Control which models are available
    AVAILABLE_MODELS = {
        'standard': {
            'name': '📊 Standard MILP + Greedy',
            'help': 'MILP + Greedy optimization',
            'enabled': True
        },
        'enhanced': {
            'name': '🚀 Enhanced MILP',
            'help': 'Advanced MILP with routing',
            'enabled': False
        },
        'genetic': {
            'name': '🧬 Genetic Algorithm',
            'help': 'Evolutionary algorithm',
            'enabled': True
        }
    }
    
    def __init__(self):
        """Initialize the Streamlit application"""
        self.initialize_session_state()
        self.data_handler = DataHandler()
        self.optimization_runner = OptimizationRunner()
        self.visualization_manager = VisualizationManager()
        self.export_manager = ExportManager()
        self.ui_components = UIComponents()
        self.documentation_renderer = DocumentationRenderer()
    
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
        self.documentation_renderer.render_introduction()
        
        if st.session_state.data_loaded:
            self.render_data_exploration()
            
            if st.session_state.optimization_complete:
                self.render_solution_analysis()
                self.render_visualization()
                # Documentation section - only available after optimization
                self.render_documentation_section()
            elif st.session_state.optimization_log:
                # Show optimization logs if optimization failed
                self.render_optimization_logs()
    
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
            
            # Depot Configuration (always visible)
            st.sidebar.markdown("**🏭 Depot Configuration:**")
            
            # Get unique postal codes for depot selection
            if 'orders_df' in st.session_state and st.session_state.orders_df is not None:
                postal_codes = sorted(st.session_state.orders_df['postal_code'].unique().tolist())
                # Add default depot if not in list
                if '08020' not in postal_codes:
                    postal_codes.append('08020')
                    postal_codes.sort()
                default_depot = '08020'
            else:
                postal_codes = ['08020']
                default_depot = '08020'
            
            depot_location = st.sidebar.selectbox(
                "Depot Location",
                options=postal_codes,
                index=postal_codes.index(default_depot) if default_depot in postal_codes else 0,
                help="Location where trucks start and end their routes"
            )
            
            # Real Distances Configuration
            use_real_distances = st.sidebar.checkbox(
                "🌍 Use Real-World Distances",
                value=getattr(st.session_state, 'use_real_distances', False),
                help="Use OpenStreetMap geocoding for accurate geographic distances (takes longer but more realistic)"
            )
            
            # Check if distance method changed and reload data if needed
            previous_setting = getattr(st.session_state, 'use_real_distances', False)
            if use_real_distances != previous_setting:
                st.session_state.use_real_distances = use_real_distances
                
                # Force reload of distance matrix if data is already loaded
                if st.session_state.data_loaded:
                    # Clear optimization state since distance matrix is changing
                    st.session_state.optimization_complete = False
                    st.session_state.optimization_log = []
                    
                    try:
                        # Ensure data handler has the session state data
                        if (hasattr(st.session_state, 'orders_df') and 
                            st.session_state.orders_df is not None):
                            
                            # Copy session state data to data handler if needed
                            if (self.data_handler.orders_df is None or 
                                self.data_handler.orders_df.empty):
                                self.data_handler.orders_df = st.session_state.orders_df
                                self.data_handler.trucks_df = st.session_state.trucks_df
                            
                            with st.sidebar.status("🔄 Updating distance matrix...", expanded=False):
                                if self.data_handler.reload_distance_matrix():
                                    # Update session state with new distance matrix
                                    st.session_state.distance_matrix = self.data_handler.distance_matrix
                                    st.rerun()  # Refresh the app to show updated data
                                else:
                                    st.sidebar.error("❌ Failed to update distance matrix")
                        else:
                            st.sidebar.warning("⚠️ No valid order data found. Please load data first.")
                            # Reload the data with new distance setting
                            st.sidebar.info("🔄 Reloading data with new distance setting...")
                            if self.data_handler.load_example_data():
                                st.session_state.orders_df = self.data_handler.orders_df
                                st.session_state.trucks_df = self.data_handler.trucks_df
                                st.session_state.distance_matrix = self.data_handler.distance_matrix
                                st.sidebar.success("✅ Data reloaded with new distance setting!")
                                st.rerun()
                            else:
                                st.sidebar.error("❌ Failed to reload data")
                                
                    except Exception as e:
                        st.sidebar.error(f"❌ Error updating distance matrix: {str(e)}")
                        import traceback
                        st.sidebar.error(f"Details: {traceback.format_exc()}")
            else:
                # Store in session state for data loading
                st.session_state.use_real_distances = use_real_distances
            
            if use_real_distances:
                st.sidebar.info("📍 Real distances use OpenStreetMap geocoding + Haversine calculation")
            
            
            # Model selection with buttons (based on configuration)
            st.sidebar.markdown("**Model Selection:**")
            
            # Initialize session state for model selection
            if 'optimization_method' not in st.session_state:
                # Set default to first enabled model
                enabled_models = [k for k, v in self.AVAILABLE_MODELS.items() if v['enabled']]
                st.session_state.optimization_method = enabled_models[0] if enabled_models else 'standard'
            
            # Create buttons only for enabled models
            model_buttons = {}
            for model_key, model_config in self.AVAILABLE_MODELS.items():
                if model_config['enabled']:
                    model_buttons[model_key] = st.sidebar.button(
                        model_config['name'],
                        help=model_config['help']
                    )
            
            # Handle button clicks
            for model_key, button_clicked in model_buttons.items():
                if button_clicked:
                    st.session_state.optimization_method = model_key
                    st.sidebar.success(f"{self.AVAILABLE_MODELS[model_key]['name']} selected")
            
            # Show current model selection
            if hasattr(st.session_state, 'optimization_method'):
                if st.session_state.optimization_method == 'enhanced':
                    st.sidebar.info("🚀 **Enhanced MILP Active**\nAdvanced routing optimization")
                elif st.session_state.optimization_method == 'genetic':
                    st.sidebar.info("🧬 **Genetic Algorithm Active**\nEvolutionary multi-objective")
                else:
                    st.sidebar.info("📊 **Standard MILP Active**\nCost optimization + greedy routes")
            else:
                st.sidebar.info("📊 **Standard MILP Active**\nCost optimization + greedy routes")
            
            # Method-specific parameters (only show for selected method)
            if hasattr(st.session_state, 'optimization_method'):
                if st.session_state.optimization_method == 'enhanced':
                    st.sidebar.markdown("**🚀 Enhanced MILP Parameters:**")
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
                        st.sidebar.info(f"⚖️ **Weights:** Cost={cost_weight:.2f}, Distance={distance_weight:.2f}")
                    
                    # Default GA values
                    population_size = 50
                    max_generations = 100
                    mutation_rate = 0.1
                    
                elif st.session_state.optimization_method == 'genetic':
                    st.sidebar.markdown("**🧬 Genetic Algorithm Parameters:**")
                    
                    # Fixed equal weights for genetic algorithm
                    cost_weight = 0.5
                    distance_weight = 0.5
                    st.sidebar.info("⚖️ **Balanced Optimization:** Cost=50%, Distance=50%")
                    
                    population_size = st.sidebar.slider(
                        "Population Size", 
                        min_value=20, max_value=100, value=50, step=10,
                        help="Number of solutions in population"
                    )
                    max_generations = st.sidebar.slider(
                        "Max Generations", 
                        min_value=50, max_value=300, value=100, step=25,
                        help="Maximum number of generations"
                    )
                    mutation_rate = st.sidebar.slider(
                        "Mutation Rate", 
                        min_value=0.05, max_value=0.3, value=0.1, step=0.05,
                        help="Probability of mutation"
                    )
                    
                else:  # Standard method
                    # Default values for standard model
                    cost_weight = 1.0
                    distance_weight = 0.0
                    population_size = 50
                    max_generations = 100
                    mutation_rate = 0.1
            else:
                # Default values when no method selected
                cost_weight = 1.0
                distance_weight = 0.0
                population_size = 50
                max_generations = 100
                mutation_rate = 0.1
            
            # Advanced Options
            st.sidebar.markdown("**Advanced Options:**")
            
            # Depot return option
            depot_return = st.sidebar.checkbox(
                "Trucks Return to Depot", 
                value=False,  # Default to False for all methods
                help="Whether trucks must return to depot after completing deliveries"
            )
            
            # Greedy route optimization always enabled
            enable_greedy_routes = True
            
            # Solver parameters
            st.sidebar.markdown("**Solver Parameters:**")
            solver_timeout = st.sidebar.slider(
                "Solver Timeout (seconds)", 
                min_value=30, max_value=600, 
                value=60 if (hasattr(st.session_state, 'optimization_method') and st.session_state.optimization_method == 'standard') else 300,
                help="Maximum time allowed for optimization (enhanced model needs more time)"
            )
            
            # Run optimization button
            if st.sidebar.button("🚀 Run Optimization", type="primary"):
                with st.spinner("Running optimization..."):
                    success = self.optimization_runner.run_optimization(
                        st.session_state.orders_df,
                        st.session_state.trucks_df,
                        st.session_state.distance_matrix,
                        solver_timeout=solver_timeout,
                        enable_validation=True,  # Always enable validation
                        optimization_method=st.session_state.optimization_method,
                        cost_weight=cost_weight,
                        distance_weight=distance_weight,
                        depot_location=depot_location,
                        depot_return=depot_return,
                        enable_greedy_routes=enable_greedy_routes,
                        population_size=population_size,
                        max_generations=max_generations,
                        mutation_rate=mutation_rate
                    )
                    
                    if success:
                        st.session_state.solution = self.optimization_runner.solution
                        st.session_state.optimization_log = self.optimization_runner.optimization_log
                        st.session_state.optimization_complete = True
                        st.session_state.optimizer_type = st.session_state.optimization_method
                        st.session_state.depot_location = depot_location
                        st.sidebar.success("✅ Optimization completed successfully!")
                    else:
                        st.sidebar.error("❌ Optimization failed")
                        
                        # Show optimization logs for debugging
                        if self.optimization_runner.optimization_log:
                            st.sidebar.error("**Optimization Logs:**")
                            for log_entry in self.optimization_runner.optimization_log[-10:]:  # Show last 10 entries
                                st.sidebar.text(log_entry)
                        
                        # Store the logs for display in main area
                        st.session_state.optimization_log = self.optimization_runner.optimization_log
        

    

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
        
        # Solution overview metrics - Updated KPIs as requested
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cost", f"€{solution['costs']['total_cost']:.0f}")
        with col2:
            total_trucks = len(st.session_state.trucks_df)
            used_trucks = len(solution['selected_trucks'])
            st.metric("Trucks Used", f"{used_trucks}/{total_trucks}")
        with col3:
            total_orders = len(st.session_state.orders_df)
            assigned_orders = len(solution['assignments_df'])
            st.metric("Orders Assigned", f"{assigned_orders}/{total_orders}")
        with col4:
            # Calculate total distance from routes_df
            if 'routes_df' in solution and not solution['routes_df'].empty:
                total_distance = solution['routes_df']['route_distance'].sum()
            elif 'total_distance' in solution.get('costs', {}):
                total_distance = solution['costs']['total_distance']
            else:
                total_distance = 0
            st.metric("Total KM", f"{total_distance:.1f}")
        
        # Solution summary
        st.markdown("### 📋 Solution Summary")
        
        summary_text = f"""
        **Selected Trucks:** {solution['selected_trucks']}
        """
        
        # Calculate total distance for summary
        if 'routes_df' in solution and not solution['routes_df'].empty:
            total_summary_distance = solution['routes_df']['route_distance'].sum()
        elif 'total_distance' in solution.get('costs', {}):
            total_summary_distance = solution['costs']['total_distance']
        else:
            total_summary_distance = 0
        
        for truck_id in solution['selected_trucks']:
            assigned_orders = [a['order_id'] for a in solution['assignments_df'].to_dict('records') if a['truck_id'] == truck_id]
            summary_text += f"\n- **Truck {truck_id}** → Orders {assigned_orders}"
            
            # Add route details if available
            if 'routes_df' in solution and not solution['routes_df'].empty:
                truck_route = solution['routes_df'][solution['routes_df']['truck_id'] == truck_id]
                if not truck_route.empty:
                    route_info = truck_route.iloc[0]
                    route_sequence = route_info.get('route_sequence', [])
                    route_distance = route_info.get('route_distance', 0)
                    
                    if route_sequence and len(route_sequence) > 1:
                        route_text = " → ".join(map(str, route_sequence))
                        summary_text += f"\n  📍 **Route:** {route_text}"
                        summary_text += f"\n  📏 **Distance:** {route_distance:.1f} km"
        
        summary_text += f"\n\n**Total Cost:** €{solution['costs']['total_cost']:.0f}"
        summary_text += f"\n**Total Distance:** {total_summary_distance:.1f} km"
        
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
    
    def render_optimization_logs(self):
        """Render the optimization logs section when optimization fails"""
        st.markdown('<h2 class="section-header">🔍 Optimization Logs</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Optimization Failed</h4>
            <p>The optimization process encountered an error. Please review the logs below for details.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.optimization_log:
            st.markdown("### 📋 Detailed Logs")
            
            # Display logs in a code block for better readability
            log_text = "\n".join(st.session_state.optimization_log)
            st.code(log_text, language="text")
            
            # Add a button to clear logs
            if st.button("🗑️ Clear Logs"):
                st.session_state.optimization_log = []
                st.rerun()
        else:
            st.info("No optimization logs available.")
    
    def render_documentation_section(self):
        """Render the documentation section with same style as other main sections"""
        st.markdown('<h2 class="section-header">📖 Documentation</h2>', unsafe_allow_html=True)
        
        # Determine documentation method based on current selection
        if hasattr(st.session_state, 'optimization_method'):
            if st.session_state.optimization_method == 'enhanced':
                doc_method = "enhanced"
            elif st.session_state.optimization_method == 'genetic':
                doc_method = "genetic"
            else:
                doc_method = "hybrid"  # Standard + Greedy (default)
        else:
            doc_method = "hybrid"  # Default to hybrid if not set
        
        # Render documentation for the selected method
        self.documentation_renderer.render_methodology(doc_method)
    

def main():
    """Main entry point for the Streamlit application"""
    app = VehicleRouterApp()
    app.run()


if __name__ == "__main__":
    main()