"""
Vehicle Router Streamlit Application

Interactive web interface for Vehicle Routing Problem optimization.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import time
from typing import Dict, Any

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

# Import configuration
from app.config import (
    OPTIMIZATION_METHOD, UI_CONFIG, OPTIMIZATION_DEFAULTS,
    METHOD_PARAMS, validate_config, get_config_summary, get_method_display_name, get_method_params
)

# Configure Streamlit page
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state=UI_CONFIG['initial_sidebar_state']
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
        # Validate configuration on startup
        config_issues = validate_config()
        if config_issues['errors']:
            st.error("❌ Configuration errors found:")
            for error in config_issues['errors']:
                st.error(f"   {error}")
            st.stop()
        
        if config_issues['warnings']:
            st.warning("⚠️ Configuration warnings:")
            for warning in config_issues['warnings']:
                st.warning(f"   {warning}")
        
        # Initialize session state first
        self.initialize_session_state()
        
        # Set up enhanced logging for Streamlit app
        self._setup_logging()
        
        # Initialize app utilities
        self.data_handler = DataHandler()
        self.optimization_runner = OptimizationRunner()
        self.visualization_manager = VisualizationManager()
        self.export_manager = ExportManager()
        self.ui_components = UIComponents()
        self.documentation_renderer = DocumentationRenderer()
        
        self.logger.info("🌐 Streamlit Vehicle Router App initialized successfully")
        self.logger.info(f"📋 Configuration: {get_config_summary()}")
    
    def _validate_max_orders_constraint(self, solution: Dict[str, Any], max_orders_per_truck: int) -> bool:
        """
        Validate that the solution respects the maximum orders per truck constraint
        
        Args:
            solution: Optimization solution dictionary
            max_orders_per_truck: Maximum allowed orders per truck
            
        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        try:
            # Count orders per truck
            truck_order_counts = {}
            for _, assignment in solution['assignments_df'].iterrows():
                truck_id = assignment['truck_id']
                if truck_id not in truck_order_counts:
                    truck_order_counts[truck_id] = 0
                truck_order_counts[truck_id] += 1
            
            # Check if any truck exceeds the limit
            violations = []
            for truck_id, order_count in truck_order_counts.items():
                if order_count > max_orders_per_truck:
                    violations.append(f"Truck {truck_id}: {order_count} orders (max: {max_orders_per_truck})")
            
            if violations:
                self.logger.error(f"❌ Max orders per truck constraint violations:")
                for violation in violations:
                    self.logger.error(f"   {violation}")
                return False
            else:
                self.logger.info(f"✅ Max orders per truck constraint satisfied (limit: {max_orders_per_truck})")
                for truck_id, order_count in truck_order_counts.items():
                    self.logger.info(f"   Truck {truck_id}: {order_count} orders")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error validating max orders constraint: {str(e)}")
            return False
    
    def _setup_logging(self):
        """Set up enhanced logging for Streamlit application"""
        from vehicle_router.logger_config import setup_app_logging, log_system_info
        
        # Generate or retrieve session ID
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())[:8]
        
        # Set up app logging with session ID
        self.logger = setup_app_logging(
            session_id=st.session_state.session_id,
            log_level="INFO",
            log_to_file=True,
            log_to_console=False,  # Streamlit handles console output
            enable_performance_tracking=True
        )
        
        # Log system information on first setup
        if 'logging_initialized' not in st.session_state:
            log_system_info(self.logger)
            st.session_state.logging_initialized = True
    
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
        
        # Set the single optimization method from configuration
        if 'optimization_method' not in st.session_state:
            st.session_state.optimization_method = OPTIMIZATION_METHOD
    
    def run(self):
        """Run the complete Streamlit application"""
        # Main header
        st.markdown('<h1 class="main-header">🚛 Vehicle Router Optimizer</h1>', unsafe_allow_html=True)
        
        # Handle data loading with progress indication
        self._handle_data_loading_progress()
        
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
        
        # Show logs at the end of the page if requested
        if hasattr(st.session_state, 'show_logs') and st.session_state.show_logs:
            self.render_log_viewer()
    
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
                # Set loading state for progress indication
                st.session_state.loading_data = True
                st.rerun()
        
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
                    # Store files in session state and set loading flag
                    st.session_state.orders_file = orders_file
                    st.session_state.trucks_file = trucks_file
                    st.session_state.loading_data = True
                    st.rerun()
        
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
                value=getattr(st.session_state, 'use_real_distances', OPTIMIZATION_DEFAULTS['use_real_distances']),
                help="Use OpenStreetMap geocoding for accurate geographic distances (takes longer but more realistic)"
            )
            
            # Check if distance method changed and reload data if needed
            previous_setting = getattr(st.session_state, 'use_real_distances', False)
            if use_real_distances != previous_setting:
                self.logger.info(f"🌍 Distance calculation method changed: {previous_setting} → {use_real_distances}")
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
                            
                            # Set updating flag to show progress at top of page
                            st.session_state.updating_distances = True
                            st.session_state.update_use_real_distances = use_real_distances
                            st.rerun()
                        else:
                            self.logger.warning("⚠️ No valid order data found. Attempting to reload data...")
                            st.sidebar.warning("⚠️ No valid order data found. Please load data first.")
                            
                            # Set reload flag to show progress at top of page
                            st.session_state.reloading_data = True
                            st.session_state.reload_use_real_distances = use_real_distances
                            st.rerun()
                                
                    except Exception as e:
                        self.logger.error(f"💥 Error updating distance matrix: {str(e)}")
                        self.logger.exception("Full traceback:")
                        # Set error flag to show at top of page
                        st.session_state.distance_error = str(e)
                        st.rerun()
            else:
                # Store in session state for data loading (default to True for real distances)
                st.session_state.use_real_distances = use_real_distances
            

            # Order constraints section (above model parameters)
            st.sidebar.markdown("**🔒 Order Constraints:**")
            max_orders_per_truck = st.sidebar.slider(
                "Max Orders per Truck",
                min_value=1, max_value=10, value=OPTIMIZATION_DEFAULTS['max_orders_per_truck'], step=1,
                help="Maximum number of orders that can be assigned to each truck"
            )
            
            # Depot return option
            depot_return = st.sidebar.checkbox(
                "Trucks Return to Depot", 
                value=OPTIMIZATION_DEFAULTS['depot_return'],
                help="Whether trucks must return to depot after completing deliveries"
            )
            
            # Show the selected optimization method (read-only)
            st.sidebar.markdown("**Optimization Method:**")
            method_display_name = get_method_display_name()
            st.sidebar.info(f"🎯 {method_display_name}")
            st.sidebar.markdown("*Configured in app/config.py*")
            

            
            # Method-specific parameters (only show for selected method)
            if hasattr(st.session_state, 'optimization_method'):
                method = st.session_state.optimization_method
                method_defaults = get_method_params()
                
                if method == 'enhanced':
                    st.sidebar.markdown("**🚀 Enhanced MILP Parameters:**")
                    cost_weight = st.sidebar.slider(
                        "Cost Weight", 
                        min_value=0.0, max_value=1.0, value=method_defaults.get('cost_weight', 0.6), step=0.1,
                        help="Weight for truck costs in objective function"
                    )
                    distance_weight = st.sidebar.slider(
                        "Distance Weight", 
                        min_value=0.0, max_value=1.0, value=method_defaults.get('distance_weight', 0.4), step=0.1,
                        help="Weight for travel distances in objective function"
                    )
                    
                    # Normalize weights to sum to 1
                    total_weight = cost_weight + distance_weight
                    if total_weight > 0:
                        cost_weight = cost_weight / total_weight
                        distance_weight = distance_weight / total_weight
                        st.sidebar.info(f"⚖️ **Weights:** Cost={cost_weight:.2f}, Distance={distance_weight:.2f}")
                    
                    # Default values for other parameters
                    population_size = method_defaults.get('population_size', 50)
                    max_generations = method_defaults.get('max_generations', 100)
                    mutation_rate = method_defaults.get('mutation_rate', 0.1)
                    
                elif method == 'genetic':
                    st.sidebar.markdown("**🧬 Genetic Algorithm Parameters:**")
                    
                    # Fixed equal weights for genetic algorithm
                    cost_weight = method_defaults.get('cost_weight', 0.5)
                    distance_weight = method_defaults.get('distance_weight', 0.5)
                    
                    population_size = st.sidebar.slider(
                        "Population Size", 
                        min_value=20, max_value=100, value=method_defaults.get('population_size', 50), step=10,
                        help="Number of solutions in population"
                    )
                    max_generations = st.sidebar.slider(
                        "Max Generations", 
                        min_value=50, max_value=300, value=method_defaults.get('max_generations', 100), step=25,
                        help="Maximum number of generations"
                    )
                    mutation_rate = st.sidebar.slider(
                        "Mutation Rate", 
                        min_value=0.05, max_value=0.3, value=method_defaults.get('mutation_rate', 0.1), step=0.05,
                        help="Probability of mutation"
                    )
                    
                else:  # Standard method
                    # Use method defaults
                    cost_weight = method_defaults.get('cost_weight', 1.0)
                    distance_weight = method_defaults.get('distance_weight', 0.0)
                    population_size = method_defaults.get('population_size', 50)
                    max_generations = method_defaults.get('max_generations', 100)
                    mutation_rate = method_defaults.get('mutation_rate', 0.1)
            else:
                # Default values when no method selected
                cost_weight = 1.0
                distance_weight = 0.0
                population_size = 50
                max_generations = 100
                mutation_rate = 0.1
            
            # Greedy route optimization always enabled for standard method
            enable_greedy_routes = True
            
            # Set solver timeout based on method configuration
            if hasattr(st.session_state, 'optimization_method'):
                method_defaults = get_method_params()
                solver_timeout = method_defaults.get('solver_timeout', OPTIMIZATION_DEFAULTS['solver_timeout'])
            else:
                solver_timeout = OPTIMIZATION_DEFAULTS['solver_timeout']
            
            # Run optimization button
            if st.sidebar.button("🚀 Run Optimization", type="primary"):
                with st.spinner("Running optimization..."):
                    self.logger.info("🚀 User initiated optimization")
                    self.logger.info(f"   Method: {st.session_state.optimization_method}")
                    self.logger.info(f"   Depot: {depot_location}")
                    self.logger.info(f"   Real distances: {use_real_distances}")
                    self.logger.info(f"   Max orders per truck: {max_orders_per_truck}")
                    self.logger.info(f"   Timeout: {solver_timeout}s (from config)")
                    
                    self.logger.perf.start_timer("Complete Optimization")
                    
                    success = self.optimization_runner.run_optimization(
                        st.session_state.orders_df,
                        st.session_state.trucks_df,
                        st.session_state.distance_matrix,
                        solver_timeout=solver_timeout,
                        optimization_method=st.session_state.optimization_method,
                        cost_weight=cost_weight,
                        distance_weight=distance_weight,
                        depot_location=depot_location,
                        depot_return=depot_return,
                        enable_greedy_routes=enable_greedy_routes,
                        max_orders_per_truck=max_orders_per_truck,
                        population_size=population_size,
                        max_generations=max_generations,
                        mutation_rate=mutation_rate
                    )
                    
                    if success:
                        opt_time = self.logger.perf.end_timer("Complete Optimization")
                        
                        solution = self.optimization_runner.solution
                        
                        # Validate max orders per truck constraint
                        constraint_satisfied = self._validate_max_orders_constraint(solution, max_orders_per_truck)
                        
                        if constraint_satisfied:
                            st.session_state.solution = solution
                            st.session_state.optimization_log = self.optimization_runner.optimization_log
                            st.session_state.optimization_complete = True
                            st.session_state.optimizer_type = st.session_state.optimization_method
                            st.session_state.depot_location = depot_location
                            
                            # Log success details
                            self.logger.info(f"✅ Optimization completed successfully in {opt_time:.2f}s")
                            self.logger.info(f"   Selected trucks: {solution['selected_trucks']}")
                            self.logger.info(f"   Total cost: €{solution['costs']['total_cost']:.0f}")
                            self.logger.info(f"   Orders assigned: {len(solution['assignments_df'])}")
                            
                            # Log memory usage
                            self.logger.perf.log_memory_usage("After optimization")
                            
                            st.sidebar.success("✅ Optimization completed successfully!")
                        else:
                            # Constraint violation detected
                            self.logger.error("❌ Solution violates max orders per truck constraint")
                            st.sidebar.error("❌ Solution violates maximum orders per truck constraint!")
                            st.sidebar.error("Please check the logs for details or try with a higher limit.")
                            
                            # Store logs for display
                            st.session_state.optimization_log = self.optimization_runner.optimization_log
                            st.session_state.optimization_complete = False
                    else:
                        self.logger.perf.end_timer("Complete Optimization")
                        self.logger.error("❌ Optimization failed")
                        
                        # Log failure details
                        if self.optimization_runner.optimization_log:
                            self.logger.error("   Optimization error logs:")
                            for log_entry in self.optimization_runner.optimization_log[-5:]:
                                self.logger.error(f"     {log_entry}")
                        
                        st.sidebar.error("❌ Optimization failed")
                        
                        # Show optimization logs for debugging
                        if self.optimization_runner.optimization_log:
                            st.sidebar.error("**Optimization Logs:**")
                            for log_entry in self.optimization_runner.optimization_log[-10:]:  # Show last 10 entries
                                st.sidebar.text(log_entry)
                        
                        # Store the logs for display in main area
                        st.session_state.optimization_log = self.optimization_runner.optimization_log
        
            # Log Management Section - at the end of sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("**📋 Log Management:**")
            
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("📖 View Logs", key="view_logs_btn"):
                    self._show_logs_modal()
            
            with col2:
                if st.button("🗑️ Clear Logs", key="clear_logs_btn"):
                    self._clear_logs()

    def _show_logs_modal(self):
        """Display logs in an expandable section in the main area"""
        st.session_state.show_logs = True
        st.rerun()
    
    def _clear_logs(self):
        """Clear all log files from logs/app directory"""
        try:
            import glob
            from pathlib import Path
            
            # Clear app logs
            app_log_dir = Path("logs/app")
            log_files_cleared = 0
            
            if app_log_dir.exists():
                log_files = list(app_log_dir.glob("*.log"))
                for log_file in log_files:
                    try:
                        log_file.unlink()
                        log_files_cleared += 1
                    except Exception as e:
                        self.logger.warning(f"Could not delete {log_file.name}: {e}")
            
            # Clear main logs too
            main_log_dir = Path("logs/main")
            if main_log_dir.exists():
                log_files = list(main_log_dir.glob("*.log"))
                for log_file in log_files:
                    try:
                        log_file.unlink()
                        log_files_cleared += 1
                    except Exception as e:
                        self.logger.warning(f"Could not delete {log_file.name}: {e}")
            
            if log_files_cleared > 0:
                st.sidebar.success(f"✅ Cleared {log_files_cleared} log files!")
                self.logger.info(f"🧹 User cleared {log_files_cleared} log files")
            else:
                st.sidebar.info("📝 No log files found to clear")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error clearing logs: {str(e)}")
            self.logger.error(f"Error clearing logs: {str(e)}")

    

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
                width='stretch',
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
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = self.visualization_manager.create_orders_distribution_chart(st.session_state.orders_df)
                st.plotly_chart(fig, width='stretch')
        
        with tab2:
            st.markdown("### Trucks Information")
            st.dataframe(
                st.session_state.trucks_df,
                width='stretch',
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
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = self.visualization_manager.create_trucks_cost_efficiency_chart(st.session_state.trucks_df)
                st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.markdown("### Distance Matrix")
            st.dataframe(
                st.session_state.distance_matrix,
                width='stretch'
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
            st.plotly_chart(fig, width='stretch')
        
        with tab2:
            st.markdown("### Solution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cost Breakdown by Truck")
                fig = self.visualization_manager.create_cost_breakdown_chart(
                    solution['costs'],
                    st.session_state.trucks_df
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("#### Volume Usage by Truck")
                fig = self.visualization_manager.create_volume_usage_chart(
                    solution['utilization'],
                    st.session_state.trucks_df
                )
                st.plotly_chart(fig, width='stretch')
    
    def _handle_data_loading_progress(self):
        """Handle data loading and distance matrix updates with progress bar at the top of the page"""
        # Handle distance matrix update
        if hasattr(st.session_state, 'updating_distances') and st.session_state.updating_distances:
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                use_real_distances = st.session_state.update_use_real_distances
                
                if use_real_distances:
                    status_text.text("🌍 Calculating real-world distances using OpenStreetMap...")
                else:
                    status_text.text("📐 Calculating simulated distances...")
                
                progress_bar.progress(30)
                
                # Copy session state data to data handler if needed
                if (hasattr(st.session_state, 'orders_df') and 
                    st.session_state.orders_df is not None):
                    self.data_handler.orders_df = st.session_state.orders_df
                    self.data_handler.trucks_df = st.session_state.trucks_df
                
                self.logger.perf.start_timer("Distance Matrix Reload")
                
                if self.data_handler.reload_distance_matrix():
                    progress_bar.progress(80)
                    reload_time = self.logger.perf.end_timer("Distance Matrix Reload")
                    
                    # Update session state with new distance matrix
                    st.session_state.distance_matrix = self.data_handler.distance_matrix
                    
                    method_desc = "real-world distances (OpenStreetMap + Haversine)" if use_real_distances else "simulated distances"
                    self.logger.info(f"✅ Distance matrix updated successfully in {reload_time:.2f}s")
                    self.logger.info(f"   Method: {method_desc}")
                    
                    progress_bar.progress(100)
                    status_text.success(f"✅ Updated to {method_desc}!")
                    
                    # Clean up session state
                    del st.session_state.updating_distances
                    del st.session_state.update_use_real_distances
                    
                    # Brief pause then refresh
                    import time
                    time.sleep(1)
                    st.rerun()
                else:
                    self.logger.perf.end_timer("Distance Matrix Reload")
                    self.logger.error("❌ Failed to update distance matrix")
                    progress_bar.progress(100)
                    status_text.error("❌ Failed to update distance matrix")
                    
                    # Clean up session state
                    del st.session_state.updating_distances
                    del st.session_state.update_use_real_distances
            return
        
        # Handle data reloading with new distance setting
        if hasattr(st.session_state, 'reloading_data') and st.session_state.reloading_data:
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                use_real_distances = st.session_state.reload_use_real_distances
                
                if use_real_distances:
                    status_text.text("🔄 Reloading data with real-world distances...")
                else:
                    status_text.text("🔄 Reloading data with simulated distances...")
                
                progress_bar.progress(20)
                
                self.logger.perf.start_timer("Data Reload with New Distance Setting")
                
                if self.data_handler.load_example_data():
                    progress_bar.progress(60)
                    
                    if use_real_distances:
                        status_text.text("🌍 Calculating real-world distances using OpenStreetMap...")
                    else:
                        status_text.text("📐 Calculating simulated distances...")
                    
                    progress_bar.progress(85)
                    reload_time = self.logger.perf.end_timer("Data Reload with New Distance Setting")
                    
                    st.session_state.orders_df = self.data_handler.orders_df
                    st.session_state.trucks_df = self.data_handler.trucks_df
                    st.session_state.distance_matrix = self.data_handler.distance_matrix
                    
                    method_desc = "real-world distances (OpenStreetMap)" if use_real_distances else "simulated distances"
                    self.logger.info(f"✅ Data reloaded with {method_desc} in {reload_time:.2f}s")
                    
                    progress_bar.progress(100)
                    status_text.success(f"✅ Data reloaded with {method_desc}!")
                    
                    # Clean up session state
                    del st.session_state.reloading_data
                    del st.session_state.reload_use_real_distances
                    
                    # Brief pause then refresh
                    import time
                    time.sleep(1)
                    st.rerun()
                else:
                    self.logger.perf.end_timer("Data Reload with New Distance Setting")
                    self.logger.error("❌ Failed to reload data")
                    progress_bar.progress(100)
                    status_text.error("❌ Failed to reload data")
                    
                    # Clean up session state
                    del st.session_state.reloading_data
                    del st.session_state.reload_use_real_distances
            return
        
        # Handle distance calculation errors
        if hasattr(st.session_state, 'distance_error') and st.session_state.distance_error:
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(100)
                status_text = st.empty()
                
                status_text.error(f"❌ Error updating distance matrix: {st.session_state.distance_error}")
                
                # Clean up session state after showing error
                del st.session_state.distance_error
            return
        
        # Handle initial data loading
        if hasattr(st.session_state, 'loading_data') and st.session_state.loading_data:
            # Create progress bar container at the top
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Check if loading example data or custom data
                if hasattr(st.session_state, 'orders_file') and hasattr(st.session_state, 'trucks_file'):
                    # Loading custom data
                    status_text.text("📤 Loading custom data...")
                    progress_bar.progress(10)
                    
                    self.logger.info("📊 User initiated custom data loading")
                    self.logger.perf.start_timer("Custom Data Loading")
                    
                    success = self.data_handler.load_custom_data(
                        st.session_state.orders_file, 
                        st.session_state.trucks_file
                    )
                    progress_bar.progress(50)
                    
                    if success:
                        status_text.text("🌍 Calculating distances...")
                        progress_bar.progress(70)
                        
                        # Check if real distances are enabled
                        use_real_distances = getattr(st.session_state, 'use_real_distances', True)
                        if use_real_distances:
                            status_text.text("🌍 Calculating real-world distances using OpenStreetMap...")
                        else:
                            status_text.text("📐 Calculating simulated distances...")
                        
                        progress_bar.progress(90)
                        
                        st.session_state.orders_df = self.data_handler.orders_df
                        st.session_state.trucks_df = self.data_handler.trucks_df
                        st.session_state.distance_matrix = self.data_handler.distance_matrix
                        st.session_state.data_loaded = True
                        
                        load_time = self.logger.perf.end_timer("Custom Data Loading")
                        self.logger.info(f"✅ Custom data loaded successfully in {load_time:.2f}s")
                        self.logger.info(f"   Orders: {len(st.session_state.orders_df)}, Trucks: {len(st.session_state.trucks_df)}")
                        
                        progress_bar.progress(100)
                        status_text.success("✅ Custom data loaded successfully!")
                        
                        # Clean up session state
                        del st.session_state.orders_file
                        del st.session_state.trucks_file
                        del st.session_state.loading_data
                        
                        # Brief pause to show completion, then rerun
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        self.logger.perf.end_timer("Custom Data Loading")
                        self.logger.error("❌ Failed to load custom data")
                        progress_bar.progress(100)
                        status_text.error("❌ Failed to load custom data")
                        del st.session_state.loading_data
                        if hasattr(st.session_state, 'orders_file'):
                            del st.session_state.orders_file
                        if hasattr(st.session_state, 'trucks_file'):
                            del st.session_state.trucks_file
                else:
                    # Loading example data
                    status_text.text("📊 Loading example data...")
                    progress_bar.progress(20)
                    
                    self.logger.info("📊 User initiated example data loading")
                    self.logger.perf.start_timer("Example Data Loading")
                    
                    success = self.data_handler.load_example_data()
                    progress_bar.progress(60)
                    
                    if success:
                        status_text.text("🌍 Calculating distances...")
                        progress_bar.progress(80)
                        
                        # Check if real distances are enabled
                        use_real_distances = getattr(st.session_state, 'use_real_distances', True)
                        if use_real_distances:
                            status_text.text("🌍 Calculating real-world distances using OpenStreetMap...")
                        else:
                            status_text.text("📐 Calculating simulated distances...")
                        
                        progress_bar.progress(95)
                        
                        st.session_state.orders_df = self.data_handler.orders_df
                        st.session_state.trucks_df = self.data_handler.trucks_df
                        st.session_state.distance_matrix = self.data_handler.distance_matrix
                        st.session_state.data_loaded = True
                        
                        load_time = self.logger.perf.end_timer("Example Data Loading")
                        self.logger.info(f"✅ Example data loaded successfully in {load_time:.2f}s")
                        self.logger.info(f"   Orders: {len(st.session_state.orders_df)}, Trucks: {len(st.session_state.trucks_df)}")
                        
                        progress_bar.progress(100)
                        status_text.success("✅ Example data loaded successfully!")
                        
                        # Clean up session state
                        del st.session_state.loading_data
                        
                        # Brief pause to show completion, then rerun
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        self.logger.perf.end_timer("Example Data Loading")
                        self.logger.error("❌ Failed to load example data")
                        progress_bar.progress(100)
                        status_text.error("❌ Failed to load example data")
                        del st.session_state.loading_data

    def render_log_viewer(self):
        """Render the log viewer section to display all log files"""
        st.markdown('<h2 class="section-header">📋 Log Viewer</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col2:
            if st.button("❌ Close Logs", key="close_logs_btn"):
                st.session_state.show_logs = False
                st.rerun()
        
        try:
            from pathlib import Path
            import os
            
            # Get all log directories
            log_base_dir = Path("logs")
            log_dirs = ["app", "main"]
            
            for log_dir_name in log_dirs:
                log_dir = log_base_dir / log_dir_name
                
                if log_dir.exists():
                    log_files = sorted(list(log_dir.glob("*.log")), key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    if log_files:
                        st.markdown(f"### 📁 {log_dir_name.title()} Logs")
                        
                        # Show log files in tabs
                        if len(log_files) == 1:
                            # Single file - no tabs needed
                            log_file = log_files[0]
                            st.markdown(f"**📄 {log_file.name}**")
                            try:
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if content.strip():
                                        st.code(content, language="text")
                                    else:
                                        st.info("Log file is empty.")
                            except Exception as e:
                                st.error(f"Could not read {log_file.name}: {str(e)}")
                        else:
                            # Multiple files - use tabs
                            tab_names = [f.name for f in log_files[:5]]  # Limit to 5 most recent
                            tabs = st.tabs(tab_names)
                            
                            for i, (tab, log_file) in enumerate(zip(tabs, log_files[:5])):
                                with tab:
                                    try:
                                        with open(log_file, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                            if content.strip():
                                                # Show file info
                                                file_size = log_file.stat().st_size / 1024  # KB
                                                mod_time = log_file.stat().st_mtime
                                                st.caption(f"Size: {file_size:.1f} KB | Modified: {os.path.basename(log_file.name)}")
                                                st.code(content, language="text")
                                            else:
                                                st.info("Log file is empty.")
                                    except Exception as e:
                                        st.error(f"Could not read {log_file.name}: {str(e)}")
                    else:
                        st.info(f"No log files found in {log_dir_name}/ directory.")
                else:
                    st.info(f"Log directory {log_dir_name}/ does not exist.")
                    
        except Exception as e:
            st.error(f"Error loading log files: {str(e)}")
            self.logger.error(f"Error in log viewer: {str(e)}")

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
        
        # Log documentation access
        if hasattr(st.session_state, 'optimization_method'):
            method = st.session_state.optimization_method
            self.logger.info(f"📖 User accessing documentation for {method} method")
        
        # Determine documentation method based on configured method
        if OPTIMIZATION_METHOD == 'enhanced':
            doc_method = "enhanced"
        elif OPTIMIZATION_METHOD == 'genetic':
            doc_method = "genetic"
        else:
            doc_method = "hybrid"  # Standard + Greedy
        
        # Render documentation for the selected method
        self.documentation_renderer.render_methodology(doc_method)
    

def main():
    """Main entry point for the Streamlit application"""
    app = VehicleRouterApp()
    app.run()


if __name__ == "__main__":
    main()