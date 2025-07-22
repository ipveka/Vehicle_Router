"""
Main Application Utilities

This module contains utility functions and classes for the main Vehicle Router application,
including configuration management, data processing, and result formatting.
"""

import logging
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer
from vehicle_router.genetic_optimizer import GeneticVrpOptimizer
from vehicle_router.validation import SolutionValidator
from vehicle_router.plotting import plot_routes, plot_costs, plot_utilization


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (Optional[str]): Optional log file path
    """
    # Create logs directory if it doesn't exist
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
            logging.FileHandler(log_file or logs_dir / "vehicle_router.log")  # File output
        ]
    )
    
    # Set specific logger levels to reduce noise
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


class DataManager:
    """
    Data Management Utility Class
    
    Handles data generation, loading, and validation for the Vehicle Router application.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the DataManager"""
        self.logger = logger
        self.data_generator = None
        self.orders_df = None
        self.trucks_df = None
        self.distance_matrix = None
    
    def generate_data(self, use_example_data: bool = True, random_seed: Optional[int] = None, 
                     depot_location: str = '08020', use_real_distances: bool = False) -> bool:
        """
        Generate input data for the optimization problem
        
        Args:
            use_example_data (bool): Whether to use example data
            random_seed (Optional[int]): Random seed for data generation
            depot_location (str): Depot location postal code
            use_real_distances (bool): Whether to use real-world distances
            
        Returns:
            bool: True if data generation successful, False otherwise
        """
        self.logger.info("Step 1: Generating input data...")
        
        try:
            # Initialize data generator
            self.data_generator = DataGenerator(
                use_example_data=use_example_data,
                random_seed=random_seed
            )
            
            # Generate orders data
            self.logger.info("Generating orders data...")
            self.orders_df = self.data_generator.generate_orders()
            
            if self.orders_df.empty:
                self.logger.error("Failed to generate orders data")
                return False
            
            # Generate trucks data
            self.logger.info("Generating trucks data...")
            self.trucks_df = self.data_generator.generate_trucks()
            
            if self.trucks_df.empty:
                self.logger.error("Failed to generate trucks data")
                return False
            
            # Generate distance matrix
            self.logger.info("Generating distance matrix...")
            postal_codes = self.orders_df['postal_code'].tolist()
            
            # Ensure depot is included in postal codes
            if depot_location not in postal_codes:
                postal_codes.append(depot_location)
                self.logger.info(f"Added depot location {depot_location} to postal codes")
                
            self.distance_matrix = self.data_generator.generate_distance_matrix(
                postal_codes, use_real_distances=use_real_distances)
            
            if self.distance_matrix.empty:
                self.logger.error("Failed to generate distance matrix")
                return False
            
            # Generate and log data summary
            data_summary = self.data_generator.get_data_summary(
                self.orders_df, self.trucks_df, self.distance_matrix
            )
            
            self.logger.info("Data generation completed successfully:")
            self.logger.info(f"  Orders: {data_summary['orders']['count']} (Total volume: {data_summary['orders']['total_volume']:.1f} m³)")
            self.logger.info(f"  Trucks: {data_summary['trucks']['count']} (Total capacity: {data_summary['trucks']['total_capacity']:.1f} m³)")
            self.logger.info(f"  Feasibility: {'Yes' if data_summary['feasibility']['capacity_sufficient'] else 'No'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            return False


class OptimizationManager:
    """
    Optimization Management Utility Class
    
    Handles the optimization process including model building, solving, and result extraction.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the OptimizationManager"""
        self.logger = logger
        self.optimizer = None
        self.solution = None
    
    def run_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                        distance_matrix: pd.DataFrame, config: Dict[str, Any]) -> bool:
        """
        Build and solve the MILP optimization model (standard or enhanced)
        
        Args:
            orders_df: Orders DataFrame
            trucks_df: Trucks DataFrame
            distance_matrix: Distance matrix DataFrame
            config: Configuration dictionary with optimization parameters
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        optimizer_type = config.get('optimizer_type', 'enhanced')
        depot_location = config.get('depot_location', '08020')
        
        self.logger.info(f"Step 2: Building and solving {optimizer_type} optimization model...")
        self.logger.info(f"Depot location: {depot_location}")
        
        try:
            # Ensure depot is in distance matrix
            if depot_location not in distance_matrix.index:
                self.logger.warning(f"Depot location {depot_location} not in distance matrix, adding it...")
                # Add depot to distance matrix with calculated distances
                from vehicle_router.data_generator import DataGenerator
                data_gen = DataGenerator()
                postal_codes = list(distance_matrix.index) + [depot_location]
                distance_matrix = data_gen.generate_distance_matrix(postal_codes)
            
            if optimizer_type == 'standard':
                # Initialize standard optimizer with new parameters
                depot_return = config.get('depot_return', False)
                enable_greedy_routes = config.get('enable_greedy_routes', True)
                
                self.logger.info("Using standard optimizer (cost minimization only)")
                self.logger.info(f"Depot return: {depot_return}")
                self.logger.info(f"Greedy route optimization: {enable_greedy_routes}")
                
                self.optimizer = VrpOptimizer(
                    orders_df, trucks_df, distance_matrix,
                    depot_location=depot_location,
                    depot_return=depot_return,
                    enable_greedy_routes=enable_greedy_routes
                )
                
                # Build MILP model
                self.logger.info("Building standard MILP model...")
                self.optimizer.build_model()
                
                # Solve optimization problem
                self.logger.info("Solving optimization problem...")
                success = self.optimizer.solve()
                
            elif optimizer_type == 'enhanced':
                # Initialize enhanced optimizer with new parameters
                depot_return = config.get('depot_return', True)  # Default True for enhanced
                
                self.logger.info("Using enhanced optimizer (cost + distance minimization)")
                self.logger.info(f"Depot return: {depot_return}")
                
                self.optimizer = EnhancedVrpOptimizer(
                    orders_df, trucks_df, distance_matrix, 
                    depot_location=depot_location,
                    depot_return=depot_return
                )
                
                # Set objective weights
                cost_weight = config.get('cost_weight', 0.6)
                distance_weight = config.get('distance_weight', 0.4)
                self.optimizer.set_objective_weights(cost_weight, distance_weight)
                self.logger.info(f"Objective weights: cost={cost_weight:.2f}, distance={distance_weight:.2f}")
                
                # Build MILP model
                self.logger.info("Building enhanced MILP model...")
                self.optimizer.build_model()
                
                # Solve optimization problem with timeout
                timeout = config.get('solver_timeout', 120)
                self.logger.info(f"Solving optimization problem (timeout: {timeout}s)...")
                success = self.optimizer.solve(timeout=timeout)
                
            elif optimizer_type == 'genetic':
                # Initialize genetic algorithm optimizer
                depot_return = config.get('depot_return', False)  # Default False for genetic
                
                self.logger.info("Using genetic algorithm optimizer (evolutionary approach)")
                self.logger.info(f"Depot return: {depot_return}")
                
                self.optimizer = GeneticVrpOptimizer(
                    orders_df, trucks_df, distance_matrix,
                    depot_location=depot_location,
                    depot_return=depot_return
                )
                
                # Set genetic algorithm parameters
                population_size = config.get('ga_population', 50)
                max_generations = config.get('ga_generations', 100)
                mutation_rate = config.get('ga_mutation', 0.1)
                
                self.optimizer.set_parameters(
                    population_size=population_size,
                    max_generations=max_generations,
                    mutation_rate=mutation_rate
                )
                
                # Set fixed objective weights (0.5/0.5 for genetic algorithm)
                self.optimizer.set_objective_weights(cost_weight=0.5, distance_weight=0.5)
                self.logger.info("Objective weights: cost=0.5, distance=0.5 (fixed for GA)")
                self.logger.info(f"GA parameters: population={population_size}, generations={max_generations}, mutation={mutation_rate}")
                
                # Solve optimization problem with timeout
                timeout = config.get('solver_timeout', 120)
                self.logger.info(f"Solving optimization problem (timeout: {timeout}s)...")
                success = self.optimizer.solve(timeout=timeout)
                
            else:
                self.logger.error(f"Unknown optimizer type: {optimizer_type}")
                return False
            
            if not success:
                self.logger.error("Optimization failed - no solution found")
                return False
            
            # Extract solution
            self.solution = self.optimizer.get_solution()
            
            if not self.solution:
                self.logger.error("Failed to extract solution from optimizer")
                return False
            
            self.logger.info("Optimization completed successfully:")
            self.logger.info(f"  Optimizer type: {optimizer_type}")
            self.logger.info(f"  Depot location: {depot_location}")
            self.logger.info(f"  Selected trucks: {self.solution['selected_trucks']}")
            self.logger.info(f"  Total cost: €{self.solution['costs']['total_cost']:.0f}")
            
            # Log distance information for enhanced model
            if optimizer_type == 'enhanced' and 'total_distance' in self.solution['costs']:
                total_distance = self.solution['costs']['total_distance']
                self.logger.info(f"  Total distance: {total_distance:.1f} km")
            
            self.logger.info(f"  Orders delivered: {len(self.solution['assignments_df'])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return False


class ValidationManager:
    """
    Solution Validation Utility Class
    
    Handles solution validation and constraint checking.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the ValidationManager"""
        self.logger = logger
        self.validator = None
    
    def validate_solution(self, solution: Dict[str, Any], orders_df: pd.DataFrame, 
                         trucks_df: pd.DataFrame) -> bool:
        """
        Validate the optimization solution for correctness
        
        Args:
            solution: Optimization solution dictionary
            orders_df: Orders DataFrame
            trucks_df: Trucks DataFrame
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("Step 3: Validating optimization solution...")
        
        try:
            # Initialize validator
            self.validator = SolutionValidator(solution, orders_df, trucks_df)
            
            # Run comprehensive validation
            validation_report = self.validator.validate_solution()
            
            if validation_report['is_valid']:
                self.logger.info("Solution validation PASSED - all constraints satisfied")
                
                # Log any warnings
                if validation_report['warnings']:
                    self.logger.warning(f"Validation warnings ({len(validation_report['warnings'])}):")
                    for warning in validation_report['warnings']:
                        self.logger.warning(f"  - {warning}")
                
                return True
            else:
                self.logger.error("Solution validation FAILED:")
                self.logger.error(f"  {validation_report['summary']}")
                self.logger.error(f"  Total errors: {validation_report['error_count']}")
                
                # Log specific validation failures
                if not validation_report['capacity_check']['is_valid']:
                    for violation in validation_report['capacity_check']['violations']:
                        self.logger.error(f"  Capacity violation: {violation['error_message']}")
                
                if not validation_report['delivery_check']['is_valid']:
                    if validation_report['delivery_check']['missing_orders']:
                        self.logger.error(f"  Missing orders: {validation_report['delivery_check']['missing_orders']}")
                    if validation_report['delivery_check']['duplicate_assignments']:
                        for dup in validation_report['delivery_check']['duplicate_assignments']:
                            self.logger.error(f"  Duplicate assignment: {dup['error_message']}")
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error in solution validation: {str(e)}")
            return False


class VisualizationManager:
    """
    Visualization Management Utility Class
    
    Handles the creation and saving of visualization plots.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the VisualizationManager"""
        self.logger = logger
    
    def create_visualizations(self, solution: Dict[str, Any], orders_df: pd.DataFrame, 
                            trucks_df: pd.DataFrame, distance_matrix: pd.DataFrame, 
                            config: Dict[str, Any]) -> None:
        """
        Generate and save visualization plots
        
        Args:
            solution: Optimization solution dictionary
            orders_df: Orders DataFrame
            trucks_df: Trucks DataFrame
            distance_matrix: Distance matrix DataFrame
            config: Configuration dictionary
        """
        self.logger.info("Step 4: Creating visualizations...")
        
        try:
            # Create plots directory
            plots_dir = Path(config['plot_directory'])
            plots_dir.mkdir(exist_ok=True)
            
            # Import matplotlib here to avoid issues if not available
            import matplotlib.pyplot as plt
            
            # Generate route visualization
            self.logger.info("Creating route visualization...")
            fig_routes = plot_routes(
                solution['routes_df'],
                orders_df,
                trucks_df,
                distance_matrix,
                save_path=plots_dir / "routes.png"
            )
            
            # Generate cost analysis
            self.logger.info("Creating cost analysis...")
            fig_costs = plot_costs(
                solution['costs'],
                trucks_df,
                save_path=plots_dir / "costs.png"
            )
            
            # Generate utilization analysis
            self.logger.info("Creating utilization analysis...")
            fig_util = plot_utilization(
                solution['utilization'],
                trucks_df,
                save_path=plots_dir / "utilization.png"
            )
            
            # Show plots if configured
            if config['show_plots']:
                plt.show()
            else:
                plt.close('all')  # Close figures to free memory
            
            self.logger.info(f"Visualizations saved to {plots_dir}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualizations")
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")


class ResultsManager:
    """
    Results Management Utility Class
    
    Handles saving results to CSV files and formatting output.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the ResultsManager"""
        self.logger = logger
    
    def save_results_to_csv(self, solution: Dict[str, Any], orders_df: pd.DataFrame, 
                           trucks_df: pd.DataFrame) -> None:
        """
        Save optimization results to CSV files in the output directory
        
        Args:
            solution: Optimization solution dictionary
            orders_df: Orders DataFrame
            trucks_df: Trucks DataFrame
        """
        self.logger.info("Step 5: Saving results to CSV files...")
        
        try:
            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save orders data
            orders_file = output_dir / "orders.csv"
            orders_df.to_csv(orders_file, index=False)
            self.logger.info(f"Orders data saved to {orders_file}")
            
            # Save trucks data
            trucks_file = output_dir / "trucks.csv"
            trucks_df.to_csv(trucks_file, index=False)
            self.logger.info(f"Trucks data saved to {trucks_file}")
            
            # Save solution assignments
            assignments_file = output_dir / "solution_assignments.csv"
            solution['assignments_df'].to_csv(assignments_file, index=False)
            self.logger.info(f"Solution assignments saved to {assignments_file}")
            
            # Save route information
            routes_file = output_dir / "solution_routes.csv"
            solution['routes_df'].to_csv(routes_file, index=False)
            self.logger.info(f"Solution routes saved to {routes_file}")
            
            # Save truck utilization data
            utilization_data = []
            for truck_id, util_info in solution['utilization'].items():
                utilization_data.append({
                    'truck_id': truck_id,
                    'capacity': util_info['capacity'],
                    'used_volume': util_info['used_volume'],
                    'utilization_percent': util_info['utilization_percent'],
                    'assigned_orders': ', '.join(util_info['assigned_orders'])
                })
            
            utilization_df = pd.DataFrame(utilization_data)
            utilization_file = output_dir / "truck_utilization.csv"
            utilization_df.to_csv(utilization_file, index=False)
            self.logger.info(f"Truck utilization saved to {utilization_file}")
            
            # Save cost breakdown
            cost_data = []
            for truck_id, cost in solution['costs']['truck_costs'].items():
                cost_data.append({
                    'truck_id': truck_id,
                    'cost': cost,
                    'selected': True
                })
            
            # Add unselected trucks with zero cost
            for _, truck in trucks_df.iterrows():
                if truck['truck_id'] not in solution['costs']['truck_costs']:
                    cost_data.append({
                        'truck_id': truck['truck_id'],
                        'cost': 0,
                        'selected': False
                    })
            
            cost_df = pd.DataFrame(cost_data)
            cost_file = output_dir / "cost_breakdown.csv"
            cost_df.to_csv(cost_file, index=False)
            self.logger.info(f"Cost breakdown saved to {cost_file}")
            
            # Save summary statistics
            summary_data = [{
                'metric': 'Total Cost',
                'value': solution['costs']['total_cost'],
                'unit': 'EUR'
            }, {
                'metric': 'Trucks Used',
                'value': len(solution['selected_trucks']),
                'unit': 'count'
            }, {
                'metric': 'Orders Delivered',
                'value': len(solution['assignments_df']),
                'unit': 'count'
            }, {
                'metric': 'Total Volume',
                'value': orders_df['volume'].sum(),
                'unit': 'm³'
            }, {
                'metric': 'Average Utilization',
                'value': sum(u['utilization_percent'] for u in solution['utilization'].values()) / len(solution['utilization']),
                'unit': '%'
            }]
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / "solution_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            self.logger.info(f"Solution summary saved to {summary_file}")
            
            self.logger.info(f"All results saved to {output_dir} directory")
            
        except Exception as e:
            self.logger.error(f"Error saving results to CSV: {str(e)}")
    
    def format_solution_output(self, solution: Dict[str, Any], optimizer_type: str, 
                              depot_location: str, optimizer) -> str:
        """
        Format the solution for console output in comparison-style format
        
        Args:
            solution: Optimization solution dictionary
            optimizer_type: Type of optimizer used (standard, enhanced, genetic)
            depot_location: Depot postal code
            optimizer: The optimizer instance
            
        Returns:
            str: Formatted solution text
        """
        if not solution:
            return "No solution available"
        
        # Method name mapping
        method_names = {
            'standard': 'Standard MILP + Greedy',
            'enhanced': 'Enhanced MILP', 
            'genetic': 'Genetic Algorithm'
        }
        
        method_name = method_names.get(optimizer_type, optimizer_type.title())
        
        lines = []
        lines.append("=" * 60)
        lines.append("🏆 VEHICLE ROUTER OPTIMIZATION RESULT")
        lines.append("=" * 60)
        
        # Configuration summary
        lines.append("\n📊 CONFIGURATION:")
        lines.append(f"  Method: {method_name}")
        lines.append(f"  Depot Location: {depot_location}")
        
        # Get execution time if available
        if hasattr(optimizer, 'solve_time'):
            lines.append(f"  Execution Time: {optimizer.solve_time:.2f}s")
        
        # Main results summary
        lines.append("\n🎯 SOLUTION SUMMARY:")
        lines.append("-" * 40)
        
        selected_trucks = solution['selected_trucks']
        total_cost = solution['costs']['total_cost']
        
        # Calculate total distance
        total_distance = 0
        if 'routes_df' in solution and not solution['routes_df'].empty:
            total_distance = solution['routes_df']['route_distance'].sum()
        elif 'total_distance' in solution.get('costs', {}):
            total_distance = solution['costs']['total_distance']
        
        lines.append(f"✅ SUCCESS: {len(selected_trucks)} trucks, €{total_cost:.0f}, {total_distance:.1f} km")
        
        # Key metrics
        lines.append("\n📈 KEY METRICS:")
        lines.append("-" * 30)
        lines.append(f"💰 Total Cost: €{total_cost:.0f}")
        lines.append(f"📏 Total Distance: {total_distance:.1f} km")
        lines.append(f"🚚 Trucks Used: {len(selected_trucks)}")
        
        # Calculate orders delivered
        orders_delivered = len(solution.get('assignments_df', []))
        lines.append(f"📦 Orders Delivered: {orders_delivered}")
        
        # Calculate average utilization
        if 'utilization' in solution:
            avg_utilization = sum(util['utilization_percent'] for util in solution['utilization'].values()) / len(solution['utilization'])
            lines.append(f"📊 Average Utilization: {avg_utilization:.1f}%")
        
        # Method-specific metrics
        if optimizer_type == 'enhanced' and hasattr(optimizer, 'objective_value'):
            lines.append(f"🎯 Multi-Objective Value: {optimizer.objective_value:.6f}")
        elif optimizer_type == 'genetic' and hasattr(optimizer, 'best_solution'):
            if hasattr(optimizer, 'generation'):
                lines.append(f"🧬 GA Generations: {optimizer.generation}")
            if optimizer.best_solution:
                lines.append(f"🏅 GA Final Fitness: {optimizer.best_solution.fitness:.6f}")
        
        # Detailed truck assignments and routes
        lines.append("\n🔬 DETAILED RESULTS:")
        lines.append("-" * 40)
        
        for truck_id in selected_trucks:
            # Get assigned orders
            assigned_orders = []
            if 'assignments_df' in solution:
                for _, row in solution['assignments_df'].iterrows():
                    if row['truck_id'] == truck_id:
                        assigned_orders.append(row['order_id'])
            
            lines.append(f"\n🚛 Truck {truck_id}: {len(assigned_orders)} orders → {assigned_orders}")
            
            # Add route information
            if 'routes_df' in solution and not solution['routes_df'].empty:
                truck_routes = solution['routes_df'][solution['routes_df']['truck_id'] == truck_id]
                if not truck_routes.empty:
                    route_info = truck_routes.iloc[0]
                    route_sequence = route_info.get('route_sequence', [])
                    route_distance = route_info.get('route_distance', 0)
                    
                    if route_sequence:
                        route_str = " → ".join(map(str, route_sequence))
                        lines.append(f"   📍 Route: {route_str}")
                        lines.append(f"   📏 Distance: {route_distance:.1f} km")
            
            # Add utilization information
            if 'utilization' in solution and truck_id in solution['utilization']:
                util = solution['utilization'][truck_id]
                lines.append(f"   📦 Utilization: {util['used_volume']:.1f}/{util['capacity']:.1f} m³ ({util['utilization_percent']:.1f}%)")
        
        return "\n".join(lines)
    
    def display_detailed_results(self, solution: Dict[str, Any], orders_df: pd.DataFrame) -> None:
        """
        Display detailed results and analysis
        
        Args:
            solution: Optimization solution dictionary
            orders_df: Orders DataFrame
        """
        print("\n" + "-" * 50)
        print("DETAILED ANALYSIS")
        print("-" * 50)
        
        # Display truck utilization details
        print("\nTruck Utilization Details:")
        for truck_id, util_info in solution['utilization'].items():
            print(f"  Truck {truck_id}:")
            print(f"    Capacity: {util_info['capacity']:.1f} m³")
            print(f"    Used: {util_info['used_volume']:.1f} m³")
            print(f"    Utilization: {util_info['utilization_percent']:.1f}%")
            print(f"    Orders: {util_info['assigned_orders']}")
        
        # Display cost breakdown
        print(f"\nCost Breakdown:")
        for truck_id, cost in solution['costs']['truck_costs'].items():
            print(f"  Truck {truck_id}: €{cost:.0f}")
        print(f"  Total: €{solution['costs']['total_cost']:.0f}")
        
        # Display efficiency metrics
        total_volume = orders_df['volume'].sum()
        total_capacity_used = sum(u['used_volume'] for u in solution['utilization'].values())
        avg_utilization = sum(u['utilization_percent'] for u in solution['utilization'].values()) / len(solution['utilization'])
        
        print(f"\nEfficiency Metrics:")
        print(f"  Cost per order: €{solution['costs']['total_cost'] / len(orders_df):.0f}")
        print(f"  Cost per m³: €{solution['costs']['total_cost'] / total_volume:.0f}")
        print(f"  Average utilization: {avg_utilization:.1f}%")
        print(f"  Total volume efficiency: {(total_capacity_used / total_volume) * 100:.1f}%")


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Validated and normalized configuration
    """
    # Validate weights
    cost_weight = config.get('cost_weight', 0.6)
    distance_weight = config.get('distance_weight', 0.4)
    
    if cost_weight < 0 or cost_weight > 1:
        raise ValueError("Cost weight must be between 0 and 1")
    if distance_weight < 0 or distance_weight > 1:
        raise ValueError("Distance weight must be between 0 and 1")
    
    # Normalize weights to sum to 1
    total_weight = cost_weight + distance_weight
    if total_weight > 0:
        config['cost_weight'] = cost_weight / total_weight
        config['distance_weight'] = distance_weight / total_weight
    
    return config