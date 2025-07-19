#!/usr/bin/env python3
"""
Vehicle Router - Main Application

This is the main entry point for the Vehicle Routing Problem optimization application.
It orchestrates the complete workflow including data generation, optimization, 
validation, and visualization with comprehensive logging and error handling.

The application workflow:
1. Generate or load input data (orders, trucks, distance matrix)
2. Build and solve the MILP optimization model
3. Validate the solution for correctness
4. Generate visualizations and reports
5. Display results in the specified console format

Usage:
    python src/main.py

The application uses the example data by default but can be configured for
different datasets and parameters.
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import vehicle_router modules
try:
    from vehicle_router.data_generator import DataGenerator
    from vehicle_router.optimizer import VrpOptimizer
    from vehicle_router.validation import SolutionValidator
    from vehicle_router.plotting import plot_routes, plot_costs, plot_utilization
except ImportError as e:
    print(f"Error importing vehicle_router modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure comprehensive logging
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


class VehicleRouterApp:
    """
    Main Vehicle Router Application Class
    
    This class orchestrates the complete vehicle routing optimization workflow
    including data generation, optimization, validation, and visualization.
    It provides comprehensive error handling, logging, and result reporting.
    
    Attributes:
        config (Dict[str, Any]): Application configuration parameters
        logger (logging.Logger): Application logger
        data_generator (DataGenerator): Data generation component
        optimizer (VrpOptimizer): Optimization engine
        validator (SolutionValidator): Solution validator
        solution (Dict[str, Any]): Optimization solution results
        
    Example:
        >>> app = VehicleRouterApp()
        >>> success = app.run()
        >>> if success:
        ...     print("Optimization completed successfully!")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Vehicle Router Application
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        # Set up logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration with defaults
        self.config = {
            'use_example_data': True,
            'random_seed': 42,
            'save_plots': True,
            'plot_directory': 'output',  # Save plots to output directory
            'show_plots': False,  # Set to True to display plots interactively
            'validation_enabled': True,
            'verbose_output': True
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize components
        self.data_generator = None
        self.optimizer = None
        self.validator = None
        self.solution = None
        
        # Initialize data containers
        self.orders_df = None
        self.trucks_df = None
        self.distance_matrix = None
        
        self.logger.info("Vehicle Router Application initialized")
        self.logger.info(f"Configuration: {self.config}")
    
    def run(self) -> bool:
        """
        Execute the complete vehicle routing optimization workflow
        
        Runs the full workflow including data generation, optimization,
        validation, and visualization with comprehensive error handling.
        
        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING VEHICLE ROUTER OPTIMIZATION WORKFLOW")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate input data
            if not self._generate_data():
                return False
            
            # Step 2: Build and solve optimization model
            if not self._optimize_routes():
                return False
            
            # Step 3: Validate solution
            if self.config['validation_enabled']:
                if not self._validate_solution():
                    return False
            
            # Step 4: Generate visualizations
            if self.config['save_plots']:
                self._create_visualizations()
            
            # Step 5: Save results to CSV
            self._save_results_to_csv()
            
            # Step 6: Display results
            self._display_results()
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            self.logger.info("=" * 60)
            self.logger.info(f"WORKFLOW COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in workflow: {str(e)}")
            self.logger.error("Workflow failed - check logs for details")
            return False
    
    def _generate_data(self) -> bool:
        """
        Generate input data for the optimization problem
        
        Returns:
            bool: True if data generation successful, False otherwise
        """
        self.logger.info("Step 1: Generating input data...")
        
        try:
            # Initialize data generator
            self.data_generator = DataGenerator(
                use_example_data=self.config['use_example_data'],
                random_seed=self.config.get('random_seed')
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
            self.distance_matrix = self.data_generator.generate_distance_matrix(postal_codes)
            
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
    
    def _optimize_routes(self) -> bool:
        """
        Build and solve the MILP optimization model
        
        Returns:
            bool: True if optimization successful, False otherwise
        """
        self.logger.info("Step 2: Building and solving optimization model...")
        
        try:
            # Initialize optimizer
            self.optimizer = VrpOptimizer(
                self.orders_df, 
                self.trucks_df, 
                self.distance_matrix
            )
            
            # Build MILP model
            self.logger.info("Building MILP model...")
            self.optimizer.build_model()
            
            # Solve optimization problem
            self.logger.info("Solving optimization problem...")
            success = self.optimizer.solve()
            
            if not success:
                self.logger.error("Optimization failed - no solution found")
                return False
            
            # Extract solution
            self.solution = self.optimizer.get_solution()
            
            if not self.solution:
                self.logger.error("Failed to extract solution from optimizer")
                return False
            
            self.logger.info("Optimization completed successfully:")
            self.logger.info(f"  Selected trucks: {self.solution['selected_trucks']}")
            self.logger.info(f"  Total cost: €{self.solution['costs']['total_cost']:.0f}")
            self.logger.info(f"  Orders delivered: {len(self.solution['assignments_df'])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return False
    
    def _validate_solution(self) -> bool:
        """
        Validate the optimization solution for correctness
        
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("Step 3: Validating optimization solution...")
        
        try:
            # Initialize validator
            self.validator = SolutionValidator(
                self.solution,
                self.orders_df,
                self.trucks_df
            )
            
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
    
    def _create_visualizations(self) -> None:
        """
        Generate and save visualization plots
        """
        self.logger.info("Step 4: Creating visualizations...")
        
        try:
            # Create plots directory
            plots_dir = Path(self.config['plot_directory'])
            plots_dir.mkdir(exist_ok=True)
            
            # Import matplotlib here to avoid issues if not available
            import matplotlib.pyplot as plt
            
            # Generate route visualization
            self.logger.info("Creating route visualization...")
            fig_routes = plot_routes(
                self.solution['routes_df'],
                self.orders_df,
                self.trucks_df,
                self.distance_matrix,
                save_path=plots_dir / "routes.png"
            )
            
            # Generate cost analysis
            self.logger.info("Creating cost analysis...")
            fig_costs = plot_costs(
                self.solution['costs'],
                self.trucks_df,
                save_path=plots_dir / "costs.png"
            )
            
            # Generate utilization analysis
            self.logger.info("Creating utilization analysis...")
            fig_util = plot_utilization(
                self.solution['utilization'],
                self.trucks_df,
                save_path=plots_dir / "utilization.png"
            )
            
            # Show plots if configured
            if self.config['show_plots']:
                plt.show()
            else:
                plt.close('all')  # Close figures to free memory
            
            self.logger.info(f"Visualizations saved to {plots_dir}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping visualizations")
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
    
    def _save_results_to_csv(self) -> None:
        """
        Save optimization results to CSV files in the output directory
        """
        self.logger.info("Step 5: Saving results to CSV files...")
        
        try:
            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save orders data
            orders_file = output_dir / "orders.csv"
            self.orders_df.to_csv(orders_file, index=False)
            self.logger.info(f"Orders data saved to {orders_file}")
            
            # Save trucks data
            trucks_file = output_dir / "trucks.csv"
            self.trucks_df.to_csv(trucks_file, index=False)
            self.logger.info(f"Trucks data saved to {trucks_file}")
            
            # Save distance matrix
            distance_file = output_dir / "distance_matrix.csv"
            self.distance_matrix.to_csv(distance_file)
            self.logger.info(f"Distance matrix saved to {distance_file}")
            
            # Save solution assignments
            assignments_file = output_dir / "solution_assignments.csv"
            self.solution['assignments_df'].to_csv(assignments_file, index=False)
            self.logger.info(f"Solution assignments saved to {assignments_file}")
            
            # Save route information
            routes_file = output_dir / "solution_routes.csv"
            self.solution['routes_df'].to_csv(routes_file, index=False)
            self.logger.info(f"Solution routes saved to {routes_file}")
            
            # Save truck utilization data
            utilization_data = []
            for truck_id, util_info in self.solution['utilization'].items():
                utilization_data.append({
                    'truck_id': truck_id,
                    'capacity': util_info['capacity'],
                    'used_volume': util_info['used_volume'],
                    'utilization_percent': util_info['utilization_percent'],
                    'assigned_orders': ', '.join(util_info['assigned_orders'])
                })
            
            import pandas as pd
            utilization_df = pd.DataFrame(utilization_data)
            utilization_file = output_dir / "truck_utilization.csv"
            utilization_df.to_csv(utilization_file, index=False)
            self.logger.info(f"Truck utilization saved to {utilization_file}")
            
            # Save cost breakdown
            cost_data = []
            for truck_id, cost in self.solution['costs']['truck_costs'].items():
                cost_data.append({
                    'truck_id': truck_id,
                    'cost': cost,
                    'selected': True
                })
            
            # Add unselected trucks with zero cost
            for _, truck in self.trucks_df.iterrows():
                if truck['truck_id'] not in self.solution['costs']['truck_costs']:
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
                'value': self.solution['costs']['total_cost'],
                'unit': 'EUR'
            }, {
                'metric': 'Trucks Used',
                'value': len(self.solution['selected_trucks']),
                'unit': 'count'
            }, {
                'metric': 'Orders Delivered',
                'value': len(self.solution['assignments_df']),
                'unit': 'count'
            }, {
                'metric': 'Total Volume',
                'value': self.orders_df['volume'].sum(),
                'unit': 'm³'
            }, {
                'metric': 'Average Utilization',
                'value': sum(u['utilization_percent'] for u in self.solution['utilization'].values()) / len(self.solution['utilization']),
                'unit': '%'
            }]
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / "solution_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            self.logger.info(f"Solution summary saved to {summary_file}")
            
            self.logger.info(f"All results saved to {output_dir} directory")
            
        except Exception as e:
            self.logger.error(f"Error saving results to CSV: {str(e)}")
    
    def _display_results(self) -> None:
        """
        Display results in the specified console format
        """
        self.logger.info("Step 5: Displaying results...")
        
        try:
            # Generate console output in the required format
            console_output = self._generate_console_output()
            
            # Display the formatted output
            print("\n" + "=" * 50)
            print(console_output)
            print("=" * 50)
            
            # Display additional detailed information if verbose mode enabled
            if self.config['verbose_output']:
                self._display_detailed_results()
                
        except Exception as e:
            self.logger.error(f"Error displaying results: {str(e)}")
    
    def _generate_console_output(self) -> str:
        """
        Generate console output in the specified format
        
        Returns:
            str: Formatted console output string
        """
        if not self.solution:
            return "No solution available"
        
        # Use the optimizer's built-in summary method
        return self.optimizer.get_solution_summary_text()
    
    def _display_detailed_results(self) -> None:
        """
        Display detailed results and analysis
        """
        print("\n" + "-" * 50)
        print("DETAILED ANALYSIS")
        print("-" * 50)
        
        # Display truck utilization details
        print("\nTruck Utilization Details:")
        for truck_id, util_info in self.solution['utilization'].items():
            print(f"  Truck {truck_id}:")
            print(f"    Capacity: {util_info['capacity']:.1f} m³")
            print(f"    Used: {util_info['used_volume']:.1f} m³")
            print(f"    Utilization: {util_info['utilization_percent']:.1f}%")
            print(f"    Orders: {util_info['assigned_orders']}")
        
        # Display cost breakdown
        print(f"\nCost Breakdown:")
        for truck_id, cost in self.solution['costs']['truck_costs'].items():
            print(f"  Truck {truck_id}: €{cost:.0f}")
        print(f"  Total: €{self.solution['costs']['total_cost']:.0f}")
        
        # Display efficiency metrics
        total_volume = self.orders_df['volume'].sum()
        total_capacity_used = sum(u['used_volume'] for u in self.solution['utilization'].values())
        avg_utilization = sum(u['utilization_percent'] for u in self.solution['utilization'].values()) / len(self.solution['utilization'])
        
        print(f"\nEfficiency Metrics:")
        print(f"  Cost per order: €{self.solution['costs']['total_cost'] / len(self.orders_df):.0f}")
        print(f"  Cost per m³: €{self.solution['costs']['total_cost'] / total_volume:.0f}")
        print(f"  Average utilization: {avg_utilization:.1f}%")
        print(f"  Total volume efficiency: {(total_capacity_used / total_volume) * 100:.1f}%")


def main():
    """
    Main entry point for the Vehicle Router application
    
    Handles command line arguments and application configuration.
    """
    # Parse command line arguments (basic implementation)
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem Optimizer')
    parser.add_argument('--random-data', action='store_true', 
                       help='Use random data instead of example data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data generation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip solution validation')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Configure application based on arguments
    config = {
        'use_example_data': not args.random_data,
        'random_seed': args.seed,
        'save_plots': not args.no_plots,
        'show_plots': args.show_plots,
        'validation_enabled': not args.no_validation,
        'verbose_output': not args.quiet,
        'plot_directory': 'output'  # Save plots to output directory
    }
    
    # Set up logging with specified level
    setup_logging(args.log_level)
    
    # Create and run application
    app = VehicleRouterApp(config)
    success = app.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()