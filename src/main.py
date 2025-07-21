#!/usr/bin/env python3
"""
Vehicle Router - Main Application

This is the main entry point for the Vehicle Routing Problem optimization application.
It orchestrates the complete workflow including data generation, optimization, 
validation, and visualization with comprehensive logging and error handling.

The application supports two optimization models:
- Standard Model: Minimizes truck operational costs only
- Enhanced Model: Minimizes both truck costs and travel distances with depot routing

Usage:
    python src/main.py [--optimizer {standard,enhanced}] [--depot POSTAL_CODE] [options]

Examples:
    python src/main.py                                    # Enhanced model with default depot
    python src/main.py --optimizer standard               # Standard cost-only model
    python src/main.py --optimizer enhanced --depot 08030 # Enhanced model with specific depot
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utility modules
try:
    from src.main_utils import (
        setup_logging, validate_configuration,
        DataManager, OptimizationManager, ValidationManager, 
        VisualizationManager, ResultsManager
    )
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class VehicleRouterApp:
    """
    Main Vehicle Router Application Class
    
    This class orchestrates the complete vehicle routing optimization workflow
    using utility managers for clean separation of concerns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Vehicle Router Application"""
        # Set up logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Validate and store configuration
        self.config = validate_configuration(config)
        
        # Initialize utility managers
        self.data_manager = DataManager(self.logger)
        self.optimization_manager = OptimizationManager(self.logger)
        self.validation_manager = ValidationManager(self.logger)
        self.visualization_manager = VisualizationManager(self.logger)
        self.results_manager = ResultsManager(self.logger)
        
        self.logger.info("Vehicle Router Application initialized")
        self.logger.info(f"Configuration: {self.config}")
    
    def run(self) -> bool:
        """Execute the complete vehicle routing optimization workflow"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING VEHICLE ROUTER OPTIMIZATION WORKFLOW")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate input data
            if not self.data_manager.generate_data(
                use_example_data=self.config['use_example_data'],
                random_seed=self.config.get('random_seed'),
                depot_location=self.config['depot_location']
            ):
                return False
            
            # Step 2: Run optimization
            if not self.optimization_manager.run_optimization(
                self.data_manager.orders_df,
                self.data_manager.trucks_df,
                self.data_manager.distance_matrix,
                self.config
            ):
                return False
            
            # Step 3: Validate solution
            if self.config['validation_enabled']:
                if not self.validation_manager.validate_solution(
                    self.optimization_manager.solution,
                    self.data_manager.orders_df,
                    self.data_manager.trucks_df
                ):
                    return False
            
            # Step 4: Generate visualizations
            if self.config['save_plots']:
                self.visualization_manager.create_visualizations(
                    self.optimization_manager.solution,
                    self.data_manager.orders_df,
                    self.data_manager.trucks_df,
                    self.data_manager.distance_matrix,
                    self.config
                )
            
            # Step 5: Save results to CSV
            self.results_manager.save_results_to_csv(
                self.optimization_manager.solution,
                self.data_manager.orders_df,
                self.data_manager.trucks_df
            )
            
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
    
    def _display_results(self) -> None:
        """Display results in the specified console format"""
        self.logger.info("Step 6: Displaying results...")
        
        try:
            # Generate console output using results manager
            console_output = self.results_manager.format_solution_output(
                self.optimization_manager.solution,
                self.config['optimizer_type'],
                self.config['depot_location'],
                self.optimization_manager.optimizer
            )
            
            # Display the formatted output
            print("\n" + "=" * 50)
            print(console_output)
            print("=" * 50)
            
            # Display additional detailed information if verbose mode enabled
            if self.config['verbose_output']:
                self.results_manager.display_detailed_results(
                    self.optimization_manager.solution,
                    self.data_manager.orders_df
                )
                
        except Exception as e:
            self.logger.error(f"Error displaying results: {str(e)}")


def main():
    """
    Main entry point for the Vehicle Router application
    """
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem Optimizer')
    
    # Simple options
    parser.add_argument('--optimizer', choices=['standard', 'enhanced'], default='standard',
                       help='Optimizer type (default: standard)')
    parser.add_argument('--depot', type=str, default='08020',
                       help='Depot postal code (default: 08020)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Simple configuration
    config = {
        'optimizer_type': args.optimizer,
        'depot_location': args.depot,
        'depot_return': True if args.optimizer == 'enhanced' else False,  # Enhanced default True, Standard default False
        'enable_greedy_routes': True,  # Enable greedy route optimization for standard optimizer
        'cost_weight': 0.6,
        'distance_weight': 0.4,
        'solver_timeout': 120,
        'use_example_data': True,
        'save_plots': True,
        'show_plots': False,
        'validation_enabled': True,
        'verbose_output': not args.quiet,
        'plot_directory': 'output'
    }
    
    # Set up logging
    setup_logging('INFO' if not args.quiet else 'WARNING')
    
    # Create and run application
    app = VehicleRouterApp(config)
    success = app.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()