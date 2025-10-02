#!/usr/bin/env python3
"""
Vehicle Router - Main Application

This is the main entry point for the Vehicle Routing Problem optimization application.
It orchestrates the complete workflow including data generation, optimization, 
validation, and visualization with comprehensive logging and error handling.

The application supports three optimization models:
- Standard Model: MILP with greedy route optimization  
- Enhanced Model: Advanced MILP with integrated cost-distance optimization
- Genetic Model: Evolutionary algorithm for multi-objective optimization

Usage:
    python src/main.py [--optimizer {standard,enhanced,genetic}] [--depot POSTAL_CODE] [options]

Examples:
    python src/main.py                                    # Standard model (default)
    python src/main.py --optimizer standard               # Standard MILP + greedy model
    python src/main.py --optimizer enhanced --depot 08030 # Enhanced MILP model
    python src/main.py --optimizer genetic                # Genetic algorithm model
"""

import sys
import time
import argparse
import logging
import glob
import os
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
        CLIVisualizationManager, ResultsManager
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
        # Set up enhanced logging
        self.logger = setup_logging(
            log_level=config.get('log_level', 'INFO')
        )
        
        # Validate and store configuration
        self.config = validate_configuration(config)
        
        # Log optimization configuration
        from vehicle_router.logger_config import log_optimization_start
        log_optimization_start(self.logger, self.config)
        
        # Initialize utility managers
        self.data_manager = DataManager(self.logger)
        self.optimization_manager = OptimizationManager(self.logger)
        self.validation_manager = ValidationManager(self.logger)
        self.visualization_manager = CLIVisualizationManager(self.logger)
        self.results_manager = ResultsManager(self.logger)
        
        self.logger.info("🚛 Vehicle Router Application initialized successfully")
        self.logger.debug(f"📋 Full configuration: {self.config}")
    
    def run(self) -> bool:
        """Execute the complete vehicle routing optimization workflow"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING VEHICLE ROUTER OPTIMIZATION WORKFLOW")
        self.logger.info("=" * 80)
        
        # Start overall timing
        self.logger.perf.start_timer("Complete Workflow")
        
        try:
            # Step 1: Generate input data
            self.logger.perf.start_timer("Data Generation")
            if not self.data_manager.generate_data(
                use_example_data=self.config['use_example_data'],
                random_seed=self.config.get('random_seed'),
                depot_location=self.config['depot_location'],
                use_real_distances=self.config.get('use_real_distances', False)
            ):
                self.logger.error("❌ Data generation failed")
                return False
            self.logger.perf.end_timer("Data Generation")
            
            # Log memory usage after data generation
            self.logger.perf.log_memory_usage("After data generation")
            
            # Step 2: Run optimization
            self.logger.perf.start_timer("Optimization")
            if not self.optimization_manager.run_optimization(
                self.data_manager.orders_df,
                self.data_manager.trucks_df,
                self.data_manager.distance_matrix,
                self.config
            ):
                self.logger.error("❌ Optimization failed")
                return False
            self.logger.perf.end_timer("Optimization")
            
            # Log memory usage after optimization
            self.logger.perf.log_memory_usage("After optimization")
            
            # Step 3: Validate solution
            if self.config['validation_enabled']:
                self.logger.perf.start_timer("Solution Validation")
                if not self.validation_manager.validate_solution(
                    self.optimization_manager.solution,
                    self.data_manager.orders_df,
                    self.data_manager.trucks_df
                ):
                    self.logger.error("❌ Solution validation failed")
                    return False
                self.logger.perf.end_timer("Solution Validation")
            
            # Step 4: Generate visualizations
            if self.config['save_plots']:
                self.logger.perf.start_timer("Visualization Generation")
                self.visualization_manager.create_visualizations(
                    self.optimization_manager.solution,
                    self.data_manager.orders_df,
                    self.data_manager.trucks_df,
                    self.data_manager.distance_matrix,
                    self.config
                )
                self.logger.perf.end_timer("Visualization Generation")
            
            # Step 5: Save results to Excel
            self.logger.perf.start_timer("Results Export")
            self.results_manager.save_results_to_excel(
                self.optimization_manager.solution,
                self.data_manager.orders_df,
                self.data_manager.trucks_df
            )
            self.logger.perf.end_timer("Results Export")
            
            # Step 6: Display results
            self._display_results()
            
            # End overall timing and log completion
            total_time = self.logger.perf.end_timer("Complete Workflow")
            
            self.logger.info("=" * 80)
            self.logger.info(f"✅ WORKFLOW COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
            self.logger.info("=" * 80)
            
            # Log final memory usage
            self.logger.perf.log_memory_usage("Workflow Complete")
            
            print(f"📄 Detailed logs saved to: logs/main/latest.log")
            
            return True
            
        except Exception as e:
            self.logger.perf.end_timer("Complete Workflow")
            self.logger.error("💥 CRITICAL ERROR in workflow:")
            self.logger.error(f"   Error: {str(e)}")
            self.logger.exception("   Full traceback:")
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


def clear_previous_logs():
    """Clear all previous log files from logs/main directory"""
    try:
        log_dir = Path("logs/main")
        if log_dir.exists():
            # Get all log files in the directory
            log_files = list(log_dir.glob("*.log"))
            
            if log_files:
                print(f"🧹 Clearing {len(log_files)} previous log files from logs/main/...")
                for log_file in log_files:
                    try:
                        log_file.unlink()
                        print(f"   ✅ Deleted: {log_file.name}")
                    except Exception as e:
                        print(f"   ⚠️  Could not delete {log_file.name}: {e}")
                print("   🎯 Log directory cleared!")
            else:
                print("📝 No previous log files found in logs/main/")
        else:
            print("📁 Creating logs/main directory...")
            log_dir.mkdir(parents=True, exist_ok=True)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not clear previous logs: {e}")


def main():
    """
    Main entry point for the Vehicle Router application
    """
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem Optimizer')
    
    # Simple options
    parser.add_argument('--optimizer', choices=['standard', 'enhanced', 'genetic'], default='standard',
                        help='Optimizer type (default: standard)')
    parser.add_argument('--depot', type=str, default='08020',
                        help='Depot postal code (default: 08020)')
    parser.add_argument('--max-orders-per-truck', type=int, default=3,
                        help='Maximum number of orders per truck (default: 3)')
    parser.add_argument('--real-distances', action='store_true',
                        help='Use real-world distances via OpenStreetMap (default: simulated)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Configuration based on optimizer type
    config = {
        'optimizer_type': args.optimizer,
        'depot_location': args.depot,
        'depot_return': False,
        'enable_greedy_routes': True,
        'max_orders_per_truck': args.max_orders_per_truck,
        'cost_weight': 0.6,
        'distance_weight': 0.4,
        'solver_timeout': 120,
        'use_example_data': True,
        'save_plots': True,
        'show_plots': False,
        'validation_enabled': True,
        'verbose_output': not args.quiet,
        'plot_directory': 'output',
        # Logging configuration
        'log_level': 'INFO' if not args.quiet else 'WARNING',
        # Genetic algorithm parameters
        'ga_population': 50,
        'ga_generations': 100,
        'ga_mutation': 0.1,
        # Distance calculation method
        'use_real_distances': args.real_distances
    }
    
    # Clear previous logs before starting the application
    clear_previous_logs()
    
    # Create and run application
    app = VehicleRouterApp(config)
    success = app.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()