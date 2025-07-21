"""
Optimization Runner Module

This module provides the OptimizationRunner class for executing the
vehicle routing optimization in the Streamlit application.
"""

import streamlit as st
import pandas as pd
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer
from vehicle_router.validation import SolutionValidator


class OptimizationRunner:
    """
    Optimization Runner for Streamlit Application
    
    This class manages the execution of vehicle routing optimization
    including solver configuration, progress tracking, and result management.
    """
    
    def __init__(self):
        """Initialize the OptimizationRunner"""
        self.optimizer = None
        self.validator = None
        self.solution = None
        self.optimization_log = []
        self.solve_time = 0.0
    
    def run_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                         distance_matrix: pd.DataFrame, solver_timeout: int = 60,
                         enable_validation: bool = True, use_enhanced_model: bool = True,
                         cost_weight: float = 0.6, distance_weight: float = 0.4,
                         depot_location: str = '08020', depot_return: bool = None,
                         enable_greedy_routes: bool = True) -> bool:
        """
        Run the complete optimization process
        
        Args:
            orders_df: Orders data
            trucks_df: Trucks data
            distance_matrix: Distance matrix
            solver_timeout: Maximum solver time in seconds
            enable_validation: Whether to validate the solution
            use_enhanced_model: Whether to use enhanced model with distance optimization
            cost_weight: Weight for truck costs in objective (0-1)
            distance_weight: Weight for distance costs in objective (0-1)
            depot_location: Depot postal code location
            depot_return: Whether trucks must return to depot (None = use optimizer default)
            enable_greedy_routes: Whether to enable greedy route optimization for standard optimizer
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        self.optimization_log = []
        start_time = time.time()
        
        try:
            # Initialize optimizer
            if use_enhanced_model:
                self._log("Initializing enhanced optimization engine with distance minimization...")
                self._log(f"Depot location: {depot_location}")
                self._log(f"Orders shape: {orders_df.shape}")
                self._log(f"Trucks shape: {trucks_df.shape}")
                self._log(f"Distance matrix shape: {distance_matrix.shape}")
                self._log(f"Distance matrix index: {list(distance_matrix.index)}")
                
                # Check if depot is in distance matrix
                if depot_location not in distance_matrix.index:
                    self._log(f"WARNING: Depot location {depot_location} not in distance matrix")
                    # Add depot to distance matrix
                    from vehicle_router.data_generator import DataGenerator
                    data_gen = DataGenerator()
                    postal_codes = list(distance_matrix.index) + [depot_location]
                    self._log(f"Generating new distance matrix with postal codes: {postal_codes}")
                    distance_matrix = data_gen.generate_distance_matrix(postal_codes)
                    self._log(f"Added depot {depot_location} to distance matrix")
                    self._log(f"New distance matrix shape: {distance_matrix.shape}")
                
                # Set depot_return default for enhanced optimizer
                if depot_return is None:
                    depot_return = True  # Enhanced optimizer default
                
                self._log("Creating EnhancedVrpOptimizer instance...")
                self._log(f"Depot return: {depot_return}")
                self.optimizer = EnhancedVrpOptimizer(
                    orders_df, trucks_df, distance_matrix, 
                    depot_location=depot_location,
                    depot_return=depot_return
                )
                self._log("EnhancedVrpOptimizer created successfully")
                self.optimizer.set_objective_weights(cost_weight, distance_weight)
                self._log(f"Objective weights: cost={cost_weight:.2f}, distance={distance_weight:.2f}")
            else:
                # Set depot_return default for standard optimizer
                if depot_return is None:
                    depot_return = False  # Standard optimizer default
                
                self._log("Initializing standard optimization engine...")
                self._log(f"Depot location: {depot_location}")
                self._log(f"Depot return: {depot_return}")
                self._log(f"Greedy route optimization: {enable_greedy_routes}")
                
                self.optimizer = VrpOptimizer(
                    orders_df, trucks_df, distance_matrix,
                    depot_location=depot_location,
                    depot_return=depot_return,
                    enable_greedy_routes=enable_greedy_routes
                )
            
            # Build model
            self._log("Building MILP optimization model...")
            self.optimizer.build_model()
            self._log("Model built successfully")
            
            # Solve optimization
            self._log("Executing optimization solver...")
            if use_enhanced_model:
                success = self.optimizer.solve(timeout=solver_timeout)
            else:
                success = self.optimizer.solve()
            
            if not success:
                self._log("ERROR: Optimization failed - no solution found")
                return False
            
            # Extract solution
            self._log("Extracting optimization results...")
            self.solution = self.optimizer.get_solution()
            
            self.solve_time = time.time() - start_time
            self._log(f"Optimization completed in {self.solve_time:.2f} seconds")
            
            # Log enhanced solution metrics
            if use_enhanced_model and 'costs' in self.solution:
                if 'total_distance' in self.solution['costs']:
                    total_distance = self.solution['costs']['total_distance']
                    self._log(f"Total distance: {total_distance:.1f} km")
                    self._log(f"Total truck cost: €{self.solution['costs']['total_truck_cost']:.0f}")
            
            # Validate solution if enabled
            if enable_validation:
                self._log("Validating solution...")
                validation_success = self._validate_solution(orders_df, trucks_df)
                if not validation_success:
                    self._log("WARNING: Solution validation failed")
                else:
                    self._log("Solution validation passed")
            
            self._log("Optimization process completed successfully")
            return True
            
        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            return False
    
    def _validate_solution(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame) -> bool:
        """
        Validate the optimization solution
        
        Args:
            orders_df: Orders data
            trucks_df: Trucks data
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        try:
            self.validator = SolutionValidator(self.solution, orders_df, trucks_df)
            validation_report = self.validator.validate_solution()
            
            if validation_report['is_valid']:
                self._log("✅ All constraints satisfied")
                return True
            else:
                self._log(f"❌ Validation failed: {validation_report['summary']}")
                return False
                
        except Exception as e:
            self._log(f"ERROR in validation: {str(e)}")
            return False
    
    def _log(self, message: str):
        """
        Add message to optimization log
        
        Args:
            message: Log message
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.optimization_log.append(log_entry)
    
    def get_optimization_log(self) -> List[str]:
        """
        Get the optimization log messages
        
        Returns:
            List[str]: Log messages
        """
        return self.optimization_log.copy()
    
    def get_solution_summary(self) -> str:
        """
        Get a formatted solution summary
        
        Returns:
            str: Formatted solution summary
        """
        if not self.solution:
            return "No solution available"
        
        summary_lines = [
            "=== OPTIMIZATION RESULTS ===",
            f"Selected Trucks: {self.solution['selected_trucks']}",
            ""
        ]
        
        # Add truck assignments
        for truck_id in self.solution['selected_trucks']:
            assigned_orders = [a['order_id'] for a in self.solution['assignments_df'].to_dict('records') 
                             if a['truck_id'] == truck_id]
            summary_lines.append(f"Truck {truck_id} → Orders {assigned_orders}")
        
        summary_lines.extend([
            "",
            f"Total Cost: €{self.solution['costs']['total_cost']:.0f}",
            f"Optimization Time: {self.solve_time:.2f} seconds",
            f"Trucks Used: {len(self.solution['selected_trucks'])}",
            f"Orders Delivered: {len(self.solution['assignments_df'])}"
        ])
        
        return "\n".join(summary_lines)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the optimization
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if not self.solution:
            return {}
        
        # Calculate utilization metrics
        utilizations = [u['utilization_percent'] for u in self.solution['utilization'].values()]
        
        return {
            'solve_time': self.solve_time,
            'total_cost': self.solution['costs']['total_cost'],
            'trucks_used': len(self.solution['selected_trucks']),
            'orders_delivered': len(self.solution['assignments_df']),
            'avg_utilization': sum(utilizations) / len(utilizations) if utilizations else 0,
            'min_utilization': min(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'optimization_status': 'optimal'
        }