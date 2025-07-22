"""
Optimization Runner Module

This module provides the OptimizationRunner class that manages the complete
optimization workflow for the Vehicle Router application, including three
optimization approaches:
1. Standard MILP (with greedy route optimization)
2. Enhanced MILP (integrated cost-distance optimization)  
3. Genetic Algorithm (evolutionary multi-objective optimization)
"""

import streamlit as st
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationRunner:
    """
    Optimization Runner for Vehicle Router Application
    
    This class manages the complete optimization workflow including data validation,
    model execution, solution processing, and result formatting. It supports three
    different optimization approaches with comprehensive error handling and logging.
    
    Attributes:
        solution (Dict[str, Any]): Optimization solution data
        optimization_log (List[str]): Log messages from optimization process
        
    Example:
        >>> runner = OptimizationRunner()
        >>> success = runner.run_optimization(
        ...     orders_df, trucks_df, distance_matrix,
        ...     optimization_method='genetic',
        ...     solver_timeout=300
        ... )
        >>> if success:
        ...     solution = runner.solution
    """
    
    def __init__(self):
        """Initialize the OptimizationRunner"""
        self.solution = None
        self.optimization_log = []
    
    def run_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                         distance_matrix: pd.DataFrame, solver_timeout: int = 60,
                         enable_validation: bool = True, optimization_method: str = 'standard',
                         cost_weight: float = 0.6, distance_weight: float = 0.4,
                         depot_location: str = '08020', depot_return: bool = None,
                         enable_greedy_routes: bool = True,
                         # Genetic algorithm parameters
                         population_size: int = 50, max_generations: int = 100,
                         mutation_rate: float = 0.1) -> bool:
        """
        Run the complete optimization process
        
        Args:
            orders_df: Orders data
            trucks_df: Trucks data
            distance_matrix: Distance matrix
            solver_timeout: Maximum solver time in seconds
            enable_validation: Whether to validate the solution
            optimization_method: Optimization method ('standard', 'enhanced', 'genetic')
            cost_weight: Weight for truck costs in objective (0-1)
            distance_weight: Weight for distance costs in objective (0-1)
            depot_location: Depot postal code location
            depot_return: Whether trucks must return to depot (None = use optimizer default)
            enable_greedy_routes: Whether to enable greedy route optimization for standard optimizer
            population_size: Population size for genetic algorithm
            max_generations: Maximum generations for genetic algorithm
            mutation_rate: Mutation rate for genetic algorithm
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        self.optimization_log = []
        start_time = time.time()
        
        try:
            # Log optimization start
            self._log(f"=== VEHICLE ROUTER OPTIMIZATION ({optimization_method.upper()}) ===")
            self._log(f"Depot location: {depot_location}")
            self._log(f"Orders: {len(orders_df)}, Trucks: {len(trucks_df)}")
            self._log(f"Distance matrix: {distance_matrix.shape}")
            
            # Initialize optimizer based on method
            if optimization_method == 'enhanced':
                success = self._run_enhanced_optimization(
                    orders_df, trucks_df, distance_matrix, solver_timeout,
                    cost_weight, distance_weight, depot_location, depot_return
                )
            elif optimization_method == 'genetic':
                success = self._run_genetic_optimization(
                    orders_df, trucks_df, distance_matrix, solver_timeout,
                    cost_weight, distance_weight, depot_location, depot_return,
                    population_size, max_generations, mutation_rate
                )
            else:  # standard
                success = self._run_standard_optimization(
                    orders_df, trucks_df, distance_matrix, solver_timeout,
                    depot_location, depot_return, enable_greedy_routes
                )
            
            # Validate solution if requested
            if success and enable_validation:
                self._log("Validating optimization solution...")
                is_valid = self._validate_solution(orders_df, trucks_df)
                if not is_valid:
                    self._log("❌ Solution validation failed")
                    return False
                else:
                    self._log("✅ Solution validation passed")
            
            # Log completion
            total_time = time.time() - start_time
            if success:
                self._log(f"🎉 Optimization completed successfully in {total_time:.2f}s")
                self._log_solution_summary()
            else:
                self._log(f"❌ Optimization failed after {total_time:.2f}s")
            
            return success
            
        except Exception as e:
            self._log(f"❌ Optimization error: {str(e)}")
            logger.exception("Optimization error details:")
            return False
    
    def _run_enhanced_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame,
                                 distance_matrix: pd.DataFrame, solver_timeout: int,
                                 cost_weight: float, distance_weight: float,
                                 depot_location: str, depot_return: bool) -> bool:
        """Run enhanced MILP optimization"""
        self._log("Initializing Enhanced MILP optimizer...")
        
        try:
            from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer
            
            # Set default depot_return for enhanced model
            if depot_return is None:
                depot_return = False
                
            optimizer = EnhancedVrpOptimizer(
                orders_df=orders_df,
                trucks_df=trucks_df,
                distance_matrix=distance_matrix,
                depot_location=depot_location,
                depot_return=depot_return
            )
            
            # Set objective weights
            optimizer.set_objective_weights(cost_weight, distance_weight)
            self._log(f"Objective weights: cost={cost_weight:.2f}, distance={distance_weight:.2f}")
            
            # Build and solve model
            self._log("Building enhanced MILP model...")
            optimizer.build_model()
            
            self._log(f"Solving enhanced model (timeout: {solver_timeout}s)...")
            success = optimizer.solve(timeout=solver_timeout)
            
            if success:
                self.solution = optimizer.get_solution()
                self._log("✅ Enhanced optimization completed successfully")
                return True
            else:
                self._log("❌ Enhanced optimization failed")
                return False
                
        except Exception as e:
            self._log(f"❌ Enhanced optimization error: {str(e)}")
            return False
    
    def _run_genetic_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame,
                                distance_matrix: pd.DataFrame, solver_timeout: int,
                                cost_weight: float, distance_weight: float,
                                depot_location: str, depot_return: bool,
                                population_size: int, max_generations: int,
                                mutation_rate: float) -> bool:
        """Run genetic algorithm optimization"""
        self._log("Initializing Genetic Algorithm optimizer...")
        
        try:
            from vehicle_router.genetic_optimizer import GeneticVrpOptimizer
            
            # Set default depot_return for genetic algorithm
            if depot_return is None:
                depot_return = False
                
            optimizer = GeneticVrpOptimizer(
                orders_df=orders_df,
                trucks_df=trucks_df,
                distance_matrix=distance_matrix,
                depot_location=depot_location,
                depot_return=depot_return
            )
            
            # Set parameters
            optimizer.set_parameters(
                population_size=population_size,
                max_generations=max_generations,
                mutation_rate=mutation_rate
            )
            
            # Set objective weights
            optimizer.set_objective_weights(cost_weight, distance_weight)
            self._log(f"GA parameters: pop={population_size}, gen={max_generations}, mut={mutation_rate:.2f}")
            self._log(f"Objective weights: cost={cost_weight:.2f}, distance={distance_weight:.2f}")
            
            # Solve
            self._log(f"Running genetic algorithm (timeout: {solver_timeout}s)...")
            success = optimizer.solve(timeout=solver_timeout)
            
            if success:
                self.solution = optimizer.get_solution()
                self._log("✅ Genetic algorithm optimization completed successfully")
                return True
            else:
                self._log("❌ Genetic algorithm optimization failed")
                return False
                
        except Exception as e:
            self._log(f"❌ Genetic algorithm error: {str(e)}")
            return False
    
    def _run_standard_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame,
                                 distance_matrix: pd.DataFrame, solver_timeout: int,
                                 depot_location: str, depot_return: bool,
                                 enable_greedy_routes: bool) -> bool:
        """Run standard MILP optimization"""
        self._log("Initializing Standard MILP optimizer...")
        
        try:
            from vehicle_router.optimizer import VrpOptimizer
            
            # Set default depot_return for standard model
            if depot_return is None:
                depot_return = False
                
            optimizer = VrpOptimizer(
                orders_df=orders_df,
                trucks_df=trucks_df,
                distance_matrix=distance_matrix,
                depot_location=depot_location,
                depot_return=depot_return,
                enable_greedy_routes=enable_greedy_routes
            )
            
            self._log(f"Greedy route optimization: {'enabled' if enable_greedy_routes else 'disabled'}")
            
            # Build and solve model
            self._log("Building standard MILP model...")
            optimizer.build_model()
            
            self._log(f"Solving standard model (timeout: {solver_timeout}s)...")
            success = optimizer.solve()
            
            if success:
                self.solution = optimizer.get_solution()
                self._log("✅ Standard optimization completed successfully")
                return True
            else:
                self._log("❌ Standard optimization failed")
                return False
                
        except Exception as e:
            self._log(f"❌ Standard optimization error: {str(e)}")
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
            from vehicle_router.validation import SolutionValidator
            
            validator = SolutionValidator(self.solution, orders_df, trucks_df)
            validation_report = validator.validate_solution()
            
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
    
    def _log_solution_summary(self) -> None:
        """Log a summary of the optimization solution"""
        if not self.solution:
            return
            
        try:
            costs = self.solution.get('costs', {})
            selected_trucks = self.solution.get('selected_trucks', [])
            
            self._log("=== SOLUTION SUMMARY ===")
            self._log(f"Selected trucks: {selected_trucks}")
            self._log(f"Total cost: €{costs.get('total_cost', 0):.0f}")
            
            if 'total_distance' in costs:
                self._log(f"Total distance: {costs['total_distance']:.1f} km")
                
            if 'utilization' in self.solution:
                utilizations = [u['utilization_percent'] for u in self.solution['utilization'].values()]
                if utilizations:
                    avg_util = sum(utilizations) / len(utilizations)
                    self._log(f"Average utilization: {avg_util:.1f}%")
            
        except Exception as e:
            self._log(f"Error logging solution summary: {str(e)}")
    
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