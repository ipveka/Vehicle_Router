"""
Optimization Runner Module

Manages the complete optimization workflow for the Vehicle Router application.
"""

import streamlit as st
import pandas as pd
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Optimization Runner for Vehicle Router Application"""
    
    def __init__(self):
        self.solution = None
        self.optimization_log = []
    
    def run_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                        distance_matrix: pd.DataFrame, optimization_method: str = 'standard',
                        solver_timeout: int = 300, depot_location: str = '08020',
                        depot_return: bool = False, max_orders_per_truck: int = 3,
                        use_real_distances: bool = True, enable_greedy_routes: bool = True,
                        cost_weight: float = 1.0, distance_weight: float = 0.0,
                        population_size: int = 50, max_generations: int = 100,
                        mutation_rate: float = 0.1) -> bool:
        """Run the complete optimization workflow"""
        self.optimization_log = []
        self.solution = None
        
        try:
            if optimization_method == 'standard':
                success = self._run_standard_optimization(
                    orders_df, trucks_df, distance_matrix,
                    solver_timeout, depot_location, depot_return,
                    max_orders_per_truck, use_real_distances, enable_greedy_routes,
                    cost_weight, distance_weight
                )
            elif optimization_method == 'enhanced':
                success = self._run_enhanced_optimization(
                    orders_df, trucks_df, distance_matrix,
                    solver_timeout, depot_location, depot_return,
                    max_orders_per_truck, use_real_distances,
                    cost_weight, distance_weight
                )
            elif optimization_method == 'genetic':
                success = self._run_genetic_optimization(
                    orders_df, trucks_df, distance_matrix,
                    solver_timeout, depot_location, depot_return,
                    max_orders_per_truck, use_real_distances,
                    cost_weight, distance_weight, population_size,
                    max_generations, mutation_rate
                )
            else:
                self._log(f"❌ Unknown optimization method: {optimization_method}")
                return False
            
            return success
            
        except Exception as e:
            self._log(f"❌ Optimization workflow error: {str(e)}")
            return False
    
    def _run_standard_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame,
                                   distance_matrix: pd.DataFrame, solver_timeout: int,
                                   depot_location: str, depot_return: bool, max_orders_per_truck: int,
                                   use_real_distances: bool, enable_greedy_routes: bool,
                                   cost_weight: float, distance_weight: float) -> bool:
        """Run Standard MILP + Greedy optimization"""
        self._log("Initializing Standard MILP + Greedy optimizer...")
        
        try:
            from vehicle_router.optimizer import VrpOptimizer
            
            optimizer = VrpOptimizer(
                orders_df=orders_df,
                trucks_df=trucks_df,
                distance_matrix=distance_matrix,
                depot_location=depot_location,
                depot_return=depot_return,
                max_orders_per_truck=max_orders_per_truck,
                enable_greedy_routes=enable_greedy_routes
            )
            
            self._log("Building Standard MILP model...")
            optimizer.build_model()
            
            self._log(f"Solving Standard MILP model (timeout: {solver_timeout}s)...")
            success = optimizer.solve(timeout=solver_timeout)
            
            if success:
                self._log("✅ Standard MILP optimization completed successfully")
                self.solution = optimizer.get_solution()
                return True
            else:
                self._log("❌ Standard MILP optimization failed")
                return False
                
        except Exception as e:
            self._log(f"❌ Standard MILP optimization error: {str(e)}")
            return False
    
    def _run_enhanced_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame,
                                   distance_matrix: pd.DataFrame, solver_timeout: int,
                                   depot_location: str, depot_return: bool, max_orders_per_truck: int,
                                   use_real_distances: bool, cost_weight: float, distance_weight: float) -> bool:
        """Run enhanced MILP optimization"""
        self._log("Initializing Enhanced MILP optimizer...")
        
        try:
            from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer
            
            optimizer = EnhancedVrpOptimizer(
                orders_df=orders_df,
                trucks_df=trucks_df,
                distance_matrix=distance_matrix,
                depot_location=depot_location,
                depot_return=depot_return,
                max_orders_per_truck=max_orders_per_truck
            )
            
            self._log("Building enhanced MILP model...")
            optimizer.build_model()
            
            self._log(f"Solving enhanced model (timeout: {solver_timeout}s)...")
            success = optimizer.solve(timeout=solver_timeout)
            
            if success:
                self._log("✅ Enhanced optimization completed successfully")
                self.solution = optimizer.get_solution()
                return True
            else:
                self._log("❌ Enhanced optimization failed")
                return False
                
        except Exception as e:
            self._log(f"❌ Enhanced optimization error: {str(e)}")
            return False
    
    def _run_genetic_optimization(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame,
                                  distance_matrix: pd.DataFrame, solver_timeout: int,
                                  depot_location: str, depot_return: bool, max_orders_per_truck: int,
                                  use_real_distances: bool, cost_weight: float, distance_weight: float,
                                  population_size: int, max_generations: int, mutation_rate: float) -> bool:
        """Run genetic algorithm optimization"""
        self._log("Initializing Genetic Algorithm optimizer...")
        
        try:
            from vehicle_router.genetic_optimizer import GeneticVrpOptimizer
            
            optimizer = GeneticVrpOptimizer(
                orders_df=orders_df,
                trucks_df=trucks_df,
                distance_matrix=distance_matrix,
                depot_location=depot_location,
                depot_return=depot_return,
                max_orders_per_truck=max_orders_per_truck
            )
            
            optimizer.set_parameters(
                population_size=population_size,
                max_generations=max_generations,
                mutation_rate=mutation_rate
            )
            
            self._log(f"Running genetic algorithm (timeout: {solver_timeout}s)...")
            success = optimizer.solve(timeout=solver_timeout)
            
            if success:
                self._log("✅ Genetic algorithm optimization completed successfully")
                self.solution = optimizer.get_solution()
                return True
            else:
                self._log("❌ Genetic algorithm optimization failed")
                return False
                
        except Exception as e:
            self._log(f"❌ Genetic algorithm error: {str(e)}")
            return False
    
    def _log(self, message: str) -> None:
        """Add message to optimization log"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.optimization_log.append(log_message)
        logger.info(message)
        
        # Also log to Streamlit session state for UI display
        if 'optimization_log' not in st.session_state:
            st.session_state.optimization_log = []
        st.session_state.optimization_log.append(log_message)