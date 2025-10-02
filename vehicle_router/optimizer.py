"""
MILP Optimization Module

This module provides the VrpOptimizer class that implements Mixed Integer Linear Programming
(MILP) optimization for the Vehicle Routing Problem with order assignment. The optimizer
uses PuLP to build and solve the optimization model with comprehensive logging and
structured result formatting.

The MILP formulation includes:
- Decision variables for order-to-truck assignments
- Objective function minimizing total operational costs
- Capacity constraints ensuring trucks don't exceed their limits
- Order assignment constraints ensuring each order is delivered exactly once

Classes:
    VrpOptimizer: Main optimization engine for VRP problem solving
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pulp
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class VrpOptimizer:
    """
    Vehicle Routing Problem MILP Optimizer
    
    This class implements a Mixed Integer Linear Programming approach to solve
    the Vehicle Routing Problem with order assignment optimization. The optimizer
    minimizes total operational costs while respecting truck capacity constraints
    and ensuring all orders are delivered.
    
    The MILP formulation includes:
    - Binary decision variables x[i,j] for order i assigned to truck j
    - Binary decision variables y[j] for truck j being used
    - Objective: minimize sum of truck costs for selected trucks
    - Constraints: capacity limits, order assignment requirements
    
    Attributes:
        orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
        trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
        distance_matrix (pd.DataFrame): Distance matrix between postal codes
        model (pulp.LpProblem): The PuLP optimization model
        decision_vars (Dict): Dictionary containing decision variables
        is_solved (bool): Whether the model has been solved successfully
        solution_data (Dict): Structured solution results
        
    Example:
        >>> optimizer = VrpOptimizer(orders_df, trucks_df, distance_matrix)
        >>> optimizer.build_model()
        >>> success = optimizer.solve()
        >>> if success:
        ...     solution = optimizer.get_solution()
    """
    
    def __init__(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                 distance_matrix: pd.DataFrame, depot_location: Optional[str] = None,
                 depot_return: bool = False, enable_greedy_routes: bool = True,
                 max_orders_per_truck: int = 3):
        """
        Initialize the VRP Optimizer with problem data
        
        Args:
            orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
            trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
            distance_matrix (pd.DataFrame): Distance matrix between postal codes
            depot_location (Optional[str]): Depot postal code. If None, uses default '08020'.
            depot_return (bool): Whether trucks must return to depot after deliveries. Default False.
            enable_greedy_routes (bool): Whether to apply greedy route optimization. Default True.
            max_orders_per_truck (int): Maximum number of orders per truck. Default 3.
            
        Raises:
            ValueError: If input data is invalid or inconsistent
            TypeError: If input data types are incorrect
        """
        logger.info("Initializing VRP Optimizer...")
        
        # Validate input data
        self._validate_input_data(orders_df, trucks_df, distance_matrix)
        
        # Store input data
        self.orders_df = orders_df.copy()
        self.trucks_df = trucks_df.copy()
        self.distance_matrix = distance_matrix.copy()
        
        # Set depot location
        if depot_location is None:
            self.depot_location = '08020'  # Default depot
        else:
            if depot_location in self.distance_matrix.index:
                self.depot_location = depot_location
            else:
                logger.warning(f"Depot location '{depot_location}' not found in distance matrix, using default '08020'")
                self.depot_location = '08020'
        
        # Store configuration options
        self.depot_return = depot_return
        self.enable_greedy_routes = enable_greedy_routes
        
        # Validate and store max orders per truck constraint
        if max_orders_per_truck < 1:
            raise ValueError("max_orders_per_truck must be at least 1")
        self.max_orders_per_truck = max_orders_per_truck
        
        logger.info(f"Depot location: {self.depot_location}")
        logger.info(f"Depot return enabled: {self.depot_return}")
        logger.info(f"Greedy route optimization enabled: {self.enable_greedy_routes}")
        logger.info(f"Maximum orders per truck: {self.max_orders_per_truck}")
        
        # Initialize optimization components
        self.model = None
        self.decision_vars = {}
        self.is_solved = False
        self.solution_data = {}
        
        # Extract problem dimensions
        self.orders = self.orders_df['order_id'].tolist()
        self.trucks = self.trucks_df['truck_id'].tolist()
        self.n_orders = len(self.orders)
        self.n_trucks = len(self.trucks)
        
        logger.info(f"Optimizer initialized with {self.n_orders} orders and {self.n_trucks} trucks")
        logger.info(f"Total order volume: {self.orders_df['volume'].sum():.1f} m³")
        logger.info(f"Total truck capacity: {self.trucks_df['capacity'].sum():.1f} m³")
    
    def _validate_input_data(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                           distance_matrix: pd.DataFrame) -> None:
        """Validate input data for consistency and completeness"""
        # Validate orders DataFrame
        if not isinstance(orders_df, pd.DataFrame):
            raise TypeError("orders_df must be a pandas DataFrame")
        
        required_order_cols = ['order_id', 'volume', 'postal_code']
        missing_cols = [col for col in required_order_cols if col not in orders_df.columns]
        if missing_cols:
            raise ValueError(f"orders_df missing required columns: {missing_cols}")
        
        if orders_df.empty or orders_df['volume'].min() <= 0:
            raise ValueError("orders_df cannot be empty and all volumes must be positive")
        
        # Validate trucks DataFrame
        if not isinstance(trucks_df, pd.DataFrame):
            raise TypeError("trucks_df must be a pandas DataFrame")
        
        required_truck_cols = ['truck_id', 'capacity', 'cost']
        missing_cols = [col for col in required_truck_cols if col not in trucks_df.columns]
        if missing_cols:
            raise ValueError(f"trucks_df missing required columns: {missing_cols}")
        
        if trucks_df.empty or trucks_df['capacity'].min() <= 0 or trucks_df['cost'].min() < 0:
            raise ValueError("trucks_df cannot be empty, capacities must be positive, costs non-negative")
        
        # Check feasibility
        total_volume = orders_df['volume'].sum()
        total_capacity = trucks_df['capacity'].sum()
        
        if total_volume > total_capacity:
            raise ValueError(f"Problem infeasible: total volume ({total_volume:.1f}) exceeds total capacity ({total_capacity:.1f})")
        
        logger.info("Input data validation completed successfully")
    
    def build_model(self) -> None:
        """Build the MILP optimization model with decision variables and constraints"""
        logger.info("Building MILP optimization model...")
        
        # Create the optimization problem
        self.model = pulp.LpProblem("Vehicle_Routing_Problem", pulp.LpMinimize)
        
        # Create decision variables and set objective
        self._create_decision_variables()
        self._set_objective_function()
        
        # Add constraints
        self._add_order_assignment_constraints()
        self._add_capacity_constraints()
        self._add_max_orders_constraints()
        self._add_truck_usage_constraints()
        
        # Log model statistics
        logger.info(f"MILP model built: {len(self.model.variables())} variables, {len(self.model.constraints)} constraints")
        logger.info(f"  Problem type: {self.model.sense}")
    
    def _create_decision_variables(self) -> None:
        """
        Create binary decision variables for the MILP model
        
        Creates two types of decision variables:
        - x[order_id, truck_id]: Binary variable indicating if order is assigned to truck
        - y[truck_id]: Binary variable indicating if truck is used in the solution
        """
        # Order assignment variables: x[i,j] = 1 if order i assigned to truck j
        self.decision_vars['x'] = {}
        for order_id in self.orders:
            for truck_id in self.trucks:
                var_name = f"assign_{order_id}_to_truck_{truck_id}"
                self.decision_vars['x'][(order_id, truck_id)] = pulp.LpVariable(
                    var_name, cat='Binary'
                )
        
        # Truck usage variables: y[j] = 1 if truck j is used
        self.decision_vars['y'] = {}
        for truck_id in self.trucks:
            var_name = f"use_truck_{truck_id}"
            self.decision_vars['y'][truck_id] = pulp.LpVariable(
                var_name, cat='Binary'
            )
        
        logger.info(f"Created {len(self.decision_vars['x'])} assignment variables")
        logger.info(f"Created {len(self.decision_vars['y'])} truck usage variables")
    
    def _set_objective_function(self) -> None:
        """
        Set the objective function to minimize total operational costs
        
        The objective function minimizes the sum of costs for all selected trucks.
        This is a simplified cost model focusing on truck selection costs.
        """
        # Get truck costs as a dictionary for easy lookup
        truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
        
        # Objective: minimize total cost of selected trucks
        objective_terms = []
        for truck_id in self.trucks:
            cost = truck_costs[truck_id]
            truck_var = self.decision_vars['y'][truck_id]
            objective_terms.append(cost * truck_var)
        
        self.model += pulp.lpSum(objective_terms), "Total_Cost_Minimization"
        
        logger.info("Objective function set to minimize total truck costs")
    
    def _add_order_assignment_constraints(self) -> None:
        """
        Add constraints ensuring each order is assigned to exactly one truck
        
        For each order i: sum over all trucks j of x[i,j] = 1
        This ensures every order is delivered exactly once.
        """
        logger.info("Adding order assignment constraints...")
        
        for order_id in self.orders:
            # Each order must be assigned to exactly one truck
            assignment_sum = pulp.lpSum([
                self.decision_vars['x'][(order_id, truck_id)] 
                for truck_id in self.trucks
            ])
            
            constraint_name = f"assign_order_{order_id}"
            self.model += assignment_sum == 1, constraint_name
        
        logger.info(f"Added {len(self.orders)} order assignment constraints")
    
    def _add_capacity_constraints(self) -> None:
        """
        Add capacity constraints ensuring no truck exceeds its capacity limit
        
        For each truck j: sum over all orders i of (volume[i] * x[i,j]) <= capacity[j]
        This ensures truck capacity limits are respected.
        """
        logger.info("Adding capacity constraints...")
        
        # Get order volumes as a dictionary for easy lookup
        order_volumes = dict(zip(self.orders_df['order_id'], self.orders_df['volume']))
        truck_capacities = dict(zip(self.trucks_df['truck_id'], self.trucks_df['capacity']))
        
        for truck_id in self.trucks:
            # Sum of assigned order volumes must not exceed truck capacity
            volume_sum = pulp.lpSum([
                order_volumes[order_id] * self.decision_vars['x'][(order_id, truck_id)]
                for order_id in self.orders
            ])
            
            capacity = truck_capacities[truck_id]
            constraint_name = f"capacity_truck_{truck_id}"
            self.model += volume_sum <= capacity, constraint_name
            
            logger.info(f"Added capacity constraint for Truck {truck_id}: max {capacity} m³")
        
        logger.info(f"Added {len(self.trucks)} capacity constraints")
    
    def _add_max_orders_constraints(self) -> None:
        """
        Add constraints ensuring no truck exceeds the maximum number of orders
        
        For each truck j: sum over all orders i of x[i,j] <= max_orders_per_truck
        This ensures each truck handles at most the specified number of orders.
        """
        logger.info(f"Adding maximum orders per truck constraints (limit: {self.max_orders_per_truck})...")
        
        for truck_id in self.trucks:
            # Sum of assigned orders must not exceed maximum
            orders_sum = pulp.lpSum([
                self.decision_vars['x'][(order_id, truck_id)]
                for order_id in self.orders
            ])
            
            constraint_name = f"max_orders_truck_{truck_id}"
            self.model += orders_sum <= self.max_orders_per_truck, constraint_name
            
            logger.info(f"Added max orders constraint for Truck {truck_id}: max {self.max_orders_per_truck} orders")
        
        logger.info(f"Added {len(self.trucks)} maximum orders constraints")
    
    def _add_truck_usage_constraints(self) -> None:
        """
        Add constraints linking truck usage to order assignments
        
        For each truck j: y[j] >= x[i,j] for all orders i
        This ensures that if any order is assigned to a truck, the truck is marked as used.
        """
        logger.info("Adding truck usage constraints...")
        
        constraint_count = 0
        for truck_id in self.trucks:
            truck_usage_var = self.decision_vars['y'][truck_id]
            
            for order_id in self.orders:
                assignment_var = self.decision_vars['x'][(order_id, truck_id)]
                
                # If order is assigned to truck, truck must be used
                constraint_name = f"usage_truck_{truck_id}_order_{order_id}"
                self.model += truck_usage_var >= assignment_var, constraint_name
                constraint_count += 1
        
        logger.info(f"Added {constraint_count} truck usage constraints")
    
    def solve(self, timeout: int = 300) -> bool:
        """Solve the MILP optimization problem"""
        if self.model is None:
            raise RuntimeError("Model must be built before solving. Call build_model() first.")
        
        logger.info("Starting MILP optimization solver...")
        start_time = time.time()
        
        try:
            self.model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))
            solve_time = time.time() - start_time
            status = pulp.LpStatus[self.model.status]
            
            logger.info(f"Solver completed in {solve_time:.2f}s with status: {status}")
            
            if self.model.status == pulp.LpStatusOptimal:
                objective_value = pulp.value(self.model.objective)
                logger.info(f"Optimal solution found with total cost: €{objective_value:.0f}")
                self.is_solved = True
                self._extract_solution()
                return True
            else:
                logger.error(f"Solver failed with status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            return False
    
    def _extract_solution(self) -> None:
        """
        Extract and structure the optimization solution from decision variables
        
        Processes the solved decision variables to create structured solution data
        including order assignments, selected trucks, costs, and utilization metrics.
        """
        logger.info("Extracting optimization solution...")
        
        # Extract order assignments
        assignments = []
        for order_id in self.orders:
            for truck_id in self.trucks:
                var_value = self.decision_vars['x'][(order_id, truck_id)].varValue
                if var_value is not None and var_value > 0.5:  # Binary variable is 1
                    assignments.append({
                        'order_id': order_id,
                        'truck_id': truck_id,
                        'assigned': True
                    })
        
        # Extract selected trucks
        selected_trucks = []
        for truck_id in self.trucks:
            var_value = self.decision_vars['y'][truck_id].varValue
            if var_value is not None and var_value > 0.5:  # Binary variable is 1
                selected_trucks.append(truck_id)
        
        selected_trucks.sort()
        
        # Calculate solution metrics
        total_cost = pulp.value(self.model.objective)
        
        # Calculate truck utilization
        truck_utilization = {}
        truck_capacities = dict(zip(self.trucks_df['truck_id'], self.trucks_df['capacity']))
        order_volumes = dict(zip(self.orders_df['order_id'], self.orders_df['volume']))
        
        for truck_id in selected_trucks:
            assigned_orders = [a['order_id'] for a in assignments if a['truck_id'] == truck_id]
            used_volume = sum(order_volumes[order_id] for order_id in assigned_orders)
            capacity = truck_capacities[truck_id]
            utilization = (used_volume / capacity) * 100 if capacity > 0 else 0
            
            truck_utilization[truck_id] = {
                'used_volume': used_volume,
                'capacity': capacity,
                'utilization_percent': utilization,
                'assigned_orders': assigned_orders
            }
        
        # Apply greedy route optimization if enabled
        if self.enable_greedy_routes:
            logger.info("Applying greedy route optimization...")
            self._optimize_routes_greedy(assignments, selected_trucks)
        
        # Store structured solution data
        self.solution_data = {
            'assignments': assignments,
            'selected_trucks': selected_trucks,
            'total_cost': total_cost,
            'truck_utilization': truck_utilization,
            'solution_summary': {
                'trucks_used': len(selected_trucks),
                'orders_delivered': len(assignments),
                'total_volume_delivered': sum(order_volumes[a['order_id']] for a in assignments),
                'average_utilization': np.mean([u['utilization_percent'] for u in truck_utilization.values()]) if truck_utilization else 0
            }
        }
        
        # Log solution summary
        logger.info("Solution extraction completed:")
        logger.info(f"  Selected trucks: {selected_trucks}")
        logger.info(f"  Total cost: €{total_cost:.0f}")
        logger.info(f"  Orders delivered: {len(assignments)}/{len(self.orders)}")
        
        for truck_id in selected_trucks:
            util_info = truck_utilization[truck_id]
            logger.info(f"  Truck {truck_id}: {util_info['used_volume']:.1f}/{util_info['capacity']:.1f} m³ ({util_info['utilization_percent']:.1f}% utilization)")
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Return structured optimization results as pandas DataFrames and dictionaries
        
        Provides comprehensive solution data including assignments, routes, costs,
        and utilization metrics in a structured format suitable for further analysis
        and visualization.
        
        Returns:
            Dict[str, Any]: Structured solution data containing:
                - assignments_df: DataFrame with order-to-truck assignments
                - routes_df: DataFrame with route information for each truck
                - costs: Dictionary with cost breakdown
                - utilization: Dictionary with truck utilization metrics
                - summary: Dictionary with high-level solution summary
                
        Raises:
            RuntimeError: If model hasn't been solved successfully
        """
        if not self.is_solved:
            raise RuntimeError("Model must be solved before getting solution. Call solve() first.")
        
        logger.info("Formatting solution data for output...")
        
        # Create assignments DataFrame
        assignments_df = pd.DataFrame(self.solution_data['assignments'])
        
        # Create routes DataFrame with basic route information for standard optimizer
        routes_data = []
        depot_location = '08020'  # Default depot for standard optimizer
        
        for truck_id in self.solution_data['selected_trucks']:
            assigned_orders = [a['order_id'] for a in self.solution_data['assignments'] if a['truck_id'] == truck_id]
            
            # Get postal codes for assigned orders
            order_postal_codes = []
            for order_id in assigned_orders:
                postal_code = self.orders_df[self.orders_df['order_id'] == order_id]['postal_code'].iloc[0]
                order_postal_codes.append(postal_code)
            
            # Use optimized route if available from greedy algorithm, otherwise create simple route
            if hasattr(self, '_optimized_routes') and truck_id in self._optimized_routes:
                # Use the optimized route from greedy algorithm
                route_sequence = self._optimized_routes[truck_id]['route_sequence']
                route_distance = self._optimized_routes[truck_id]['route_distance']
            else:
                # Create a simple route sequence: depot -> orders -> (depot if depot_return)
                sorted_postal_codes = sorted(order_postal_codes)
                route_sequence = [self.depot_location] + sorted_postal_codes
                if self.depot_return:
                    route_sequence.append(self.depot_location)
                
                # Calculate basic route distance using distance matrix if available
                route_distance = self._calculate_route_distance(route_sequence)
            
            routes_data.append({
                'truck_id': truck_id,
                'assigned_orders': assigned_orders,
                'postal_codes': order_postal_codes,
                'route_sequence': route_sequence,
                'route_distance': route_distance,
                'num_orders': len(assigned_orders),
                'depot_location': depot_location
            })
        
        routes_df = pd.DataFrame(routes_data)
        
        # Prepare cost breakdown
        truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
        cost_breakdown = {
            'truck_costs': {truck_id: truck_costs[truck_id] for truck_id in self.solution_data['selected_trucks']},
            'total_truck_cost': sum(truck_costs[truck_id] for truck_id in self.solution_data['selected_trucks']),
            'total_cost': self.solution_data['total_cost']
        }
        
        # Prepare final solution structure
        solution = {
            'assignments_df': assignments_df,
            'routes_df': routes_df,
            'costs': cost_breakdown,
            'utilization': self.solution_data['truck_utilization'],
            'summary': self.solution_data['solution_summary'],
            'selected_trucks': self.solution_data['selected_trucks'],
            'optimization_status': 'optimal'
        }
        
        logger.info("Solution data formatted successfully")
        return solution
    
    def get_solution_summary_text(self) -> str:
        """
        Generate a human-readable text summary of the optimization solution
        
        Returns:
            str: Formatted text summary matching the required output format
        """
        if not self.is_solved:
            return "No solution available - model not solved"
        
        lines = ["=== VEHICLE ROUTER ==="]
        
        # Selected trucks
        selected_trucks = self.solution_data['selected_trucks']
        lines.append(f"Selected Trucks: {selected_trucks}")
        
        # Truck assignments
        for truck_id in selected_trucks:
            assigned_orders = [a['order_id'] for a in self.solution_data['assignments'] if a['truck_id'] == truck_id]
            lines.append(f"Truck {truck_id} -> Orders {assigned_orders}")
        
        # Total cost
        total_cost = self.solution_data['total_cost']
        lines.append(f"Total Cost: €{total_cost:.0f}")
        
        return "\n".join(lines)
    
    def _optimize_routes_greedy(self, assignments: List[Dict], selected_trucks: List[int]) -> None:
        """
        Apply greedy route optimization to minimize travel distances for each truck
        
        This algorithm tests all possible route permutations for each truck and selects
        the one with minimum total distance. It considers depot_return setting.
        
        Args:
            assignments (List[Dict]): Order assignments from MILP solution
            selected_trucks (List[int]): List of selected truck IDs
        """
        from itertools import permutations
        
        logger.info("🔍 Starting greedy route optimization algorithm...")
        logger.info(f"   Processing {len(selected_trucks)} trucks with depot_return={self.depot_return}")
        
        total_distance_before = 0
        total_distance_after = 0
        
        for truck_id in selected_trucks:
            logger.info(f"🚛 Optimizing routes for Truck {truck_id}...")
            
            # Get orders assigned to this truck
            assigned_orders = [a['order_id'] for a in assignments if a['truck_id'] == truck_id]
            
            if len(assigned_orders) <= 1:
                logger.info(f"   Truck {truck_id} has {len(assigned_orders)} order(s), no optimization needed")
                continue
            
            # Get postal codes for assigned orders
            order_locations = []
            for order_id in assigned_orders:
                postal_code = self.orders_df[self.orders_df['order_id'] == order_id]['postal_code'].iloc[0]
                order_locations.append(postal_code)
            
            logger.info(f"   Orders: {assigned_orders}")
            logger.info(f"   Locations: {order_locations}")
            
            # Calculate current route distance (simple sorted order)
            current_route = [self.depot_location] + sorted(order_locations)
            if self.depot_return:
                current_route.append(self.depot_location)
            
            current_distance = self._calculate_route_distance(current_route)
            total_distance_before += current_distance
            
            logger.info(f"   Current route: {' → '.join(current_route)}")
            logger.info(f"   Current distance: {current_distance:.1f} km")
            
            # Test all permutations of order locations
            best_route = current_route
            best_distance = current_distance
            permutation_count = 0
            
            logger.info(f"   Testing {len(list(permutations(order_locations)))} route permutations...")
            
            for perm in permutations(order_locations):
                permutation_count += 1
                test_route = [self.depot_location] + list(perm)
                if self.depot_return:
                    test_route.append(self.depot_location)
                
                test_distance = self._calculate_route_distance(test_route)
                
                if test_distance < best_distance:
                    best_route = test_route
                    best_distance = test_distance
                    logger.info(f"   ✨ New best route found: {' → '.join(best_route)} ({best_distance:.1f} km)")
            
            total_distance_after += best_distance
            
            # Update the route sequence in solution data (will be used in get_solution)
            improvement = current_distance - best_distance
            improvement_pct = (improvement / current_distance * 100) if current_distance > 0 else 0
            
            logger.info(f"   🎯 Final optimized route: {' → '.join(best_route)}")
            logger.info(f"   📊 Distance improvement: {improvement:.1f} km ({improvement_pct:.1f}%)")
            logger.info(f"   📈 Tested {permutation_count} permutations")
            
            # Store the optimized route (will be used in get_solution method)
            if not hasattr(self, '_optimized_routes'):
                self._optimized_routes = {}
            self._optimized_routes[truck_id] = {
                'route_sequence': best_route,
                'route_distance': best_distance,
                'improvement': improvement,
                'permutations_tested': permutation_count
            }
        
        # Log overall improvement
        total_improvement = total_distance_before - total_distance_after
        total_improvement_pct = (total_improvement / total_distance_before * 100) if total_distance_before > 0 else 0
        
        logger.info("🏁 Greedy route optimization completed!")
        logger.info(f"   📊 Total distance before: {total_distance_before:.1f} km")
        logger.info(f"   📊 Total distance after: {total_distance_after:.1f} km")
        logger.info(f"   🎯 Total improvement: {total_improvement:.1f} km ({total_improvement_pct:.1f}%)")
    
    def _calculate_route_distance(self, route_sequence: List[str]) -> float:
        """
        Calculate total distance for a route sequence
        
        Args:
            route_sequence (List[str]): List of postal codes in route order
            
        Returns:
            float: Total distance in km
        """
        if len(route_sequence) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route_sequence) - 1):
            from_loc = route_sequence[i]
            to_loc = route_sequence[i + 1]
            
            if from_loc in self.distance_matrix.index and to_loc in self.distance_matrix.columns:
                distance = self.distance_matrix.loc[from_loc, to_loc]
                total_distance += distance
            else:
                logger.warning(f"Distance not found for {from_loc} → {to_loc}")
        
        return total_distance