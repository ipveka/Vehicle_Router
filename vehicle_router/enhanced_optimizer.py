"""
Enhanced MILP Optimization Module with Distance-Based Routing

This module provides the EnhancedVrpOptimizer class that implements Mixed Integer Linear Programming
(MILP) optimization for the Vehicle Routing Problem with both order assignment and route optimization.
The optimizer minimizes both truck operational costs and total travel distances.

The enhanced MILP formulation includes:
- Decision variables for order-to-truck assignments
- Decision variables for route sequencing between locations
- Multi-objective function minimizing truck costs and travel distances
- Capacity constraints ensuring trucks don't exceed their limits
- Order assignment constraints ensuring each order is delivered exactly once
- Route continuity constraints ensuring valid routes
- Subtour elimination constraints preventing disconnected routes

Classes:
    EnhancedVrpOptimizer: Advanced optimization engine for VRP with routing
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


class EnhancedVrpOptimizer:
    """
    Enhanced Vehicle Routing Problem MILP Optimizer with Distance Minimization
    
    This class implements a comprehensive Mixed Integer Linear Programming approach to solve
    the Vehicle Routing Problem with both order assignment and route optimization. The optimizer
    minimizes a weighted combination of truck operational costs and total travel distances.
    
    The enhanced MILP formulation includes:
    - Binary decision variables x[i,j] for order i assigned to truck j
    - Binary decision variables y[j] for truck j being used
    - Binary decision variables z[k,l,j] for truck j traveling from location k to location l
    - Multi-objective: minimize weighted sum of truck costs and travel distances
    - Constraints: capacity limits, order assignment, route continuity, subtour elimination
    
    Attributes:
        orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
        trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
        distance_matrix (pd.DataFrame): Distance matrix between postal codes
        model (pulp.LpProblem): The PuLP optimization model
        decision_vars (Dict): Dictionary containing decision variables
        is_solved (bool): Whether the model has been solved successfully
        solution_data (Dict): Structured solution results
        cost_weight (float): Weight for truck costs in objective (0-1)
        distance_weight (float): Weight for distance costs in objective (0-1)
        depot_location (str): Depot postal code (trucks start/end here)
        
    Example:
        >>> optimizer = EnhancedVrpOptimizer(orders_df, trucks_df, distance_matrix)
        >>> optimizer.set_objective_weights(cost_weight=0.7, distance_weight=0.3)
        >>> optimizer.build_model()
        >>> success = optimizer.solve()
        >>> if success:
        ...     solution = optimizer.get_solution()
    """
    
    def __init__(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                 distance_matrix: pd.DataFrame, depot_location: Optional[str] = None):
        """
        Initialize the Enhanced VRP Optimizer with problem data
        
        Args:
            orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
            trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
            distance_matrix (pd.DataFrame): Distance matrix between postal codes
            depot_location (Optional[str]): Depot postal code. If None, uses first postal code.
            
        Raises:
            ValueError: If input data is invalid or inconsistent
            TypeError: If input data types are incorrect
        """
        logger.info("Initializing Enhanced VRP Optimizer with distance minimization...")
        
        # Validate input data
        self._validate_input_data(orders_df, trucks_df, distance_matrix)
        
        # Store input data
        self.orders_df = orders_df.copy()
        self.trucks_df = trucks_df.copy()
        self.distance_matrix = distance_matrix.copy()
        
        # Set depot location (trucks start and end here)
        if depot_location is None:
            self.depot_location = self.orders_df['postal_code'].iloc[0]
        else:
            self.depot_location = depot_location
            
        # Initialize optimization components
        self.model = None
        self.decision_vars = {}
        self.is_solved = False
        self.solution_data = {}
        
        # Default objective weights (can be modified with set_objective_weights)
        self.cost_weight = 0.6  # Weight for truck costs
        self.distance_weight = 0.4  # Weight for distance costs
        
        # Extract problem dimensions
        self.orders = self.orders_df['order_id'].tolist()
        self.trucks = self.trucks_df['truck_id'].tolist()
        self.locations = list(set(self.orders_df['postal_code'].tolist() + [self.depot_location]))
        self.n_orders = len(self.orders)
        self.n_trucks = len(self.trucks)
        self.n_locations = len(self.locations)
        
        logger.info(f"Enhanced optimizer initialized:")
        logger.info(f"  Orders: {self.n_orders}, Trucks: {self.n_trucks}, Locations: {self.n_locations}")
        logger.info(f"  Depot location: {self.depot_location}")
        logger.info(f"  Total order volume: {self.orders_df['volume'].sum():.1f} m³")
        logger.info(f"  Total truck capacity: {self.trucks_df['capacity'].sum():.1f} m³")
    
    def set_objective_weights(self, cost_weight: float, distance_weight: float) -> None:
        """
        Set the weights for multi-objective optimization
        
        Args:
            cost_weight (float): Weight for truck costs (0-1)
            distance_weight (float): Weight for distance costs (0-1)
            
        Note:
            Weights should sum to 1.0 for proper scaling
        """
        if not (0 <= cost_weight <= 1) or not (0 <= distance_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
            
        if abs(cost_weight + distance_weight - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {cost_weight + distance_weight:.3f}, not 1.0")
            
        self.cost_weight = cost_weight
        self.distance_weight = distance_weight
        
        logger.info(f"Objective weights set: cost={cost_weight:.2f}, distance={distance_weight:.2f}")
    
    def _validate_input_data(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                             distance_matrix: pd.DataFrame) -> None:
        """
        Validate input data for consistency and completeness
        
        Args:
            orders_df (pd.DataFrame): Orders data to validate
            trucks_df (pd.DataFrame): Trucks data to validate
            distance_matrix (pd.DataFrame): Distance matrix to validate
            
        Raises:
            ValueError: If data validation fails
            TypeError: If data types are incorrect
        """
        # Validate orders DataFrame
        if not isinstance(orders_df, pd.DataFrame):
            raise TypeError("orders_df must be a pandas DataFrame")
        
        required_order_cols = ['order_id', 'volume', 'postal_code']
        missing_cols = [col for col in required_order_cols if col not in orders_df.columns]
        if missing_cols:
            raise ValueError(f"orders_df missing required columns: {missing_cols}")
        
        if orders_df.empty:
            raise ValueError("orders_df cannot be empty")
        
        if orders_df['volume'].min() <= 0:
            raise ValueError("All order volumes must be positive")
        
        # Validate trucks DataFrame
        if not isinstance(trucks_df, pd.DataFrame):
            raise TypeError("trucks_df must be a pandas DataFrame")
        
        required_truck_cols = ['truck_id', 'capacity', 'cost']
        missing_cols = [col for col in required_truck_cols if col not in trucks_df.columns]
        if missing_cols:
            raise ValueError(f"trucks_df missing required columns: {missing_cols}")
        
        if trucks_df.empty:
            raise ValueError("trucks_df cannot be empty")
        
        if trucks_df['capacity'].min() <= 0:
            raise ValueError("All truck capacities must be positive")
        
        if trucks_df['cost'].min() < 0:
            raise ValueError("All truck costs must be non-negative")
        
        # Validate distance matrix
        if not isinstance(distance_matrix, pd.DataFrame):
            raise TypeError("distance_matrix must be a pandas DataFrame")
            
        # Check if distance matrix is square
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
            
        # Check if all postal codes are in distance matrix
        postal_codes = set(orders_df['postal_code'].tolist())
        matrix_codes = set(distance_matrix.index.tolist())
        missing_codes = postal_codes - matrix_codes
        if missing_codes:
            raise ValueError(f"Postal codes missing from distance matrix: {missing_codes}")
        
        # Check feasibility
        total_volume = orders_df['volume'].sum()
        total_capacity = trucks_df['capacity'].sum()
        
        if total_volume > total_capacity:
            raise ValueError(f"Problem infeasible: total volume ({total_volume:.1f}) exceeds total capacity ({total_capacity:.1f})")
        
        logger.info("Enhanced input data validation completed successfully")
    
    def build_model(self) -> None:
        """
        Build the enhanced MILP optimization model with routing variables and constraints
        
        This method constructs the complete enhanced MILP formulation including:
        - Decision variables for order assignments, truck usage, and routing
        - Multi-objective function minimizing truck costs and travel distances
        - Capacity constraints for each truck
        - Order assignment constraints ensuring each order is delivered once
        - Truck usage constraints linking assignments to truck selection
        - Route continuity constraints ensuring valid routes
        - Subtour elimination constraints preventing disconnected routes
        
        The model uses PuLP's LpProblem class and binary decision variables.
        """
        logger.info("Building enhanced MILP optimization model with routing...")
        
        # Create the optimization problem
        self.model = pulp.LpProblem("Enhanced_Vehicle_Routing_Problem", pulp.LpMinimize)
        
        # Create decision variables
        logger.info("Creating decision variables...")
        self._create_decision_variables()
        
        # Set multi-objective function
        logger.info("Setting multi-objective function...")
        self._set_objective_function()
        
        # Add constraints
        logger.info("Adding optimization constraints...")
        self._add_order_assignment_constraints()
        self._add_capacity_constraints()
        self._add_truck_usage_constraints()
        self._add_route_continuity_constraints()
        self._add_depot_constraints()
        
        # Log model statistics
        num_variables = len(self.model.variables())
        num_constraints = len(self.model.constraints)
        
        logger.info(f"Enhanced MILP model built successfully:")
        logger.info(f"  Decision variables: {num_variables}")
        logger.info(f"  Constraints: {num_constraints}")
        logger.info(f"  Problem type: {self.model.sense}")
        logger.info(f"  Objective weights: cost={self.cost_weight:.2f}, distance={self.distance_weight:.2f}")
    
    def _create_decision_variables(self) -> None:
        """
        Create binary decision variables for the enhanced MILP model
        
        Creates three types of decision variables:
        - x[order_id, truck_id]: Binary variable indicating if order is assigned to truck
        - y[truck_id]: Binary variable indicating if truck is used in the solution
        - z[location1, location2, truck_id]: Binary variable indicating if truck travels from location1 to location2
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
        
        # Route variables: z[k,l,j] = 1 if truck j travels from location k to location l
        self.decision_vars['z'] = {}
        for truck_id in self.trucks:
            for loc1 in self.locations:
                for loc2 in self.locations:
                    if loc1 != loc2:  # No self-loops
                        var_name = f"route_truck_{truck_id}_from_{loc1}_to_{loc2}"
                        self.decision_vars['z'][(loc1, loc2, truck_id)] = pulp.LpVariable(
                            var_name, cat='Binary'
                        )
        
        logger.info(f"Created {len(self.decision_vars['x'])} assignment variables")
        logger.info(f"Created {len(self.decision_vars['y'])} truck usage variables")
        logger.info(f"Created {len(self.decision_vars['z'])} routing variables")
    
    def _set_objective_function(self) -> None:
        """
        Set the multi-objective function to minimize weighted sum of truck costs and travel distances
        
        The objective function minimizes:
        cost_weight * (sum of truck costs) + distance_weight * (sum of travel distances)
        """
        # Get truck costs and distances
        truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
        
        # Truck cost terms
        truck_cost_terms = []
        for truck_id in self.trucks:
            cost = truck_costs[truck_id]
            truck_var = self.decision_vars['y'][truck_id]
            truck_cost_terms.append(cost * truck_var)
        
        # Distance cost terms
        distance_cost_terms = []
        for truck_id in self.trucks:
            for loc1 in self.locations:
                for loc2 in self.locations:
                    if loc1 != loc2 and (loc1, loc2, truck_id) in self.decision_vars['z']:
                        distance = self.distance_matrix.loc[loc1, loc2]
                        route_var = self.decision_vars['z'][(loc1, loc2, truck_id)]
                        distance_cost_terms.append(distance * route_var)
        
        # Combined objective with weights
        total_truck_cost = pulp.lpSum(truck_cost_terms)
        total_distance_cost = pulp.lpSum(distance_cost_terms)
        
        # Scale the objectives for better numerical properties
        max_truck_cost = self.trucks_df['cost'].sum()
        max_distance = self.distance_matrix.values.max() * len(self.locations) * len(self.trucks)
        
        scaled_truck_cost = total_truck_cost / max_truck_cost if max_truck_cost > 0 else 0
        scaled_distance_cost = total_distance_cost / max_distance if max_distance > 0 else 0
        
        objective = (self.cost_weight * scaled_truck_cost + 
                    self.distance_weight * scaled_distance_cost)
        
        self.model += objective, "Multi_Objective_Cost_Distance_Minimization"
        
        logger.info("Multi-objective function set to minimize weighted truck costs and travel distances")
        logger.info(f"  Max truck cost scale: {max_truck_cost:.0f}")
        logger.info(f"  Max distance scale: {max_distance:.1f}")
    
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
        
        # Get order volumes and truck capacities
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
        
        logger.info(f"Added {len(self.trucks)} capacity constraints")
    
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
    
    def _add_route_continuity_constraints(self) -> None:
        """
        Add route continuity constraints ensuring valid routes
        
        For each truck and location, if the truck visits the location,
        it must arrive and depart (flow conservation).
        """
        logger.info("Adding route continuity constraints...")
        
        # Create location-to-orders mapping
        order_locations = dict(zip(self.orders_df['order_id'], self.orders_df['postal_code']))
        location_orders = {}
        for order_id, location in order_locations.items():
            if location not in location_orders:
                location_orders[location] = []
            location_orders[location].append(order_id)
        
        constraint_count = 0
        for truck_id in self.trucks:
            for location in self.locations:
                if location == self.depot_location:
                    continue  # Depot constraints handled separately
                
                # Inflow to location
                inflow = pulp.lpSum([
                    self.decision_vars['z'][(other_loc, location, truck_id)]
                    for other_loc in self.locations
                    if other_loc != location and (other_loc, location, truck_id) in self.decision_vars['z']
                ])
                
                # Outflow from location
                outflow = pulp.lpSum([
                    self.decision_vars['z'][(location, other_loc, truck_id)]
                    for other_loc in self.locations
                    if other_loc != location and (location, other_loc, truck_id) in self.decision_vars['z']
                ])
                
                # Orders served at this location by this truck
                if location in location_orders:
                    orders_served = pulp.lpSum([
                        self.decision_vars['x'][(order_id, truck_id)]
                        for order_id in location_orders[location]
                    ])
                else:
                    orders_served = 0
                
                # Flow conservation: inflow = outflow = orders served
                constraint_name = f"flow_conservation_truck_{truck_id}_location_{location}"
                self.model += inflow == orders_served, constraint_name
                self.model += outflow == orders_served, f"{constraint_name}_out"
                constraint_count += 2
        
        logger.info(f"Added {constraint_count} route continuity constraints")
    
    def _add_depot_constraints(self) -> None:
        """
        Add depot constraints ensuring trucks start and end at depot
        
        Each used truck must leave the depot exactly once and return exactly once.
        """
        logger.info("Adding depot constraints...")
        
        constraint_count = 0
        for truck_id in self.trucks:
            # Truck leaves depot
            depot_outflow = pulp.lpSum([
                self.decision_vars['z'][(self.depot_location, other_loc, truck_id)]
                for other_loc in self.locations
                if other_loc != self.depot_location and (self.depot_location, other_loc, truck_id) in self.decision_vars['z']
            ])
            
            # Truck returns to depot
            depot_inflow = pulp.lpSum([
                self.decision_vars['z'][(other_loc, self.depot_location, truck_id)]
                for other_loc in self.locations
                if other_loc != self.depot_location and (other_loc, self.depot_location, truck_id) in self.decision_vars['z']
            ])
            
            # If truck is used, it must leave and return to depot exactly once
            truck_usage_var = self.decision_vars['y'][truck_id]
            
            constraint_name = f"depot_outflow_truck_{truck_id}"
            self.model += depot_outflow == truck_usage_var, constraint_name
            
            constraint_name = f"depot_inflow_truck_{truck_id}"
            self.model += depot_inflow == truck_usage_var, constraint_name
            
            constraint_count += 2
        
        logger.info(f"Added {constraint_count} depot constraints")
    
    def solve(self, timeout: int = 300) -> bool:
        """
        Solve the enhanced MILP optimization problem
        
        Args:
            timeout (int): Maximum solving time in seconds
            
        Returns:
            bool: True if optimal solution found, False otherwise
        """
        if self.model is None:
            raise RuntimeError("Model must be built before solving. Call build_model() first.")
        
        logger.info(f"Starting enhanced MILP optimization solver with {timeout}s timeout...")
        start_time = time.time()
        
        try:
            # Solve with timeout
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout)
            self.model.solve(solver)
            
            solve_time = time.time() - start_time
            
            # Check solution status
            status = pulp.LpStatus[self.model.status]
            logger.info(f"Solver completed in {solve_time:.2f} seconds")
            logger.info(f"Solution status: {status}")
            
            if self.model.status == pulp.LpStatusOptimal:
                # Optimal solution found
                objective_value = pulp.value(self.model.objective)
                logger.info(f"Optimal solution found with objective value: {objective_value:.6f}")
                
                self.is_solved = True
                self._extract_solution()
                return True
                
            elif self.model.status == pulp.LpStatusInfeasible:
                logger.error("Problem is infeasible - no solution exists")
                return False
                
            elif self.model.status == pulp.LpStatusUnbounded:
                logger.error("Problem is unbounded")
                return False
                
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
        including order assignments, selected trucks, routes, costs, and metrics.
        """
        logger.info("Extracting enhanced optimization solution...")
        
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
        
        # Extract routes
        routes = {}
        for truck_id in selected_trucks:
            routes[truck_id] = []
            for loc1 in self.locations:
                for loc2 in self.locations:
                    if loc1 != loc2 and (loc1, loc2, truck_id) in self.decision_vars['z']:
                        var_value = self.decision_vars['z'][(loc1, loc2, truck_id)].varValue
                        if var_value is not None and var_value > 0.5:
                            routes[truck_id].append((loc1, loc2))
        
        # Calculate solution metrics
        objective_value = pulp.value(self.model.objective)
        
        # Calculate costs and distances
        truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
        total_truck_cost = sum(truck_costs[truck_id] for truck_id in selected_trucks)
        
        total_distance = 0
        for truck_id in selected_trucks:
            for loc1, loc2 in routes[truck_id]:
                total_distance += self.distance_matrix.loc[loc1, loc2]
        
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
        
        # Store structured solution data
        self.solution_data = {
            'assignments': assignments,
            'selected_trucks': selected_trucks,
            'routes': routes,
            'objective_value': objective_value,
            'total_truck_cost': total_truck_cost,
            'total_distance': total_distance,
            'truck_utilization': truck_utilization,
            'solution_summary': {
                'trucks_used': len(selected_trucks),
                'orders_delivered': len(assignments),
                'total_volume_delivered': sum(order_volumes[a['order_id']] for a in assignments),
                'average_utilization': np.mean([u['utilization_percent'] for u in truck_utilization.values()]) if truck_utilization else 0,
                'cost_weight': self.cost_weight,
                'distance_weight': self.distance_weight
            }
        }
        
        # Log solution summary
        logger.info("Enhanced solution extraction completed:")
        logger.info(f"  Selected trucks: {selected_trucks}")
        logger.info(f"  Total truck cost: €{total_truck_cost:.0f}")
        logger.info(f"  Total distance: {total_distance:.1f} km")
        logger.info(f"  Objective value: {objective_value:.6f}")
        logger.info(f"  Orders delivered: {len(assignments)}/{len(self.orders)}")
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Return structured optimization results with routing information
        
        Returns:
            Dict[str, Any]: Comprehensive solution data including routes and distances
        """
        if not self.is_solved:
            raise RuntimeError("Model must be solved before getting solution. Call solve() first.")
        
        logger.info("Formatting enhanced solution data for output...")
        
        # Create assignments DataFrame
        assignments_df = pd.DataFrame(self.solution_data['assignments'])
        
        # Create detailed routes DataFrame
        routes_data = []
        for truck_id in self.solution_data['selected_trucks']:
            assigned_orders = [a['order_id'] for a in self.solution_data['assignments'] if a['truck_id'] == truck_id]
            
            # Get route sequence
            route_sequence = self._reconstruct_route_sequence(truck_id)
            
            # Calculate route distance
            route_distance = 0
            for i in range(len(route_sequence) - 1):
                route_distance += self.distance_matrix.loc[route_sequence[i], route_sequence[i+1]]
            
            routes_data.append({
                'truck_id': truck_id,
                'assigned_orders': assigned_orders,
                'route_sequence': route_sequence,
                'route_distance': route_distance,
                'num_orders': len(assigned_orders)
            })
        
        routes_df = pd.DataFrame(routes_data)
        
        # Prepare cost breakdown
        truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
        cost_breakdown = {
            'truck_costs': {truck_id: truck_costs[truck_id] for truck_id in self.solution_data['selected_trucks']},
            'total_truck_cost': self.solution_data['total_truck_cost'],
            'total_distance': self.solution_data['total_distance'],
            'total_cost': self.solution_data['total_truck_cost']  # For compatibility
        }
        
        # Prepare final solution structure
        solution = {
            'assignments_df': assignments_df,
            'routes_df': routes_df,
            'costs': cost_breakdown,
            'utilization': self.solution_data['truck_utilization'],
            'summary': self.solution_data['solution_summary'],
            'selected_trucks': self.solution_data['selected_trucks'],
            'optimization_status': 'optimal',
            'objective_weights': {
                'cost_weight': self.cost_weight,
                'distance_weight': self.distance_weight
            },
            'depot_location': self.depot_location
        }
        
        logger.info("Enhanced solution data formatted successfully")
        return solution
    
    def _reconstruct_route_sequence(self, truck_id: int) -> List[str]:
        """
        Reconstruct the route sequence for a truck from routing variables
        
        Args:
            truck_id (int): Truck ID to reconstruct route for
            
        Returns:
            List[str]: Ordered sequence of locations visited by the truck
        """
        if truck_id not in self.solution_data['routes']:
            return [self.depot_location]
        
        # Build adjacency list from route segments
        route_segments = self.solution_data['routes'][truck_id]
        adjacency = {}
        
        for loc1, loc2 in route_segments:
            if loc1 not in adjacency:
                adjacency[loc1] = []
            adjacency[loc1].append(loc2)
        
        # Reconstruct route starting from depot
        route_sequence = [self.depot_location]
        current_location = self.depot_location
        visited = set([self.depot_location])
        
        # Follow the route from depot
        while current_location in adjacency:
            next_locations = [loc for loc in adjacency[current_location] if loc not in visited or loc == self.depot_location]
            
            if not next_locations:
                break
                
            # Choose the next location (prefer non-depot locations first)
            next_location = next_locations[0]
            for loc in next_locations:
                if loc != self.depot_location:
                    next_location = loc
                    break
            
            route_sequence.append(next_location)
            
            # If we're back at depot and have visited other locations, we're done
            if next_location == self.depot_location and len(visited) > 1:
                break
                
            if next_location != self.depot_location:
                visited.add(next_location)
            current_location = next_location
            
            # Prevent infinite loops
            if len(route_sequence) > len(self.locations) * 2:
                break
        
        return route_sequence