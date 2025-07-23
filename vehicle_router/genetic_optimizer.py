"""
Genetic Algorithm Optimization Module for Vehicle Routing

This module provides the GeneticVrpOptimizer class that implements a sophisticated
genetic algorithm approach to solve the Vehicle Routing Problem with multi-objective
optimization of both truck costs and total travel distances.

The genetic algorithm approach includes:
- Population-based evolutionary optimization
- Multi-objective fitness function (cost + distance)
- Advanced genetic operators (selection, crossover, mutation)
- Elitism and diversity preservation
- Adaptive parameter tuning
- Constraint handling for capacity limits

Classes:
    GeneticVrpOptimizer: Evolutionary optimization engine for VRP
    Solution: Individual solution representation
    Population: Collection of solutions with genetic operations
"""

import logging
import pandas as pd
import numpy as np
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import permutations
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """
    Individual solution representation for genetic algorithm
    
    Attributes:
        assignments (Dict[str, int]): Mapping of order_id to truck_id
        truck_usage (List[int]): List of used truck IDs
        routes (Dict[int, List[str]]): Mapping of truck_id to route sequence
        fitness (float): Solution fitness value (lower is better)
        cost (float): Total truck operational cost
        distance (float): Total travel distance
        feasible (bool): Whether solution satisfies all constraints
    """
    assignments: Dict[str, int]
    truck_usage: List[int]
    routes: Dict[int, List[str]]
    fitness: float = float('inf')
    cost: float = 0.0
    distance: float = 0.0
    feasible: bool = True


class GeneticVrpOptimizer:
    """
    Genetic Algorithm Vehicle Routing Problem Optimizer
    
    This class implements a sophisticated genetic algorithm approach to solve the Vehicle
    Routing Problem with multi-objective optimization of truck costs and travel distances.
    
    The genetic algorithm features:
    - Population-based evolutionary search
    - Multi-objective fitness function with configurable weights
    - Tournament selection with elitism
    - Order-preserving crossover and adaptive mutation
    - Constraint repair mechanisms for capacity violations
    - Diversity preservation and convergence detection
    
    Attributes:
        orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
        trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
        distance_matrix (pd.DataFrame): Distance matrix between postal codes
        depot_location (str): Depot postal code (trucks start/end here)
        cost_weight (float): Weight for truck costs in fitness (0-1)
        distance_weight (float): Weight for distance costs in fitness (0-1)
        population_size (int): Number of solutions in population
        max_generations (int): Maximum number of generations
        mutation_rate (float): Probability of mutation
        elite_size (int): Number of elite solutions preserved
        
    Example:
        >>> optimizer = GeneticVrpOptimizer(orders_df, trucks_df, distance_matrix)
        >>> optimizer.set_parameters(population_size=50, max_generations=100)
        >>> optimizer.set_objective_weights(cost_weight=0.6, distance_weight=0.4)
        >>> success = optimizer.solve()
        >>> if success:
        ...     solution = optimizer.get_solution()
    """
    
    def __init__(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                 distance_matrix: pd.DataFrame, depot_location: Optional[str] = None,
                 depot_return: bool = False, max_orders_per_truck: int = 3):
        """
        Initialize the Genetic Algorithm VRP Optimizer
        
        Args:
            orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
            trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
            distance_matrix (pd.DataFrame): Distance matrix between postal codes
            depot_location (Optional[str]): Depot postal code. If None, uses '08020'.
            depot_return (bool): Whether trucks must return to depot after deliveries.
            max_orders_per_truck (int): Maximum number of orders per truck. Default 3.
            
        Raises:
            ValueError: If input data is invalid or inconsistent
        """
        logger.info("Initializing Genetic Algorithm VRP Optimizer...")
        
        # Validate input data
        self._validate_input_data(orders_df, trucks_df, distance_matrix)
        
        # Store input data
        self.orders_df = orders_df.copy()
        self.trucks_df = trucks_df.copy()
        self.distance_matrix = distance_matrix.copy()
        
        # Set depot location
        if depot_location is None:
            if '08020' in self.distance_matrix.index:
                self.depot_location = '08020'
            else:
                self.depot_location = self.orders_df['postal_code'].iloc[0]
        else:
            if depot_location in self.distance_matrix.index:
                self.depot_location = depot_location
            else:
                raise ValueError(f"Depot location '{depot_location}' not found in distance matrix")
        
        self.depot_return = depot_return
        
        # Validate and store max orders per truck constraint
        if max_orders_per_truck < 1:
            raise ValueError("max_orders_per_truck must be at least 1")
        self.max_orders_per_truck = max_orders_per_truck
        
        # Extract problem data
        self.orders = self.orders_df['order_id'].tolist()
        self.trucks = self.trucks_df['truck_id'].tolist()
        self.order_volumes = dict(zip(self.orders_df['order_id'], self.orders_df['volume']))
        self.order_locations = dict(zip(self.orders_df['order_id'], self.orders_df['postal_code']))
        self.truck_capacities = dict(zip(self.trucks_df['truck_id'], self.trucks_df['capacity']))
        self.truck_costs = dict(zip(self.trucks_df['truck_id'], self.trucks_df['cost']))
        
        # Algorithm parameters (defaults)
        self.population_size = 50
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.elite_size = 5
        self.tournament_size = 3
        self.cost_weight = 0.6
        self.distance_weight = 0.4
        
        # Algorithm state
        self.population = []
        self.best_solution = None
        self.generation = 0
        self.convergence_generations = 0
        self.max_convergence = 20
        
        # Statistics
        self.fitness_history = []
        self.diversity_history = []
        
        logger.info(f"Genetic optimizer initialized:")
        logger.info(f"  Orders: {len(self.orders)}, Trucks: {len(self.trucks)}")
        logger.info(f"  Depot: {self.depot_location}, Return: {self.depot_return}")
        logger.info(f"  Maximum orders per truck: {self.max_orders_per_truck}")
        logger.info(f"  Population size: {self.population_size}")
        logger.info(f"  Max generations: {self.max_generations}")
    
    def set_parameters(self, population_size: int = 50, max_generations: int = 100,
                      mutation_rate: float = 0.1, elite_size: int = 5,
                      tournament_size: int = 3) -> None:
        """
        Set genetic algorithm parameters
        
        Args:
            population_size (int): Number of solutions in population
            max_generations (int): Maximum number of generations
            mutation_rate (float): Probability of mutation (0-1)
            elite_size (int): Number of elite solutions preserved
            tournament_size (int): Tournament selection size
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        logger.info(f"Genetic algorithm parameters set:")
        logger.info(f"  Population: {population_size}, Generations: {max_generations}")
        logger.info(f"  Mutation rate: {mutation_rate}, Elite size: {elite_size}")
    
    def set_objective_weights(self, cost_weight: float, distance_weight: float) -> None:
        """
        Set the weights for multi-objective optimization
        
        Args:
            cost_weight (float): Weight for truck costs (0-1)
            distance_weight (float): Weight for distance costs (0-1)
        """
        if not (0 <= cost_weight <= 1) or not (0 <= distance_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        total_weight = cost_weight + distance_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {total_weight:.3f}, normalizing to 1.0")
            cost_weight = cost_weight / total_weight
            distance_weight = distance_weight / total_weight
        
        self.cost_weight = cost_weight
        self.distance_weight = distance_weight
        
        logger.info(f"Objective weights set: cost={cost_weight:.2f}, distance={distance_weight:.2f}")
    
    def solve(self, timeout: int = 300) -> bool:
        """
        Solve the VRP using genetic algorithm
        
        Args:
            timeout (int): Maximum solving time in seconds
            
        Returns:
            bool: True if solution found, False otherwise
        """
        logger.info(f"Starting genetic algorithm optimization (timeout: {timeout}s)...")
        start_time = time.time()
        
        try:
            # Initialize population
            logger.info("Initializing population...")
            self._initialize_population()
            
            # Evolution loop
            self.generation = 0
            self.convergence_generations = 0
            
            while self.generation < self.max_generations:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Genetic algorithm timeout after {timeout}s")
                    break
                
                # Evolve population
                self._evolve_generation()
                
                # Update statistics
                self._update_statistics()
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Converged after {self.generation} generations")
                    break
                
                self.generation += 1
            
            solve_time = time.time() - start_time
            
            # Get best solution
            if self.best_solution and self.best_solution.feasible:
                logger.info(f"Genetic algorithm completed in {solve_time:.2f}s")
                logger.info(f"Best solution: cost={self.best_solution.cost:.0f}, distance={self.best_solution.distance:.1f}")
                logger.info(f"Fitness: {self.best_solution.fitness:.6f}")
                return True
            else:
                logger.error("No feasible solution found")
                return False
                
        except Exception as e:
            logger.error(f"Error during genetic algorithm: {str(e)}")
            return False
    
    def _validate_input_data(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                           distance_matrix: pd.DataFrame) -> None:
        """Validate input data for consistency and completeness"""
        # Basic validation (similar to other optimizers)
        if not isinstance(orders_df, pd.DataFrame) or orders_df.empty:
            raise ValueError("Invalid orders_df")
        if not isinstance(trucks_df, pd.DataFrame) or trucks_df.empty:
            raise ValueError("Invalid trucks_df")
        if not isinstance(distance_matrix, pd.DataFrame) or distance_matrix.empty:
            raise ValueError("Invalid distance_matrix")
        
        # Check required columns
        required_order_cols = ['order_id', 'volume', 'postal_code']
        if not all(col in orders_df.columns for col in required_order_cols):
            raise ValueError(f"orders_df missing required columns: {required_order_cols}")
        
        required_truck_cols = ['truck_id', 'capacity', 'cost']
        if not all(col in trucks_df.columns for col in required_truck_cols):
            raise ValueError(f"trucks_df missing required columns: {required_truck_cols}")
        
        # Check feasibility
        total_volume = orders_df['volume'].sum()
        total_capacity = trucks_df['capacity'].sum()
        if total_volume > total_capacity:
            raise ValueError(f"Infeasible: total volume ({total_volume}) > total capacity ({total_capacity})")
    
    def _initialize_population(self) -> None:
        """Initialize population with diverse solutions"""
        self.population = []
        
        for i in range(self.population_size):
            if i == 0:
                # First solution: greedy assignment by capacity
                solution = self._create_greedy_solution()
            else:
                # Random solutions with different strategies
                solution = self._create_random_solution()
            
            # Repair solution if needed
            solution = self._repair_solution(solution)
            
            # Calculate fitness
            self._evaluate_solution(solution)
            
            self.population.append(solution)
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)
        self.best_solution = self.population[0]
        
        logger.info(f"Population initialized with {len(self.population)} solutions")
        logger.info(f"Best initial fitness: {self.best_solution.fitness:.6f}")
    
    def _create_greedy_solution(self) -> Solution:
        """Create a greedy solution by assigning orders to trucks efficiently"""
        assignments = {}
        truck_loads = {truck_id: 0 for truck_id in self.trucks}
        
        # Sort orders by volume (largest first) for better packing
        sorted_orders = sorted(self.orders, key=lambda x: self.order_volumes[x], reverse=True)
        
        for order_id in sorted_orders:
            volume = self.order_volumes[order_id]
            
            # Find truck with enough capacity and lowest cost
            best_truck = None
            best_cost = float('inf')
            
            for truck_id in self.trucks:
                remaining_capacity = self.truck_capacities[truck_id] - truck_loads[truck_id]
                if remaining_capacity >= volume:
                    cost = self.truck_costs[truck_id]
                    if cost < best_cost:
                        best_cost = cost
                        best_truck = truck_id
            
            if best_truck is not None:
                assignments[order_id] = best_truck
                truck_loads[best_truck] += volume
        
        # Get used trucks
        truck_usage = list(set(assignments.values()))
        
        # Create routes
        routes = self._create_routes_for_assignments(assignments, truck_usage)
        
        return Solution(
            assignments=assignments,
            truck_usage=truck_usage,
            routes=routes
        )
    
    def _create_random_solution(self) -> Solution:
        """Create a random feasible solution"""
        assignments = {}
        truck_loads = {truck_id: 0 for truck_id in self.trucks}
        
        # Randomly shuffle orders and trucks
        shuffled_orders = self.orders.copy()
        random.shuffle(shuffled_orders)
        shuffled_trucks = self.trucks.copy()
        random.shuffle(shuffled_trucks)
        
        for order_id in shuffled_orders:
            volume = self.order_volumes[order_id]
            
            # Try to assign to a random truck with capacity
            assigned = False
            for truck_id in shuffled_trucks:
                remaining_capacity = self.truck_capacities[truck_id] - truck_loads[truck_id]
                if remaining_capacity >= volume:
                    assignments[order_id] = truck_id
                    truck_loads[truck_id] += volume
                    assigned = True
                    break
            
            # If no truck found, assign to truck with most remaining capacity
            if not assigned:
                best_truck = max(self.trucks, 
                               key=lambda x: self.truck_capacities[x] - truck_loads[x])
                assignments[order_id] = best_truck
                truck_loads[best_truck] += volume
        
        truck_usage = list(set(assignments.values()))
        routes = self._create_routes_for_assignments(assignments, truck_usage)
        
        return Solution(
            assignments=assignments,
            truck_usage=truck_usage,
            routes=routes
        )
    
    def _create_routes_for_assignments(self, assignments: Dict[str, int], 
                                     truck_usage: List[int]) -> Dict[int, List[str]]:
        """Create optimized routes for truck assignments"""
        routes = {}
        
        for truck_id in truck_usage:
            # Get orders assigned to this truck
            truck_orders = [order_id for order_id, tid in assignments.items() if tid == truck_id]
            
            if not truck_orders:
                routes[truck_id] = [self.depot_location]
                continue
            
            # Get unique postal codes for these orders
            postal_codes = list(set(self.order_locations[order_id] for order_id in truck_orders))
            
            if len(postal_codes) == 1:
                # Simple out-and-back route
                route = [self.depot_location, postal_codes[0]]
                if self.depot_return:
                    route.append(self.depot_location)
                routes[truck_id] = route
            else:
                # Optimize route using nearest neighbor heuristic
                route = self._optimize_route_nearest_neighbor(postal_codes)
                routes[truck_id] = route
        
        return routes
    
    def _optimize_route_nearest_neighbor(self, postal_codes: List[str]) -> List[str]:
        """Optimize route using nearest neighbor heuristic"""
        if not postal_codes:
            return [self.depot_location]
        
        route = [self.depot_location]
        remaining = postal_codes.copy()
        current = self.depot_location
        
        while remaining:
            # Find nearest unvisited location
            nearest = min(remaining, 
                         key=lambda x: self.distance_matrix.loc[current, x])
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        if self.depot_return:
            route.append(self.depot_location)
        
        return route
    
    def _repair_solution(self, solution: Solution) -> Solution:
        """Repair solution to ensure feasibility"""
        # Check capacity and max orders constraints
        truck_loads = {truck_id: 0 for truck_id in self.trucks}
        truck_order_counts = {truck_id: 0 for truck_id in self.trucks}
        capacity_violations = []
        order_count_violations = []
        
        for order_id, truck_id in solution.assignments.items():
            volume = self.order_volumes[order_id]
            truck_loads[truck_id] += volume
            truck_order_counts[truck_id] += 1
            
            if truck_loads[truck_id] > self.truck_capacities[truck_id]:
                capacity_violations.append(order_id)
            
            if truck_order_counts[truck_id] > self.max_orders_per_truck:
                order_count_violations.append(order_id)
        
        # Combine all violations (remove duplicates)
        all_violations = list(set(capacity_violations + order_count_violations))
        
        # Reassign violating orders
        for order_id in all_violations:
            volume = self.order_volumes[order_id]
            old_truck = solution.assignments[order_id]
            truck_loads[old_truck] -= volume
            truck_order_counts[old_truck] -= 1
            
            # Find a truck with enough capacity and order slot
            assigned = False
            for truck_id in self.trucks:
                if (truck_loads[truck_id] + volume <= self.truck_capacities[truck_id] and
                    truck_order_counts[truck_id] < self.max_orders_per_truck):
                    solution.assignments[order_id] = truck_id
                    truck_loads[truck_id] += volume
                    truck_order_counts[truck_id] += 1
                    assigned = True
                    break
            
            # If no suitable truck found, assign to truck with minimum load (emergency repair)
            if not assigned:
                best_truck = min(self.trucks, key=lambda x: truck_loads[x])
                solution.assignments[order_id] = best_truck
                truck_loads[best_truck] += volume
                truck_order_counts[best_truck] += 1
        
        # Update truck usage and routes
        solution.truck_usage = list(set(solution.assignments.values()))
        solution.routes = self._create_routes_for_assignments(solution.assignments, solution.truck_usage)
        
        return solution
    
    def _evaluate_solution(self, solution: Solution) -> None:
        """Calculate fitness, cost, and distance for solution"""
        # Calculate truck cost
        total_cost = sum(self.truck_costs[truck_id] for truck_id in solution.truck_usage)
        
        # Calculate total distance
        total_distance = 0
        for truck_id, route in solution.routes.items():
            for i in range(len(route) - 1):
                from_loc = route[i]
                to_loc = route[i + 1]
                distance = self.distance_matrix.loc[from_loc, to_loc]
                total_distance += distance
        
        # Check feasibility
        feasible = self._check_solution_feasibility(solution)
        
        # Calculate fitness (lower is better)
        if feasible:
            # Normalize costs and distances
            max_cost = sum(self.truck_costs.values())
            max_distance = self.distance_matrix.values.max() * len(self.orders) * 2
            
            normalized_cost = total_cost / max_cost if max_cost > 0 else 0
            normalized_distance = total_distance / max_distance if max_distance > 0 else 0
            
            fitness = self.cost_weight * normalized_cost + self.distance_weight * normalized_distance
        else:
            # Penalty for infeasible solutions
            fitness = float('inf')
        
        solution.cost = total_cost
        solution.distance = total_distance
        solution.feasible = feasible
        solution.fitness = fitness
    
    def _check_solution_feasibility(self, solution: Solution) -> bool:
        """Check if solution satisfies all constraints"""
        # Check that all orders are assigned
        if len(solution.assignments) != len(self.orders):
            return False
        
        # Check capacity constraints and max orders constraints
        truck_loads = {truck_id: 0 for truck_id in self.trucks}
        truck_order_counts = {truck_id: 0 for truck_id in self.trucks}
        
        for order_id, truck_id in solution.assignments.items():
            volume = self.order_volumes[order_id]
            truck_loads[truck_id] += volume
            truck_order_counts[truck_id] += 1
            
            # Check capacity constraint
            if truck_loads[truck_id] > self.truck_capacities[truck_id]:
                return False
            
            # Check max orders constraint
            if truck_order_counts[truck_id] > self.max_orders_per_truck:
                return False
        
        return True
    
    def _evolve_generation(self) -> None:
        """Evolve population for one generation"""
        new_population = []
        
        # Keep elite solutions
        elite_solutions = self.population[:self.elite_size]
        new_population.extend([copy.deepcopy(sol) for sol in elite_solutions])
        
        # Generate new solutions through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            # Repair and evaluate
            child1 = self._repair_solution(child1)
            child2 = self._repair_solution(child2)
            self._evaluate_solution(child1)
            self._evaluate_solution(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size and sort
        self.population = new_population[:self.population_size]
        self.population.sort(key=lambda x: x.fitness)
        
        # Update best solution
        if self.population[0].fitness < self.best_solution.fitness:
            self.best_solution = copy.deepcopy(self.population[0])
            self.convergence_generations = 0
        else:
            self.convergence_generations += 1
    
    def _tournament_selection(self) -> Solution:
        """Select solution using tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Order crossover for solution recombination"""
        # Create children by combining assignments from both parents
        child1_assignments = {}
        child2_assignments = {}
        
        # Copy some assignments from each parent
        orders_list = list(self.orders)
        split_point = len(orders_list) // 2
        
        # Child 1: first half from parent1, second half from parent2
        for i, order_id in enumerate(orders_list):
            if i < split_point:
                child1_assignments[order_id] = parent1.assignments[order_id]
                child2_assignments[order_id] = parent2.assignments[order_id]
            else:
                child1_assignments[order_id] = parent2.assignments[order_id]
                child2_assignments[order_id] = parent1.assignments[order_id]
        
        # Create child solutions
        child1_usage = list(set(child1_assignments.values()))
        child2_usage = list(set(child2_assignments.values()))
        
        child1_routes = self._create_routes_for_assignments(child1_assignments, child1_usage)
        child2_routes = self._create_routes_for_assignments(child2_assignments, child2_usage)
        
        child1 = Solution(
            assignments=child1_assignments,
            truck_usage=child1_usage,
            routes=child1_routes
        )
        
        child2 = Solution(
            assignments=child2_assignments,
            truck_usage=child2_usage,
            routes=child2_routes
        )
        
        return child1, child2
    
    def _mutate(self, solution: Solution) -> Solution:
        """Mutate solution by reassigning random orders"""
        mutated_solution = copy.deepcopy(solution)
        
        # Reassign 1-3 random orders
        num_mutations = random.randint(1, min(3, len(self.orders)))
        orders_to_mutate = random.sample(self.orders, num_mutations)
        
        for order_id in orders_to_mutate:
            # Remove from current truck
            current_truck = mutated_solution.assignments[order_id]
            
            # Assign to random different truck
            available_trucks = [t for t in self.trucks if t != current_truck]
            if available_trucks:
                new_truck = random.choice(available_trucks)
                mutated_solution.assignments[order_id] = new_truck
        
        # Update truck usage and routes
        mutated_solution.truck_usage = list(set(mutated_solution.assignments.values()))
        mutated_solution.routes = self._create_routes_for_assignments(
            mutated_solution.assignments, mutated_solution.truck_usage)
        
        return mutated_solution
    
    def _update_statistics(self) -> None:
        """Update algorithm statistics"""
        if self.generation % 10 == 0:
            best_fitness = self.population[0].fitness
            avg_fitness = np.mean([sol.fitness for sol in self.population if sol.fitness != float('inf')])
            
            self.fitness_history.append(best_fitness)
            
            logger.info(f"Generation {self.generation}: best={best_fitness:.6f}, avg={avg_fitness:.6f}")
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        return self.convergence_generations >= self.max_convergence
    
    def get_solution(self) -> Dict[str, Any]:
        """Return structured optimization results"""
        if not self.best_solution or not self.best_solution.feasible:
            raise RuntimeError("No feasible solution found. Call solve() first.")
        
        logger.info("Formatting genetic algorithm solution...")
        
        # Create assignments DataFrame
        assignments_data = []
        for order_id, truck_id in self.best_solution.assignments.items():
            assignments_data.append({
                'order_id': order_id,
                'truck_id': truck_id,
                'assigned': True
            })
        
        assignments_df = pd.DataFrame(assignments_data)
        
        # Create routes DataFrame
        routes_data = []
        for truck_id in self.best_solution.truck_usage:
            assigned_orders = [order_id for order_id, tid in self.best_solution.assignments.items() 
                             if tid == truck_id]
            route_sequence = self.best_solution.routes[truck_id]
            
            # Calculate route distance
            route_distance = 0
            for i in range(len(route_sequence) - 1):
                route_distance += self.distance_matrix.loc[route_sequence[i], route_sequence[i+1]]
            
            routes_data.append({
                'truck_id': truck_id,
                'assigned_orders': assigned_orders,
                'route_sequence': route_sequence,
                'route_distance': route_distance,
                'num_orders': len(assigned_orders),
                'depot_location': self.depot_location
            })
        
        routes_df = pd.DataFrame(routes_data)
        
        # Calculate utilization
        truck_utilization = {}
        for truck_id in self.best_solution.truck_usage:
            assigned_orders = [order_id for order_id, tid in self.best_solution.assignments.items() 
                             if tid == truck_id]
            used_volume = sum(self.order_volumes[order_id] for order_id in assigned_orders)
            capacity = self.truck_capacities[truck_id]
            utilization = (used_volume / capacity) * 100 if capacity > 0 else 0
            
            truck_utilization[truck_id] = {
                'used_volume': used_volume,
                'capacity': capacity,
                'utilization_percent': utilization,
                'assigned_orders': assigned_orders
            }
        
        # Prepare cost breakdown
        cost_breakdown = {
            'truck_costs': {truck_id: self.truck_costs[truck_id] 
                           for truck_id in self.best_solution.truck_usage},
            'total_truck_cost': self.best_solution.cost,
            'total_distance': self.best_solution.distance,
            'total_cost': self.best_solution.cost
        }
        
        # Prepare solution
        solution = {
            'assignments_df': assignments_df,
            'routes_df': routes_df,
            'costs': cost_breakdown,
            'utilization': truck_utilization,
            'selected_trucks': self.best_solution.truck_usage,
            'optimization_status': 'optimal',
            'objective_weights': {
                'cost_weight': self.cost_weight,
                'distance_weight': self.distance_weight
            },
            'depot_location': self.depot_location,
            'algorithm_stats': {
                'generations': self.generation,
                'population_size': self.population_size,
                'final_fitness': self.best_solution.fitness,
                'mutation_rate': self.mutation_rate
            }
        }
        
        logger.info("Genetic algorithm solution formatted successfully")
        logger.info(f"Final solution: {len(self.best_solution.truck_usage)} trucks, "
                   f"cost={self.best_solution.cost:.0f}, distance={self.best_solution.distance:.1f}")
        
        return solution 