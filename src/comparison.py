#!/usr/bin/env python3
"""
Vehicle Router Optimization Methods Comparison Script

This script runs all three optimization methods (Standard MILP + Greedy, Enhanced MILP, 
and Genetic Algorithm) on the same dataset and provides a comprehensive comparison 
of their performance, results, and characteristics.

Usage:
    python src/comparison.py [options]

Options:
    --timeout SECONDS      Solver timeout in seconds (default: 120)
    --depot POSTAL_CODE    Depot location postal code (default: '08020')
    --depot-return         Enable depot return (default: False)
    --cost-weight FLOAT    Cost weight for multi-objective methods (default: 0.6)
    --distance-weight FLOAT Distance weight for multi-objective methods (default: 0.4)
    --ga-population INT    Genetic algorithm population size (default: 50)
    --ga-generations INT   Genetic algorithm max generations (default: 100)
    --ga-mutation FLOAT    Genetic algorithm mutation rate (default: 0.1)

    --real-distances       Use real-world distances instead of simulated
    --quiet                Reduce output verbosity
    --help                 Show this help message

Example:
    python src/comparison.py --timeout 180 --depot-return --real-distances
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from vehicle_router.data_generator import DataGenerator
from vehicle_router.optimizer import VrpOptimizer
from vehicle_router.enhanced_optimizer import EnhancedVrpOptimizer
from vehicle_router.genetic_optimizer import GeneticVrpOptimizer


class OptimizationComparison:
    """
    Comprehensive comparison of all three optimization methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize comparison with configuration"""
        self.config = config
        self.results = {}
        self.start_time = time.time()
        
        # Configure logging
        log_level = logging.WARNING if config.get('quiet', False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='[%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Generate test data
        self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate consistent test data for all methods"""
        self.logger.info("Generating test data...")
        
        data_gen = DataGenerator(use_example_data=True)
        self.orders_df = data_gen.generate_orders()
        self.trucks_df = data_gen.generate_trucks()
        
        # Include depot in distance matrix
        postal_codes = self.orders_df['postal_code'].tolist()
        if self.config['depot_location'] not in postal_codes:
            postal_codes.append(self.config['depot_location'])
        
        self.distance_matrix = data_gen.generate_distance_matrix(
            postal_codes, use_real_distances=self.config.get('use_real_distances', False))
        
        # Log problem characteristics
        total_volume = self.orders_df['volume'].sum()
        total_capacity = self.trucks_df['capacity'].sum()
        
        self.logger.info(f"Test problem characteristics:")
        self.logger.info(f"  Orders: {len(self.orders_df)} (Total volume: {total_volume:.0f} m³)")
        self.logger.info(f"  Trucks: {len(self.trucks_df)} (Total capacity: {total_capacity:.0f} m³)")
        self.logger.info(f"  Locations: {len(self.distance_matrix)} (including depot: {self.config['depot_location']})")
        self.logger.info(f"  Depot return: {self.config['depot_return']}")
        
    def run_standard_milp(self) -> Dict[str, Any]:
        """Run Standard MILP + Greedy optimization"""
        self.logger.info("🔄 Running Standard MILP + Greedy...")
        
        start_time = time.time()
        result = {
            'method': 'Standard MILP + Greedy',
            'status': 'FAILED',
            'execution_time': 0,
            'error': None
        }
        
        try:
            optimizer = VrpOptimizer(
                orders_df=self.orders_df,
                trucks_df=self.trucks_df,
                distance_matrix=self.distance_matrix,
                depot_location=self.config['depot_location'],
                depot_return=self.config['depot_return'],
                enable_greedy_routes=True
            )
            
            optimizer.build_model()
            success = optimizer.solve()
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            
            if success:
                solution = optimizer.get_solution()
                
                # Calculate total distance from routes
                total_distance = solution['routes_df']['route_distance'].sum()
                
                result.update({
                    'status': 'SUCCESS',
                    'selected_trucks': solution['selected_trucks'],
                    'total_cost': solution['costs']['total_cost'],
                    'total_distance': total_distance,
                    'truck_utilization': {
                        truck_id: data['utilization_percent'] 
                        for truck_id, data in solution['utilization'].items()
                    },
                    'routes': solution['routes_df'].to_dict('records'),
                    'solver_status': getattr(optimizer, 'solver_status', 'Unknown'),
                    'num_variables': getattr(optimizer, 'num_variables', 0),
                    'num_constraints': getattr(optimizer, 'num_constraints', 0),
                })
                
                self.logger.info(f"  ✅ Success: {len(solution['selected_trucks'])} trucks, €{solution['costs']['total_cost']:.0f}, {total_distance:.1f} km ({execution_time:.2f}s)")
            else:
                result['error'] = 'Optimization failed'
                self.logger.info(f"  ❌ Failed ({execution_time:.2f}s)")
                
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            self.logger.info(f"  ❌ Error: {str(e)[:50]}...")
            
        return result
    
    def run_enhanced_milp(self) -> Dict[str, Any]:
        """Run Enhanced MILP optimization"""
        self.logger.info("🚀 Running Enhanced MILP...")
        
        start_time = time.time()
        result = {
            'method': 'Enhanced MILP',
            'status': 'FAILED',
            'execution_time': 0,
            'error': None
        }
        
        try:
            optimizer = EnhancedVrpOptimizer(
                orders_df=self.orders_df,
                trucks_df=self.trucks_df,
                distance_matrix=self.distance_matrix,
                depot_location=self.config['depot_location'],
                depot_return=self.config['depot_return']
            )
            
            optimizer.set_objective_weights(
                cost_weight=self.config['cost_weight'],
                distance_weight=self.config['distance_weight']
            )
            
            optimizer.build_model()
            success = optimizer.solve(timeout=self.config['timeout'])
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            
            if success:
                solution = optimizer.get_solution()
                
                result.update({
                    'status': 'SUCCESS',
                    'selected_trucks': solution['selected_trucks'],
                    'total_cost': solution['costs']['total_cost'],
                    'total_distance': solution['costs']['total_distance'],
                    'objective_value': solution['costs'].get('objective_value', 0),
                    'truck_utilization': {
                        truck_id: data['utilization_percent'] 
                        for truck_id, data in solution['utilization'].items()
                    },
                    'routes': solution['routes_df'].to_dict('records'),
                    'solver_status': getattr(optimizer, 'solver_status', 'Unknown'),
                    'num_variables': getattr(optimizer, 'num_variables', 0),
                    'num_constraints': getattr(optimizer, 'num_constraints', 0),
                    'cost_weight': self.config['cost_weight'],
                    'distance_weight': self.config['distance_weight']
                })
                
                self.logger.info(f"  ✅ Success: {len(solution['selected_trucks'])} trucks, €{solution['costs']['total_cost']:.0f}, {solution['costs']['total_distance']:.1f} km ({execution_time:.2f}s)")
            else:
                result['error'] = 'Optimization failed'
                self.logger.info(f"  ❌ Failed ({execution_time:.2f}s)")
                
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            self.logger.info(f"  ❌ Error: {str(e)[:50]}...")
            
        return result
    
    def run_genetic_algorithm(self) -> Dict[str, Any]:
        """Run Genetic Algorithm optimization"""
        self.logger.info("🧬 Running Genetic Algorithm...")
        
        start_time = time.time()
        result = {
            'method': 'Genetic Algorithm',
            'status': 'FAILED',
            'execution_time': 0,
            'error': None
        }
        
        try:
            optimizer = GeneticVrpOptimizer(
                orders_df=self.orders_df,
                trucks_df=self.trucks_df,
                distance_matrix=self.distance_matrix,
                depot_location=self.config['depot_location'],
                depot_return=self.config['depot_return']
            )
            
            optimizer.set_parameters(
                population_size=self.config['ga_population'],
                max_generations=self.config['ga_generations'],
                mutation_rate=self.config['ga_mutation']
            )
            
            optimizer.set_objective_weights(
                cost_weight=self.config['cost_weight'],
                distance_weight=self.config['distance_weight']
            )
            
            success = optimizer.solve(timeout=self.config['timeout'])
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            
            if success:
                solution = optimizer.get_solution()
                
                result.update({
                    'status': 'SUCCESS',
                    'selected_trucks': solution['selected_trucks'],
                    'total_cost': solution['costs']['total_cost'],
                    'total_distance': solution['costs']['total_distance'],
                    'final_fitness': solution['algorithm_stats']['final_fitness'],
                    'generations': solution['algorithm_stats']['generations'],
                    'population_size': solution['algorithm_stats']['population_size'],
                    'mutation_rate': solution['algorithm_stats']['mutation_rate'],
                    'truck_utilization': {
                        truck_id: data['utilization_percent'] 
                        for truck_id, data in solution['utilization'].items()
                    },
                    'routes': solution['routes_df'].to_dict('records'),
                    'cost_weight': self.config['cost_weight'],
                    'distance_weight': self.config['distance_weight']
                })
                
                self.logger.info(f"  ✅ Success: {len(solution['selected_trucks'])} trucks, €{solution['costs']['total_cost']:.0f}, {solution['costs']['total_distance']:.1f} km, {solution['algorithm_stats']['generations']} gen ({execution_time:.2f}s)")
            else:
                result['error'] = 'Optimization failed'
                self.logger.info(f"  ❌ Failed ({execution_time:.2f}s)")
                
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            self.logger.info(f"  ❌ Error: {str(e)[:50]}...")
            
        return result
    
    def run_comparison(self):
        """Run all three optimization methods and compare results"""
        self.logger.info("🏁 Starting comprehensive optimization comparison...")
        self.logger.info("=" * 60)
        
        # Run all three methods
        self.results['standard'] = self.run_standard_milp()
        self.results['enhanced'] = self.run_enhanced_milp()
        self.results['genetic'] = self.run_genetic_algorithm()
        
        # Generate comparison report
        self._generate_comparison_report()
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("🏆 VEHICLE ROUTER OPTIMIZATION METHODS COMPARISON")
        print("=" * 70)
        
        # Configuration summary
        print("\n📊 TEST CONFIGURATION:")
        print(f"  Problem Size: {len(self.orders_df)} orders, {len(self.trucks_df)} trucks")
        print(f"  Depot Location: {self.config['depot_location']}")
        print(f"  Depot Return: {self.config['depot_return']}")
        print(f"  Solver Timeout: {self.config['timeout']}s")
        print(f"  Multi-Objective Weights: {self.config['cost_weight']:.1f} cost, {self.config['distance_weight']:.1f} distance")
        
        # Results table
        print("\n🎯 OPTIMIZATION RESULTS:")
        print("-" * 70)
        
        successful_results = [r for r in self.results.values() if r['status'] == 'SUCCESS']
        
        if successful_results:
            # Header
            print(f"{'METHOD':<25} {'TRUCKS':<8} {'COST':<10} {'DISTANCE':<12} {'TIME':<8}")
            print("-" * 70)
            
            # Results
            for method_key, result in self.results.items():
                if result['status'] == 'SUCCESS':
                    trucks_str = str(result['selected_trucks'])
                    cost_str = f"€{result['total_cost']:.0f}"
                    distance_str = f"{result['total_distance']:.1f} km"
                    time_str = f"{result['execution_time']:.2f}s"
                    
                    print(f"{result['method']:<25} {trucks_str:<8} {cost_str:<10} {distance_str:<12} {time_str:<8}")
                else:
                    error_msg = result['error'][:20] + "..." if len(result['error']) > 20 else result['error']
                    print(f"{result['method']:<25} {'FAILED':<8} {'':<10} {error_msg:<12} {result['execution_time']:.2f}s")
        
        # Performance analysis
        if len(successful_results) > 1:
            print("\n📈 PERFORMANCE ANALYSIS:")
            print("-" * 40)
            
            # Best cost
            best_cost = min(r['total_cost'] for r in successful_results)
            best_cost_methods = [r['method'] for r in successful_results if r['total_cost'] == best_cost]
            print(f"🏅 Best Cost: €{best_cost:.0f} ({', '.join(best_cost_methods)})")
            
            # Best distance
            best_distance = min(r['total_distance'] for r in successful_results)
            best_distance_methods = [r['method'] for r in successful_results if r['total_distance'] == best_distance]
            print(f"🏅 Best Distance: {best_distance:.1f} km ({', '.join(best_distance_methods)})")
            
            # Fastest execution
            fastest_time = min(r['execution_time'] for r in successful_results)
            fastest_methods = [r['method'] for r in successful_results if r['execution_time'] == fastest_time]
            print(f"🏅 Fastest Execution: {fastest_time:.2f}s ({', '.join(fastest_methods)})")
            
            # Distance efficiency comparison
            if len(successful_results) >= 2:
                distances = [r['total_distance'] for r in successful_results]
                max_distance = max(distances)
                min_distance = min(distances)
                
                if max_distance > min_distance:
                    improvement = ((max_distance - min_distance) / max_distance) * 100
                    print(f"📊 Distance Efficiency Range: {improvement:.1f}% improvement from best to worst")
        
        # Detailed method comparison
        print("\n🔬 DETAILED METHOD ANALYSIS:")
        print("-" * 50)
        
        for method_key, result in self.results.items():
            print(f"\n{result['method']}:")
            if result['status'] == 'SUCCESS':
                print(f"  Status: ✅ SUCCESS")
                print(f"  Execution Time: {result['execution_time']:.2f}s")
                print(f"  Selected Trucks: {result['selected_trucks']}")
                print(f"  Total Cost: €{result['total_cost']:.0f}")
                print(f"  Total Distance: {result['total_distance']:.1f} km")
                
                # Method-specific details
                if 'objective_value' in result:
                    print(f"  Multi-Objective Value: {result['objective_value']:.6f}")
                if 'generations' in result:
                    print(f"  GA Generations: {result['generations']}")
                    print(f"  GA Final Fitness: {result['final_fitness']:.6f}")
                
                # Truck utilization
                avg_utilization = sum(result['truck_utilization'].values()) / len(result['truck_utilization'])
                print(f"  Average Utilization: {avg_utilization:.1f}%")
                
                # Route details
                print(f"  📋 Route Details:")
                for route in result['routes']:
                    truck_id = route['truck_id']
                    assigned_orders = route['assigned_orders']
                    route_sequence = route.get('route_sequence', [])
                    route_distance = route.get('route_distance', 0)
                    num_orders = route.get('num_orders', len(assigned_orders) if isinstance(assigned_orders, list) else 0)
                    
                    print(f"    🚛 Truck {truck_id}: {num_orders} orders → {assigned_orders}")
                    if route_sequence:
                        route_str = " → ".join(route_sequence)
                        print(f"       Route: {route_str} ({route_distance:.1f} km)")
                    else:
                        print(f"       Distance: {route_distance:.1f} km")
                
            else:
                print(f"  Status: ❌ FAILED")
                print(f"  Error: {result['error']}")
                print(f"  Execution Time: {result['execution_time']:.2f}s")
        
        # Method recommendations
        if successful_results:
            self._generate_recommendations(successful_results)
        
        print(f"\n⏱️  Total Comparison Time: {total_time:.2f}s")
        print("=" * 70)
    
    def _generate_recommendations(self, successful_results):
        """Generate method recommendations based on results"""
        print("\n🎯 RECOMMENDATIONS:")
        print("=" * 50)
        
        if len(successful_results) == 1:
            method = successful_results[0]['method']
            print(f"✅ **{method}** - Only successful method")
            return
        
        # Calculate scores for different criteria
        scores = {}
        for result in successful_results:
            method = result['method']
            scores[method] = {
                'cost_rank': 0,
                'distance_rank': 0,
                'speed_rank': 0,
                'balanced_score': 0
            }
        
        # Rank by cost (lower is better)
        sorted_by_cost = sorted(successful_results, key=lambda x: x['total_cost'])
        for i, result in enumerate(sorted_by_cost):
            scores[result['method']]['cost_rank'] = len(sorted_by_cost) - i
        
        # Rank by distance (lower is better) 
        sorted_by_distance = sorted(successful_results, key=lambda x: x['total_distance'])
        for i, result in enumerate(sorted_by_distance):
            scores[result['method']]['distance_rank'] = len(sorted_by_distance) - i
        
        # Rank by speed (faster is better)
        sorted_by_speed = sorted(successful_results, key=lambda x: x['execution_time'])
        for i, result in enumerate(sorted_by_speed):
            scores[result['method']]['speed_rank'] = len(sorted_by_speed) - i
        
        # Calculate balanced scores (equal weight for cost, distance, speed)
        for method in scores:
            scores[method]['balanced_score'] = (
                scores[method]['cost_rank'] + 
                scores[method]['distance_rank'] + 
                scores[method]['speed_rank']
            ) / 3
        
        # Find best performers
        best_cost = max(scores.items(), key=lambda x: x[1]['cost_rank'])
        best_distance = max(scores.items(), key=lambda x: x[1]['distance_rank'])
        best_speed = max(scores.items(), key=lambda x: x[1]['speed_rank'])
        best_balanced = max(scores.items(), key=lambda x: x[1]['balanced_score'])
        
        print(f"💰 **Best for Cost Optimization:** {best_cost[0]}")
        print(f"📏 **Best for Distance Optimization:** {best_distance[0]}")
        print(f"⚡ **Best for Speed:** {best_speed[0]}")
        print(f"⚖️  **Best Overall Balance:** {best_balanced[0]}")
        
        # Provide specific recommendations
        print(f"\n💡 **SPECIFIC RECOMMENDATIONS:**")
        
        # Cost-focused recommendation
        cost_leader = sorted_by_cost[0]
        if cost_leader['total_cost'] < sorted_by_cost[-1]['total_cost']:
            savings = sorted_by_cost[-1]['total_cost'] - cost_leader['total_cost']
            savings_pct = (savings / sorted_by_cost[-1]['total_cost']) * 100
            print(f"💰 For **minimum cost**: Use **{cost_leader['method']}** (saves €{savings:.0f}, {savings_pct:.1f}% reduction)")
        
        # Distance-focused recommendation  
        distance_leader = sorted_by_distance[0]
        if distance_leader['total_distance'] < sorted_by_distance[-1]['total_distance']:
            distance_savings = sorted_by_distance[-1]['total_distance'] - distance_leader['total_distance']
            distance_pct = (distance_savings / sorted_by_distance[-1]['total_distance']) * 100
            print(f"📏 For **minimum distance**: Use **{distance_leader['method']}** (saves {distance_savings:.1f} km, {distance_pct:.1f}% reduction)")
        
        # Speed recommendation
        speed_leader = sorted_by_speed[0]
        if speed_leader['execution_time'] < sorted_by_speed[-1]['execution_time']:
            time_savings = sorted_by_speed[-1]['execution_time'] - speed_leader['execution_time']
            print(f"⚡ For **fastest results**: Use **{speed_leader['method']}** ({time_savings:.2f}s faster)")
        
        # Overall recommendation
        print(f"\n🏆 **OVERALL WINNER:** **{best_balanced[0]}** (best balanced performance)")
        
        # Method-specific advice
        print(f"\n📋 **WHEN TO USE EACH METHOD:**")
        
        method_advice = {
            'Standard MILP + Greedy': '• Quick daily routing\n• Cost is priority\n• Fast decisions needed',
            'Enhanced MILP': '• High-quality routes required\n• Balanced cost-distance optimization\n• Medium computation time acceptable',
            'Genetic Algorithm': '• Large problem sizes\n• Exploration of diverse solutions\n• Longer computation time acceptable'
        }
        
        for result in successful_results:
            method = result['method']
            if method in method_advice:
                print(f"\n**{method}:**")
                print(method_advice[method])
    

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare all three Vehicle Router optimization methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--timeout', type=int, default=120,
                       help='Solver timeout in seconds (default: 120)')
    
    parser.add_argument('--depot', type=str, default='08020',
                       help='Depot location postal code (default: 08020)')
    
    parser.add_argument('--depot-return', action='store_true',
                       help='Enable depot return (default: False)')
    
    parser.add_argument('--cost-weight', type=float, default=0.6,
                       help='Cost weight for multi-objective methods (default: 0.6)')
    
    parser.add_argument('--distance-weight', type=float, default=0.4,
                       help='Distance weight for multi-objective methods (default: 0.4)')
    
    parser.add_argument('--ga-population', type=int, default=50,
                       help='Genetic algorithm population size (default: 50)')
    
    parser.add_argument('--ga-generations', type=int, default=100,
                       help='Genetic algorithm max generations (default: 100)')
    
    parser.add_argument('--ga-mutation', type=float, default=0.1,
                       help='Genetic algorithm mutation rate (default: 0.1)')
    

    parser.add_argument('--real-distances', action='store_true',
                       help='Use real-world distances instead of simulated')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    return parser.parse_args()


def main():
    """Main comparison function"""
    args = parse_arguments()
    
    # Validate weights
    if abs(args.cost_weight + args.distance_weight - 1.0) > 0.001:
        print("Error: Cost weight and distance weight must sum to 1.0")
        sys.exit(1)
    
    # Configuration
    config = {
        'timeout': args.timeout,
        'depot_location': args.depot,
        'depot_return': args.depot_return,
        'cost_weight': args.cost_weight,
        'distance_weight': args.distance_weight,
        'ga_population': args.ga_population,
        'ga_generations': args.ga_generations,
        'ga_mutation': args.ga_mutation,
        'use_real_distances': args.real_distances,
        'quiet': args.quiet
    }
    
    # Run comparison
    comparison = OptimizationComparison(config)
    comparison.run_comparison()


if __name__ == "__main__":
    main() 