"""
Solution Validation Module

This module provides the SolutionValidator class for validating optimization results
from the Vehicle Routing Problem solver. It performs comprehensive checks to ensure
solution correctness and constraint satisfaction.

The validator checks:
- Truck capacity constraints are satisfied
- All orders are delivered exactly once
- Route feasibility and logic
- Solution completeness and consistency

Classes:
    SolutionValidator: Main class for validating VRP optimization solutions
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SolutionValidator:
    """
    Solution Validator for Vehicle Routing Problem
    
    This class validates optimization solutions from the VRP solver to ensure
    all constraints are satisfied and the solution is feasible. It performs
    comprehensive checks including capacity constraints, order assignments,
    and route feasibility.
    
    The validator checks:
    - Capacity constraints: No truck exceeds its capacity limit
    - Order delivery: All orders are assigned to exactly one truck
    - Route feasibility: Routes are logically consistent
    - Solution completeness: All required data is present
    
    Attributes:
        solution (Dict): The optimization solution to validate
        orders_df (pd.DataFrame): Original orders data
        trucks_df (pd.DataFrame): Original trucks data
        validation_results (Dict): Results of validation checks
        
    Example:
        >>> validator = SolutionValidator(solution, orders_df, trucks_df)
        >>> validation_report = validator.validate_solution()
        >>> if validation_report['is_valid']:
        ...     print("Solution is valid!")
    """
    
    def __init__(self, solution: Dict[str, Any], orders_df: pd.DataFrame, 
                 trucks_df: pd.DataFrame):
        """
        Initialize the Solution Validator with solution and input data
        
        Args:
            solution (Dict[str, Any]): The optimization solution to validate
            orders_df (pd.DataFrame): Original orders data with columns [order_id, volume, postal_code]
            trucks_df (pd.DataFrame): Original trucks data with columns [truck_id, capacity, cost]
            
        Raises:
            ValueError: If input data is invalid or inconsistent
            TypeError: If input data types are incorrect
        """
        logger.info("Initializing Solution Validator...")
        
        # Validate input parameters
        self._validate_inputs(solution, orders_df, trucks_df)
        
        # Store input data
        self.solution = solution
        self.orders_df = orders_df.copy()
        self.trucks_df = trucks_df.copy()
        
        # Initialize validation results
        self.validation_results = {}
        
        # Extract key data for validation
        self.order_volumes = dict(zip(self.orders_df['order_id'], self.orders_df['volume']))
        self.truck_capacities = dict(zip(self.trucks_df['truck_id'], self.trucks_df['capacity']))
        self.all_order_ids = set(self.orders_df['order_id'].tolist())
        self.all_truck_ids = set(self.trucks_df['truck_id'].tolist())
        
        logger.info(f"Validator initialized for {len(self.all_order_ids)} orders and {len(self.all_truck_ids)} trucks")
    
    def _validate_inputs(self, solution: Dict[str, Any], orders_df: pd.DataFrame, 
                        trucks_df: pd.DataFrame) -> None:
        """
        Validate input parameters for the validator
        
        Args:
            solution (Dict[str, Any]): Solution data to validate
            orders_df (pd.DataFrame): Orders data to validate
            trucks_df (pd.DataFrame): Trucks data to validate
            
        Raises:
            ValueError: If validation fails
            TypeError: If data types are incorrect
        """
        # Validate solution structure
        if not isinstance(solution, dict):
            raise TypeError("solution must be a dictionary")
        
        required_solution_keys = ['assignments_df', 'selected_trucks']
        missing_keys = [key for key in required_solution_keys if key not in solution]
        if missing_keys:
            raise ValueError(f"solution missing required keys: {missing_keys}")
        
        # Validate orders DataFrame
        if not isinstance(orders_df, pd.DataFrame):
            raise TypeError("orders_df must be a pandas DataFrame")
        
        required_order_cols = ['order_id', 'volume', 'postal_code']
        missing_cols = [col for col in required_order_cols if col not in orders_df.columns]
        if missing_cols:
            raise ValueError(f"orders_df missing required columns: {missing_cols}")
        
        # Validate trucks DataFrame
        if not isinstance(trucks_df, pd.DataFrame):
            raise TypeError("trucks_df must be a pandas DataFrame")
        
        required_truck_cols = ['truck_id', 'capacity', 'cost']
        missing_cols = [col for col in required_truck_cols if col not in trucks_df.columns]
        if missing_cols:
            raise ValueError(f"trucks_df missing required columns: {missing_cols}")
        
        logger.info("Input validation completed successfully")
    
    def check_capacity(self) -> Dict[str, Any]:
        """
        Verify that truck capacity constraints are satisfied
        
        Checks that the total volume of orders assigned to each truck does not
        exceed the truck's capacity limit. Reports violations with detailed
        information about capacity usage.
        
        Returns:
            Dict[str, Any]: Capacity validation results containing:
                - is_valid (bool): Whether all capacity constraints are satisfied
                - violations (List): List of capacity violations with details
                - truck_utilization (Dict): Capacity utilization for each truck
                - summary (str): Human-readable summary of capacity check
        """
        logger.info("Checking truck capacity constraints...")
        
        capacity_results = {
            'is_valid': True,
            'violations': [],
            'truck_utilization': {},
            'summary': ''
        }
        
        try:
            # Get assignments from solution
            assignments_df = self.solution['assignments_df']
            selected_trucks = self.solution['selected_trucks']
            
            # Check capacity for each selected truck
            for truck_id in selected_trucks:
                # Get orders assigned to this truck
                truck_assignments = assignments_df[assignments_df['truck_id'] == truck_id]
                assigned_order_ids = truck_assignments['order_id'].tolist()
                
                # Calculate total volume assigned to truck
                total_volume = sum(self.order_volumes[order_id] for order_id in assigned_order_ids)
                truck_capacity = self.truck_capacities[truck_id]
                
                # Calculate utilization
                utilization_percent = (total_volume / truck_capacity) * 100 if truck_capacity > 0 else 0
                
                # Store utilization data
                capacity_results['truck_utilization'][truck_id] = {
                    'assigned_orders': assigned_order_ids,
                    'total_volume': total_volume,
                    'capacity': truck_capacity,
                    'utilization_percent': utilization_percent,
                    'remaining_capacity': truck_capacity - total_volume
                }
                
                # Check for capacity violation
                if total_volume > truck_capacity:
                    capacity_results['is_valid'] = False
                    violation = {
                        'truck_id': truck_id,
                        'assigned_volume': total_volume,
                        'capacity': truck_capacity,
                        'excess_volume': total_volume - truck_capacity,
                        'assigned_orders': assigned_order_ids,
                        'error_message': f"Truck {truck_id} exceeds capacity: {total_volume:.1f} m³ > {truck_capacity:.1f} m³"
                    }
                    capacity_results['violations'].append(violation)
                    logger.error(violation['error_message'])
                else:
                    logger.info(f"Truck {truck_id}: {total_volume:.1f}/{truck_capacity:.1f} m³ ({utilization_percent:.1f}% utilization)")
            
            # Generate summary
            if capacity_results['is_valid']:
                avg_utilization = np.mean([u['utilization_percent'] for u in capacity_results['truck_utilization'].values()])
                capacity_results['summary'] = f"All capacity constraints satisfied. Average utilization: {avg_utilization:.1f}%"
                logger.info("Capacity constraint check: PASSED")
            else:
                num_violations = len(capacity_results['violations'])
                capacity_results['summary'] = f"Capacity constraint violations found: {num_violations} trucks exceed capacity"
                logger.error(f"Capacity constraint check: FAILED ({num_violations} violations)")
        
        except Exception as e:
            capacity_results['is_valid'] = False
            capacity_results['summary'] = f"Error during capacity check: {str(e)}"
            logger.error(f"Error in capacity check: {str(e)}")
        
        return capacity_results
    
    def check_all_orders_delivered(self) -> Dict[str, Any]:
        """
        Ensure that every order is assigned to exactly one truck
        
        Verifies that all orders from the original dataset are present in the
        solution and that no order is assigned to multiple trucks or left
        unassigned.
        
        Returns:
            Dict[str, Any]: Order delivery validation results containing:
                - is_valid (bool): Whether all orders are properly assigned
                - missing_orders (List): Orders not assigned to any truck
                - duplicate_assignments (List): Orders assigned to multiple trucks
                - assignment_summary (Dict): Summary of order assignments
                - summary (str): Human-readable summary of delivery check
        """
        logger.info("Checking that all orders are delivered...")
        
        delivery_results = {
            'is_valid': True,
            'missing_orders': [],
            'duplicate_assignments': [],
            'assignment_summary': {},
            'summary': ''
        }
        
        try:
            # Get assignments from solution
            assignments_df = self.solution['assignments_df']
            
            # Count assignments per order
            assignment_counts = assignments_df['order_id'].value_counts()
            assigned_order_ids = set(assignments_df['order_id'].tolist())
            
            # Check for missing orders (not assigned to any truck)
            missing_orders = self.all_order_ids - assigned_order_ids
            if missing_orders:
                delivery_results['is_valid'] = False
                delivery_results['missing_orders'] = list(missing_orders)
                logger.error(f"Missing orders not assigned to any truck: {list(missing_orders)}")
            
            # Check for duplicate assignments (order assigned to multiple trucks)
            duplicate_orders = assignment_counts[assignment_counts > 1].index.tolist()
            if duplicate_orders:
                delivery_results['is_valid'] = False
                for order_id in duplicate_orders:
                    trucks_assigned = assignments_df[assignments_df['order_id'] == order_id]['truck_id'].tolist()
                    duplicate_info = {
                        'order_id': order_id,
                        'assigned_trucks': trucks_assigned,
                        'assignment_count': len(trucks_assigned),
                        'error_message': f"Order {order_id} assigned to multiple trucks: {trucks_assigned}"
                    }
                    delivery_results['duplicate_assignments'].append(duplicate_info)
                    logger.error(duplicate_info['error_message'])
            
            # Create assignment summary
            delivery_results['assignment_summary'] = {
                'total_orders': len(self.all_order_ids),
                'assigned_orders': len(assigned_order_ids),
                'missing_orders': len(missing_orders),
                'duplicate_assignments': len(duplicate_orders),
                'properly_assigned': len(self.all_order_ids) - len(missing_orders) - len(duplicate_orders)
            }
            
            # Generate summary
            if delivery_results['is_valid']:
                delivery_results['summary'] = f"All {len(self.all_order_ids)} orders properly assigned to exactly one truck"
                logger.info("Order delivery check: PASSED")
            else:
                issues = []
                if missing_orders:
                    issues.append(f"{len(missing_orders)} missing")
                if duplicate_orders:
                    issues.append(f"{len(duplicate_orders)} duplicated")
                delivery_results['summary'] = f"Order assignment issues: {', '.join(issues)}"
                logger.error(f"Order delivery check: FAILED ({delivery_results['summary']})")
        
        except Exception as e:
            delivery_results['is_valid'] = False
            delivery_results['summary'] = f"Error during order delivery check: {str(e)}"
            logger.error(f"Error in order delivery check: {str(e)}")
        
        return delivery_results
    
    def check_route_feasibility(self) -> Dict[str, Any]:
        """
        Validate route logic and feasibility
        
        Checks that the routes are logically consistent, including:
        - Selected trucks have at least one order assigned
        - No empty trucks are marked as selected
        - Route data is consistent with assignments
        
        Returns:
            Dict[str, Any]: Route feasibility validation results containing:
                - is_valid (bool): Whether routes are feasible
                - empty_trucks (List): Selected trucks with no orders
                - unselected_with_orders (List): Unselected trucks with orders
                - route_summary (Dict): Summary of route information
                - summary (str): Human-readable summary of route check
        """
        logger.info("Checking route feasibility...")
        
        route_results = {
            'is_valid': True,
            'empty_trucks': [],
            'unselected_with_orders': [],
            'route_summary': {},
            'summary': ''
        }
        
        try:
            # Get solution data
            assignments_df = self.solution['assignments_df']
            selected_trucks = self.solution['selected_trucks']
            
            # Get trucks with orders assigned
            trucks_with_orders = set(assignments_df['truck_id'].tolist())
            selected_trucks_set = set(selected_trucks)
            
            # Check for selected trucks with no orders (empty trucks)
            empty_trucks = selected_trucks_set - trucks_with_orders
            if empty_trucks:
                route_results['is_valid'] = False
                route_results['empty_trucks'] = list(empty_trucks)
                logger.error(f"Selected trucks with no orders assigned: {list(empty_trucks)}")
            
            # Check for unselected trucks with orders assigned
            unselected_with_orders = trucks_with_orders - selected_trucks_set
            if unselected_with_orders:
                route_results['is_valid'] = False
                route_results['unselected_with_orders'] = list(unselected_with_orders)
                logger.error(f"Unselected trucks with orders assigned: {list(unselected_with_orders)}")
            
            # Create route summary
            route_results['route_summary'] = {
                'total_trucks_available': len(self.all_truck_ids),
                'trucks_selected': len(selected_trucks),
                'trucks_with_orders': len(trucks_with_orders),
                'empty_selected_trucks': len(empty_trucks),
                'unselected_trucks_with_orders': len(unselected_with_orders)
            }
            
            # Log route information for each selected truck
            for truck_id in selected_trucks:
                truck_orders = assignments_df[assignments_df['truck_id'] == truck_id]['order_id'].tolist()
                logger.info(f"Truck {truck_id}: {len(truck_orders)} orders assigned {truck_orders}")
            
            # Generate summary
            if route_results['is_valid']:
                route_results['summary'] = f"Route feasibility check passed: {len(selected_trucks)} trucks properly selected and assigned"
                logger.info("Route feasibility check: PASSED")
            else:
                issues = []
                if empty_trucks:
                    issues.append(f"{len(empty_trucks)} empty selected trucks")
                if unselected_with_orders:
                    issues.append(f"{len(unselected_with_orders)} unselected trucks with orders")
                route_results['summary'] = f"Route feasibility issues: {', '.join(issues)}"
                logger.error(f"Route feasibility check: FAILED ({route_results['summary']})")
        
        except Exception as e:
            route_results['is_valid'] = False
            route_results['summary'] = f"Error during route feasibility check: {str(e)}"
            logger.error(f"Error in route feasibility check: {str(e)}")
        
        return route_results
    
    def validate_solution(self) -> Dict[str, Any]:
        """
        Run all validation checks and return comprehensive validation report
        
        Performs all validation checks including capacity constraints, order
        delivery verification, and route feasibility. Combines results into
        a comprehensive report with overall validation status.
        
        Returns:
            Dict[str, Any]: Comprehensive validation report containing:
                - is_valid (bool): Overall validation status
                - capacity_check (Dict): Results from capacity validation
                - delivery_check (Dict): Results from order delivery validation
                - route_check (Dict): Results from route feasibility validation
                - summary (str): Overall validation summary
                - error_count (int): Total number of validation errors
                - warnings (List): List of warnings (non-critical issues)
        """
        logger.info("Starting comprehensive solution validation...")
        
        # Run individual validation checks
        capacity_results = self.check_capacity()
        delivery_results = self.check_all_orders_delivered()
        route_results = self.check_route_feasibility()
        
        # Combine results
        overall_valid = (capacity_results['is_valid'] and 
                        delivery_results['is_valid'] and 
                        route_results['is_valid'])
        
        # Count total errors
        error_count = 0
        error_count += len(capacity_results.get('violations', []))
        error_count += len(delivery_results.get('missing_orders', []))
        error_count += len(delivery_results.get('duplicate_assignments', []))
        error_count += len(route_results.get('empty_trucks', []))
        error_count += len(route_results.get('unselected_with_orders', []))
        
        # Generate warnings for non-critical issues
        warnings = []
        
        # Check for low utilization (warning, not error)
        if capacity_results['is_valid'] and capacity_results['truck_utilization']:
            for truck_id, util_info in capacity_results['truck_utilization'].items():
                if util_info['utilization_percent'] < 50:
                    warnings.append(f"Truck {truck_id} has low utilization: {util_info['utilization_percent']:.1f}%")
        
        # Create comprehensive validation report
        validation_report = {
            'is_valid': overall_valid,
            'capacity_check': capacity_results,
            'delivery_check': delivery_results,
            'route_check': route_results,
            'error_count': error_count,
            'warnings': warnings,
            'summary': ''
        }
        
        # Generate overall summary
        if overall_valid:
            validation_report['summary'] = f"Solution validation PASSED: All constraints satisfied"
            if warnings:
                validation_report['summary'] += f" ({len(warnings)} warnings)"
            logger.info("=== SOLUTION VALIDATION PASSED ===")
        else:
            failed_checks = []
            if not capacity_results['is_valid']:
                failed_checks.append("capacity constraints")
            if not delivery_results['is_valid']:
                failed_checks.append("order delivery")
            if not route_results['is_valid']:
                failed_checks.append("route feasibility")
            
            validation_report['summary'] = f"Solution validation FAILED: Issues with {', '.join(failed_checks)} ({error_count} total errors)"
            logger.error("=== SOLUTION VALIDATION FAILED ===")
        
        # Store validation results for future reference
        self.validation_results = validation_report
        
        logger.info(f"Validation completed: {validation_report['summary']}")
        return validation_report
    
    def get_validation_summary_text(self) -> str:
        """
        Generate a human-readable text summary of the validation results
        
        Returns:
            str: Formatted text summary of validation results
        """
        if not self.validation_results:
            return "No validation results available - run validate_solution() first"
        
        lines = ["=== SOLUTION VALIDATION REPORT ==="]
        
        # Overall status
        status = "PASSED" if self.validation_results['is_valid'] else "FAILED"
        lines.append(f"Overall Status: {status}")
        lines.append("")
        
        # Individual check results
        checks = [
            ("Capacity Constraints", self.validation_results['capacity_check']),
            ("Order Delivery", self.validation_results['delivery_check']),
            ("Route Feasibility", self.validation_results['route_check'])
        ]
        
        for check_name, check_results in checks:
            status = "PASSED" if check_results['is_valid'] else "FAILED"
            lines.append(f"{check_name}: {status}")
            if check_results['summary']:
                lines.append(f"  {check_results['summary']}")
        
        # Warnings
        if self.validation_results['warnings']:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.validation_results['warnings']:
                lines.append(f"  - {warning}")
        
        # Error summary
        if self.validation_results['error_count'] > 0:
            lines.append("")
            lines.append(f"Total Errors: {self.validation_results['error_count']}")
        
        return "\n".join(lines)