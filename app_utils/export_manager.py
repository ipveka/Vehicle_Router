"""Export Manager Module for creating Excel reports, CSV files, and detailed text reports"""

import pandas as pd
import io
import time
from typing import Dict, Any, Optional
import json


class ExportManager:
    """Export Manager for Streamlit Application handling various export formats"""
    
    def __init__(self):
        pass
    
    def create_excel_report(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                           solution: Dict[str, Any]) -> bytes:
        """Create a comprehensive Excel report with multiple sheets"""
        # Create Excel writer object
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = self._create_summary_data(orders_df, trucks_df, solution)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Orders
            orders_df.to_excel(writer, sheet_name='Orders', index=False)
            
            # Sheet 3: Trucks
            trucks_df.to_excel(writer, sheet_name='Trucks', index=False)
            
            # Sheet 4: Assignments
            solution['assignments_df'].to_excel(writer, sheet_name='Assignments', index=False)
            
            # Sheet 5: Routes
            solution['routes_df'].to_excel(writer, sheet_name='Routes', index=False)
            
            # Sheet 6: Utilization
            utilization_data = []
            for truck_id, util_info in solution['utilization'].items():
                utilization_data.append({
                    'Truck ID': truck_id,
                    'Capacity (m³)': util_info['capacity'],
                    'Used Volume (m³)': util_info['used_volume'],
                    'Utilization (%)': util_info['utilization_percent'],
                    'Assigned Orders': ', '.join(util_info['assigned_orders'])
                })
            
            utilization_df = pd.DataFrame(utilization_data)
            utilization_df.to_excel(writer, sheet_name='Utilization', index=False)
            
            # Sheet 7: Cost Breakdown
            cost_data = []
            for truck_id, cost in solution['costs']['truck_costs'].items():
                cost_data.append({
                    'Truck ID': truck_id,
                    'Cost (€)': cost,
                    'Selected': True
                })
            
            cost_df = pd.DataFrame(cost_data)
            cost_df.to_excel(writer, sheet_name='Cost Breakdown', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def create_summary_csv(self, solution: Dict[str, Any]) -> str:
        """
        Create a CSV summary of the solution
        
        Args:
            solution: Optimization solution
            
        Returns:
            str: CSV content as string
        """
        summary_data = []
        
        # Basic solution info
        summary_data.append(['Metric', 'Value', 'Unit'])
        summary_data.append(['Total Cost', solution['costs']['total_cost'], 'EUR'])
        summary_data.append(['Trucks Used', len(solution['selected_trucks']), 'count'])
        summary_data.append(['Orders Delivered', len(solution['assignments_df']), 'count'])
        
        # Utilization metrics
        if solution['utilization']:
            avg_util = sum(u['utilization_percent'] for u in solution['utilization'].values()) / len(solution['utilization'])
            summary_data.append(['Average Utilization', avg_util, '%'])
            
            min_util = min(u['utilization_percent'] for u in solution['utilization'].values())
            summary_data.append(['Minimum Utilization', min_util, '%'])
            
            max_util = max(u['utilization_percent'] for u in solution['utilization'].values())
            summary_data.append(['Maximum Utilization', max_util, '%'])
        
        # Convert to CSV string
        csv_lines = []
        for row in summary_data:
            csv_lines.append(','.join(str(item) for item in row))
        
        return '\n'.join(csv_lines)
    
    def create_detailed_report(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                              solution: Dict[str, Any]) -> str:
        """
        Create a detailed text report
        
        Args:
            orders_df: Orders data
            trucks_df: Trucks data
            solution: Optimization solution
            
        Returns:
            str: Detailed report as text
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 60,
            "VEHICLE ROUTER OPTIMIZATION REPORT",
            "=" * 60,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ])
        
        # Problem Overview
        report_lines.extend([
            "PROBLEM OVERVIEW",
            "-" * 20,
            f"Total Orders: {len(orders_df)}",
            f"Total Volume: {orders_df['volume'].sum():.1f} m³",
            f"Available Trucks: {len(trucks_df)}",
            f"Total Capacity: {trucks_df['capacity'].sum():.1f} m³",
            ""
        ])
        
        # Solution Summary
        report_lines.extend([
            "SOLUTION SUMMARY",
            "-" * 20,
            f"Selected Trucks: {solution['selected_trucks']}",
            f"Total Cost: €{solution['costs']['total_cost']:.0f}",
            f"Trucks Used: {len(solution['selected_trucks'])}/{len(trucks_df)}",
            f"Orders Delivered: {len(solution['assignments_df'])}/{len(orders_df)}",
            ""
        ])
        
        # Truck Assignments
        report_lines.extend([
            "TRUCK ASSIGNMENTS",
            "-" * 20
        ])
        
        for truck_id in solution['selected_trucks']:
            assigned_orders = [a['order_id'] for a in solution['assignments_df'].to_dict('records') 
                             if a['truck_id'] == truck_id]
            truck_info = trucks_df[trucks_df['truck_id'] == truck_id].iloc[0]
            util_info = solution['utilization'][truck_id]
            
            report_lines.extend([
                f"Truck {truck_id}:",
                f"  Capacity: {truck_info['capacity']:.1f} m³",
                f"  Cost: €{truck_info['cost']:.0f}",
                f"  Assigned Orders: {assigned_orders}",
                f"  Used Volume: {util_info['used_volume']:.1f} m³",
                f"  Utilization: {util_info['utilization_percent']:.1f}%",
                ""
            ])
        
        # Performance Metrics
        if solution['utilization']:
            utilizations = [u['utilization_percent'] for u in solution['utilization'].values()]
            avg_util = sum(utilizations) / len(utilizations)
            
            report_lines.extend([
                "PERFORMANCE METRICS",
                "-" * 20,
                f"Average Utilization: {avg_util:.1f}%",
                f"Minimum Utilization: {min(utilizations):.1f}%",
                f"Maximum Utilization: {max(utilizations):.1f}%",
                f"Cost per Order: €{solution['costs']['total_cost'] / len(orders_df):.2f}",
                f"Cost per m³: €{solution['costs']['total_cost'] / orders_df['volume'].sum():.2f}",
                ""
            ])
        
        # Order Details
        report_lines.extend([
            "ORDER DETAILS",
            "-" * 20
        ])
        
        for _, order in orders_df.iterrows():
            # Find which truck this order is assigned to
            assignment = solution['assignments_df'][solution['assignments_df']['order_id'] == order['order_id']]
            if not assignment.empty:
                truck_id = assignment.iloc[0]['truck_id']
                report_lines.append(f"Order {order['order_id']}: {order['volume']:.1f} m³ at {order['postal_code']} → Truck {truck_id}")
        
        report_lines.append("")
        
        # Footer
        report_lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return '\n'.join(report_lines)
    
    def _create_summary_data(self, orders_df: pd.DataFrame, trucks_df: pd.DataFrame, 
                            solution: Dict[str, Any]) -> list:
        """
        Create summary data for Excel export
        
        Args:
            orders_df: Orders data
            trucks_df: Trucks data
            solution: Optimization solution
            
        Returns:
            list: Summary data as list of dictionaries
        """
        summary_data = []
        
        # Basic metrics
        summary_data.append({
            'Metric': 'Total Cost',
            'Value': solution['costs']['total_cost'],
            'Unit': 'EUR'
        })
        
        summary_data.append({
            'Metric': 'Trucks Used',
            'Value': len(solution['selected_trucks']),
            'Unit': 'count'
        })
        
        summary_data.append({
            'Metric': 'Orders Delivered',
            'Value': len(solution['assignments_df']),
            'Unit': 'count'
        })
        
        summary_data.append({
            'Metric': 'Total Volume',
            'Value': orders_df['volume'].sum(),
            'Unit': 'm³'
        })
        
        summary_data.append({
            'Metric': 'Total Capacity',
            'Value': trucks_df['capacity'].sum(),
            'Unit': 'm³'
        })
        
        # Utilization metrics
        if solution['utilization']:
            utilizations = [u['utilization_percent'] for u in solution['utilization'].values()]
            
            summary_data.append({
                'Metric': 'Average Utilization',
                'Value': sum(utilizations) / len(utilizations),
                'Unit': '%'
            })
            
            summary_data.append({
                'Metric': 'Minimum Utilization',
                'Value': min(utilizations),
                'Unit': '%'
            })
            
            summary_data.append({
                'Metric': 'Maximum Utilization',
                'Value': max(utilizations),
                'Unit': '%'
            })
        
        # Cost efficiency
        summary_data.append({
            'Metric': 'Cost per Order',
            'Value': solution['costs']['total_cost'] / len(orders_df) if len(orders_df) > 0 else 0,
            'Unit': 'EUR/order'
        })
        
        summary_data.append({
            'Metric': 'Cost per m³',
            'Value': solution['costs']['total_cost'] / orders_df['volume'].sum() if orders_df['volume'].sum() > 0 else 0,
            'Unit': 'EUR/m³'
        })
        
        return summary_data