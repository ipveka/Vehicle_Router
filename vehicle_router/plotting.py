"""
Plotting and Visualization Module

This module provides comprehensive visualization capabilities for the Vehicle Routing Problem
optimization results. It includes functions for plotting delivery routes, cost analysis,
and capacity utilization using matplotlib and seaborn for clean, professional visualizations.

The module supports:
- Route visualization on 2D grids showing truck paths and order locations
- Cost breakdown charts displaying truck cost contributions
- Capacity utilization analysis showing truck efficiency metrics
- Clean styling with proper labels, legends, and color schemes

Functions:
    plot_routes: Visualize delivery routes on a 2D coordinate grid
    plot_costs: Show bar chart of truck cost contributions
    plot_utilization: Display capacity utilization rates for selected trucks
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set seaborn style for clean, professional plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def plot_routes(solution_df: pd.DataFrame, orders_df: pd.DataFrame, 
                trucks_df: pd.DataFrame, distance_matrix: pd.DataFrame = None,
                figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize delivery routes on a 2D grid showing truck paths and order locations
    
    Creates a comprehensive route visualization displaying:
    - Order locations as points with labels and volume information
    - Truck routes as colored lines connecting assigned orders
    - Legend showing truck assignments and costs
    - Grid layout based on postal code coordinates
    
    Args:
        solution_df (pd.DataFrame): Solution data with truck assignments
        orders_df (pd.DataFrame): Orders data with columns [order_id, volume, postal_code]
        trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
        distance_matrix (pd.DataFrame, optional): Distance matrix for route optimization
        figsize (Tuple[int, int]): Figure size as (width, height)
        save_path (Optional[str]): Path to save the plot image
        
    Returns:
        plt.Figure: The matplotlib figure object
        
    Example:
        >>> fig = plot_routes(solution['routes_df'], orders_df, trucks_df)
        >>> plt.show()
    """
    logger.info("Creating route visualization plot...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert postal codes to coordinates (assuming sequential postal codes)
    # Extract numeric part of postal codes for positioning
    postal_coords = {}
    for _, order in orders_df.iterrows():
        postal_code = order['postal_code']
        # Convert postal code to coordinate (e.g., 08027 -> x=0, 08028 -> x=1, etc.)
        coord_x = int(postal_code) - int(orders_df['postal_code'].min())
        coord_y = 0  # Simple 1D layout on x-axis
        postal_coords[postal_code] = (coord_x, coord_y)
    
    # Define colors for different trucks
    colors = plt.cm.Set1(np.linspace(0, 1, len(solution_df)))
    truck_colors = {}
    
    # Plot order locations
    logger.info("Plotting order locations...")
    for _, order in orders_df.iterrows():
        postal_code = order['postal_code']
        x, y = postal_coords[postal_code]
        
        # Plot order as a circle with size proportional to volume
        volume = order['volume']
        size = max(100, volume * 3)  # Scale size based on volume
        
        ax.scatter(x, y, s=size, c='lightblue', edgecolors='navy', 
                  alpha=0.7, zorder=3, linewidth=2)
        
        # Add order label with volume information
        ax.annotate(f"Order {order['order_id']}\n{volume}m³", 
                   (x, y), xytext=(0, 15), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot truck routes
    logger.info("Plotting truck routes...")
    legend_elements = []
    
    for idx, (_, route) in enumerate(solution_df.iterrows()):
        truck_id = route['truck_id']
        assigned_orders = route['assigned_orders']
        postal_codes = route['postal_codes']
        
        if len(assigned_orders) == 0:
            continue
            
        color = colors[idx]
        truck_colors[truck_id] = color
        
        # Get truck information
        truck_info = trucks_df[trucks_df['truck_id'] == truck_id].iloc[0]
        truck_cost = truck_info['cost']
        truck_capacity = truck_info['capacity']
        
        # Calculate route coordinates
        route_coords = [postal_coords[pc] for pc in postal_codes]
        
        if len(route_coords) > 1:
            # Plot route as connected line segments
            x_coords = [coord[0] for coord in route_coords]
            y_coords = [coord[1] for coord in route_coords]
            
            ax.plot(x_coords, y_coords, color=color, linewidth=3, 
                   alpha=0.8, zorder=2, marker='o', markersize=8)
            
            # Add arrows to show direction
            for i in range(len(x_coords) - 1):
                dx = x_coords[i+1] - x_coords[i]
                dy = y_coords[i+1] - y_coords[i]
                ax.annotate('', xy=(x_coords[i+1], y_coords[i+1]), 
                           xytext=(x_coords[i], y_coords[i]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Add to legend
        total_volume = sum(orders_df[orders_df['order_id'].isin(assigned_orders)]['volume'])
        utilization = (total_volume / truck_capacity) * 100
        
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=3,
                      label=f'Truck {truck_id}: €{truck_cost:.0f} ({utilization:.1f}% full)')
        )
    
    # Customize plot appearance
    ax.set_xlabel('Postal Code Distance (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Route Layout', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Routing Optimization - Delivery Routes', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with padding
    x_coords = [coord[0] for coord in postal_coords.values()]
    ax.set_xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
    ax.set_ylim(-0.5, 0.5)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    # Add summary text box
    total_cost = sum(trucks_df[trucks_df['truck_id'].isin(solution_df['truck_id'])]['cost'])
    total_orders = len(orders_df)
    trucks_used = len(solution_df)
    
    summary_text = f"Summary:\n• {trucks_used} trucks used\n• {total_orders} orders delivered\n• Total cost: €{total_cost:.0f}"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Route plot saved to {save_path}")
    
    logger.info("Route visualization completed")
    return fig


def plot_costs(cost_summary: Dict[str, Any], trucks_df: pd.DataFrame,
               figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> plt.Figure:
    """
    Show bar chart of truck cost contributions with detailed breakdown
    
    Creates a comprehensive cost analysis visualization displaying:
    - Individual truck costs as bars with different colors
    - Total cost summary and breakdown
    - Cost per unit capacity analysis
    - Clean styling with proper labels and formatting
    
    Args:
        cost_summary (Dict[str, Any]): Cost breakdown data from optimization solution
        trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
        figsize (Tuple[int, int]): Figure size as (width, height)
        save_path (Optional[str]): Path to save the plot image
        
    Returns:
        plt.Figure: The matplotlib figure object
        
    Example:
        >>> fig = plot_costs(solution['costs'], trucks_df)
        >>> plt.show()
    """
    logger.info("Creating cost analysis plot...")
    
    # Extract cost data
    truck_costs = cost_summary.get('truck_costs', {})
    total_cost = cost_summary.get('total_cost', 0)
    
    if not truck_costs:
        logger.warning("No truck cost data available for plotting")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No cost data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=16)
        return fig
    
    # Prepare data for plotting
    truck_ids = list(truck_costs.keys())
    costs = list(truck_costs.values())
    
    # Get additional truck information
    truck_info = {}
    for truck_id in truck_ids:
        truck_data = trucks_df[trucks_df['truck_id'] == truck_id].iloc[0]
        truck_info[truck_id] = {
            'capacity': truck_data['capacity'],
            'cost_per_m3': truck_data['cost'] / truck_data['capacity'] if truck_data['capacity'] > 0 else 0
        }
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Truck costs bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(truck_ids)))
    bars = ax1.bar(range(len(truck_ids)), costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize first subplot
    ax1.set_xlabel('Truck ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cost (€)', fontsize=12, fontweight='bold')
    ax1.set_title('Selected Truck Costs', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(truck_ids)))
    ax1.set_xticklabels([f'Truck {tid}' for tid in truck_ids])
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(costs) * 0.01,
                f'€{cost:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(costs) * 1.15)
    
    # Plot 2: Cost efficiency (cost per m³)
    cost_per_m3 = [truck_info[tid]['cost_per_m3'] for tid in truck_ids]
    bars2 = ax2.bar(range(len(truck_ids)), cost_per_m3, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize second subplot
    ax2.set_xlabel('Truck ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost per m³ (€/m³)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(truck_ids)))
    ax2.set_xticklabels([f'Truck {tid}' for tid in truck_ids])
    
    # Add value labels on bars
    for i, (bar, cost_eff) in enumerate(zip(bars2, cost_per_m3)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(cost_per_m3) * 0.01,
                f'€{cost_eff:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(cost_per_m3) * 1.15)
    
    # Add overall summary
    fig.suptitle(f'Cost Analysis - Total: €{total_cost:.0f}', fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary statistics text
    avg_cost = np.mean(costs)
    avg_efficiency = np.mean(cost_per_m3)
    summary_text = f"Average Cost: €{avg_cost:.0f}\nAverage Efficiency: €{avg_efficiency:.1f}/m³"
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Cost analysis plot saved to {save_path}")
    
    logger.info("Cost analysis plot completed")
    return fig


def plot_utilization(utilization_data: Dict[int, Dict[str, Any]], trucks_df: pd.DataFrame,
                    figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> plt.Figure:
    """
    Display capacity utilization rates for selected trucks with detailed metrics
    
    Creates a comprehensive utilization analysis showing:
    - Capacity utilization percentages as horizontal bar chart
    - Used vs available capacity comparison
    - Efficiency metrics and recommendations
    - Color coding based on utilization levels
    
    Args:
        utilization_data (Dict[int, Dict[str, Any]]): Truck utilization data from solution
        trucks_df (pd.DataFrame): Trucks data with columns [truck_id, capacity, cost]
        figsize (Tuple[int, int]): Figure size as (width, height)
        save_path (Optional[str]): Path to save the plot image
        
    Returns:
        plt.Figure: The matplotlib figure object
        
    Example:
        >>> fig = plot_utilization(solution['utilization'], trucks_df)
        >>> plt.show()
    """
    logger.info("Creating capacity utilization plot...")
    
    if not utilization_data:
        logger.warning("No utilization data available for plotting")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No utilization data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=16)
        return fig
    
    # Prepare data for plotting
    truck_ids = list(utilization_data.keys())
    truck_ids.sort()  # Sort for consistent ordering
    
    utilization_percentages = []
    used_volumes = []
    capacities = []
    assigned_orders_list = []
    
    for truck_id in truck_ids:
        util_info = utilization_data[truck_id]
        utilization_percentages.append(util_info['utilization_percent'])
        used_volumes.append(util_info['used_volume'])
        capacities.append(util_info['capacity'])
        assigned_orders_list.append(util_info['assigned_orders'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Plot 1: Utilization percentages as horizontal bars
    # Color code based on utilization levels
    colors = []
    for util in utilization_percentages:
        if util >= 90:
            colors.append('darkgreen')  # Excellent utilization
        elif util >= 75:
            colors.append('green')      # Good utilization
        elif util >= 50:
            colors.append('orange')     # Moderate utilization
        else:
            colors.append('red')        # Poor utilization
    
    y_positions = range(len(truck_ids))
    bars = ax1.barh(y_positions, utilization_percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize first subplot
    ax1.set_xlabel('Capacity Utilization (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Truck ID', fontsize=12, fontweight='bold')
    ax1.set_title('Truck Capacity Utilization Analysis', fontsize=14, fontweight='bold')
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([f'Truck {tid}' for tid in truck_ids])
    
    # Add percentage labels on bars
    for i, (bar, util, used_vol, capacity) in enumerate(zip(bars, utilization_percentages, used_volumes, capacities)):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{util:.1f}% ({used_vol:.1f}/{capacity:.1f}m³)', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Add reference lines
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax1.axvline(x=75, color='orange', linestyle='--', alpha=0.7, label='75% threshold')
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.7, label='90% threshold')
    
    ax1.set_xlim(0, 105)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend(loc='lower right')
    
    # Plot 2: Volume breakdown stacked bar chart
    unused_volumes = [cap - used for cap, used in zip(capacities, used_volumes)]
    
    ax2.barh(y_positions, used_volumes, color='steelblue', alpha=0.8, label='Used Volume', edgecolor='black')
    ax2.barh(y_positions, unused_volumes, left=used_volumes, color='lightgray', alpha=0.8, 
            label='Unused Volume', edgecolor='black')
    
    # Customize second subplot
    ax2.set_xlabel('Volume (m³)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Truck ID', fontsize=12, fontweight='bold')
    ax2.set_title('Volume Usage Breakdown', fontsize=14, fontweight='bold')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f'Truck {tid}' for tid in truck_ids])
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add volume labels
    for i, (used_vol, capacity) in enumerate(zip(used_volumes, capacities)):
        ax2.text(capacity + max(capacities) * 0.01, i,
                f'{capacity:.1f}m³ total', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Calculate and display summary statistics
    avg_utilization = np.mean(utilization_percentages)
    total_used = sum(used_volumes)
    total_capacity = sum(capacities)
    overall_utilization = (total_used / total_capacity) * 100 if total_capacity > 0 else 0
    
    # Add summary text box
    summary_text = (f"Utilization Summary:\n"
                   f"• Average: {avg_utilization:.1f}%\n"
                   f"• Overall: {overall_utilization:.1f}%\n"
                   f"• Total Used: {total_used:.1f}m³\n"
                   f"• Total Capacity: {total_capacity:.1f}m³")
    
    fig.text(0.02, 0.02, summary_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Add efficiency recommendations
    recommendations = []
    if avg_utilization < 50:
        recommendations.append("• Consider using fewer, larger trucks")
    elif avg_utilization > 95:
        recommendations.append("• Excellent utilization achieved")
    else:
        recommendations.append("• Good utilization balance")
    
    if len(set(utilization_percentages)) > 1:
        min_util_truck = truck_ids[utilization_percentages.index(min(utilization_percentages))]
        recommendations.append(f"• Truck {min_util_truck} has lowest utilization")
    
    if recommendations:
        rec_text = "Recommendations:\n" + "\n".join(recommendations)
        fig.text(0.98, 0.02, rec_text, fontsize=10, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Utilization plot saved to {save_path}")
    
    logger.info("Capacity utilization plot completed")
    return fig


def create_comprehensive_dashboard(solution: Dict[str, Any], orders_df: pd.DataFrame, 
                                 trucks_df: pd.DataFrame, distance_matrix: pd.DataFrame = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive dashboard combining all visualization types
    
    Generates a multi-panel dashboard showing:
    - Route visualization
    - Cost analysis
    - Capacity utilization
    - Summary statistics and key metrics
    
    Args:
        solution (Dict[str, Any]): Complete solution data from optimizer
        orders_df (pd.DataFrame): Orders data
        trucks_df (pd.DataFrame): Trucks data
        distance_matrix (pd.DataFrame, optional): Distance matrix
        save_path (Optional[str]): Path to save the dashboard image
        
    Returns:
        plt.Figure: The matplotlib figure object with complete dashboard
    """
    logger.info("Creating comprehensive visualization dashboard...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Define subplot layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 1, 1])
    
    # Route plot (large, spans two rows on left)
    ax_routes = fig.add_subplot(gs[:, 0])
    
    # Cost plot (top right)
    ax_costs = fig.add_subplot(gs[0, 1])
    
    # Utilization plot (top far right)
    ax_util = fig.add_subplot(gs[0, 2])
    
    # Summary statistics (bottom right, spans two columns)
    ax_summary = fig.add_subplot(gs[1, 1:])
    
    # Generate individual plots within the dashboard
    try:
        # Routes plot
        plot_routes_in_axis(ax_routes, solution['routes_df'], orders_df, trucks_df)
        
        # Costs plot
        plot_costs_in_axis(ax_costs, solution['costs'], trucks_df)
        
        # Utilization plot
        plot_utilization_in_axis(ax_util, solution['utilization'], trucks_df)
        
        # Summary statistics
        create_summary_panel(ax_summary, solution, orders_df, trucks_df)
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        # Create error message
        fig.text(0.5, 0.5, f'Error creating dashboard: {str(e)}', 
                ha='center', va='center', fontsize=16, color='red')
    
    plt.suptitle('Vehicle Routing Optimization - Complete Analysis Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save dashboard if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
    
    logger.info("Comprehensive dashboard completed")
    return fig


# Helper functions for dashboard creation
def plot_routes_in_axis(ax, routes_df, orders_df, trucks_df):
    """Helper function to create routes plot in specific axis"""
    # Simplified version of plot_routes for dashboard
    postal_coords = {}
    for _, order in orders_df.iterrows():
        postal_code = order['postal_code']
        coord_x = int(postal_code) - int(orders_df['postal_code'].min())
        postal_coords[postal_code] = (coord_x, 0)
    
    # Plot orders
    for _, order in orders_df.iterrows():
        postal_code = order['postal_code']
        x, y = postal_coords[postal_code]
        volume = order['volume']
        size = max(50, volume * 2)
        ax.scatter(x, y, s=size, c='lightblue', edgecolors='navy', alpha=0.7)
        ax.annotate(f"{order['order_id']}", (x, y), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=8)
    
    ax.set_title('Delivery Routes', fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_costs_in_axis(ax, costs, trucks_df):
    """Helper function to create costs plot in specific axis"""
    truck_costs = costs.get('truck_costs', {})
    if truck_costs:
        truck_ids = list(truck_costs.keys())
        cost_values = list(truck_costs.values())
        ax.bar(range(len(truck_ids)), cost_values, alpha=0.8)
        ax.set_xticks(range(len(truck_ids)))
        ax.set_xticklabels([f'T{tid}' for tid in truck_ids])
        ax.set_title('Truck Costs', fontweight='bold')
        ax.set_ylabel('Cost (€)')


def plot_utilization_in_axis(ax, utilization_data, trucks_df):
    """Helper function to create utilization plot in specific axis"""
    if utilization_data:
        truck_ids = list(utilization_data.keys())
        utilizations = [utilization_data[tid]['utilization_percent'] for tid in truck_ids]
        colors = ['green' if u >= 75 else 'orange' if u >= 50 else 'red' for u in utilizations]
        ax.barh(range(len(truck_ids)), utilizations, color=colors, alpha=0.8)
        ax.set_yticks(range(len(truck_ids)))
        ax.set_yticklabels([f'T{tid}' for tid in truck_ids])
        ax.set_title('Utilization %', fontweight='bold')
        ax.set_xlabel('Utilization (%)')


def create_summary_panel(ax, solution, orders_df, trucks_df):
    """Helper function to create summary statistics panel"""
    ax.axis('off')  # Turn off axis for text display
    
    # Calculate summary statistics
    total_cost = solution['costs']['total_cost']
    trucks_used = len(solution['selected_trucks'])
    orders_delivered = len(orders_df)
    avg_utilization = np.mean([u['utilization_percent'] for u in solution['utilization'].values()])
    
    summary_text = f"""
OPTIMIZATION SUMMARY

Selected Trucks: {solution['selected_trucks']}
Total Cost: €{total_cost:.0f}
Trucks Used: {trucks_used}/{len(trucks_df)}
Orders Delivered: {orders_delivered}
Average Utilization: {avg_utilization:.1f}%

EFFICIENCY METRICS
Cost per Order: €{total_cost/orders_delivered:.0f}
Total Volume: {orders_df['volume'].sum():.1f}m³
Total Capacity Used: {sum(u['used_volume'] for u in solution['utilization'].values()):.1f}m³
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))