"""
Visualization Manager Module

This module provides the VisualizationManager class for creating
interactive charts and visualizations using Plotly in the Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional


class VisualizationManager:
    """
    Visualization Manager for Streamlit Application
    
    This class creates interactive charts and visualizations for the
    Vehicle Router Streamlit application using Plotly.
    """
    
    def __init__(self):
        """Initialize the VisualizationManager"""
        self.color_palette = px.colors.qualitative.Set1
    
    def create_orders_volume_chart(self, orders_df: pd.DataFrame):
        """Create a bar chart showing order volumes"""
        fig = px.bar(
            orders_df, 
            x='order_id', 
            y='volume',
            title='Order Volumes',
            labels={'volume': 'Volume (m³)', 'order_id': 'Order ID'},
            color='volume',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def create_orders_location_chart(self, orders_df: pd.DataFrame):
        """Create a scatter plot showing order locations"""
        # Convert postal codes to numeric for plotting
        orders_df_plot = orders_df.copy()
        orders_df_plot['postal_numeric'] = orders_df_plot['postal_code'].astype(str).str.extract('(\d+)').astype(int)
        
        fig = px.scatter(
            orders_df_plot,
            x='postal_numeric',
            y=[1] * len(orders_df_plot),  # All on same y-level
            size='volume',
            hover_data=['order_id', 'volume', 'postal_code'],
            title='Order Locations',
            labels={'postal_numeric': 'Postal Code', 'y': ''},
            size_max=30
        )
        fig.update_layout(
            yaxis=dict(showticklabels=False, showgrid=False),
            showlegend=False
        )
        return fig
    
    def create_trucks_capacity_chart(self, trucks_df: pd.DataFrame):
        """Create a bar chart showing truck capacities"""
        fig = px.bar(
            trucks_df,
            x='truck_id',
            y='capacity',
            title='Truck Capacities',
            labels={'capacity': 'Capacity (m³)', 'truck_id': 'Truck ID'},
            color='capacity',
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def create_trucks_cost_efficiency_chart(self, trucks_df: pd.DataFrame):
        """Create a scatter plot showing cost vs capacity efficiency"""
        trucks_df_plot = trucks_df.copy()
        trucks_df_plot['cost_per_m3'] = trucks_df_plot['cost'] / trucks_df_plot['capacity']
        
        fig = px.scatter(
            trucks_df_plot,
            x='capacity',
            y='cost',
            size='cost_per_m3',
            hover_data=['truck_id', 'cost_per_m3'],
            title='Truck Cost vs Capacity',
            labels={'capacity': 'Capacity (m³)', 'cost': 'Cost (€)'},
            size_max=20
        )
        return fig
    
    def create_distance_heatmap(self, distance_matrix: pd.DataFrame):
        """Create a heatmap of the distance matrix"""
        fig = px.imshow(
            distance_matrix.values,
            x=distance_matrix.columns,
            y=distance_matrix.index,
            title='Distance Matrix Heatmap',
            labels={'x': 'To Postal Code', 'y': 'From Postal Code', 'color': 'Distance (km)'},
            color_continuous_scale='Viridis'
        )
        return fig
    
    def create_route_visualization(self, routes_df: pd.DataFrame, orders_df: pd.DataFrame, trucks_df: pd.DataFrame):
        """Create a route visualization plot"""
        fig = go.Figure()
        
        # Convert postal codes to coordinates
        postal_coords = {}
        for _, order in orders_df.iterrows():
            postal_code = order['postal_code']
            coord_x = int(postal_code) - int(orders_df['postal_code'].min())
            postal_coords[postal_code] = coord_x
        
        # Plot order locations
        for _, order in orders_df.iterrows():
            postal_code = order['postal_code']
            x_coord = postal_coords[postal_code]
            
            fig.add_trace(go.Scatter(
                x=[x_coord],
                y=[0],
                mode='markers+text',
                marker=dict(size=max(10, order['volume']), color='lightblue', line=dict(width=2, color='navy')),
                text=f"Order {order['order_id']}<br>{order['volume']}m³",
                textposition="top center",
                name=f"Order {order['order_id']}",
                showlegend=False
            ))
        
        # Plot truck routes
        colors = px.colors.qualitative.Set1
        for idx, (_, route) in enumerate(routes_df.iterrows()):
            truck_id = route['truck_id']
            assigned_orders = route['assigned_orders']
            
            if len(assigned_orders) > 1:
                # Get coordinates for route
                route_coords = []
                for order_id in assigned_orders:
                    order_postal = orders_df[orders_df['order_id'] == order_id]['postal_code'].iloc[0]
                    route_coords.append(postal_coords[order_postal])
                
                # Plot route line
                fig.add_trace(go.Scatter(
                    x=route_coords,
                    y=[0] * len(route_coords),
                    mode='lines+markers',
                    line=dict(width=3, color=colors[idx % len(colors)]),
                    marker=dict(size=8),
                    name=f'Truck {truck_id}',
                    showlegend=True
                ))
        
        fig.update_layout(
            title='Delivery Routes Visualization',
            xaxis_title='Postal Code Distance (km)',
            yaxis=dict(showticklabels=False, showgrid=False),
            height=400
        )
        
        return fig
    
    def create_cost_breakdown_chart(self, costs: Dict[str, Any], trucks_df: pd.DataFrame):
        """Create a cost breakdown chart"""
        truck_costs = costs.get('truck_costs', {})
        
        if not truck_costs:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(text="No cost data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        truck_ids = list(truck_costs.keys())
        cost_values = list(truck_costs.values())
        
        fig = px.bar(
            x=truck_ids,
            y=cost_values,
            title='Cost Breakdown by Truck',
            labels={'x': 'Truck ID', 'y': 'Cost (€)'},
            color=cost_values,
            color_continuous_scale='Reds'
        )
        
        # Add total cost annotation
        total_cost = sum(cost_values)
        fig.add_annotation(
            text=f"Total Cost: €{total_cost:.0f}",
            x=0.5, y=0.95,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def create_cost_efficiency_chart(self, costs: Dict[str, Any], utilization: Dict[int, Dict[str, Any]], trucks_df: pd.DataFrame):
        """Create a cost efficiency analysis chart"""
        truck_costs = costs.get('truck_costs', {})
        
        if not truck_costs or not utilization:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Prepare data
        truck_ids = []
        cost_per_m3 = []
        utilization_pct = []
        
        for truck_id in truck_costs.keys():
            if truck_id in utilization:
                truck_ids.append(truck_id)
                used_volume = utilization[truck_id]['used_volume']
                cost = truck_costs[truck_id]
                cost_per_m3.append(cost / used_volume if used_volume > 0 else 0)
                utilization_pct.append(utilization[truck_id]['utilization_percent'])
        
        fig = px.scatter(
            x=utilization_pct,
            y=cost_per_m3,
            size=[10] * len(truck_ids),
            hover_data={'Truck ID': truck_ids},
            title='Cost Efficiency vs Utilization',
            labels={'x': 'Utilization (%)', 'y': 'Cost per m³ (€/m³)'}
        )
        
        return fig
    
    def create_utilization_chart(self, utilization: Dict[int, Dict[str, Any]], trucks_df: pd.DataFrame):
        """Create a utilization analysis chart"""
        if not utilization:
            fig = go.Figure()
            fig.add_annotation(text="No utilization data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        truck_ids = list(utilization.keys())
        utilization_pct = [utilization[tid]['utilization_percent'] for tid in truck_ids]
        used_volumes = [utilization[tid]['used_volume'] for tid in truck_ids]
        capacities = [utilization[tid]['capacity'] for tid in truck_ids]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add utilization bars
        colors = ['green' if u >= 75 else 'orange' if u >= 50 else 'red' for u in utilization_pct]
        
        fig.add_trace(go.Bar(
            y=[f'Truck {tid}' for tid in truck_ids],
            x=utilization_pct,
            orientation='h',
            marker_color=colors,
            text=[f'{u:.1f}%' for u in utilization_pct],
            textposition='inside',
            name='Utilization'
        ))
        
        fig.update_layout(
            title='Truck Capacity Utilization',
            xaxis_title='Utilization (%)',
            yaxis_title='Truck ID',
            showlegend=False
        )
        
        # Add reference lines
        fig.add_vline(x=50, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_vline(x=75, line_dash="dash", line_color="orange", opacity=0.7)
        fig.add_vline(x=90, line_dash="dash", line_color="green", opacity=0.7)
        
        return fig
    
    def create_performance_dashboard(self, solution: Dict[str, Any], orders_df: pd.DataFrame, trucks_df: pd.DataFrame):
        """Create a comprehensive performance dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cost Breakdown', 'Utilization Analysis', 'Order Distribution', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # Cost breakdown (top left)
        truck_costs = solution['costs'].get('truck_costs', {})
        if truck_costs:
            fig.add_trace(
                go.Bar(x=list(truck_costs.keys()), y=list(truck_costs.values()), name="Cost"),
                row=1, col=1
            )
        
        # Utilization (top right)
        if solution['utilization']:
            truck_ids = list(solution['utilization'].keys())
            utilizations = [solution['utilization'][tid]['utilization_percent'] for tid in truck_ids]
            fig.add_trace(
                go.Bar(x=truck_ids, y=utilizations, name="Utilization"),
                row=1, col=2
            )
        
        # Order distribution (bottom left)
        volume_by_truck = {}
        for truck_id in solution['selected_trucks']:
            assigned_orders = [a['order_id'] for a in solution['assignments_df'].to_dict('records') 
                             if a['truck_id'] == truck_id]
            total_volume = orders_df[orders_df['order_id'].isin(assigned_orders)]['volume'].sum()
            volume_by_truck[f'Truck {truck_id}'] = total_volume
        
        if volume_by_truck:
            fig.add_trace(
                go.Pie(labels=list(volume_by_truck.keys()), values=list(volume_by_truck.values()), name="Volume"),
                row=2, col=1
            )
        
        # Performance indicator (bottom right)
        avg_util = np.mean([u['utilization_percent'] for u in solution['utilization'].values()]) if solution['utilization'] else 0
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_util,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Utilization (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Performance Dashboard")
        return fig