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
        """Create a vertical bar chart showing order volumes"""
        if orders_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No order data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create vertical bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=orders_df['order_id'].tolist(),
            y=orders_df['volume'].tolist(),
            text=[f"{vol:.0f}" for vol in orders_df['volume']],
            textposition='outside',
            marker_color='cornflowerblue',
            name='Volume'
        ))
        
        fig.update_layout(
            title='Order Volumes (m³)',
            xaxis_title='Order ID',
            yaxis_title='Volume (m³)',
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, orders_df['volume'].max() * 1.15])
        )
        
        return fig
    
    def create_orders_distribution_chart(self, orders_df: pd.DataFrame):
        """Create a pie chart showing order volume percentage distribution"""
        if orders_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No order data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate percentage distribution
        total_volume = orders_df['volume'].sum()
        percentages = (orders_df['volume'] / total_volume * 100).round(1)
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=orders_df['order_id'].tolist(),
            values=orders_df['volume'].tolist(),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>' +
                         'Volume: %{value:.1f} m³<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#000000', width=1)
            )
        )])
        
        fig.update_layout(
            title='Order Volume Distribution',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_trucks_capacity_chart(self, trucks_df: pd.DataFrame):
        """Create a vertical bar chart showing truck capacities"""
        if trucks_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No truck data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create vertical bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trucks_df['truck_id'].astype(str).tolist(),
            y=trucks_df['capacity'].tolist(),
            text=[f"{cap:.0f}" for cap in trucks_df['capacity']],
            textposition='outside',
            marker_color='cornflowerblue',
            name='Capacity'
        ))
        
        fig.update_layout(
            title='Truck Capacities (m³)',
            xaxis_title='Truck ID',
            yaxis_title='Capacity (m³)',
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, trucks_df['capacity'].max() * 1.15])
        )
        
        return fig
    
    def create_trucks_cost_efficiency_chart(self, trucks_df: pd.DataFrame):
        """Create a vertical bar chart showing cost vs capacity"""
        if trucks_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No truck data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create vertical bar chart showing cost
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trucks_df['truck_id'].astype(str).tolist(),
            y=trucks_df['cost'].tolist(),
            text=[f"€{cost:.0f}" for cost in trucks_df['cost']],
            textposition='outside',
            marker_color='cornflowerblue',
            name='Cost'
        ))
        
        fig.update_layout(
            title='Truck Costs (€)',
            xaxis_title='Truck ID',
            yaxis_title='Cost (€)',
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, trucks_df['cost'].max() * 1.15])
        )
        
        return fig
    
    def create_distance_heatmap(self, distance_matrix: pd.DataFrame):
        """Create a heatmap of the distance matrix"""
        if distance_matrix.empty:
            fig = go.Figure()
            fig.add_annotation(text="No distance data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        try:
            # Convert to numpy array and ensure it's numeric
            z_values = distance_matrix.values.astype(float)
            
            # Create text annotations for each cell
            text_values = []
            for i in range(len(distance_matrix.index)):
                row_text = []
                for j in range(len(distance_matrix.columns)):
                    value = z_values[i, j]
                    if value == 0:
                        row_text.append("0")
                    else:
                        row_text.append(f"{value:.1f}")
                text_values.append(row_text)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=list(distance_matrix.columns),
                y=list(distance_matrix.index),
                colorscale='RdYlBu_r',  # Red-Yellow-Blue reversed (red for high distances)
                text=text_values,
                texttemplate="%{text}km",
                textfont={"size": 12, "color": "black"},
                hoverongaps=False,
                colorbar=dict(title="Distance (km)", titleside="right"),
                showscale=True
            ))
            
            fig.update_layout(
                title='Distance Matrix Between Postal Codes',
                xaxis_title='To Postal Code',
                yaxis_title='From Postal Code',
                width=600,
                height=500,
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed')  # Reverse y-axis to match matrix convention
            )
            
            return fig
            
        except Exception as e:
            # Fallback in case of any errors
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating heatmap: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    

    
    def create_route_visualization(self, routes_df: pd.DataFrame, orders_df: pd.DataFrame, trucks_df: pd.DataFrame):
        """Create individual route plots for each truck with proper route sequences and distance calculation"""
        
        # Get depot location from routes data or use default
        depot_location = '08020'  # Default depot
        if not routes_df.empty:
            # Try to get depot from solution data
            if 'depot_location' in routes_df.columns:
                depot_location = routes_df.iloc[0]['depot_location']
            elif 'route_sequence' in routes_df.columns:
                # Try to find depot from route sequences (usually first and last)
                for _, route in routes_df.iterrows():
                    if isinstance(route['route_sequence'], list) and len(route['route_sequence']) > 0:
                        depot_location = route['route_sequence'][0]
                        break
        
        # Create subplots - one per truck only
        n_trucks = len(routes_df) if not routes_df.empty else 0
        if n_trucks == 0:
            # No routes to display
            fig = go.Figure()
            fig.add_annotation(text="No routes to display", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate total distance for title
        total_distance = sum(route.get('route_distance', 0) for _, route in routes_df.iterrows())
        
        # Create subplot titles with proper distance calculation
        subplot_titles = []
        for _, route in routes_df.iterrows():
            truck_id = route['truck_id']
            n_orders = len(route['assigned_orders'])
            distance = route.get('route_distance', 0)
            subplot_titles.append(f"🚚 Truck {truck_id} - {n_orders} Orders ({distance:.1f} km)")
        
        fig = make_subplots(
            rows=n_trucks, 
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}] for _ in range(n_trucks)]
        )
        
        # Convert postal codes to coordinates for visualization
        postal_coords = {}
        unique_postal_codes = list(set(orders_df['postal_code'].tolist()))
        
        # Add depot to unique postal codes if not already there
        if depot_location not in unique_postal_codes:
            unique_postal_codes.append(depot_location)
        
        # Create coordinate system based on postal code values for better spacing
        sorted_codes = sorted(unique_postal_codes)
        for postal_code in sorted_codes:
            # Use actual postal code differences for proportional spacing
            coord_x = int(postal_code) - int(min(sorted_codes))
            postal_coords[postal_code] = coord_x
        
        # Colors for trucks
        colors = px.colors.qualitative.Set1
        depot_x = postal_coords[depot_location]
        
        # INDIVIDUAL TRUCK PLOTS
        for idx, (_, route) in enumerate(routes_df.iterrows()):
            truck_id = route['truck_id']
            assigned_orders = route['assigned_orders']
            color = colors[idx % len(colors)]
            subplot_row = idx + 1
            route_distance = route.get('route_distance', 0)
            route_sequence = route.get('route_sequence', [])
            
            # Add depot to this truck's subplot
            fig.add_trace(
                go.Scatter(
                    x=[depot_x],
                    y=[0],
                    mode='markers+text',
                    marker=dict(size=35, color='red', symbol='star', line=dict(width=3, color='darkred')),
                    text=f"DEPOT<br>{depot_location}",
                    textposition="bottom center",
                    name="Depot",
                    showlegend=False,
                    hovertemplate=f"<b>Depot</b><br>Location: {depot_location}<extra></extra>"
                ),
                row=subplot_row, col=1
            )
            
            # Add only the orders assigned to this truck
            for order_id in assigned_orders:
                order_row = orders_df[orders_df['order_id'] == order_id]
                if not order_row.empty:
                    order = order_row.iloc[0]
                    postal_code = order['postal_code']
                    x_coord = postal_coords[postal_code]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[x_coord],
                            y=[0],
                            mode='markers+text',
                            marker=dict(size=max(25, order['volume']/1.2), color='lightgreen', line=dict(width=2, color='darkgreen')),
                            text=f"{order_id}<br>{order['volume']}m³",
                            textposition="top center",
                            name=f"Order {order_id}",
                            showlegend=False,
                            hovertemplate=f"<b>Order {order_id}</b><br>Volume: {order['volume']} m³<br>Location: {postal_code}<extra></extra>"
                        ),
                        row=subplot_row, col=1
                    )
            
            # Add the ACTUAL route sequence for this truck
            if route_sequence and isinstance(route_sequence, list) and len(route_sequence) > 1:
                # Get coordinates for the actual route sequence
                route_x_coords = []
                valid_sequence = []
                
                for loc in route_sequence:
                    if loc in postal_coords:
                        route_x_coords.append(postal_coords[loc])
                        valid_sequence.append(loc)
                
                if len(route_x_coords) > 1:
                    # Plot the route line following the EXACT sequence
                    fig.add_trace(
                        go.Scatter(
                            x=route_x_coords,
                            y=[0] * len(route_x_coords),
                            mode='lines',
                            line=dict(width=3, color=color, dash='solid'),  # Reduced line width
                            name=f'Route',
                            showlegend=False,
                            hovertemplate=f'<b>Truck {truck_id} Route</b><br>Distance: {route_distance:.1f} km<br>Sequence: {" → ".join(valid_sequence)}<extra></extra>'
                        ),
                        row=subplot_row, col=1
                    )
                    
                    # Add directional arrows between consecutive points
                    for i in range(len(route_x_coords) - 1):
                        start_x = route_x_coords[i]
                        end_x = route_x_coords[i + 1]
                        
                        # Calculate arrow position (closer to end point)
                        arrow_x = start_x + 0.7 * (end_x - start_x)
                        
                        # Calculate arrow direction and size
                        arrow_dx = 0.3 if end_x > start_x else -0.3
                        
                        fig.add_annotation(
                            x=arrow_x,
                            y=0,
                            ax=arrow_x - arrow_dx,
                            ay=0,
                            axref=f'x{subplot_row}',
                            ayref=f'y{subplot_row}',
                            arrowhead=2,
                            arrowsize=1.5,
                            arrowwidth=3,
                            arrowcolor=color,
                            showarrow=True,
                            row=subplot_row, col=1
                        )
                    
                    # Add route sequence text at the top
                    route_text = " → ".join(valid_sequence)
                    fig.add_annotation(
                        x=sum(route_x_coords) / len(route_x_coords),
                        y=0.5,
                        text=f"<b>Route:</b> {route_text}",
                        showarrow=False,
                        font=dict(size=10, color=color),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=color,
                        borderwidth=1,
                        row=subplot_row, col=1
                    )
            
            # Add truck info box
            truck_info = trucks_df[trucks_df['truck_id'] == truck_id].iloc[0]
            truck_cost = truck_info['cost']
            truck_capacity = truck_info['capacity']
            total_volume = sum(orders_df[orders_df['order_id'].isin(assigned_orders)]['volume'])
            utilization = (total_volume / truck_capacity) * 100 if truck_capacity > 0 else 0
            
            info_text = f"Cost: €{truck_cost:.0f} | Capacity: {truck_capacity:.0f}m³ | Utilization: {utilization:.1f}%"
            fig.add_annotation(
                x=0.02,
                y=0.02,
                text=info_text,
                showarrow=False,
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                xref="paper",
                yref="paper",
                row=subplot_row, col=1
            )
        
        # Update layout for all subplots with total distance in title
        fig.update_layout(
            height=280 * n_trucks,  # Slightly more height for better visibility
            title_text=f"🚛 Vehicle Routes - Total Distance: {total_distance:.1f} km",
            title_x=0.5,
            title_font=dict(size=16, color='darkblue'),
            showlegend=False,
            hovermode='closest'
        )
        
        # Update x and y axes for all subplots
        for i in range(n_trucks):
            row_num = i + 1
            fig.update_xaxes(
                title_text="Postal Code Distance" if row_num == n_trucks else "",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                row=row_num, col=1
            )
            fig.update_yaxes(
                showticklabels=False,
                showgrid=False,
                range=[-0.8, 0.8],
                row=row_num, col=1
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
        
        # Create blue bar chart with values on top
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[str(tid) for tid in truck_ids],
            y=cost_values,
            text=[f"€{cost:.0f}" for cost in cost_values],
            textposition='outside',
            marker_color='cornflowerblue',
            name='Cost'
        ))
        
        fig.update_layout(
            title='Cost Breakdown by Truck',
            xaxis_title='Truck ID',
            yaxis_title='Cost (€)',
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, max(cost_values) * 1.15])
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
        
        # Create horizontal bar chart with blue bars
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=[f'Truck {tid}' for tid in truck_ids],
            x=utilization_pct,
            orientation='h',
            marker_color='cornflowerblue',
            text=[f'{u:.1f}%' for u in utilization_pct],
            textposition='inside',
            name='Utilization'
        ))
        
        fig.update_layout(
            title='Truck Capacity Utilization',
            xaxis_title='Utilization (%)',
            yaxis_title='Truck ID',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_volume_usage_chart(self, utilization: Dict[int, Dict[str, Any]], trucks_df: pd.DataFrame):
        """Create a volume usage breakdown chart"""
        if not utilization:
            fig = go.Figure()
            fig.add_annotation(text="No utilization data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        truck_ids = list(utilization.keys())
        used_volumes = [utilization[tid]['used_volume'] for tid in truck_ids]
        
        # Create vertical bar chart with blue bars and values on top
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'Truck {tid}' for tid in truck_ids],
            y=used_volumes,
            text=[f"{vol:.1f} m³" for vol in used_volumes],
            textposition='outside',
            marker_color='cornflowerblue',
            name='Volume Used'
        ))
        
        fig.update_layout(
            title='Volume Usage Breakdown by Truck',
            xaxis_title='Truck ID',
            yaxis_title='Volume Used (m³)',
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, max(used_volumes) * 1.15] if used_volumes else [0, 1])
        )
        
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