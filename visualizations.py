import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class ExpenseVisualizer:
    def __init__(self, df):
        self.df = df.copy()
        self.color_palette = px.colors.qualitative.Set3
    
    def create_category_pie_chart(self):
        """Create pie chart for expense categories"""
        if 'category' not in self.df.columns:
            return go.Figure()
        
        category_spending = self.df.groupby('category')['amount'].sum().reset_index()
        
        fig = px.pie(
            category_spending,
            values='amount',
            names='category',
            title='Spending by Category',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                         'Amount: $%{value:,.2f}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            height=400,
            font=dict(size=12)
        )
        
        return fig
    
    def create_monthly_spending_chart(self):
        """Create bar chart for monthly spending"""
        monthly_data = self.df.copy()
        monthly_data['month_year'] = monthly_data['date'].dt.to_period('M').astype(str)
        monthly_spending = monthly_data.groupby('month_year')['amount'].sum().reset_index()
        
        fig = px.bar(
            monthly_spending,
            x='month_year',
            y='amount',
            title='Monthly Spending Trends',
            color='amount',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                         'Total Spending: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            showlegend=False,
            height=400
        )
        
        # Format y-axis as currency
        fig.update_yaxes(tickformat='$,.0f')
        
        return fig
    
    def create_spending_timeline(self):
        """Create timeline chart showing daily spending"""
        daily_spending = self.df.groupby('date')['amount'].sum().reset_index()
        
        fig = px.line(
            daily_spending,
            x='date',
            y='amount',
            title='Daily Spending Timeline',
            markers=True
        )
        
        fig.update_traces(
            line_color='rgb(67, 67, 67)',
            marker_size=4,
            hovertemplate='<b>%{x}</b><br>' +
                         'Daily Spending: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        )
        
        # Add moving average line
        if len(daily_spending) > 7:
            daily_spending['ma_7'] = daily_spending['amount'].rolling(window=7, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=daily_spending['date'],
                    y=daily_spending['ma_7'],
                    mode='lines',
                    name='7-day Average',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='<b>%{x}</b><br>' +
                                 '7-day Average: $%{y:,.2f}<br>' +
                                 '<extra></extra>'
                )
            )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            height=400,
            hovermode='x unified'
        )
        
        fig.update_yaxes(tickformat='$,.0f')
        
        return fig
    
    def create_weekly_pattern(self):
        """Create chart showing spending patterns by day of week"""
        weekly_data = self.df.copy()
        weekly_data['day_of_week'] = weekly_data['date'].dt.day_name()
        weekly_data['day_num'] = weekly_data['date'].dt.dayofweek
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_spending = weekly_data.groupby(['day_of_week', 'day_num'])['amount'].agg(['sum', 'mean', 'count']).reset_index()
        weekly_spending['day_of_week'] = pd.Categorical(weekly_spending['day_of_week'], categories=day_order, ordered=True)
        weekly_spending = weekly_spending.sort_values('day_of_week')
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for total spending
        fig.add_trace(
            go.Bar(
                x=weekly_spending['day_of_week'],
                y=weekly_spending['sum'],
                name='Total Spending',
                marker_color='lightblue',
                yaxis='y',
                hovertemplate='<b>%{x}</b><br>' +
                             'Total: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add line chart for average spending
        fig.add_trace(
            go.Scatter(
                x=weekly_spending['day_of_week'],
                y=weekly_spending['mean'],
                mode='lines+markers',
                name='Average Spending',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>' +
                             'Average: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='Weekly Spending Patterns',
            xaxis_title='Day of Week',
            height=400,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text='Total Spending ($)', tickformat='$,.0f', secondary_y=False)
        fig.update_yaxes(title_text='Average Spending ($)', tickformat='$,.0f', secondary_y=True)
        
        return fig
    
    def create_spending_heatmap(self):
        """Create heatmap showing spending patterns by day and hour"""
        heatmap_data = self.df.copy()
        
        # Extract day of week and hour (if available)
        heatmap_data['day_of_week'] = heatmap_data['date'].dt.day_name()
        
        # If we don't have hour information, use a default or random assignment
        if 'hour' not in heatmap_data.columns:
            # For demonstration, we'll create a pattern based on transaction patterns
            # In real scenarios, this would come from actual timestamp data
            np.random.seed(42)  # For reproducible results
            # Create normalized probabilities that sum to 1.0
            probabilities = np.array([0.05, 0.08, 0.10, 0.15, 0.12, 0.10, 0.08, 0.10, 0.08, 0.08, 0.06, 0.10])
            probabilities = probabilities / probabilities.sum()  # Normalize to sum to 1.0
            
            heatmap_data['hour'] = np.random.choice(
                [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                size=len(heatmap_data),
                p=probabilities
            )
        
        # Group by day and hour
        heatmap_summary = heatmap_data.groupby(['day_of_week', 'hour'])['amount'].sum().reset_index()
        
        # Pivot for heatmap
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = heatmap_summary.pivot(index='day_of_week', columns='hour', values='amount').fillna(0)
        pivot_data = pivot_data.reindex(day_order)
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Spending ($)"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='Viridis',
            title='Spending Heatmap by Day and Hour'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_cluster_visualization(self, df_with_clusters):
        """Create visualization of expense clusters"""
        if 'cluster' not in df_with_clusters.columns:
            return go.Figure()
        
        # Create scatter plot with amount vs day of week, colored by cluster
        cluster_data = df_with_clusters.copy()
        cluster_data['day_of_week'] = cluster_data['date'].dt.dayofweek
        
        fig = px.scatter(
            cluster_data,
            x='day_of_week',
            y='amount',
            color='cluster',
            size='amount',
            hover_data=['description'],
            title='Expense Clusters Analysis',
            labels={'day_of_week': 'Day of Week (0=Monday)', 'amount': 'Amount ($)'},
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            hovertemplate='<b>Cluster %{color}</b><br>' +
                         'Day: %{x}<br>' +
                         'Amount: $%{y:,.2f}<br>' +
                         'Description: %{customdata[0]}<br>' +
                         '<extra></extra>'
        )
        
        # Update x-axis to show day names
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            legend_title="Cluster"
        )
        
        fig.update_yaxes(tickformat='$,.0f')
        
        return fig
    
    def create_top_merchants_chart(self):
        """Create horizontal bar chart for top merchants/descriptions"""
        top_merchants = self.df.groupby('description')['amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
        top_merchants = top_merchants.reset_index()
        
        fig = px.bar(
            top_merchants,
            x='sum',
            y='description',
            orientation='h',
            title='Top 10 Merchants by Total Spending',
            labels={'sum': 'Total Spending ($)', 'description': 'Merchant'},
            color='sum',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>' +
                         'Total Spending: $%{x:,.2f}<br>' +
                         'Transactions: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=top_merchants['count']
        )
        
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        fig.update_xaxes(tickformat='$,.0f')
        
        return fig
    
    def create_amount_distribution(self):
        """Create histogram showing distribution of transaction amounts"""
        fig = px.histogram(
            self.df,
            x='amount',
            nbins=30,
            title='Distribution of Transaction Amounts',
            labels={'amount': 'Transaction Amount ($)', 'count': 'Frequency'},
            color_discrete_sequence=['skyblue']
        )
        
        # Add statistical lines
        mean_amount = self.df['amount'].mean()
        median_amount = self.df['amount'].median()
        
        fig.add_vline(
            x=mean_amount,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_amount:.2f}"
        )
        
        fig.add_vline(
            x=median_amount,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: ${median_amount:.2f}"
        )
        
        fig.update_layout(height=400)
        fig.update_xaxes(tickformat='$,.0f')
        
        return fig
    
    def create_category_trends(self):
        """Create line chart showing category spending trends over time"""
        if 'category' not in self.df.columns:
            return go.Figure()
        
        # Group by month and category
        trends_data = self.df.copy()
        trends_data['month_year'] = trends_data['date'].dt.to_period('M').astype(str)
        category_trends = trends_data.groupby(['month_year', 'category'])['amount'].sum().reset_index()
        
        fig = px.line(
            category_trends,
            x='month_year',
            y='amount',
            color='category',
            title='Category Spending Trends Over Time',
            labels={'month_year': 'Month', 'amount': 'Amount ($)'},
            markers=True,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Amount: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            height=400,
            hovermode='x unified'
        )
        
        fig.update_yaxes(tickformat='$,.0f')
        
        return fig
