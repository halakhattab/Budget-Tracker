import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def format_currency(amount):
    """Format amount as currency string"""
    if pd.isna(amount):
        return "$0.00"
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format value as percentage string"""
    if pd.isna(value):
        return "0.0%"
    return f"{value:.1f}%"

def get_date_range_options():
    """Get common date range options for filtering"""
    today = datetime.now().date()
    return {
        'Last 7 days': (today - timedelta(days=7), today),
        'Last 30 days': (today - timedelta(days=30), today),
        'Last 3 months': (today - timedelta(days=90), today),
        'Last 6 months': (today - timedelta(days=180), today),
        'Last year': (today - timedelta(days=365), today),
        'This month': (today.replace(day=1), today),
        'This year': (today.replace(month=1, day=1), today)
    }

def validate_dataframe(df):
    """Validate that dataframe has required columns and data types"""
    required_columns = ['date', 'amount', 'description']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            raise ValueError("Cannot convert 'date' column to datetime")
    
    if not pd.api.types.is_numeric_dtype(df['amount']):
        try:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount'])
        except:
            raise ValueError("Cannot convert 'amount' column to numeric")
    
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("No valid data found after validation")
    
    return df

def generate_summary_stats(df):
    """Generate comprehensive summary statistics"""
    if len(df) == 0:
        return {}
    
    stats = {
        'basic_stats': {
            'total_transactions': len(df),
            'total_amount': df['amount'].sum(),
            'mean_amount': df['amount'].mean(),
            'median_amount': df['amount'].median(),
            'std_amount': df['amount'].std(),
            'min_amount': df['amount'].min(),
            'max_amount': df['amount'].max()
        },
        'date_stats': {
            'date_range_start': df['date'].min(),
            'date_range_end': df['date'].max(),
            'date_span_days': (df['date'].max() - df['date'].min()).days
        },
        'categorical_stats': {}
    }
    
    # Category statistics if available
    if 'category' in df.columns:
        stats['categorical_stats'] = {
            'unique_categories': df['category'].nunique(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'category_spending': df.groupby('category')['amount'].sum().to_dict()
        }
    
    return stats

def detect_anomalies(df, method='iqr', threshold=2.5):
    """Detect anomalous transactions based on amount"""
    if len(df) < 10:  # Need sufficient data for anomaly detection
        return df.copy()
    
    df_with_anomalies = df.copy()
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_with_anomalies['is_anomaly'] = (
            (df['amount'] < lower_bound) | 
            (df['amount'] > upper_bound)
        )
    
    elif method == 'zscore':
        # Z-score method
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        
        if std_amount > 0:
            z_scores = np.abs((df['amount'] - mean_amount) / std_amount)
            df_with_anomalies['is_anomaly'] = z_scores > threshold
        else:
            df_with_anomalies['is_anomaly'] = False
    
    else:
        df_with_anomalies['is_anomaly'] = False
    
    return df_with_anomalies

def calculate_spending_velocity(df):
    """Calculate spending velocity (rate of spending over time)"""
    if len(df) < 2:
        return 0
    
    df_sorted = df.sort_values('date')
    date_span = (df_sorted['date'].iloc[-1] - df_sorted['date'].iloc[0]).days
    
    if date_span == 0:
        return df['amount'].sum()  # All transactions on same day
    
    total_spending = df['amount'].sum()
    velocity = total_spending / date_span  # Amount per day
    
    return velocity

def categorize_transaction_size(amount):
    """Categorize transaction by size"""
    if amount < 10:
        return 'Micro'
    elif amount < 50:
        return 'Small'
    elif amount < 200:
        return 'Medium'
    elif amount < 1000:
        return 'Large'
    else:
        return 'Very Large'

def get_spending_insights_data(df):
    """Extract data for generating spending insights"""
    if len(df) == 0:
        return {}
    
    insights_data = {}
    
    # Time-based insights
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
    
    insights_data['daily_patterns'] = df.groupby('day_of_week')['amount'].agg(['sum', 'mean', 'count']).to_dict()
    insights_data['monthly_patterns'] = df.groupby('month')['amount'].agg(['sum', 'mean', 'count']).to_dict()
    insights_data['weekend_vs_weekday'] = {
        'weekend': {
            'total': df[df['is_weekend']]['amount'].sum(),
            'mean': df[df['is_weekend']]['amount'].mean(),
            'count': df[df['is_weekend']].shape[0]
        },
        'weekday': {
            'total': df[~df['is_weekend']]['amount'].sum(),
            'mean': df[~df['is_weekend']]['amount'].mean(),
            'count': df[~df['is_weekend']].shape[0]
        }
    }
    
    # Amount-based insights
    df['transaction_size'] = df['amount'].apply(categorize_transaction_size)
    insights_data['transaction_sizes'] = df.groupby('transaction_size')['amount'].agg(['sum', 'count']).to_dict()
    
    # Trend analysis
    if len(df) > 7:
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        # Split into first and last thirds for trend analysis
        n = len(df_sorted)
        first_third = df_sorted.iloc[:n//3]
        last_third = df_sorted.iloc[-n//3:]
        
        first_avg = first_third['amount'].mean()
        last_avg = last_third['amount'].mean()
        
        insights_data['trend_analysis'] = {
            'first_period_avg': first_avg,
            'last_period_avg': last_avg,
            'trend_direction': 'increasing' if last_avg > first_avg * 1.1 else 'decreasing' if last_avg < first_avg * 0.9 else 'stable'
        }
    
    # Top merchants/descriptions
    insights_data['top_merchants'] = df.groupby('description')['amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(5).to_dict()
    
    return insights_data

def export_analysis_report(df, ml_results=None):
    """Generate a comprehensive analysis report"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary_statistics': generate_summary_stats(df),
        'spending_insights': get_spending_insights_data(df)
    }
    
    if ml_results:
        report['ml_analysis'] = {
            'cluster_count': len(ml_results.get('cluster_summary', {})),
            'category_suggestions_count': len(ml_results.get('category_suggestions', {})),
            'spending_patterns': ml_results.get('spending_patterns', {})
        }
    
    return report

@st.cache_data
def load_sample_data():
    """Load sample data for testing purposes"""
    # This function would only be used if explicitly requested by user
    # Following the guidelines, we don't generate sample data by default
    return pd.DataFrame()

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def clean_currency_string(currency_str):
    """Clean currency string and extract numeric value"""
    if pd.isna(currency_str):
        return 0
    
    # Remove common currency symbols and formatting
    cleaned = str(currency_str).replace('$', '').replace(',', '').replace('€', '').replace('£', '').strip()
    
    try:
        return float(cleaned)
    except ValueError:
        return 0

def get_color_palette(n_colors):
    """Get a color palette with n colors"""
    import plotly.colors as pc
    
    if n_colors <= 10:
        return pc.qualitative.Set3[:n_colors]
    else:
        # For more colors, interpolate between existing ones
        base_colors = pc.qualitative.Set3
        extended_colors = []
        for i in range(n_colors):
            color_index = i % len(base_colors)
            extended_colors.append(base_colors[color_index])
        return extended_colors
