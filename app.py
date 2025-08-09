import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from ml_analyzer import MLAnalyzer
from visualizations import ExpenseVisualizer
from utils import format_currency, get_date_range_options

# Configure page
st.set_page_config(page_title="Expense Analyzer",
                   page_icon="💰",
                   layout="wide",
                   initial_sidebar_state="expanded")


def main():
    st.title("💰 Smart Expense Analyzer")
    st.markdown(
        "Upload your expense data and get intelligent insights about your spending patterns!"
    )

    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None

    # Sidebar for file upload and filters
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'pdf'],
            help="Upload CSV or PDF files containing expense data")

        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                try:
                    processor = DataProcessor()
                    df = processor.process_file(uploaded_file)
                    st.session_state.processed_data = df
                    st.success(f"✅ Processed {len(df)} transactions")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.session_state.processed_data = None

    # Main content
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data

        # Sidebar filters
        with st.sidebar:
            st.header("🔍 Filters")

            # Date range filter
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            date_range = st.date_input("Date Range",
                                       value=(min_date, max_date),
                                       min_value=min_date,
                                       max_value=max_date)

            # Amount range filter
            min_amount = float(df['amount'].min())
            max_amount = float(df['amount'].max())

            amount_range = st.slider("Amount Range",
                                     min_value=min_amount,
                                     max_value=max_amount,
                                     value=(min_amount, max_amount),
                                     format="$%.2f")

            # Category filter
            if 'category' in df.columns:
                categories = df['category'].unique().tolist()
                selected_categories = st.multiselect("Categories",
                                                     options=categories,
                                                     default=categories)
            else:
                selected_categories = []

        # Apply filters
        filtered_df = df.copy()

        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= start_date)
                & (filtered_df['date'].dt.date <= end_date)]

        filtered_df = filtered_df[(filtered_df['amount'] >= amount_range[0])
                                  & (filtered_df['amount'] <= amount_range[1])]

        if selected_categories and 'category' in df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(
                selected_categories)]

        # Main dashboard
        col1, col2, col3, col4 = st.columns(
            [1.2, 1, 1, 1])  # Give more space to total expenses

        with col1:
            total_expenses = filtered_df['amount'].sum()
            st.metric("Total Expenses", format_currency(total_expenses))

        with col2:
            avg_transaction = filtered_df['amount'].mean()
            st.metric("Avg Transaction", format_currency(avg_transaction))

        with col3:
            transaction_count = len(filtered_df)
            st.metric("Transactions", transaction_count)

        with col4:
            if len(filtered_df) > 0:
                date_span = (filtered_df['date'].max() -
                             filtered_df['date'].min()).days
                daily_avg = total_expenses / max(date_span, 1)
                st.metric("Daily Average", format_currency(daily_avg))

        # Visualizations
        visualizer = ExpenseVisualizer(filtered_df)

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Overview", "🤖 ML Analysis", "📈 Trends", "💡 Insights"])

        with tab1:
            st.subheader("Spending Overview")

            col1, col2 = st.columns(2)

            with col1:
                # Category pie chart
                if 'category' in filtered_df.columns:
                    fig_pie = visualizer.create_category_pie_chart()
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Categories will be available after ML analysis")

            with col2:
                # Monthly spending bar chart
                fig_monthly = visualizer.create_monthly_spending_chart()
                st.plotly_chart(fig_monthly, use_container_width=True)

            # Transaction timeline
            st.subheader("Transaction Timeline")
            fig_timeline = visualizer.create_spending_timeline()
            st.plotly_chart(fig_timeline, use_container_width=True)

        with tab2:
            st.subheader("🤖 Machine Learning Analysis")

            if st.button("Run ML Analysis", type="primary"):
                with st.spinner("Analyzing spending patterns..."):
                    try:
                        ml_analyzer = MLAnalyzer()
                        results = ml_analyzer.analyze_expenses(df)
                        st.session_state.ml_results = results
                        st.success("✅ ML Analysis Complete!")
                    except Exception as e:
                        st.error(f"❌ ML Analysis failed: {str(e)}")

            if st.session_state.ml_results is not None:
                results = st.session_state.ml_results
                df_with_clusters = results['data_with_clusters']

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Spending Clusters")
                    fig_clusters = visualizer.create_cluster_visualization(
                        df_with_clusters)
                    st.plotly_chart(fig_clusters, use_container_width=True)

                with col2:
                    st.subheader("Cluster Summary")
                    cluster_summary = results['cluster_summary']
                    for cluster_id, summary in cluster_summary.items():
                        with st.expander(
                                f"Cluster {cluster_id} ({summary['count']} transactions)"
                        ):
                            st.write(
                                f"**Average Amount:** {format_currency(summary['avg_amount'])}"
                            )
                            st.write(
                                f"**Most Common Day:** {summary['common_day']}"
                            )
                            st.write(
                                f"**Suggested Category:** {summary['suggested_category']}"
                            )

                            if summary['common_descriptions']:
                                st.write("**Common Descriptions:**")
                                for desc in summary['common_descriptions'][:5]:
                                    st.write(f"• {desc}")

                # Category suggestions
                st.subheader("Smart Category Suggestions")
                category_suggestions = results.get('category_suggestions', {})

                if category_suggestions:
                    for original_desc, suggested_category in list(
                            category_suggestions.items())[:10]:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(original_desc[:50] +
                                     "..." if len(original_desc) >
                                     50 else original_desc)
                        with col2:
                            st.write(f"→ **{suggested_category}**")
                        with col3:
                            if st.button("Apply",
                                         key=f"apply_{hash(original_desc)}"):
                                # Update the dataframe with suggested category
                                mask = df['description'].str.contains(
                                    original_desc, case=False, na=False)
                                df.loc[mask, 'category'] = suggested_category
                                st.session_state.processed_data = df
                                st.rerun()

        with tab3:
            st.subheader("📈 Spending Trends")

            # Weekly spending pattern
            fig_weekly = visualizer.create_weekly_pattern()
            st.plotly_chart(fig_weekly, use_container_width=True)

            # Daily spending heatmap
            fig_heatmap = visualizer.create_spending_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Top merchants/vendors
            if 'description' in filtered_df.columns:
                st.subheader("Top Merchants")
                top_merchants = filtered_df.groupby(
                    'description')['amount'].agg(['sum', 'count']).sort_values(
                        'sum', ascending=False).head(10)

                for idx, (merchant,
                          data) in enumerate(top_merchants.iterrows(), 1):
                    col1, col2, col3 = st.columns([1, 3, 2])
                    with col1:
                        st.write(f"#{idx}")
                    with col2:
                        st.write(merchant[:40] +
                                 "..." if len(merchant) > 40 else merchant)
                    with col3:
                        st.write(
                            f"{format_currency(data['sum'])} ({data['count']} transactions)"
                        )

        with tab4:
            st.subheader("💡 Spending Insights & Recommendations")

            # Generate insights
            insights = generate_spending_insights(filtered_df)

            for insight in insights:
                st.info(insight)

            # Spending goals section
            st.subheader("🎯 Set Spending Goals")

            col1, col2 = st.columns(2)

            with col1:
                monthly_goal = st.number_input(
                    "Monthly Spending Goal",
                    min_value=0.0,
                    value=float(filtered_df['amount'].sum()),
                    format="%.2f")

            with col2:
                current_month_spending = filtered_df[
                    filtered_df['date'].dt.month ==
                    datetime.now().month]['amount'].sum()

                if current_month_spending > monthly_goal:
                    st.error(
                        f"⚠️ Over budget by {format_currency(current_month_spending - monthly_goal)}"
                    )
                else:
                    st.success(
                        f"✅ Under budget by {format_currency(monthly_goal - current_month_spending)}"
                    )

        # Data table
        st.subheader("📋 Transaction Data")

        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_all = st.checkbox("Show all transactions", value=False)
        with col2:
            sort_by = st.selectbox("Sort by",
                                   ["date", "amount", "description"])
        with col3:
            sort_order = st.selectbox("Order", ["Descending", "Ascending"])

        # Sort data
        ascending = sort_order == "Ascending"
        display_df = filtered_df.sort_values(sort_by, ascending=ascending)

        if not show_all:
            display_df = display_df.head(100)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Export options
        st.subheader("📤 Export Data")
        col1, col2 = st.columns(2)

        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv")

        with col2:
            if st.session_state.ml_results is not None:
                results_csv = st.session_state.ml_results[
                    'data_with_clusters'].to_csv(index=False)
                st.download_button(
                    "Download with ML Results",
                    results_csv,
                    file_name=
                    f"expenses_analyzed_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

    else:
        # Landing page
        st.info("👆 Please upload a CSV or PDF file to get started")

        # Instructions
        with st.expander("📝 File Format Instructions"):
            st.subheader("CSV Format")
            st.write("Your CSV should contain columns for:")
            st.write("• Date (various formats supported)")
            st.write("• Amount (numerical values)")
            st.write("• Description (transaction details)")
            st.write("• Category (optional)")

            st.subheader("PDF Format")
            st.write("PDF files should contain:")
            st.write("• Bank statements or expense reports")
            st.write("• Clear transaction data with dates and amounts")
            st.write("• Text-based content (not scanned images)")

        # Sample data structure
        with st.expander("📊 Sample Data Structure"):
            sample_data = pd.DataFrame({
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'amount': [25.50, 150.00, 8.99],
                'description':
                ['Coffee Shop', 'Grocery Store', 'Online Subscription'],
                'category': ['Food', 'Groceries', 'Entertainment']
            })
            st.dataframe(sample_data, use_container_width=True)


def generate_spending_insights(df):
    """Generate actionable spending insights"""
    insights = []

    if len(df) == 0:
        return ["No data available for analysis."]

    # Analyze spending patterns
    total_spending = df['amount'].sum()
    avg_transaction = df['amount'].mean()

    # High spending transactions
    high_spending_threshold = df['amount'].quantile(0.9)
    high_spending_count = (df['amount'] > high_spending_threshold).sum()

    if high_spending_count > 0:
        high_spending_total = df[df['amount'] >
                                 high_spending_threshold]['amount'].sum()
        high_spending_pct = (high_spending_total / total_spending) * 100
        insights.append(
            f"💸 Your top 10% of transactions account for {high_spending_pct:.1f}% "
            f"({format_currency(high_spending_total)}) of total spending. "
            "Consider reviewing these larger expenses for potential savings.")

    # Frequent small transactions
    small_transaction_threshold = avg_transaction * 0.3
    small_transactions = df[df['amount'] < small_transaction_threshold]

    if len(small_transactions) > len(df) * 0.3:
        small_total = small_transactions['amount'].sum()
        small_threshold_formatted = format_currency(
            small_transaction_threshold)
        small_total_formatted = format_currency(small_total)
        insights.append(
            f"🔍 You have many small transactions (under {small_threshold_formatted})"
            f"totaling {small_total_formatted}. These micro-expenses can add up significantly."
        )

    # Weekend vs weekday spending
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
    weekend_avg = df[df['is_weekend']]['amount'].mean()
    weekday_avg = df[~df['is_weekend']]['amount'].mean()

    if weekend_avg > weekday_avg * 1.2:
        weekend_formatted = format_currency(weekend_avg)
        weekday_formatted = format_currency(weekday_avg)
        insights.append(
            f"🎉 Your weekend spending ({weekend_formatted} avg) is significantly higher than weekday spending ({weekday_formatted} avg). Consider planning weekend activities with a budget in mind."
        )

    # Monthly trend
    if len(df) > 30:
        df['month'] = df['date'].dt.to_period('M')
        monthly_spending = df.groupby('month')['amount'].sum()

        if len(monthly_spending) > 1:
            latest_month = monthly_spending.iloc[-1]
            previous_month = monthly_spending.iloc[-2]

            if latest_month > previous_month * 1.1:
                increase = latest_month - previous_month
                insights.append(
                    f"📈 Your spending increased by {format_currency(increase)} "
                    f"({((latest_month/previous_month - 1) * 100):.1f}%) from last month. "
                    "Review recent expenses to identify the cause.")
            elif latest_month < previous_month * 0.9:
                decrease = previous_month - latest_month
                insights.append(
                    f"📉 Great job! You reduced spending by {format_currency(decrease)} "
                    f"({((1 - latest_month/previous_month) * 100):.1f}%) from last month."
                )

    # Category insights (if available)
    if 'category' in df.columns:
        category_spending = df.groupby('category')['amount'].sum().sort_values(
            ascending=False)
        top_category = category_spending.index[0]
        top_category_amount = category_spending.iloc[0]
        top_category_pct = (top_category_amount / total_spending) * 100

        insights.append(
            f"🏆 Your highest spending category is '{top_category}' with "
            f"{format_currency(top_category_amount)} ({top_category_pct:.1f}% of total spending)."
        )

    if not insights:
        insights.append(
            "📊 Upload more transaction data to get personalized insights!")

    return insights


if __name__ == "__main__":
    main()
