import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
from collections import Counter
import streamlit as st

class MLAnalyzer:
    def __init__(self):
        self.category_keywords = {
            'Food & Dining': [
                'restaurant', 'cafe', 'coffee', 'food', 'pizza', 'burger', 'dining',
                'mcdonalds', 'starbucks', 'subway', 'kfc', 'taco', 'sushi', 'bar',
                'pub', 'bistro', 'kitchen', 'grill', 'deli', 'bakery', 'donut'
            ],
            'Groceries': [
                'grocery', 'supermarket', 'walmart', 'target', 'kroger', 'safeway',
                'market', 'store', 'shop', 'mart', 'food', 'fresh', 'organic'
            ],
            'Transportation': [
                'gas', 'fuel', 'shell', 'exxon', 'chevron', 'bp', 'uber', 'lyft',
                'taxi', 'bus', 'train', 'metro', 'parking', 'toll', 'car',
                'auto', 'repair', 'service', 'oil change'
            ],
            'Shopping': [
                'amazon', 'ebay', 'shopping', 'retail', 'store', 'mall', 'outlet',
                'department', 'clothing', 'fashion', 'shoes', 'electronics',
                'best buy', 'apple', 'home depot', 'lowes'
            ],
            'Entertainment': [
                'movie', 'theater', 'cinema', 'netflix', 'spotify', 'music',
                'game', 'entertainment', 'ticket', 'event', 'concert', 'show',
                'streaming', 'subscription', 'youtube', 'disney'
            ],
            'Healthcare': [
                'pharmacy', 'doctor', 'hospital', 'medical', 'health', 'dental',
                'clinic', 'cvs', 'walgreens', 'prescription', 'medicine',
                'insurance', 'copay'
            ],
            'Utilities': [
                'electric', 'water', 'gas', 'internet', 'phone', 'cable',
                'utility', 'bill', 'service', 'verizon', 'att', 'comcast',
                'spectrum', 'energy', 'power'
            ],
            'Finance': [
                'bank', 'fee', 'atm', 'interest', 'loan', 'credit', 'payment',
                'transfer', 'finance', 'investment', 'insurance', 'tax'
            ],
            'Travel': [
                'hotel', 'flight', 'airline', 'travel', 'booking', 'airbnb',
                'rental', 'vacation', 'trip', 'airport', 'baggage'
            ],
            'Personal Care': [
                'salon', 'barber', 'spa', 'beauty', 'cosmetics', 'personal',
                'care', 'hygiene', 'haircut', 'massage'
            ]
        }
    
    def analyze_expenses(self, df):
        """Perform comprehensive ML analysis on expense data"""
        if len(df) < 5:
            raise ValueError("Need at least 5 transactions for meaningful analysis")
        
        results = {}
        
        # 1. Feature engineering
        features_df = self._engineer_features(df)
        
        # 2. Clustering analysis
        clustered_data, cluster_info = self._perform_clustering(features_df, df)
        results['data_with_clusters'] = clustered_data
        results['cluster_summary'] = cluster_info
        
        # 3. Category prediction
        category_suggestions = self._predict_categories(df)
        results['category_suggestions'] = category_suggestions
        
        # 4. Spending pattern analysis
        patterns = self._analyze_spending_patterns(df)
        results['spending_patterns'] = patterns
        
        return results
    
    def _engineer_features(self, df):
        """Create features for machine learning"""
        features_df = df.copy()
        
        # Time-based features
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['hour'] = features_df['date'].dt.hour if 'time' in str(features_df['date'].dtype) else 12
        features_df['month'] = features_df['date'].dt.month
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        features_df['log_amount'] = np.log1p(features_df['amount'])
        features_df['amount_quantile'] = pd.qcut(features_df['amount'], q=5, labels=False)
        
        # Description-based features
        features_df['description_length'] = features_df['description'].str.len()
        features_df['has_numbers'] = features_df['description'].str.contains(r'\d').astype(int)
        features_df['word_count'] = features_df['description'].str.split().str.len()
        
        # Clean description for text analysis
        features_df['clean_description'] = features_df['description'].apply(self._clean_text)
        
        return features_df
    
    def _clean_text(self, text):
        """Clean text for better analysis"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _perform_clustering(self, features_df, original_df):
        """Perform K-means clustering on transactions"""
        # Prepare features for clustering
        clustering_features = []
        
        # Numerical features
        numerical_cols = ['log_amount', 'day_of_week', 'month', 'is_weekend', 
                         'description_length', 'word_count']
        
        for col in numerical_cols:
            if col in features_df.columns:
                clustering_features.append(features_df[col])
        
        # Text features using TF-IDF
        if 'clean_description' in features_df.columns:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=50,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                text_features = vectorizer.fit_transform(features_df['clean_description'])
                text_df = pd.DataFrame(
                    text_features.toarray(),
                    columns=[f'text_{i}' for i in range(text_features.shape[1])]
                )
                clustering_features.append(text_df)
            except:
                pass  # Skip text features if TF-IDF fails
        
        # Combine all features
        if clustering_features:
            X = pd.concat(clustering_features, axis=1)
        else:
            # Fallback to basic features
            X = features_df[['log_amount', 'day_of_week']].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        n_clusters = min(8, max(3, len(features_df) // 10))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to original data
        clustered_data = original_df.copy()
        clustered_data['cluster'] = cluster_labels
        
        # Generate cluster summaries
        cluster_summary = self._generate_cluster_summary(clustered_data)
        
        return clustered_data, cluster_summary
    
    def _generate_cluster_summary(self, clustered_data):
        """Generate summary information for each cluster"""
        cluster_summary = {}
        
        for cluster_id in clustered_data['cluster'].unique():
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
            
            # Basic statistics
            avg_amount = cluster_data['amount'].mean()
            count = len(cluster_data)
            
            # Most common day of week
            common_day_num = cluster_data['date'].dt.dayofweek.mode().iloc[0]
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            common_day = day_names[common_day_num]
            
            # Most common descriptions
            common_descriptions = cluster_data['description'].value_counts().head(5).index.tolist()
            
            # Suggest category based on descriptions
            suggested_category = self._suggest_category_for_cluster(cluster_data)
            
            cluster_summary[cluster_id] = {
                'count': count,
                'avg_amount': avg_amount,
                'total_amount': cluster_data['amount'].sum(),
                'common_day': common_day,
                'common_descriptions': common_descriptions,
                'suggested_category': suggested_category,
                'date_range': {
                    'start': cluster_data['date'].min(),
                    'end': cluster_data['date'].max()
                }
            }
        
        return cluster_summary
    
    def _suggest_category_for_cluster(self, cluster_data):
        """Suggest category for a cluster based on transaction descriptions"""
        # Combine all descriptions in the cluster
        all_descriptions = ' '.join(cluster_data['description'].astype(str).str.lower())
        
        # Score each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_descriptions)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'Other' if no match
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'Other'
    
    def _predict_categories(self, df):
        """Predict categories for transactions based on description"""
        category_suggestions = {}
        
        for idx, row in df.iterrows():
            if pd.isna(row.get('category')) or row.get('category') == '':
                description = str(row['description']).lower()
                predicted_category = self._predict_single_category(description)
                
                if predicted_category != 'Other':
                    category_suggestions[row['description']] = predicted_category
        
        return category_suggestions
    
    def _predict_single_category(self, description):
        """Predict category for a single transaction description"""
        description = description.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'Other'
    
    def _analyze_spending_patterns(self, df):
        """Analyze spending patterns and trends"""
        patterns = {}
        
        # Weekly pattern analysis
        weekly_spending = df.groupby(df['date'].dt.day_name())['amount'].mean()
        patterns['weekly_average'] = weekly_spending.to_dict()
        
        # Monthly pattern analysis
        monthly_spending = df.groupby(df['date'].dt.month)['amount'].sum()
        patterns['monthly_totals'] = monthly_spending.to_dict()
        
        # Time-based patterns
        if len(df) > 30:  # Only if we have enough data
            # Trend analysis
            df_sorted = df.sort_values('date')
            df_sorted['cumulative_spending'] = df_sorted['amount'].cumsum()
            
            # Simple trend calculation
            recent_avg = df_sorted.tail(10)['amount'].mean()
            older_avg = df_sorted.head(10)['amount'].mean()
            
            if recent_avg > older_avg * 1.1:
                patterns['trend'] = 'increasing'
            elif recent_avg < older_avg * 0.9:
                patterns['trend'] = 'decreasing'
            else:
                patterns['trend'] = 'stable'
        
        # High-value transaction analysis
        high_value_threshold = df['amount'].quantile(0.9)
        high_value_transactions = df[df['amount'] > high_value_threshold]
        
        patterns['high_value_analysis'] = {
            'threshold': high_value_threshold,
            'count': len(high_value_transactions),
            'total_amount': high_value_transactions['amount'].sum(),
            'common_descriptions': high_value_transactions['description'].value_counts().head(3).to_dict()
        }
        
        return patterns
    
    def get_spending_recommendations(self, analysis_results):
        """Generate spending recommendations based on ML analysis"""
        recommendations = []
        
        if 'cluster_summary' in analysis_results:
            # Find the cluster with highest average spending
            cluster_summary = analysis_results['cluster_summary']
            
            if cluster_summary:
                highest_spending_cluster = max(
                    cluster_summary.items(),
                    key=lambda x: x[1]['avg_amount']
                )[1]
                
                recommendations.append(
                    f"Your highest spending pattern averages "
                    f"${highest_spending_cluster['avg_amount']:.2f} per transaction "
                    f"in the '{highest_spending_cluster['suggested_category']}' category. "
                    f"Consider setting a budget limit for this category."
                )
        
        if 'spending_patterns' in analysis_results:
            patterns = analysis_results['spending_patterns']
            
            if 'trend' in patterns:
                if patterns['trend'] == 'increasing':
                    recommendations.append(
                        "Your spending trend is increasing. "
                        "Review recent transactions to identify areas for cost reduction."
                    )
                elif patterns['trend'] == 'decreasing':
                    recommendations.append(
                        "Great job! Your spending trend is decreasing. "
                        "Keep up the good financial discipline."
                    )
        
        return recommendations
