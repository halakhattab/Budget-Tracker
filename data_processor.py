import pandas as pd
import numpy as np
import re
from datetime import datetime
import streamlit as st
import pdfplumber
from io import StringIO
import tempfile
import os

class DataProcessor:
    def __init__(self):
        self.common_date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%m-%d-%Y', '%d-%m-%Y', '%B %d, %Y', '%b %d, %Y',
            '%d %B %Y', '%d %b %Y', '%Y%m%d'
        ]
        
        self.common_amount_patterns = [
            r'\$?([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)',
            r'\$([0-9,]+\.?[0-9]*)',
            r'([0-9]+,[0-9]{3}\.?[0-9]*)',
            r'([0-9]+\.?[0-9]{2})'
        ]
    
    def process_file(self, uploaded_file):
        """Process uploaded CSV or PDF file and return standardized DataFrame"""
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            return self._process_csv(uploaded_file)
        elif file_type == 'pdf':
            return self._process_pdf(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_csv(self, uploaded_file):
        """Process CSV file and standardize columns"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            # Clean column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Validate and clean data
            df = self._validate_and_clean_data(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing CSV: {str(e)}")
    
    def _process_pdf(self, uploaded_file):
        """Extract transaction data from PDF file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            transactions = []
            
            with pdfplumber.open(tmp_file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        page_transactions = self._extract_transactions_from_text(text)
                        transactions.extend(page_transactions)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if not transactions:
                raise ValueError("No transaction data found in PDF")
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Validate and clean data
            df = self._validate_and_clean_data(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
    
    def _extract_transactions_from_text(self, text):
        """Extract transaction data from PDF text using regex patterns"""
        transactions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for patterns that contain date and amount
            transaction = self._parse_transaction_line(line)
            if transaction:
                transactions.append(transaction)
        
        return transactions
    
    def _parse_transaction_line(self, line):
        """Parse a single line of text to extract transaction details"""
        # Common transaction patterns for bank statements
        patterns = [
            # Pattern: MM/DD/YYYY Description Amount
            r'(\d{1,2}/\d{1,2}/\d{4})\s+(.+?)\s+\$?([0-9,]+\.?\d{0,2})$',
            # Pattern: DD/MM/YYYY Description Amount
            r'(\d{1,2}/\d{1,2}/\d{4})\s+(.+?)\s+([0-9,]+\.?\d{0,2})$',
            # Pattern: YYYY-MM-DD Description Amount
            r'(\d{4}-\d{1,2}-\d{1,2})\s+(.+?)\s+\$?([0-9,]+\.?\d{0,2})$',
            # Pattern: Date Description -Amount (for debits)
            r'(\d{1,2}/\d{1,2}/\d{4})\s+(.+?)\s+-?\$?([0-9,]+\.?\d{0,2})$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                date_str, description, amount_str = match.groups()
                
                # Parse date
                parsed_date = self._parse_date(date_str)
                if not parsed_date:
                    continue
                
                # Parse amount
                amount = self._parse_amount(amount_str)
                if amount is None or amount <= 0:
                    continue
                
                return {
                    'date': parsed_date,
                    'description': description.strip(),
                    'amount': amount
                }
        
        return None
    
    def _standardize_columns(self, df):
        """Standardize column names to expected format"""
        column_mapping = {}
        
        # Date column
        date_candidates = ['date', 'transaction_date', 'trans_date', 'dt', 'timestamp']
        for col in df.columns:
            if any(candidate in col for candidate in date_candidates):
                column_mapping[col] = 'date'
                break
        
        # Amount column
        amount_candidates = ['amount', 'value', 'sum', 'total', 'cost', 'price', 'expense']
        for col in df.columns:
            if any(candidate in col for candidate in amount_candidates):
                column_mapping[col] = 'amount'
                break
        
        # Description column
        desc_candidates = ['description', 'desc', 'merchant', 'vendor', 'payee', 'transaction', 'details']
        for col in df.columns:
            if any(candidate in col for candidate in desc_candidates):
                column_mapping[col] = 'description'
                break
        
        # Category column
        cat_candidates = ['category', 'cat', 'type', 'classification']
        for col in df.columns:
            if any(candidate in col for candidate in cat_candidates):
                column_mapping[col] = 'category'
                break
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        if 'date' not in df.columns:
            raise ValueError("Could not identify date column. Please ensure your CSV has a date column.")
        
        if 'amount' not in df.columns:
            raise ValueError("Could not identify amount column. Please ensure your CSV has an amount column.")
        
        if 'description' not in df.columns:
            # Create description from available columns
            desc_cols = [col for col in df.columns if col not in ['date', 'amount', 'category']]
            if desc_cols:
                df['description'] = df[desc_cols].apply(
                    lambda x: ' '.join(x.astype(str)), axis=1
                )
            else:
                df['description'] = 'Transaction'
        
        return df
    
    def _validate_and_clean_data(self, df):
        """Validate and clean the transaction data"""
        original_length = len(df)
        
        # Remove empty rows
        df = df.dropna(subset=['date', 'amount'])
        
        # Parse dates
        df['date'] = df['date'].apply(self._parse_date)
        df = df.dropna(subset=['date'])
        
        # Parse amounts
        df['amount'] = df['amount'].apply(self._parse_amount)
        df = df[df['amount'] > 0]  # Remove negative or zero amounts
        
        # Clean descriptions
        df['description'] = df['description'].astype(str).str.strip()
        df = df[df['description'] != '']
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'amount', 'description'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Log data cleaning results
        cleaned_length = len(df)
        if original_length != cleaned_length:
            st.info(f"Data cleaned: {original_length} → {cleaned_length} transactions")
        
        if len(df) == 0:
            raise ValueError("No valid transactions found after data cleaning")
        
        return df
    
    def _parse_date(self, date_str):
        """Parse date string using multiple formats"""
        if pd.isna(date_str):
            return None
        
        # If already a datetime object
        if isinstance(date_str, datetime):
            return date_str
        
        date_str = str(date_str).strip()
        
        # Try each date format
        for fmt in self.common_date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try pandas date parser as last resort
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    def _parse_amount(self, amount_str):
        """Parse amount string to float"""
        if pd.isna(amount_str):
            return None
        
        # If already a number
        if isinstance(amount_str, (int, float)):
            return abs(float(amount_str))
        
        amount_str = str(amount_str).strip()
        
        # Remove currency symbols and commas
        cleaned_amount = re.sub(r'[$,€£¥]', '', amount_str)
        cleaned_amount = cleaned_amount.replace('(', '-').replace(')', '')
        
        # Try to extract number
        for pattern in self.common_amount_patterns:
            match = re.search(pattern, cleaned_amount)
            if match:
                try:
                    amount = float(match.group(1).replace(',', ''))
                    return abs(amount)  # Return absolute value
                except ValueError:
                    continue
        
        # Direct float conversion as last resort
        try:
            return abs(float(cleaned_amount))
        except ValueError:
            return None
    
    def get_data_summary(self, df):
        """Generate summary statistics of the processed data"""
        if df is None or len(df) == 0:
            return {}
        
        return {
            'total_transactions': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'total_amount': df['amount'].sum(),
            'average_amount': df['amount'].mean(),
            'median_amount': df['amount'].median(),
            'unique_descriptions': df['description'].nunique(),
            'has_categories': 'category' in df.columns
        }
