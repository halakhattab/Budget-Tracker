# Smart Expense Analyzer

## Overview

The Smart Expense Analyzer is a Streamlit-based web application that helps users analyze their spending patterns through intelligent data processing and machine learning insights. The application accepts CSV and PDF files containing expense data, automatically processes and categorizes transactions, and provides comprehensive visualizations and analytics to help users understand their financial habits.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses Streamlit as the primary web framework, providing a reactive single-page application interface. The main application flow is orchestrated through `app.py`, which manages session state for processed data and ML results. The interface features a sidebar for file uploads and filters, with the main content area displaying visualizations and insights.

### Data Processing Pipeline
The system implements a modular data processing architecture through the `DataProcessor` class. This component handles multiple input formats (CSV and PDF) with robust error handling for different encodings and file structures. The processor includes intelligent column mapping to standardize expense data into a consistent format with required fields: date, amount, and description.

### Machine Learning Components
The `MLAnalyzer` class provides automated expense categorization using keyword-based classification and clustering algorithms. The system includes predefined category mappings for common expense types (Food & Dining, Transportation, Healthcare, etc.) and uses TF-IDF vectorization with K-means clustering for pattern recognition in spending behavior.

### Visualization Engine
The `ExpenseVisualizer` class leverages Plotly for interactive data visualizations, including pie charts for category breakdown, time-series analysis for spending trends, and comparative analytics. The visualization system uses a consistent color palette and provides hover interactions with detailed transaction information.

### Data Validation and Utilities
The architecture includes comprehensive data validation through utility functions that ensure data integrity, proper date formatting, and numeric conversion. The system provides helper functions for currency formatting, percentage calculations, and date range filtering options.

## External Dependencies

- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing support
- **Plotly**: Interactive visualization library (plotly.express and plotly.graph_objects)
- **Scikit-learn**: Machine learning algorithms for clustering and text analysis
- **PDFplumber**: PDF text extraction for processing PDF expense reports
- **DateTime**: Built-in Python library for date/time operations

The application is designed to run as a standalone web service without external database dependencies, storing processed data in Streamlit's session state for the duration of user sessions.