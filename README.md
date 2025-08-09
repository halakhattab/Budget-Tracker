# Budget Tracker

A Streamlit-based web application for tracking, analyzing, and visualizing personal finances.  
Users can upload CSV transaction data to see summaries, spending breakdowns, and insights using machine learning.

## Features
- **CSV Upload** — Import budget or transaction files.
- **Data Processing** — Cleans and structures the data for analysis.
- **Visualizations** — Interactive charts to explore spending trends.
- **Machine Learning Insights** — Detects spending patterns and categories.
- **Sample Dataset** — Includes `BudgetTracker_Sample.csv` for testing.

## Live Demo
Access the live app here: **[Budget Tracker on Streamlit](https://budget-tracker-yfhdqbcbqxfgecxhocr2xn.streamlit.app/)**

## Installation
Clone this repository:
```bash
git clone https://github.com/your-username/budget-tracker.git
cd budget-tracker
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run locally:
```bash
streamlit run app.py
```

## File Overview
- `app.py` — Main Streamlit application.
- `data_processor.py` — Cleans and preprocesses CSV data.
- `ml_analyzer.py` — Runs machine learning models for insights.
- `visualizations.py` — Creates interactive charts and graphs.
- `utils.py` — Helper functions.
- `BudgetTracker_Sample.csv` — Example dataset.
- `requirements.txt` — Python dependencies.

## Deployment
This app is deployed on **Streamlit Community Cloud**.  

## License
MIT License – free to use and modify.
