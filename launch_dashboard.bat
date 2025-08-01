@echo off
echo Starting Retail Performance Analysis Dashboard...
echo.
conda run --live-stream --name base python data_cleaning.py
echo Dashboard will open in your web browser at http://localhost:8501
echo.
echo To stop the dashboard, press Ctrl+C in this window
echo.
conda run --live-stream --name base streamlit run streamlit_dashboard.py
pause