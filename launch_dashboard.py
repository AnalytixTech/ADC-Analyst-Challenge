#!/usr/bin/env python3
"""
Simple launcher for the Streamlit dashboard
"""
import subprocess
import sys
import webbrowser
import time
import os

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    
    print("🚀 Starting Retail Performance Analysis Dashboard...")
    print("📊 Loading interactive visualizations...")
    print()
    
    try:
        # Change to the correct directory
        os.chdir(r"c:\Users\ToyeebKazeem\Documents\ADC\Analyst Challenge")
        
        # Start the Streamlit app
        print("🌐 Dashboard will open at: http://localhost:8501")
        print("⏹️  To stop the dashboard, press Ctrl+C")
        print()
        
        # Launch Streamlit
        subprocess.run([
            "conda", "run", "--live-stream", "--name", "base", 
            "streamlit", "run", "streamlit_dashboard.py"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n🔧 Try running this command manually:")
        print('conda run --live-stream --name base streamlit run streamlit_dashboard.py')

if __name__ == "__main__":
    launch_dashboard()
