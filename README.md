# 📊 Retail Performance Analysis Dashboard

## Overview

Interactive Streamlit dashboard for analyzing retail performance data, identifying revenue patterns, seasonal trends, and providing strategic business recommendations.

## Features

### 🎯 **Six Interactive Analysis Sections:**

1. **📊 Executive Summary**
   - Key business metrics overview
   - Revenue and growth visualizations
   - Critical insights summary

2. **📈 Revenue Trends**
   - Daily revenue analysis with moving averages
   - Peak performance days identification
   - Trend visualization

3. **🎯 Seasonal Patterns**
   - Monthly seasonal analysis
   - Weekly pattern identification
   - Peak/low period comparisons

4. **⏰ Time Analysis**
   - Hourly activity patterns
   - Peak/slow hours identification
   - Optimal timing insights

5. **🌍 Geographic Insights**
   - Country-wise revenue distribution
   - Geographic performance metrics
   - International market analysis

6. **💡 Recommendations**
   - Strategic promotion timing
   - Restocking strategies
   - Revenue optimization tactics
   - Cyclical behavior insights

## 🚀 Quick Start

### Option 1: Using the Batch File (Easiest)

1. Double-click `launch_dashboard.bat`
2. Dashboard will open automatically in your browser
3. Navigate through different sections using the sidebar

### Option 2: Command Line

```bash
# Navigate to the project directory
cd "c:\Users\ToyeebKazeem\Documents\ADC\Analyst Challenge"

# Run the dashboard
conda run --live-stream --name base streamlit run streamlit_dashboard.py
```

### Option 3: Direct Streamlit Command

```bash
streamlit run streamlit_dashboard.py
```

## 📊 Dashboard Sections Guide

### Executive Summary

- **Purpose:** High-level overview of business performance
- **Key Metrics:** Total revenue, transactions, customers, daily averages
- **Visualizations:** Monthly patterns, growth trends
- **Best For:** Initial presentation, stakeholder briefings

### Revenue Trends

- **Purpose:** Detailed daily revenue analysis
- **Features:** Moving averages, peak day identification
- **Best For:** Understanding daily fluctuations, identifying outliers

### Seasonal Patterns

- **Purpose:** Monthly and weekly pattern analysis
- **Features:** Peak/low period identification, seasonal comparisons
- **Best For:** Planning seasonal campaigns, inventory management

### Time Analysis

- **Purpose:** Hourly activity pattern analysis
- **Features:** Peak hours, slow periods, optimal timing
- **Best For:** Staff scheduling, promotional timing

### Geographic Insights

- **Purpose:** Revenue distribution by country
- **Features:** Top markets, international performance
- **Best For:** Market expansion planning, geographic focus

### Recommendations

- **Purpose:** Strategic business recommendations
- **Features:** Promotion timing, restocking strategies, optimization tactics
- **Best For:** Action planning, strategic decision making

## 📈 Key Findings Summary

### Peak Performance Periods

- **Months:** November, December, October
- **Days:** Thursday, Tuesday, Wednesday
- **Hours:** 12:00 PM, 10:00 AM, 1:00 PM

### Low Performance Periods

- **Months:** February, April, January
- **Days:** Sunday, Monday
- **Hours:** 6:00 AM, 8:00 PM, 7:00 AM

### Growth Metrics

- **Year-over-Year Growth:** +1,356%
- **Total Revenue:** £8,911,408
- **Average Daily Revenue:** £29,218
- **Revenue Volatility:** 61.1%

## 🎯 Strategic Recommendations

### Promotion Timing

- Target February-April for major promotions
- Launch holiday campaigns in October-November
- Focus on Tuesday-Wednesday promotions
- Schedule flash sales during 6-9 AM

### Inventory Management

- Heavy restocking in September-October
- Higher inventory Thursday-Sunday
- Stock up before 10 AM-4 PM peak hours
- Reduce inventory January-February

### Revenue Optimization

- Focus marketing spend on October-December
- Implement dynamic pricing during peak hours
- Offer volume discounts on slower days
- Create bundled offers during low-revenue months

## 🛠️ Technical Requirements

### Dependencies

- Python 3.7+
- streamlit
- pandas
- numpy
- plotly
- matplotlib (for backup charts)

### Data Requirements

- `data.csv` file in the same directory
- CSV should contain: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## 📱 Dashboard Navigation Tips

1. **Sidebar Navigation:** Use radio buttons to switch between analysis sections
2. **Data Overview:** Check sidebar for quick data statistics
3. **Interactive Charts:** Hover for detailed information, zoom/pan as needed
4. **Full Screen:** Click the expand icon on charts for detailed view
5. **Export Options:** Use Plotly's built-in export features for charts

## 🎬 For Video Presentations

### Recommended Flow

1. Start with **Executive Summary** for overview
2. Move to **Seasonal Patterns** to show cyclical behavior
3. Demonstrate **Time Analysis** for daily/hourly insights
4. Show **Revenue Trends** for growth story
5. Present **Geographic Insights** for market analysis
6. Conclude with **Recommendations** for actionable insights

### Screen Recording Tips

- Use full-screen mode for better visibility
- Navigate slowly between sections
- Pause to highlight key insights
- Use the hover features to show interactivity
- Zoom into specific chart areas for detailed analysis

## 🔧 Troubleshooting

### Common Issues

1. **Dashboard won't start:** Ensure all dependencies are installed
2. **Data not loading:** Check that `data.csv` is in the correct directory
3. **Charts not displaying:** Try refreshing the browser page
4. **Performance issues:** Close other browser tabs, restart the dashboard

### Support

- Check the terminal/command prompt for error messages
- Ensure Python environment is properly configured
- Verify all required packages are installed

## 📄 Files Included

- `streamlit_dashboard.py` - Main dashboard application
- `launch_dashboard.bat` - Easy launch script for Windows
- `data.csv` - Retail transaction data
- `README.md` - This documentation file
- Previous analysis files (retail_analysis.py, charts, etc.)

---

**Created for ADC Analyst Challenge - July 2025**
**Interactive Retail Performance Analysis Dashboard**
