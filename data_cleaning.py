#!/usr/bin/env python3
"""
Data Cleaning and Preprocessing Script for Retail Performance Analysis
This script performs comprehensive data cleaning and feature engineering on the raw retail dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(filepath):
    """
    Load raw retail data from CSV file
    
    Args:
        filepath (str): Path to the raw data CSV file
        
    Returns:
        pd.DataFrame: Raw retail transaction data
    """
    print("📁 STEP 1: Loading Raw Data")
    print("=" * 50)
    
    try:
        # Load data with appropriate encoding for international characters
        df_raw = pd.read_csv(filepath, encoding='latin-1')
        
        print(f"✅ Successfully loaded data from: {filepath}")
        print(f"📊 Raw data shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        print(f"💾 Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display basic info about the dataset
        print(f"\n📋 Column Information:")
        for col in df_raw.columns:
            print(f"   • {col}: {df_raw[col].dtype}")
        
        print(f"\n📅 Date Range (before cleaning):")
        if 'InvoiceDate' in df_raw.columns:
            try:
                dates = pd.to_datetime(df_raw['InvoiceDate'], errors='coerce')
                print(f"   • From: {dates.min()}")
                print(f"   • To: {dates.max()}")
            except:
                print("   • Date parsing will be handled in cleaning step")
        
        return df_raw
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_data_quality(df):
    """
    Analyze data quality issues before cleaning
    
    Args:
        df (pd.DataFrame): Raw dataset to analyze
        
    Returns:
        dict: Dictionary containing data quality metrics
    """
    print("\n🔍 STEP 2: Data Quality Analysis")
    print("=" * 50)
    
    quality_report = {}
    
    # Missing values analysis
    print("🔢 Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    for col in df.columns:
        missing_count = missing_counts[col]
        missing_pct = missing_percentages[col]
        quality_report[f'{col}_missing'] = missing_count
        
        if missing_count > 0:
            print(f"   • {col}: {missing_count:,} missing ({missing_pct:.2f}%)")
        else:
            print(f"   • {col}: No missing values ✅")
    
    # Data type analysis
    print(f"\n🧮 Data Type Analysis:")
    for col, dtype in df.dtypes.items():
        print(f"   • {col}: {dtype}")
    
    # Unique values for categorical columns
    print(f"\n🏷️ Categorical Data Summary:")
    categorical_cols = ['Country', 'StockCode', 'Description']
    for col in categorical_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            print(f"   • {col}: {unique_count:,} unique values out of {total_count:,} total")
    
    # Numerical data summary
    print(f"\n📊 Numerical Data Summary:")
    numerical_cols = ['Quantity', 'UnitPrice']
    for col in numerical_cols:
        if col in df.columns:
            print(f"   • {col}:")
            print(f"     - Min: {df[col].min()}")
            print(f"     - Max: {df[col].max()}")
            print(f"     - Mean: {df[col].mean():.2f}")
            print(f"     - Negative values: {(df[col] < 0).sum():,}")
            print(f"     - Zero values: {(df[col] == 0).sum():,}")
    
    quality_report['total_rows'] = len(df)
    quality_report['total_columns'] = len(df.columns)
    
    return quality_report

def clean_and_transform_data(df_raw):
    """
    Perform comprehensive data cleaning and transformation
    
    Args:
        df_raw (pd.DataFrame): Raw dataset to clean
        
    Returns:
        pd.DataFrame: Cleaned and transformed dataset
    """
    print("\n🧹 STEP 3: Data Cleaning and Transformation")
    print("=" * 50)
    
    # Create a copy to avoid modifying original data
    df = df_raw.copy()
    original_rows = len(df)
    
    # 3.1: Date conversion and validation
    print("📅 Sub-step 3.1: Date Processing")
    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        invalid_dates = df['InvoiceDate'].isnull().sum()
        
        if invalid_dates > 0:
            print(f"   ⚠️  Found {invalid_dates:,} invalid dates - removing these rows")
            df = df.dropna(subset=['InvoiceDate'])
        else:
            print("   ✅ All dates successfully parsed")
            
        print(f"   📅 Final date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        
    except Exception as e:
        print(f"   ❌ Error processing dates: {e}")
        return None
    
    # 3.2: Revenue calculation
    print("\n💰 Sub-step 3.2: Revenue Calculation")
    try:
        # Check for non-numeric values in Quantity and UnitPrice
        quantity_numeric = pd.to_numeric(df['Quantity'], errors='coerce')
        price_numeric = pd.to_numeric(df['UnitPrice'], errors='coerce')
        
        # Count non-numeric values
        quantity_invalid = quantity_numeric.isnull().sum() - df['Quantity'].isnull().sum()
        price_invalid = price_numeric.isnull().sum() - df['UnitPrice'].isnull().sum()
        
        if quantity_invalid > 0:
            print(f"   ⚠️  Found {quantity_invalid:,} non-numeric Quantity values")
        if price_invalid > 0:
            print(f"   ⚠️  Found {price_invalid:,} non-numeric UnitPrice values")
        
        # Use numeric versions
        df['Quantity'] = quantity_numeric
        df['UnitPrice'] = price_numeric
        
        # Calculate revenue
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        print("   ✅ Revenue calculated as Quantity × UnitPrice")
        
    except Exception as e:
        print(f"   ❌ Error calculating revenue: {e}")
        return None
    
    # 3.3: Remove invalid transactions
    print("\n🚫 Sub-step 3.3: Removing Invalid Transactions")
    
    # Remove negative quantities (returns/cancellations)
    negative_qty = (df['Quantity'] <= 0).sum()
    if negative_qty > 0:
        print(f"   🔄 Removing {negative_qty:,} transactions with negative/zero quantities (returns/cancellations)")
        df = df[df['Quantity'] > 0]
    
    # Remove zero or negative unit prices
    invalid_price = (df['UnitPrice'] <= 0).sum()
    if invalid_price > 0:
        print(f"   💸 Removing {invalid_price:,} transactions with zero/negative unit prices")
        df = df[df['UnitPrice'] > 0]
    
    # Remove missing customer IDs (guest purchases or data quality issues)
    missing_customers = df['CustomerID'].isnull().sum()
    if missing_customers > 0:
        print(f"   👤 Removing {missing_customers:,} transactions with missing CustomerID")
        df = df.dropna(subset=['CustomerID'])
    
    # Remove any remaining null revenue values
    null_revenue = df['Revenue'].isnull().sum()
    if null_revenue > 0:
        print(f"   💰 Removing {null_revenue:,} transactions with null revenue")
        df = df.dropna(subset=['Revenue'])
    
    rows_after_cleaning = len(df)
    rows_removed = original_rows - rows_after_cleaning
    removal_percentage = (rows_removed / original_rows) * 100
    
    print(f"   📊 Cleaning Summary:")
    print(f"     • Original rows: {original_rows:,}")
    print(f"     • Rows removed: {rows_removed:,} ({removal_percentage:.1f}%)")
    print(f"     • Final rows: {rows_after_cleaning:,}")
    
    # 3.4: Feature engineering - Time components
    print("\n⏰ Sub-step 3.4: Time Feature Engineering")
    
    try:
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['MonthName'] = df['InvoiceDate'].dt.strftime('%B')
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Date'] = df['InvoiceDate'].dt.date
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['WeekOfYear'] = df['InvoiceDate'].dt.isocalendar().week
        
        print("   ✅ Created time-based features:")
        print("     • Year, Month, MonthName")
        print("     • DayOfWeek, Hour, Date")
        print("     • Quarter, WeekOfYear")
        
        # Summary of time range
        years = df['Year'].unique()
        months = df['Month'].unique()
        print(f"   📅 Data spans: {len(years)} year(s), {len(months)} month(s)")
        
    except Exception as e:
        print(f"   ❌ Error creating time features: {e}")
        return None
    
    # 3.5: Data type optimization
    print("\n🔧 Sub-step 3.5: Data Type Optimization")
    
    try:
        # Convert categorical columns to category type for memory efficiency
        categorical_columns = ['Country', 'StockCode', 'Description', 'MonthName', 'DayOfWeek']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert CustomerID to integer
        df['CustomerID'] = df['CustomerID'].astype('int32')
        
        # Optimize numeric columns
        df['Quantity'] = df['Quantity'].astype('int32')
        df['Year'] = df['Year'].astype('int16')
        df['Month'] = df['Month'].astype('int8')
        df['Hour'] = df['Hour'].astype('int8')
        df['Quarter'] = df['Quarter'].astype('int8')
        df['WeekOfYear'] = df['WeekOfYear'].astype('int8')
        
        print("   ✅ Optimized data types for memory efficiency")
        print(f"   💾 Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"   ⚠️  Warning - Data type optimization failed: {e}")
        # Continue without optimization if it fails
    
    return df

def generate_cleaning_summary(df_raw, df_clean, quality_report):
    """
    Generate a comprehensive summary of the cleaning process
    
    Args:
        df_raw (pd.DataFrame): Original raw dataset
        df_clean (pd.DataFrame): Cleaned dataset
        quality_report (dict): Data quality metrics
        
    Returns:
        dict: Cleaning summary statistics
    """
    print("\n📋 STEP 4: Cleaning Summary Report")
    print("=" * 50)
    
    summary = {}
    
    # Basic statistics
    original_rows = len(df_raw) if df_raw is not None else 0
    cleaned_rows = len(df_clean) if df_clean is not None else 0
    rows_removed = original_rows - cleaned_rows
    retention_rate = (cleaned_rows / original_rows) * 100 if original_rows > 0 else 0
    
    summary['original_rows'] = original_rows
    summary['cleaned_rows'] = cleaned_rows
    summary['rows_removed'] = rows_removed
    summary['retention_rate'] = retention_rate
    
    print(f"📊 Data Transformation Summary:")
    print(f"   • Original dataset: {original_rows:,} rows")
    print(f"   • Cleaned dataset: {cleaned_rows:,} rows")
    print(f"   • Rows removed: {rows_removed:,} ({100-retention_rate:.1f}%)")
    print(f"   • Data retention rate: {retention_rate:.1f}%")
    
    if df_clean is not None:
        # Revenue statistics
        total_revenue = df_clean['Revenue'].sum()
        avg_transaction = df_clean['Revenue'].mean()
        max_transaction = df_clean['Revenue'].max()
        min_transaction = df_clean['Revenue'].min()
        
        summary['total_revenue'] = total_revenue
        summary['avg_transaction'] = avg_transaction
        summary['max_transaction'] = max_transaction
        summary['min_transaction'] = min_transaction
        
        print(f"\n💰 Revenue Analysis:")
        print(f"   • Total revenue: £{total_revenue:,.2f}")
        print(f"   • Average transaction: £{avg_transaction:.2f}")
        print(f"   • Transaction range: £{min_transaction:.2f} - £{max_transaction:,.2f}")
        
        # Customer and transaction statistics
        unique_customers = df_clean['CustomerID'].nunique()
        unique_products = df_clean['StockCode'].nunique()
        unique_countries = df_clean['Country'].nunique()
        unique_invoices = df_clean['InvoiceNo'].nunique()
        
        summary['unique_customers'] = unique_customers
        summary['unique_products'] = unique_products
        summary['unique_countries'] = unique_countries
        summary['unique_invoices'] = unique_invoices
        
        print(f"\n👥 Business Metrics:")
        print(f"   • Unique customers: {unique_customers:,}")
        print(f"   • Unique products: {unique_products:,}")
        print(f"   • Countries served: {unique_countries}")
        print(f"   • Unique invoices: {unique_invoices:,}")
        
        # Time range
        date_range = df_clean['InvoiceDate'].max() - df_clean['InvoiceDate'].min()
        summary['date_range_days'] = date_range.days
        
        print(f"\n📅 Time Period:")
        print(f"   • From: {df_clean['InvoiceDate'].min()}")
        print(f"   • To: {df_clean['InvoiceDate'].max()}")
        print(f"   • Duration: {date_range.days} days")
        
        # Top countries by revenue
        print(f"\n🌍 Top 5 Countries by Revenue:")
        top_countries = df_clean.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5)
        for country, revenue in top_countries.items():
            print(f"   • {country}: £{revenue:,.0f}")
    
    return summary

def save_cleaned_data(df_clean, output_path='data_cleaned.csv'):
    """
    Save the cleaned dataset to a CSV file
    
    Args:
        df_clean (pd.DataFrame): Cleaned dataset
        output_path (str): Path for the output file
        
    Returns:
        bool: Success status
    """
    print(f"\n💾 STEP 5: Saving Cleaned Data")
    print("=" * 50)
    
    try:
        # Save to CSV
        df_clean.to_csv(output_path, index=False, encoding='utf-8')
        
        file_size = pd.read_csv(output_path).memory_usage(deep=True).sum() / 1024**2
        
        print(f"✅ Successfully saved cleaned data to: {output_path}")
        print(f"📁 File size: {file_size:.2f} MB")
        print(f"📊 Columns saved: {len(df_clean.columns)}")
        print(f"📋 Column list: {', '.join(df_clean.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving cleaned data: {e}")
        return False

def main():
    """
    Main function to execute the complete data cleaning pipeline
    """
    print("🚀 RETAIL DATA CLEANING PIPELINE")
    print("=" * 80)
    print("This script performs comprehensive data cleaning and preprocessing")
    print("for retail performance analysis.\n")
    
    # Configuration
    input_file = 'data.csv'
    output_file = 'data_cleaned.csv'
    
    # Execute cleaning pipeline
    try:
        # Step 1: Load raw data
        df_raw = load_raw_data(input_file)
        if df_raw is None:
            print("❌ Failed to load data. Exiting.")
            return
        
        # Step 2: Analyze data quality
        quality_report = analyze_data_quality(df_raw)
        
        # Step 3: Clean and transform data
        df_clean = clean_and_transform_data(df_raw)
        if df_clean is None:
            print("❌ Data cleaning failed. Exiting.")
            return
        
        # Step 4: Generate summary
        summary = generate_cleaning_summary(df_raw, df_clean, quality_report)
        
        # Step 5: Save cleaned data
        success = save_cleaned_data(df_clean, output_file)
        
        if success:
            print(f"\n🎉 DATA CLEANING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📁 Input file: {input_file}")
            print(f"📁 Output file: {output_file}")
            print(f"📊 Data retention: {summary.get('retention_rate', 0):.1f}%")
            print(f"💰 Total revenue: £{summary.get('total_revenue', 0):,.2f}")
            print(f"👥 Customers: {summary.get('unique_customers', 0):,}")
            print("\n✅ Ready for analysis in Streamlit dashboard!")
        else:
            print("\n❌ Data cleaning completed but saving failed.")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
