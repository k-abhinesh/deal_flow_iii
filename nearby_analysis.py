import pandas as pd
import streamlit as st
from geopy.distance import geodesic
import plotly.express as px

@st.cache_data # Use Streamlit's data caching for performance
def load_training_data(filepath='processing_outputs/cleaned_data.csv'):
    """
    Loads the training data from a CSV file.
    Caches the data in memory to avoid reloading on every run.
    """
    try:
        df = pd.read_csv(filepath)
        # --- Ensure necessary columns exist for the analysis ---
        bhk_count_cols = [f'{bhk}_count' for bhk in ['1RK', '1BHK', '1_5BHK', '2BHK', '2_5BHK', '3BHK', '3_5BHK', '4BHK', '4_5BHK', '5BHK']]
        required_cols = [
            'latitude', 'longitude', 'project_name', 'total_project_cost_inr', 
            'total_unit_count', 'avg_price', 'total_open_area_sqm', 'product_type',
            '2BHK_mean_carpet_area', '3BHK_mean_carpet_area', 'project_start_date',
            'project_completion_date', 'project_duration'
        ] + bhk_count_cols

        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"The training data at '{filepath}' is missing required columns for analysis: {', '.join(missing_cols)}. Please check the file.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"Training data file not found. Please ensure the file exists at: '{filepath}'")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the training data: {e}")
        return None

def _calculate_stats(df):
    """Helper function to calculate a standard set of statistics on a dataframe."""
    if df.empty:
        return {"properties_found": 0}
    
    return {
        "properties_found": len(df),
        "avg_project_cost_cr": (df['total_project_cost_inr'].mean() / 10**7),
        "min_project_cost_cr": (df['total_project_cost_inr'].min() / 10**7),
        "max_project_cost_cr": (df['total_project_cost_inr'].max() / 10**7),
        "median_project_cost_cr": (df['total_project_cost_inr'].median() / 10**7),
        "avg_unit_count": df['total_unit_count'].mean(),
        "min_unit_count": df['total_unit_count'].min(),
        "max_unit_count": df['total_unit_count'].max(),
        "median_unit_count": df['total_unit_count'].median(),
        "avg_sale_price_sqft": df['avg_price'].mean() * 1000, 
        "min_sale_price_sqft": df['avg_price'].min() * 1000,
        "max_sale_price_sqft": df['avg_price'].max() * 1000,
        "median_sale_price_sqft": df['avg_price'].median() * 1000,
        "min_project_completion_date" : df['project_completion_date'].min().strftime('%b %Y'),
        "max_project_completion_date" : df['project_completion_date'].max().strftime('%b %Y'),
        "min_project_duration": df['project_duration'].min(),
        "max_project_duration": df['project_duration'].max()
    }

def _format_preview_df(df):
    """Helper function to format a dataframe for UI preview."""
    if df.empty:
        return pd.DataFrame()

    bhk_types = ['1RK', '1BHK', '1_5BHK', '2BHK', '2_5BHK', '3BHK', '3_5BHK', '4BHK', '4_5BHK', '5BHK']
    def get_available_bhks(row):
        available = [bhk.replace('_', '.') for bhk in bhk_types if f'{bhk}_count' in row.index and pd.notna(row[f'{bhk}_count']) and row[f'{bhk}_count'] > 0]
        return ', '.join(available) if available else 'N/A'

    df['available_units_str'] = df.apply(get_available_bhks, axis=1)

    preview_cols = [
        'project_name', 'product_type', 'distance_km', "total_area_of_land_sqm", 'available_units_str',
        'total_unit_count', 'total_project_cost_inr', 'avg_price', 'project_start_date',
        'project_completion_date', 'project_duration',
    ]
    preview_df = df[preview_cols].copy()

    preview_df.rename(columns={
        'project_name': 'Project Name', 'product_type': 'Product Type',
        'far': 'FAR', 'distance_km': 'Distance',
        "total_area_of_land_sqm": 'Land Area (sq.m)', 'available_units_str': 'Available Units',
        'total_unit_count': 'Total Units', 'total_project_cost_inr': 'Project Cost',
        'avg_price': 'Avg. Price (per sq.ft)', 'project_start_date': 'Start Date',
        'project_completion_date': 'Completion Date', 'project_duration': 'Duration (Years)'
    }, inplace=True)
    
    preview_df['Distance'] = preview_df['Distance'].apply(lambda x: f"{x:.2f} km")
    preview_df['Project Cost'] = preview_df['Project Cost'].apply(lambda x: f"₹ {x / 10**7:.2f} Cr")
    preview_df['Avg. Price (per sq.ft)'] = preview_df['Avg. Price (per sq.ft)'].apply(lambda x: f"₹ {int(x * 1000):,}")
    preview_df['Start Date'] = preview_df['Start Date'].dt.strftime('%b %Y')
    preview_df['Completion Date'] = preview_df['Completion Date'].dt.strftime('%b %Y')

    return preview_df.sort_values('Distance')

def get_nearby_properties_analysis(lat, lon, radius_km, df, user_land_area):
    """
    Finds properties within a radius, calculates stats, and then repeats the
    analysis for a subset of properties with a similar land area.
    """
    if df is None:
        return None

    center_point = (lat, lon)
    df_copy = df.copy()
    date_cols = ['project_start_date', 'project_completion_date']
    for col in date_cols:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    df_copy.dropna(subset=date_cols, inplace=True) # Drop rows where dates couldn't be parsed

    df_copy['distance_km'] = df_copy.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), center_point).km,
        axis=1
    )
    
    nearby_df = df_copy[df_copy['distance_km'] <= radius_km].copy()
    
    # --- 1. Analysis for ALL nearby properties ---
    all_nearby_stats = _calculate_stats(nearby_df)
    all_nearby_preview_df = _format_preview_df(nearby_df)

    # --- 2. Analysis for NEARBY properties with SIMILAR LAND AREA ---
    lower_bound = user_land_area * 0.70
    upper_bound = user_land_area * 1.30
    
    similar_area_df = nearby_df[
        (nearby_df['total_area_of_land_sqm'] >= lower_bound) & 
        (nearby_df['total_area_of_land_sqm'] <= upper_bound)
    ].copy()
    
    similar_area_stats = _calculate_stats(similar_area_df)
    similar_area_preview_df = _format_preview_df(similar_area_df)

    # --- Return all results in a structured dictionary ---
    return {
        "all_nearby": {
            "stats": all_nearby_stats,
            "df": all_nearby_preview_df
        },
        "similar_nearby": {
            "stats": similar_area_stats,
            "df": similar_area_preview_df
        }
    }
