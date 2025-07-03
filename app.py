import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from location_processing import *

# =====================================================
# APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Real Estate Project Details Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODELS FUNCTION
# =====================================================

@st.cache_resource
def load_models(model_dir='models'):
    """Load all trained models"""
    try:
        models = {}
        scalers = {}
        
        # Load each model and scaler
        model_names = ['bhk_presence', 'unit_count', 'carpet_area', 'project_metrics']
        
        for name in model_names:
            with open(f'{model_dir}/{name}_model.pkl', 'rb') as f:
                models[name] = pickle.load(f)
            
            with open(f'{model_dir}/{name}_scaler.pkl', 'rb') as f:
                scalers[name] = pickle.load(f)
        
        # Load feature names
        with open(f'{model_dir}/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return models, scalers, feature_names, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_project(input_data, models, scalers, feature_names, selected_bhk_types=None):
    """Make predictions for new project data"""
    try:
        # Ensure input data has all required features
        X_new = input_data[feature_names].values.reshape(1, -1)
        
        # Step 1: Handle BHK Presence
        if selected_bhk_types is not None:
            # User specified BHK types - create presence array manually
            bhk_types = ['1RK', '1BHK', '1.5BHK', '2BHK', '2.5BHK', '3BHK', '3.5BHK', '4BHK', '4.5BHK', '5BHK']
            presence_pred = np.array([[1 if bhk in selected_bhk_types else 0 for bhk in bhk_types]])
        else:
            # Use model to predict BHK presence
            X_scaled = scalers['bhk_presence'].transform(X_new)
            presence_pred = models['bhk_presence'].predict(X_scaled)
        
        # Step 2: Predict Unit Counts
        X_enhanced = np.hstack([X_new, presence_pred])
        X_scaled = scalers['unit_count'].transform(X_enhanced)
        count_pred = models['unit_count'].predict(X_scaled)
        count_pred = np.maximum(count_pred, 0)  # Ensure non-negative
        
        # Step 3: Predict Carpet Areas
        X_enhanced = np.hstack([X_new, presence_pred, count_pred])
        X_scaled = scalers['carpet_area'].transform(X_enhanced)
        carpet_pred = models['carpet_area'].predict(X_scaled)
        carpet_pred = np.maximum(carpet_pred, 0)  # Ensure non-negative
        
        # Step 4: Predict Project Metrics
        X_enhanced = np.hstack([X_new, presence_pred, count_pred, carpet_pred])
        X_scaled = scalers['project_metrics'].transform(X_enhanced)
        project_pred = models['project_metrics'].predict(X_scaled)
        
        return {
            'bhk_presence': presence_pred[0],
            'unit_counts': count_pred[0],
            'carpet_areas': carpet_pred[0],
            'project_metrics': project_pred[0]
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
# =====================================================
# UI HELPER FUNCTIONS
# =====================================================

def create_input_form():
    """Create input form for project parameters"""
    
    st.sidebar.header("🏗️ Project Parameters")
    
    # Basic project details
    st.sidebar.subheader("Basic Details")
    total_area = st.sidebar.number_input(
        "Total Land Area (sqm)", 
        min_value=100, 
        max_value=100000, 
        value=5000,
        help="Total area of land in square meters"
    )
    
    far = st.sidebar.number_input(
        "Floor Area Ratio (FAR)", 
        min_value=0.1, 
        max_value=10.0, 
        value=2.5,
        help="Ratio of total floor area to land area"
    )
    
    # Location details
    st.sidebar.subheader("Location Details")
    lat = st.sidebar.number_input(
        "Latitude", 
        min_value=10.0, 
        max_value=20.0, 
        value=12.9300773
    )
    
    long = st.sidebar.number_input(
        "Longitude", 
        min_value=70.0, 
        max_value=80.0, 
        value=77.6950959
    )
    
    
    # Product type and Taluk
    st.sidebar.subheader("Project Type and Configuration")
    product_type = st.sidebar.selectbox(
        "Product Type",
        ["Apartment", "Villa"],
        help="Type of residential project"
    )

    customize_bhk = st.sidebar.checkbox(
    "Customize BHK Types", 
    value=False,
    help="Select specific BHK types for your project"
)

    selected_bhk_types = None
    if customize_bhk:
        bhk_options = ['1RK', '1BHK', '1.5BHK', '2BHK', '2.5BHK', '3BHK', '3.5BHK', '4BHK', '4.5BHK', '5BHK']
        selected_bhk_types = st.sidebar.multiselect(
            "Select BHK Types",
            options=bhk_options,
            default=[],
            help="Choose which BHK types to include in your project"
        )
        if customize_bhk and len(selected_bhk_types) == 0:
            st.sidebar.warning("Please select at least one BHK type")
        
    return {
        'total_area_of_land_sqm': total_area,
        'far': far,
        'latitude': lat,
        'longitude': long,
        'product_type': product_type,
        'selected_bhk_types': selected_bhk_types
    }

def prepare_input_features(user_input, feature_names):
    """Convert user input to model features"""
    
    # Create base features
    input_dict = {
        'total_area_of_land_sqm': user_input['total_area_of_land_sqm'],
        'far': user_input['far'],
        'product_type_Apartment': 1 if user_input['product_type'] == 'Apartment' else 0,
        'product_type_Villa': 1 if user_input['product_type'] == 'Villa' else 0,
    }


    
    # Add taluk dummies (set all to 0 first, then set selected taluk to 1)
    taluk_features = [f for f in feature_names if f.startswith('taluk_')]
    for taluk_feature in taluk_features:
        input_dict[taluk_feature] = 0
    
    #Set selected taluk to 1
    selected_taluk_feature = get_taluk(user_input['latitude'], user_input['longitude'])
    if selected_taluk_feature in input_dict:
        input_dict[selected_taluk_feature] = 1

    location_details = get_all_location_details(user_input['latitude'], user_input['longitude'])

    input_dict["airport_distance_kms"] = location_details["airport_distance_kms"]
    input_dict["ksr_jn_distance_kms"] = location_details["ksr_jn_distance_kms"]
    input_dict["yeshwantpur_jn_distance_kms"] = location_details["yeshwantpur_jn_distance_kms"]
    input_dict["nearest_metro_dist_km"] = location_details["nearest_metro"][0]
    input_dict["nearest_major_road_dist_kms"] = location_details["nearest_major_road"][0]
    
    # Create DataFrame with all features in correct order
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all features are present and in correct order
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    return input_df[feature_names]

def display_predictions(predictions, user_input):
    """Display prediction results with visualizations"""
    
    if predictions is None:
        st.error("Unable to generate predictions")
        return
    
    # BHK Type names
    bhk_types = ['1RK', '1BHK', '1.5BHK', '2BHK', '2.5BHK', '3BHK', '3.5BHK', '4BHK', '4.5BHK', '5BHK']
    
    # Create tabs for different prediction categories
    tab1, tab2, tab3 = st.tabs(["🏠Location Metrics", "📊 Inventory Configuration", "💰 Project Metrics"])
    
    with tab1:
        st.subheader("Location Details")
        
        location_metrics = get_all_location_details(user_input['latitude'], user_input['longitude'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            taluk = location_metrics["taluk"]
            st.metric(
                "Taluk", 
                taluk,
                help="Taluk of the location provided"
            )
        
        with col2:
            airport_distance = location_metrics["airport_distance_kms"]
            st.metric(
                f"Distance to Airport", 
                f"{airport_distance:.1f} kms",
                help="Distance to Bengaluru airport"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            ksr_jn_distance = location_metrics["ksr_jn_distance_kms"]
            st.metric(
                "Distance to KSR Jn Railway Station", 
                f"{ksr_jn_distance:.1f} kms",
                help="KSR Bengaluru City Junction distance"
            )
        
        with col4:
            yeshwantpur_jn_distance = location_metrics["yeshwantpur_jn_distance_kms"]
            st.metric(
                f"Distance to Yeshwantpur Railway Station", 
                f"{yeshwantpur_jn_distance:.1f} kms",
                help="Yeshwantpur Jn Railway Station"
            )

        col5, col6 = st.columns(2)
        
        with col5:
            nearest_metro = location_metrics["nearest_metro"][1]
            st.metric(
                "Nearest Metro Station", 
                f"{nearest_metro}",
                help="Nearest Metro Station Name"
            )
        
        with col6:
            nearest_metro_dist = location_metrics["nearest_metro"][0]
            st.metric(
                "Distance to the Metro Station", 
                f"{nearest_metro_dist:.1f} kms",
                help="Distance to the nearest Metro Station"
            )

        col7, col8 = st.columns(2)
        
        with col7:
            nearest_major_road = location_metrics["nearest_major_road"][1]
            st.metric(
                "Nearest Major Road", 
                f"{nearest_major_road}",
                help="Nearest Major Road Name"
            )
        
        with col8:
            nearest_major_road_dist = location_metrics["nearest_major_road"][0]
            st.metric(
                "Distance to the Major Road", 
                f"{nearest_major_road_dist:.1f} kms",
                help="Distance to the nearest Major Road"
            )
    
    with tab2:
        st.subheader("Inventory Predictions")

        # Add this info message
        if st.session_state.get('user_input', {}).get('selected_bhk_types') is not None:
            st.info("🎯 Predictions based on your selected BHK types")
        else:
            st.info("🤖 BHK types predicted by AI model")

        project_metrics = predictions['project_metrics']
        avg_price = max(0, project_metrics[2])

        counts = predictions['unit_counts']
        count_df = pd.DataFrame({
            'BHK Type': bhk_types,
            'Predicted Count': [max(0, round(count)) for count in counts]
        })
        carpet_areas = predictions['carpet_areas']
        area_df = pd.DataFrame({
            'BHK Type': bhk_types,
            'Mean Carpet Area (sqft)': [max(0, round(area* 10.764)) for area in carpet_areas]})
        unit_df = pd.merge(count_df, area_df, on='BHK Type')
        # Only show BHK types with count > 0
        unit_df_filtered = unit_df[unit_df['Predicted Count'] > 0]
        unit_df_filtered["Mean Price (₹) (Lakhs)"] = round(unit_df_filtered["Mean Carpet Area (sqft)"] * avg_price/100,2)
        if not unit_df_filtered.empty:
            st.dataframe(unit_df_filtered, use_container_width=True)
            
            # Bar chart
            fig = px.pie(
                values=unit_df_filtered['Predicted Count'],
                names=unit_df_filtered['BHK Type'],
                title="Unit Mix"
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                unit_df_filtered, 
                x='BHK Type', 
                y='Mean Carpet Area (sqft)',
                title="Predicted Carpet Areas by BHK Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            total_units = unit_df_filtered['Predicted Count'].sum()

            
            st.info(f"**Total Predicted Units:** {total_units}")
        else:
            st.warning("No units predicted")
    
    with tab3:
        st.subheader("Project-Level Metrics")
        
        project_metrics = predictions['project_metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            open_area = max(0, project_metrics[0])
            st.metric(
                "Total Open Area", 
                f"{open_area:,.0f} sqm",
                help="Predicted total open/common area"
            )
        
        with col2:
            total_cost = max(0, project_metrics[1])
            st.metric(
                "Total Project Cost", 
                f"₹{total_cost/10000000:.1f} Cr",
                help="Predicted total project cost in Crores"
            )
        
        with col3:
            avg_price = max(0, project_metrics[2])
            st.metric(
                "Average Unit Price", 
                f"₹{avg_price:.1f}K/sq.ft",
                help="Predicted average unit price in Lakhs"
            )

# =====================================================
# MAIN APP
# =====================================================

def main():
    # App header
    st.title("🏠 Real Estate Project Predictor")
    st.markdown("### Predict unit mix, counts, carpet areas, and project metrics for residential developments")
    
    # Load models
    models, scalers, feature_names, model_loaded = load_models()
    
    if not model_loaded:
        st.error("❌ Could not load prediction models. Please ensure model files are in the 'models' directory.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Get user input
    user_input = create_input_form()

    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
    # Add validation for custom BHK selection
        if user_input['selected_bhk_types'] is not None and len(user_input['selected_bhk_types']) == 0:
            st.error("Please select at least one BHK type or disable BHK customization")
        else:
            with st.spinner("Generating predictions..."):
                # Prepare input features
                input_features = prepare_input_features(user_input, feature_names)
                
                # Make predictions with selected BHK types
                predictions = predict_project(
                    input_features, 
                    models, 
                    scalers, 
                    feature_names, 
                    selected_bhk_types=user_input['selected_bhk_types']
                )
                
                # Store predictions in session state
                st.session_state['predictions'] = predictions
                st.session_state['user_input'] = user_input
        
    # # Main content area
    if 'predictions' in st.session_state and st.session_state['predictions'] is not None:
        display_predictions(st.session_state['predictions'], st.session_state['user_input'])
    else:
        st.info("👈 Configure project parameters and click 'Generate Predictions' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on historical data patterns and should be used as estimates for planning purposes.")

if __name__ == "__main__":
    main()