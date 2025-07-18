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
import warnings
warnings.filterwarnings('ignore')
# =====================================================
# Import functions from your new analysis file
# =====================================================
from nearby_analysis import load_training_data, get_nearby_properties_analysis

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'English_India')
    except locale.Error:
        print("Indian locale not found. Using default formatting.")

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
# LOAD MODELS FUNCTION (Unchanged)
# =====================================================
@st.cache_resource
def load_models(model_dir='models'):
    """Load all trained models, scalers, and feature names based on the new pipeline."""
    try:
        models = {}
        scalers = {}
        model_names = ['bhk_presence', 'unit_count', 'carpet_area', 'bhk_unit_count', 'avg_price', 'total_project_cost', 'open_area']
        for name in model_names:
            with open(os.path.join(model_dir, f'{name}_model.pkl'), 'rb') as f:
                models[name] = pickle.load(f)
            with open(os.path.join(model_dir, f'{name}_scaler.pkl'), 'rb') as f:
                scalers[name] = pickle.load(f)
        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        bhk_mapping = {'1RK': 0.5, '1BHK': 1.0, '1_5BHK': 1.5, '2BHK': 2.0, '2_5BHK': 2.5, '3BHK': 3.0, '3_5BHK': 3.5, '4BHK': 4.0, '4_5BHK': 4.5, '5BHK': 5.0}
        return models, scalers, feature_names, bhk_mapping, True
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure all model files are in the '{model_dir}' directory.")
        return None, None, None, None, False
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None, None, None, None, False

# =====================================================
# PREDICTION FUNCTION (Unchanged)
# =====================================================
def predict_project(input_df, models, scalers, feature_names, bhk_mapping, selected_bhk_types=None):
    """Make predictions for new project data using the updated sequential pipeline."""
    try:
        X_base = input_df[feature_names].values.reshape(1, -1)
        if selected_bhk_types:
            bhk_presence_pred = np.array([[1 if bhk in selected_bhk_types else 0 for bhk in bhk_mapping.keys()]])
        else:
            X_scaled = scalers['bhk_presence'].transform(X_base)
            bhk_presence_pred = (models['bhk_presence'].predict(X_scaled) > 0.5).astype(int)
        X_enhanced_total_units = np.hstack([X_base, bhk_presence_pred])
        X_scaled = scalers['unit_count'].transform(X_enhanced_total_units)
        total_unit_count_pred = max(0, round(models['unit_count'].predict(X_scaled)[0]))
        predicted_bhk_types = [bhk for i, bhk in enumerate(bhk_mapping.keys()) if bhk_presence_pred[0, i] == 1]
        unit_counts, carpet_areas, avg_prices = {}, {}, {}
        for bhk_str in predicted_bhk_types:
            bhk_numeric = bhk_mapping[bhk_str]
            features_carpet = np.append(X_base, [bhk_numeric, total_unit_count_pred]).reshape(1, -1)
            mean_carpet_area_pred = max(0, models['carpet_area'].predict(scalers['carpet_area'].transform(features_carpet))[0])
            carpet_areas[bhk_str] = mean_carpet_area_pred
            features_count = np.append(features_carpet, [mean_carpet_area_pred]).reshape(1, -1)
            bhk_count_pred = max(0, round(models['bhk_unit_count'].predict(scalers['bhk_unit_count'].transform(features_count))[0]))
            unit_counts[bhk_str] = bhk_count_pred
            features_price = np.append(X_base, [bhk_numeric, mean_carpet_area_pred]).reshape(1, -1)
            avg_price_pred = max(0, models['avg_price'].predict(scalers['avg_price'].transform(features_price))[0])
            avg_prices[bhk_str] = avg_price_pred
        full_bhk_counts = [unit_counts.get(bhk, 0) for bhk in bhk_mapping.keys()]
        full_carpet_areas = [carpet_areas.get(bhk, 0) for bhk in bhk_mapping.keys()]
        X_final = np.hstack([X_base, bhk_presence_pred, np.array([full_bhk_counts]), np.array([full_carpet_areas]), [[total_unit_count_pred]]])
        total_cost_pred = max(0, models['total_project_cost'].predict(scalers['total_project_cost'].transform(X_final))[0])
        open_area_pred = max(0, models['open_area'].predict(scalers['open_area'].transform(X_final))[0])
        return {'unit_counts': unit_counts, 'carpet_areas': carpet_areas, 'avg_prices': avg_prices, 'total_project_cost': total_cost_pred, 'total_open_area': open_area_pred, 'total_units': total_unit_count_pred}
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# =====================================================
# UI HELPER FUNCTIONS (Input form unchanged)
# =====================================================
def create_input_form():
    """Create input form for project parameters (UI unchanged)."""
    st.sidebar.header("🏗️ Project Parameters")
    st.sidebar.subheader("Basic Details")
    total_area = st.sidebar.number_input("Total Land Area (sqm)", min_value=100, max_value=200000, value=10000)
    far = st.sidebar.number_input("Floor Area Ratio (FAR)", min_value=0.1, max_value=10.0, value=2.5)
    st.sidebar.subheader("Location Details")
    lat = st.sidebar.number_input("Latitude", min_value=12.0, max_value=14.0, value=12.9300773, format="%.7f")
    long = st.sidebar.number_input("Longitude", min_value=77.0, max_value=78.0, value=77.6950959, format="%.7f")
    st.sidebar.subheader("Project Type and Configuration")
    product_type = st.sidebar.selectbox("Product Type", ["Apartment", "Villa"])
    customize_bhk = st.sidebar.checkbox("Customize BHK Types", value=False)
    selected_bhk_types = None
    if customize_bhk:
        bhk_options = ['1RK', '1BHK', '1_5BHK', '2BHK', '2_5BHK', '3BHK', '3_5BHK', '4BHK', '4_5BHK', '5BHK']
        selected_bhk_types = st.sidebar.multiselect("Select BHK Types", options=bhk_options, default=['2BHK', '3BHK'])
        if not selected_bhk_types:
            st.sidebar.warning("Please select at least one BHK type.")
    return {'total_area_of_land_sqm': total_area, 'far': far, 'latitude': lat, 'longitude': long, 'product_type': product_type, 'selected_bhk_types': selected_bhk_types}

def prepare_input_features(user_input, feature_names):
    """Convert user input to a DataFrame for the model (Unchanged)."""
    location_details = get_all_location_details(user_input['latitude'], user_input['longitude'])
    input_dict = {'total_area_of_land_sqm': user_input['total_area_of_land_sqm'], 'far': user_input['far'], 'built_up_area': user_input['total_area_of_land_sqm'] * user_input['far'], 'product_type_Apartment': 1 if user_input['product_type'] == 'Apartment' else 0, 'product_type_Villa': 1 if user_input['product_type'] == 'Villa' else 0, "airport_distance_kms": location_details["airport_distance_kms"], "ksr_jn_distance_kms": location_details["ksr_jn_distance_kms"], "yeshwantpur_jn_distance_kms": location_details["yeshwantpur_jn_distance_kms"], "nearest_metro_dist_kms": location_details["nearest_metro"][0], "nearest_major_road_dist_kms": location_details["nearest_major_road"][0],}
    taluk_features = [f for f in feature_names if f.startswith('taluk_')]
    for taluk_feature in taluk_features:
        input_dict[taluk_feature] = 0
    selected_taluk_feature = location_details.get("taluk_feature_name")
    if selected_taluk_feature and selected_taluk_feature in taluk_features:
        input_dict[selected_taluk_feature] = 1
    input_df = pd.DataFrame([input_dict])
    return input_df[feature_names]

# =====================================================
# UPDATED: Display function to handle dual analysis
# =====================================================
def display_predictions(predictions, user_input, nearby_analysis):
    """Display prediction results and market context analysis."""
    if not predictions:
        st.error("Unable to generate predictions.")
        return

    # --- Data Preparation Step for Predictions ---
    location_metrics = get_all_location_details(user_input['latitude'], user_input['longitude'])
    unit_counts = predictions.get('unit_counts', {})
    carpet_areas = predictions.get('carpet_areas', {})
    avg_prices = predictions.get('avg_prices', {})
    total_open_area = predictions.get('total_open_area', 0)
    total_project_cost = predictions.get('total_project_cost', 0)
    results, total_units_sum, total_sale_price_sum = [], 0, 0.0
    for bhk, count in unit_counts.items():
        if count > 0:
            area_sqm = carpet_areas.get(bhk, 0)
            price_per_sqft = avg_prices.get(bhk, 0) * 1000
            avg_sale_price_raw = price_per_sqft * area_sqm * 10.764
            total_sale_price_raw = avg_sale_price_raw * count
            total_units_sum += count
            total_sale_price_sum += total_sale_price_raw
            results.append({'BHK Type': bhk, 'No. of Units': count, 'Mean Carpet Area (sqft)': round(area_sqm * 10.764), 'Avg. Price (per sq.ft)': f"₹{int(price_per_sqft):n}", 'Avg. Sale Price (per Unit)': f"₹{avg_sale_price_raw/10**7:.2f} Cr", "Total Sale Price": f"₹{total_sale_price_raw/10**7:.2f} Cr"})
    result_df, final_df = pd.DataFrame(), pd.DataFrame()
    if results:
        result_df = pd.DataFrame(results)
        total_row = pd.DataFrame([{'BHK Type': 'Total', 'No. of Units': total_units_sum, 'Mean Carpet Area (sqft)': '-', 'Avg. Price (per sq.ft)': '-', 'Avg. Sale Price (per Unit)': '-', 'Total Sale Price': f"₹{total_sale_price_sum/10**7:.2f} Cr"}])
        final_df = pd.concat([result_df, total_row], ignore_index=True)

    # --- UI Rendering Step ---
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Location Details", "📈 Market Context", "📊 Predicted Inventory", "💰 Predicted Project Metrics"])
    
    with tab1:
        st.subheader("Your Project's Location Details")
        col1, col2 = st.columns(2)
        with col1: st.metric("Taluk", location_metrics.get("taluk", "N/A"))
        with col2: st.metric("Distance to Airport", f"{location_metrics.get('airport_distance_kms', 0):.1f} kms")
        col3, col4 = st.columns(2)
        with col3: st.metric("Distance to KSR Jn Railway Station", f"{location_metrics['ksr_jn_distance_kms']:.1f} kms", help="KSR Bengaluru City Junction distance")
        with col4: st.metric(f"Distance to Yeshwantpur Railway Station", f"{location_metrics['yeshwantpur_jn_distance_kms']:.1f} kms", help="Yeshwantpur Jn Railway Station")
        col5, col6 = st.columns(2)
        with col5: st.metric("Nearest Metro Station", f"{location_metrics['nearest_metro'][1]}", help="Nearest Metro Station Name")
        with col6: st.metric("Distance to the Metro Station", f"{location_metrics['nearest_metro'][0]:.1f} kms", help="Distance to the nearest Metro Station")
        col7, col8 = st.columns(2)
        with col7: st.metric("Nearest Major Road", f"{location_metrics['nearest_major_road'][1]}", help="Nearest Major Road Name")
        with col8: st.metric("Distance to the Major Road", f"{location_metrics['nearest_major_road'][0]:.1f} kms", help="Distance to the nearest Major Road") 

    with tab2:    
        # --- SECTION 1: Display Analysis for ALL nearby properties ---
        st.subheader(f"Nearby Projects")
        all_nearby_analysis = nearby_analysis.get('all_nearby', {})
        all_nearby_stats = all_nearby_analysis.get('stats', {})
        all_nearby_df = all_nearby_analysis.get('df')

        if all_nearby_stats and all_nearby_stats.get("properties_found", 0) > 0:
            st.success(f"Found **{all_nearby_stats['properties_found']}** projects for a general comparison.")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Mean Project Cost", f"₹ {all_nearby_stats.get('avg_project_cost_cr', 0):.2f} Cr")
            with col2: st.metric("Unit Count Range", f"{all_nearby_stats.get('min_unit_count', 0):.0f} - {all_nearby_stats.get('max_unit_count', 0):.0f}")
            with col3: st.metric("Sale Price (per sq.ft.)", f"₹ {all_nearby_stats.get('min_sale_price_sqft', 0)/1000:.1f}K - ₹ {all_nearby_stats.get('max_sale_price_sqft', 0)/1000:.1f}K")
            col4, col5, col6 = st.columns(3)
            with col4: st.metric("Median Project Cost", f"₹ {all_nearby_stats.get('median_project_cost_cr', 0):.2f} Cr")
            with col5: st.metric("Project Duration", f"{all_nearby_stats.get('min_project_duration', 0):.1f} - {all_nearby_stats.get('max_project_duration', 0):.1f} Years")
            with col6: st.metric("Completion Date", f"{all_nearby_stats.get('min_project_completion_date', 0)} - {all_nearby_stats.get('max_project_completion_date', 0)}")

            with st.expander("View All Nearby Projects Data"):
                if all_nearby_df is not None and not all_nearby_df.empty:
                    st.dataframe(all_nearby_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No registered projects found within a 5km radius in the dataset.")

        st.markdown("---")

        # --- SECTION 2: Display Analysis for SIMILAR properties ---
        st.subheader(f"Nearby Projects: Similar Land Area")
        lower_bound = user_input['total_area_of_land_sqm'] * 0.7
        upper_bound = user_input['total_area_of_land_sqm'] * 1.3
        st.markdown(f"Comparing to projects with a land area between **{lower_bound:,.0f} sqm** and **{upper_bound:,.0f} sqm**.")

        similar_nearby_analysis = nearby_analysis.get('similar_nearby', {})
        similar_stats = similar_nearby_analysis.get('stats', {})
        similar_df = similar_nearby_analysis.get('df')

        if similar_stats and similar_stats.get("properties_found", 0) > 0:
            st.success(f"Found **{similar_stats['properties_found']}** similar sized projects for a direct comparison.")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Mean Project Cost", f"₹ {similar_stats.get('avg_project_cost_cr', 0):.2f} Cr")
            with col2: st.metric("Unit Count Range", f"{similar_stats.get('min_unit_count', 0):.0f} - {similar_stats.get('max_unit_count', 0):.0f}")
            with col3: st.metric("Sale Price (per sq.ft.)", f"₹ {similar_stats.get('min_sale_price_sqft', 0)/1000:.1f}K - ₹ {similar_stats.get('max_sale_price_sqft', 0)/1000:.1f}K")
            col4, col5, col6 = st.columns(3)
            with col4: st.metric("Median Project Cost", f"₹ {similar_stats.get('median_project_cost_cr', 0):.2f} Cr")
            with col5: st.metric("Project Duration", f"{similar_stats.get('min_project_duration', 0):.1f} - {similar_stats.get('max_project_duration', 0):.1f} Years")
            with col6: st.metric("Completion Date", f"{similar_stats.get('min_project_completion_date', 0)} - {similar_stats.get('max_project_completion_date', 0)}")

            with st.expander("View Similar Sized Projects Data"):
                if similar_df is not None and not similar_df.empty:
                    st.dataframe(similar_df, use_container_width=True, hide_index=True)
        else:
            st.info("No projects of a comparable size were found in the immediate vicinity.")

    with tab3:
        st.subheader("Predicted Inventory Configuration")
        if user_input.get('selected_bhk_types'): st.info("🎯 Predictions based on your selected BHK types.")
        else: st.info("🤖 AI-predicted BHK types.")
        if not final_df.empty:
            st.dataframe(final_df, use_container_width=True, hide_index=True)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(px.pie(result_df, values='No. of Units', names='BHK Type', title="Predicted Unit Mix"), use_container_width=True)
            with col2: st.plotly_chart(px.bar(result_df, x='BHK Type', y='Mean Carpet Area (sqft)', title="Predicted Carpet Areas"), use_container_width=True)
        else:
            st.warning("No units predicted for the given configuration.")

    with tab4:
        st.subheader("Overall Predicted Project Metrics")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Open Area", f"{total_open_area:,.0f} sqm")
        with col2: st.metric("Total Project Cost", f"₹ {total_project_cost / 10**7:.2f} Cr")
        with col3: st.metric("Total Sale Price", f"₹ {total_sale_price_sum / 10**7:.2f} Cr")

# =====================================================
# Main app logic
# =====================================================
def main():
    st.title("🏠 Real Estate Project Predictor")
    
    models, scalers, feature_names, bhk_mapping, models_loaded = load_models()
    training_df = load_training_data('processing_outputs/price_predicted_data.csv')
    
    if not models_loaded or training_df is None:
        st.error("Application cannot start. Please check that all model and data files are present and correctly named.")
        return
    
    st.success("✅ Models and analysis data loaded successfully!")
    
    user_input = create_input_form()

    if st.sidebar.button("🔮 Generate Predictions & Analysis", type="primary", use_container_width=True):
        if user_input['selected_bhk_types'] is not None and not user_input['selected_bhk_types']:
            st.sidebar.error("Please select at least one BHK type or disable customization.")
        else:
            with st.spinner("🧠 Running AI predictions and market analysis..."):
                # 1. Get model predictions
                input_df = prepare_input_features(user_input, feature_names)
                predictions = predict_project(input_df, models, scalers, feature_names, bhk_mapping, user_input['selected_bhk_types'])
                st.session_state['predictions'] = predictions
                
                # 2. Get nearby market analysis
                analysis_results = get_nearby_properties_analysis(
                    user_input['latitude'], 
                    user_input['longitude'], 
                    5, # radius in km
                    training_df,
                    user_input['total_area_of_land_sqm'] # Pass the user's land area
                )
                st.session_state['nearby_analysis'] = analysis_results
                
                st.session_state['user_input'] = user_input
    
    if 'predictions' in st.session_state and 'nearby_analysis' in st.session_state:
        display_predictions(
            st.session_state['predictions'], 
            st.session_state['user_input'],
            st.session_state['nearby_analysis']
        )
    else:
        st.info("👈 Enter your project's parameters in the sidebar and click the button to generate.")

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>**Note:** Predictions are based on historical data patterns and should be used as estimates for planning purposes.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
