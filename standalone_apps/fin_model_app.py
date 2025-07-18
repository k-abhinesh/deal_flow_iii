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
from nearby_analysis import load_training_data, get_nearby_properties_analysis
import json
from urllib.parse import urlencode
import dynamic_schedules # We will keep the logic separate for cleanliness
from datetime import date

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
    page_title="Integrated Real Estate Analysis & Financial Model",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# INITIALIZE SESSION STATE
# =====================================================
if 'view' not in st.session_state:
    st.session_state.view = 'predictor'
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'nearby_analysis' not in st.session_state:
    st.session_state.nearby_analysis = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}
if 'fm_defaults' not in st.session_state:
    st.session_state.fm_defaults = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'schedules_generated' not in st.session_state:
    st.session_state.schedules_generated = False
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0
if 'dynamic_end_date' not in st.session_state:
    st.session_state.dynamic_end_date = None


# ======================================================================================
# ALL FUNCTIONS FROM BOTH APPS (ML and Financial) ARE PLACED HERE
# ======================================================================================

# --- ML Model Functions ---
@st.cache_resource
def load_models(model_dir='models'):
    """Load all trained models, scalers, and feature names."""
    try:
        models, scalers = {}, {}
        model_names = ['bhk_presence', 'unit_count', 'carpet_area', 'bhk_unit_count', 'avg_price', 'total_project_cost', 'open_area']
        for name in model_names:
            with open(os.path.join(model_dir, f'{name}_model.pkl'), 'rb') as f: models[name] = pickle.load(f)
            with open(os.path.join(model_dir, f'{name}_scaler.pkl'), 'rb') as f: scalers[name] = pickle.load(f)
        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f: feature_names = pickle.load(f)
        bhk_mapping = {'1RK': 0.5, '1BHK': 1.0, '1_5BHK': 1.5, '2BHK': 2.0, '2_5BHK': 2.5, '3BHK': 3.0, '3_5BHK': 3.5, '4BHK': 4.0, '4_5BHK': 4.5, '5BHK': 5.0}
        return models, scalers, feature_names, bhk_mapping, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

def predict_project(input_df, models, scalers, feature_names, bhk_mapping, selected_bhk_types=None):
    """Make predictions for new project data."""
    try:
        X_base = input_df[feature_names].values.reshape(1, -1)
        bhk_presence_pred = np.array([[1 if bhk in selected_bhk_types else 0 for bhk in bhk_mapping.keys()]]) if selected_bhk_types else (models['bhk_presence'].predict(scalers['bhk_presence'].transform(X_base)) > 0.5).astype(int)
        X_enhanced_total_units = np.hstack([X_base, bhk_presence_pred])
        total_unit_count_pred = max(0, round(models['unit_count'].predict(scalers['unit_count'].transform(X_enhanced_total_units))[0]))
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
        st.error(f"Prediction error: {e}")
        return None

def prepare_input_features(user_input, feature_names):
    """Convert user input to a DataFrame for the model."""
    location_details = get_all_location_details(user_input['latitude'], user_input['longitude'])
    input_dict = {'total_area_of_land_sqm': user_input['total_area_of_land_sqm'], 'far': user_input['far'], 'built_up_area': user_input['total_area_of_land_sqm'] * user_input['far'], 'product_type_Apartment': 1 if user_input['product_type'] == 'Apartment' else 0, 'product_type_Villa': 1 if user_input['product_type'] == 'Villa' else 0, "airport_distance_kms": location_details["airport_distance_kms"], "ksr_jn_distance_kms": location_details["ksr_jn_distance_kms"], "yeshwantpur_jn_distance_kms": location_details["yeshwantpur_jn_distance_kms"], "nearest_metro_dist_kms": location_details["nearest_metro"][0], "nearest_major_road_dist_kms": location_details["nearest_major_road"][0],}
    taluk_features = [f for f in feature_names if f.startswith('taluk_')]
    for taluk_feature in taluk_features: input_dict[taluk_feature] = 0
    selected_taluk_feature = location_details.get("taluk_feature_name")
    if selected_taluk_feature and selected_taluk_feature in taluk_features: input_dict[selected_taluk_feature] = 1
    return pd.DataFrame([input_dict])[feature_names]

# --- Financial Model Functions ---
def validate_schedules(inputs):
    """Checks if the distribution percentages sum to 100%."""
    errors = []
    tolerance = 0.001
    df_construct = inputs['construction_phasing']
    q_cols_construct = [col for col in df_construct.columns if isinstance(col, str) and col.startswith('Q')]
    for _, row in df_construct.iterrows():
        total = row[q_cols_construct].sum()
        if abs(total - 1.0) > tolerance: errors.append(f"Construction Phasing Error: '{row['Stage']}' sums to {total:.2%}.")
    df_other = inputs['other_costs_phasing']
    q_cols_other = [col for col in df_other.columns if isinstance(col, str) and col.startswith('Q')]
    for _, row in df_other.iterrows():
        total = row[q_cols_other].sum()
        if abs(total - 1.0) > tolerance and total > 0: errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' sums to {total:.2%}.")
    if abs(inputs['sales_dist']['Distribution'].sum() - 1.0) > tolerance: errors.append("Sales Distribution Error: Does not sum to 100%.")
    if abs(inputs['collection_dist']['Distribution'].sum() - 1.0) > tolerance: errors.append("Collections Distribution Error: Does not sum to 100%.")
    return errors

def calculate_financial_model(inputs):
    """Main function to run all financial calculations."""
    intermediate_data = {}
    all_q_strs = set()
    for key in ['construction_phasing', 'other_costs_phasing', 'sales_dist', 'collection_dist', 'admin_cost']:
        df = inputs.get(key)
        if df is None or df.empty: continue
        quarter_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Q')]
        if not quarter_cols and 'Quarter' in df.columns: quarter_cols = df['Quarter'].unique()
        all_q_strs.update(quarter_cols)
    all_timestamps = [pd.Period(f"{q.split(' ')[1]}{q.split(' ')[0]}", freq='Q').to_timestamp() for q in all_q_strs if q and ' ' in q]
    start_date, end_date = (min(all_timestamps), max(all_timestamps)) if all_timestamps else (pd.to_datetime(inputs['start_date']), st.session_state.get('dynamic_end_date', pd.to_datetime(inputs['start_date']) + pd.DateOffset(years=3)))
    master_timeline = pd.date_range(start=start_date, end=end_date, freq='Q')
    master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline]
    
    total_bua = inputs['land_area'] * inputs['far']
    total_saleable_area = inputs.get('total_saleable_area', total_bua * inputs.get('efficiency', 0.45) * 1.1)
    townhouse_sa = total_saleable_area - inputs.get('clubhouse_area', 0)
    intermediate_data['area_calcs'] = pd.DataFrame([{"Metric": "Land Area (sq ft)", "Value": inputs['land_area']}, {"Metric": "FAR", "Value": inputs['far']}, {"Metric": "Total BUA (sq ft)", "Value": total_bua}, {"Metric": "Total Saleable Area (sq ft)", "Value": total_saleable_area}]).set_index("Metric")
    
    sales_dist = pd.Series(inputs['sales_dist']['Distribution'].fillna(0).values, index=inputs['sales_dist']['Quarter'].values)
    first_sale_quarter = sales_dist[sales_dist > 0].index[0] if not sales_dist[sales_dist > 0].empty else None
    project_timeline = pd.date_range(start=inputs['start_date'], end=inputs['end_date'], freq='Q')
    project_timeline_labels = [f"Q{q.quarter} {q.year}" for q in project_timeline]
    sales_prices, escalation_count, sales_started = [], 0, False
    for quarter_label in master_timeline_labels:
        if not sales_started and quarter_label == first_sale_quarter: sales_started = True
        price = inputs['unit_price']
        if sales_started and quarter_label in project_timeline_labels:
            price *= ((1 + inputs['escalation']) ** escalation_count); escalation_count += 1
        sales_prices.append(price)
    revenue_details = pd.DataFrame({'Quarterly Sales Distribution': sales_dist.reindex(master_timeline_labels, fill_value=0), 'Escalated Price per Sq Ft': pd.Series(sales_prices, index=master_timeline_labels)})
    revenue_details['Quarterly Revenue'] = townhouse_sa * revenue_details['Quarterly Sales Distribution'] * revenue_details['Escalated Price per Sq Ft']
    total_revenue = revenue_details['Quarterly Revenue'].sum()
    intermediate_data['revenue_details'] = revenue_details
    
    construction_stage_costs = inputs['construction_stage_costs'].set_index('Stage')
    construction_stage_costs['Total Cost'] = construction_stage_costs['Cost Rate (per sq ft of Total SA)'].fillna(0) * total_saleable_area
    total_construction_cost = construction_stage_costs['Total Cost'].sum()
    total_consultant_cost, total_marketing_cost, total_misc_approval_cost = total_saleable_area * inputs['consultant_cost_psf'], total_revenue * inputs['marketing_cost_pct_revenue'], total_saleable_area * inputs['misc_approval_cost_psf']
    cost_summary = pd.DataFrame([{"Cost Category": "Total Construction Cost", "Value": total_construction_cost}, {"Cost Category": "Total Consultant Cost", "Value": total_consultant_cost}, {"Cost Category": "Total Marketing Cost", "Value": total_marketing_cost}, {"Cost Category": "Total Misc. Approval Cost", "Value": total_misc_approval_cost}, {"Cost Category": "Plan Sanction (Lump Sum)", "Value": inputs['plan_sanction_cost']}, {"Cost Category": "Sales Lounge (Lump Sum)", "Value": inputs['sales_lounge_cost']}]).set_index("Cost Category")
    intermediate_data['cost_summary'] = cost_summary
    
    cashflow = pd.DataFrame(index=master_timeline_labels)
    collection_dist_series = pd.Series(inputs['collection_dist']['Distribution'].fillna(0).values, index=inputs['collection_dist']['Quarter'].values)
    cashflow['Collections'] = collection_dist_series.reindex(master_timeline_labels, fill_value=0) * total_revenue
    total_brokerage_cost = cashflow['Collections'].sum() * inputs['brokerage_fee']
    cashflow['Cost: Brokerage'] = cashflow['Collections'] * inputs['brokerage_fee']
    cost_summary.loc["Total Brokerage Cost"] = total_brokerage_cost
    
    construction_phasing = inputs['construction_phasing'].set_index('Stage')
    for stage_name, stage_data in construction_stage_costs.iterrows():
        cashflow[f'Cost: {stage_name}'] = construction_phasing.loc[stage_name].reindex(master_timeline_labels, fill_value=0).astype(float) * stage_data['Total Cost']
    
    other_costs_phasing = inputs['other_costs_phasing'].set_index('Cost Item')
    cost_items_map = {'Consultant Cost': total_consultant_cost, 'Marketing Cost': total_marketing_cost, 'Misc. Approval Cost': total_misc_approval_cost, 'Plan Sanction': inputs['plan_sanction_cost'], 'Sales Lounge': inputs['sales_lounge_cost']}
    for item, total_cost in cost_items_map.items():
        if item in other_costs_phasing.index: cashflow[f'Cost: {item}'] = other_costs_phasing.loc[item].reindex(master_timeline_labels, fill_value=0).astype(float) * total_cost
    
    admin_cost_schedule = pd.Series(inputs['admin_cost']['Cost per Quarter'].fillna(0).values, index=inputs['admin_cost']['Quarter'].values)
    cashflow['Cost: Admin'] = admin_cost_schedule.reindex(master_timeline_labels, fill_value=0)
    cost_columns = [col for col in cashflow.columns if 'Cost:' in col]
    cashflow[cost_columns] = cashflow[cost_columns].fillna(0)
    cashflow['Total Expenses'] = cashflow[cost_columns].sum(axis=1)
    cashflow['EBITDA'] = cashflow['Collections'] - cashflow['Total Expenses']
    
    financials = pd.DataFrame({'Surplus/(Deficit)': cashflow['EBITDA']}, index=master_timeline_labels)
    opening_cash, equity_required, closing_cash_list, equity_inflow_list = 0.0, 0, [], []
    for deficit in financials['Surplus/(Deficit)']:
        pre_funding_balance = opening_cash + deficit
        inflow = -pre_funding_balance if pre_funding_balance < 0 else 0
        equity_required += inflow
        equity_inflow_list.append(inflow)
        closing_cash = pre_funding_balance + inflow
        closing_cash_list.append(closing_cash)
        opening_cash = closing_cash
    
    financials['Closing Cash'] = closing_cash_list
    financials['Equity Inflow'] = equity_inflow_list
    financials['Opening Cash'] = financials['Closing Cash'].shift(1).fillna(0)
    
    total_project_cost = cashflow['Total Expenses'].sum()
    pat = (total_revenue - total_project_cost) * (1 - 0.25)
    kpis = {"Projected Revenue": f"₹{total_revenue/1e7:.2f} Cr", "Total Construction Cost": f"₹{total_construction_cost/1e7:.2f} Cr", "Total Project Cost": f"₹{total_project_cost/1e7:.2f} Cr", "Profit After Tax": f"₹{pat/1e7:.2f} Cr", "Equity Required": f"₹{equity_required/1e7:.2f} Cr"}
    return {"kpis": kpis, "cashflow": cashflow, "financials": financials, "intermediate": intermediate_data, "master_timeline": master_timeline}

# =====================================================
# UI RENDERING FUNCTIONS
# =====================================================

def render_prediction_stage():
    """Renders the entire ML prediction stage, including sidebar and main content."""
    # --- 1. Render Sidebar and Get Inputs ---
    st.sidebar.header("🏗️ Project Parameters")
    st.sidebar.subheader("Basic Details")
    total_area = st.sidebar.number_input("Total Land Area (sqm)", min_value=100, max_value=200000, value=10000, key="ml_land_area")
    far = st.sidebar.number_input("Floor Area Ratio (FAR)", min_value=0.1, max_value=10.0, value=2.5, key="ml_far")
    st.sidebar.subheader("Location Details")
    lat = st.sidebar.number_input("Latitude", min_value=12.0, max_value=14.0, value=12.9300773, format="%.7f", key="ml_lat")
    long = st.sidebar.number_input("Longitude", min_value=77.0, max_value=78.0, value=77.6950959, format="%.7f", key="ml_long")
    st.sidebar.subheader("Project Type and Configuration")
    product_type = st.sidebar.selectbox("Product Type", ["Apartment", "Villa"], key="ml_product_type")
    customize_bhk = st.sidebar.checkbox("Customize BHK Types", value=False, key="ml_customize_bhk")
    selected_bhk_types = None
    if customize_bhk:
        bhk_options = ['1RK', '1BHK', '1_5BHK', '2BHK', '2_5BHK', '3BHK', '3_5BHK', '4BHK', '4_5BHK', '5BHK']
        selected_bhk_types = st.sidebar.multiselect("Select BHK Types", options=bhk_options, default=['2BHK', '3BHK'], key="ml_bhk_select")
        if not selected_bhk_types: st.sidebar.warning("Please select at least one BHK type.")
    
    user_input = {'total_area_of_land_sqm': total_area, 'far': far, 'latitude': lat, 'longitude': long, 'product_type': product_type, 'selected_bhk_types': selected_bhk_types}

    # --- 2. Handle Button Click ---
    if st.sidebar.button("🔮 Generate Predictions & Analysis", type="primary", use_container_width=True):
        if customize_bhk and not selected_bhk_types:
            st.sidebar.error("Please select at least one BHK type or disable customization.")
        else:
            with st.spinner("🧠 Running AI predictions and market analysis..."):
                input_df = prepare_input_features(user_input, st.session_state.feature_names)
                st.session_state.predictions = predict_project(input_df, st.session_state.models, st.session_state.scalers, st.session_state.feature_names, st.session_state.bhk_mapping, user_input['selected_bhk_types'])
                st.session_state.nearby_analysis = get_nearby_properties_analysis(lat, long, 5, st.session_state.training_df, total_area)
                st.session_state.user_input = user_input
    
    # --- 3. Render Main Content ---
    if st.session_state.predictions:
        st.header("Step 1: Prediction Results")
        predictions = st.session_state.predictions
        user_input = st.session_state.user_input
        nearby_analysis = st.session_state.nearby_analysis
        
        location_metrics = get_all_location_details(user_input['latitude'], user_input['longitude'])
        unit_counts, carpet_areas, avg_prices = predictions.get('unit_counts', {}), predictions.get('carpet_areas', {}), predictions.get('avg_prices', {})
        total_open_area, total_project_cost = predictions.get('total_open_area', 0), predictions.get('total_project_cost', 0)
        results, total_units_sum, total_sale_price_sum, total_weighted_price_sqft, total_carpet_area_sqft = [], 0, 0.0, 0, 0
        
        for bhk, count in unit_counts.items():
            if count > 0:
                area_sqm, area_sqft = carpet_areas.get(bhk, 0), carpet_areas.get(bhk, 0) * 10.764
                price_per_sqft = avg_prices.get(bhk, 0) * 1000
                avg_sale_price_raw, total_sale_price_raw = price_per_sqft * area_sqft, price_per_sqft * area_sqft * count
                total_units_sum += count
                total_sale_price_sum += total_sale_price_raw
                total_carpet_area_sqft += area_sqft * count
                total_weighted_price_sqft += price_per_sqft * area_sqft * count
                results.append({'BHK Type': bhk, 'No. of Units': count, 'Mean Carpet Area (sqft)': round(area_sqft), 'Avg. Price (per sq.ft)': f"₹{int(price_per_sqft):n}", 'Avg. Sale Price (per Unit)': f"₹{avg_sale_price_raw/10**7:.2f} Cr", "Total Sale Price": f"₹{total_sale_price_raw/10**7:.2f} Cr"})
        
        final_df = pd.DataFrame()
        if results:
            result_df = pd.DataFrame(results)
            total_row = pd.DataFrame([{'BHK Type': 'Total', 'No. of Units': total_units_sum, 'Mean Carpet Area (sqft)': '-', 'Avg. Price (per sq.ft)': '-', 'Avg. Sale Price (per Unit)': '-', 'Total Sale Price': f"₹{total_sale_price_sum/10**7:.2f} Cr"}])
            final_df = pd.concat([result_df, total_row], ignore_index=True)
        
        overall_avg_price_psf = total_weighted_price_sqft / total_carpet_area_sqft if total_carpet_area_sqft > 0 else 0
        total_saleable_area = total_carpet_area_sqft * 1.1

        tab1, tab2, tab3, tab4 = st.tabs(["📍 Location Details", "📈 Market Context", "📊 Predicted Inventory", "💰 Predicted Project Metrics"])
        with tab1:
            st.subheader("Your Project's Location Details")
            col1, col2 = st.columns(2)
            with col1: st.metric("Taluk", location_metrics.get("taluk", "N/A"))
            with col2: st.metric("Distance to Airport", f"{location_metrics.get('airport_distance_kms', 0):.1f} kms")
        with tab2:
            st.subheader(f"Nearby Projects")
            all_nearby_analysis = nearby_analysis.get('all_nearby', {})
            all_nearby_stats = all_nearby_analysis.get('stats', {})
            if all_nearby_stats and all_nearby_stats.get("properties_found", 0) > 0:
                st.success(f"Found **{all_nearby_stats['properties_found']}** projects for a general comparison.")
        with tab3:
            st.subheader("Predicted Inventory Configuration")
            if not final_df.empty: st.dataframe(final_df, use_container_width=True, hide_index=True)
        with tab4:
            st.subheader("Overall Predicted Project Metrics")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Open Area", f"{total_open_area:,.0f} sqm")
            with col2: st.metric("Total Project Cost", f"₹ {total_project_cost / 10**7:.2f} Cr")
            with col3: st.metric("Total Sale Price", f"₹ {total_sale_price_sum / 10**7:.2f} Cr")

        st.markdown("---")
        st.header("Step 2: Proceed to Financial Model")
        if st.button("➡️ Launch Financial Model with these Predictions", type="primary"):
            st.session_state.fm_defaults = {
                'land_area': round(user_input['total_area_of_land_sqm'] * 10.764),
                'far': user_input['far'],
                'unit_price': round(overall_avg_price_psf),
                'total_saleable_area': round(total_saleable_area)
            }
            st.session_state.view = 'financial_model'
            st.rerun()
    else:
        st.header("Step 1: Generate Market Predictions")
        st.info("👈 Enter your project's parameters in the sidebar and click 'Generate Predictions' to begin.")

def render_financial_model_ui():
    """Renders the UI for the Financial Modeling part of the app."""
    st.header("Step 2: Dynamic Financial Model")
    
    with st.sidebar:
        st.header("Financial Model Inputs")
        st.info("Inputs are pre-filled from ML predictions. You can edit them as needed.")
        if st.button("⬅️ Back to Predictions"):
            st.session_state.view = 'predictor'
            st.session_state.predictions = None
            st.session_state.results = None
            st.session_state.schedules_generated = False
            st.rerun()

        st.subheader("1. Project Definition")
        defaults = st.session_state.get('fm_defaults', {})
        base_inputs = {
            'land_area': st.number_input("Land Area (sq ft)", value=defaults.get('land_area', 217800)),
            'far': st.number_input("FAR (Floor Area Ratio)", value=defaults.get('far', 1.75)),
            'num_floors': st.number_input("Number of Floors", value=14),
            'unit_price': st.number_input("Base Price (per sq ft of SA)", value=defaults.get('unit_price', 17500)),
            'start_date': st.date_input("Project Start Date", value=pd.to_datetime("2025-06-30")),
            'total_saleable_area': st.number_input("Total Saleable Area (sq ft)", value=defaults.get('total_saleable_area', 300000))
        }
        
        st.subheader("2. Financial & Cost Assumptions")
        financial_assumptions = {
            'escalation': st.slider("Price Escalation (% per Quarter)", 0.0, 5.0, 1.5) / 100,
            'brokerage_fee': st.slider("Brokerage Fee (% of Collections)", 0.0, 5.0, 2.95) / 100,
            'marketing_cost_pct_revenue': st.slider("Marketing Cost (% of Revenue)", 0.0, 10.0, 3.0) / 100,
            'plan_sanction_cost': st.number_input("Plan Sanction Cost (Lump Sum)", value=40000000),
            'sales_lounge_cost': st.number_input("Sales Lounge Cost (Lump Sum)", value=50000000),
            'consultant_cost_psf': st.number_input("Consultant Cost (per sq ft of Total SA)", value=300),
            'misc_approval_cost_psf': st.number_input("Misc. Approval Cost (per sq ft of Total SA)", value=50),
        }
        base_inputs.update(financial_assumptions)
        
        st.subheader("Admin Timeline")
        admin_dates = {'admin_start_date': st.date_input("Admin Cost Start Date", value=pd.to_datetime("2025-06-30")), 'admin_end_date': st.date_input("Admin Cost End Date", value=pd.to_datetime("2032-09-30"))}
        base_inputs.update(admin_dates)

        st.subheader("3. Detailed Phasing Schedules")
        st.info("Set primary inputs above, then generate schedules.")

        def generate_schedules(current_inputs):
            st.session_state.reset_counter += 1
            construction_phasing_df, dynamic_end_date = dynamic_schedules.generate_construction_phasing(current_inputs['start_date'], current_inputs['land_area'], current_inputs['num_floors'])
            st.session_state.dynamic_end_date, st.session_state.construction_phasing_df = dynamic_end_date, construction_phasing_df
            post_project_q, extended_end_date = pd.Period(dynamic_end_date, freq='Q') + 8, (pd.Period(dynamic_end_date, freq='Q') + 8).end_time
            admin_end_date_ts = pd.to_datetime(current_inputs['admin_end_date'])
            master_timeline_range = pd.date_range(start=current_inputs['start_date'], end=max(extended_end_date, admin_end_date_ts), freq='Q')
            master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline_range]
            st.session_state.other_costs_phasing_df = dynamic_schedules.generate_other_costs_phasing(master_timeline_labels, dynamic_end_date)
            st.session_state.sales_dist_df = dynamic_schedules.generate_sales_distribution(master_timeline_labels, dynamic_end_date)
            st.session_state.collection_dist_df = dynamic_schedules.generate_collection_distribution(st.session_state.sales_dist_df, master_timeline_labels)
            admin_timeline_range = pd.date_range(start=current_inputs['admin_start_date'], end=current_inputs['admin_end_date'], freq='Q')
            admin_timeline_labels = [f"Q{q.quarter} {q.year}" for q in admin_timeline_range]
            st.session_state.admin_cost_df = dynamic_schedules.generate_admin_cost_schedule(current_inputs['total_saleable_area'], admin_timeline_labels)
            st.session_state.schedules_generated = True
            st.success(f"Schedules generated. Project end date: {dynamic_end_date.strftime('%Y-%m-%d')}")

        if st.button("Generate / Reset Schedules ⚙️", type="primary") or not st.session_state.schedules_generated:
            generate_schedules(base_inputs)

        schedule_inputs = {}
        with st.expander("🔨 Construction Costs & Phasing", expanded=True):
            schedule_inputs['construction_stage_costs'] = st.data_editor(pd.DataFrame([{"Stage": "Excavation and Foundation", "Cost Rate (per sq ft of Total SA)": 900}, {"Stage": "RCC", "Cost Rate (per sq ft of Total SA)": 1200}, {"Stage": "MEP", "Cost Rate (per sq ft of Total SA)": 800}, {"Stage": "Finishing", "Cost Rate (per sq ft of Total SA)": 1400}, {"Stage": "Infra and Amenities", "Cost Rate (per sq ft of Total SA)": 700}]), key="csc_editor", hide_index=True)
            schedule_inputs['construction_phasing'] = st.data_editor(st.session_state.get('construction_phasing_df', pd.DataFrame()), key=f"cp_editor_{st.session_state.reset_counter}", hide_index=True)
        with st.expander("💰 Sales & Collections Phasing"):
            schedule_inputs['sales_dist'] = st.data_editor(st.session_state.get('sales_dist_df', pd.DataFrame()), key=f"sd_editor_{st.session_state.reset_counter}", hide_index=True)
            schedule_inputs['collection_dist'] = st.data_editor(st.session_state.get('collection_dist_df', pd.DataFrame()), key=f"cd_editor_{st.session_state.reset_counter}", hide_index=True)
        with st.expander("📋 Other Costs Phasing"):
            schedule_inputs['other_costs_phasing'] = st.data_editor(st.session_state.get('other_costs_phasing_df', pd.DataFrame()), key=f"ocp_editor_{st.session_state.reset_counter}", hide_index=True)
        with st.expander("👥 Admin Cost Schedule"):
            schedule_inputs['admin_cost'] = st.data_editor(st.session_state.get('admin_cost_df', pd.DataFrame()), key=f"ac_editor_{st.session_state.reset_counter}", hide_index=True)

        if st.button("Calculate Project Financials 🚀", type="primary"):
            final_inputs = {**base_inputs, **schedule_inputs}
            final_inputs['end_date'] = st.session_state.get('dynamic_end_date', base_inputs.get('start_date'))
            validation_errors = validate_schedules(final_inputs)
            if validation_errors:
                for error in validation_errors: st.error(error)
            else:
                with st.spinner("Running financial model..."): st.session_state['results'] = calculate_financial_model(final_inputs)

    # --- Main Panel for Outputs ---
    if st.session_state['results']:
        results, kpis, cashflow, financials, master_timeline, intermediate = st.session_state['results'], st.session_state['results']['kpis'], st.session_state['results']['cashflow'], st.session_state['results']['financials'], st.session_state['results']['master_timeline'], st.session_state['results']['intermediate']
        CR = 1e7
        colors = {'collections': '#63b179', 'expenses': '#e67c73', 'net_cash': '#4285f4', 'equity': '#f4b400', 'surplus': '#8f8f8f'}
        table_styles = [{'selector': 'th, td', 'props': [('text-align', 'center')]}]
        cashflow_display, financials_display = (cashflow / CR), (financials / CR)
        cost_summary_display = intermediate['cost_summary'].copy(); cost_summary_display['Value'] /= CR
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dashboard", "🏗️ Area & Revenue", "💰 Cost Breakdowns", "🧾 Cash Flow Details", "📊 Funding & Financials"])
        with tab1:
            st.header("Financial Summary")
            cols = st.columns(len(kpis))
            for i, (key, value) in enumerate(kpis.items()): cols[i].metric(key, value)
            st.markdown("---"); st.header("Visualizations"); st.subheader("Quarterly Cash Flow Analysis")
            fig_cashflow = go.Figure()
            fig_cashflow.add_trace(go.Bar(x=master_timeline, y=cashflow_display['Collections'], name='Collections (Cr)', marker_color=colors['collections']))
            fig_cashflow.add_trace(go.Bar(x=master_timeline, y=-cashflow_display['Total Expenses'], name='Expenses (Cr)', marker_color=colors['expenses']))
            fig_cashflow.add_trace(go.Scatter(x=master_timeline, y=cashflow_display['EBITDA'], name='Net Cash Flow (Cr)', mode='lines+markers', line=dict(color=colors['net_cash'])))
            fig_cashflow.update_layout(barmode='relative', height=400, title_text="Quarterly Inflows, Outflows, and Net Position", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Amount (in Cr)")
            st.plotly_chart(fig_cashflow, use_container_width=True)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Quarterly Funding View (in Cr)")
                fig_funding = go.Figure()
                fig_funding.add_trace(go.Bar(x=master_timeline, y=financials_display['Equity Inflow'], name='Equity Inflow', marker_color=colors['equity']))
                fig_funding.add_trace(go.Bar(x=master_timeline, y=financials_display['Surplus/(Deficit)'], name='Surplus/(Deficit)', marker_color=colors['surplus']))
                fig_funding.update_layout(barmode='relative', height=400, yaxis_title="Amount (in Cr)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_funding, use_container_width=True)
            with col2:
                st.subheader("Cash Balance Over Time (in Cr)")
                fig_balance = go.Figure()
                fig_balance.add_trace(go.Scatter(x=master_timeline, y=financials_display['Closing Cash'], name='Cash Balance', fill='tozeroy', line=dict(color=colors['net_cash'])))
                fig_balance.update_layout(height=400, yaxis_title="Amount (in Cr)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_balance, use_container_width=True)
        with tab2:
            st.header("Area & Revenue Calculations")
            st.dataframe(intermediate['area_calcs'].style.set_table_styles(table_styles).format(formatter={"Value": lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x}))
        with tab3:
            st.header("Total Cost Breakdowns")
            st.dataframe(cost_summary_display.style.set_table_styles(table_styles).format("₹{:,.2f} Cr"))
        with tab4:
            st.header("Quarterly Cash Flow Details (in Cr)")
            st.dataframe(cashflow_display.style.set_table_styles(table_styles).format("₹{:,.2f}"))
        with tab5:
            st.header("Funding & Financials (in Cr)")
            st.dataframe(financials_display.style.set_table_styles(table_styles).format("₹{:,.2f}"))

# =====================================================
# Main App Router
# =====================================================
def main():
    st.title("Integrated Real Estate Analysis & Financial Model")
    
    # Load all necessary data once
    if 'models_loaded' not in st.session_state:
        st.session_state.models, st.session_state.scalers, st.session_state.feature_names, st.session_state.bhk_mapping, st.session_state.models_loaded = load_models()
        st.session_state.training_df = load_training_data('processing_outputs/price_predicted_data.csv')
    
    if not st.session_state.models_loaded or st.session_state.training_df is None:
        st.error("Application cannot start. Please check that all model and data files are present.")
        return
    
    # --- View Router ---
    if st.session_state.view == 'predictor':
        render_prediction_stage()
    elif st.session_state.view == 'show_predictions':
        render_prediction_stage()
    elif st.session_state.view == 'financial_model':
        render_financial_model_ui()

if __name__ == "__main__":
    main()
