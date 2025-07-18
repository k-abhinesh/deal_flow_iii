import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from location_processing import get_all_location_details

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
        print("BHK_types_predicted:", predicted_bhk_types)
        unit_counts, carpet_areas, avg_prices = {}, {}, {}
        for bhk_str in predicted_bhk_types:
            bhk_numeric = bhk_mapping[bhk_str]
            features_carpet = np.append(X_base, [bhk_numeric]).reshape(1, -1)
            print("Features_carpet",len(features_carpet))
            mean_carpet_area_pred = max(0, models['carpet_area'].predict(scalers['carpet_area'].transform(features_carpet))[0])
            carpet_areas[bhk_str] = mean_carpet_area_pred
            features_count = np.append(features_carpet, [total_unit_count_pred,mean_carpet_area_pred]).reshape(1, -1)
            print("Features_count",len(features_count))
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
