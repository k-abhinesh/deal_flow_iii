import streamlit as st
from ml_functions import load_models, predict_project, prepare_input_features
from finance_functions import validate_schedules, calculate_financial_model
from ui_components import render_prediction_stage, render_financial_model_ui
from nearby_analysis import load_training_data
import warnings
warnings.filterwarnings('ignore')

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
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False


# =====================================================
# Main App Router
# =====================================================
def main():
    st.title("Integrated Real Estate Analysis & Financial Model")
    
    # Load all necessary data once
    if not st.session_state.models_loaded:
        models, scalers, feature_names, bhk_mapping, loaded_ok = load_models()
        if loaded_ok:
            st.session_state.models = models
            st.session_state.scalers = scalers
            st.session_state.feature_names = feature_names
            st.session_state.bhk_mapping = bhk_mapping
            st.session_state.training_df = load_training_data('processing_outputs/price_predicted_data.csv')
            st.session_state.models_loaded = True
        else:
            st.error("Application cannot start. Please check that all model and data files are present.")
            return

    # --- View Router ---
    if st.session_state.view == 'predictor':
        render_prediction_stage()
    elif st.session_state.view == 'financial_model':
        render_financial_model_ui()

if __name__ == "__main__":
    main()
