import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from location_processing import get_all_location_details
import dynamic_schedules
from finance_functions import validate_schedules, calculate_financial_model
from ml_functions import prepare_input_features, predict_project
from nearby_analysis import get_nearby_properties_analysis
import plotly.express as px
import numpy as np

def render_prediction_stage():
    """Renders the entire ML prediction stage, including sidebar and main content."""
    st.sidebar.header("🏗️ Project Parameters")
    st.sidebar.subheader("Basic Details")
    
    user_input_defaults = st.session_state.get('user_input', {})
    
    total_area = st.sidebar.number_input("Total Land Area (sqm)", min_value=100, max_value=200000, 
                                       value=user_input_defaults.get('total_area_of_land_sqm', 10000), 
                                       key="ml_land_area")
    far = st.sidebar.number_input("Floor Area Ratio (FAR)", min_value=0.1, max_value=10.0, 
                                value=user_input_defaults.get('far', 2.5), 
                                key="ml_far")
    st.sidebar.subheader("Location Details")
    lat = st.sidebar.number_input("Latitude", min_value=12.0, max_value=14.0, 
                                value=user_input_defaults.get('latitude', 12.9300773), 
                                format="%.7f", key="ml_lat")
    long = st.sidebar.number_input("Longitude", min_value=77.0, max_value=78.0, 
                                 value=user_input_defaults.get('longitude', 77.6950959), 
                                 format="%.7f", key="ml_long")
    st.sidebar.subheader("Project Type and Configuration")
    
    product_type_options = ["Apartment", "Villa"]
    default_product_type_index = product_type_options.index(user_input_defaults.get('product_type', 'Apartment'))
    product_type = st.sidebar.selectbox("Product Type", product_type_options, 
                                      index=default_product_type_index, 
                                      key="ml_product_type")

    customize_bhk = st.sidebar.checkbox("Customize BHK Types", 
                                      value=bool(user_input_defaults.get('selected_bhk_types')), 
                                      key="ml_customize_bhk")
    selected_bhk_types = None
    if customize_bhk:
        bhk_options = ['1RK', '1BHK', '1_5BHK', '2BHK', '2_5BHK', '3BHK', '3_5BHK', '4BHK', '4_5BHK', '5BHK']
        selected_bhk_types = st.sidebar.multiselect("Select BHK Types", options=bhk_options, 
                                                  default=user_input_defaults.get('selected_bhk_types', ['2BHK', '3BHK']), 
                                                  key="ml_bhk_select")
        if not selected_bhk_types: st.sidebar.warning("Please select at least one BHK type.")
    
    user_input = {'total_area_of_land_sqm': total_area, 'far': far, 'latitude': lat, 'longitude': long, 'product_type': product_type, 'selected_bhk_types': selected_bhk_types}

    if st.sidebar.button("🔮 Generate Predictions & Analysis", type="primary", use_container_width=True):
        if customize_bhk and not selected_bhk_types:
            st.sidebar.error("Please select at least one BHK type or disable customization.")
        else:
            with st.spinner("🧠 Running AI predictions and market analysis..."):
                st.session_state.user_input = user_input
                input_df = prepare_input_features(user_input, st.session_state.feature_names)
                st.session_state.predictions = predict_project(input_df, st.session_state.models, st.session_state.scalers, st.session_state.feature_names, st.session_state.bhk_mapping, user_input['selected_bhk_types'])
                st.session_state.nearby_analysis = get_nearby_properties_analysis(lat, long, 5, st.session_state.training_df, total_area)
    
    if st.session_state.predictions:
        render_prediction_results()
    else:
        st.header("Step 1: Generate Market Predictions")
        st.info("👈 Enter your project's parameters in the sidebar and click 'Generate Predictions' to begin.")

def render_prediction_results():
    """Renders the prediction results and the button to proceed to financial modeling."""
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
            with col3: st.metric("Sale Price (per sq.ft.)", f"₹ {similar_stats.get('min_sale_price_sqft', 0)/1000:.1f}K - ₹ {all_nearby_stats.get('max_sale_price_sqft', 0)/1000:.1f}K")
            col4, col5, col6 = st.columns(3)
            with col4: st.metric("Median Project Cost", f"₹ {similar_stats.get('median_project_cost_cr', 0):.2f} Cr")
            with col5: st.metric("Project Duration", f"{similar_stats.get('min_project_duration', 0):.1f} - {all_nearby_stats.get('max_project_duration', 0):.1f} Years")
            with col6: st.metric("Completion Date", f"{similar_stats.get('min_project_completion_date', 0)} - {all_nearby_stats.get('max_project_completion_date', 0)}")

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

def render_financial_model_ui():
    """Renders the UI for the Financial Modeling part of the app."""
    st.header("Step 2: Dynamic Financial Model")
    
    with st.sidebar:
        st.header("Financial Model Inputs")
        st.info("Inputs are pre-filled from ML predictions. You can edit them as needed.")
        if st.button("⬅️ Back to Predictions"):
            st.session_state.view = 'predictor'
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
            'debt_interest_rate': st.slider("Interest Rate on Debt (%)", 0.0, 20.0, 13.0) / 100.0,
        }
        base_inputs.update(financial_assumptions)
        
        st.subheader("3. Detailed Phasing Schedules")
        st.info("Set primary inputs above, then generate schedules.")

        def generate_schedules(current_inputs):
            st.session_state.reset_counter += 1
            construction_phasing_df, dynamic_end_date = dynamic_schedules.generate_construction_phasing(current_inputs['start_date'], current_inputs['land_area'], current_inputs['num_floors'])
            st.session_state.dynamic_end_date, st.session_state.construction_phasing_df = dynamic_end_date, construction_phasing_df
            
            master_timeline_range = pd.date_range(start=current_inputs['start_date'], end=dynamic_end_date + pd.DateOffset(years=2), freq='Q')
            master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline_range]
            
            sales_dist_df = dynamic_schedules.generate_sales_distribution(master_timeline_labels, dynamic_end_date)
            st.session_state.sales_dist_df = sales_dist_df
            
            st.session_state.payment_phasing_df = dynamic_schedules.generate_payment_phasing(master_timeline_labels, construction_phasing_df, sales_dist_df)

            last_sales_quarter_label = sales_dist_df[sales_dist_df['Distribution'] > 0]['Quarter'].iloc[-1]
            st.session_state.other_costs_phasing_df = dynamic_schedules.generate_other_costs_phasing(master_timeline_labels, dynamic_end_date, last_sales_quarter_label)
            
            st.session_state.schedules_generated = True
            st.success(f"Schedules generated. Project end date: {dynamic_end_date.strftime('%Y-%m-%d')}")

        if st.button("Generate / Reset Schedules ⚙️", type="primary") or not st.session_state.schedules_generated:
            generate_schedules(base_inputs)

        schedule_inputs = {}
        percent_format = "%.1f"
        
        with st.expander("🔨 Construction Costs & Phasing", expanded=True):
            schedule_inputs['construction_stage_costs'] = st.data_editor(pd.DataFrame([{"Stage": "Excavation and Foundation", "Cost Rate (per sq ft of Total SA)": 900}, {"Stage": "RCC", "Cost Rate (per sq ft of Total SA)": 1200}, {"Stage": "MEP", "Cost Rate (per sq ft of Total SA)": 800}, {"Stage": "Finishing", "Cost Rate (per sq ft of Total SA)": 1400}, {"Stage": "Infra and Amenities", "Cost Rate (per sq ft of Total SA)": 700}]), key="csc_editor", hide_index=True)
            
            construction_df = st.session_state.get('construction_phasing_df', pd.DataFrame()).copy()
            q_cols = [col for col in construction_df.columns if col.startswith('Q')]
            construction_df[q_cols] = construction_df[q_cols].replace(0, np.nan)
            construction_q_cols = {col: st.column_config.NumberColumn(format=f"{percent_format}%%") for col in construction_df.columns if col.startswith('Q')}
            schedule_inputs['construction_phasing'] = st.data_editor(construction_df, column_config=construction_q_cols, key=f"cp_editor_{st.session_state.reset_counter}", hide_index=True)

        with st.expander("💰 Sales & Payment Phasing"):
            st.subheader("Sales Phasing")
            sales_df = st.session_state.get('sales_dist_df', pd.DataFrame()).copy()
            sales_df['Distribution'] = sales_df['Distribution'].replace(0, np.nan)
            sales_config = {"Distribution": st.column_config.NumberColumn(format=f"{percent_format}%%")}
            schedule_inputs['sales_dist'] = st.data_editor(sales_df, column_config=sales_config, key=f"sd_editor_{st.session_state.reset_counter}", hide_index=True, num_rows="dynamic")
            
            st.markdown("---")
            st.subheader("Payment Phasing")
            payment_df = st.session_state.get('payment_phasing_df', pd.DataFrame()).copy()
            payment_df['Payment (%)'] = payment_df['Payment (%)'].replace(0, np.nan)
            payment_config = {"Payment (%)": st.column_config.NumberColumn(format=f"{percent_format}%%")}
            schedule_inputs['payment_phasing'] = st.data_editor(
                payment_df, 
                column_config=payment_config, 
                key=f"pp_editor_{st.session_state.reset_counter}", 
                hide_index=True,
                num_rows="fixed"
            )

        with st.expander("📋 Other Costs Phasing"):
            other_costs_df = st.session_state.get('other_costs_phasing_df', pd.DataFrame()).copy()
            q_cols_other = [col for col in other_costs_df.columns if col.startswith('Q')]
            other_costs_df[q_cols_other] = other_costs_df[q_cols_other].replace(0, np.nan)
            other_q_cols_config = {col: st.column_config.NumberColumn(format=f"{percent_format}%%") for col in other_costs_df.columns if col.startswith('Q')}
            schedule_inputs['other_costs_phasing'] = st.data_editor(other_costs_df, column_config=other_q_cols_config, key=f"ocp_editor_{st.session_state.reset_counter}", hide_index=True)
        
        if st.button("Calculate Project Financials 🚀", type="primary"):
            calc_schedule_inputs = {k: v.copy() for k, v in schedule_inputs.items()}
            for key in ['construction_phasing', 'other_costs_phasing', 'sales_dist', 'payment_phasing']:
                if key in calc_schedule_inputs and not calc_schedule_inputs[key].empty:
                    calc_schedule_inputs[key] = calc_schedule_inputs[key].fillna(0)
            
            final_inputs = {**base_inputs, **calc_schedule_inputs}
            final_inputs['end_date'] = st.session_state.get('dynamic_end_date', base_inputs.get('start_date'))
            
            validation_errors = validate_schedules(final_inputs)
            if validation_errors:
                for error in validation_errors: st.error(error)
            else:
                with st.spinner("Running financial model..."): 
                    st.session_state['results'] = calculate_financial_model(final_inputs)

    if st.session_state['results']:
        results, kpis, cashflow, financials, master_timeline, intermediate = st.session_state['results'], st.session_state['results']['kpis'], st.session_state['results']['cashflow'], st.session_state['results']['financials'], st.session_state['results']['master_timeline'], st.session_state['results']['intermediate']
        CR = 1e7
        colors = {'collections': '#63b179', 'expenses': '#e67c73', 'net_cash': '#4285f4', 'equity': '#f4b400', 'surplus': '#8f8f8f', "debt": "#FF6961"}
        table_styles = [{'selector': 'th, td', 'props': [('text-align', 'center')]}]
        cashflow_display, financials_display = (cashflow / CR), (financials / CR)
        cost_summary_display = intermediate['cost_summary'].copy(); cost_summary_display['Value'] /= CR
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dashboard", "💰 Cost Breakdowns", "🔎 Collection Details", "🧾 Cash Flow Details", "📊 Funding & Financials"])
        with tab1:
            st.header("Financial Summary")
            kpi_count = len(kpis)
            for i in range(kpi_count // 3 + 1):
                cols = st.columns(3)
                for j in range(3):
                    if i * 3 + j < kpi_count:
                        key, value = list(kpis.items())[i * 3 + j]
                        cols[j].metric(key, value)
            st.markdown("---"); st.header("Visualizations"); st.subheader("Quarterly Cash Flow Analysis")
            fig_cashflow = go.Figure()
            fig_cashflow.add_trace(go.Bar(x=master_timeline, y=cashflow_display['Collections'], name='Collections (Cr)', marker_color=colors['collections']))
            fig_cashflow.add_trace(go.Bar(x=master_timeline, y=-cashflow_display['Total Expenses'], name='Expenses (Cr)', marker_color=colors['expenses']))
            fig_cashflow.add_trace(go.Scatter(x=master_timeline, y=cashflow_display['Net Cash Flow'], name='Net Cash Flow (Cr)', mode='lines+markers', line=dict(color=colors['net_cash'])))
            fig_cashflow.update_layout(barmode='relative', height=400, title_text="Quarterly Inflows, Outflows, and Net Position", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Amount (in Cr)")
            st.plotly_chart(fig_cashflow, use_container_width=True)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Quarterly Funding View (in Cr)")
                fig_funding = go.Figure()
                fig_funding.add_trace(go.Bar(x=master_timeline, y=financials_display['Equity Inflow'], name='Equity Inflow', marker_color=colors['equity']))
                fig_funding.add_trace(go.Bar(x=master_timeline, y=financials_display['Debt Drawdown'], name='Debt Drawdown', marker_color=colors['debt']))
                fig_funding.add_trace(go.Bar(x=master_timeline, y=financials_display['Net Cash Flow'], name='Surplus/Deficit', marker_color=colors['surplus']))
                fig_funding.update_layout(barmode='relative', height=400, yaxis_title="Amount (in Cr)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_funding, use_container_width=True)
            with col2:
                st.subheader("Cash Balance Over Time (in Cr)")
                fig_balance = go.Figure()
                fig_balance.add_trace(go.Scatter(x=master_timeline, y=financials_display['Closing Cash'], name='Cash Balance', fill='tozeroy', line=dict(color=colors['net_cash'])))
                fig_balance.add_trace(go.Scatter(x=master_timeline, y=financials_display['Closing Debt'], name='Debt Balance', line=dict(color='red')))
                fig_balance.update_layout(height=400, yaxis_title="Amount (in Cr)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_balance, use_container_width=True)
        with tab2:
            st.header("Total Cost Breakdowns")
            st.dataframe(cost_summary_display.style.set_table_styles(table_styles).format("₹{:,.2f} Cr"))
        with tab3:
            st.header("Collection Details (in Cr)")
            collections_breakdown_display = intermediate['collections_breakdown'].copy()
            collections_breakdown_display_t = (collections_breakdown_display / CR).T
            st.dataframe(collections_breakdown_display_t.style.format(lambda val: "-" if abs(val) < 1e-9 else f"₹{val:,.2f}"))
        with tab4:
            st.header("Quarterly Cash Flow Details (in Cr)")
            construction_cost_cols = [f"Cost: {stage}" for stage in ["Excavation and Foundation", "RCC", "MEP", "Finishing", "Infra and Amenities"]]
            
            cashflow_for_display = cashflow.copy()
            cashflow_for_display['Cost: Total Construction'] = cashflow_for_display[construction_cost_cols].sum(axis=1)
            
            main_cashflow_cols = [col for col in cashflow_for_display.columns if col not in construction_cost_cols and col != 'Cost: Interest']
            cashflow_summary = cashflow_for_display[main_cashflow_cols]
            
            cost_cols = [col for col in cashflow_summary.columns if col.startswith('Cost: ')]
            final_order = ['Collections'] + cost_cols + ['Total Expenses', 'Net Cash Flow']
            cashflow_summary = cashflow_summary[final_order]

            cashflow_summary_t = (cashflow_summary / CR).T
            cashflow_summary_t['Total'] = cashflow_summary_t.sum(axis=1)
            cashflow_summary_t.insert(0, 'Total', cashflow_summary_t.pop('Total'))
            st.dataframe(cashflow_summary_t.style.format(lambda val: "-" if abs(val) < 1e-9 else f"₹{val:,.2f}"))

            st.subheader("Detailed Construction Costs (in Cr)")
            construction_details = cashflow[construction_cost_cols].copy()
            construction_details_t = (construction_details / CR).T
            construction_details_t['Total'] = construction_details_t.sum(axis=1)
            construction_details_t.insert(0, 'Total', construction_details_t.pop('Total'))
            construction_details_t.loc['Total'] = construction_details_t.sum(axis=0)
            st.dataframe(construction_details_t.style.format(lambda val: "-" if abs(val) < 1e-9 else f"₹{val:,.2f}"))
        with tab5:
            st.header("Funding & Financials (in Cr)")
            financials_to_display = financials_display[['Opening Cash', 'Net Cash Flow', 'Equity Inflow', 'Debt Drawdown', 'Debt Repayment', 'Closing Cash', 'Opening Debt', 'Closing Debt', 'Interest Expense']].copy()
            financials_t = (financials_to_display).T
            st.dataframe(financials_t.style.format(lambda val: "-" if abs(val) < 1e-9 else f"₹{val:,.2f}"))