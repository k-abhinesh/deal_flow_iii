import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import plotly.graph_objects as go
import dynamic_schedules

# ======================================================================================
# App Configuration
# ======================================================================================
st.set_page_config(layout="wide", page_title="Real Estate Financial Model")

# ======================================================================================
# Validation Function
# ======================================================================================
def validate_schedules(inputs):
    """
    Checks if the distribution percentages in user-edited schedules sum to 100%.
    Returns a list of error messages. An empty list means validation passed.
    """
    errors = []
    tolerance = 0.001 # Use a tolerance for floating point comparisons

    # 1. Validate Construction Phasing
    df_construct = inputs['construction_phasing']
    q_cols_construct = [col for col in df_construct.columns if isinstance(col, str) and col.startswith('Q')]
    for index, row in df_construct.iterrows():
        total = row[q_cols_construct].sum()
        if abs(total - 1.0) > tolerance:
            errors.append(f"Construction Phasing Error: '{row['Stage']}' distribution sums to {total:.2%}, not 100%.")

    # 2. Validate Other Costs Phasing
    df_other = inputs['other_costs_phasing']
    q_cols_other = [col for col in df_other.columns if isinstance(col, str) and col.startswith('Q')]
    for index, row in df_other.iterrows():
        total = row[q_cols_other].sum()
        if abs(total - 1.0) > tolerance:
            # Don't flag items that are supposed to be zero
            if inputs['plan_sanction_cost'] > 0 and row['Cost Item'] == 'Plan Sanction' and total > 0:
                 errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' distribution sums to {total:.2%}, not 100%.")
            if inputs['sales_lounge_cost'] > 0 and row['Cost Item'] == 'Sales Lounge' and total > 0:
                 errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' distribution sums to {total:.2%}, not 100%.")
            if inputs['consultant_cost_psf'] > 0 and row['Cost Item'] == 'Consultant Cost' and total > 0:
                 errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' distribution sums to {total:.2%}, not 100%.")
            if inputs['marketing_cost_pct_revenue'] > 0 and row['Cost Item'] == 'Marketing Cost' and total > 0:
                 errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' distribution sums to {total:.2%}, not 100%.")
            if inputs['misc_approval_cost_psf'] > 0 and row['Cost Item'] == 'Misc. Approval Cost' and total > 0:
                 errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' distribution sums to {total:.2%}, not 100%.")


    # 3. Validate Sales Distribution
    total_sales = inputs['sales_dist']['Distribution'].sum()
    if abs(total_sales - 1.0) > tolerance:
        errors.append(f"Sales Distribution Error: Total distribution sums to {total_sales:.2%}, not 100%.")

    # 4. Validate Collection Distribution
    total_collections = inputs['collection_dist']['Distribution'].sum()
    if abs(total_collections - 1.0) > tolerance:
        errors.append(f"Collections Distribution Error: Total distribution sums to {total_collections:.2%}, not 100%.")
        
    return errors

# ======================================================================================
# Core Calculation Engine
# ======================================================================================
def calculate_financial_model(inputs):
    """Main function to run all financial calculations based on user inputs."""
    intermediate_data = {}

    # --- 1. Master Timeline Calculation ---
    all_q_strs = set()
    for key in ['construction_phasing', 'other_costs_phasing', 'sales_dist', 'collection_dist', 'admin_cost']:
        df = inputs.get(key)
        if df is None or df.empty:
            continue
        quarter_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Q')]
        if not quarter_cols and 'Quarter' in df.columns:
             quarter_cols = df['Quarter'].unique()
        all_q_strs.update(quarter_cols)
    
    all_timestamps = []
    for q in all_q_strs:
        if q and isinstance(q, str) and ' ' in q:
            parts = q.split(' ')
            formatted_q_str = f"{parts[1]}{parts[0]}" # "2027Q4"
            all_timestamps.append(pd.Period(formatted_q_str, freq='Q').to_timestamp())
    
    if not all_timestamps:
        start_date = pd.to_datetime(inputs['start_date'])
        end_date = st.session_state.get('dynamic_end_date', start_date + pd.DateOffset(years=3))
    else:
        start_date = min(all_timestamps)
        end_date = max(all_timestamps)

    master_timeline = pd.date_range(start=start_date, end=end_date, freq='Q')
    master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline]

    # --- 2. Area, Revenue, and Cost Totals ---
    total_bua = inputs['land_area'] * inputs['far']
    total_carpet_area = total_bua * inputs['efficiency']
    total_saleable_area = total_carpet_area * 1.1
    townhouse_sa = total_saleable_area - inputs['clubhouse_area']
    
    area_calcs = pd.DataFrame([
        {"Metric": "Land Area (sq ft)", "Value": inputs['land_area']},
        {"Metric": "FAR", "Value": inputs['far']},
        {"Metric": "Total BUA (sq ft)", "Value": total_bua},
        {"Metric": "Efficiency %", "Value": f"{inputs['efficiency']:.2%}"},
        {"Metric": "Total Carpet Area (sq ft)", "Value": total_carpet_area},
        {"Metric": "Saleable Area Factor", "Value": "110%"},
        {"Metric": "Total Saleable Area (sq ft)", "Value": total_saleable_area},
        {"Metric": "Clubhouse & Common Area (sq ft)", "Value": inputs['clubhouse_area']},
        {"Metric": "Townhouse Saleable Area (sq ft)", "Value": townhouse_sa},
    ]).set_index("Metric")
    intermediate_data['area_calcs'] = area_calcs

    sales_dist = pd.Series(inputs['sales_dist']['Distribution'].fillna(0).values, index=inputs['sales_dist']['Quarter'].values)
    first_sale_quarter = sales_dist[sales_dist > 0].index[0] if not sales_dist[sales_dist > 0].empty else None

    project_timeline = pd.date_range(start=inputs['start_date'], end=inputs['end_date'], freq='Q')
    project_timeline_labels = [f"Q{q.quarter} {q.year}" for q in project_timeline]
    
    sales_prices = []
    escalation_count = 0
    sales_started = False
    for quarter_label in master_timeline_labels:
        if not sales_started and quarter_label == first_sale_quarter:
            sales_started = True
        
        price = inputs['unit_price']
        if sales_started and quarter_label in project_timeline_labels:
            price *= ((1 + inputs['escalation']) ** escalation_count)
            escalation_count += 1
        sales_prices.append(price)

    sales_prices_s = pd.Series(sales_prices, index=master_timeline_labels)
    
    revenue_details = pd.DataFrame(index=master_timeline_labels)
    revenue_details['Quarterly Sales Distribution'] = sales_dist.reindex(master_timeline_labels, fill_value=0)
    revenue_details['Escalated Price per Sq Ft'] = sales_prices_s
    revenue_details['Quarterly Revenue'] = townhouse_sa * revenue_details['Quarterly Sales Distribution'] * revenue_details['Escalated Price per Sq Ft']
    total_revenue = revenue_details['Quarterly Revenue'].sum()
    intermediate_data['revenue_details'] = revenue_details

    construction_stage_costs = inputs['construction_stage_costs'].set_index('Stage')
    construction_stage_costs['Total Cost'] = construction_stage_costs['Cost Rate (per sq ft of Total SA)'].fillna(0) * total_saleable_area
    total_construction_cost = construction_stage_costs['Total Cost'].sum()

    total_consultant_cost = total_saleable_area * inputs['consultant_cost_psf']
    total_marketing_cost = total_revenue * inputs['marketing_cost_pct_revenue']
    total_misc_approval_cost = total_saleable_area * inputs['misc_approval_cost_psf']
    
    cost_summary = pd.DataFrame([
        {"Cost Category": "Total Construction Cost", "Value": total_construction_cost},
        {"Cost Category": "Total Consultant Cost", "Value": total_consultant_cost},
        {"Cost Category": "Total Marketing Cost", "Value": total_marketing_cost},
        {"Cost Category": "Total Misc. Approval Cost", "Value": total_misc_approval_cost},
        {"Cost Category": "Plan Sanction (Lump Sum)", "Value": inputs['plan_sanction_cost']},
        {"Cost Category": "Sales Lounge (Lump Sum)", "Value": inputs['sales_lounge_cost']},
    ]).set_index("Cost Category")
    intermediate_data['cost_summary'] = cost_summary

    # --- 3. Main Cashflow DataFrame ---
    cashflow = pd.DataFrame(index=master_timeline_labels)

    collection_dist_series = pd.Series(inputs['collection_dist']['Distribution'].fillna(0).values, index=inputs['collection_dist']['Quarter'].values)
    cashflow['Collections'] = collection_dist_series.reindex(master_timeline_labels, fill_value=0) * total_revenue
    
    total_brokerage_cost = cashflow['Collections'].sum() * inputs['brokerage_fee']
    cashflow['Cost: Brokerage'] = cashflow['Collections'] * inputs['brokerage_fee']
    cost_summary.loc["Total Brokerage Cost"] = total_brokerage_cost


    construction_phasing = inputs['construction_phasing'].set_index('Stage')
    for stage_name, stage_data in construction_stage_costs.iterrows():
        total_stage_cost = stage_data['Total Cost']
        phasing_s = construction_phasing.loc[stage_name]
        cashflow[f'Cost: {stage_name}'] = phasing_s.reindex(master_timeline_labels, fill_value=0).astype(float) * total_stage_cost

    other_costs_phasing = inputs['other_costs_phasing'].set_index('Cost Item')
    cost_items_map = {
        'Consultant Cost': total_consultant_cost, 'Marketing Cost': total_marketing_cost,
        'Misc. Approval Cost': total_misc_approval_cost, 'Plan Sanction': inputs['plan_sanction_cost'],
        'Sales Lounge': inputs['sales_lounge_cost']
    }
    for item, total_cost in cost_items_map.items():
        if item in other_costs_phasing.index:
            phasing_s = other_costs_phasing.loc[item]
            cashflow[f'Cost: {item}'] = phasing_s.reindex(master_timeline_labels, fill_value=0).astype(float) * total_cost
    
    admin_cost_schedule = pd.Series(inputs['admin_cost']['Cost per Quarter'].fillna(0).values, index=inputs['admin_cost']['Quarter'].values)
    cashflow['Cost: Admin'] = admin_cost_schedule.reindex(master_timeline_labels, fill_value=0)

    cost_columns = [col for col in cashflow.columns if 'Cost:' in col]
    cashflow[cost_columns] = cashflow[cost_columns].fillna(0)
    cashflow['Total Expenses'] = cashflow[cost_columns].sum(axis=1)
    cashflow['EBITDA'] = cashflow['Collections'] - cashflow['Total Expenses']

    # --- 4. Funding Requirements ---
    financials = pd.DataFrame(index=master_timeline_labels)
    financials['Surplus/(Deficit)'] = cashflow['EBITDA']
    financials['Opening Cash'] = 0.0
    
    opening_cash = 0.0
    equity_required = 0
    closing_cash_list = []
    equity_inflow_list = []
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

    # --- 5. Compile Final KPIs ---
    total_project_cost = cashflow['Total Expenses'].sum()
    pat = (total_revenue - total_project_cost) * (1 - 0.25)
    
    kpis = {
        "Projected Revenue": f"₹{total_revenue/1e7:.2f} Cr",
        "Total Construction Cost": f"₹{total_construction_cost/1e7:.2f} Cr",
        "Total Project Cost": f"₹{total_project_cost/1e7:.2f} Cr",
        "Profit After Tax": f"₹{pat/1e7:.2f} Cr",
        "Equity Required": f"₹{equity_required/1e7:.2f} Cr",
    }
    
    return {"kpis": kpis, "cashflow": cashflow, "financials": financials, "intermediate": intermediate_data, "master_timeline": master_timeline}

# ======================================================================================
# Streamlit User Interface
# ======================================================================================
# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'schedules_generated' not in st.session_state:
    st.session_state['schedules_generated'] = False
if 'reset_counter' not in st.session_state:
    st.session_state['reset_counter'] = 0


st.title("Dynamic Real Estate Financial Model 🏗️")
st.write(f"**As of:** {date.today().strftime('%B %d, %Y')} | **Location:** Bengaluru, Karnataka")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Project Inputs")

    st.subheader("1. Project Definition")
    base_inputs = {
        'land_area': st.number_input("Land Area (sq ft)", value=217800, min_value=1000),
        'far': st.number_input("FAR (Floor Area Ratio)", value=1.75, min_value=0.1),
        'num_floors': st.number_input("Number of Floors", value=14, min_value=1),
        'efficiency': st.slider("Efficiency % (BUA to Carpet)", 0.0, 1.0, 0.45),
        'clubhouse_area': st.number_input("Clubhouse & Common Area (sq ft)", value=80000),
        'unit_price': st.number_input("Base Price (per sq ft of SA)", value=17500),
        'start_date': st.date_input("Project Start Date", value=pd.to_datetime("2025-06-30")),
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
    admin_dates = {
        'admin_start_date': st.date_input("Admin Cost Start Date", value=pd.to_datetime("2025-06-30")),
        'admin_end_date': st.date_input("Admin Cost End Date", value=pd.to_datetime("2032-09-30"))
    }
    base_inputs.update(admin_dates)

    st.subheader("3. Detailed Phasing Schedules")
    st.info("Set primary inputs above, then generate schedules.")

    # --- DYNAMIC SCHEDULE GENERATION ---
    def generate_schedules(current_inputs):
        # FIX: Increment a counter to force data_editors to re-render
        st.session_state.reset_counter += 1
        
        construction_phasing_df, dynamic_end_date = dynamic_schedules.generate_construction_phasing(
            current_inputs['start_date'], current_inputs['land_area'], current_inputs['num_floors']
        )
        st.session_state.dynamic_end_date = dynamic_end_date
        st.session_state.construction_phasing_df = construction_phasing_df
        
        temp_total_bua = current_inputs['land_area'] * current_inputs['far']
        temp_total_carpet_area = temp_total_bua * current_inputs['efficiency']
        temp_total_saleable_area = temp_total_carpet_area * 1.1

        post_project_q = pd.Period(dynamic_end_date, freq='Q') + 8 
        extended_end_date = post_project_q.end_time
        
        admin_end_date_ts = pd.to_datetime(current_inputs['admin_end_date'])
        master_timeline_range = pd.date_range(start=current_inputs['start_date'], end=max(extended_end_date, admin_end_date_ts), freq='Q')
        master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline_range]

        st.session_state.other_costs_phasing_df = dynamic_schedules.generate_other_costs_phasing(master_timeline_labels, dynamic_end_date)
        st.session_state.sales_dist_df = dynamic_schedules.generate_sales_distribution(master_timeline_labels, dynamic_end_date)
        st.session_state.collection_dist_df = dynamic_schedules.generate_collection_distribution(st.session_state.sales_dist_df, master_timeline_labels)
        
        admin_timeline_range = pd.date_range(start=current_inputs['admin_start_date'], end=current_inputs['admin_end_date'], freq='Q')
        admin_timeline_labels = [f"Q{q.quarter} {q.year}" for q in admin_timeline_range]
        st.session_state.admin_cost_df = dynamic_schedules.generate_admin_cost_schedule(temp_total_saleable_area, admin_timeline_labels)
        
        st.session_state.schedules_generated = True
        st.success(f"Schedules generated. Project end date: {dynamic_end_date.strftime('%Y-%m-%d')}")


    if st.button("Generate / Reset Schedules ⚙️", type="primary") or not st.session_state.schedules_generated:
        generate_schedules(base_inputs)

    # --- Display Editable Schedules ---
    schedule_inputs = {}
    with st.expander("🔨 Construction Costs & Phasing", expanded=True):
        schedule_inputs['construction_stage_costs'] = st.data_editor(
            pd.DataFrame([
                {"Stage": "Excavation and Foundation", "Cost Rate (per sq ft of Total SA)": 900},
                {"Stage": "RCC", "Cost Rate (per sq ft of Total SA)": 1200},
                {"Stage": "MEP", "Cost Rate (per sq ft of Total SA)": 800},
                {"Stage": "Finishing", "Cost Rate (per sq ft of Total SA)": 1400},
                {"Stage": "Infra and Amenities", "Cost Rate (per sq ft of Total SA)": 700}
            ]), 
            key="csc_editor", hide_index=True
        )
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
            for error in validation_errors:
                st.error(error)
        else:
            with st.spinner("Running financial model..."):
                st.session_state['results'] = calculate_financial_model(final_inputs)

# --- Main Panel for Outputs ---
if st.session_state['results']:
    results = st.session_state['results']
    kpis, cashflow, financials, master_timeline = results['kpis'], results['cashflow'], results['financials'], results['master_timeline']
    intermediate = results['intermediate']
    CR = 1e7

    colors = {'collections': '#63b179', 'expenses': '#e67c73', 'net_cash': '#4285f4', 'equity': '#f4b400', 'surplus': '#8f8f8f'}
    table_styles = [{'selector': 'th, td', 'props': [('text-align', 'center')]}]

    cashflow_display = (cashflow / CR)
    financials_display = (financials / CR)
    cost_summary_display = intermediate['cost_summary'].copy()
    cost_summary_display['Value'] /= CR
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard", 
        "🏗️ Area & Revenue", 
        "💰 Cost Breakdowns", 
        "🧾 Cash Flow Details", 
        "📊 Funding & Financials"
    ])

    with tab1:
        st.header("Financial Summary")
        cols = st.columns(len(kpis))
        for i, (key, value) in enumerate(kpis.items()):
            cols[i].metric(key, value)
        
        st.markdown("---")
        st.header("Visualizations")
        
        st.subheader("Quarterly Cash Flow Analysis")
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
        st.subheader("Summary of Total Project Costs (Pre-Phasing)")
        st.dataframe(cost_summary_display.style.set_table_styles(table_styles).format("₹{:,.2f} Cr"))
        st.info("These are the total calculated costs before they are distributed across the project timeline in the cash flow.")

    with tab4:
        st.header("Quarterly Cash Flow Details (in Cr)")
        st.dataframe(cashflow_display.style.set_table_styles(table_styles).format("₹{:,.2f}"))

    with tab5:
        st.header("Funding & Financials (in Cr)")
        st.dataframe(financials_display.style.set_table_styles(table_styles).format("₹{:,.2f}"))

else:
    st.info("Configure your project in the sidebar and click 'Calculate Project Financials' to see the results.")
