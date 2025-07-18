import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import plotly.graph_objects as go
import plotly.express as px

# ======================================================================================
# App Configuration
# ======================================================================================
st.set_page_config(layout="wide", page_title="Real Estate Financial Model")

# ======================================================================================
# Core Calculation Engine
# ======================================================================================
def calculate_financial_model(inputs):
    """Main function to run all financial calculations based on user inputs."""
    # --- Intermediate Data Storage ---
    intermediate_data = {}

    # 1. Generate Timelines
    project_timeline = pd.date_range(start=inputs['start_date'], end=inputs['end_date'], freq='Q')
    admin_timeline = pd.date_range(start=inputs['admin_start_date'], end=inputs['admin_end_date'], freq='Q')
    
    combined_start = min(inputs['start_date'], inputs['admin_start_date'])
    combined_end = max(inputs['end_date'], inputs['admin_end_date'])
    master_timeline = pd.date_range(start=combined_start, end=combined_end, freq='Q')
    master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline]

    # 2. Calculate Area, Revenue, and Cost Totals
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
    
    first_sale_quarter_series = sales_dist[sales_dist > 0]
    first_sale_quarter = first_sale_quarter_series.index[0] if not first_sale_quarter_series.empty else None

    project_timeline_labels_list = [f"Q{q.quarter} {q.year}" for q in project_timeline]
    
    sales_prices = []
    escalation_count = 0
    sales_started = False

    for quarter_label in project_timeline_labels_list:
        if not sales_started and quarter_label == first_sale_quarter:
            sales_started = True

        if sales_started:
            price = inputs['unit_price'] * ((1 + inputs['escalation']) ** escalation_count)
            escalation_count += 1
        else:
            price = inputs['unit_price']
        
        sales_prices.append(price)

    sales_prices_s = pd.Series(sales_prices, index=project_timeline_labels_list)
    
    revenue_details = pd.DataFrame({
        'Quarterly Sales Distribution': sales_dist,
        'Escalated Price per Sq Ft': sales_prices_s
    }).reindex(master_timeline_labels, fill_value=0)
    revenue_details['Quarterly Revenue'] = townhouse_sa * revenue_details['Quarterly Sales Distribution'] * revenue_details['Escalated Price per Sq Ft']
    total_revenue = revenue_details['Quarterly Revenue'].sum()
    intermediate_data['revenue_details'] = revenue_details

    construction_stage_costs = inputs['construction_stage_costs'].set_index('Stage')
    construction_stage_costs['Total Cost'] = construction_stage_costs['Cost Rate (per sq ft of Total SA)'].fillna(0) * total_saleable_area
    total_construction_cost = construction_stage_costs['Total Cost'].sum()

    total_consultant_cost = total_saleable_area * inputs['consultant_cost_psf']
    total_marketing_cost = (inputs['unit_price'] * 1.18 * 0.03) * total_saleable_area
    total_misc_approval_cost = total_saleable_area * inputs['misc_approval_cost_psf']
    
    # 3. Create the Main Cashflow DataFrame
    cashflow = pd.DataFrame(index=master_timeline_labels)

    cumulative_revenue_s = revenue_details['Quarterly Revenue'].cumsum().reindex(master_timeline_labels).ffill().fillna(0)
    collection_dist_series = pd.Series(
        inputs['collection_dist']['Distribution'].fillna(0).values,
        index=inputs['collection_dist']['Quarter'].values
    )
    cumulative_collection_pct = collection_dist_series.cumsum().reindex(master_timeline_labels).ffill().fillna(0)
    total_collections_due = (cumulative_revenue_s * cumulative_collection_pct)
    preceding_collections_due = total_collections_due.shift(1).fillna(0)
    quarterly_collections = total_collections_due - preceding_collections_due
    
    collection_details = pd.DataFrame({
        "Cumulative Revenue": cumulative_revenue_s,
        "Cumulative Collection %": cumulative_collection_pct,
        "Total Collections Due by Quarter End": total_collections_due,
        "Quarterly Collections": quarterly_collections
    })
    intermediate_data['collection_details'] = collection_details
    
    cashflow['Collections'] = quarterly_collections

    total_brokerage_cost = cashflow['Collections'].sum() * inputs['brokerage_fee']
    cashflow['Cost: Brokerage'] = cashflow['Collections'] * inputs['brokerage_fee']

    cost_summary = pd.DataFrame([
        {"Cost Category": "Total Construction Cost", "Value": total_construction_cost},
        {"Cost Category": "Total Consultant Cost", "Value": total_consultant_cost},
        {"Cost Category": "Total Marketing Cost", "Value": total_marketing_cost},
        {"Cost Category": "Total Misc. Approval Cost", "Value": total_misc_approval_cost},
        {"Cost Category": "Total Brokerage Cost", "Value": total_brokerage_cost},
        {"Cost Category": "Plan Sanction (Lump Sum)", "Value": inputs['plan_sanction_cost']},
        {"Cost Category": "Sales Lounge (Lump Sum)", "Value": inputs['sales_lounge_cost']},
    ]).set_index("Cost Category")
    intermediate_data['cost_summary'] = cost_summary
    
    construction_phasing = inputs['construction_phasing'].set_index('Stage')
    for stage_name, stage_data in construction_stage_costs.iterrows():
        specific_stage_total_cost = stage_data['Total Cost']
        stage_phasing_dist = construction_phasing.loc[stage_name].fillna(0)
        phased_cost = specific_stage_total_cost * stage_phasing_dist
        cashflow[f'Cost: {stage_name}'] = phased_cost

    other_costs_phasing = inputs['other_costs_phasing'].set_index('Cost Item')
    cost_items_map = {
        'Consultant Cost': total_consultant_cost,
        'Marketing Cost': total_marketing_cost,
        'Misc. Approval Cost': total_misc_approval_cost,
        'Plan Sanction': inputs['plan_sanction_cost'],
        'Sales Lounge': inputs['sales_lounge_cost']
    }
    for item_name, specific_item_total_cost in cost_items_map.items():
        if item_name in other_costs_phasing.index:
            item_phasing_dist = other_costs_phasing.loc[item_name].fillna(0)
            phased_cost = specific_item_total_cost * item_phasing_dist
            cashflow[f'Cost: {item_name}'] = phased_cost
    
    admin_cost_schedule = pd.Series(inputs['admin_cost']['Cost per Quarter'].fillna(0).values, index=inputs['admin_cost']['Quarter'].values)
    cashflow['Cost: Admin'] = admin_cost_schedule.reindex(master_timeline_labels, fill_value=0)

    cost_columns = [col for col in cashflow.columns if 'Cost:' in col]
    cashflow[cost_columns] = cashflow[cost_columns].fillna(0)

    cashflow['Total Expenses'] = cashflow[cost_columns].sum(axis=1)
    cashflow['EBITDA'] = cashflow['Collections'] - cashflow['Total Expenses']

    # 4. Calculate Funding Requirements
    financials = pd.DataFrame(index=master_timeline_labels)
    financials['Surplus/(Deficit)'] = cashflow['EBITDA']
    financials['Opening Cash'] = 0.0
    financials['Equity Inflow'] = 0.0
    financials['Closing Cash'] = 0.0
    
    equity_required = 0
    for i in range(len(financials)):
        opening_cash = financials.iloc[i-1]['Closing Cash'] if i > 0 else 0
        financials.iloc[i, financials.columns.get_loc('Opening Cash')] = opening_cash
        pre_funding_balance = opening_cash + financials.iloc[i]['Surplus/(Deficit)']
        if pre_funding_balance < 0:
            inflow_needed = -pre_funding_balance
            financials.iloc[i, financials.columns.get_loc('Equity Inflow')] = inflow_needed
            equity_required += inflow_needed
        financials.iloc[i, financials.columns.get_loc('Closing Cash')] = pre_funding_balance + financials.iloc[i]['Equity Inflow']

    # 5. Compile Final KPIs
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
if 'results' not in st.session_state:
    st.session_state['results'] = None

st.title("Dynamic Real Estate Financial Model 🏗️")
st.write(f"**As of:** {date.today().strftime('%B %d, %Y')} | **Location:** Bengaluru, Karnataka")

with st.sidebar:
    st.header("Project Inputs")

    st.subheader("1. Project Definition")
    inputs = {
        'land_area': st.number_input("Land Area (sq ft)", value=217800),
        'far': st.number_input("FAR (Floor Area Ratio)", value=1.75),
        'efficiency': st.slider("Efficiency % (BUA to Carpet)", 0.0, 1.0, 0.45),
        'clubhouse_area': st.number_input("Clubhouse & Common Area (sq ft)", value=80000),
        'unit_price': st.number_input("Base Price (per sq ft of Saleable Area)", value=17500),
        'start_date': st.date_input("Project Start Date", value=pd.to_datetime("2025-06-30")),
        'end_date': st.date_input("Project End Date", value=pd.to_datetime("2029-12-31"))
    }
    
    st.subheader("2. Financial & Cost Assumptions")
    inputs.update({
        'escalation': st.slider("Price Escalation (% per Quarter)", 0.0, 5.0, 1.5) / 100,
        'brokerage_fee': st.slider("Brokerage Fee (% of Revenue)", 0.0, 5.0, 2.95) / 100,
        'plan_sanction_cost': st.number_input("Plan Sanction Cost (Lump Sum)", value=40000000),
        'sales_lounge_cost': st.number_input("Sales Lounge Cost (Lump Sum)", value=50000000),
        'consultant_cost_psf': st.number_input("Consultant Cost (per sq ft of Total SA)", value=300),
        'misc_approval_cost_psf': st.number_input("Misc. Approval Cost (per sq ft of Total SA)", value=50),
    })
    
    st.subheader("3. Detailed Phasing Schedules")
    st.info("Default values are populated from the source Excel file.")

    project_timeline_labels = [f"Q{q.quarter} {q.year}" for q in pd.date_range(start=inputs['start_date'], end=inputs['end_date'], freq='Q')]
    
    with st.expander("🔨 Construction Costs & Phasing", expanded=True):
        csc_df = pd.DataFrame([
            {"Stage": "Excavation and Foundation", "Cost Rate (per sq ft of Total SA)": 900},
            {"Stage": "RCC", "Cost Rate (per sq ft of Total SA)": 1200},
            {"Stage": "MEP", "Cost Rate (per sq ft of Total SA)": 800},
            {"Stage": "Finishing", "Cost Rate (per sq ft of Total SA)": 1400},
            {"Stage": "Infra and Amenities", "Cost Rate (per sq ft of Total SA)": 700}
        ])
        inputs['construction_stage_costs'] = st.data_editor(csc_df, key="csc_editor", hide_index=True)
        cp_data = {
            'Stage': csc_df['Stage'].tolist(),
            'Q2 2025': [0.0, 0.0, 0.0, 0.0, 0.0], 'Q3 2025': [0.0, 0.0, 0.0, 0.0, 0.0],
            'Q4 2025': [0.1, 0.0, 0.0, 0.0, 0.0], 'Q1 2026': [0.2, 0.0, 0.0, 0.0, 0.0],
            'Q2 2026': [0.2, 0.0, 0.0, 0.0, 0.0], 'Q3 2026': [0.2, 0.0, 0.0, 0.0, 0.0],
            'Q4 2026': [0.3, 0.1, 0.0, 0.0, 0.0], 'Q1 2027': [0.0, 0.15, 0.0, 0.0, 0.0],
            'Q2 2027': [0.0, 0.2, 0.0, 0.0, 0.0], 'Q3 2027': [0.0, 0.2, 0.0, 0.0, 0.0],
            'Q4 2027': [0.0, 0.25, 0.0, 0.0, 0.0], 'Q1 2028': [0.0, 0.1, 0.2, 0.0, 0.0],
            'Q2 2028': [0.0, 0.0, 0.25, 0.0, 0.0], 'Q3 2028': [0.0, 0.0, 0.25, 0.2, 0.0],
            'Q4 2028': [0.0, 0.0, 0.3, 0.25, 0.0], 'Q1 2029': [0.0, 0.0, 0.0, 0.25, 0.0],
            'Q2 2029': [0.0, 0.0, 0.0, 0.3, 0.25], 'Q3 2029': [0.0, 0.0, 0.0, 0.0, 0.25],
            'Q4 2029': [0.0, 0.0, 0.0, 0.0, 0.5]
        }
        inputs['construction_phasing'] = st.data_editor(pd.DataFrame(cp_data), key="cp_editor", hide_index=True)

    with st.expander("💰 Sales & Collections Phasing"):
        sales_dist_data = {'Quarter': project_timeline_labels, 'Distribution': [0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]}
        inputs['sales_dist'] = st.data_editor(pd.DataFrame(sales_dist_data), key="sd_editor", hide_index=True)
        collection_dist_data = {'Quarter': project_timeline_labels, 'Distribution': [0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]}
        inputs['collection_dist'] = st.data_editor(pd.DataFrame(collection_dist_data), key="cd_editor", hide_index=True)

    with st.expander("📋 Other Costs Phasing"):
        ocp_data = {
            'Cost Item': ['Consultant Cost', 'Marketing Cost', 'Misc. Approval Cost', 'Plan Sanction', 'Sales Lounge'],
            'Q2 2025': [0.1, 0.0, 0.0, 0.0, 0.0], 
            'Q3 2025': [0.15, 0.05, 0.0, 1.0, 0.5],
            'Q4 2025': [0.15, 0.1, 0.0, 0.0, 0.5], 
            'Q1 2026': [0.05, 0.1, 0.0, 0.0, 0.0],
            'Q2 2026': [0.025, 0.1, 0.0, 0.0, 0.0], 
            'Q3 2026': [0.025, 0.05, 0.0, 0.0, 0.0],
            'Q4 2026': [0.05, 0.075, 0.0, 0.0, 0.0], 
            'Q1 2027': [0.025, 0.075, 0.0, 0.0, 0.0],
            'Q2 2027': [0.025, 0.075, 0.0, 0.0, 0.0], 
            'Q3 2027': [0.025, 0.075, 0.0, 0.0, 0.0],
            'Q4 2027': [0.025, 0.075, 0.0, 0.0, 0.0], 
            'Q1 2028': [0.025, 0.075, 0.0, 0.0, 0.0],
            'Q2 2028': [0.025, 0.075, 0.0, 0.0, 0.0], 
            'Q3 2028': [0.05, 0.075, 0.0, 0.0, 0.0],
            'Q4 2028': [0.05, 0.0, 0.0, 0.0, 0.0], 
            'Q1 2029': [0.05, 0.0, 0.0, 0.0, 0.0],
            'Q2 2029': [0.05, 0.0, 0.0, 0.0, 0.0], 
            'Q3 2029': [0.05, 0.0, 0.0, 0.0, 0.0],
            'Q4 2029': [0.05, 0.0, 1.0, 0.0, 0.0]
        }
        inputs['other_costs_phasing'] = st.data_editor(pd.DataFrame(ocp_data), key="ocp_editor", hide_index=True)

    with st.expander("👥 Admin Cost Schedule"):
        inputs['admin_start_date'] = st.date_input("Admin Cost Start Date", value=pd.to_datetime("2025-06-30"))
        inputs['admin_end_date'] = st.date_input("Admin Cost End Date", value=pd.to_datetime("2032-09-30"))
        admin_timeline_labels = [f"Q{q.quarter} {q.year}" for q in pd.date_range(start=inputs['admin_start_date'], end=inputs['admin_end_date'], freq='Q')]
        admin_cost_values = [0.0, 0.0, 1300000, 1300000, 1300000, 1500000, 2000000, 2000000, 2000000, 2000000, 2400000, 2400000, 2400000, 2400000, 2400000, 2400000, 2400000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000, 2825000]
        if len(admin_timeline_labels) > len(admin_cost_values):
            admin_cost_values.extend([0.0] * (len(admin_timeline_labels) - len(admin_cost_values)))
        admin_cost_data = {'Quarter': admin_timeline_labels, 'Cost per Quarter': admin_cost_values[:len(admin_timeline_labels)]}
        inputs['admin_cost'] = st.data_editor(pd.DataFrame(admin_cost_data), key="ac_editor", hide_index=True)

    if st.button("Calculate Project Financials 🚀", type="primary"):
        with st.spinner("Running financial model..."):
            st.session_state['results'] = calculate_financial_model(inputs)

# Main Panel for displaying all outputs
if st.session_state['results']:
    results = st.session_state['results']
    kpis = results['kpis']
    cashflow = results['cashflow']
    financials = results['financials']
    intermediate = results['intermediate']
    master_timeline = results['master_timeline']
    
    CR = 1e7

    # --- Chart Colors ---
    colors = {
        'collections': '#63b179', # Soft Green
        'expenses': '#e67c73',    # Soft Red
        'net_cash': '#4285f4',    # Google Blue
        'equity': '#f4b400',      # Google Yellow
        'surplus': '#8f8f8f'      # Grey
    }
    
    # --- Define a reusable style for center alignment ---
    table_styles = [{'selector': 'th, td', 'props': [('text-align', 'center')]}]

    # --- Create Display-Ready DataFrames (in Cr) ---
    cashflow_display = (cashflow / CR)
    financials_display = (financials / CR)

    financials_for_plotting = financials_display.copy()
    financials_for_plotting.index = master_timeline

    revenue_details_display = intermediate['revenue_details'].copy()
    revenue_details_display['Quarterly Revenue'] /= CR

    collection_details_display = intermediate['collection_details'].copy()
    collection_details_display["Cumulative Revenue"] /= CR
    collection_details_display["Total Collections Due by Quarter End"] /= CR
    collection_details_display["Quarterly Collections"] /= CR
    
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
        fig_cashflow.add_trace(go.Bar(
            x=master_timeline, y=cashflow_display['Collections'], name='Collections (Cr)', 
            marker_color=colors['collections'],
            hovertemplate='<b>%{x|%Y-Q%q}</b><br>Collections: %{y:.2f} Cr<extra></extra>'
        ))
        fig_cashflow.add_trace(go.Bar(
            x=master_timeline, y=-cashflow_display['Total Expenses'], name='Expenses (Cr)', 
            marker_color=colors['expenses'],
            hovertemplate='<b>%{x|%Y-Q%q}</b><br>Expenses: %{y:.2f} Cr<extra></extra>'
        ))
        fig_cashflow.add_trace(go.Scatter(
            x=master_timeline, y=cashflow_display['EBITDA'], name='Net Cash Flow (Cr)', 
            mode='lines+markers', line=dict(color=colors['net_cash']),
            hovertemplate='<b>%{x|%Y-Q%q}</b><br>Net Cash Flow: %{y:.2f} Cr<extra></extra>'
        ))
        fig_cashflow.update_layout(
            barmode='relative', 
            height=400, 
            title_text="Quarterly Inflows, Outflows, and Net Position", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Amount (in Cr)"
        )
        st.plotly_chart(fig_cashflow, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Quarterly Funding View (in Cr)")
            fig_funding = go.Figure()
            fig_funding.add_trace(go.Bar(
                x=master_timeline, y=financials_display['Equity Inflow'], name='Equity Inflow',
                marker_color=colors['equity'],
                hovertemplate='<b>%{x|%Y-Q%q}</b><br>Equity Inflow: %{y:.2f} Cr<extra></extra>'
            ))
            fig_funding.add_trace(go.Bar(
                x=master_timeline, y=financials_display['Surplus/(Deficit)'], name='Surplus/(Deficit)',
                marker_color=colors['surplus'],
                hovertemplate='<b>%{x|%Y-Q%q}</b><br>Surplus/(Deficit): %{y:.2f} Cr<extra></extra>'
            ))
            fig_funding.update_layout(
                barmode='relative', height=400, yaxis_title="Amount (in Cr)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_funding, use_container_width=True)

        with col2:
            st.subheader("Cash Balance Over Time (in Cr)")
            fig_balance = go.Figure()
            fig_balance.add_trace(go.Scatter(
                x=master_timeline, y=financials_display['Closing Cash'], name='Cash Balance',
                fill='tozeroy', line=dict(color=colors['net_cash']),
                hovertemplate='<b>%{x|%Y-Q%q}</b><br>Cash Balance: %{y:.2f} Cr<extra></extra>'
            ))
            fig_balance.update_layout(
                height=400, yaxis_title="Amount (in Cr)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_balance, use_container_width=True)

    with tab2:
        st.header("Area & Revenue Calculations")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Area Buildup")
            st.dataframe(intermediate['area_calcs'].style.set_table_styles(table_styles).format(formatter={"Value": lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x}))
        with col2:
            st.subheader("Total Revenue Calculation")
            st.dataframe(revenue_details_display.style.set_table_styles(table_styles).format({
                'Quarterly Sales Distribution': '{:.2%}',
                'Escalated Price per Sq Ft': '₹{:,.0f}',
                'Quarterly Revenue': '₹{:,.2f} Cr'
            }))
        
        st.markdown("---")
        st.header("Collections Calculation")
        st.dataframe(collection_details_display.style.set_table_styles(table_styles).format({
            "Cumulative Revenue": "₹{:,.2f} Cr",
            "Cumulative Collection %": "{:.2%}",
            "Total Collections Due by Quarter End": "₹{:,.2f} Cr",
            "Quarterly Collections": "₹{:,.2f} Cr"
        }))

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
