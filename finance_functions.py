import pandas as pd
import streamlit as st

def validate_schedules(inputs):
    """Checks if the distribution percentages in user-edited schedules sum to 100%."""
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
    
    # FIX: Renamed 'EBITDA' to 'Net Cash Flow'
    cashflow['Net Cash Flow'] = cashflow['Collections'] - cashflow['Total Expenses']
    
    # --- 4. Funding Requirements ---
    # FIX: Updated to use 'Net Cash Flow'
    financials = pd.DataFrame({'Surplus/(Deficit)': cashflow['Net Cash Flow']}, index=master_timeline_labels)
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
    kpis = {"Projected Revenue": f"₹{total_revenue/1e7:.2f} Cr", "Total Construction Cost": f"₹{total_construction_cost/1e7:.2f} Cr", "Total Project Cost": f"₹{total_project_cost/1e7:.2f} Cr", "Equity Required": f"₹{equity_required/1e7:.2f} Cr"}
    return {"kpis": kpis, "cashflow": cashflow, "financials": financials, "intermediate": intermediate_data, "master_timeline": master_timeline}
