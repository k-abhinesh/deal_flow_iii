import pandas as pd
import streamlit as st

def validate_schedules(inputs):
    """Checks if the distribution percentages in user-edited schedules sum to 100%."""
    errors = []
    tolerance = 0.1 # Use a tolerance for user-edited percentages
    
    df_construct = inputs['construction_phasing']
    q_cols_construct = [col for col in df_construct.columns if isinstance(col, str) and col.startswith('Q')]
    for _, row in df_construct.iterrows():
        total = row[q_cols_construct].sum()
        if abs(total - 100.0) > tolerance: errors.append(f"Construction Phasing Error: '{row['Stage']}' sums to {total:.2f}%, not 100%.")
    
    df_other = inputs['other_costs_phasing']
    q_cols_other = [col for col in df_other.columns if isinstance(col, str) and col.startswith('Q')]
    for _, row in df_other.iterrows():
        total = row[q_cols_other].sum()
        if abs(total - 100.0) > tolerance and total > 0: errors.append(f"Other Costs Phasing Error: '{row['Cost Item']}' sums to {total:.2f}%, not 100%.")
    
    if abs(inputs['sales_dist']['Distribution'].sum() - 100.0) > tolerance: errors.append(f"Sales Distribution Error: Sums to {inputs['sales_dist']['Distribution'].sum():.2f}%, not 100%.")
    
    if 'payment_phasing' in inputs:
        total_pct = inputs['payment_phasing']['Payment (%)'].sum()
        if abs(total_pct - 100.0) > tolerance:
            errors.append(f"Payment Phasing Error: Percentages sum to {total_pct:.1f}%, not 100%.")
            
    return errors

def calculate_financial_model(inputs):
    """Main function to run all financial calculations."""
    intermediate_data = {}
    all_q_strs = set()
    for key in ['construction_phasing', 'other_costs_phasing', 'sales_dist', 'payment_phasing']:
        df = inputs.get(key)
        if df is None or df.empty: continue
        quarter_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Q')]
        if not quarter_cols and 'Quarter' in df.columns: quarter_cols = df['Quarter'].unique()
        all_q_strs.update(quarter_cols)
    
    def reformat_q_string_for_period(q_str):
        if not isinstance(q_str, str) or ' ' not in q_str: return None
        parts = q_str.split(' ')
        return f"{parts[1]}{parts[0]}"

    all_timestamps = [pd.Period(reformat_q_string_for_period(q), freq='Q').to_timestamp() for q in all_q_strs if q and ' ' in q]
    start_date, end_date = (min(all_timestamps), max(all_timestamps)) if all_timestamps else (pd.to_datetime(inputs['start_date']), st.session_state.get('dynamic_end_date', pd.to_datetime(inputs['start_date']) + pd.DateOffset(years=3)))
    master_timeline = pd.date_range(start=start_date, end=end_date, freq='Q')
    master_timeline_labels = [f"Q{q.quarter} {q.year}" for q in master_timeline]
    
    total_bua = inputs['land_area'] * inputs['far']
    total_saleable_area = inputs.get('total_saleable_area', total_bua * 1.1)
    townhouse_sa = total_saleable_area
    intermediate_data['area_calcs'] = pd.DataFrame([{"Metric": "Land Area (sq ft)", "Value": inputs['land_area']}, {"Metric": "FAR", "Value": inputs['far']}, {"Metric": "Total BUA (sq ft)", "Value": total_bua}, {"Metric": "Total Saleable Area (sq ft)", "Value": total_saleable_area}]).set_index("Metric")
    
    sales_dist = pd.Series(inputs['sales_dist']['Distribution'].fillna(0).values / 100.0, index=inputs['sales_dist']['Quarter'].values)
    first_sale_quarter = sales_dist[sales_dist > 0].index[0] if not sales_dist[sales_dist > 0].empty else None
    
    project_end_date = inputs.get('end_date', master_timeline[-1].to_pydatetime())
    project_timeline = pd.date_range(start=inputs['start_date'], end=project_end_date, freq='Q')
    project_timeline_labels = [f"Q{q.quarter} {q.year}" for q in project_timeline]
    
    sales_prices, escalation_count, sales_started = [], 0, False
    price = inputs['unit_price'] # Correct: Initialize price BEFORE the loop
    for quarter_label in master_timeline_labels:
        if not sales_started and quarter_label == first_sale_quarter:
            sales_started = True

        # Once sales start, escalate the price in each subsequent quarter
        # This check is now independent of any end_date
        if sales_started:
            # Apply escalation for every quarter after the first sale
            if escalation_count > 0:
                price *= (1 + inputs['escalation'])
            escalation_count += 1

        sales_prices.append(price)
        
    revenue_details = pd.DataFrame({'Quarterly Sales Distribution': sales_dist.reindex(master_timeline_labels, fill_value=0), 'Escalated Price per Sq Ft': pd.Series(sales_prices, index=master_timeline_labels)})
    revenue_details['Quarterly Revenue'] = townhouse_sa * revenue_details['Quarterly Sales Distribution'] * revenue_details['Escalated Price per Sq Ft']
    total_revenue = revenue_details['Quarterly Revenue'].sum()
    intermediate_data['revenue_details'] = revenue_details
    
    cashflow = pd.DataFrame(index=master_timeline_labels)
    
    # --- CORRECTED Collection Logic based on Absolute Payment Phasing ---
    def to_period_index(index):
        # Converts 'Q1 2025' into a sortable Period object
        # Corrected regex to properly format the date string
        return pd.PeriodIndex(index.str.replace(r'Q(\d)\s(\d{4})', r'\2-Q\1', regex=True), freq='Q')

    payment_phasing = pd.Series(inputs['payment_phasing']['Payment (%)'].fillna(0).values / 100.0, index=inputs['payment_phasing']['Quarter'].values)
    payment_phasing.index = to_period_index(payment_phasing.index)
    payment_phasing = payment_phasing.sort_index() # Ensure chronological order
    payment_phasing_cumulative = payment_phasing.cumsum()
    
    sales_revenue_by_quarter = revenue_details['Quarterly Revenue']
    sales_revenue_by_quarter.index = to_period_index(sales_revenue_by_quarter.index)
    
    # Create columns for the breakdown table only from quarters with actual sales
    positive_sales_quarters = sales_revenue_by_quarter[sales_revenue_by_quarter > 0].index
    breakdown_columns = [f"Sales from Q{q.quarter} {q.year}" for q in positive_sales_quarters]
    collections_breakdown = pd.DataFrame(0.0, index=master_timeline_labels, columns=breakdown_columns)

    for sale_q_period, revenue_from_sale in sales_revenue_by_quarter.items():
        if revenue_from_sale > 0:
            sale_q_str = f"Q{sale_q_period.quarter} {sale_q_period.year}"
            
            # Catch-up payment: Collect all payments due up to and including the sale quarter
            due_pct = payment_phasing_cumulative.asof(sale_q_period)
            if pd.notna(due_pct):
                collections_breakdown.loc[sale_q_str, f"Sales from {sale_q_str}"] += revenue_from_sale * due_pct
            
            # Future payments: Schedule payments due after the sale quarter
            future_payments = payment_phasing[payment_phasing.index > sale_q_period]
            for payment_q_period, pct in future_payments.items():
                payment_q_str = f"Q{payment_q_period.quarter} {payment_q_period.year}"
                if payment_q_str in collections_breakdown.index:
                    collections_breakdown.loc[payment_q_str, f"Sales from {sale_q_str}"] += revenue_from_sale * pct

    total_sales_data = {
        f"Sales from Q{q.quarter} {q.year}": revenue
        for q, revenue in sales_revenue_by_quarter.items() if revenue > 0
    }
    total_sales_row = pd.DataFrame(total_sales_data, index=['Total Sales'])
    collections_breakdown = pd.concat([total_sales_row, collections_breakdown])
    collections_breakdown = collections_breakdown.fillna(0) # Clean up any NaNs from concatenation

    collections_breakdown['Total Collections'] = collections_breakdown.sum(axis=1)
    cashflow['Collections'] = collections_breakdown['Total Collections']
    intermediate_data['collections_breakdown'] = collections_breakdown

    # --- Cost Calculations ---
    construction_stage_costs = inputs['construction_stage_costs'].set_index('Stage')
    construction_stage_costs['Total Cost'] = construction_stage_costs['Cost Rate (per sq ft of Total SA)'].fillna(0) * total_saleable_area
    total_construction_cost = construction_stage_costs['Total Cost'].sum()
    intermediate_data['construction_breakdown'] = construction_stage_costs['Total Cost']

    total_consultant_cost, total_marketing_cost, total_misc_approval_cost = total_saleable_area * inputs['consultant_cost_psf'], total_revenue * inputs['marketing_cost_pct_revenue'], total_saleable_area * inputs['misc_approval_cost_psf']
    total_admin_cost = total_revenue * 0.025

    cost_summary = pd.DataFrame([{"Cost Category": "Total Construction Cost", "Value": total_construction_cost}, {"Cost Category": "Total Consultant Cost", "Value": total_consultant_cost}, {"Cost Category": "Total Marketing Cost", "Value": total_marketing_cost}, {"Cost Category": "Total Misc. Approval Cost", "Value": total_misc_approval_cost}, {"Cost Category": "Plan Sanction (Lump Sum)", "Value": inputs['plan_sanction_cost']}, {"Cost Category": "Sales Lounge (Lump Sum)", "Value": inputs['sales_lounge_cost']}, {"Cost Category": "Total Admin Cost", "Value": total_admin_cost}]).set_index("Cost Category")
    
    total_brokerage_cost = cashflow['Collections'].sum() * inputs['brokerage_fee']
    cashflow['Cost: Brokerage'] = cashflow['Collections'] * inputs['brokerage_fee']
    cost_summary.loc["Total Brokerage Cost"] = total_brokerage_cost
    
    construction_phasing = inputs['construction_phasing'].set_index('Stage')
    q_cols = [col for col in construction_phasing.columns if col.startswith('Q')]
    construction_phasing[q_cols] /= 100.0
    for stage_name, stage_data in construction_stage_costs.iterrows():
        cashflow[f'Cost: {stage_name}'] = construction_phasing.loc[stage_name].reindex(master_timeline_labels, fill_value=0).astype(float) * stage_data['Total Cost']
    
    other_costs_phasing = inputs['other_costs_phasing'].set_index('Cost Item')
    other_q_cols = [col for col in other_costs_phasing.columns if isinstance(col, str) and col.startswith('Q')]
    other_costs_phasing[other_q_cols] /= 100.0
    cost_items_map = {'Consultant Cost': total_consultant_cost, 'Marketing Cost': total_marketing_cost, 'Misc. Approval Cost': total_misc_approval_cost, 'Plan Sanction': inputs['plan_sanction_cost'], 'Sales Lounge': inputs['sales_lounge_cost'], 'Admin Cost': total_admin_cost}
    for item, total_cost in cost_items_map.items():
        if item in other_costs_phasing.index: cashflow[f'Cost: {item}'] = other_costs_phasing.loc[item].reindex(master_timeline_labels, fill_value=0).astype(float) * total_cost
    
    cost_columns = [col for col in cashflow.columns if col.startswith('Cost: ')]
    cashflow['Total Expenses (pre-interest)'] = cashflow[cost_columns].sum(axis=1)
    
    # --- Financials (Debt/Equity) Calculations ---
    financials = pd.DataFrame(index=master_timeline_labels)
    quarterly_interest_rate = inputs['debt_interest_rate'] / 4.0
    
    opening_cash_list, equity_inflow_list, debt_drawdown_list = [], [], []
    interest_expense_list, debt_repayment_list, closing_cash_list = [], [], []
    opening_debt_list, closing_debt_list = [], []
    
    opening_cash, opening_debt = 0.0, 0.0
    for q in master_timeline_labels:
        collections = cashflow.loc[q, 'Collections']
        expenses_pre_interest = cashflow.loc[q, 'Total Expenses (pre-interest)']
        interest_expense = opening_debt * quarterly_interest_rate
        cash_flow_after_interest = collections - expenses_pre_interest - interest_expense
        pre_financing_balance = opening_cash + cash_flow_after_interest
        
        equity_inflow, debt_drawdown, debt_repayment = 0.0, 0.0, 0.0
        if pre_financing_balance < 0:
            shortfall = -pre_financing_balance
            if collections > 0:
                debt_drawdown = shortfall
            else:
                equity_inflow = shortfall
        else:
            debt_repayment = min(pre_financing_balance, opening_debt)

        closing_cash = pre_financing_balance + equity_inflow + debt_drawdown - debt_repayment
        closing_debt = opening_debt + debt_drawdown - debt_repayment

        opening_cash_list.append(opening_cash)
        equity_inflow_list.append(equity_inflow)
        debt_drawdown_list.append(debt_drawdown)
        interest_expense_list.append(interest_expense)
        debt_repayment_list.append(debt_repayment)
        closing_cash_list.append(closing_cash)
        opening_debt_list.append(opening_debt)
        closing_debt_list.append(closing_debt)
        
        opening_cash = closing_cash
        opening_debt = closing_debt

    cashflow['Cost: Interest'] = interest_expense_list
    cashflow['Total Expenses'] = cashflow['Total Expenses (pre-interest)'] + cashflow['Cost: Interest']
    cashflow['Net Cash Flow'] = cashflow['Collections'] - cashflow['Total Expenses']
    
    financials['Opening Cash'] = opening_cash_list
    financials['Net Cash Flow'] = cashflow['Net Cash Flow']
    financials['Equity Inflow'] = equity_inflow_list
    financials['Debt Drawdown'] = debt_drawdown_list
    financials['Debt Repayment'] = debt_repayment_list
    financials['Closing Cash'] = closing_cash_list
    financials['Opening Debt'] = opening_debt_list
    financials['Closing Debt'] = closing_debt_list
    financials['Interest Expense'] = interest_expense_list
    
    total_project_cost = cashflow['Total Expenses'].sum()
    equity_required = sum(equity_inflow_list)
    debt_required = sum(debt_drawdown_list)
    interest_on_debt = sum(interest_expense_list)
    
    kpis = {"Projected Revenue": f"₹{total_revenue/1e7:.2f} Cr", 
            "Total Construction Cost": f"₹{total_construction_cost/1e7:.2f} Cr",
            "Total Project Cost": f"₹{total_project_cost/1e7:.2f} Cr",
            "Equity Required": f"₹{equity_required/1e7:.2f} Cr",
            "Debt Required": f"₹{debt_required/1e7:.2f} Cr",
            "Interest on Debt": f"₹{interest_on_debt/1e7:.2f} Cr"}
    
    interest_cost_df = pd.DataFrame({'Value': cashflow['Cost: Interest'].sum()}, index=['Total Interest Cost'])
    cost_summary = pd.concat([cost_summary, interest_cost_df])
    intermediate_data['cost_summary'] = cost_summary
    
    return {"kpis": kpis, "cashflow": cashflow, "financials": financials, "intermediate": intermediate_data, "master_timeline": master_timeline}
