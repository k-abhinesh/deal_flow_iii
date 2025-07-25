import pandas as pd
import numpy as np
from math import ceil

def _normalize_distribution(series):
    """Helper function to ensure a distribution series sums to exactly 1.0."""
    if series.sum() == 0:
        return series
    return series / series.sum()

def generate_construction_phasing(start_date, land_area_sqft, num_floors):
    """
    Dynamically calculates construction timeline and generates the phasing distribution.
    Returns the DataFrame (with extra empty columns) and the calculated project end date.
    """
    # --- Calculate Durations in Quarters based on Rules ---
    land_area_acres = land_area_sqft / 43560
    
    if land_area_acres < 1:
        excavation_q = 2
    elif 1 <= land_area_acres <= 2:
        excavation_q = 3
    else:
        excavation_q = 4
        
    rcc_months = num_floors / 2
    rcc_q = ceil(rcc_months / 3)
    
    if 1 <= num_floors <= 9: mep_q, mep_overlap_q = 2, 0
    elif 10 <= num_floors <= 25: mep_q, mep_overlap_q = 4, 1
    elif 26 <= num_floors <= 35: mep_q, mep_overlap_q = 5, 2
    else: mep_q, mep_overlap_q = 6, 3
        
    if 1 <= num_floors <= 9: finishing_q, finishing_overlap_q = 2, 0
    elif 10 <= num_floors <= 25: finishing_q, finishing_overlap_q = 4, 1
    elif 26 <= num_floors <= 35: finishing_q, finishing_overlap_q = 5, 2
    else: finishing_q, finishing_overlap_q = 6, 3

    if land_area_acres < 1: infra_q = 2
    elif 1 <= land_area_acres <= 2: infra_q = 3
    else: infra_q = 4
    infra_overlap_q = 1

    # --- Determine total timeline length by simulating stage indices ---
    excavation_start_idx = 2
    excavation_end_idx = excavation_start_idx + excavation_q - 1
    rcc_start_idx = excavation_end_idx + 1
    rcc_end_idx = rcc_start_idx + rcc_q - 1
    mep_start_idx = max(rcc_start_idx, rcc_end_idx - mep_overlap_q + 1)
    mep_end_idx = mep_start_idx + mep_q - 1
    finishing_start_idx = max(mep_start_idx, mep_end_idx - finishing_overlap_q + 1)
    finishing_end_idx = finishing_start_idx + finishing_q - 1
    infra_start_idx = max(finishing_start_idx, finishing_end_idx - infra_overlap_q + 1)
    infra_end_idx = infra_start_idx + infra_q - 1
    
    total_construction_quarters = infra_end_idx + 1
    
    # --- Generate Timeline and DataFrame ---
    extended_total_quarters = total_construction_quarters + 8
    project_timeline = pd.date_range(start=start_date, periods=extended_total_quarters, freq='Q')
    project_timeline_labels = [f"Q{q.quarter} {q.year}" for q in project_timeline]
    dynamic_project_end_date = pd.date_range(start=start_date, periods=total_construction_quarters, freq='Q')[-1].date()

    stages = ["Excavation and Foundation", "RCC", "MEP", "Finishing", "Infra and Amenities"]
    df = pd.DataFrame(0.0, index=stages, columns=project_timeline_labels)

    # --- Map Durations to Timeline ---
    df.loc["Excavation and Foundation", project_timeline_labels[excavation_start_idx : excavation_end_idx + 1]] = 1.0 / excavation_q
    df.loc["RCC", project_timeline_labels[rcc_start_idx : rcc_end_idx + 1]] = 1.0 / rcc_q
    df.loc["MEP", project_timeline_labels[mep_start_idx : mep_end_idx + 1]] = 1.0 / mep_q
    df.loc["Finishing", project_timeline_labels[finishing_start_idx : finishing_end_idx + 1]] = 1.0 / finishing_q
    df.loc["Infra and Amenities", project_timeline_labels[infra_start_idx : infra_end_idx + 1]] = 1.0 / infra_q

    # Convert to percentage
    q_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Q')]
    df[q_cols] = df[q_cols] * 100

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Stage'})
    
    return df, dynamic_project_end_date

def generate_payment_phasing(master_timeline_q, construction_phasing_df, sales_dist_df):
    """
    Generates a default absolute payment phasing schedule based on milestones.
    """
    q_cols = [col for col in construction_phasing_df.columns if col.startswith('Q')]
    
    def get_milestone_end_quarter(stage_name):
        stage_row = construction_phasing_df[construction_phasing_df['Stage'] == stage_name]
        if stage_row.empty: return None
        for q in reversed(q_cols):
            if stage_row[q].iloc[0] > 0:
                return q
        return None

    milestone_pcts = {
        'On Excavation Completion': (get_milestone_end_quarter("Excavation and Foundation"), 15.0),
        'On RCC Completion': (get_milestone_end_quarter("RCC"), 30.0),
        'On MEP Completion': (get_milestone_end_quarter("MEP"), 15.0),
        'On Finishing Completion': (get_milestone_end_quarter("Finishing"), 20.0),
        'On Infra/Handover': (get_milestone_end_quarter("Infra and Amenities"), 10.0)
    }

    dist = pd.Series(0.0, index=master_timeline_q)
    
    # Place booking fee in the first quarter with sales
    first_sales_q = sales_dist_df[sales_dist_df['Distribution'] > 0]['Quarter'].iloc[0]
    if first_sales_q in dist.index:
        dist[first_sales_q] += 10.0

    # Add milestone payments
    for milestone, (quarter, pct) in milestone_pcts.items():
        if quarter and quarter in dist.index:
            dist[quarter] += pct
    
    if dist.sum() > 0:
        dist_normalized = (dist / dist.sum()) * 100
    else:
        dist_normalized = dist
    
    return pd.DataFrame({'Quarter': dist_normalized.index, 'Payment (%)': dist_normalized.values})


def generate_other_costs_phasing(master_timeline_q, project_end_date, last_sales_quarter_label):
    """
    Generates phasing for other costs based on project rules.
    """
    project_end_date = pd.to_datetime(project_end_date)
    
    cost_items = ['Consultant Cost', 'Marketing Cost', 'Misc. Approval Cost', 'Plan Sanction', 'Sales Lounge', 'Admin Cost']
    df = pd.DataFrame(0.0, index=cost_items, columns=master_timeline_q)
    
    construction_end_period = pd.Period(project_end_date, freq='Q')
    construction_timeline_labels = [q for q in master_timeline_q if pd.Period(f"{q.split(' ')[1]}{q.split(' ')[0]}", freq='Q') <= construction_end_period]
    total_construction_q = len(construction_timeline_labels)

    consultant_dist = pd.Series(0.0, index=construction_timeline_labels)
    if total_construction_q > 0: consultant_dist.iloc[0] = 0.05
    if total_construction_q > 1: consultant_dist.iloc[1] = 0.15
    if total_construction_q > 2: consultant_dist.iloc[2] = 0.20
    if total_construction_q > 3: consultant_dist.iloc[-1] = 0.05

    if total_construction_q > 4:
        fixed_pct_sum = consultant_dist.sum()
        remaining_pct = 1.0 - fixed_pct_sum
        middle_quarters = consultant_dist[consultant_dist == 0.0].index
        if not middle_quarters.empty and remaining_pct > 0:
            pct_per_q = remaining_pct / len(middle_quarters)
            consultant_dist[middle_quarters] = pct_per_q
            
    df.loc['Consultant Cost', construction_timeline_labels] = _normalize_distribution(consultant_dist).values

    last_sales_period = pd.Period(f"{last_sales_quarter_label.split(' ')[1]}{last_sales_quarter_label.split(' ')[0]}", freq='Q')
    marketing_timeline_labels = [q for q in master_timeline_q if pd.Period(f"{q.split(' ')[1]}{q.split(' ')[0]}", freq='Q') <= last_sales_period]
    total_marketing_q = len(marketing_timeline_labels)

    marketing_dist = pd.Series(0.0, index=marketing_timeline_labels)
    if total_marketing_q > 1: marketing_dist.iloc[1] = 0.05
    if total_marketing_q > 2: marketing_dist.iloc[2] = 0.15
    if total_marketing_q > 3: marketing_dist.iloc[3] = 0.15
    
    if total_marketing_q > 4:
        remaining_pct = 1.0 - marketing_dist.sum()
        distribute_quarters = marketing_dist.index[4:]
        if not distribute_quarters.empty and remaining_pct > 0:
            pct_per_q = remaining_pct / len(distribute_quarters)
            marketing_dist[distribute_quarters] = pct_per_q

    df.loc['Marketing Cost', marketing_timeline_labels] = _normalize_distribution(marketing_dist).values

    first_quarter_label = master_timeline_q[0]
    first_admin_period = pd.Period(f"{first_quarter_label.split(' ')[1]}{first_quarter_label.split(' ')[0]}", freq='Q')
    
    admin_timeline_labels = [q for q in master_timeline_q if first_admin_period <= pd.Period(f"{q.split(' ')[1]}{q.split(' ')[0]}", freq='Q') <= last_sales_period]
    total_admin_q = len(admin_timeline_labels)

    admin_dist = pd.Series(0.0, index=admin_timeline_labels)
    if total_admin_q > 0: admin_dist.iloc[0] = 0.005
    if total_admin_q > 1: admin_dist.iloc[1] = 0.005

    if total_admin_q > 2:
        remaining_pct = 1.0 - admin_dist.sum()
        distribute_quarters = admin_dist.index[2:]
        if not distribute_quarters.empty and remaining_pct > 0:
            pct_per_q = remaining_pct / len(distribute_quarters)
            admin_dist[distribute_quarters] = pct_per_q
            
    df.loc['Admin Cost', admin_timeline_labels] = _normalize_distribution(admin_dist).values

    if len(master_timeline_q) > 1: df.loc['Plan Sanction', master_timeline_q[1]] = 1.0
    if len(master_timeline_q) > 2:
        df.loc['Sales Lounge', master_timeline_q[1]] = 0.5
        df.loc['Sales Lounge', master_timeline_q[2]] = 0.5

    project_end_q_label = f"Q{project_end_date.quarter} {project_end_date.year}"
    if project_end_q_label in df.columns:
        df.loc['Misc. Approval Cost', project_end_q_label] = 1.0
    
    q_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Q')]
    df[q_cols] = df[q_cols] * 100

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Cost Item'})
    return df

def generate_sales_distribution(master_timeline_q, project_end_date):
    """
    Generates the sales distribution based on project rules.
    """
    project_end_date = pd.to_datetime(project_end_date)
    
    df = pd.DataFrame({'Quarter': master_timeline_q, 'Distribution': 0.0})
    df.set_index('Quarter', inplace=True)
    
    sales_start_idx = 2
    
    if len(df) > sales_start_idx:     df.iloc[sales_start_idx, 0] = 0.15
    if len(df) > sales_start_idx + 1: df.iloc[sales_start_idx + 1, 0] = 0.15
        
    post_construction_start_period = pd.Period(project_end_date, freq='Q') + 1
    post_construction_labels = []
    for i in range(4):
        current_period = post_construction_start_period + i
        current_label = f"Q{current_period.quarter} {current_period.year}"
        if current_label in df.index:
            df.loc[current_label, 'Distribution'] = 0.05
            post_construction_labels.append(current_label)
    
    remaining_pct = 1.0 - df['Distribution'].sum()
    
    dist_start_idx = sales_start_idx + 2
    dist_end_idx = -1
    if post_construction_labels:
        try:
            dist_end_idx = df.index.get_loc(post_construction_labels[0])
        except KeyError:
            dist_end_idx = len(df)
    else:
        dist_end_idx = len(df)

    if dist_end_idx > dist_start_idx:
        num_quarters_for_dist = dist_end_idx - dist_start_idx
        if num_quarters_for_dist > 0:
            pct_per_q = remaining_pct / num_quarters_for_dist
            df.iloc[dist_start_idx:dist_end_idx, 0] = pct_per_q
    
    df['Distribution'] = _normalize_distribution(df['Distribution'])
    
    df['Distribution'] = df['Distribution'] * 100
    
    return df.reset_index()
