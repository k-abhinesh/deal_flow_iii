import pandas as pd
import numpy as np
from math import ceil
import os

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

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Stage'})
    
    return df, dynamic_project_end_date


def generate_other_costs_phasing(master_timeline_q, project_end_date):
    """
    Generates phasing for other costs based on project rules.
    """
    project_end_date = pd.to_datetime(project_end_date)
    
    cost_items = ['Consultant Cost', 'Marketing Cost', 'Misc. Approval Cost', 'Plan Sanction', 'Sales Lounge']
    df = pd.DataFrame(0.0, index=cost_items, columns=master_timeline_q)
    
    construction_end_period = pd.Period(project_end_date, freq='Q')
    construction_timeline_labels = [
        q for q in master_timeline_q 
        if pd.Period(f"{q.split(' ')[1]}{q.split(' ')[0]}", freq='Q') <= construction_end_period
    ]
    total_construction_q = len(construction_timeline_labels)

    if len(master_timeline_q) > 1: df.loc['Plan Sanction', master_timeline_q[1]] = 1.0
    if len(master_timeline_q) > 2:
        df.loc['Sales Lounge', master_timeline_q[1]] = 0.5
        df.loc['Sales Lounge', master_timeline_q[2]] = 0.5

    project_end_q_label = f"Q{project_end_date.quarter} {project_end_date.year}"
    if project_end_q_label in df.columns:
        df.loc['Misc. Approval Cost', project_end_q_label] = 1.0

    consultant_dist = pd.Series(0.0, index=construction_timeline_labels)
    if total_construction_q > 0: consultant_dist.iloc[0] = 0.05
    if total_construction_q > 1: consultant_dist.iloc[1] = 0.15
    if total_construction_q > 2: consultant_dist.iloc[2] = 0.20
    
    if total_construction_q > 8:
        middle_start_idx = 3
        middle_end_idx = total_construction_q - 5
        consultant_dist.iloc[middle_end_idx:] = 0.05
        if middle_end_idx > middle_start_idx:
            middle_len = middle_end_idx - middle_start_idx
            middle_values = np.linspace(0.05, 0.025, middle_len)
            consultant_dist.iloc[middle_start_idx:middle_end_idx] = middle_values
    df.loc['Consultant Cost', construction_timeline_labels] = _normalize_distribution(consultant_dist).values

    # Updated marketing cost distribution logic
    marketing_dist = pd.Series(0.0, index=construction_timeline_labels)
    if total_construction_q > 1: marketing_dist.iloc[1] = 0.05  # Q2: 5%
    if total_construction_q > 2: marketing_dist.iloc[2] = 0.15  # Q3: 15%
    if total_construction_q > 3: marketing_dist.iloc[3] = 0.15  # Q4: 15%
    
    # After the initial ramp-up, immediately start a gradual decline to 5%
    if total_construction_q > 4:
        taper_start_idx = 4 # Start tapering from the 5th quarter
        taper_len = total_construction_q - taper_start_idx
        if taper_len > 0:
            # FIX: Create a smooth decline from 10% down to 5%
            taper_values = np.linspace(0.10, 0.05, taper_len) 
            marketing_dist.iloc[taper_start_idx:] = taper_values
            
    df.loc['Marketing Cost', construction_timeline_labels] = _normalize_distribution(marketing_dist).values
    
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
        
    post_project_q = pd.Period(project_end_date, freq='Q') + 1
    post_project_q_label = f"Q{post_project_q.quarter} {post_project_q.year}"
    if post_project_q_label in df.index:
        df.loc[post_project_q_label, 'Distribution'] = 0.20
    
    remaining_pct = 1.0 - df['Distribution'].sum()
    dist_start_idx = sales_start_idx + 2
    
    try:
        dist_end_idx = df.index.get_loc(post_project_q_label)
    except KeyError:
        dist_end_idx = len(df)

    if dist_end_idx > dist_start_idx:
        num_quarters_for_dist = dist_end_idx - dist_start_idx
        if num_quarters_for_dist > 0:
            pct_per_q = remaining_pct / num_quarters_for_dist
            df.iloc[dist_start_idx:dist_end_idx, 0] = pct_per_q
    
    df['Distribution'] = _normalize_distribution(df['Distribution'])
    return df.reset_index()


def generate_collection_distribution(sales_dist_df, master_timeline_q):
    """
    Generates a collection distribution based on a lag from sales.
    """
    sales_series = sales_dist_df.set_index('Quarter')['Distribution']
    collections = pd.Series(0.0, index=master_timeline_q)

    for quarter, sales_pct in sales_series.items():
        if sales_pct > 0:
            try:
                q_idx = master_timeline_q.index(quarter)
                if q_idx < len(collections):     collections.iloc[q_idx] += sales_pct * 0.10
                if q_idx + 1 < len(collections): collections.iloc[q_idx + 1] += sales_pct * 0.40
                if q_idx + 2 < len(collections): collections.iloc[q_idx + 2] += sales_pct * 0.30
                if q_idx + 3 < len(collections): collections.iloc[q_idx + 3] += sales_pct * 0.20
            except (ValueError, IndexError):
                continue

    collections = _normalize_distribution(collections)
    return pd.DataFrame({'Quarter': collections.index, 'Distribution': collections.values})


def generate_admin_cost_schedule(total_saleable_area, admin_timeline_q):
    """
    Generates the admin cost schedule with extra empty columns for editing.
    """
    if not admin_timeline_q or total_saleable_area == 0:
        return pd.DataFrame({'Quarter': admin_timeline_q, 'Cost per Quarter': 0.0})

    total_admin_cost = 1500 * total_saleable_area
    quarterly_admin_cost = total_admin_cost / len(admin_timeline_q)
    
    extended_admin_timeline = list(admin_timeline_q)
    if admin_timeline_q:
        last_q_str = admin_timeline_q[-1]
        last_q = pd.Period(f"{last_q_str.split(' ')[1]}{last_q_str.split(' ')[0]}", freq='Q')
        for i in range(1, 9):
            next_q = last_q + i
            extended_admin_timeline.append(f"Q{next_q.quarter} {next_q.year}")
        
    df = pd.DataFrame({
        'Quarter': admin_timeline_q,
        'Cost per Quarter': quarterly_admin_cost
    })
    
    df = df.set_index('Quarter').reindex(extended_admin_timeline, fill_value=0).reset_index()
    
    return df
