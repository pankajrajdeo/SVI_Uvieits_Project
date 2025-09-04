#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
from scipy import stats
import re
from datetime import datetime
import matplotlib as mpl

# Set global plot style and font
try:
    mpl.rcParams['font.family'] = 'Arial'
except:
    print("Arial font not found, using default sans-serif.")
sns.set_theme(style="whitegrid")

# Create output directory if it doesn't exist
output_dir = 'svi_treatment_delay_analysis'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
# Updated file path to use the new CSV file
file_path = 'SVI_Pankaj_data_1_updated_merged_new.csv'
df = pd.read_csv(file_path, low_memory=False)

# Check for date columns in the new CSV file
print("\nChecking date columns in the new CSV file...")
date_columns = [col for col in df.columns if 'date' in col.lower()]
print(f"Found {len(date_columns)} date-related columns in the CSV file.")

# Check specifically for the columns we're interested in
jia_date_cols = [
    'date_diagnosis', 'date of diagnosis', 
    'date of arth diagy', 'date of arth onsety',
    'date of arth diagd', 'date of arth diagm',
    'date of arth diagnosi', 'date of arthritis onset'
]

uveitis_date_cols = [
    'date of uv diagnosisy', 'date of uv onsety',
    'date of uv diagnosisd', 'date of uv diagnosism',
    'date of uveitis diagnosis', 'date of uv onsetd',
    'date of uv onsetm'
]

medication_date_cols = [
    'medication start date (list distinct)',
    'eye drop start date (list distinct)'
]

# Check which of these columns exist in the new file
jia_cols_in_file = [col for col in jia_date_cols if col in df.columns]
uveitis_cols_in_file = [col for col in uveitis_date_cols if col in df.columns]
med_cols_in_file = [col for col in medication_date_cols if col in df.columns]

print("\nJIA date columns found in the file:")
for col in jia_cols_in_file:
    sample_values = df[col].dropna().head(3).tolist()
    print(f"  - {col}: {sample_values}")

print("\nUveitis date columns found in the file:")
for col in uveitis_cols_in_file:
    sample_values = df[col].dropna().head(3).tolist()
    print(f"  - {col}: {sample_values}")

print("\nMedication date columns found in the file:")
for col in med_cols_in_file:
    sample_values = df[col].dropna().head(3).tolist()
    print(f"  - {col}: {sample_values}")

# Define systemic immunosuppressant medications
SYSTEMIC_MEDS = [
    'methotrexate', 'adalimumab', 'infliximab', 'golimumab', 'certolizumab',
    'etanercept', 'abatacept', 'tocilizumab', 'rituximab', 'anakinra',
    'canakinumab', 'azathioprine', 'mycophenolate', 'cyclosporine',
    'tacrolimus', 'cyclophosphamide', 'leflunomide'
]

# Calculate SVI
print("\nCalculating SVI...")

# Function to calculate total SVI score using existing SVI theme columns
def calculate_svi(row):
    # List of SVI component columns and their directions
    svi_components = {
        'svi_socioeconomic (list distinct)': 1,       # Higher is worse (more socioeconomic vulnerability)
        'svi_household_comp (list distinct)': 1,       # Higher is worse (more household composition vulnerability)
        'svi_minority (list distinct)': 1,             # Higher is worse (more minority status & language vulnerability)
        'svi_housing_transportation (list distinct)': 1 # Higher is worse (more housing type & transportation vulnerability)
    }
    
    values = []
    for component, direction in svi_components.items():
        if pd.notna(row[component]):
            # Try to convert to float if it's stored as string
            try:
                value = float(row[component])
                values.append(value * direction)
            except (ValueError, TypeError):
                # Skip if not convertible to float
                pass
    
    # Return mean if we have values, NaN otherwise
    return np.mean(values) if values else np.nan

# Calculate total SVI score
df['svi_total'] = df.apply(calculate_svi, axis=1)

# Create SVI quartiles
# Handle cases where svi_total is NaN - these won't be assigned a quartile
df['svi_quartile'] = np.nan 
valid_svi_indices = df[df['svi_total'].notna()].index
df.loc[valid_svi_indices, 'svi_quartile'] = pd.qcut(
    df.loc[valid_svi_indices, 'svi_total'], 
    q=4, 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
df['svi_quartile'] = df['svi_quartile'].astype('category') # Keep it as category

# Create diagnosis groups
print("Creating diagnosis groups...")

# Check for JIA
jia_ht_cols = ['dx code (list distinct)', 'dx name (list distinct)', 'diagnosis of arthritis', 
               'ilar_code (list distinct)', 'ilar_code_display_value (list distinct)', 'ILAR code']  # Columns to check for JIA codes
jia_codes = ['M08.0', 'M08.2', 'M08.3', 'M08.4', 'M08.8', 'M08.9', 'M08', 'JIA', 'JRA', 'JUVENILE', 'ARTHRITIS']

def has_jia(row, cols=jia_ht_cols, codes=jia_codes):
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).upper()
            if any(code.upper() in value for code in codes):
                return True
    return False

# Check for Uveitis
def has_uveitis(row):
    # Using 'diagnosis of uveitis' column
    if 'diagnosis of uveitis' in row.index and pd.notna(row['diagnosis of uveitis']):
        value = str(row['diagnosis of uveitis']).upper()
        if 'UVEITIS' in value:
            return True
    # Using 'uveitis curr' column
    if 'uveitis curr' in row.index and pd.notna(row['uveitis curr']):
        value = str(row['uveitis curr']).upper()
        if value == 'YES' or value == '1' or value == 'TRUE':
            return True
    # Using 'uveitis curr fup' column
    if 'uveitis curr fup' in row.index and pd.notna(row['uveitis curr fup']):
        value = str(row['uveitis curr fup']).upper()
        if value == 'YES' or value == '1' or value == 'TRUE':
            return True
    return False

# Apply diagnosis group logic
df['has_jia'] = df.apply(has_jia, axis=1)
df['has_uveitis'] = df.apply(has_uveitis, axis=1)

# Create diagnosis group
df['diagnosis_group'] = 'Other'
df.loc[df['has_jia'] & ~df['has_uveitis'], 'diagnosis_group'] = 'JIA-Only'
df.loc[df['has_uveitis'], 'diagnosis_group'] = 'Uveitis'

# Function to parse dates from semicolon-separated list
def parse_date_list(date_string):
    if pd.isna(date_string) or date_string == '':
        return pd.NaT # Return Not a Time for missing input
    
    # Find all dates in the format YYYY-MM-DD
    full_dates = re.findall(r'\d{4}-\d{2}-\d{2}', str(date_string))
    
    # Find years (including those like 2019.0)
    # Match 4 digits possibly followed by .0, ensuring it's a standalone year
    years = re.findall(r'\b(\d{4})(?:\.0)?\b', str(date_string))
    
    parsed_dates = []
    
    # Process full dates first
    for date_str in full_dates:
        try:
            date = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(date):
                parsed_dates.append(date)
        except Exception as e:
            # print(f"Error parsing full date '{date_str}': {e}") # Optional: for debugging
            pass
    
    # If full dates were found, return the earliest one
    if parsed_dates:
        return min(parsed_dates)

    # If no full dates, process years
    valid_year_dates = []
    if years:
        for year in years:
            try:
                y_int = int(year)
                # Basic check for plausible year range
                if 1950 < y_int < 2050:
                    # Approximate as January 1st of that year
                    date = pd.to_datetime(f"{y_int}-01-01", errors='coerce')
                    if pd.notna(date):
                         valid_year_dates.append(date)
            except Exception as e:
                # print(f"Error processing year '{year}': {e}") # Optional: for debugging
                pass
                
    # Return the earliest valid year date if any found
    if valid_year_dates:
        return min(valid_year_dates)
        
    # If neither full dates nor valid years are found
    return pd.NaT

# Extract baseline dates
print("Extracting baseline dates...")

# --- Updated JIA and Uveitis date columns based on available columns ---
# Use the columns we found in the file
JIA_DATE_COLS = jia_cols_in_file
UVEITIS_DATE_COLS = uveitis_cols_in_file

# Helper function to combine date columns
def get_combined_date_string(row, cols):
    values = []
    for col in cols:
        # Check if column exists before trying to access it
        if col in row.index and pd.notna(row[col]):
            values.append(str(row[col]))
    return ';'.join(values) if values else ""

# Apply parsing using the available column lists
df['jia_baseline_date_str'] = df.apply(lambda row: get_combined_date_string(row, JIA_DATE_COLS), axis=1)
df['jia_baseline_date'] = df['jia_baseline_date_str'].apply(parse_date_list)

df['uveitis_baseline_date_str'] = df.apply(lambda row: get_combined_date_string(row, UVEITIS_DATE_COLS), axis=1)
df['uveitis_baseline_date'] = df['uveitis_baseline_date_str'].apply(parse_date_list)

# Choose appropriate baseline based on diagnosis group
df['baseline_date'] = df.apply(
    lambda row: row['uveitis_baseline_date'] if row['diagnosis_group'] == 'Uveitis' 
    else row['jia_baseline_date'] if row['diagnosis_group'] == 'JIA-Only'
    else pd.NaT, # Use NaT for Not a Time
    axis=1
)

# Function to find earliest systemic medication start date
def find_earliest_systemic_start(row):
    med_name_col = 'medication name (list distinct)'
    med_eye_col = 'cmeyetrt (list distinct)'
    med_date_col = 'medication start date (list distinct)'

    # Check if columns exist in the row
    has_med_name = med_name_col in row.index and pd.notna(row[med_name_col])
    has_med_eye = med_eye_col in row.index and pd.notna(row[med_eye_col])
    has_med_date = med_date_col in row.index and pd.notna(row[med_date_col])

    if not (has_med_name or has_med_eye) or not has_med_date:
        return pd.NaT
    
    # Extract medication lists and corresponding dates
    med_list1 = str(row[med_name_col]).split(';') if has_med_name else []
    med_list2 = str(row[med_eye_col]).split(';') if has_med_eye else []
    
    # Attempt to parse dates from the date column
    date_list_str = str(row[med_date_col]) if has_med_date else ""
    
    # We need a way to associate dates with meds if they are semicolon-separated
    # For now, let's parse *all* dates listed and find the minimum associated with *any* systemic med.
    
    all_potential_dates = []
    full_dates = re.findall(r'\d{4}-\d{2}-\d{2}', date_list_str)
    years = re.findall(r'\b\d{4}\b', date_list_str)

    for date_str in full_dates:
        try:
            date = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(date):
                all_potential_dates.append(date)
        except: pass
        
    if not all_potential_dates: # Only look at years if no full dates found
        for year in years:
            try:
                 y_int = int(year)
                 if 1950 < y_int < 2050: # Plausibility check
                    date = pd.to_datetime(f"{year}-01-01", errors='coerce')
                    if pd.notna(date):
                        all_potential_dates.append(date)
            except: pass
            
    if not all_potential_dates:
         return pd.NaT # No valid start dates found

    # Check if any listed med is systemic
    all_meds_combined = [item.strip().lower() for item in med_list1 + med_list2 if item.strip()]
    is_systemic_present = any(
        systemic_med.lower() in med 
        for med in all_meds_combined 
        for systemic_med in SYSTEMIC_MEDS
    )

    # If a systemic med is listed, return the earliest date found in the start date column
    if is_systemic_present:
        return min(all_potential_dates)
    else:
        return pd.NaT # No systemic med found

# Find earliest systemic medication start date
print("Finding earliest systemic medication start dates...")
df['earliest_systemic_start_date'] = df.apply(find_earliest_systemic_start, axis=1)

# Calculate delay in days
print("Calculating treatment delay...")
df['treatment_delay_days'] = df.apply(
    lambda row: (row['earliest_systemic_start_date'] - row['baseline_date']).days
    if pd.notna(row['earliest_systemic_start_date']) and pd.notna(row['baseline_date']) 
    and row['earliest_systemic_start_date'] >= row['baseline_date']
    else np.nan, # Keep as NaN if dates invalid or start date is before baseline
    axis=1
)

# Print percentage of records with exact dates vs year-only dates
print("\nDate precision analysis:")
# For JIA baseline dates
jia_dates = df['jia_baseline_date'].dropna()
jia_total = len(jia_dates)
jia_year_only = sum(1 for d in jia_dates if d.month == 1 and d.day == 1)
jia_exact = jia_total - jia_year_only
print(f"JIA baseline dates: {jia_total} total, {jia_exact} exact ({jia_exact/jia_total*100:.1f}% if total > 0), {jia_year_only} year-only ({jia_year_only/jia_total*100:.1f}% if total > 0)")

# For Uveitis baseline dates
uveitis_dates = df['uveitis_baseline_date'].dropna()
uveitis_total = len(uveitis_dates)
uveitis_year_only = sum(1 for d in uveitis_dates if d.month == 1 and d.day == 1)
uveitis_exact = uveitis_total - uveitis_year_only
print(f"Uveitis baseline dates: {uveitis_total} total, {uveitis_exact} exact ({uveitis_exact/uveitis_total*100:.1f}% if total > 0), {uveitis_year_only} year-only ({uveitis_year_only/uveitis_total*100:.1f}% if total > 0)")

# For medication start dates
med_dates = df['earliest_systemic_start_date'].dropna()
med_total = len(med_dates)
med_year_only = sum(1 for d in med_dates if d.month == 1 and d.day == 1)
med_exact = med_total - med_year_only
print(f"Medication start dates: {med_total} total, {med_exact} exact ({med_exact/med_total*100:.1f}% if total > 0), {med_year_only} year-only ({med_year_only/med_total*100:.1f}% if total > 0)")

# Filter to valid delays (positive or zero) and diagnosis groups of interest
valid_df = df[
    (df['diagnosis_group'].isin(['JIA-Only', 'Uveitis'])) & 
    (pd.notna(df['treatment_delay_days'])) &
    (df['treatment_delay_days'] >= 0)  # Only include non-negative values
].copy()

# Filter out rows with missing SVI values for statistical analysis and plotting SVI relationship
valid_with_svi_df = valid_df[pd.notna(valid_df['svi_total'])].copy()

# Print basic stats
print("\n--- SVI vs Treatment Delay Analysis ---")
print(f"Total patients with valid treatment delay data: {len(valid_df)}")
print(f"Patients with valid SVI data for analysis: {len(valid_with_svi_df)}")
print(f"JIA-Only (Total with delay): {len(valid_df[valid_df['diagnosis_group'] == 'JIA-Only'])}")
print(f"JIA-Only with SVI: {len(valid_with_svi_df[valid_with_svi_df['diagnosis_group'] == 'JIA-Only'])}")
print(f"Uveitis (Total with delay): {len(valid_df[valid_df['diagnosis_group'] == 'Uveitis'])}")
print(f"Uveitis with SVI: {len(valid_with_svi_df[valid_with_svi_df['diagnosis_group'] == 'Uveitis'])}")

# Convert delay days to weeks for better interpretability
valid_df['treatment_delay_weeks'] = valid_df['treatment_delay_days'] / 7
valid_with_svi_df['treatment_delay_weeks'] = valid_with_svi_df['treatment_delay_days'] / 7

# Summary statistics (including missing SVI quartiles - reflects full cohort with delay)
# Use original valid_df for this, but group by SVI Quartile (will show NaNs)
summary_stats_all = valid_df.groupby(['diagnosis_group', 'svi_quartile'], observed=False)['treatment_delay_weeks'].agg(['count', 'mean', 'median', 'std']).reset_index()
summary_stats_all = summary_stats_all.sort_values(['diagnosis_group', 'svi_quartile'])
print("\nSummary Statistics (Delay in Weeks - All with Valid Delay):")
print(summary_stats_all)

# Save summary statistics for all
summary_stats_all.to_csv(f"{output_dir}/treatment_delay_summary_stats_all.csv", index=False)

# Summary statistics (excluding missing SVI quartiles - for SVI analysis)
summary_stats_with_svi = valid_with_svi_df.groupby(['diagnosis_group', 'svi_quartile'], observed=False)['treatment_delay_weeks'].agg(['count', 'mean', 'median', 'std']).reset_index()
summary_stats_with_svi = summary_stats_with_svi.sort_values(['diagnosis_group', 'svi_quartile'])
print("\nSummary Statistics (Delay in Weeks - Only Patients with SVI):")
print(summary_stats_with_svi)

# Save filtered summary statistics
summary_stats_with_svi.to_csv(f"{output_dir}/treatment_delay_summary_stats_with_svi.csv", index=False)

# Perform Kruskal-Wallis test for each diagnosis group (only patients with SVI data)
print("\nKruskal-Wallis Test Results (Comparing Delay across SVI Quartiles):")
kruskal_results = {}
for group in ['JIA-Only', 'Uveitis']:
    group_data = valid_with_svi_df[valid_with_svi_df['diagnosis_group'] == group]
    if len(group_data['svi_quartile'].unique()) > 1:  # Ensure more than one quartile group
        quartiles_with_data = group_data['svi_quartile'].dropna().unique()
        if len(quartiles_with_data) > 1:
            try:
                # Extract data for each quartile present
                quartile_data = [
                    group_data[group_data['svi_quartile'] == q]['treatment_delay_weeks'].dropna()
                    for q in ['Q1', 'Q2', 'Q3', 'Q4'] # Use defined order
                    if q in quartiles_with_data and len(group_data[group_data['svi_quartile'] == q]) > 0
                ]
                
                # Only run test if we have data in multiple quartiles
                if len(quartile_data) > 1 and all(len(d) > 0 for d in quartile_data):
                    h_stat, p_val = stats.kruskal(*quartile_data)
                    print(f"{group}: H={h_stat:.2f}, p={p_val:.4f}")
                    kruskal_results[group] = p_val
                else:
                    print(f"{group}: Not enough data across multiple quartiles for Kruskal-Wallis test.")
                    kruskal_results[group] = np.nan
            except Exception as e:
                print(f"{group}: Error running Kruskal-Wallis test: {e}")
                kruskal_results[group] = np.nan
        else:
             print(f"{group}: Only one SVI quartile found with data.")
             kruskal_results[group] = np.nan
    else:
        print(f"{group}: Not enough data or SVI quartiles for statistical testing.")
        kruskal_results[group] = np.nan

# Test correlation between SVI total and treatment delay (only patients with SVI data)
print("\nSpearman Correlation Results (SVI Total vs Delay):")
correlation_results = {}
for group in ['JIA-Only', 'Uveitis']:
    group_data = valid_with_svi_df[valid_with_svi_df['diagnosis_group'] == group]
    # Ensure we have enough non-NaN pairs for correlation
    valid_corr_data = group_data[['svi_total', 'treatment_delay_weeks']].dropna()
    if len(valid_corr_data) >= 5:  # Need at least a few points for correlation
        try:
            corr, p_val = stats.spearmanr(
                valid_corr_data['svi_total'].values,
                valid_corr_data['treatment_delay_weeks'].values
            )
            print(f"{group}: rho={corr:.3f}, p={p_val:.4f}")
            correlation_results[group] = {'rho': corr, 'p_value': p_val}
        except Exception as e:
            print(f"{group}: Error running Spearman correlation: {e}")
            correlation_results[group] = {'rho': np.nan, 'p_value': np.nan}
    else:
        print(f"{group}: Not enough non-NaN data pairs for correlation testing.")
        correlation_results[group] = {'rho': np.nan, 'p_value': np.nan}

# Perform Mann-Whitney U tests to compare JIA-Only vs Uveitis for each SVI quartile
print("\nMann-Whitney U Test Results (Comparing JIA-Only vs Uveitis within each SVI Quartile):")
group_comparison_results = {}
for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    jia_data = valid_with_svi_df[(valid_with_svi_df['diagnosis_group'] == 'JIA-Only') & 
                                (valid_with_svi_df['svi_quartile'] == quartile)]['treatment_delay_weeks']
    uveitis_data = valid_with_svi_df[(valid_with_svi_df['diagnosis_group'] == 'Uveitis') & 
                                    (valid_with_svi_df['svi_quartile'] == quartile)]['treatment_delay_weeks']
    
    # Only perform test if we have data in both groups
    if len(jia_data) > 0 and len(uveitis_data) > 0:
        try:
            stat, p_val = stats.mannwhitneyu(jia_data, uveitis_data, alternative='two-sided')
            print(f"{quartile}: U={stat:.2f}, p={p_val:.4f}, JIA n={len(jia_data)}, Uveitis n={len(uveitis_data)}")
            group_comparison_results[quartile] = p_val
        except Exception as e:
            print(f"{quartile}: Error running Mann-Whitney U test: {e}")
            group_comparison_results[quartile] = np.nan
    else:
        print(f"{quartile}: Insufficient data for statistical comparison (JIA n={len(jia_data)}, Uveitis n={len(uveitis_data)})")
        group_comparison_results[quartile] = np.nan

# --- Create Visualizations ---
print("\nGenerating visualizations...")

# Plotting Style - Consistent with other scripts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'  # Set global font to Times New Roman
sns.set_context("paper", font_scale=1.2)

# Define color scheme
COLORS = {'JIA-Only': '#1f77b4', 'Uveitis': '#ff7f0e'}
QUARTILE_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']

# Plot 1: Boxplots for treatment delay by SVI quartile, separated by group
colors = {'JIA-Only': '#1f77b4', 'Uveitis': '#ff7f0e'}
quartile_names = ['Q1', 'Q2', 'Q3', 'Q4']

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True) # Adjusted figure size
fig.suptitle('Time from Diagnosis to First Systemic Medication by SVI Quartile', fontsize=16, y=0.98)

for i, group in enumerate(['JIA-Only', 'Uveitis']):
    ax = axes[i]
    group_data_svi = valid_with_svi_df[valid_with_svi_df['diagnosis_group'] == group]
    
    # Prepare data for boxplot, ensuring correct order
    plot_data = [group_data_svi[group_data_svi['svi_quartile'] == q]['treatment_delay_weeks'].dropna() for q in quartile_names]
    
    if not group_data_svi.empty:
        bp = ax.boxplot(
            plot_data, 
            labels=quartile_names, 
            patch_artist=True,
            widths=0.6,
            showmeans=False, # Hide default mean marker
            boxprops=dict(facecolor=colors[group], alpha=0.7, edgecolor='black'),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color='black', linestyle='-'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5, markeredgecolor='none', markersize=5) # Improved flier style
        )
        
        ax.set_title(group, fontsize=14)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        
        if i == 0:
            ax.set_ylabel('Time to Systemic Immunosuppression (Weeks)', fontsize=12)
        
        ax.set_xlabel('SVI Quartile', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7) # Keep grid lines subtle
        
        # Calculate y limits for text placement
        y_min, y_max = ax.get_ylim()
        # Adjust position based on axis range to avoid overlap
        text_y_pos_n = y_min - (y_max - y_min) * 0.05 
        text_y_pos_mean = y_max + (y_max - y_min) * 0.02 # Place mean slightly above boxes

        for j, q in enumerate(quartile_names):
            quartile_subset = group_data_svi[group_data_svi['svi_quartile'] == q]['treatment_delay_weeks'].dropna()
            count = len(quartile_subset)
            mean_val = quartile_subset.mean()
            
            # Add n count text
            ax.text(j + 1, text_y_pos_n, f'n={count}', ha='center', va='top', fontsize=10) # j+1 because boxplot positions are 1-based
            
            # Add mean value text only if mean is calculable
            if pd.notna(mean_val):
                # Add mean text above the box
                 ax.text(j + 1, mean_val + (y_max - y_min) * 0.03 , f'{mean_val:.1f}', 
                         ha='center', va='bottom', fontsize=9, color='darkred', fontweight='bold')
                 # Add a small marker for the mean
                 ax.plot(j + 1, mean_val, marker='D', color='darkred', markersize=4, alpha=0.8)

        # Add Mann-Whitney U p-value annotation
        p_val = group_comparison_results.get(q, np.nan)
        if pd.notna(p_val):
            ax.text(0.5, 0.85, f'Mann-Whitney U: p={p_val:.3f}', 
                    ha='center', va='top', transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='lightgrey'))
            
        # Auto-adjust y-limit based on IQR to handle outliers better
        all_delays = group_data_svi['treatment_delay_weeks'].dropna()
        if not all_delays.empty:
            q1 = all_delays.quantile(0.25)
            q3 = all_delays.quantile(0.75)
            iqr = q3 - q1
            # Consider maximum of data or upper whisker for plot limit, plus mean value position
            upper_bound = max(all_delays.max(), q3 + 1.5 * iqr) 
            current_means = [d.mean() for d in plot_data if not d.empty] 
            if current_means: # Check if means list is not empty
                 upper_bound = max(upper_bound, max(current_means) if current_means else upper_bound)

            ax.set_ylim(bottom=y_min, top=upper_bound * 1.15) # Add more padding for mean text
            
    else:
        ax.text(0.5, 0.5, "No data with SVI", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
        ax.set_title(group, fontsize=14)
        if i == 0: ax.set_ylabel('Time to Systemic Immunosuppression (Weeks)', fontsize=12)
        ax.set_xlabel('SVI Quartile', fontsize=12)
        # Use ax.set_xticks and ax.set_xticklabels for consistency
        ax.set_xticks(range(len(quartile_names))) # Positions 0, 1, 2, 3
        ax.set_xticklabels(quartile_names)


plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout slightly more for bottom text
plt.savefig(f"{output_dir}/treatment_delay_boxplot_by_group.png", dpi=300)
plt.close(fig) # Close the figure explicitly

# Plot 2: Bar chart for mean treatment delay by SVI quartile, separated by group
# Use the summary stats dataframe already calculated
plt.figure(figsize=(12, 7))

# Define order for plotting
svi_order = ['Q1', 'Q2', 'Q3', 'Q4']
hue_order = ['JIA-Only', 'Uveitis'] # Define hue order for consistency

# Filter summary_stats_with_svi for plotting
plot_df = summary_stats_with_svi[summary_stats_with_svi['svi_quartile'].isin(svi_order)].copy()

# Ensure svi_quartile is categorical with the correct order
plot_df['svi_quartile'] = pd.Categorical(plot_df['svi_quartile'], categories=svi_order, ordered=True)

# Create the bar plot
barplot = sns.barplot(
    data=plot_df,
    x='svi_quartile',
    y='mean',
    hue='diagnosis_group',
    hue_order=hue_order, # Ensure consistent hue order
    palette=colors,
    edgecolor='black',
    alpha=0.8
)

# Add error bars directly using the DataFrame
# This approach doesn't rely on mapping patch indices to DataFrame rows
for idx, row in plot_df.iterrows():
    quartile = row['svi_quartile']
    group = row['diagnosis_group']
    mean_val = row['mean']
    std_val = row['std'] if pd.notna(row['std']) else 0
    count = row['count']  # Get the count (N) from summary stats
    
    # Find the corresponding bar position
    x_pos = svi_order.index(quartile)
    hue_offset = hue_order.index(group) - (len(hue_order) - 1)/2
    width = 0.8 / len(hue_order)  # Standard width adjustment for hue groups
    
    # Calculate the center position for this specific bar
    center_x = x_pos + hue_offset * width
    
    # Calculate lower error bar, but don't let it go below 0
    # For treatment time, negative values don't make sense
    lower_err = min(mean_val, std_val) if not pd.isna(std_val) else 0
    upper_err = std_val if not pd.isna(std_val) else 0
    
    # Add the error bar with asymmetric error values to prevent negative extension
    plt.errorbar(
        x=center_x,
        y=mean_val,
        yerr=[[lower_err], [upper_err]],  # Asymmetric error bars: [[lower], [upper]]
        fmt='none',
        c='black',
        capsize=4
    )
    
    # Add count (N) below the bar
    plt.text(
        center_x, 
        -5,  # Position below x-axis
        f'N={count}',
        ha='center',
        va='top',
        fontsize=9,
        color='black'
    )
    
    # Add mean value on top of the bar
    plt.text(
        center_x,
        mean_val + upper_err + 3,  # Position above the error bar
        f'{mean_val:.1f}',
        ha='center',
        va='bottom',
        fontsize=9,
        color='darkred',
        fontweight='bold'
    )

# Add between-group p-values (JIA vs Uveitis) for each quartile
for quartile in svi_order:
    x_pos = svi_order.index(quartile)
    p_val = group_comparison_results.get(quartile, np.nan)
    
    if pd.notna(p_val):
        # Get the max height of bars and error bars for this quartile to position the text
        quartile_data = plot_df[plot_df['svi_quartile'] == quartile]
        
        if not quartile_data.empty:
            # Calculate the maximum height including error bars
            max_height = 0
            for _, row in quartile_data.iterrows():
                mean = row['mean']
                std = row['std'] if pd.notna(row['std']) else 0
                max_height = max(max_height, mean + std)
            
            # Add p-value text with bracket - position below the bars near x-axis
            plt.text(
                x_pos,  # Center the text above the quartile
                -15,  # Position below the bars, above the N counts
                f'p={p_val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='black',
                fontweight='bold'
            )
            
            # Add bracket connecting the two bars below the x-axis
            # Plot a line with a little bracket at each end
            bracket_width = 0.35  # Width of bracket
            y_pos = -10  # Position bracket below the bars, above the text
            
            # Horizontal line
            plt.plot([x_pos - bracket_width, x_pos + bracket_width], 
                     [y_pos, y_pos], 
                     'k-', linewidth=1)
            
            # Left vertical tick
            plt.plot([x_pos - bracket_width, x_pos - bracket_width], 
                     [y_pos, y_pos + 2], 
                     'k-', linewidth=1)
            
            # Right vertical tick
            plt.plot([x_pos + bracket_width, x_pos + bracket_width], 
                     [y_pos, y_pos + 2], 
                     'k-', linewidth=1)

# Set y-axis minimum to 0 
plt.ylim(bottom=0)  # Ensure y-axis starts at 0

plt.title('Mean Time to First Systemic Medication by SVI Quartile', fontsize=16)
plt.xlabel('SVI Quartile', fontsize=12)
plt.ylabel('Mean Time to Systemic Immunosuppression (Weeks)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(title='Diagnosis Group', fontsize=11, title_fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig(f"{output_dir}/treatment_delay_barchart_by_group.png", dpi=300)
plt.close() # Close the figure

# Plot 3: Scatter plot of SVI total vs treatment delay, separated by group (Renamed from Plot 2)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True) # Adjusted figsize
fig.suptitle('Relationship Between SVI Total and Time to First Systemic Medication', fontsize=16, y=0.98)

for i, group in enumerate(['JIA-Only', 'Uveitis']):
    ax = axes[i]
    group_data_svi = valid_with_svi_df[valid_with_svi_df['diagnosis_group'] == group]
    
    if not group_data_svi.empty:
        # Get data without NaN values needed for plotting and correlation
        plot_data = group_data_svi[['svi_total', 'treatment_delay_weeks']].dropna()

        if not plot_data.empty:
            ax.scatter(
                plot_data['svi_total'], 
                plot_data['treatment_delay_weeks'],
                c=colors[group], 
                alpha=0.6, 
                s=50, # Adjusted size
                edgecolors='w', # Add white edge for clarity
                linewidth=0.5
            )
            
            # Add trendline only if enough points
            if len(plot_data) >= 5:
                try:
                    x = plot_data['svi_total']
                    y = plot_data['treatment_delay_weeks']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Create smooth line
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = p(x_line)
                    
                    ax.plot(x_line, y_line, color=sns.desaturate(colors[group], 0.7), linestyle='--', linewidth=2)
                    
                    # Add correlation info from pre-calculated results
                    corr_info = correlation_results.get(group, np.nan)
                    if corr_info and pd.notna(corr_info['rho']):
                        p_val = corr_info['p_value']
                        ax.text(0.05, 0.90, f"Spearman Correlation: rho={corr_info['rho']:.3f}, p={p_val:.4f}", 
                                transform=ax.transAxes, fontsize=11,
                                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='lightgrey')) # Added background box
                except Exception as e:
                    print(f"Could not fit trend line for {group}: {e}")

        ax.set_title(group, fontsize=14)
        ax.set_xlabel('SVI Total (Higher = More Vulnerable)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Time to Systemic Immunosuppression (Weeks)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=11)
        
    else:
        ax.text(0.5, 0.5, "No data with SVI", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
        ax.set_title(group, fontsize=14)
        ax.set_xlabel('SVI Total (Higher = More Vulnerable)', fontsize=12)
        if i == 0: ax.set_ylabel('Time to Systemic Immunosuppression (Weeks)', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
plt.savefig(f"{output_dir}/treatment_delay_scatter_by_group.png", dpi=300)
plt.close(fig) # Close the figure

print(f"Analysis complete. Results saved to {output_dir}/") 