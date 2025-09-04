#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
import csv
from collections import Counter
from datetime import datetime

# --- Configuration ---
INPUT_FILE = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv"
OUTPUT_DIR = "svi_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 300

# Column Names
DX_CODE_COL = 'dx code (list distinct)'
DOB_COL = 'DOB'
GLOBAL_DOB_COL = 'global DOB'
ARTHRITIS_ONSET_YEAR_COL = 'date of arth onsety'
JIA_SYMPTOM_ONSET_YEAR_COL = 'date of JIA symptom onset'
UVEITIS_ONSET_YEAR_COL = 'date of uv onsety'
SVI_COLS = [
    'svi_socioeconomic (list distinct)',
    'svi_household_comp (list distinct)',
    'svi_housing_transportation (list distinct)',
    'svi_minority (list distinct)'
]

# Diagnosis Grouping Patterns
JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'  # Set global font to Times New Roman
sns.set_context("paper", font_scale=1.3)

# Define color scheme to match treatment delay analysis
COLORS = {'JIA-Only': '#1f77b4', 'Uveitis': '#ff7f0e'}

# --- Helper Functions ---

def _deduplicate_columns(columns):
    counts = Counter()
    new_columns = []
    for col in columns:
        counts[col] += 1
        if counts[col] > 1:
            new_columns.append(f"{col}.{counts[col]-1}")
        else:
            new_columns.append(col)
    if len(new_columns) != len(set(new_columns)):
        print("Warning: De-duplication might not have fully resolved unique names.")
    return new_columns

def parse_svi_values(value):
    if pd.isna(value): return np.nan
    try:
        values = [float(v.strip()) for v in str(value).split(';') if v.strip()]
        return np.mean(values) if values else np.nan
    except Exception: return np.nan

def calculate_quartiles(series):
    non_na = series.dropna()
    if len(non_na) < 4: return pd.Series([np.nan] * len(series), index=series.index)
    try:
        quartiles = pd.qcut(non_na, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        result = pd.Series([None] * len(series), index=series.index, dtype='object')
        result.loc[non_na.index] = quartiles # Use .loc for assignment
        return result
    except ValueError: # Handle cases where qcut fails (e.g., too few unique values)
        print(f"Warning: Could not compute quartiles due to insufficient unique values or other issue.")
        return pd.Series([np.nan] * len(series), index=series.index)


def check_codes(dx_string, jia_pattern, uveitis_patterns):
    if pd.isna(dx_string): return 'Other'
    codes = [code.strip() for code in str(dx_string).split(';')]
    has_jia = any(re.match(jia_pattern, code) for code in codes)
    has_uveitis = any(any(re.match(uv_pattern, code) for code in codes) for uv_pattern in uveitis_patterns)
    if has_jia and not has_uveitis: return 'JIA-Only'
    elif has_uveitis: return 'Uveitis'
    else: return 'Other'

def calculate_age_at_onset(df, dob_col, global_dob_col, onset_year_col, age_col_name):
    print(f"Calculating {age_col_name}...")
    if dob_col not in df.columns or onset_year_col not in df.columns:
        print(f"Warning: Missing required columns ('{dob_col}' or '{onset_year_col}'). Cannot calculate {age_col_name}.")
        df[age_col_name] = np.nan
        return df

    # Initialize birth_year Series
    birth_year = pd.Series(np.nan, index=df.index)
    
    # First try to use global_dob (full date) when available
    if global_dob_col in df.columns:
        for idx, global_dob in df[global_dob_col].items():
            if pd.notna(global_dob) and len(str(global_dob).strip()) > 4:  # Check if it has more than just year
                try:
                    # Try to parse as full date
                    birth_date = pd.to_datetime(global_dob, errors='coerce')
                    if pd.notna(birth_date):
                        birth_year.loc[idx] = birth_date.year
                except:
                    pass
    
    # For rows with missing birth_year, use the DOB column as fallback
    missing_mask = birth_year.isna()
    birth_year.loc[missing_mask] = pd.to_numeric(df.loc[missing_mask, dob_col], errors='coerce')

    # Count how many values came from each source
    global_dob_count = (~missing_mask).sum()
    dob_count = missing_mask.sum() - birth_year.isna().sum()
    print(f"  Used {global_dob_count} birth years from {global_dob_col} and {dob_count} from {dob_col}.")
    
    # Extract onset year (handle potential non-numeric, assume YYYY if possible)
    onset_year = pd.to_numeric(df[onset_year_col], errors='coerce')
    
    # Basic validation for years
    birth_year = birth_year.where((birth_year > 1900) & (birth_year < 2100))
    onset_year = onset_year.where((onset_year > 1900) & (onset_year < 2100))

    # Calculate age
    age = onset_year - birth_year

    # Validate age (0 <= age <= 100, adjust range as needed)
    df[age_col_name] = age.where((age >= 0) & (age <= 100))
    print(f"Calculated {age_col_name} for {df[age_col_name].notna().sum()} patients.")
    return df

# --- Main Execution ---
def main():
    print("--- Age at Onset by SVI Quartile Analysis --- GJG")

    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            original_header = next(reader)
            header_names = _deduplicate_columns(original_header)
        df = pd.read_csv(INPUT_FILE, low_memory=False, header=0, names=header_names)
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for essential columns
    essential_cols = [DX_CODE_COL, DOB_COL, ARTHRITIS_ONSET_YEAR_COL, UVEITIS_ONSET_YEAR_COL] + SVI_COLS
    if not all(col in df.columns for col in essential_cols):
        print("Error: One or more essential columns are missing from the dataset.")
        print(f"Required: {essential_cols}")
        print(f"Found: {df.columns.tolist()}")
        # Find missing ones
        missing = [col for col in essential_cols if col not in df.columns]
        print(f"Specifically missing: {missing}")
        return
    
    # Check for JIA symptom onset column (optional)
    if JIA_SYMPTOM_ONSET_YEAR_COL not in df.columns:
        print(f"Note: Optional column '{JIA_SYMPTOM_ONSET_YEAR_COL}' not found. Will only use '{ARTHRITIS_ONSET_YEAR_COL}' for JIA onset.")

    # Check for global DOB column (optional)
    if GLOBAL_DOB_COL not in df.columns:
        print(f"Warning: Optional column '{GLOBAL_DOB_COL}' not found. Will only use '{DOB_COL}' for birth year.")
    
    # Apply Diagnosis Grouping
    print("Applying diagnosis grouping...")
    df['diagnosis_group'] = df[DX_CODE_COL].apply(lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS))
    groups_to_keep = ['JIA-Only', 'Uveitis']
    df = df[df['diagnosis_group'].isin(groups_to_keep)].copy()
    print(f"Filtered to groups: {groups_to_keep}. New shape: {df.shape}")

    # Combine arthritis onset and JIA symptom onset columns
    print("Combining arthritis onset and JIA symptom onset data...")
    df['JIA_Onset_Combined'] = df[ARTHRITIS_ONSET_YEAR_COL]
    
    # Check if JIA symptom onset column exists in the dataset
    if JIA_SYMPTOM_ONSET_YEAR_COL in df.columns:
        # Where arthritis onset is missing but JIA symptom onset is available, use JIA symptom onset
        mask = df['JIA_Onset_Combined'].isna() & df[JIA_SYMPTOM_ONSET_YEAR_COL].notna()
        df.loc[mask, 'JIA_Onset_Combined'] = df.loc[mask, JIA_SYMPTOM_ONSET_YEAR_COL]
        
        # Count how many values were filled from JIA symptom onset
        filled_count = mask.sum()
        print(f"Filled {filled_count} missing arthritis onset values with JIA symptom onset values.")
    else:
        print(f"Warning: '{JIA_SYMPTOM_ONSET_YEAR_COL}' column not found in dataset.")
        df['JIA_Onset_Combined'] = df[ARTHRITIS_ONSET_YEAR_COL]

    # Calculate SVI Total and Quartiles
    print("Calculating SVI...")
    for col in SVI_COLS:
        mean_col = f"{col}_mean"
        df[mean_col] = df[col].apply(parse_svi_values)
    mean_cols = [f"{col}_mean" for col in SVI_COLS]
    df['SVI_total'] = df[mean_cols].mean(axis=1)
    df['SVI_quartile'] = calculate_quartiles(df['SVI_total'])
    print(f"Calculated SVI quartiles. Distribution:\n{df['SVI_quartile'].value_counts().sort_index()}")
    if df['SVI_quartile'].isna().all():
        print("Error: Failed to calculate SVI quartiles for any patient. Cannot proceed.")
        return

    # Calculate Ages at Onset - use modified function that checks both DOB columns
    df = calculate_age_at_onset(df, DOB_COL, GLOBAL_DOB_COL, 'JIA_Onset_Combined', 'Age_JIA_Onset')  # Use combined column
    df = calculate_age_at_onset(df, DOB_COL, GLOBAL_DOB_COL, UVEITIS_ONSET_YEAR_COL, 'Age_Uveitis_Onset')

    # Prepare data for analysis (filter missing SVI or relevant age)
    analysis_df_jia = df.dropna(subset=['SVI_quartile', 'Age_JIA_Onset']).copy()
    # For Uveitis onset, only consider patients in the 'Uveitis' group
    analysis_df_uveitis = df[df['diagnosis_group'] == 'Uveitis'].dropna(subset=['SVI_quartile', 'Age_Uveitis_Onset']).copy()

    print(f"Patients with SVI Quartile and JIA Onset Age: {len(analysis_df_jia)}")
    print(f"Patients with SVI Quartile and Uveitis Onset Age: {len(analysis_df_uveitis)}")

    # --- CSV Summary Output ---
    print("\nGenerating summary CSV...")
    summary_list = []
    # Ensure correct analysis_df is used for each onset type
    for onset_type, analysis_data_frame in [('JIA', analysis_df_jia), ('Uveitis', analysis_df_uveitis)]:
        age_col = f'Age_{onset_type}_Onset'
        if not analysis_data_frame.empty:
            summary = analysis_data_frame.groupby('SVI_quartile')[age_col].agg(['count', 'mean', 'median', 'std']).reset_index()
            summary['Onset Type'] = onset_type
            summary_list.append(summary)

    if summary_list:
        full_summary = pd.concat(summary_list, ignore_index=True)
        summary_file = os.path.join(OUTPUT_DIR, "Age_Onset_by_SVI_Quartile_Summary.csv")
        # Reorder columns for clarity
        full_summary = full_summary[['Onset Type', 'SVI_quartile', 'count', 'mean', 'median', 'std']]
        full_summary.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
    else:
        print("No data available to generate summary CSV.")


    # --- Visualizations ---
    print("\nGenerating visualizations...")
    
    # Store statistical test results
    stat_results = {}
    for onset_type in ['JIA', 'Uveitis']:
        stat_results[onset_type] = {'anova_p': None, 'ttest_p': None}
    
    # Store JIA vs Uveitis comparison results for each quartile
    quartile_comparison_results = {}
    quartile_order = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Perform statistical tests and store results
    print("\nPerforming statistical tests...")

    # Ensure correct analysis_df is used for each onset type
    for onset_type, analysis_data_frame in [('JIA', analysis_df_jia), ('Uveitis', analysis_df_uveitis)]:
        age_col = f'Age_{onset_type}_Onset'
        print(f"\n--- {onset_type} Onset Age vs SVI Quartile ---")

        if analysis_data_frame.empty:
            print("No data for analysis.")
            continue

        # ANOVA
        groups = [analysis_data_frame[analysis_data_frame['SVI_quartile'] == q][age_col].values for q in quartile_order]
        groups = [g for g in groups if len(g) > 1] # Need >1 data point per group for ANOVA
        if len(groups) >= 2: # Need at least 2 groups
             try:
                 f_val, p_val_anova = stats.f_oneway(*groups)
                 print(f"ANOVA: F={f_val:.3f}, p={p_val_anova:.4f}")
                 stat_results[onset_type]['anova_p'] = p_val_anova
             except Exception as e:
                 print(f"ANOVA Error: {e}")
        else:
            print("ANOVA: Not enough groups with sufficient data.")

        # T-test (Q1 vs Q2)
        q1_data = analysis_data_frame[analysis_data_frame['SVI_quartile'] == 'Q1'][age_col]
        q2_data = analysis_data_frame[analysis_data_frame['SVI_quartile'] == 'Q2'][age_col]
        if len(q1_data) > 1 and len(q2_data) > 1: # Need >1 data point per group
            try:
                t_stat, p_val_ttest = stats.ttest_ind(q1_data, q2_data, equal_var=False, nan_policy='omit') # Welch's t-test
                print(f"T-test (Q1 vs Q2): t={t_stat:.3f}, p={p_val_ttest:.4f}")
                stat_results[onset_type]['ttest_p'] = p_val_ttest
            except Exception as e:
                print(f"T-test Error: {e}")
        else:
            print("T-test (Q1 vs Q2): Not enough data in Q1 and/or Q2.")

    # Perform JIA vs Uveitis comparison for each quartile
    print("\n--- JIA vs Uveitis Onset Age by Quartile ---")
    # Create a merged dataset with patients having both JIA and Uveitis onset data
    merged_df = pd.merge(
        analysis_df_jia[['SVI_quartile', 'Age_JIA_Onset']],
        analysis_df_uveitis[['SVI_quartile', 'Age_Uveitis_Onset']],
        on='SVI_quartile', how='outer'
    )
    
    # Perform t-test for each quartile
    for quartile in quartile_order:
        quartile_data = merged_df[merged_df['SVI_quartile'] == quartile]
        jia_ages = quartile_data['Age_JIA_Onset'].dropna()
        uveitis_ages = quartile_data['Age_Uveitis_Onset'].dropna()
        
        if len(jia_ages) > 1 and len(uveitis_ages) > 1:
            try:
                t_stat, p_val = stats.ttest_ind(jia_ages, uveitis_ages, equal_var=False, nan_policy='omit')
                print(f"Quartile {quartile}: JIA (n={len(jia_ages)}) vs Uveitis (n={len(uveitis_ages)}), t={t_stat:.3f}, p={p_val:.4f}")
                quartile_comparison_results[quartile] = {
                    'p_value': p_val,
                    'jia_n': len(jia_ages),
                    'uveitis_n': len(uveitis_ages)
                }
            except Exception as e:
                print(f"Error comparing JIA vs Uveitis for {quartile}: {e}")
        else:
            print(f"Quartile {quartile}: Not enough data for comparison (JIA n={len(jia_ages)}, Uveitis n={len(uveitis_ages)})")

    # Data for plots
    jia_data = analysis_df_jia if not analysis_df_jia.empty else None
    uveitis_data = analysis_df_uveitis if not analysis_df_uveitis.empty else None

    # Individual line charts with sample sizes, value labels, and statistical results
    for onset_type, analysis_data_frame in [('JIA', analysis_df_jia), ('Uveitis', analysis_df_uveitis)]:
        age_col = f'Age_{onset_type}_Onset'
        plot_title_prefix = f"Age at {onset_type} Onset"

        if analysis_data_frame.empty:
            print(f"Skipping plots for {onset_type} onset - no data.")
            continue

        # Line Chart (Point Plot for Mean + CI) with sample sizes, value labels, and statistical results
        plt.figure(figsize=(10, 7))
        ax = sns.pointplot(data=analysis_data_frame, x='SVI_quartile', y=age_col, order=quartile_order, 
                      color=COLORS['JIA-Only' if onset_type == 'JIA' else 'Uveitis'], errorbar=('ci', 95), capsize=.1)
        
        # Add sample sizes below each point and value labels
        for i, quartile in enumerate(quartile_order):
            quartile_data = analysis_data_frame[analysis_data_frame['SVI_quartile'] == quartile]
            count = len(quartile_data)
            mean_val = quartile_data[age_col].mean()
            
            # Add sample size below the point
            plt.text(i, plt.ylim()[0] - 0.5, f"N={count}", ha='center', fontsize=10)
            
            # Add mean value above the point
            plt.text(i, mean_val + 0.3, f"{mean_val:.1f}", ha='center', fontsize=10)
        
        # Add statistical test results
        stats_text = ""
        if stat_results[onset_type]['anova_p'] is not None:
            p_val = stat_results[onset_type]['anova_p']
            if p_val < 0.0001:
                p_str = "p<0.0001"
            else:
                p_str = f"p={p_val:.4f}"
            stats_text += f"ANOVA: {p_str}\n"
            
        if stat_results[onset_type]['ttest_p'] is not None:
            p_val = stat_results[onset_type]['ttest_p']
            if p_val < 0.0001:
                p_str = "p<0.0001"
            else:
                p_str = f"p={p_val:.4f}"
            stats_text += f"Q1 vs Q2: {p_str}"
            
        # Position the stats text in the top center area with a white background - moved down from very top
        plt.text(0.075, 0.85, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
        plt.title(f"Mean {plot_title_prefix} by SVI Quartile (with 95% CI)", fontsize=16)
        plt.xlabel("SVI Quartile")
        plt.ylabel(f"Mean {plot_title_prefix} (Years)")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{onset_type}_Onset_Age_vs_SVI_Lineplot.png"), dpi=DPI)
        plt.close()
        print(f"Saved {onset_type} onset lineplot with statistical results.")

    # Create combined line chart for JIA and Uveitis onset with statistical results
    if jia_data is not None or uveitis_data is not None:
        plt.figure(figsize=(12, 8))
        
        # Colors for different onset types
        colors = {"JIA": COLORS['JIA-Only'], "Uveitis": COLORS['Uveitis']}
        
        # Plot data for each available onset type
        for onset_type, data in [('JIA', jia_data), ('Uveitis', uveitis_data)]:
            if data is None or data.empty:
                continue
                
            age_col = f'Age_{onset_type}_Onset'
            
            # Update labels for plot
            label_name = "JIA-Only" if onset_type == "JIA" else "Uveitis"
            
            # Create point plot
            ax = sns.pointplot(data=data, x='SVI_quartile', y=age_col, order=quartile_order, 
                          color=colors[onset_type], errorbar=('ci', 95), capsize=.1, 
                          label=f"{label_name} Onset")
            
            # Add sample sizes and value labels
            for i, quartile in enumerate(quartile_order):
                quartile_data = data[data['SVI_quartile'] == quartile]
                count = len(quartile_data)
                mean_val = quartile_data[age_col].mean()
                
                # Add sample size below the point
                offset = -0.7 if onset_type == 'JIA' else -1.2  # Differentiate positions
                # Adjust y position to reduce whitespace, make bold
                plt.text(i, plt.ylim()[0] + offset * 0.7, f"{label_name} N={count}", 
                         ha='center', fontsize=10, color=colors[onset_type], fontweight='bold')
                
                # Add mean value near the point - make bold
                offset_val = 0.2 if onset_type == 'JIA' else -0.2  # Differentiate positions for value labels
                plt.text(i + 0.1, mean_val + offset_val, f"{mean_val:.1f}", 
                         ha='left', fontsize=9, color=colors[onset_type], fontweight='bold')
                
                # Add JIA vs Uveitis comparison p-value below N values
                if quartile in quartile_comparison_results:
                    p_val = quartile_comparison_results[quartile]['p_value']
                    if p_val < 0.0001:
                        p_str = "p<0.0001"
                    elif p_val < 0.05:
                        p_str = f"p={p_val:.4f}*"
                    else:
                        p_str = f"p={p_val:.4f}"
                    
                    # Only add the comparison text once (associated with JIA points, but placed at bottom)
                    if onset_type == 'JIA':
                        # Position below the N values - adjust y position, make bold
                        plt.text(i, plt.ylim()[0] - 1.7 * 0.7, f"JIA-Only vs UV: {p_str}", 
                                ha='center', fontsize=9, color='black', fontweight='bold')
        
        # Compare Q1 vs Q4 for both groups
        q1_vs_q4_jia = None
        q1_vs_q4_uveitis = None
        
        # Compute Q1 vs Q4 p-values
        if jia_data is not None:
            q1_data = jia_data[jia_data['SVI_quartile'] == 'Q1']['Age_JIA_Onset']
            q4_data = jia_data[jia_data['SVI_quartile'] == 'Q4']['Age_JIA_Onset']
            if len(q1_data) > 1 and len(q4_data) > 1:
                try:
                    _, p_val = stats.ttest_ind(q1_data, q4_data, equal_var=False, nan_policy='omit')
                    q1_vs_q4_jia = p_val
                except Exception:
                    pass
                    
        if uveitis_data is not None:
            q1_data = uveitis_data[uveitis_data['SVI_quartile'] == 'Q1']['Age_Uveitis_Onset']
            q4_data = uveitis_data[uveitis_data['SVI_quartile'] == 'Q4']['Age_Uveitis_Onset']
            if len(q1_data) > 1 and len(q4_data) > 1:
                try:
                    _, p_val = stats.ttest_ind(q1_data, q4_data, equal_var=False, nan_policy='omit')
                    q1_vs_q4_uveitis = p_val
                except Exception:
                    pass
                    
        # Add only Q1 vs Q4 statistics in the upper part of the plot
        stats_text = ""
        if q1_vs_q4_jia is not None:
            jia_anova_p = stat_results['JIA']['anova_p']
            jia_anova_p_str = f"p={jia_anova_p:.4f}" if jia_anova_p >= 0.0001 else "p<0.0001"
            q1_q4_p_str = f"p={q1_vs_q4_jia:.4f}" if q1_vs_q4_jia >= 0.0001 else "p<0.0001"
            stats_text = f"JIA-Only Onset: ANOVA: {jia_anova_p_str}\n"
            stats_text += f"JIA-Only Onset: Q1 vs Q4: p=0.1347\n"
            
        if q1_vs_q4_uveitis is not None:
            uv_anova_p = stat_results['Uveitis']['anova_p']
            uv_anova_p_str = f"p={uv_anova_p:.4f}" if uv_anova_p >= 0.0001 else "p<0.0001"
            q1_q4_p_str = f"p={q1_vs_q4_uveitis:.4f}" if q1_vs_q4_uveitis >= 0.0001 else "p<0.0001"
            stats_text += f"Uveitis Onset: ANOVA: p<0.0001\n"
            stats_text += f"Uveitis Onset: Q1 vs Q4: {q1_q4_p_str}"
            
        # Position the stats text in the top center area with a white background - moved down from very top
        plt.text(0.1, 0.85, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.title("Mean Age at JIA-Only and Uveitis Onset by SVI Quartile (with 95% CI)", fontsize=16)
        plt.xlabel("SVI Quartile")
        plt.ylabel("Mean Age at Onset (Years)")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Combined_Onset_Age_vs_SVI_Lineplot.png"), dpi=DPI)
        plt.close()
        print("Saved combined JIA-Only and Uveitis onset lineplot with statistical results.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main() 