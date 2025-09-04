#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
import math
from collections import Counter
import warnings

# Ignore specific warnings if necessary
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- Configuration ---
INPUT_FILE = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged.csv"
OUTPUT_DIR = "svi_complications_analysis_lifetime"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 300

# --- Column Names ---
# SVI Columns
SVI_COLS = [
    'svi_socioeconomic (list distinct)',
    'svi_household_comp (list distinct)',
    'svi_housing_transportation (list distinct)',
    'svi_minority (list distinct)'
]
SVI_TOTAL_COL = 'SVI_total'
SVI_QUARTILE_COL = 'SVI_quartile'

# Diagnosis Column
DX_CODE_COL = 'dx code (list distinct)'
DIAGNOSIS_GROUP_COL = 'Diagnosis_Group'
JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']
GROUPS_TO_ANALYZE = ['JIA-Only', 'Any Uveitis']
DIAGNOSIS_GROUP_ORDER = ['JIA-Only', 'Any Uveitis']
QUARTILE_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']

# Complication Columns & Keywords (Based on old script + refinement)
COL_OSSURG = 'ossurg (list distinct)'
COL_OSSURGOTH = 'ossurgoth (list distinct)'
COL_PROC_NAME = 'procedure name (list distinct)'
COL_CMEYETRT = 'cmeyetrt (list distinct)'
COL_MED_NAME = 'medication name (list distinct)'
COL_DX_NAME = 'dx name (list distinct)'
# Potentially check COL_DX_CODE? 'dx code (list distinct)' (col 502, 503?) - Needs ICD codes
# Check 'comorbidities'? (col 494) - Free text?
# Check 'oscompoth (list distinct)' (col 89) - Other complication text

KW_CATARACT = ['cataract', 'phaco', 'lens', 'extract']
KW_GLAUCOMA_PROC = ['glaucoma', 'trabeculotomy', 'tube', 'shunt', 'valve', 'trabeculect', 'ITRACK']
KW_GLAUCOMA_MED = ['timolol', 'dorzolamide', 'brimonidine', 'latanoprost', 'travoprost', 'cosopt', 'azopt', 'alphagan', 'lumigan']
KW_SYNECHIAE = ['synechi'] # Broadened, includes 'synechiolysis'
KW_SURGERY_GENERAL = ['surgery', 'surgical', 'operation'] # Add general terms?

# Output Columns (removed _1_2yr suffix)
COMP_CATARACT = 'Cataract'
COMP_GLAUCOMA = 'Glaucoma'
COMP_SYNECHIAE = 'Synechiae'
COMP_SURGERY = 'Surgery'
COMP_TOTAL_COUNT = 'Total_Complications'
COMP_ANY = 'Any_Complication'

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# --- Helper Functions ---

def _deduplicate_columns(columns):
    # ... (same as before) ...
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
    # ... (same as before) ...
    if pd.isna(value): return np.nan
    try:
        str_value = str(value)
        values = [float(v.strip()) for v in str_value.split(';') if v.strip() and v.strip().lower() != 'nan']
        return np.mean(values) if values else np.nan
    except ValueError:
        return np.nan
    except Exception:
        return np.nan
        
def calculate_svi_total_and_quartiles(df, svi_cols, total_col, quartile_col):
    # ... (same as before, ensure it handles missing SVI columns gracefully) ...
    print("Calculating SVI...")
    component_mean_cols = []
    svi_cols_found = [col for col in svi_cols if col in df.columns]
    print(f"  Found SVI component columns: {svi_cols_found}")
    missing_svi_cols = set(svi_cols) - set(svi_cols_found)
    if missing_svi_cols:
        print(f"  Warning: Missing SVI columns: {missing_svi_cols}")

    for col in svi_cols_found:
        mean_col = f"{col}_mean"
        df[mean_col] = df[col].apply(parse_svi_values)
        component_mean_cols.append(mean_col)

    if not component_mean_cols:
        print("Error: No SVI component columns found or processed. Cannot calculate total SVI.")
        df[total_col] = np.nan
        df[quartile_col] = 'Unknown'
        return df

    df[total_col] = df[component_mean_cols].mean(axis=1, skipna=True)
    print(f"Calculated total SVI for {df[total_col].notna().sum()} patients.")

    df[quartile_col] = pd.Series(dtype=object)
    non_na_svi = df[total_col].dropna()
    if len(non_na_svi) >= 4:
        try:
            quartiles = pd.qcut(non_na_svi, 4, labels=QUARTILE_ORDER)
            df[quartile_col] = quartiles.reindex(df.index)
            print(f"Calculated SVI quartiles. Distribution:\n{df[quartile_col].value_counts().sort_index()}")
            df[quartile_col] = pd.Categorical(df[quartile_col], categories=QUARTILE_ORDER + ['Unknown'], ordered=False)
        except ValueError as e:
            print(f"Warning: Could not compute SVI quartiles: {e}. Assigning Unknown.")
            df[quartile_col] = pd.Categorical(pd.Series('Unknown', index=df.index), categories=QUARTILE_ORDER + ['Unknown'], ordered=False)
    else:
        print("Warning: Not enough non-NA SVI values to compute quartiles. Assigning Unknown.")
        df[quartile_col] = pd.Categorical(pd.Series('Unknown', index=df.index), categories=QUARTILE_ORDER + ['Unknown'], ordered=False)

    df[quartile_col] = df[quartile_col].fillna('Unknown')
    if df[quartile_col].eq('Unknown').all() and not df[total_col].isna().all():
         print("Warning: Failed to calculate SVI quartiles, but total SVI was calculated.")
    return df

def check_codes(dx_string, jia_pattern, uveitis_patterns):
    # ... (same as before) ...
    if pd.isna(dx_string): return 'Other'
    codes = [code.strip() for code in str(dx_string).split(';') if code.strip()]
    has_jia = any(re.search(jia_pattern, code, re.IGNORECASE) for code in codes)
    has_uveitis = any(any(re.search(uv_pattern, code, re.IGNORECASE) for code in codes) for uv_pattern in uveitis_patterns)
    
    if has_jia and not has_uveitis:
        return 'JIA-Only'
    elif has_uveitis:
        return 'Any Uveitis'
    elif has_jia:
         return 'JIA-Only'
    else:
        return 'Other'

# --- Complication Identification Helper ---
def contains_any_keyword(text_value, keywords):
    """Checks if any of the keywords are contained in the text_value."""
    if pd.isna(text_value): return False
    text = str(text_value).lower()
    return any(keyword.lower() in text for keyword in keywords)

# --- Main Execution ---
def main():
    print("--- SVI vs Complications (Lifetime) Analysis ---")

    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        # Load all potentially relevant columns first
        all_cols_in_file = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
        cols_to_load = list(set(
            SVI_COLS + 
            [DX_CODE_COL] +  
            [COL_OSSURG, COL_OSSURGOTH, COL_PROC_NAME, COL_CMEYETRT, COL_MED_NAME, COL_DX_NAME] + 
            ['oscompoth (list distinct)', 'comorbidities'] # Add cols for enhanced synechiae check
        ))
        cols_present = [col for col in cols_to_load if col in all_cols_in_file]
        missing_cols = set(cols_to_load) - set(cols_present)
        if missing_cols:
            print(f"Warning: The following requested columns are missing from the CSV and will be ignored: {missing_cols}")
            
        df = pd.read_csv(INPUT_FILE, usecols=cols_present, low_memory=False)
        print(f"Data loaded. Shape: {df.shape}")
        # Deduplication might still be needed if column names loaded have issues
        # df.columns = _deduplicate_columns(df.columns.tolist())

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. Calculate SVI
    df = calculate_svi_total_and_quartiles(df, SVI_COLS, SVI_TOTAL_COL, SVI_QUARTILE_COL)

    # 2. Determine Diagnosis Group
    if DX_CODE_COL not in df.columns:
        print(f"Error: Diagnosis code column '{DX_CODE_COL}' not found.")
        return
    print(f"\nDetermining diagnosis groups...")
    df[DIAGNOSIS_GROUP_COL] = df[DX_CODE_COL].apply(lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS))
    print(f"Diagnosis group distribution:\n{df[DIAGNOSIS_GROUP_COL].value_counts()}")

    # 3. Identify Complications (Lifetime Prevalence - No Time Window)
    print("\nIdentifying lifetime complications...")
    df[COMP_CATARACT] = 0
    df[COMP_GLAUCOMA] = 0
    df[COMP_SYNECHIAE] = 0
    df[COMP_SURGERY] = 0
    
    # Define column sets to check for each complication
    cataract_cols = [COL_OSSURG, COL_OSSURGOTH, COL_PROC_NAME, COL_DX_NAME]
    glaucoma_proc_cols = [COL_OSSURG, COL_PROC_NAME, COL_DX_NAME]
    glaucoma_med_cols = [COL_CMEYETRT, COL_MED_NAME]
    synechiae_cols = [COL_OSSURGOTH, COL_DX_NAME, 'oscompoth (list distinct)']
    surgery_cols = [COL_OSSURG, COL_OSSURGOTH, COL_PROC_NAME]
    
    # Apply checks for each complication type across all patients
    cols_present_in_df = [col for col in df.columns if col in df.columns]
    
    # Check for cataract complications
    cataract_cols_present = [col for col in cataract_cols if col in cols_present_in_df]
    if cataract_cols_present:
        print(f"Checking for Cataracts in columns: {cataract_cols_present}")
        df[COMP_CATARACT] = df[cataract_cols_present].apply(
            lambda row: any(contains_any_keyword(row[col], KW_CATARACT) 
                          for col in cataract_cols_present if col in row.index), 
            axis=1
        ).astype(int)
    
    # Check for glaucoma complications (procedures/surgery)
    glaucoma_proc_cols_present = [col for col in glaucoma_proc_cols if col in cols_present_in_df]
    has_glaucoma_proc = pd.Series(False, index=df.index)
    if glaucoma_proc_cols_present:
        print(f"Checking for Glaucoma (procedures) in columns: {glaucoma_proc_cols_present}")
        has_glaucoma_proc = df[glaucoma_proc_cols_present].apply(
            lambda row: any(contains_any_keyword(row[col], KW_GLAUCOMA_PROC) 
                          for col in glaucoma_proc_cols_present if col in row.index), 
            axis=1
        )
    
    # Check for glaucoma complications (medications)
    glaucoma_med_cols_present = [col for col in glaucoma_med_cols if col in cols_present_in_df]
    has_glaucoma_med = pd.Series(False, index=df.index)
    if glaucoma_med_cols_present:
        print(f"Checking for Glaucoma (medications) in columns: {glaucoma_med_cols_present}")
        has_glaucoma_med = df[glaucoma_med_cols_present].apply(
            lambda row: any(contains_any_keyword(row[col], KW_GLAUCOMA_MED) 
                          for col in glaucoma_med_cols_present if col in row.index), 
            axis=1
        )
    
    # Combine glaucoma flags
    df[COMP_GLAUCOMA] = (has_glaucoma_proc | has_glaucoma_med).astype(int)
    
    # Check for synechiae complications
    synechiae_cols_present = [col for col in synechiae_cols if col in cols_present_in_df]
    if synechiae_cols_present:
        print(f"Checking for Synechiae in columns: {synechiae_cols_present}")
        df[COMP_SYNECHIAE] = df[synechiae_cols_present].apply(
            lambda row: any(contains_any_keyword(row[col], KW_SYNECHIAE) 
                          for col in synechiae_cols_present if col in row.index), 
            axis=1
        ).astype(int)
    
    # Check for surgery complications
    surgery_cols_present = [col for col in surgery_cols if col in cols_present_in_df]
    if surgery_cols_present:
        print(f"Checking for Surgery in columns: {surgery_cols_present}")
        # Use broader surgery keywords: all specific complications plus general surgery terms
        all_surgery_kw = KW_CATARACT + KW_GLAUCOMA_PROC + KW_SYNECHIAE + ['vitrec', 'iridec'] + KW_SURGERY_GENERAL
        df[COMP_SURGERY] = df[surgery_cols_present].apply(
            lambda row: any(contains_any_keyword(row[col], all_surgery_kw) 
                          for col in surgery_cols_present if col in row.index), 
            axis=1
        ).astype(int)
    
    # Calculate total and any complication flags
    comp_cols = [COMP_CATARACT, COMP_GLAUCOMA, COMP_SYNECHIAE, COMP_SURGERY]
    df[COMP_TOTAL_COUNT] = df[comp_cols].sum(axis=1)
    df[COMP_ANY] = (df[COMP_TOTAL_COUNT] > 0).astype(int)

    print("\nComplication prevalence (lifetime):")
    for col in comp_cols + [COMP_ANY, COMP_TOTAL_COUNT]:
        if col == COMP_TOTAL_COUNT:
             print(f"- Mean {col}: {df[col].mean():.2f} (Std: {df[col].std():.2f})")
        else:
             count = df[col].sum()
             perc = (count / len(df) * 100)
             print(f"- {col}: {count} patients ({perc:.1f}% of {len(df)} total patients)")

    # 4. Prepare Data for Analysis
    analysis_df = df[
        (df[SVI_QUARTILE_COL] != 'Unknown') &
        df[DIAGNOSIS_GROUP_COL].isin(GROUPS_TO_ANALYZE)
    ].copy()

    print(f"\nPrepared analysis dataset with {len(analysis_df)} patients having SVI Quartile and Diagnosis Group.")
    if len(analysis_df) == 0:
        print("No patients remaining after filtering for analysis. Exiting.")
        return
        
    print(f"Distribution in analysis set:")
    print(f"- Diagnosis Group:\n{analysis_df[DIAGNOSIS_GROUP_COL].value_counts()}")
    print(f"- SVI Quartile:\n{analysis_df[SVI_QUARTILE_COL].value_counts().sort_index()}")
    print(f"- Any Complication (lifetime):\n{analysis_df[COMP_ANY].value_counts()}")
    print(f"- Total Complications (lifetime) Mean: {analysis_df[COMP_TOTAL_COUNT].mean():.2f}")
    
    jia_only_present = 'JIA-Only' in analysis_df[DIAGNOSIS_GROUP_COL].unique()
    uveitis_present = 'Any Uveitis' in analysis_df[DIAGNOSIS_GROUP_COL].unique()

    # 5. Analyze Complication Count (ANOVA/Kruskal-Wallis)
    print(f"\n--- Analyzing Mean Complication Count ({COMP_TOTAL_COUNT}) by SVI Quartile & Diagnosis Group ---")
    count_stats = analysis_df.groupby([DIAGNOSIS_GROUP_COL, SVI_QUARTILE_COL], observed=False)[COMP_TOTAL_COUNT].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).reset_index()
    print(count_stats)
    count_stats_path = os.path.join(OUTPUT_DIR, "svi_vs_comp_count_stats.csv")
    count_stats.to_csv(count_stats_path, index=False)
    print(f"Saved count stats to {count_stats_path}")
    
    # Run stats within each group
    for group in GROUPS_TO_ANALYZE:
        subset = analysis_df[analysis_df[DIAGNOSIS_GROUP_COL] == group]
        if len(subset) < 2 or len(subset[SVI_QUARTILE_COL].unique()) < 2:
             print(f"Skipping ANOVA/K-W test for {group} (insufficient data or groups)")
             continue
        groups_for_test = [subset[COMP_TOTAL_COUNT][subset[SVI_QUARTILE_COL] == q].dropna() for q in QUARTILE_ORDER]
        groups_for_test = [g for g in groups_for_test if len(g) > 0]
        if len(groups_for_test) < 2: 
             print(f"Skipping ANOVA/K-W test for {group} (less than 2 non-empty quartiles)")
             continue
             
        # Check normality assumption (optional, K-W is non-parametric)
        # Use Kruskal-Wallis as counts are likely not normally distributed
        try:
            h_stat, p_val = stats.kruskal(*groups_for_test)
            print(f"Kruskal-Wallis test for {group} ({COMP_TOTAL_COUNT} vs SVI Quartile): H={h_stat:.2f}, p={p_val:.4f}")
        except Exception as e:
            print(f"Could not perform Kruskal-Wallis test for {group}: {e}")

    # 6. Analyze Proportion with Any Complication (Chi-squared/Fisher)
    print(f"\n--- Analyzing Proportion with Any Complication ({COMP_ANY}) by SVI Quartile & Diagnosis Group ---")
    proportion_stats = analysis_df.groupby([DIAGNOSIS_GROUP_COL, SVI_QUARTILE_COL], observed=False)[COMP_ANY].agg(
        total_count='count',
        any_comp_count='sum' 
    ).reset_index()
    proportion_stats['proportion_any_comp'] = proportion_stats['any_comp_count'] / proportion_stats['total_count']
    proportion_stats['proportion_any_comp'] = proportion_stats['proportion_any_comp'].fillna(0)
    print(proportion_stats)
    prop_stats_path = os.path.join(OUTPUT_DIR, "svi_vs_any_comp_proportion_stats.csv")
    proportion_stats.to_csv(prop_stats_path, index=False)
    print(f"Saved proportion stats to {prop_stats_path}")

    # Perform comparison tests (Group vs Group within Quartile)
    print("\nPerforming comparison tests (JIA vs Uveitis within each Quartile)...")
    comparison_results = []
    if jia_only_present and uveitis_present:
        for quartile in QUARTILE_ORDER:
            subset = analysis_df[analysis_df[SVI_QUARTILE_COL] == quartile]
            if len(subset[DIAGNOSIS_GROUP_COL].unique()) == 2:
                contingency_table = pd.crosstab(subset[DIAGNOSIS_GROUP_COL], subset[COMP_ANY])
                if contingency_table.shape == (2, 2): # Expecting 2 groups, 2 outcomes (0 or 1)
                    use_fisher = (contingency_table.values < 5).any()
                    test_name = "Fisher's Exact" if use_fisher else "Chi-squared"
                    try:
                        if use_fisher:
                            oddsr, p = stats.fisher_exact(contingency_table)
                            stat_val = oddsr # Store odds ratio for Fisher
                        else:
                            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                            stat_val = chi2 # Store chi2 stat
                        print(f"  {quartile}: {test_name}, p={p:.4f}")
                        comparison_results.append({'Quartile': quartile, 'Test': test_name, 'Stat': stat_val, 'p_value': p})
                    except Exception as e:
                        print(f"  {quartile}: Could not perform {test_name} test: {e}")
                        comparison_results.append({'Quartile': quartile, 'Test': test_name, 'Stat': np.nan, 'p_value': np.nan})
                else:
                    print(f"  {quartile}: Skipping comparison - contingency table shape {contingency_table.shape} not 2x2.")
                    comparison_results.append({'Quartile': quartile, 'Test': 'Skipped', 'Stat': np.nan, 'p_value': np.nan})
            else:
                print(f"  {quartile}: Skipping comparison - only one diagnosis group present.")
                comparison_results.append({'Quartile': quartile, 'Test': 'Skipped', 'Stat': np.nan, 'p_value': np.nan})
    else:
        print("Skipping JIA vs Uveitis comparison tests - only one group present in analysis data.")
        for quartile in QUARTILE_ORDER:
             comparison_results.append({'Quartile': quartile, 'Test': 'Skipped', 'Stat': np.nan, 'p_value': np.nan})
             
    df_comp_results = pd.DataFrame(comparison_results)
    comp_results_path = os.path.join(OUTPUT_DIR, "svi_vs_any_comp_comparison_results.csv")
    df_comp_results.to_csv(comp_results_path, index=False)
    print(f"Saved comparison test results to {comp_results_path}")

    # 7. Visualization
    print("\nCreating plots...")
    
    # Plot 1: Mean Complication Count
    plt.figure(figsize=(12, 7))
    # Filter out 'Unknown' SVI category from the data being plotted
    plot_data_mean = count_stats[count_stats[SVI_QUARTILE_COL].isin(QUARTILE_ORDER)]
    ax1 = sns.barplot(data=plot_data_mean, 
                      x=SVI_QUARTILE_COL, y='mean', 
                      hue=DIAGNOSIS_GROUP_COL, hue_order=DIAGNOSIS_GROUP_ORDER,
                      palette=['skyblue', 'salmon'])
    plt.title('Mean Number of Complications (Lifetime) by SVI Quartile', fontsize=16, pad=20)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel(f'Mean {COMP_TOTAL_COUNT}', fontsize=12)
    # Add N counts and potentially mean values?
    for i, quartile in enumerate(QUARTILE_ORDER):
         jia_row = plot_data_mean[(plot_data_mean[DIAGNOSIS_GROUP_COL] == 'JIA-Only') & (plot_data_mean[SVI_QUARTILE_COL] == quartile)]
         uveitis_row = plot_data_mean[(plot_data_mean[DIAGNOSIS_GROUP_COL] == 'Any Uveitis') & (plot_data_mean[SVI_QUARTILE_COL] == quartile)]
         n_jia = jia_row['count'].iloc[0] if not jia_row.empty else 0
         n_uveitis = uveitis_row['count'].iloc[0] if not uveitis_row.empty else 0
         xtick_label = f"{quartile}\n(JIA:N={n_jia}, Uveitis:N={n_uveitis})"
         ax1.text(i, -0.08, xtick_label, ha='center', va='top', fontsize=9, transform=ax1.get_xaxis_transform())
    # ax1.set_xticklabels([]) # Keep the original labels Q1, Q2, etc.
    ax1.set_xlim(-0.5, len(QUARTILE_ORDER) - 0.5) # Set x-limits tightly
    plt.legend(title='Diagnosis Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plot1_path = os.path.join(OUTPUT_DIR, "svi_vs_mean_comp_count_plot.png")
    plt.savefig(plot1_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Mean Count Plot to {plot1_path}")

    # Plot 2: Proportion with Any Complication
    plt.figure(figsize=(12, 7))
    # Filter out 'Unknown' SVI category from the data being plotted
    plot_data_prop = proportion_stats[proportion_stats[SVI_QUARTILE_COL].isin(QUARTILE_ORDER)]
    ax2 = sns.barplot(data=plot_data_prop, 
                      x=SVI_QUARTILE_COL, y='proportion_any_comp', 
                      hue=DIAGNOSIS_GROUP_COL, hue_order=DIAGNOSIS_GROUP_ORDER,
                      palette=['skyblue', 'salmon'])
    plt.title('Proportion with Any Complication (Lifetime) by SVI Quartile', fontsize=16, pad=20)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel(f'Proportion with {COMP_ANY}', fontsize=12)
    plt.ylim(0, max(1.0, plot_data_prop['proportion_any_comp'].max() * 1.15 + 0.1)) 
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
    # Add proportion percentages on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='{:.0%}', label_type='edge', padding=2, fontsize=9)
        
    # Add N counts and comparison p-values
    y_max_prop = plot_data_prop['proportion_any_comp'].max() if not plot_data_prop.empty else 0
    text_p_val_y_prop = ax2.get_ylim()[1] * 1.02 # Position p-values slightly above the plot area
    for i, quartile in enumerate(QUARTILE_ORDER):
        jia_row = plot_data_prop[(plot_data_prop[DIAGNOSIS_GROUP_COL] == 'JIA-Only') & (plot_data_prop[SVI_QUARTILE_COL] == quartile)]
        uveitis_row = plot_data_prop[(plot_data_prop[DIAGNOSIS_GROUP_COL] == 'Any Uveitis') & (plot_data_prop[SVI_QUARTILE_COL] == quartile)]
        n_jia = jia_row['total_count'].iloc[0] if not jia_row.empty else 0
        n_uveitis = uveitis_row['total_count'].iloc[0] if not uveitis_row.empty else 0
        xtick_label = f"{quartile}\n(JIA:N={n_jia}, Uveitis:N={n_uveitis})"
        ax2.text(i, -0.08, xtick_label, ha='center', va='top', fontsize=9, transform=ax2.get_xaxis_transform())
        
        # Add comparison p-value
        p_val_row = df_comp_results[df_comp_results['Quartile'] == quartile]
        p_val = p_val_row['p_value'].iloc[0] if not p_val_row.empty else np.nan
        test_type = p_val_row['Test'].iloc[0] if not p_val_row.empty else "Skipped"
        if test_type != 'Skipped' and pd.notna(p_val):
            p_text = f"p={p_val:.3f}"
            if p_val < 0.001: p_text = "p<0.001"
            elif p_val < 0.05: p_text += "*"
            ax2.text(i, text_p_val_y_prop, p_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            p_display_text = "NA (1 group)" if (n_jia == 0 or n_uveitis == 0) and (n_jia > 0 or n_uveitis > 0) else "p=NA"
            ax2.text(i, text_p_val_y_prop, p_display_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    # ax2.set_xticklabels([]) # Keep the original labels Q1, Q2, etc.
    ax2.set_xlim(-0.5, len(QUARTILE_ORDER) - 0.5) # Set x-limits tightly
    plt.legend(title='Diagnosis Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
    plot2_path = os.path.join(OUTPUT_DIR, "svi_vs_any_comp_proportion_plot.png")
    plt.savefig(plot2_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved Proportion Plot to {plot2_path}")

    # 8. Create stacked bar chart of all complications by SVI quartile for both groups
    print("\nCreating stacked bar chart of all complications...")
    
    # First, calculate the proportion for each complication by diagnosis group and SVI quartile
    stacked_data = []
    for comp_col in comp_cols:
        comp_data = analysis_df.groupby([DIAGNOSIS_GROUP_COL, SVI_QUARTILE_COL], observed=False)[comp_col].agg(
            total_count='count',
            comp_count='sum'
        ).reset_index()
        comp_data['proportion'] = comp_data['comp_count'] / comp_data['total_count']
        comp_data['Complication'] = comp_col
        stacked_data.append(comp_data)
    
    stacked_df = pd.concat(stacked_data, ignore_index=True)
    
    # Create separate plots for Each Diagnosis Group (Grouped Bar + N Counts)
    print("\nCreating grouped bar charts with N counts for each group...")
    for diagnosis_group in DIAGNOSIS_GROUP_ORDER:
        group_data = stacked_df[stacked_df[DIAGNOSIS_GROUP_COL] == diagnosis_group]
        
        plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        
        # Create pivot table for easier plotting
        pivot_data = group_data.pivot(index=SVI_QUARTILE_COL, columns='Complication', values='proportion')
        comp_counts_pivot = group_data.pivot(index=SVI_QUARTILE_COL, columns='Complication', values='comp_count')
        
        # Ensure all quartiles are present and sort
        for q in QUARTILE_ORDER:
            if q not in pivot_data.index:
                pivot_data.loc[q] = 0
                comp_counts_pivot.loc[q] = 0
        pivot_data = pivot_data.loc[QUARTILE_ORDER]
        comp_counts_pivot = comp_counts_pivot.loc[QUARTILE_ORDER].fillna(0).astype(int)
        
        # Define colors
        colors = {COMP_CATARACT: 'orange', COMP_GLAUCOMA: 'green', COMP_SYNECHIAE: 'red', COMP_SURGERY: 'purple'}
        complication_order = [c for c in [COMP_CATARACT, COMP_GLAUCOMA, COMP_SYNECHIAE, COMP_SURGERY] if c in pivot_data.columns]
        pivot_data = pivot_data[complication_order]
        comp_counts_pivot = comp_counts_pivot[complication_order]
        
        # Plot grouped bars
        pivot_data.plot(kind='bar', stacked=False, ax=ax, color=[colors.get(col, 'grey') for col in complication_order], 
                        width=0.8, figsize=(14, 8))
        
        # Add N counts above each bar
        for i, comp in enumerate(complication_order):
            container = ax.containers[i]
            labels = [f"N={int(n)}" for n in comp_counts_pivot[comp]]
            ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=9, rotation=0)
            
        # Formatting
        plt.title(f'{diagnosis_group}: Complications Prevalence by SVI Quartile', fontsize=16, pad=20)
        plt.xlabel('SVI Quartile', fontsize=14)
        plt.ylabel('Prevalence (%)', fontsize=14)
        plt.xticks(range(len(QUARTILE_ORDER)), QUARTILE_ORDER, rotation=0)
        plt.ylim(0, max(1.05, pivot_data.max().max() * 1.2)) # Adjust ylim based on data
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add total N counts below x-axis
        total_counts_per_quartile = group_data.groupby(SVI_QUARTILE_COL)['total_count'].first().reindex(QUARTILE_ORDER, fill_value=0)
        for i, quartile in enumerate(QUARTILE_ORDER):
            n_count = total_counts_per_quartile.loc[quartile]
            plt.text(i, -0.07, f"Total N={n_count}", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)
        
        # Add p-values (calculated earlier) to the legend
        comp_p_values = {}
        try:
            # Calculate p-values for each complication within this group
            for comp_col in complication_order:
                subset = analysis_df[analysis_df[DIAGNOSIS_GROUP_COL] == diagnosis_group]
                if len(subset) < 5 or len(subset[SVI_QUARTILE_COL].unique()) < 2:
                    comp_p_values[comp_col] = None
                    continue
                contingency_table = pd.crosstab(subset[SVI_QUARTILE_COL], subset[comp_col])
                if contingency_table.shape[0] <= 1 or contingency_table.shape[1] <=1: # Need at least 2 rows and 2 columns for test
                     comp_p_values[comp_col] = None
                     continue
                if contingency_table.shape[0] > 2 or contingency_table.shape[1] > 2:
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                elif (contingency_table.values < 5).any():
                    _, p_val = stats.fisher_exact(contingency_table)
                else:
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
                comp_p_values[comp_col] = p_val
        except Exception as e:
             print(f"P-value calculation error for {diagnosis_group}: {e}")

        legend_labels = []
        for comp_col in complication_order:
            label = comp_col
            p_val = comp_p_values.get(comp_col)
            if p_val is not None:
                if p_val < 0.001: label += " (p<0.001)"
                elif p_val < 0.05: label += f" (p={p_val:.3f}*)"
                else: label += f" (p={p_val:.3f})"
            legend_labels.append(label)
        
        # Use existing handles from the plot for the legend
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, loc='upper right', fontsize=10, title="Complication (p-value vs Quartile)")
        
        plt.tight_layout()
        grouped_plot_path = os.path.join(OUTPUT_DIR, f"grouped_complications_{diagnosis_group.lower().replace('-', '_').replace(' ', '_')}.png")
        plt.savefig(grouped_plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped complications plot for {diagnosis_group} to {grouped_plot_path}")

    # Modify the combined plot (~line 595 onwards)
    # Change to Grouped Bar Chart with distinct group colors
    print("\nCreating combined grouped bar chart with distinct group colors...")
    
    # Prepare data in a pivot format suitable for grouped bar chart
    combined_pivot = stacked_df.pivot_table(index=SVI_QUARTILE_COL, 
                                              columns=[DIAGNOSIS_GROUP_COL, 'Complication'], 
                                              values='proportion')
    combined_counts_pivot = stacked_df.pivot_table(index=SVI_QUARTILE_COL, 
                                                  columns=[DIAGNOSIS_GROUP_COL, 'Complication'], 
                                                  values='comp_count')

    # Ensure all quartiles and desired columns are present
    complication_order = [c for c in [COMP_CATARACT, COMP_GLAUCOMA, COMP_SYNECHIAE, COMP_SURGERY] if c in stacked_df['Complication'].unique()]
    multi_index = pd.MultiIndex.from_product([DIAGNOSIS_GROUP_ORDER, complication_order], names=[DIAGNOSIS_GROUP_COL, 'Complication'])
    combined_pivot = combined_pivot.reindex(index=QUARTILE_ORDER, columns=multi_index, fill_value=0)
    combined_counts_pivot = combined_counts_pivot.reindex(index=QUARTILE_ORDER, columns=multi_index, fill_value=0).astype(int)
    
    # Define distinct color shades for groups
    # Standard/Darker for Any Uveitis, Lighter for JIA-Only
    uveitis_colors = {COMP_CATARACT: '#FFA500', COMP_GLAUCOMA: '#228B22', COMP_SYNECHIAE: '#DC143C', COMP_SURGERY: '#8A2BE2'} # Orange, ForestGreen, Crimson, BlueViolet
    jia_colors = {COMP_CATARACT: '#FFDAB9', COMP_GLAUCOMA: '#90EE90', COMP_SYNECHIAE: '#FFA07A', COMP_SURGERY: '#E6E6FA'} # PeachPuff, LightGreen, LightSalmon, Lavender

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 10))
    n_groups = len(DIAGNOSIS_GROUP_ORDER)
    n_complications = len(complication_order)
    n_quartiles = len(QUARTILE_ORDER)
    total_bar_group_width = 0.8 # Total width for all bars in a quartile
    single_bar_width = total_bar_group_width / (n_groups * n_complications)
    # Adjust spacing if needed, maybe increase total width slightly if bars are too thin
    
    indices = np.arange(n_quartiles)

    for i, comp in enumerate(complication_order):
        for j, group in enumerate(DIAGNOSIS_GROUP_ORDER):
            # Calculate position for this specific bar (Quartile -> Group -> Complication)
            # Group bars by complication first, then by diagnosis group within the complication
            group_offset_within_quartile = (i - n_complications / 2 + 0.5) * (n_groups * single_bar_width) 
            bar_offset_within_complication = (j - n_groups / 2 + 0.5) * single_bar_width
            
            positions = indices + group_offset_within_quartile + bar_offset_within_complication

            proportions = combined_pivot[(group, comp)]
            counts = combined_counts_pivot[(group, comp)]
            
            # Select color based on group
            color = jia_colors.get(comp, 'lightgrey') if group == 'JIA-Only' else uveitis_colors.get(comp, 'grey')
            
            label = f"{group} - {comp}" # Label needed for the legend later
            
            bars = ax.bar(positions, proportions, single_bar_width, 
                        label=label, color=color)
            
            # Add N counts
            labels = [f"N={n}" if n > 0 else "" for n in counts] # Don't label N=0
            ax.bar_label(bars, labels=labels, label_type='edge', padding=2, fontsize=8, rotation=90)
            
    # Formatting
    ax.set_title('Complications Prevalence by SVI Quartile and Diagnosis Group', fontsize=18, pad=20)
    ax.set_xlabel('SVI Quartile', fontsize=16)
    ax.set_ylabel('Prevalence (%)', fontsize=16)
    ax.set_xticks(indices)
    ax.set_xticklabels(QUARTILE_ORDER, fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    # Dynamically set ylim based on max value + label space needed
    max_h = combined_pivot.max().max()
    ax.set_ylim(0, max(1.05, max_h * 1.25)) # Increase top margin slightly for N labels
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add total N counts below x-axis
    for i, quartile in enumerate(QUARTILE_ORDER):
        # Ensure we access counts correctly, handling potential missing rows
        jia_row = proportion_stats[(proportion_stats[DIAGNOSIS_GROUP_COL] == 'JIA-Only') & (proportion_stats[SVI_QUARTILE_COL] == quartile)]
        uveitis_row = proportion_stats[(proportion_stats[DIAGNOSIS_GROUP_COL] == 'Any Uveitis') & (proportion_stats[SVI_QUARTILE_COL] == quartile)]
        jia_total_n = jia_row['total_count'].iloc[0] if not jia_row.empty else 0
        uveitis_total_n = uveitis_row['total_count'].iloc[0] if not uveitis_row.empty else 0
        ax.text(i, -0.07, f"JIA: N={jia_total_n}\nUveitis: N={uveitis_total_n}", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    # Add complication colors (using the darker/standard shade)
    for comp in complication_order:
        legend_elements.append(Patch(facecolor=uveitis_colors.get(comp, 'grey'), label=comp))
    # Add group indicators (using simple color patches)
    legend_elements.append(Patch(facecolor=jia_colors[COMP_CATARACT], label='JIA-Only (Lighter Shade)')) # Use one example color
    legend_elements.append(Patch(facecolor=uveitis_colors[COMP_CATARACT], label='Any Uveitis (Darker Shade)'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, title="Complication / Group Shade")
    
    plt.tight_layout()
    combined_grouped_plot_path = os.path.join(OUTPUT_DIR, "combined_grouped_complications_by_group.png")
    plt.savefig(combined_grouped_plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved combined grouped complications plot to {combined_grouped_plot_path}")

    print("\n--- Complications Analysis Complete ---")

if __name__ == "__main__":
    main() 