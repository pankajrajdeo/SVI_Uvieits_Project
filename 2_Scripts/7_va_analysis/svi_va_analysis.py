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
INPUT_FILE = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv"
OUTPUT_DIR = "svi_va_analysis"
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

# Visual Acuity Columns to Investigate
VA_COLS_TO_LOAD = [
    'exam left vadcc (list distinct)',
    'exam right vadcc (list distinct)',
    'exam left vadsc (list distinct)',
    'exam right vadsc (list distinct)',
    # Adding other potentially relevant columns from user list for deeper check
    'exam left vadcco (list distinct)',
    'exam right vadcco (list distinct)',
    'exam left vadsco (list distinct)',
    'exam right vadsco (list distinct)',
    'exam left vancc (list distinct)',
    'exam right vancc (list distinct)',
    'exam left vansc (list distinct)',
    'exam right vansc (list distinct)',
    'exam left vadnr (list distinct)',
    'exam right vadnr (list distinct)',
    'exam left vadnro (list distinct)',
    'exam right vadnro (list distinct)'
    # Ignoring 'x' (modifier) and 'test'/'testo' columns for now
]

# Define specific column names for primary analysis types
VA_L_VADCC = 'exam left vadcc (list distinct)'
VA_R_VADCC = 'exam right vadcc (list distinct)'
VA_L_VADSC = 'exam left vadsc (list distinct)'
VA_R_VADSC = 'exam right vadsc (list distinct)'

# LogMAR Conversion Values
LOGMAR_EQUIV = {
    'NLP': 3.0, 
    'LP': 2.7,  
    'HM': 2.3,  
    'CF': 1.9   
}

# Derived Column Names
WORST_LOGMAR_VADCC_COL = 'Worst_Overall_LogMAR_VADCC'
WORST_LOGMAR_VADSC_COL = 'Worst_Overall_LogMAR_VADSC'
WORST_LOGMAR_COMBINED_DIST_COL = 'Worst_Overall_LogMAR_Combined_Distance'

# Analysis Outcome Columns
WORSE_VA_THRESHOLD_LOGMAR = 0.4 # Corresponds to 20/50
WORSE_VA_BINARY_COL = 'Worse_VA_20_50_Combined_Distance' # Based on combined distance VA

# Grouping Columns
DIAGNOSIS_GROUP_COL = 'Diagnosis_Group'
JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']
GROUPS_TO_ANALYZE = ['JIA-Only', 'Uveitis']
DIAGNOSIS_GROUP_ORDER = ['JIA-Only', 'Uveitis']
QUARTILE_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
# Define color scheme to match treatment delay analysis
# COLORS = {'JIA-Only': '#1f77b4', 'Uveitis': '#ff7f0e'}  # Removed custom color dictionary

# --- Helper Functions (Copied and slightly adapted) ---

def _deduplicate_columns(columns):
    """Handles duplicate column names by appending .N"""
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
    """Parses semi-colon separated SVI values and calculates the mean."""
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
    """Calculates SVI component means, total SVI, and quartiles."""
    print("Calculating SVI...")
    component_mean_cols = []
    svi_cols_found = [col for col in svi_cols if col in df.columns]
    print(f"  Found SVI component columns: {svi_cols_found}")

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
    if len(non_na_svi) >= 4:  # Need at least 4 values for quartiles
        try:
            # Use pd.qcut to create quartiles
            quartiles = pd.qcut(non_na_svi, 4, labels=QUARTILE_ORDER)
            
            # Create result series (initially all None)
            result = pd.Series([None] * len(df), index=df.index, dtype='object')
            
            # Assign quartile values to those with non-NA SVI
            for idx, q in zip(non_na_svi.index, quartiles):
                result[idx] = q
            
            df[quartile_col] = result
            
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
    """Classifies patients into 'JIA-Only', 'Uveitis', or 'Other'."""
    if pd.isna(dx_string): return 'Other'
    codes = [code.strip() for code in str(dx_string).split(';') if code.strip()]
    has_jia = any(re.search(jia_pattern, code, re.IGNORECASE) for code in codes)
    has_uveitis = any(any(re.search(uv_pattern, code, re.IGNORECASE) for code in codes) for uv_pattern in uveitis_patterns)
    
    if has_jia and not has_uveitis:
        return 'JIA-Only'
    elif has_uveitis:
        return 'Uveitis'
    elif has_jia:
         return 'JIA-Only' # If only JIA is present, classify as JIA-Only
    else:
        return 'Other'

# --- Visual Acuity Processing Functions (Same core logic) ---

def parse_single_va(va_record):
    """Converts a single VA record (e.g., '20/40', 'CF', 'HM', 'LP') to LogMAR.
       Returns LogMAR value or np.nan if invalid.
    """
    if pd.isna(va_record): return np.nan
    va_str = str(va_record).strip().upper()

    # Handle non-Snellen first
    if va_str == 'NLP': return LOGMAR_EQUIV['NLP']
    if va_str == 'LP': return LOGMAR_EQUIV['LP']
    if va_str == 'HM': return LOGMAR_EQUIV['HM']
    if 'CF' in va_str: return LOGMAR_EQUIV['CF'] # Catches "CF", "CF 1FT" etc.

    # Handle Snellen (e.g., "20/100", "20/20-1", "6/6")
    # Remove extra annotations like PH, -1, +2
    va_str = re.sub(r'\s*PH.*|\[+-].*|\(.*\)', '', va_str).strip()
    match = re.match(r'^(\d+)/(\d+)$', va_str)
    if match:
        try:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if numerator > 0 and denominator > 0:
                # Ensure minimum LogMAR is 0 (20/20 or better)
                # Allow for values like 20/10, 20/15
                logmar = math.log10(denominator / numerator)
                # Set floor for better than 20/20 (e.g. 20/15 -> -0.12, 20/10 -> -0.3). Clamp at 0 for simplicity?
                # Let's keep negative logmar for now to represent better than 20/20
                # Clamp values worse than NLP
                return min(logmar, LOGMAR_EQUIV['NLP'])
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.nan
            
    # Handle other notations if needed (e.g. decimals?) - currently ignored
    # Example: "0.5" could be decimal VA, convert if required: log10(1/0.5) = 0.3
    try:
        dec_va = float(va_str)
        if 0 < dec_va <= 2.0: # Assuming decimal VA range
             logmar = math.log10(1.0 / dec_va)
             return min(logmar, LOGMAR_EQUIV['NLP'])
    except ValueError:
        pass

    # If nothing matches, return NaN
    return np.nan

def va_string_to_logmar_list(va_string):
    """Parses a semicolon-separated VA string and returns a list of valid LogMAR values.
    """
    if pd.isna(va_string): return []
    records = str(va_string).split(';')
    logmar_values = []
    for record in records:
        logmar = parse_single_va(record)
        if pd.notna(logmar):
            logmar_values.append(logmar)
    return logmar_values

def get_worst_logmar(logmar_list):
    """Returns the maximum (worst) LogMAR value from a list, or np.nan if empty.
    """
    if not logmar_list: # Check if list is empty
        return np.nan
    try:
        # Ensure all elements are numeric before calling max
        numeric_logmars = [val for val in logmar_list if isinstance(val, (int, float)) and pd.notna(val)]
        if not numeric_logmars:
             return np.nan
        return max(numeric_logmars)
    except Exception:
        return np.nan # Catch any unexpected errors during max()

# --- New Function to calculate worst overall logmar from specific columns ---
def calculate_worst_overall_logmar(df, left_va_col, right_va_col, output_col_name):
    """Calculates the worst overall LogMAR from specified left/right VA columns."""
    print(f"\nCalculating worst overall LogMAR using: {left_va_col} & {right_va_col}")
    
    # Check if columns exist
    if left_va_col not in df.columns or right_va_col not in df.columns:
        print(f"  Error: Input columns {left_va_col} or {right_va_col} not found.")
        df[output_col_name] = np.nan
        return df
        
    left_logmar_list_col = f"{left_va_col}_LogMAR_List"
    right_logmar_list_col = f"{right_va_col}_LogMAR_List"
    worst_left_col = f"{left_va_col}_Worst_LogMAR"
    worst_right_col = f"{right_va_col}_Worst_LogMAR"
    
    df[left_logmar_list_col] = df[left_va_col].apply(va_string_to_logmar_list)
    df[right_logmar_list_col] = df[right_va_col].apply(va_string_to_logmar_list)

    df[worst_left_col] = df[left_logmar_list_col].apply(get_worst_logmar)
    df[worst_right_col] = df[right_logmar_list_col].apply(get_worst_logmar)

    # Calculate overall worst LogMAR 
    df[output_col_name] = df[[worst_left_col, worst_right_col]].max(axis=1, skipna=False) 
    df[output_col_name] = df[output_col_name].fillna(df[worst_left_col]).fillna(df[worst_right_col])
    
    valid_count = df[output_col_name].notna().sum()
    print(f"  Found worst overall LogMAR for {valid_count} patients into column '{output_col_name}'.")
    if valid_count > 0:
        print(f"  Distribution:\n{df[output_col_name].describe()}")
    return df

# --- New Function for LogMAR Analysis per Group ---
def run_logmar_analysis_for_group(df_subset, group_name, logmar_col, svi_quartile_col, output_dir):
    """Performs ANOVA and generates box plots for LogMAR for a specific diagnosis group."""
    print(f"\n--- Analyzing Continuous LogMAR for Group: {group_name} ({len(df_subset)} patients) ---")
    output_prefix = os.path.join(output_dir, group_name.replace(' ', '_'))

    # --- Descriptive Statistics by Quartile --- 
    print(f"Analyzing {logmar_col} by SVI quartile...")
    svi_logmar_stats = df_subset.groupby(svi_quartile_col, observed=False)[logmar_col].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    svi_logmar_stats = svi_logmar_stats.set_index(svi_quartile_col).reindex(QUARTILE_ORDER).reset_index()
    svi_logmar_stats['count'] = svi_logmar_stats['count'].fillna(0).astype(int)
    # Keep other stats as NaN if group is missing
    print(svi_logmar_stats)
    stats_file = f"{output_prefix}_logmar_by_svi_stats.csv"
    svi_logmar_stats.to_csv(stats_file, index=False)
    print(f"Saved LogMAR stats to {stats_file}")

    # --- Statistical Testing: ANOVA --- 
    print("\nPerforming statistical tests (ANOVA)...")
    present_quartiles = [q for q in QUARTILE_ORDER if q in df_subset[svi_quartile_col].unique()]
    groups_for_anova = [df_subset[logmar_col][df_subset[svi_quartile_col] == q] for q in present_quartiles]
    groups_for_anova = [g.dropna() for g in groups_for_anova if len(g.dropna()) >= 2] # Need at least 2 data points per group

    anova_f_stat, anova_p_value = np.nan, np.nan
    anova_p_text = "ANOVA: Insufficient data"
    if len(groups_for_anova) >= 2: # Need at least 2 groups for ANOVA
        try:
            anova_f_stat, anova_p_value = stats.f_oneway(*groups_for_anova)
            p_str = f"p={anova_p_value:.4f}"
            if anova_p_value < 0.001: p_str = "p<0.001"
            elif anova_p_value < 0.05: p_str += "*"
            anova_p_text = f"ANOVA (Across Quartiles): F={anova_f_stat:.2f}, {p_str}"
            print(f"ANOVA Result for {group_name}: {anova_p_text}")
        except Exception as e:
            print(f"Could not perform ANOVA for {group_name}: {e}")
            anova_p_text = "ANOVA: Error"
    else:
        print(f"ANOVA Skipped for {group_name}: Not enough groups with sufficient data.")
    
    # --- Statistical Testing: Q1 vs Q4 --- 
    print("\nPerforming Mann-Whitney U test (Q1 vs Q4)...")
    q1_vs_q4_text = "Q1 vs Q4: Insufficient data"
    if 'Q1' in present_quartiles and 'Q4' in present_quartiles:
        q1_data = df_subset[df_subset[svi_quartile_col] == 'Q1'][logmar_col].dropna()
        q4_data = df_subset[df_subset[svi_quartile_col] == 'Q4'][logmar_col].dropna()
        
        if len(q1_data) >= 2 and len(q4_data) >= 2:  # Need at least 2 data points per group
            try:
                u_stat, p_val = stats.mannwhitneyu(q1_data, q4_data, alternative='two-sided')
                p_str = f"p={p_val:.4f}"
                if p_val < 0.001: p_str = "p<0.001"
                elif p_val < 0.05: p_str += "*"
                q1_vs_q4_text = f"Q1 vs Q4: U={u_stat:.2f}, {p_str}"
                print(f"Mann-Whitney U Result for {group_name}, Q1 vs Q4: {q1_vs_q4_text}")
            except Exception as e:
                print(f"Could not perform Mann-Whitney U test for {group_name}, Q1 vs Q4: {e}")
                q1_vs_q4_text = "Q1 vs Q4: Error"
        else:
            print(f"Mann-Whitney U Skipped for {group_name}, Q1 vs Q4: Not enough data points")
    else:
        print(f"Mann-Whitney U Skipped for {group_name}, Q1 vs Q4: One or both quartiles missing")
        
    # --- Create Plot Labels with N counts and Mean values --- 
    plot_labels = {}
    for q in present_quartiles:
        count = svi_logmar_stats.loc[svi_logmar_stats[svi_quartile_col] == q, 'count'].iloc[0]
        mean_val = svi_logmar_stats.loc[svi_logmar_stats[svi_quartile_col] == q, 'mean'].iloc[0]
        mean_str = f"{mean_val:.2f}" if pd.notna(mean_val) else "NaN"
        plot_labels[q] = f"{q}\n(N={count})\nMean={mean_str}" 
    ordered_plot_labels = [plot_labels.get(q, f"{q}\n(N=0)\nMean=NaN") for q in present_quartiles]

    # --- Visualization: Box Plot --- 
    if not present_quartiles:
        print("No SVI quartiles present in this group for plotting LogMAR.")
        return
        
    plt.figure(figsize=(10, 7))
    ax_box = plt.gca()
    # Use original viridis palette instead of custom colors
    sns.boxplot(x=svi_quartile_col, y=logmar_col, data=df_subset, 
                order=present_quartiles, 
                palette='viridis',  # Reverted to original palette
                ax=ax_box, 
                showmeans=True, meanprops={"marker":"^",
                                          "markerfacecolor":"white", 
                                          "markeredgecolor":"black",
                                          "markersize":"8"})
    # Add mean value text near the mean marker
    for i, q in enumerate(present_quartiles):
        mean_val = svi_logmar_stats.loc[svi_logmar_stats[svi_quartile_col] == q, 'mean'].iloc[0]
        if pd.notna(mean_val):
            ax_box.text(i, mean_val + 0.02 * ax_box.get_ylim()[1], f'{mean_val:.2f}', 
                        horizontalalignment='center', size='small', color='black', weight='semibold')

    ax_box.set_title(f'Worst Overall LogMAR by SVI Quartile ({group_name})', fontsize=14)
    ax_box.set_xlabel('SVI Quartile', fontsize=12)
    ax_box.set_ylabel('Worst Overall LogMAR (Higher=Worse)', fontsize=12)
    ax_box.tick_params(axis='x', labelsize=11)
    ax_box.tick_params(axis='y', labelsize=11)
    ax_box.set_xticks(range(len(present_quartiles)))
    ax_box.set_xticklabels(ordered_plot_labels, fontsize=9)
    
    # Combine statistical test results
    stats_text = f"{anova_p_text}\n{q1_vs_q4_text}"
    ax_box.text(0.05, 0.95, stats_text, transform=ax_box.transAxes, fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, ec='lightgrey'))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plot_file = f"{output_prefix}_logmar_by_svi_boxplot.png"
    plt.savefig(plot_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved LogMAR boxplot to {plot_file}")

# --- Main Execution --- 
def main():
    print("--- SVI vs Visual Acuity Analysis (Extended Check + LogMAR Analysis) ---")

    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        all_cols = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
        # Ensure essential columns are loaded
        required_cols = VA_COLS_TO_LOAD + SVI_COLS + [DX_CODE_COL]
        cols_to_load_present = [col for col in required_cols if col in all_cols]
        cols_to_load_present = list(set(cols_to_load_present))
        
        print(f"Attempting to load columns: {cols_to_load_present}")
        df = pd.read_csv(INPUT_FILE, usecols=cols_to_load_present, low_memory=False)
        
        original_header = df.columns.tolist()
        header_names = _deduplicate_columns(original_header)
        df.columns = header_names
        print(f"Data loaded. Shape: {df.shape}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except ValueError as e:
         if "is not in list" in str(e):
             # Identify missing columns
             missing_cols = set(required_cols) - set(all_cols)
             print(f"Error: One or more required columns not found in CSV: {missing_cols}")
         else:
              print(f"Error loading specific columns: {e}")
         return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. Calculate SVI
    df = calculate_svi_total_and_quartiles(df, SVI_COLS, SVI_TOTAL_COL, SVI_QUARTILE_COL)

    # 2. Determine Diagnosis Group
    if DX_CODE_COL not in df.columns:
        print(f"Error: Diagnosis code column '{DX_CODE_COL}' not found. Cannot determine groups.")
        return
    print(f"\nDetermining diagnosis groups (JIA-Only, Uveitis)...")
    df[DIAGNOSIS_GROUP_COL] = df[DX_CODE_COL].apply(lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS))
    print(f"Diagnosis group distribution:\n{df[DIAGNOSIS_GROUP_COL].value_counts()}")

    # 3. Process Visual Acuity - Stepwise Check
    print("\n--- Processing Visual Acuity Columns Stepwise ---")
    
    # 3a. VADCC (Corrected Distance)
    df = calculate_worst_overall_logmar(df, VA_L_VADCC, VA_R_VADCC, WORST_LOGMAR_VADCC_COL)
    
    # 3b. VADSC (Uncorrected Distance)
    df = calculate_worst_overall_logmar(df, VA_L_VADSC, VA_R_VADSC, WORST_LOGMAR_VADSC_COL)
    
    # 3c. Combined Distance (VADCC or VADSC)
    print(f"\nCalculating worst overall LogMAR using Combined Distance (VADCC or VADSC)")
    df['Combined_Left_LogMAR_List'] = df[f'{VA_L_VADCC}_LogMAR_List'] + df[f'{VA_L_VADSC}_LogMAR_List']
    df['Combined_Right_LogMAR_List'] = df[f'{VA_R_VADCC}_LogMAR_List'] + df[f'{VA_R_VADSC}_LogMAR_List']
    df['Worst_Combined_Left_LogMAR'] = df['Combined_Left_LogMAR_List'].apply(get_worst_logmar)
    df['Worst_Combined_Right_LogMAR'] = df['Combined_Right_LogMAR_List'].apply(get_worst_logmar)
    df[WORST_LOGMAR_COMBINED_DIST_COL] = df[['Worst_Combined_Left_LogMAR', 'Worst_Combined_Right_LogMAR']].max(axis=1, skipna=False)
    df[WORST_LOGMAR_COMBINED_DIST_COL] = df[WORST_LOGMAR_COMBINED_DIST_COL].fillna(df['Worst_Combined_Left_LogMAR']).fillna(df['Worst_Combined_Right_LogMAR'])
    valid_count_combined = df[WORST_LOGMAR_COMBINED_DIST_COL].notna().sum()
    print(f"  Found worst overall Combined Distance LogMAR for {valid_count_combined} patients into column '{WORST_LOGMAR_COMBINED_DIST_COL}'.")
    if valid_count_combined > 0:
        print(f"  Distribution:\n{df[WORST_LOGMAR_COMBINED_DIST_COL].describe()}")
        
    # 3d. Check other columns for uniquely captured low vision (CF/HM/LP/NLP)
    print("\nChecking other sparse columns for unique low vision entries (CF/HM/LP/NLP)...")
    low_vision_cols_to_check = [
        'exam left vadcco (list distinct)', 'exam right vadcco (list distinct)',
        'exam left vadsco (list distinct)', 'exam right vadsco (list distinct)',
        'exam left vancc (list distinct)', 'exam right vancc (list distinct)',
        'exam left vansc (list distinct)', 'exam right vansc (list distinct)',
        'exam left vadnr (list distinct)', 'exam right vadnr (list distinct)',
        'exam left vadnro (list distinct)', 'exam right vadnro (list distinct)'
    ]
    patients_with_only_sparse_low_vision = 0
    df['Has_Low_Vision_In_Sparse'] = False
    df['Worst_Sparse_Low_Vision_LogMAR'] = np.nan

    for index, row in df.iterrows():
        if pd.notna(row[WORST_LOGMAR_COMBINED_DIST_COL]): continue
        sparse_logmars = []
        for col in low_vision_cols_to_check:
            if col in df.columns:
                logmar_list = va_string_to_logmar_list(row[col])
                low_vision_logmars = [l for l in logmar_list if l >= LOGMAR_EQUIV['CF']]
                sparse_logmars.extend(low_vision_logmars)
        if sparse_logmars:
            worst_sparse_logmar = max(sparse_logmars)
            df.loc[index, 'Has_Low_Vision_In_Sparse'] = True
            df.loc[index, 'Worst_Sparse_Low_Vision_LogMAR'] = worst_sparse_logmar
            patients_with_only_sparse_low_vision += 1
            
    print(f"  Found {patients_with_only_sparse_low_vision} additional patients with only CF/HM/LP/NLP found in sparse columns.")
    if patients_with_only_sparse_low_vision > 0:
         print("  Updating combined LogMAR column with these sparse low vision findings...")
         df[WORST_LOGMAR_COMBINED_DIST_COL] = df[WORST_LOGMAR_COMBINED_DIST_COL].fillna(df['Worst_Sparse_Low_Vision_LogMAR'])
         new_valid_count = df[WORST_LOGMAR_COMBINED_DIST_COL].notna().sum()
         print(f"  Final worst overall LogMAR (Distance OR Sparse Low Vision) available for {new_valid_count} patients.")
         if new_valid_count > 0: print(f"  Final Distribution:\n{df[WORST_LOGMAR_COMBINED_DIST_COL].describe()}")
    else:
         print("  No additional patients gained from sparse low vision columns.")
         print(f"  Using Combined Distance LogMAR ({valid_count_combined} patients) for final analysis.")

    # --- Analysis using the most comprehensive VA data ---
    final_logmar_col = WORST_LOGMAR_COMBINED_DIST_COL # Use the column derived above
    final_valid_va_count = df[final_logmar_col].notna().sum()

    if final_valid_va_count == 0: # Check if any VA data exists at all
        print("\nError: No valid visual acuity data could be finalized for analysis.")
        return

    # Create Binary Worse VA Outcome based on the final LogMAR column
    df[WORSE_VA_BINARY_COL] = df[final_logmar_col] >= WORSE_VA_THRESHOLD_LOGMAR
    print(f"\nBinary Worse VA ({WORSE_VA_BINARY_COL}, >=20/50 based on final LogMAR) distribution:\n{df[WORSE_VA_BINARY_COL].value_counts(dropna=False)}")

    # 4. Prepare Data for Final Analysis
    analysis_df = df[
        df[final_logmar_col].notna() &
        (df[SVI_QUARTILE_COL] != 'Unknown') &
        df[DIAGNOSIS_GROUP_COL].isin(GROUPS_TO_ANALYZE)
    ].copy()

    print(f"\nPrepared FINAL analysis dataset with {len(analysis_df)} patients having valid VA, SVI Quartile, and Diagnosis Group.")
    if len(analysis_df) == 0:
        print("No patients remaining after filtering for final analysis. Exiting.")
        return
        
    print(f"Distribution in FINAL analysis set:")
    print(f"- Diagnosis Group:\n{analysis_df[DIAGNOSIS_GROUP_COL].value_counts()}")
    print(f"- SVI Quartile:\n{analysis_df[SVI_QUARTILE_COL].value_counts().sort_index()}")
    print(f"- Worse VA (>=20/50):\n{analysis_df[WORSE_VA_BINARY_COL].value_counts()}")

    # Check if JIA-Only group is present
    jia_only_present = 'JIA-Only' in analysis_df[DIAGNOSIS_GROUP_COL].unique()
    if not jia_only_present:
        print("\nWARNING: JIA-Only group still has 0 patients in the final analysis set.")
    else:
        print("\nINFO: JIA-Only group is present in the final analysis set.")

    # 5. Run Per-Group LogMAR Analysis (ANOVA, Boxplots)
    logmar_analysis_output_dir = os.path.join(OUTPUT_DIR, "logmar_analysis_by_group")
    os.makedirs(logmar_analysis_output_dir, exist_ok=True)
    for group in GROUPS_TO_ANALYZE:
        df_group_subset = analysis_df[analysis_df[DIAGNOSIS_GROUP_COL] == group]
        if not df_group_subset.empty:
            run_logmar_analysis_for_group(
                df_subset=df_group_subset, 
                group_name=group, 
                logmar_col=final_logmar_col, 
                svi_quartile_col=SVI_QUARTILE_COL, 
                output_dir=logmar_analysis_output_dir
            )
        else:
             print(f"\n--- Skipping Continuous LogMAR Analysis for Group: {group} (0 patients in final analysis set) ---")

    # 6. Analyze Proportion of Worse VA by SVI Quartile and Diagnosis Group (Final)
    print(f"\n--- Analyzing Proportion with Worse VA (>=20/50) using FINAL data ---")

    proportion_stats = analysis_df.groupby([DIAGNOSIS_GROUP_COL, SVI_QUARTILE_COL], observed=False)[WORSE_VA_BINARY_COL].agg(
        total_count='count',
        worse_va_count='sum' 
    ).reset_index()
    
    proportion_stats['proportion_worse_va'] = proportion_stats['worse_va_count'] / proportion_stats['total_count']
    proportion_stats['proportion_worse_va'] = proportion_stats['proportion_worse_va'].fillna(0)

    proportion_stats[SVI_QUARTILE_COL] = pd.Categorical(proportion_stats[SVI_QUARTILE_COL], categories=QUARTILE_ORDER, ordered=True)
    proportion_stats = proportion_stats.sort_values(by=[DIAGNOSIS_GROUP_COL, SVI_QUARTILE_COL])

    print("FINAL Proportion of patients with Worse VA (>=20/50):")
    print(proportion_stats)
    output_stats_path = os.path.join(OUTPUT_DIR, "svi_vs_worse_va_proportions_FINAL.csv")
    proportion_stats.to_csv(output_stats_path, index=False)
    print(f"Saved FINAL proportion statistics to {output_stats_path}")

    # 7. Statistical Testing (Chi-squared within each quartile) (Final)
    print("\nPerforming Chi-squared tests (JIA-Only vs Uveitis within each SVI quartile - FINAL)...")
    test_results = []
    for quartile in QUARTILE_ORDER:
        subset = analysis_df[analysis_df[SVI_QUARTILE_COL] == quartile]
        if len(subset[DIAGNOSIS_GROUP_COL].unique()) == len(GROUPS_TO_ANALYZE):
            contingency_table = pd.crosstab(subset[DIAGNOSIS_GROUP_COL], subset[WORSE_VA_BINARY_COL])
            if contingency_table.shape == (len(GROUPS_TO_ANALYZE), 2): 
                if (contingency_table.values < 5).any(): print(f"  Warning: Low expected count in {quartile}, Fisher's exact test might be more appropriate.")
                try:
                    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                    print(f"  {quartile}: Chi2={chi2:.2f}, p={p:.4f}")
                    test_results.append({'Quartile': quartile, 'Chi2': chi2, 'p_value': p, 'dof': dof})
                except ValueError as e:
                     print(f"  {quartile}: Could not perform Chi-squared test: {e}")
                     test_results.append({'Quartile': quartile, 'Chi2': np.nan, 'p_value': np.nan, 'dof': np.nan})
            else:
                print(f"  {quartile}: Skipping test - Contingency table unexpected shape {contingency_table.shape}.")
                test_results.append({'Quartile': quartile, 'Chi2': np.nan, 'p_value': np.nan, 'dof': np.nan})
        else:
            print(f"  {quartile}: Skipping test - Both JIA-Only and Uveitis not present in this quartile.")
            test_results.append({'Quartile': quartile, 'Chi2': np.nan, 'p_value': np.nan, 'dof': np.nan})

    df_test_results = pd.DataFrame(test_results)
    output_chi2_path = os.path.join(OUTPUT_DIR, "svi_vs_worse_va_chi2_results_FINAL.csv")
    df_test_results.to_csv(output_chi2_path, index=False)
    print(f"Saved FINAL Chi-squared results to {output_chi2_path}")

    # 8. Visualization - Proportion Bar Plot (Final)
    print("\nCreating FINAL Proportion Bar Plot...")
    plt.figure(figsize=(12, 7))
    ax_bar = sns.barplot(data=proportion_stats, x=SVI_QUARTILE_COL, y='proportion_worse_va', 
                       hue=DIAGNOSIS_GROUP_COL, hue_order=DIAGNOSIS_GROUP_ORDER,
                       palette=['steelblue', 'darkorange'])  # Reverted to original colors

    plt.title('Proportion of Patients with Worse Visual Acuity (>= 20/50) by SVI Quartile', fontsize=16, pad=20)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Proportion with VA ≥ 20/50', fontsize=12)
    plt.ylim(0, max(1.0, proportion_stats['proportion_worse_va'].max() * 1.15 + 0.1)) 
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    y_max = proportion_stats['proportion_worse_va'].max() if not proportion_stats.empty else 0
    text_y_offset_bar = 0.015 * ax_bar.get_ylim()[1] # Offset for proportion text on bars
    text_p_val_y = y_max + 0.05 * ax_bar.get_ylim()[1] # Reposition p-values slightly higher

    # Add proportion text ON the bars
    for container in ax_bar.containers:
        ax_bar.bar_label(container, fmt='{:.0%}', label_type='edge', padding=2, fontsize=9)

    # Add N counts and p-values (below x-axis and above bars)
    for i, quartile in enumerate(QUARTILE_ORDER):
        jia_row = proportion_stats.loc[(proportion_stats[DIAGNOSIS_GROUP_COL] == 'JIA-Only') & (proportion_stats[SVI_QUARTILE_COL] == quartile)]
        n_jia = jia_row['total_count'].iloc[0] if not jia_row.empty else 0

        uveitis_row = proportion_stats.loc[(proportion_stats[DIAGNOSIS_GROUP_COL] == 'Uveitis') & (proportion_stats[SVI_QUARTILE_COL] == quartile)]
        n_uveitis = uveitis_row['total_count'].iloc[0] if not uveitis_row.empty else 0
        
        p_val_row = df_test_results[df_test_results['Quartile'] == quartile]
        p_val = p_val_row['p_value'].iloc[0] if not p_val_row.empty else np.nan
        
        xtick_label = f"{quartile}\n(JIA:N={n_jia}, Uveitis:N={n_uveitis})"
        ax_bar.text(i, -0.08, xtick_label, ha='center', va='top', fontsize=9, transform=ax_bar.get_xaxis_transform())
        
        if pd.notna(p_val):
            p_text = f"p={p_val:.3f}"
            if p_val < 0.001: p_text = "p<0.001"
            elif p_val < 0.05: p_text += "*"
            ax_bar.text(i, text_p_val_y, p_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            if n_jia == 0 or n_uveitis == 0: 
                 p_display_text = "NA (1 group)"
            elif n_jia > 0 and n_uveitis > 0:
                 p_display_text = "p=NA (Test Fail)" 
            else: 
                 p_display_text = "p=NA"
            ax_bar.text(i, text_p_val_y, p_display_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
             
    ax_bar.set_xticklabels([]) 
    ax_bar.set_xlabel("SVI Quartile (with N counts)", fontsize=12) 
    plt.legend(title='Diagnosis Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_plot_path = os.path.join(OUTPUT_DIR, "svi_vs_worse_va_proportion_plot_FINAL.png")
    plt.savefig(output_plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved FINAL Proportion Bar Plot to {output_plot_path}")

    print("\n--- Extended VA Check & Analysis Complete ---")

if __name__ == "__main__":
    main() 