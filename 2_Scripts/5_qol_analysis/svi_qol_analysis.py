#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os
from datetime import datetime
import re
from collections import Counter
import warnings

# Ignore specific warnings if necessary
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- Configuration ---
INPUT_FILE = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv"
OUTPUT_DIR = "svi_qol_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 300

# Column Names
SVI_COLS = [
    'svi_socioeconomic (list distinct)',      # col 483
    'svi_household_comp (list distinct)',     # col 480
    'svi_housing_transportation (list distinct)', # col 481
    'svi_minority (list distinct)'            # col 484
]
SVI_TOTAL_COL = 'SVI_total'
SVI_QUARTILE_COL = 'SVI_quartile'

DX_CODE_COL = 'dx code (list distinct)' # col 502

# --- QOL Configuration ---
# Define patterns to identify QOL columns

# PedsQL Patterns - Enhanced with more keywords
PEDSQL_CHILD_SUFFIX = r'pedsql .* c \(list distinct\)'
PEDSQL_PATTERNS = {
    'Physical': ['bath', 'lift', 'exercise', 'run', 'sport', 'energy', 'chore', 'walk', 'hurt', 'ache', 'low energy'],
    'Emotional': ['afraid', 'angry', 'sad', 'worried', 'scared', 'sleep', 'trouble sleep', 'fear', 'mad', 'upset'],
    'Social': ['friend', 'play', 'peer', 'keepup', 'behind', 'get along', 'tease', 'game', 'playdate', 'social'],
    'School': ['school', 'class', 'classroom', 'pay attention', 'forget', 'homework', 'miss school', 'lesson']
}

# CHAQ Columns
CHAQ_COLUMNS = ['child dress', 'child walk', 'child cut meat', 'child tub bath', 'child toilet', 
                'child sweater', 'child shampoo', 'child socks', 'child nails', 'child stand']
CHAQ_PATTERN = r'(' + '|'.join(CHAQ_COLUMNS) + r')'

# EQ-5D-Y Vision Columns
EQVISION_PREFIX = 'eqv5y'

# Pain and Function Slider Columns
PAIN_SLIDER_PATTERN = r'pain slider child'
FUNCTION_SLIDER_PATTERN = r'functioning slider child'

# Output Column Names
QOL_SCORE_COLS = {
    'pedsql': 'pedsql_total_score_child',
    'chaq': 'chaq_function_score', 
    'eqvision': 'eqvision_qol_score',
    'pain': 'pain_slider_score', 
    'function': 'function_slider_score'
}

DIAGNOSIS_GROUP_COL = 'Diagnosis_Group'

# Diagnosis Grouping Patterns
JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']

# Analysis Groups
GROUPS_TO_ANALYZE = ['JIA-Only', 'Uveitis']
DIAGNOSIS_GROUP_ORDER = ['JIA-Only', 'Uveitis'] # Consistent order

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
# Ensure 'Times New Roman' is set for Matplotlib and Seaborn
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_theme(style='whitegrid', font='Times New Roman', rc={'font.family': 'Times New Roman', 'font.sans-serif': ['Times New Roman']})
sns.set_context("paper", font_scale=1.2, rc={"font.family": "Times New Roman"}) # Reinforce for context
QUARTILE_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']

# --- Helper Functions ---

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

def compare_q1_vs_q4(df, value_col, quartile_col='SVI_quartile'):
    """Performs Mann-Whitney U test between Q1 and Q4 groups."""
    print("\nPerforming Q1 vs Q4 comparison (Mann-Whitney U test)...")
    
    q1_data = df[df[quartile_col] == 'Q1'][value_col].dropna()
    q4_data = df[df[quartile_col] == 'Q4'][value_col].dropna()
    
    q1_n = len(q1_data)
    q4_n = len(q4_data)
    
    if q1_n < 2 or q4_n < 2:
        print(f"Insufficient data for Q1 vs Q4 comparison: Q1 (n={q1_n}), Q4 (n={q4_n})")
        return np.nan, f"Q1 vs Q4: Insufficient data"
    
    try:
        u_stat, p_value = stats.mannwhitneyu(q1_data, q4_data, alternative='two-sided')
        p_text = f"p={p_value:.4f}"
        if p_value < 0.001:
            p_text = "p<0.001"
        elif p_value < 0.05:
            p_text += "*"
        
        result_text = f"Q1 vs Q4: {p_text}"
        print(f"Q1 (n={q1_n}) vs Q4 (n={q4_n}): U={u_stat:.1f}, {p_text}")
        return p_value, result_text
    except Exception as e:
        print(f"Could not perform Mann-Whitney U test: {e}")
        return np.nan, "Q1 vs Q4: Error"

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
    for col in svi_cols:
        if col in df.columns:
            mean_col = f"{col}_mean"
            df[mean_col] = df[col].apply(parse_svi_values)
            component_mean_cols.append(mean_col)
        else:
            print(f"Warning: SVI component column '{col}' not found.")

    if not component_mean_cols:
        print("Error: No SVI component columns found or processed. Cannot calculate total SVI.")
        df[total_col] = np.nan
        df[quartile_col] = 'Unknown'
        return df

    df[total_col] = df[component_mean_cols].mean(axis=1, skipna=True)
    print(f"Calculated total SVI for {df[total_col].notna().sum()} patients.")

    df[quartile_col] = pd.Series(dtype=object)
    non_na_svi = df[total_col].dropna()
    if len(non_na_svi) >= 4:  # Changed from 2 to 4 (need at least 4 values for 4 quartiles)
        try:
            quartiles = pd.qcut(non_na_svi, 4, labels=QUARTILE_ORDER)  # Changed from 2 to 4 quartiles
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
         return 'JIA-Only'
    else:
        return 'Other'

# --- QOL Processing Functions ---

# 1. PedsQL Processing
def process_pedsql_response(value):
    """Converts PedsQL text response to numerical score (0-4, higher=worse)."""
    if pd.isna(value): return np.nan
    val_str = str(value).lower()
    # PedsQL: 0=Never, 1=Almost Never, 2=Sometimes, 3=Often, 4=Almost Always
    if "almost always" in val_str: return 4
    if "often" in val_str: return 3
    if "sometimes" in val_str: return 2
    if "almost never" in val_str: return 1
    if "never" in val_str: return 0 # Assuming "0=Never" is the base
    # Fallback: try extracting highest number if text format failed
    numbers = re.findall(r'\d+', str(value))
    if numbers: 
        num = max([int(n) for n in numbers])
        if 0 <= num <= 4: return num
    return np.nan

def standardize_pedsql(score):
    """Converts 0-4 PedsQL score (higher=worse) to 0-100 scale (higher=better)."""
    if pd.isna(score) or not (0 <= score <= 4): return np.nan
    # PedsQL standard transformation: (4 - score) * 25
    return (4 - score) * 25

def calculate_pedsql_domain_score(df, domain_name, domain_patterns, child_suffix_pattern):
    """Calculates a PedsQL domain score (0-100, higher=better)."""
    print(f"  Calculating PedsQL {domain_name} Score...")
    domain_cols = []
    # Find columns matching domain keywords and child suffix pattern
    for col in df.columns:
        col_lower = col.lower()
        is_child_report = re.search(child_suffix_pattern, col_lower)
        has_keyword = any(pattern.lower() in col_lower for pattern in domain_patterns)
        if is_child_report and has_keyword:
            domain_cols.append(col)
            
    if not domain_cols:
        print(f"    Warning: No columns found for PedsQL {domain_name} domain.")
        return None # Return None if no columns found for this domain
        
    print(f"    Found {len(domain_cols)} columns for {domain_name}: {domain_cols[:5]}...")
    
    # Process and standardize each item
    standardized_cols = []
    for col in domain_cols:
        processed_col = f"{col}_processed"
        standardized_col = f"{col}_standardized"
        df[processed_col] = df[col].apply(process_pedsql_response)
        df[standardized_col] = df[processed_col].apply(standardize_pedsql)
        standardized_cols.append(standardized_col)
        
    # Calculate domain score (mean of standardized items)
    domain_score_col = f"pedsql_{domain_name.lower()}_score_child"
    df[domain_score_col] = df[standardized_cols].mean(axis=1, skipna=True) # Use skipna=True
    print(f"    Calculated {domain_score_col} for {df[domain_score_col].notna().sum()} patients.")
    return domain_score_col # Return the name of the created column

# 2. CHAQ Processing
def process_chaq_response(value):
    """Converts CHAQ text response to numerical score (0-3, higher=worse)."""
    if pd.isna(value): return np.nan
    val_str = str(value).lower()
    
    # CHAQ: 0=Without ANY Difficulty, 1=With SOME Difficulty, 2=With MUCH Difficulty, 3=UNABLE To Do
    if "without any difficulty" in val_str or "without difficulty" in val_str: return 0
    if "with some difficulty" in val_str: return 1
    if "with much difficulty" in val_str: return 2
    if "unable to do" in val_str: return 3
    
    # Fallback: try extracting highest number if text format failed
    numbers = re.findall(r'\d+', str(value))
    if numbers: 
        num = max([int(n) for n in numbers])
        if 0 <= num <= 3: return num
    return np.nan

def standardize_chaq(score):
    """Converts 0-3 CHAQ score (higher=worse) to 0-100 scale (higher=better)."""
    if pd.isna(score) or not (0 <= score <= 3): return np.nan
    # CHAQ standard transformation: (3 - score) * 33.33
    return (3 - score) * (100/3)

def calculate_chaq_score(df, chaq_pattern):
    """Calculates CHAQ function score (0-100, higher=better)."""
    print("\nCalculating CHAQ Function Score...")
    chaq_cols = []
    # Find CHAQ columns
    for col in df.columns:
        if re.search(chaq_pattern, col.lower()):
            chaq_cols.append(col)
    
    if not chaq_cols:
        print("  Warning: No CHAQ columns found.")
        return None
    
    print(f"  Found {len(chaq_cols)} CHAQ columns: {chaq_cols[:5]}...")
    
    # Process CHAQ responses
    chaq_processed_cols = []
    for col in chaq_cols:
        processed_col = f"{col}_processed"
        df[processed_col] = df[col].apply(process_chaq_response)
        chaq_processed_cols.append(processed_col)
    
    # Calculate CHAQ disability index (0-3, higher = worse function)
    # CHAQ disability index is the max score across all items
    score_col = QOL_SCORE_COLS['chaq']
    df['chaq_disability_index'] = df[chaq_processed_cols].max(axis=1)
    
    # Convert to CHAQ function score (0-100, higher = better function)
    df[score_col] = df['chaq_disability_index'].apply(standardize_chaq)
    
    print(f"  Calculated {score_col} for {df[score_col].notna().sum()} patients.")
    if df[score_col].notna().sum() > 0:
        print(f"  {score_col} summary:\n{df[score_col].describe()}")
    
    return score_col

# 3. EQ-Vision Processing
def process_eqvision_response(value):
    """Converts EQ-Vision text response to numerical score (0-2, higher=worse)."""
    if pd.isna(value): return np.nan
    val_str = str(value).lower()
    
    # EQ-Vision uses 0-2 scale based on comprehensive script logic
    # 0=No problems/Never hard, 1=A little/Sometimes hard, 2=Very/Always hard
    if "very hard" in val_str or "always hard" in val_str or "a lot hard" in val_str or "quite hard" in val_str:
         return 2
    if "a little hard" in val_str or "little bit hard" in val_str or "sometimes hard" in val_str or "somewhat hard" in val_str: 
        return 1
    if "no problems" in val_str or "no difficulty" in val_str or "never hard" in val_str:
         return 0 

    # Fallback: try extracting highest number (0, 1, or 2)
    numbers = re.findall(r'\d+', str(value))
    if numbers: 
        num = max([int(n) for n in numbers])
        if 0 <= num <= 2: return num
    return np.nan

def standardize_eqvision(score):
    """Converts 0-2 EQ-Vision score (higher=worse) to 0-100 scale (higher=better)."""
    if pd.isna(score) or not (0 <= score <= 2): return np.nan
    # EQ-Vision standard transformation based on 0-2 scale: (2 - score) * 50
    return (2 - score) * 50

def calculate_eqvision_score(df, eqvision_prefix):
    """Calculates EQ-Vision score (0-100, higher=better)."""
    print("\nCalculating EQ-Vision Score...")
    eqvision_cols = []
    # Find EQ-Vision columns
    for col in df.columns:
        if eqvision_prefix in col.lower():
            eqvision_cols.append(col)
    
    if not eqvision_cols:
        print("  Warning: No EQ-Vision columns found.")
        return None
    
    print(f"  Found {len(eqvision_cols)} EQ-Vision columns: {eqvision_cols[:5]}...")
    
    # Process and standardize each item
    standardized_cols = []
    for col in eqvision_cols:
        processed_col = f"{col}_processed"
        standardized_col = f"{col}_standardized"
        df[processed_col] = df[col].apply(process_eqvision_response)
        df[standardized_col] = df[processed_col].apply(standardize_eqvision)
        standardized_cols.append(standardized_col)
    
    # Calculate EQ-Vision score (mean of standardized items)
    score_col = QOL_SCORE_COLS['eqvision']
    df[score_col] = df[standardized_cols].mean(axis=1, skipna=True)
    
    print(f"  Calculated {score_col} for {df[score_col].notna().sum()} patients.")
    if df[score_col].notna().sum() > 0:
        print(f"  {score_col} summary:\n{df[score_col].describe()}")
    
    return score_col

# 4. Slider Processing Helper
def parse_slider_values(value):
    """Parses potential multi-value slider entries (0-100 scale assumed)."""
    if pd.isna(value): return np.nan
    
    valid_scores = []
    try:
        parts = str(value).split(';')
        for part in parts:
            cleaned_part = part.strip()
            # Try to extract leading number
            match = re.match(r'^([0-9.]+)', cleaned_part)
            if match:
                num_str = match.group(1)
                try:
                    num = float(num_str)
                    # Assume 0-100 scale based on observed data
                    if 0 <= num <= 100:
                        valid_scores.append(num)
                except ValueError:
                    continue # Ignore parts that are not valid numbers
    except Exception as e:
        print(f"Warning: Error parsing slider value '{value}': {e}")
        return np.nan
        
    if valid_scores:
        return np.mean(valid_scores)
    else:
        return np.nan

# 5. Pain Slider Processing (Revised)
def calculate_pain_slider_score(df, pain_slider_pattern):
    """Calculates Pain Slider score (0-100, higher=better) assuming raw 0-100 scale."""
    print("\nCalculating Pain Slider Score...")
    pain_slider_cols = []
    for col in df.columns:
        if re.search(pain_slider_pattern, col.lower()):
            pain_slider_cols.append(col)
    
    if not pain_slider_cols:
        print("  Warning: No Pain Slider column found.")
        return None
    
    # Use the first matching column (handle potential duplicates like _alt1)
    pain_col = pain_slider_cols[0]
    print(f"  Using Pain Slider column: {pain_col}")
    
    # Parse values using the helper function (gets avg of valid 0-100 numbers)
    df['pain_slider_raw_avg'] = df[pain_col].apply(parse_slider_values)
    
    # Convert raw average (0-100, higher=worse) to QOL score (0-100, higher=better)
    score_col = QOL_SCORE_COLS['pain']
    df[score_col] = 100 - df['pain_slider_raw_avg']
    
    valid_count = df[score_col].notna().sum()
    print(f"  Calculated {score_col} for {valid_count} patients.")
    if valid_count > 0:
        print(f"  {score_col} summary:\n{df[score_col].describe()}")
    
    return score_col

# 6. Function Slider Processing (Revised)
def calculate_function_slider_score(df, function_slider_pattern):
    """Calculates Function Slider score (0-100, higher=better) assuming raw 0-100 scale."""
    print("\nCalculating Function Slider Score...")
    function_slider_cols = []
    for col in df.columns:
        if re.search(function_slider_pattern, col.lower()):
            function_slider_cols.append(col)
    
    if not function_slider_cols:
        print("  Warning: No Function Slider column found.")
        return None
    
    # Use the first matching column
    function_col = function_slider_cols[0]
    print(f"  Found Function Slider column: {function_col}")
    
    # Parse values using the helper function (gets avg of valid 0-100 numbers)
    # This raw average (0-100, higher=better) IS the final score
    score_col = QOL_SCORE_COLS['function']
    df[score_col] = df[function_col].apply(parse_slider_values)
    
    valid_count = df[score_col].notna().sum()
    print(f"  Calculated {score_col} for {valid_count} patients.")
    if valid_count > 0:
        print(f"  {score_col} summary:\n{df[score_col].describe()}")
    
    return score_col

# --- Analysis Functions (Adapted from steroid duration script) ---
def run_qol_analysis_for_group(df_subset, group_name, qol_col, svi_total_col, svi_quartile_col, output_dir):
    """Performs analysis (stats, plots) for QOL for a specific diagnosis group.
    Returns a dictionary with regression results (slope, p-value) if calculated, else None.
    """
    print(f"\n--- Analyzing Group: {group_name} ({len(df_subset)} patients) ---")
    output_prefix = os.path.join(output_dir, group_name.replace(' ', '_'))
    regression_results = None # Initialize return value

    # Font properties
    font_props_axis_labels = {'fontname': 'Times New Roman', 'fontsize': 12}
    font_props_title = {'fontname': 'Times New Roman', 'fontsize': 14}
    font_props_tick_labels = {'fontname': 'Times New Roman', 'fontsize': 9} # Adjusted for N-counts
    font_props_stat_text = {'fontname': 'Times New Roman', 'fontsize': 9}

    # --- Descriptive Statistics ---
    print(f"Descriptive Statistics for {qol_col}:")
    qol_stats_desc = df_subset[qol_col].describe()
    print(qol_stats_desc)
    qol_stats_desc.to_csv(f"{output_prefix}_{qol_col}_descriptive_stats.csv")

    # --- Analysis by SVI Quartile ---
    print(f"\nAnalyzing {qol_col} by SVI quartile...")
    svi_qol_stats = df_subset.groupby(svi_quartile_col, observed=False)[qol_col].agg(['count', 'mean', 'median', 'std']).reset_index()
    svi_qol_stats = svi_qol_stats.set_index(svi_quartile_col).reindex(QUARTILE_ORDER).reset_index()
    svi_qol_stats['count'] = svi_qol_stats['count'].fillna(0).astype(int)
    svi_qol_stats['mean'] = svi_qol_stats['mean'].fillna(np.nan)
    print(svi_qol_stats)
    svi_qol_stats.to_csv(f"{output_prefix}_{qol_col}_by_svi_stats.csv", index=False)

    # --- Statistical Testing: ANOVA --- 
    print("\nPerforming statistical tests (ANOVA)...")
    present_quartiles = [q for q in QUARTILE_ORDER if q in df_subset[svi_quartile_col].unique()]
    groups = [df_subset[qol_col][df_subset[svi_quartile_col] == q] for q in present_quartiles]
    groups = [g.dropna() for g in groups if len(g.dropna()) >= 2] 

    anova_f_stat, anova_p_value = np.nan, np.nan
    anova_p_text = "ANOVA: Insufficient data"
    if len(groups) >= 2: 
        try:
            anova_f_stat, anova_p_value = stats.f_oneway(*groups)
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
    
    # --- Calculate Q1 vs Q4 p-value ---
    q1_q4_p_value, q1_q4_text = compare_q1_vs_q4(df_subset, qol_col, svi_quartile_col)
        
    # --- Create Plot Labels with N counts and Mean values ---
    plot_labels = {}
    for q in present_quartiles:
         count = svi_qol_stats.loc[svi_qol_stats[svi_quartile_col] == q, 'count'].iloc[0]
         mean_val = svi_qol_stats.loc[svi_qol_stats[svi_quartile_col] == q, 'mean'].iloc[0]
         mean_str = f"{mean_val:.1f}" if pd.notna(mean_val) else "NaN"
         plot_labels[q] = f"{q}\n(N={count})\nMean={mean_str}" 
    ordered_plot_labels = [plot_labels.get(q, f"{q}\n(N=0)\nMean=NaN") for q in present_quartiles]

    # --- Visualization: Box Plot & Bar Plot --- 
    if not present_quartiles:
        print("No SVI quartiles present in this group for plotting.")
        print(f"Linear Regression Skipped for {group_name}: No data for plots.") 
        return regression_results # Return None if no plot/regression data
        
    plt.figure(figsize=(12, 7))
    qol_title_name = qol_col.replace('_', ' ').title()
    
    # Box Plot
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    sns.boxplot(x=svi_quartile_col, y=qol_col, data=df_subset, order=present_quartiles, palette='viridis', ax=ax1)
    ax1.set_title(f'{qol_title_name} by SVI Quartile ({group_name})', **font_props_title)
    ax1.set_xlabel('SVI Quartile', **font_props_axis_labels)
    ax1.set_ylabel(f'{qol_title_name} (Higher=Better)', **font_props_axis_labels)
    ax1.set_xticks(range(len(present_quartiles)))
    ax1.set_xticklabels(ordered_plot_labels, **font_props_tick_labels)
    for tick in ax1.get_yticklabels(): tick.set_fontname('Times New Roman')
    
    # Position text boxes in the top left corner
    stat_text = f"{anova_p_text}\n{q1_q4_text}"
    ax1.text(0.02, 0.97, stat_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), **font_props_stat_text)
    
    # Bar Plot
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    plot_data = svi_qol_stats[svi_qol_stats[svi_quartile_col].isin(present_quartiles)]
    sem_vals = plot_data['std'] / np.sqrt(plot_data['count'].replace(0, np.nan))
    bar_colors = sns.color_palette('viridis', len(present_quartiles))
    ax2.bar(range(len(present_quartiles)), plot_data['mean'], yerr=sem_vals.fillna(0), capsize=4, color=bar_colors)
    ax2.set_title(f'Mean {qol_title_name} by SVI Quartile ({group_name})', **font_props_title)
    ax2.set_xlabel('SVI Quartile', **font_props_axis_labels)
    ax2.set_ylabel(f'Mean {qol_title_name} (Higher=Better)', **font_props_axis_labels)
    ax2.set_xticks(range(len(present_quartiles)))
    ax2.set_xticklabels(ordered_plot_labels, **font_props_tick_labels)
    for tick in ax2.get_yticklabels(): tick.set_fontname('Times New Roman')
    
    # Position text boxes in the top left corner 
    ax2.text(0.02, 0.97, stat_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), **font_props_stat_text)

    plt.subplots_adjust(bottom=0.2) 
    plt.savefig(f"{output_prefix}_{qol_col}_by_svi_plots.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {output_prefix}_{qol_col}_by_svi_plots.png")

    # --- Statistical Testing: Regression --- 
    print("\nPerforming statistical tests (Linear Regression)...")
    regression_data = df_subset[[svi_total_col, qol_col]].dropna()
    if len(regression_data) >= 10: 
        X = regression_data[svi_total_col]
        y = regression_data[qol_col]
        X = sm.add_constant(X) 
        try:
            model = sm.OLS(y, X).fit()
            print(f"Linear Regression Summary ({group_name} - {qol_col}):")
            # Only print summary, don't save as separate file here
            # print(model.summary())
            reg_slope = model.params[svi_total_col]
            reg_intercept = model.params['const']
            reg_r_value = np.sqrt(model.rsquared)
            reg_p_value = model.pvalues[svi_total_col]
            
            # Store results to return
            regression_results = {'slope': reg_slope, 'p_value': reg_p_value, 'r_squared': model.rsquared, 'q1_q4_p_value': q1_q4_p_value}
            print(f"  Regression: Slope={reg_slope:.3f}, R2={model.rsquared:.3f}, p={reg_p_value:.4f}")

            plt.figure(figsize=(8, 6))
            sns.regplot(x=svi_total_col, y=qol_col, data=regression_data, scatter_kws={'alpha':0.5})
            p_val_reg_str = f"p={reg_p_value:.4f}"
            if reg_p_value < 0.001: p_val_reg_str = "p<0.001"
            elif reg_p_value < 0.05: p_val_reg_str += "*"
            plt.title(f'SVI Total vs {qol_title_name} ({group_name})\nR² = {model.rsquared:.2f}, {p_val_reg_str}', **font_props_title)
            plt.xlabel('SVI Total Score', **font_props_axis_labels)
            plt.ylabel(f'{qol_title_name} (Higher=Better)', **font_props_axis_labels)
            for tick in plt.gca().get_xticklabels(): tick.set_fontname('Times New Roman')
            for tick in plt.gca().get_yticklabels(): tick.set_fontname('Times New Roman')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{output_prefix}_{qol_col}_svi_regression.png", dpi=DPI, bbox_inches='tight')
            plt.close()
        except Exception as e:
             print(f"Could not perform Linear Regression for {group_name} - {qol_col}: {e}")
             # regression_results remains None
    else:
        print(f"Linear Regression Skipped for {group_name} - {qol_col}: Insufficient data (need >= 10 points). Found {len(regression_data)}.")
        # regression_results remains None
        
    return regression_results # Return the dictionary or None

def create_qol_dual_bar_chart(df_analysis, group_col, quartile_col, qol_col, output_dir):
    """Creates a dual bar chart comparing QOL across SVI quartiles for JIA-Only vs Uveitis."""
    print(f"\n--- Creating Dual Bar Chart for {qol_col} (JIA-Only vs Uveitis) ---")
    output_prefix = os.path.join(output_dir, f"Dual_Comparison_{qol_col}")
    qol_title_name = qol_col.replace('_', ' ').title()

    # Font properties
    font_props_legend = {'family': 'Times New Roman', 'size': 10}
    font_props_axis_labels = {'fontname': 'Times New Roman', 'fontsize': 12}
    font_props_title = {'fontname': 'Times New Roman', 'fontsize': 14}
    font_props_tick_labels = {'fontname': 'Times New Roman', 'fontsize': 10} # Adjusted for N-counts
    font_props_pvalue_text = {'fontname': 'Times New Roman', 'fontsize': 9}
    font_props_stat_text = {'fontname': 'Times New Roman', 'fontsize': 9, 'fontweight':'bold'} # For p-values between bars

    if df_analysis.empty:
        print("Skipping dual bar chart: No analysis data available.")
        return

    summary_stats = df_analysis.groupby([group_col, quartile_col], observed=False)[qol_col].agg(['mean', 'count', 'std']).unstack(group_col)
    idx = pd.Index(QUARTILE_ORDER, name=quartile_col)
    summary_stats = summary_stats.reindex(idx)
    
    present_groups = [group for group in GROUPS_TO_ANALYZE if ('mean', group) in summary_stats.columns]
    if len(present_groups) == 0:
         print("Skipping dual bar chart: No target groups found in the data after aggregation.")
         return
    
    sem_data = summary_stats.get('std', pd.DataFrame()) / np.sqrt(summary_stats.get('count', pd.DataFrame()).replace(0, np.nan))
    means = summary_stats.get('mean', pd.DataFrame()).fillna(np.nan) 
    counts = summary_stats.get('count', pd.DataFrame()).fillna(0).astype(int)
    stderr = sem_data.fillna(0)

    for group in GROUPS_TO_ANALYZE:
        if group not in means.columns: means[group] = np.nan
        if group not in counts.columns: counts[group] = 0
        if group not in stderr.columns: stderr[group] = 0

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    bar_width = 0.35
    x = np.arange(len(QUARTILE_ORDER))

    rects1 = ax.bar(x - bar_width/2, means.get('JIA-Only', np.nan), bar_width, 
                    yerr=stderr.get('JIA-Only', 0), label='JIA-Only', capsize=4, color='#1f77b4')
    rects2 = ax.bar(x + bar_width/2, means.get('Uveitis', np.nan), bar_width, 
                    yerr=stderr.get('Uveitis', 0), label='Uveitis', capsize=4, color='#ff7f0e')

    ax.set_xlabel('SVI Quartile', **font_props_axis_labels)
    ax.set_ylabel(f'Mean {qol_title_name} (Higher=Better)', **font_props_axis_labels)
    ax.set_title(f'Mean {qol_title_name} by SVI Quartile: JIA-Only vs Uveitis', **font_props_title)
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTILE_ORDER, **font_props_tick_labels)
    ax.legend(prop=font_props_legend)

    def create_label(q_name_label):
        n1 = counts.get('JIA-Only', pd.Series(0, index=counts.index)).get(q_name_label, 0)
        n2 = counts.get('Uveitis', pd.Series(0, index=counts.index)).get(q_name_label, 0)
        return f"{q_name_label}\n(JIA:N={n1}, Uveitis:N={n2})"
    ax.set_xticklabels([create_label(q_name) for q_name in QUARTILE_ORDER], **font_props_tick_labels)
    for tick in ax.get_yticklabels(): tick.set_fontname('Times New Roman')

    y_min, y_max = ax.get_ylim()
    y_range = max(1, y_max - y_min)
    p_values_comparison_dict = {}
    q1_q4_p_values_dict = {}

    if 'JIA-Only' in present_groups:
        jia_subset = df_analysis[df_analysis[group_col] == 'JIA-Only']
        jia_q1q4_p, jia_q1q4_text = compare_q1_vs_q4(jia_subset, qol_col, quartile_col)
        q1_q4_p_values_dict['JIA-Only'] = (jia_q1q4_p, jia_q1q4_text)
    
    if 'Uveitis' in present_groups:
        uv_subset = df_analysis[df_analysis[group_col] == 'Uveitis']
        uv_q1q4_p, uv_q1q4_text = compare_q1_vs_q4(uv_subset, qol_col, quartile_col)
        q1_q4_p_values_dict['Uveitis'] = (uv_q1q4_p, uv_q1q4_text)
    
    if q1_q4_p_values_dict:
        q1q4_text_plot = ""
        for group, (p_val, text) in q1_q4_p_values_dict.items():
            q1q4_text_plot += f"{group} {text}\n"
        ax.text(0.98, 0.98, q1q4_text_plot.strip(), transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), **font_props_pvalue_text)

    for i, q_name_iter in enumerate(QUARTILE_ORDER):
        group1_data = df_analysis[(df_analysis[group_col] == 'JIA-Only') & (df_analysis[quartile_col] == q_name_iter)][qol_col].dropna() if 'JIA-Only' in present_groups else pd.Series(dtype=float)
        group2_data = df_analysis[(df_analysis[group_col] == 'Uveitis') & (df_analysis[quartile_col] == q_name_iter)][qol_col].dropna() if 'Uveitis' in present_groups else pd.Series(dtype=float)
        p_val_text_iter = "NA"
        if 'JIA-Only' in present_groups and 'Uveitis' in present_groups and len(group1_data) >= 2 and len(group2_data) >= 2:
            try:
                t_stat, p_val_iter = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                p_values_comparison_dict[q_name_iter] = p_val_iter
                if p_val_iter < 0.001: p_val_text_iter = "p<0.001"
                elif p_val_iter < 0.01: p_val_text_iter = "p<0.01"
                elif p_val_iter < 0.05: p_val_text_iter = f"p={p_val_iter:.3f}*"
                else: p_val_text_iter = f"p={p_val_iter:.3f}"
            except Exception:
                p_val_text_iter = "Error"
                p_values_comparison_dict[q_name_iter] = np.nan
        elif len(present_groups) < 2: p_val_text_iter = "Single Group"
        else: p_val_text_iter = "Insufficient N"
        p_values_comparison_dict[q_name_iter] = p_values_comparison_dict.get(q_name_iter, np.nan)
        
        bar1_mean = means.get('JIA-Only', pd.Series(np.nan, index=means.index)).get(q_name_iter, np.nan)
        bar2_mean = means.get('Uveitis', pd.Series(np.nan, index=means.index)).get(q_name_iter, np.nan)
        bar1_sem_val = stderr.get('JIA-Only', pd.Series(0, index=stderr.index)).get(q_name_iter, 0)
        bar2_sem_val = stderr.get('Uveitis', pd.Series(0, index=stderr.index)).get(q_name_iter, 0)
        
        text_y = max(bar1_mean + bar1_sem_val if pd.notna(bar1_mean) else y_min,
                     bar2_mean + bar2_sem_val if pd.notna(bar2_mean) else y_min)
        text_y += y_range * 0.03
        
        ax.text(i, text_y, p_val_text_iter, ha='center', va='bottom', **font_props_stat_text)

    plt.subplots_adjust(bottom=0.15, right=0.85)
    plt.savefig(f"{output_prefix}_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    summary_list = []
    for q_name_csv in QUARTILE_ORDER:
         for group_csv in GROUPS_TO_ANALYZE:
              mean_val = means.get(group_csv, pd.Series(np.nan, index=means.index)).get(q_name_csv, np.nan)
              count_val = counts.get(group_csv, pd.Series(0, index=counts.index)).get(q_name_csv, 0)
              sem_val_csv = stderr.get(group_csv, pd.Series(np.nan, index=stderr.index)).get(q_name_csv, np.nan)
              summary_list.append({
                  quartile_col: q_name_csv, group_col: group_csv,
                  'Mean QOL': mean_val, 'Count': count_val, 'SEM': sem_val_csv,
                  'Comparison p-value': p_values_comparison_dict.get(q_name_csv, np.nan),
                  'Q1_Q4_p_value': q1_q4_p_values_dict.get(group_csv, (np.nan, ''))[0] if q_name_csv == 'Q1' else np.nan
              })
    final_summary = pd.DataFrame(summary_list)
    final_summary = final_summary.sort_values(by=[group_col, quartile_col])
    final_summary.to_csv(f"{output_prefix}_comparison_summary.csv", index=False)
    print(f"Saved comparison summary data to {output_prefix}_comparison_summary.csv")

# --- Summary Plot Function (Revised) ---
def create_revised_summary_plot(trend_data, regression_results, output_dir):
    """Creates a revised 2x2 summary plot for clinical interpretation."""
    print(f"\n\n{'='*80}")
    print(f"=== CREATING REVISED SUMMARY PLOT ===")
    print(f"{'='*80}")
    
    if trend_data.empty:
        print("Error: No trend data available for summary plot.")
        return
        
    # Measures to include in the trend plots and regression summary
    plot_measures = [
        QOL_SCORE_COLS.get('pedsql'),
        QOL_SCORE_COLS.get('eqvision'),
        QOL_SCORE_COLS.get('chaq')
    ]
    plot_measures = [m for m in plot_measures if m and m in trend_data['QOL_Measure'].unique()]
    if len(plot_measures) < 1:
        print("Error: None of the key QOL measures (PedsQL, EQ-Vision, CHAQ) have data for plotting trends.")
        return
        
    # Use the mapped labels for plotting
    label_map = {
        QOL_SCORE_COLS.get('pedsql'): 'PedsQL Total',
        QOL_SCORE_COLS.get('chaq'): 'CHAQ Func',
        QOL_SCORE_COLS.get('eqvision'): 'EQ-Vision',
        QOL_SCORE_COLS.get('pain'): 'Pain Slider',
        QOL_SCORE_COLS.get('function'): 'Func Slider'
    }
    label_map = {k: v for k, v in label_map.items() if k}
    trend_data['QOL_Label'] = trend_data['QOL_Measure'].map(label_map)
    plot_labels = [label_map[m] for m in plot_measures]

    # --- Create 2x2 Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('SVI vs QOL Trends and Regression Summary', fontsize=18, y=1.03, fontname='Times New Roman')
    sns.set_style("whitegrid")
    palette = ['#1f77b4', '#ff7f0e']  # Blue for JIA-Only, Orange for Uveitis
    font_props_legend = {'family': 'Times New Roman', 'size': 10}
    font_props_axis_labels = {'fontname': 'Times New Roman', 'fontsize': 12}
    font_props_title = {'fontname': 'Times New Roman', 'fontsize': 14}
    font_props_tick_labels = {'fontname': 'Times New Roman', 'fontsize': 11}
    font_props_text = {'fontname': 'Times New Roman', 'fontsize': 9}

    # --- Trend Plots (Panels 1-3) ---
    axes_flat = [axes[0,0], axes[0,1], axes[1,0]]
    for i, measure in enumerate(plot_measures):
        ax = axes_flat[i]
        measure_label = label_map[measure]
        data_plot = trend_data[trend_data['QOL_Measure'] == measure].copy()
        
        if not data_plot.empty:
            # Ensure SVI_quartile is categorical and ordered for plotting
            data_plot['SVI_quartile'] = pd.Categorical(data_plot['SVI_quartile'], categories=QUARTILE_ORDER, ordered=True)
            data_plot = data_plot.sort_values('SVI_quartile')

            sns.lineplot(data=data_plot, x='SVI_quartile', y='Mean QOL', hue=DIAGNOSIS_GROUP_COL,
                         hue_order=DIAGNOSIS_GROUP_ORDER, marker='o', err_style="bars", errorbar=('se', 1), 
                         palette=palette, ax=ax)
            ax.set_title(f'{measure_label} Trend by SVI Quartile', **font_props_title)
            ax.set_xlabel('SVI Quartile', **font_props_axis_labels)
            ax.set_ylabel(f'Mean {measure_label} (Higher=Better)', **font_props_axis_labels)
            ax.legend(title='Diagnosis Group', prop=font_props_legend)
            ax.grid(True, linestyle='--', alpha=0.7)
            for tick in ax.get_xticklabels(): tick.set_fontname('Times New Roman')
            for tick in ax.get_yticklabels(): tick.set_fontname('Times New Roman')
            
            # Add Q1 vs Q4 p-values
            q1_vs_q4_text = ""
            for group in DIAGNOSIS_GROUP_ORDER:
                if group in data_plot[DIAGNOSIS_GROUP_COL].unique():
                    group_data = data_plot[data_plot[DIAGNOSIS_GROUP_COL] == group]
                    q1_q4_p_values = group_data[group_data['SVI_quartile'] == 'Q1']['Q1_Q4_p_value'].dropna()
                    
                    if not q1_q4_p_values.empty:
                        p_val = q1_q4_p_values.iloc[0]
                        if pd.notna(p_val):
                            p_text = f"p={p_val:.4f}"
                            if p_val < 0.001: p_text = "p<0.001"
                            elif p_val < 0.01: p_text = "p<0.01"
                            elif p_val < 0.05: p_text = f"p={p_val:.3f}*"
                            
                            q1_vs_q4_text += f"{group} Q1 vs Q4: {p_text}\n"
            
            if q1_vs_q4_text:
                # Position in top-right corner to avoid overlap with high bars
                ax.text(0.98, 0.98, q1_vs_q4_text.strip(), transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), **font_props_text)
        else:
            ax.text(0.5, 0.5, f'No data for {measure_label}', ha='center', va='center', **font_props_axis_labels)
            ax.set_title(f'{measure_label} Trend by SVI Quartile', **font_props_title)
            
    # If only 1 or 2 measures plotted, hide unused axes
    for j in range(len(plot_measures), 3):
        axes_flat[j].set_visible(False)

    # --- Panel 4: Regression Slopes --- 
    ax4 = axes[1, 1]
    reg_plot_data = []
    measure_labels_reg = []
    for measure in plot_measures: # Only plot regression for measures shown in trends
        measure_label = label_map[measure]
        measure_labels_reg.append(measure_label)
        for group in DIAGNOSIS_GROUP_ORDER:
            result = regression_results.get((measure, group), None)
            if result:
                reg_plot_data.append({
                    'QOL_Label': measure_label,
                    DIAGNOSIS_GROUP_COL: group,
                    'Slope': result['slope'],
                    'p_value': result['p_value']
                })
            else:
                 # Add entry with NaN slope if regression failed or insufficient data
                 reg_plot_data.append({
                    'QOL_Label': measure_label,
                    DIAGNOSIS_GROUP_COL: group,
                    'Slope': np.nan,
                    'p_value': np.nan
                 })
                 
    if reg_plot_data:
        df_reg = pd.DataFrame(reg_plot_data)
        sns.barplot(data=df_reg, x='QOL_Label', y='Slope', hue=DIAGNOSIS_GROUP_COL,
                    order=measure_labels_reg, # Use labels for consistent order
                    hue_order=DIAGNOSIS_GROUP_ORDER, palette=palette, ax=ax4)
                    
        # Add significance markers
        for i, measure_label_reg in enumerate(measure_labels_reg):
            for k, group in enumerate(DIAGNOSIS_GROUP_ORDER):
                group_data = df_reg[(df_reg['QOL_Label'] == measure_label_reg) & (df_reg[DIAGNOSIS_GROUP_COL] == group)]
                if not group_data.empty:
                    p_val = group_data['p_value'].iloc[0]
                    slope_val = group_data['Slope'].iloc[0]
                    if pd.notna(p_val) and p_val < 0.05:
                        # Position star above or below bar based on slope
                        x_pos = i + (k - 0.5) * ax4.patches[0].get_width() # Adjust x position based on hue
                        y_pos = slope_val + (0.05 * np.sign(slope_val) if slope_val != 0 else 0.05) * ax4.get_ylim()[1]
                        ax4.text(x_pos, y_pos, "*", ha='center', va='bottom' if slope_val >= 0 else 'top', 
                                 fontsize=14, color='red', fontweight='bold', fontname='Times New Roman')
                                 
        ax4.set_title('SVI Impact on QOL (Regression Slope)', **font_props_title)
        ax4.set_xlabel('', **font_props_axis_labels)
        ax4.set_ylabel('Change in QOL Score per SVI Unit', **font_props_axis_labels)
        ax4.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Line at zero slope
        ax4.legend(title='Diagnosis Group', prop=font_props_legend)
        ax4.tick_params(axis='x', rotation=0) # Horizontal labels
        for tick in ax4.get_xticklabels(): tick.set_fontname('Times New Roman')
        for tick in ax4.get_yticklabels(): tick.set_fontname('Times New Roman')
    else:
        ax4.text(0.5, 0.5, 'No regression data available', ha='center', va='center', **font_props_axis_labels)
        ax4.set_title('SVI Impact on QOL (Regression Slope)', **font_props_title)
        
    # --- Final Adjustments and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    summary_plot_path = os.path.join(output_dir, "SVI_QOL_Clinical_Summary_Plot.png") # New name
    plt.savefig(summary_plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved revised summary plot to {summary_plot_path}")

# --- Multi-panel QoL Figure ---
def create_multi_panel_qol_figure(combined_data, output_dir):
    """Creates a multi-panel figure showing SVI impact on key QoL measures.
    
    Parameters:
    -----------
    combined_data : pandas.DataFrame
        Combined trend data from all QoL analyses
    output_dir : str
        Directory to save the output figure
    """
    print(f"\n\n{'='*80}")
    print(f"=== CREATING MULTI-PANEL QOL FIGURE (FIGURE 3) ===")
    print(f"{'='*80}")
    
    # Font properties (ensure these are defined here)
    font_props_legend = {'family': 'Times New Roman', 'size': 10}
    font_props_axis_labels = {'fontname': 'Times New Roman', 'fontsize': 12}
    font_props_title = {'fontname': 'Times New Roman', 'fontsize': 14}
    font_props_tick_labels = {'fontname': 'Times New Roman', 'fontsize': 11} 
    font_props_pvalue_text = {'fontname': 'Times New Roman', 'fontsize': 9} # For Q1 vs Q4 text box
    font_props_stat_text = {'fontname': 'Times New Roman', 'fontsize': 8, 'fontweight': 'bold'} # For between-bar p-values

    if combined_data.empty:
        print("Error: No data available for multi-panel QoL figure.")
        return
    
    # Define the key QoL measures to include
    key_measures = [
        QOL_SCORE_COLS.get('pedsql', ''),  # PedsQL Total Score
        QOL_SCORE_COLS.get('chaq', ''),    # CHAQ Function Score
        QOL_SCORE_COLS.get('eqvision', '') # EQ-Vision Score
    ]
    
    # Filter for available measures
    available_measures = [m for m in key_measures if m and m in combined_data['QOL_Measure'].unique()]
    
    if not available_measures:
        print("Error: None of the key QoL measures have data for multi-panel figure.")
        return
    
    # Labels for the measures
    label_map = {
        QOL_SCORE_COLS.get('pedsql', ''): 'PedsQL Child Total Score',
        QOL_SCORE_COLS.get('chaq', ''): 'CHAQ Function Score',
        QOL_SCORE_COLS.get('eqvision', ''): 'EQ-Vision Score'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(available_measures), figsize=(16, 6), sharey=False, 
                            gridspec_kw={'wspace': 0.3})
    
    # If only one measure is available, make axes iterable
    if len(available_measures) == 1:
        axes = [axes]
    
    # Define color palette
    palette = ['#1f77b4', '#ff7f0e']  # Blue for JIA-Only, Orange for Uveitis
    
    # Create each panel
    for i, measure in enumerate(available_measures):
        ax = axes[i]
        measure_label = label_map.get(measure, measure.replace('_', ' ').title())
        data_plot = combined_data[combined_data['QOL_Measure'] == measure].copy()
        
        if not data_plot.empty:
            # Ensure SVI_quartile is categorical and ordered
            data_plot['SVI_quartile'] = pd.Categorical(data_plot['SVI_quartile'], 
                                                      categories=QUARTILE_ORDER, ordered=True)
            data_plot = data_plot.sort_values('SVI_quartile')
            
            # Create bar chart
            bar_width = 0.35
            x = np.arange(len(QUARTILE_ORDER))
            
            # Group data
            means = data_plot.pivot_table(
                index='SVI_quartile', 
                columns=DIAGNOSIS_GROUP_COL,
                values='Mean QOL',
                aggfunc='first'
            ).reindex(QUARTILE_ORDER)
            
            sems = data_plot.pivot_table(
                index='SVI_quartile', 
                columns=DIAGNOSIS_GROUP_COL,
                values='SEM',
                aggfunc='first'
            ).reindex(QUARTILE_ORDER)
            
            # Create bars
            ax.bar(x - bar_width/2, means.get('JIA-Only', [np.nan]*len(QUARTILE_ORDER)), 
                  bar_width, yerr=sems.get('JIA-Only', [0]*len(QUARTILE_ORDER)), 
                  label='JIA-Only', color=palette[0], capsize=4)
            
            ax.bar(x + bar_width/2, means.get('Uveitis', [np.nan]*len(QUARTILE_ORDER)), 
                  bar_width, yerr=sems.get('Uveitis', [0]*len(QUARTILE_ORDER)), 
                  label='Uveitis', color=palette[1], capsize=4)
            
            # Set labels and title
            ax.set_xlabel('SVI Quartile', **font_props_axis_labels)
            ax.set_ylabel(f'Mean {measure_label}', **font_props_axis_labels)
            ax.set_title(measure_label, **font_props_title)
            ax.set_xticks(x)
            
            # Add counts below x-ticks
            counts = data_plot.pivot_table(
                index='SVI_quartile', 
                columns=DIAGNOSIS_GROUP_COL,
                values='Count',
                aggfunc='first'
            ).reindex(QUARTILE_ORDER)
            
            # Create x-tick labels with counts
            xlabels = []
            for q in QUARTILE_ORDER:
                jia_count = counts.get('JIA-Only', pd.Series(0, index=counts.index)).get(q, 0)
                uv_count = counts.get('Uveitis', pd.Series(0, index=counts.index)).get(q, 0)
                xlabels.append(f"{q}\n(J:{jia_count}, U:{uv_count})")
            
            ax.set_xticklabels(xlabels, **font_props_tick_labels)
            
            # Calculate Q1 vs Q4 p-values for each group
            q1_vs_q4_text = ""
            for group in DIAGNOSIS_GROUP_ORDER:
                if group in data_plot[DIAGNOSIS_GROUP_COL].unique():
                    group_data = data_plot[data_plot[DIAGNOSIS_GROUP_COL] == group]
                    q1_q4_p_values = group_data[group_data['SVI_quartile'] == 'Q1']['Q1_Q4_p_value'].dropna()
                    
                    if not q1_q4_p_values.empty:
                        p_val = q1_q4_p_values.iloc[0]
                        if pd.notna(p_val):
                            p_text = f"p={p_val:.4f}"
                            if p_val < 0.001: p_text = "p<0.001"
                            elif p_val < 0.01: p_text = "p<0.01"
                            elif p_val < 0.05: p_text = f"p={p_val:.3f}*"
                            
                            q1_vs_q4_text += f"{group} Q1 vs Q4: {p_text}\n"
            
            # Add Q1 vs Q4 p-values to the plot if any are available - REPOSITIONED to top-right
            if q1_vs_q4_text:
                # Use top-right position instead of top-left to avoid overlap with high bars
                ax.text(0.98, 0.98, q1_vs_q4_text.strip(), transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), **font_props_pvalue_text)
            
            # Add statistical significance markers
            p_values = data_plot.pivot_table(
                index='SVI_quartile',
                values='Comparison p-value', 
                aggfunc='first'
            ).reindex(QUARTILE_ORDER)
            
            # Get y-axis limits
            y_min, y_max = ax.get_ylim()
            y_range = max(1.0, y_max - y_min)
            
            for j, q in enumerate(QUARTILE_ORDER):
                p_val = p_values.get(q, np.nan)
                
                # Get maximum bar height
                jia_mean = means.get('JIA-Only', pd.Series(dtype=float)).get(q, np.nan)
                uv_mean = means.get('Uveitis', pd.Series(dtype=float)).get(q, np.nan)
                
                jia_sem = sems.get('JIA-Only', pd.Series(dtype=float)).get(q, 0)
                uv_sem = sems.get('Uveitis', pd.Series(dtype=float)).get(q, 0)
                
                # Calculate text position
                text_y = max(
                    jia_mean + jia_sem if pd.notna(jia_mean) else y_min,
                    uv_mean + uv_sem if pd.notna(uv_mean) else y_min
                ) + y_range * 0.03
                
                # Add p-value text
                if pd.notna(p_val):
                    if p_val < 0.001:
                        p_text = "p<0.001***"
                    elif p_val < 0.01:
                        p_text = "p<0.01**"
                    elif p_val < 0.05:
                        p_text = "p<0.05*"
                    else:
                        p_text = f"p={p_val:.3f}"
                    
                    ax.text(j, text_y, p_text, ha='center', va='bottom', **font_props_stat_text)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
        else:
            # No data available for this measure
            ax.text(0.5, 0.5, f'No data for {measure_label}', 
                   ha='center', va='center', transform=ax.transAxes, **font_props_axis_labels)
            ax.set_title(measure_label, **font_props_title)
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Diagnosis Group', 
              loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=2, frameon=True, prop=font_props_legend)
    
    # Add overall title
    fig.suptitle('SVI Impact on Key Quality of Life (QoL) Measures', fontsize=16, y=0.98, fontname='Times New Roman')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(output_dir, "Figure3_SVI_QoL_MultiPanel.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-panel QoL figure to {output_path}")
    return output_path

# --- Main Execution ---
def main():
    print("--- SVI vs Multiple QOL Measures Analysis ---")

    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
             original_header = f.readline().strip().split(',')
             header_names = _deduplicate_columns(original_header)
        df = pd.read_csv(INPUT_FILE, low_memory=False, header=0, names=header_names)
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. Calculate SVI
    df = calculate_svi_total_and_quartiles(df, SVI_COLS, SVI_TOTAL_COL, SVI_QUARTILE_COL)

    # 2. Calculate QOL scores
    available_qol_scores = []
    
    # 2.1 PedsQL Score
    print("\n=== Calculating PedsQL Scores ===")
    pedsql_domain_cols = []
    for domain, patterns in PEDSQL_PATTERNS.items():
        score_col_name = calculate_pedsql_domain_score(df, domain, patterns, PEDSQL_CHILD_SUFFIX)
        if score_col_name:
            pedsql_domain_cols.append(score_col_name)
    
    if pedsql_domain_cols:
        pedsql_score_col = QOL_SCORE_COLS['pedsql']
        print(f"\nCalculating PedsQL Total Score ({pedsql_score_col})...")
        df[pedsql_score_col] = df[pedsql_domain_cols].mean(axis=1, skipna=True)
        valid_qol_count = df[pedsql_score_col].notna().sum()
        print(f"Calculated {pedsql_score_col} for {valid_qol_count} patients.")
        if valid_qol_count > 0:
            print(f"Overall {pedsql_score_col} summary:\n{df[pedsql_score_col].describe()}")
            available_qol_scores.append(pedsql_score_col)
    
    # 2.2 CHAQ Score
    chaq_score_col = calculate_chaq_score(df, CHAQ_PATTERN)
    if chaq_score_col:
        available_qol_scores.append(chaq_score_col)
    
    # 2.3 EQ-Vision Score
    eqvision_score_col = calculate_eqvision_score(df, EQVISION_PREFIX)
    if eqvision_score_col:
        available_qol_scores.append(eqvision_score_col)
    
    # 2.4 Pain Slider Score
    pain_score_col = calculate_pain_slider_score(df, PAIN_SLIDER_PATTERN)
    if pain_score_col:
        available_qol_scores.append(pain_score_col)
    
    # 2.5 Function Slider Score
    function_score_col = calculate_function_slider_score(df, FUNCTION_SLIDER_PATTERN)
    if function_score_col:
        available_qol_scores.append(function_score_col)
    
    if not available_qol_scores:
        print("Error: No QOL scores could be calculated. Exiting.")
        return
    
    print("\n=== Available QOL Scores for Analysis ===")
    summary_csv_paths = {} # Store paths for the summary plot trend data
    all_regression_results = {} # Store regression results
    for score in available_qol_scores:
        count = df[score].notna().sum()
        print(f"- {score}: {count} patients with data")

    # 3. Determine Diagnosis Group
    print(f"\nDetermining diagnosis groups (JIA-Only, Uveitis)...")
    df[DIAGNOSIS_GROUP_COL] = df[DX_CODE_COL].apply(lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS))
    print(f"Diagnosis group distribution:\n{df[DIAGNOSIS_GROUP_COL].value_counts()}")

    # 4. Analyze Each QOL Score
    combined_trend_data = [] # Collect data for trend plots
    for qol_col in available_qol_scores:
        print(f"\n\n{'='*80}")
        print(f"=== ANALYZING {qol_col.upper()} ===")
        print(f"{'='*80}")
        
        # 4.1 Filter for this specific QOL score analysis
        analysis_df = df[
            df[qol_col].notna() &
            (df[SVI_QUARTILE_COL] != 'Unknown') &
            df[DIAGNOSIS_GROUP_COL].isin(GROUPS_TO_ANALYZE)
        ].copy()
        
        # Create QOL-specific output subdirectory
        qol_output_dir = os.path.join(OUTPUT_DIR, qol_col)
        os.makedirs(qol_output_dir, exist_ok=True)
        
        print(f"Dataset for {qol_col} analysis includes {len(analysis_df)} patients.")
        if len(analysis_df) == 0:
            print(f"No patients with {qol_col} data. Skipping analysis.")
            continue
        
        print(f"Distribution in {qol_col} analysis set:\n{analysis_df[DIAGNOSIS_GROUP_COL].value_counts()}")
        print(f"SVI Quartile distribution in {qol_col} set:\n{analysis_df[SVI_QUARTILE_COL].value_counts().sort_index()}")

        # 4.2 Run Analysis per Group for this QOL score
        group_regression_results = {} # Temporarily store results for this QOL measure
        for group in GROUPS_TO_ANALYZE:
            df_subset = analysis_df[analysis_df[DIAGNOSIS_GROUP_COL] == group].copy()
            if not df_subset.empty:
                 # Run analysis and capture regression results
                 reg_result = run_qol_analysis_for_group(
                     df_subset=df_subset, 
                     group_name=group, 
                     qol_col=qol_col, 
                     svi_total_col=SVI_TOTAL_COL, 
                     svi_quartile_col=SVI_QUARTILE_COL, 
                     output_dir=qol_output_dir
                 )
                 if reg_result:
                    all_regression_results[(qol_col, group)] = reg_result # Store globally
            else:
                print(f"\n--- Skipping analysis for Group: {group} (0 patients) ---")
                
        # 4.3 Create Combined Comparison Chart for this QOL score
        summary_csv_path = os.path.join(qol_output_dir, f"Dual_Comparison_{qol_col}_comparison_summary.csv")
        create_qol_dual_bar_chart(
            df_analysis=analysis_df, 
            group_col=DIAGNOSIS_GROUP_COL, 
            quartile_col=SVI_QUARTILE_COL, 
            qol_col=qol_col, 
            output_dir=qol_output_dir
        )
        # Collect data from the generated summary CSV for trend plots
        try:
             df_trend = pd.read_csv(summary_csv_path)
             df_trend['QOL_Measure'] = qol_col
             combined_trend_data.append(df_trend)
        except Exception as e:
             print(f"Warning: Could not read {summary_csv_path} for trend data: {e}")

    # 5. Create Final Revised Summary Plot
    if combined_trend_data:
        final_trend_df = pd.concat(combined_trend_data, ignore_index=True)
        create_revised_summary_plot(final_trend_df, all_regression_results, OUTPUT_DIR)
        
        # 6. Create Multi-panel QoL Figure (Figure 3)
        create_multi_panel_qol_figure(final_trend_df, OUTPUT_DIR)
    else:
         print("Warning: No trend data collected, skipping final summary plots generation.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main() 