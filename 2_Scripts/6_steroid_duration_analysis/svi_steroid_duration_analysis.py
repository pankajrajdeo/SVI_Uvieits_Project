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
warnings.simplefilter(action='ignore', category=UserWarning) # For seaborn/matplotlib tick label warnings

# --- Configuration ---
INPUT_FILE = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv"
OUTPUT_DIR = "svi_steroid_duration_analysis"
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

STEROID_TREATMENT_COL = 'cmeyetrt (list distinct)' # col 43
MEDICATION_NAME_COL = 'medication name (list distinct)' # col 538
# SIMPLE_GENERIC_NAME_COL = 'simple_generic_name (list distinct)' # col 537 - Alternative if needed

START_DATE_COL = 'eye drop start date (list distinct)' # col 39
END_DATE_COL = 'eye drop end date (list distinct)' # col 40

DX_CODE_COL = 'dx code (list distinct)' # col 502

# Output Columns
HAS_STEROID_COL = 'has_steroid_drops'
DURATION_COL = 'steroid_duration_days'
DIAGNOSIS_GROUP_COL = 'Diagnosis_Group'

# Keywords and Patterns
STEROID_KEYWORDS = ['steroi', 'prednisolone', 'difluprednate', 'fluorometholone', 'dexamethasone', 'loteprednol']
EYE_CONTEXT_KEYWORDS = ['ophth', 'eye', ' op ', 'ocul', ' oc ', 'opti', 'oint', 'ung', 'gtt'] # Added ointment/gtt

JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']

# Analysis Groups
GROUPS_TO_ANALYZE = ['JIA-Only', 'Any Uveitis']

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
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
        df[quartile_col] = 'Unknown' # Assign string directly if calculation fails
        return df

    df[total_col] = df[component_mean_cols].mean(axis=1, skipna=True)
    print(f"Calculated total SVI for {df[total_col].notna().sum()} patients.")

    # Initialize column as object type first to allow 'Unknown'
    df[quartile_col] = pd.Series(dtype=object)
    
    non_na_svi = df[total_col].dropna()
    if len(non_na_svi) >= 4:  # Need at least 4 values for quartiles
        try:
            # Calculate quartiles
            quartile_values = np.percentile(non_na_svi, [25, 50, 75])
            q1_val, q2_val, q3_val = quartile_values
            
            # Create result series (initialized with None)
            result = pd.Series([None] * len(df), index=df.index, dtype='object')
            
            # Assign quartiles
            result[non_na_svi[non_na_svi <= q1_val].index] = "Q1"
            result[non_na_svi[(non_na_svi > q1_val) & (non_na_svi <= q2_val)].index] = "Q2"
            result[non_na_svi[(non_na_svi > q2_val) & (non_na_svi <= q3_val)].index] = "Q3"
            result[non_na_svi[non_na_svi > q3_val].index] = "Q4"
            
            df[quartile_col] = result
            
            print(f"Calculated SVI quartiles. Distribution:\n{df[quartile_col].value_counts().sort_index()}")
            
            # Convert to Categorical *after* potential split, adding 'Unknown'
            df[quartile_col] = pd.Categorical(
                df[quartile_col],
                categories=QUARTILE_ORDER + ['Unknown'],
                ordered=False # Order doesn't matter once 'Unknown' is added
            )
            
        except ValueError as e:
            print(f"Warning: Could not compute SVI quartiles: {e}. Assigning Unknown.")
            # If quartile calculation fails, ensure the column exists and can accept 'Unknown'
            df[quartile_col] = pd.Categorical(
                pd.Series('Unknown', index=df.index),
                categories=QUARTILE_ORDER + ['Unknown'],
                ordered=False
            )
    else:
        print(f"Warning: Not enough non-NA SVI values to compute quartiles (need at least 4, found {len(non_na_svi)}). Assigning Unknown.")
        df[quartile_col] = pd.Categorical(
             pd.Series('Unknown', index=df.index),
             categories=QUARTILE_ORDER + ['Unknown'],
             ordered=False
         )

    # Now fillna should work as 'Unknown' is a valid category
    df[quartile_col] = df[quartile_col].fillna('Unknown')
    
    if df[quartile_col].eq('Unknown').all() and not df[total_col].isna().all():
         print("Warning: Failed to calculate SVI quartiles, but total SVI was calculated.")
         
    return df

def identify_steroid_drops(row, steroid_col, med_col, steroid_kws, eye_kws):
    """Checks for steroid eye drops based on dedicated column and general meds column with context."""
    # 1. Check dedicated steroid eye drop treatment column
    if pd.notna(row[steroid_col]):
        treatments = str(row[steroid_col]).lower()
        if any(kw in treatments for kw in steroid_kws):
            return True
            
    # 2. Check general medication column (if not found above)
    if pd.notna(row[med_col]):
        meds_str = str(row[med_col]).lower()
        # Check each medication individually for steroid + eye context
        for med in meds_str.split(';'):
            med_lower = med.strip()
            if not med_lower: continue
            has_steroid_kw = any(kw in med_lower for kw in steroid_kws)
            has_eye_kw = any(kw in med_lower for kw in eye_kws)
            if has_steroid_kw and has_eye_kw:
                return True
                
    return False

def parse_dates(date_str):
    """Parses a potentially semi-colon separated date string into a list of datetime objects."""
    if pd.isna(date_str): return []
    dates = []
    for part in str(date_str).split(';'):
        part = part.strip()
        if not part: continue
        try:
            # Attempt standard YYYY-MM-DD first
            dates.append(datetime.strptime(part, '%Y-%m-%d'))
        except ValueError:
            # Add other potential format parsers here if needed
            # e.g., try: dates.append(datetime.strptime(part, '%m/%d/%Y'))
            # except ValueError: print(f"Warning: Could not parse date '{part}'")
            pass # Ignore unparseable dates for now
    return sorted(list(set(dates)))

def calculate_steroid_duration(row, has_steroid_col, start_col, end_col):
    """Calculates duration in days between earliest start and latest end date."""
    if not row[has_steroid_col]:
        return np.nan
        
    start_dates = parse_dates(row[start_col])
    end_dates = parse_dates(row[end_col])
    
    if not start_dates or not end_dates:
        return np.nan
        
    earliest_start = min(start_dates)
    latest_end = max(end_dates)
    
    if latest_end < earliest_start:
        # Handle case where end date is before start date (data issue?)
        return 0 # Or np.nan depending on desired handling
        
    duration = (latest_end - earliest_start).days
    # Add 1 if duration should be inclusive of start/end day, otherwise use difference
    # duration = (latest_end - earliest_start).days + 1 
    return duration

def check_codes(dx_string, jia_pattern, uveitis_patterns):
    """Classifies patients into 'JIA-Only', 'Any Uveitis', or 'Other'."""
    if pd.isna(dx_string): return 'Other'
    codes = [code.strip() for code in str(dx_string).split(';') if code.strip()]
    
    has_jia = any(re.search(jia_pattern, code, re.IGNORECASE) for code in codes)
    has_uveitis = any(any(re.search(uv_pattern, code, re.IGNORECASE) for code in codes) for uv_pattern in uveitis_patterns)
    
    if has_jia and not has_uveitis:
        return 'JIA-Only'
    elif has_uveitis: # Includes patients who might also have JIA
        return 'Any Uveitis'
    elif has_jia: # Fallback if somehow missed by JIA-Only (e.g., no Uveitis codes checked but JIA present)
         return 'JIA-Only'
    else:
        return 'Other' # Neither specific JIA nor Uveitis codes found

def run_analysis_for_group(df_subset, group_name, duration_col, svi_total_col, svi_quartile_col, output_dir):
    """Performs analysis (stats, plots) for a specific diagnosis group."""
    print(f"\n--- Analyzing Group: {group_name} ({len(df_subset)} patients) ---")
    output_prefix = os.path.join(output_dir, group_name.replace(' ', '_'))

    # --- Descriptive Statistics ---
    print(f"Descriptive Statistics for {duration_col}:")
    duration_stats_desc = df_subset[duration_col].describe()
    print(duration_stats_desc)
    duration_stats_desc.to_csv(f"{output_prefix}_duration_descriptive_stats.csv")

    # --- Analysis by SVI Quartile ---
    print(f"\nAnalyzing {duration_col} by SVI quartile...")
    svi_duration_stats = df_subset.groupby(svi_quartile_col, observed=False)[duration_col].agg(['count', 'mean', 'median', 'std']).reset_index()
    # Ensure all quartiles are present for consistent tables/plots, fill missing with 0/NaN
    svi_duration_stats = svi_duration_stats.set_index(svi_quartile_col).reindex(QUARTILE_ORDER).reset_index()
    # Fill NaN counts with 0 for label generation
    svi_duration_stats['count'] = svi_duration_stats['count'].fillna(0).astype(int)
    # Fill NaN means for label generation
    svi_duration_stats['mean'] = svi_duration_stats['mean'].fillna(np.nan) 
    print(svi_duration_stats)
    svi_duration_stats.to_csv(f"{output_prefix}_duration_by_svi_stats.csv", index=False)

    # --- Statistical Testing: ANOVA --- 
    print("\nPerforming statistical tests (ANOVA)...")
    present_quartiles = [q for q in QUARTILE_ORDER if q in df_subset[svi_quartile_col].unique()]
    groups = [df_subset[duration_col][df_subset[svi_quartile_col] == q] for q in present_quartiles]
    groups = [g.dropna() for g in groups if len(g.dropna()) >= 2] # Need at least 2 data points per group

    anova_f_stat, anova_p_value = np.nan, np.nan
    anova_p_text = "ANOVA: Insufficient data"
    if len(groups) >= 2: # Need at least 2 groups for ANOVA
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
    
    # --- Test Q1 vs Q4 specifically ---
    print("\nPerforming Q1 vs Q4 comparison...")
    q1_vs_q4_text = "Q1 vs Q4: Insufficient data"
    if 'Q1' in present_quartiles and 'Q4' in present_quartiles:
        q1_data = df_subset[df_subset[svi_quartile_col] == 'Q1'][duration_col].dropna()
        q4_data = df_subset[df_subset[svi_quartile_col] == 'Q4'][duration_col].dropna()
        
        if len(q1_data) >= 2 and len(q4_data) >= 2:
            try:
                # Use Mann-Whitney U test (non-parametric)
                u_stat, p_val = stats.mannwhitneyu(q1_data, q4_data, alternative='two-sided')
                p_str = f"p={p_val:.4f}"
                if p_val < 0.001: p_str = "p<0.001"
                elif p_val < 0.05: p_str += "*"
                q1_vs_q4_text = f"Q1 vs Q4: U={u_stat:.1f}, {p_str}"
                print(f"Q1 vs Q4 comparison for {group_name}: {q1_vs_q4_text}")
            except Exception as e:
                print(f"Could not perform Q1 vs Q4 comparison for {group_name}: {e}")
                q1_vs_q4_text = "Q1 vs Q4: Error"
        else:
            print(f"Q1 vs Q4 comparison skipped for {group_name}: Not enough data.")
    else:
        print(f"Q1 vs Q4 comparison skipped for {group_name}: One or both quartiles missing.")
        
    # --- Create Plot Labels with N counts and Mean values ---
    plot_labels = {}
    for q in present_quartiles:
         count = svi_duration_stats.loc[svi_duration_stats[svi_quartile_col] == q, 'count'].iloc[0]
         mean_val = svi_duration_stats.loc[svi_duration_stats[svi_quartile_col] == q, 'mean'].iloc[0]
         mean_str = f"{mean_val:.1f}" if pd.notna(mean_val) else "NaN"
         plot_labels[q] = f"{q}\n(N={count})\nMean={mean_str}" # Add mean to label
    
    ordered_plot_labels = [plot_labels.get(q, f"{q}\n(N=0)\nMean=NaN") for q in present_quartiles]

    # --- Visualization: Box Plot --- 
    if not present_quartiles:
        print("No SVI quartiles present in this group for plotting.")
        # Skip regression if no plots
        print(f"Linear Regression Skipped for {group_name}: No data for plots.") 
        return 
        
    plt.figure(figsize=(10, 6)) # Adjusted size for single plot
    
    # Box Plot
    ax = plt.gca()
    sns.boxplot(x=svi_quartile_col, y=duration_col, data=df_subset, order=present_quartiles, palette='viridis', ax=ax)
    ax.set_title(f'Steroid Duration by SVI Quartile ({group_name})', fontsize=14)
    ax.set_xlabel('SVI Quartile', fontsize=12)
    ax.set_ylabel('Duration (Days)', fontsize=12)
    # Set labels with N counts and Mean
    ax.set_xticks(range(len(present_quartiles)))
    ax.set_xticklabels(ordered_plot_labels, fontsize=9) # Adjust font size if needed
    
    # Add statistical test results in a box
    stats_text = f"{anova_p_text}\n{q1_vs_q4_text}"
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(f"{output_prefix}_duration_by_svi_boxplot.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to {output_prefix}_duration_by_svi_boxplot.png")

    # --- Statistical Testing: Regression --- 
    print("\nPerforming statistical tests (Linear Regression)...")
    regression_data = df_subset[[svi_total_col, duration_col]].dropna()
    
    reg_slope, reg_intercept, reg_r_value, reg_p_value, reg_std_err = np.nan, np.nan, np.nan, np.nan, np.nan
    if len(regression_data) >= 10: # Require minimum number of points for regression
        X = regression_data[svi_total_col]
        y = regression_data[duration_col]
        X = sm.add_constant(X) # Add intercept term
        try:
            model = sm.OLS(y, X).fit()
            print(f"Linear Regression Summary ({group_name}):")
            print(model.summary())
            
            # Extract key results for reporting
            reg_slope = model.params[svi_total_col]
            reg_intercept = model.params['const']
            reg_r_value = np.sqrt(model.rsquared)
            reg_p_value = model.pvalues[svi_total_col]
            
            # Visualization
            plt.figure(figsize=(8, 6))
            sns.regplot(x=svi_total_col, y=duration_col, data=regression_data, scatter_kws={'alpha':0.5})
            # Include p-value in the title
            p_val_reg_str = f"p={reg_p_value:.4f}"
            if reg_p_value < 0.001: p_val_reg_str = "p<0.001"
            elif reg_p_value < 0.05: p_val_reg_str += "*"
            plt.title(f'SVI Total vs Steroid Duration ({group_name})\nR² = {model.rsquared:.2f}, {p_val_reg_str}', fontsize=14)
            plt.xlabel('SVI Total Score', fontsize=12)
            plt.ylabel('Steroid Duration (Days)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{output_prefix}_svi_duration_regression.png", dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"Saved regression plot to {output_prefix}_svi_duration_regression.png")
            
        except Exception as e:
             print(f"Could not perform Linear Regression for {group_name}: {e}")
    else:
        print(f"Linear Regression Skipped for {group_name}: Insufficient data (need >= 10 points with SVI and duration). Found {len(regression_data)}.")

def create_dual_bar_chart(df_analysis, group_col, quartile_col, duration_col, output_dir):
    """Creates a dual bar chart comparing duration across SVI quartiles for JIA-Only vs Any Uveitis."""
    print("\n--- Creating Dual Bar Chart (JIA-Only vs Any Uveitis) ---")
    output_prefix = os.path.join(output_dir, "Dual_Comparison")

    if df_analysis.empty:
        print("Skipping dual bar chart: No analysis data available.")
        return

    # Calculate means, counts, stds per group and quartile
    # Use observed=False to potentially include groups even if not present in the final subset
    summary_stats = df_analysis.groupby([group_col, quartile_col], observed=False)[duration_col].agg(['mean', 'count', 'std']).unstack(group_col)

    # Reindex to ensure all quartiles are present and fill missing stats
    idx = pd.Index(QUARTILE_ORDER, name=quartile_col)
    summary_stats = summary_stats.reindex(idx)
    
    # Check which groups are actually present in the columns
    present_groups = [group for group in GROUPS_TO_ANALYZE if ('mean', group) in summary_stats.columns]
    print(f"Groups present in analysis data for comparison: {present_groups}")
    if len(present_groups) == 0:
         print("Skipping dual bar chart: No target groups found in the data after aggregation.")
         return
    
    # Calculate SEM = std / sqrt(count)
    sem = summary_stats.get('std', pd.DataFrame()) / np.sqrt(summary_stats.get('count', pd.DataFrame()))

    # Get means, counts, stderr, handling missing groups
    means = summary_stats.get('mean', pd.DataFrame()).fillna(0)
    counts = summary_stats.get('count', pd.DataFrame()).fillna(0).astype(int)
    stderr = sem.fillna(0)

    # Ensure columns exist for both groups, even if empty
    for group in GROUPS_TO_ANALYZE:
        if group not in means.columns:
            means[group] = 0
        if group not in counts.columns:
            counts[group] = 0
        if group not in stderr.columns:
             stderr[group] = 0

    # Create figure
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    bar_width = 0.35
    x = np.arange(len(QUARTILE_ORDER))

    # Plot bars only for present groups, using placeholders for missing ones
    rects1 = ax.bar(x - bar_width/2, means.get('JIA-Only', 0), bar_width, 
                    yerr=stderr.get('JIA-Only', 0), label='JIA-Only', capsize=4, color='steelblue')
    rects2 = ax.bar(x + bar_width/2, means.get('Any Uveitis', 0), bar_width, 
                    yerr=stderr.get('Any Uveitis', 0), label='Any Uveitis', capsize=4, color='darkorange')

    # Add labels, title, legend
    ax.set_xlabel('SVI Quartile', fontsize=12)
    ax.set_ylabel('Mean Steroid Duration (Days)', fontsize=12)
    ax.set_title('Mean Steroid Duration by SVI Quartile: JIA-Only vs Any Uveitis', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTILE_ORDER)
    ax.legend()

    # Add N counts to labels
    def create_label(q):
        # Safely get counts using .get with default 0
        n1 = counts.get('JIA-Only', pd.Series(0, index=counts.index)).get(q, 0)
        n2 = counts.get('Any Uveitis', pd.Series(0, index=counts.index)).get(q, 0)
        return f"{q}\n(JIA:N={n1}, Uveitis:N={n2})"
    
    ax.set_xticklabels([create_label(q) for q in QUARTILE_ORDER])

    # Add p-values from t-tests
    y_min, y_max = ax.get_ylim()
    y_range = max(1, y_max - y_min) # Ensure y_range is at least 1 to avoid division by zero if all bars are 0
    
    p_values_comparison = {}
    for i, q in enumerate(QUARTILE_ORDER):
        # Get data only if groups exist
        group1_data = df_analysis[(df_analysis[group_col] == 'JIA-Only') & (df_analysis[quartile_col] == q)][duration_col].dropna() if 'JIA-Only' in present_groups else pd.Series(dtype=float)
        group2_data = df_analysis[(df_analysis[group_col] == 'Any Uveitis') & (df_analysis[quartile_col] == q)][duration_col].dropna() if 'Any Uveitis' in present_groups else pd.Series(dtype=float)

        p_val_text = "NA"
        # Only run t-test if both groups are present AND have sufficient data
        if 'JIA-Only' in present_groups and 'Any Uveitis' in present_groups and len(group1_data) >= 2 and len(group2_data) >= 2:
            try:
                t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False) # Welch's t-test
                p_values_comparison[q] = p_val
                if p_val < 0.001: p_val_text = "p<0.001"
                elif p_val < 0.01: p_val_text = "p<0.01"
                elif p_val < 0.05: p_val_text = f"p={p_val:.3f}*"
                else: p_val_text = f"p={p_val:.3f}"
            except Exception:
                p_val_text = "Error"
                p_values_comparison[q] = np.nan
        elif len(present_groups) < 2:
             p_val_text = "Single Group"
             p_values_comparison[q] = np.nan
        else: # Both groups conceptually exist, but one or both have N < 2
             p_val_text = "Insufficient N"
             p_values_comparison[q] = np.nan

        # Position p-value text
        # Use .get() for safe access to potentially missing group data
        bar1_height = means.get('JIA-Only', pd.Series(0, index=means.index)).get(q, 0) + stderr.get('JIA-Only', pd.Series(0, index=stderr.index)).get(q, 0)
        bar2_height = means.get('Any Uveitis', pd.Series(0, index=means.index)).get(q, 0) + stderr.get('Any Uveitis', pd.Series(0, index=stderr.index)).get(q, 0)
        text_y = max(bar1_height, bar2_height) + y_range * 0.03
        ax.text(i, text_y, p_val_text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout slightly for labels
    plt.savefig(f"{output_prefix}_duration_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved dual comparison plot to {output_prefix}_duration_comparison.png")

    # Save comparison summary data
    # Create DataFrame manually to ensure correct structure even with missing groups
    summary_list = []
    for q in QUARTILE_ORDER:
         for group in GROUPS_TO_ANALYZE:
              mean_val = means.get(group, pd.Series(np.nan, index=means.index)).get(q, np.nan)
              count_val = counts.get(group, pd.Series(0, index=counts.index)).get(q, 0)
              sem_val = stderr.get(group, pd.Series(np.nan, index=stderr.index)).get(q, np.nan)
              summary_list.append({
                  quartile_col: q,
                  group_col: group,
                  'Mean Duration': mean_val,
                  'Count': count_val,
                  'SEM': sem_val,
                  'Comparison p-value': p_values_comparison.get(q, np.nan)
              })
    final_summary = pd.DataFrame(summary_list)
    final_summary = final_summary.sort_values(by=[group_col, quartile_col])
    
    final_summary.to_csv(f"{output_prefix}_duration_comparison_summary.csv", index=False)
    print(f"Saved comparison summary data to {output_prefix}_duration_comparison_summary.csv")

# --- Main Execution ---
def main():
    print("--- SVI vs Steroid Eye Drop Duration Analysis ---")

    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        # Handle potential duplicate columns during load
        # Read header first
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
             original_header = f.readline().strip().split(',')
             header_names = _deduplicate_columns(original_header)
        # Read data using potentially modified names
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

    # 2. Identify Steroid Eye Drops
    print(f"\nIdentifying steroid eye drops...")
    df[HAS_STEROID_COL] = df.apply(identify_steroid_drops, 
                                  args=(STEROID_TREATMENT_COL, MEDICATION_NAME_COL, STEROID_KEYWORDS, EYE_CONTEXT_KEYWORDS), 
                                  axis=1)
    steroid_count = df[HAS_STEROID_COL].sum()
    print(f"Identified {steroid_count} patients with potential steroid eye drops.")

    # 3. Calculate Duration
    print(f"\nCalculating steroid eye drop duration...")
    df[DURATION_COL] = df.apply(calculate_steroid_duration, 
                                args=(HAS_STEROID_COL, START_DATE_COL, END_DATE_COL), 
                                axis=1)
    valid_duration_count = df[DURATION_COL].notna().sum()
    print(f"Calculated duration for {valid_duration_count} patients with steroid drops and valid dates.")
    # Optional: Describe duration for those who have it
    if valid_duration_count > 0:
        print(f"Overall duration summary (days):\n{df[DURATION_COL].describe()}")


    # 4. Determine Diagnosis Group
    print(f"\nDetermining diagnosis groups (JIA-Only, Any Uveitis)...")
    df[DIAGNOSIS_GROUP_COL] = df[DX_CODE_COL].apply(lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS))
    print(f"Diagnosis group distribution:\n{df[DIAGNOSIS_GROUP_COL].value_counts()}")

    # --- Debugging Print Statements --- 
    print(f"\nChecking counts before final filter:")
    # Use observed=True if SVI_half is categorical and might have unused categories
    print(df.groupby(DIAGNOSIS_GROUP_COL, observed=True)[HAS_STEROID_COL].value_counts().unstack(fill_value=0))
    
    # Check how many JIA-Only steroid users have valid duration
    jia_only_steroid_users = df[(df[DIAGNOSIS_GROUP_COL] == 'JIA-Only') & df[HAS_STEROID_COL]].copy() # Use copy to avoid SettingWithCopyWarning
    print(f"\nJIA-Only patients identified with steroid drops: {len(jia_only_steroid_users)}")
    if len(jia_only_steroid_users) > 0:
        print(f"  - With calculable duration ({DURATION_COL}.notna()): {jia_only_steroid_users[DURATION_COL].notna().sum()}")
        print(f"  - With valid SVI Quartile ({SVI_QUARTILE_COL} != 'Unknown'): {jia_only_steroid_users[SVI_QUARTILE_COL].ne('Unknown').sum()}")
        
        # --- Added Check: Print date columns for these 12 patients --- 
        print("\n  Raw Date Column Contents for JIA-Only Steroid Users:")
        print(jia_only_steroid_users[[START_DATE_COL, END_DATE_COL]].to_string())
        # --- End Added Check ---
    # --- End Debugging --- 
    
    # 5. Filter for Analysis
    print(f"\nFiltering data for analysis...")
    analysis_df = df[
        df[HAS_STEROID_COL] &
        df[DURATION_COL].notna() &
        (df[DURATION_COL] >= 0) & # Exclude potential negative durations if calculation logic allows
        (df[SVI_QUARTILE_COL] != 'Unknown') &
        df[DIAGNOSIS_GROUP_COL].isin(GROUPS_TO_ANALYZE)
    ].copy()
    
    print(f"Final dataset for analysis includes {len(analysis_df)} patients.")
    if len(analysis_df) == 0:
        print("No patients remaining after filtering. Cannot proceed with analysis.")
        return
        
    print(f"Distribution in final analysis set:\n{analysis_df[DIAGNOSIS_GROUP_COL].value_counts()}")
    print(f"SVI Quartile distribution in final set:\n{analysis_df[SVI_QUARTILE_COL].value_counts().sort_index()}")

    # 6. Run Analysis per Group
    for group in GROUPS_TO_ANALYZE:
        df_subset = analysis_df[analysis_df[DIAGNOSIS_GROUP_COL] == group].copy()
        if not df_subset.empty:
             run_analysis_for_group(
                 df_subset=df_subset, 
                 group_name=group, 
                 duration_col=DURATION_COL, 
                 svi_total_col=SVI_TOTAL_COL, 
                 svi_quartile_col=SVI_QUARTILE_COL, 
                 output_dir=OUTPUT_DIR
             )
        else:
            print(f"\n--- Skipping analysis for Group: {group} (0 patients) ---")

    # 7. Create Combined Comparison Chart
    create_dual_bar_chart(
        df_analysis=analysis_df, 
        group_col=DIAGNOSIS_GROUP_COL, 
        quartile_col=SVI_QUARTILE_COL, 
        duration_col=DURATION_COL, 
        output_dir=OUTPUT_DIR
    )

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main() 