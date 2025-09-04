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
import warnings

# Ignore specific warnings if necessary (e.g., from seaborn)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
INPUT_FILE = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv"
OUTPUT_DIR = "/Users/rajlq7/Desktop/SVI/svi_jia_subtype_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 300

# Column Names
DX_CODE_COL = 'dx code (list distinct)' # Used for both JIA subtype and JIA/Uveitis grouping
SVI_COLS = [
    'svi_socioeconomic (list distinct)',
    'svi_household_comp (list distinct)',
    'svi_housing_transportation (list distinct)',
    'svi_minority (list distinct)'
]
SVI_TOTAL_COL = 'SVI_total'
SVI_QUARTILE_COL = 'SVI_quartile'
JIA_SUBTYPE_COL = 'JIA_Subtype' # The final mapped subtype column
DIAGNOSIS_GROUP_COL = 'Diagnosis_Group' # 'JIA-Only' or 'Any Uveitis'

# Source columns for JIA Subtype (based on generate_table1_comparison.py)
JIA_SUBTYPE_PRIMARY_COL = 'ilar_code_display_value (list distinct)'
JIA_SUBTYPE_ALT1_COL = 'ILAR code'
JIA_SUBTYPE_ALT2_COL = 'jia subtype'
COMBINED_JIA_SOURCE_COL = 'combined_jia_subtype_source' # Intermediate column after combining sources

# Diagnosis Grouping Patterns
JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']

# JIA Subtype Mapping (Based on logic from generate_table1_comparison.py)
JIA_SUBTYPE_MAP = {
    'Oligoarticular': r'M08\.4',
    'Polyarticular RF+': r'M08\.0',
    'Polyarticular RF-': r'M08\.2',
    'Systemic': r'M08\.1', # Note: generate_table1 used M08.1, confirm if this is correct Systemic code
    'Psoriatic': r'M08\.3', # Note: generate_table1 used M08.3, confirm if this is correct Psoriatic code
    'Enthesitis-related': r'M08\.8', # Note: generate_table1 used M08.8, confirm if this is correct ERA code
    'Undifferentiated': r'M08\.9'
}
# Order for plotting consistency (using final consolidated categories)
JIA_SUBTYPE_ORDER = [
    'Oligoarticular, Persistent', 'Oligoarticular, Extended', 'Oligoarticular (unspecified)',
    'Polyarticular RF-', 'Polyarticular RF+',
    'Systemic JIA', 'Psoriatic Arthritis', 'Enthesitis Related Arthritis',
    'Undifferentiated Arthritis', 'Other'
]


# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

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
        # Handle potential non-string values first
        str_value = str(value)
        values = [float(v.strip()) for v in str_value.split(';') if v.strip()]
        return np.mean(values) if values else np.nan
    except ValueError: # Catch errors if conversion to float fails
        # print(f"Warning: Could not parse SVI value '{value}' to float.")
        return np.nan
    except Exception as e:
        # print(f"Warning: Error parsing SVI value '{value}': {e}")
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
        df[quartile_col] = np.nan
        return df

    # Calculate Total SVI (mean of component means)
    df[total_col] = df[component_mean_cols].mean(axis=1)
    print(f"Calculated total SVI for {df[total_col].notna().sum()} patients.")

    # Calculate Quartiles based on Total SVI
    non_na_svi = df[total_col].dropna()
    if len(non_na_svi) >= 2: # Changed from >= 4
        try:
            quartiles = pd.qcut(non_na_svi, 2, labels=["Q1", "Q2"]) # Changed from 4 to 2
            # Ensure the result series matches the original dataframe's index
            df[quartile_col] = quartiles.reindex(df.index)
            print(f"Calculated SVI quartiles. Distribution:\n{df[quartile_col].value_counts().sort_index()}")
        except ValueError as e:
            print(f"Warning: Could not compute SVI quartiles: {e}. Assigning NaN.")
            df[quartile_col] = np.nan
    else:
        print("Warning: Not enough non-NA SVI values to compute quartiles (need at least 2). Assigning NaN.") # Updated message
        df[quartile_col] = np.nan
        
    if df[quartile_col].isna().all() and not df[total_col].isna().all():
         print("Warning: Failed to calculate SVI quartiles, but total SVI was calculated.")
         
    return df


def map_jia_subtype(source_string):
    """Maps a combined JIA subtype source string to final categories based on keywords/codes and precedence."""
    if pd.isna(source_string):
        return 'Other' # Treat NaN as Other

    # Normalize the source string: lower case, split by semicolon
    terms = [term.strip().lower() for term in str(source_string).split(';') if term.strip()]
    if not terms:
        return 'Other'

    # Define keywords for each subtype (lowercase)
    # Ensure M codes have the dot escaped for potential regex use, though simple `in` is used here.
    mapping_keywords = {
        'Oligoarticular, Persistent': ['oligoarticular persistent', 'persistent oligoarticular'],
        'Oligoarticular, Extended': ['oligoarticular extended', 'extended oligoarticular'],
        'Polyarticular RF+': ['polyarticular rf+', 'polyarticular rheumatoid factor positive', 'm08.0'],
        'Polyarticular RF-': ['polyarticular rf-', 'polyarticular rheumatoid factor negative', 'm08.2'],
        'Psoriatic Arthritis': ['psoriatic arthritis', 'psoriatic', 'm08.3'],
        'Enthesitis Related Arthritis': ['enthesitis related arthritis', 'enthesitis-related', 'm08.8'],
        'Systemic JIA': ['systemic jia', 'systemic', 'm08.1'],
        'Undifferentiated Arthritis': ['undifferentiated arthritis', 'undifferentiated', 'm08.9'],
        'Oligoarticular (unspecified)': ['oligoarticular', 'm08.4'], # Check this last among Oligo types
        'Other': ['other']
    }

    # Check order reflects the desired precedence
    check_order = [
        'Oligoarticular, Persistent',
        'Oligoarticular, Extended',
        'Polyarticular RF+', # Check before RF-
        'Polyarticular RF-',
        'Psoriatic Arthritis',
        'Enthesitis Related Arthritis',
        'Systemic JIA',
        'Undifferentiated Arthritis',
        'Oligoarticular (unspecified)', # Check general oligo last
        'Other'
    ]

    # Iterate through the check_order (precedence)
    for subtype in check_order:
        keywords = mapping_keywords[subtype]
        # Check if ANY term in the source string matches ANY keyword for this subtype
        for term in terms:
            if any(keyword in term for keyword in keywords):
                return subtype # Return the first matched subtype based on precedence

    # Fallback if no keywords match at all after checking all terms according to precedence
    return 'Other'

def apply_final_subtype_grouping(raw_subtype_string):
    """Maps raw subtype strings to the final consolidated categories."""
    if pd.isna(raw_subtype_string):
        return 'Other' # Map NAs to Other
        
    raw_lower = str(raw_subtype_string).strip().lower()
    if not raw_lower:
        return 'Other' # Map empty strings to Other

    # Apply grouping rules (ensure checks happen in a logical order if needed)
    if 'persistent oligoarticular' in raw_lower or raw_lower == 'oligoarticular persistent':
        return 'Oligoarticular, Persistent'
    elif 'extended oligoarticular' in raw_lower or raw_lower == 'oligoarticular extended':
        return 'Oligoarticular, Extended'
    elif 'oligoarticular (unspecified)' in raw_lower or raw_lower == 'oligoarticular': # Check general Oligo after specific ones
        return 'Oligoarticular (unspecified)'
    elif 'polyarticular rf (+)' in raw_lower or raw_lower == 'polyarticular rf+':
        return 'Polyarticular RF+'
    elif 'polyarticular rf (-)' in raw_lower or raw_lower == 'polyarticular rf-':
        return 'Polyarticular RF-'
    elif 'systemic jia' in raw_lower or raw_lower == 'systemic':
        return 'Systemic JIA'
    elif 'psoriatic arthritis' in raw_lower or raw_lower == 'psoriatic':
        return 'Psoriatic Arthritis'
    elif 'enthesitis related arthritis' in raw_lower or raw_lower == 'enthesitis-related':
        return 'Enthesitis Related Arthritis'
    elif 'undifferentiated arthritis' in raw_lower or raw_lower == 'undifferentiated':
        return 'Undifferentiated Arthritis'
    elif raw_lower == 'other':
         return 'Other'
    else:
        # If a raw string doesn't match any specific rule, classify as 'Other'
        # This handles cases like M-codes if they weren't caught by keywords
        # print(f"Warning: Raw subtype '{raw_subtype_string}' mapped to Other.") 
        return 'Other'

def check_codes(dx_string, jia_pattern, uveitis_patterns):
    """Classifies patients into 'JIA-Only', 'Any Uveitis', or 'Other'."""
    if pd.isna(dx_string): return 'Other'
    codes = [code.strip() for code in str(dx_string).split(';') if code.strip()]
    
    # Check for specific JIA codes (M08.x) - Note: this check might be redundant if map_jia_subtype handles it
    # has_jia = any(re.match(jia_pattern, code) for code in codes)
    
    # Simplified check: Does the mapped subtype indicate JIA?
    mapped_subtype = map_jia_subtype(dx_string) # Call the updated mapping function
    is_jia = mapped_subtype != 'Other' # Assume any mapped subtype means JIA
    
    # Check for any Uveitis codes (H20, H30, H44)
    has_uveitis = any(any(re.match(uv_pattern, code, re.IGNORECASE) for code in codes) for uv_pattern in uveitis_patterns)
    
    if is_jia and not has_uveitis:
        return 'JIA-Only'
    elif has_uveitis: # Includes patients who might also have JIA
        return 'Any Uveitis'
    elif is_jia: # Fallback if somehow missed by JIA-Only
         return 'JIA-Only'
    else:
        return 'Other' # Neither JIA nor Uveitis codes found

# --- Helper function to parse semicolon-separated strings ---
def parse_semicolon_string_column(df, col_name):
    """Parses a column containing semicolon-separated strings into lists of strings."""
    if col_name not in df.columns:
        print(f"Warning: Column '{col_name}' not found for parsing.")
        return pd.Series([None] * len(df), index=df.index)
        
    def parse_string(value):
        if pd.isna(value):
            return None
        try:
            # Split, strip whitespace, and filter out empty strings
            return [item.strip() for item in str(value).split(';') if item.strip()]
        except Exception:
            return None # Return None if any error occurs during parsing
            
    return df[col_name].apply(parse_string)


# --- Function to combine JIA subtypes from multiple source columns ---
def combine_jia_subtypes(df, primary_col, alt_col1, alt_col2, output_col):
    """
    Combines JIA subtype data from three columns with priority:
    primary_col > alt_col1 > alt_col2. Stores the first found value in output_col.
    """
    print(f"\n--- Combining JIA subtypes from multiple columns into '{output_col}' ---")
    
    # Check which columns exist in the dataframe
    primary_exists = primary_col in df.columns
    alt1_exists = alt_col1 in df.columns
    alt2_exists = alt_col2 in df.columns
    
    if not (primary_exists or alt1_exists or alt2_exists):
        print("Warning: None of the JIA subtype source columns found. Cannot combine.")
        df[output_col] = None
        return df
    
    # Initialize the output column with None
    df[output_col] = None
    df['_source_col'] = None # Track which column provided the value
    
    temp_cols = [] # Keep track of temporary columns to drop later

    # Function to get the first non-empty item from a parsed list
    def get_first_item(parsed_list):
        return parsed_list[0] if isinstance(parsed_list, list) and len(parsed_list) > 0 else None

    # --- Process Primary Column --- 
    if primary_exists:
        parsed_col_name = 'primary_jia_parsed'
        first_item_col_name = 'primary_jia'
        temp_cols.extend([parsed_col_name, first_item_col_name])
        
        df[parsed_col_name] = parse_semicolon_string_column(df, primary_col)
        df[first_item_col_name] = df[parsed_col_name].apply(get_first_item)
        
        # Assign primary column values first (highest priority)
        mask = df[first_item_col_name].notna()
        df.loc[mask, output_col] = df.loc[mask, first_item_col_name]
        df.loc[mask, '_source_col'] = primary_col
        primary_count = mask.sum()
        print(f"Found {primary_count} potential subtypes from primary '{primary_col}'")
    else:
        print(f"Primary column '{primary_col}' not found.")
        primary_count = 0

    # --- Process Alt Column 1 --- 
    alt1_fill_count = 0
    if alt1_exists:
        parsed_col_name = 'alt1_jia_parsed'
        first_item_col_name = 'alt1_jia'
        temp_cols.extend([parsed_col_name, first_item_col_name])
        
        df[parsed_col_name] = parse_semicolon_string_column(df, alt_col1)
        df[first_item_col_name] = df[parsed_col_name].apply(get_first_item)
        
        # Only fill in where output is still None
        mask = (df[output_col].isna()) & (df[first_item_col_name].notna())
        df.loc[mask, output_col] = df.loc[mask, first_item_col_name]
        df.loc[mask, '_source_col'] = alt_col1
        alt1_fill_count = mask.sum()
        print(f"Filled {alt1_fill_count} missing subtypes from alt1 '{alt_col1}'")
    else:
        print(f"Alternative column 1 '{alt_col1}' not found.")

    # --- Process Alt Column 2 --- 
    alt2_fill_count = 0
    if alt2_exists:
        parsed_col_name = 'alt2_jia_parsed'
        first_item_col_name = 'alt2_jia'
        temp_cols.extend([parsed_col_name, first_item_col_name])
        
        df[parsed_col_name] = parse_semicolon_string_column(df, alt_col2)
        df[first_item_col_name] = df[parsed_col_name].apply(get_first_item)
        
        # Only fill in where output is still None
        mask = (df[output_col].isna()) & (df[first_item_col_name].notna())
        df.loc[mask, output_col] = df.loc[mask, first_item_col_name]
        df.loc[mask, '_source_col'] = alt_col2
        alt2_fill_count = mask.sum()
        print(f"Filled {alt2_fill_count} missing subtypes from alt2 '{alt_col2}'")
    else:
        print(f"Alternative column 2 '{alt_col2}' not found.")

    # Report final coverage
    total_filled = df[output_col].notna().sum()
    print(f"Combined source column '{output_col}' has {total_filled}/{len(df)} values filled.")
    print(f"Source distribution:\n{df['_source_col'].value_counts(dropna=False)}")
    
    # Clean up temporary columns
    df = df.drop(columns=[col for col in temp_cols if col in df.columns] + ['_source_col'], errors='ignore')
    
    return df

# --- Analysis Functions ---

def run_analysis_for_group(df_group, group_name, subtype_col, svi_col, svi_quartile_col, subtype_order, output_dir):
    """Runs analysis for a specific group (e.g., JIA-Only, Any Uveitis)."""
    print(f"\n--- Analysis for {group_name} ---")
    if df_group.empty:
        print(f"No data for {group_name}. Skipping analysis.")
        return None, None

    # Ensure SVI Quartile column is present
    if svi_quartile_col not in df_group.columns:
        print(f"Error: SVI Quartile column '{svi_quartile_col}' not found in {group_name} data.")
        return None, None

    # --- Summary Statistics by SVI Quartile ---
    print(f"\nSummary statistics for {group_name} by SVI Quartile:")
    summary_stats = df_group.groupby(svi_quartile_col)[svi_col].agg(['count', 'mean', 'median', 'std']).reset_index()
    print(summary_stats)
    summary_stats_file = os.path.join(output_dir, f"{group_name}_SVI_Summary_Stats.csv")
    summary_stats.to_csv(summary_stats_file, index=False)
    print(f"Saved SVI summary stats for {group_name} to {summary_stats_file}")

    # --- JIA Subtype Distribution by SVI Quartile (Counts and Percentages) ---
    print(f"\nJIA Subtype distribution for {group_name} by SVI Quartile:")
    # Ensure subtype_order is used for consistent row ordering, and fill missing with 0
    subtype_counts = pd.crosstab(df_group[subtype_col], df_group[svi_quartile_col])
    subtype_counts = subtype_counts.reindex(subtype_order, fill_value=0) # Use official order
    # Ensure quartiles are in Q1, Q2 order if present
    quartile_order_present = [q for q in ['Q1', 'Q2'] if q in subtype_counts.columns]
    subtype_counts = subtype_counts[quartile_order_present] if quartile_order_present else subtype_counts # Reorder columns

    print("Counts:")
    print(subtype_counts)

    # Calculate percentages within each SVI quartile (column-wise)
    subtype_percentages = subtype_counts.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0, axis=0).round(1)
    print("\nPercentages (% within each SVI Quartile):")
    print(subtype_percentages)

    # Save to CSV
    counts_file = os.path.join(output_dir, f"{group_name}_Subtype_Counts_by_SVI.csv")
    subtype_counts.to_csv(counts_file)
    print(f"Saved subtype counts for {group_name} to {counts_file}")
    percentages_file = os.path.join(output_dir, f"{group_name}_Subtype_Percentages_by_SVI.csv")
    subtype_percentages.to_csv(percentages_file)
    print(f"Saved subtype percentages for {group_name} to {percentages_file}")

    # --- Statistical Tests for SVI Total between Quartiles ---
    print(f"\nStatistical tests for SVI Total ({svi_col}) for {group_name}:")
    quartile_order = ['Q1', 'Q2'] # Updated to Q1, Q2
    
    svi_by_quartile = [df_group[df_group[svi_quartile_col] == q][svi_col].dropna() for q in quartile_order]
    svi_by_quartile_valid = [g for g in svi_by_quartile if len(g) > 1]

    # ANOVA (or Kruskal-Wallis if assumptions not met - here simplified to ANOVA for 2 groups it's like t-test)
    if len(svi_by_quartile_valid) >= 2:
        try:
            f_val, p_anova = stats.f_oneway(*svi_by_quartile_valid)
            print(f"ANOVA across SVI quartiles: F={f_val:.3f}, p={p_anova:.4f}")
        except Exception as e:
            print(f"ANOVA Error: {e}")
    else:
        print("ANOVA: Not enough groups with sufficient data.")

    # T-test (Q1 vs Q2) - if both quartiles exist and have data
    if 'Q1' in df_group[svi_quartile_col].unique() and 'Q2' in df_group[svi_quartile_col].unique():
        group1_svi = df_group[df_group[svi_quartile_col] == 'Q1'][svi_col].dropna()
        group2_svi = df_group[df_group[svi_quartile_col] == 'Q2'][svi_col].dropna()
        if len(group1_svi) > 1 and len(group2_svi) > 1:
            try:
                t_stat, p_ttest = stats.ttest_ind(group1_svi, group2_svi, equal_var=False) # Welch's t-test
                print(f"T-test (Q1 vs Q2 SVI Total): t={t_stat:.3f}, p={p_ttest:.4f}")
            except Exception as e:
                print(f"T-test (Q1 vs Q2) Error: {e}")
        else:
            print("T-test (Q1 vs Q2): Not enough data in Q1 and/or Q2 for SVI Total comparison.")
    else:
        print("T-test (Q1 vs Q2): Q1 and/or Q2 not present in the data for SVI Total comparison.")

    # --- Chi-squared test for JIA subtype distribution across SVI quartiles ---
    # Only run if there are at least two quartiles with data and multiple subtypes
    if subtype_counts.shape[1] >=2 and subtype_counts.shape[0] > 1:
        # Filter out rows (subtypes) that are all zero, as they can cause issues with chi2
        subtype_counts_filtered = subtype_counts[(subtype_counts.T != 0).any()]
        if subtype_counts_filtered.shape[0] > 1: # Need at least 2 subtypes after filtering
            try:
                chi2, p_chi2, dof, expected = stats.chi2_contingency(subtype_counts_filtered)
                print(f"\nChi-squared test for JIA subtype distribution across SVI quartiles ({group_name}):")
                print(f"Chi2={chi2:.3f}, p={p_chi2:.4f}, DOF={dof}")
            except ValueError as e:
                print(f"Chi-squared test error for {group_name} (possibly due to low counts): {e}")
        else:
            print(f"Chi-squared test for {group_name}: Not enough subtype diversity after filtering zero rows.")
    else:
        print(f"Chi-squared test for {group_name}: Not enough SVI quartiles with data or subtype diversity.")

    # --- Visualizations ---
    # Stacked Bar Chart of JIA Subtype by SVI Quartile
    if not subtype_percentages.empty:
        try:
            subtype_percentages.T.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')
            plt.title(f'JIA Subtype Distribution by SVI Quartile ({group_name})')
            plt.xlabel('SVI Quartile')
            plt.ylabel('Percentage of Patients (%)')
            plt.xticks(rotation=0)
            plt.legend(title='JIA Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            plot_file = os.path.join(output_dir, f"{group_name}_Subtype_by_SVI_Quartile_StackedBar.png")
            plt.savefig(plot_file, dpi=DPI)
            plt.close()
            print(f"Saved stacked bar chart for {group_name} to {plot_file}")
        except Exception as e:
            print(f"Error generating stacked bar chart for {group_name}: {e}")
    else:
        print(f"Skipping stacked bar chart for {group_name} as there are no subtype percentages.")

    # Boxplot/Violin plot of SVI Total by JIA Subtype (if enough data)
    # Filter out subtypes with very few data points for robustness of plot
    min_subtype_samples = 5 
    df_plot_svi_subtype = df_group.groupby(subtype_col).filter(lambda x: len(x) >= min_subtype_samples)
    # Ensure the subtype_col is treated as categorical with the correct order for plotting
    if not df_plot_svi_subtype.empty and subtype_col in df_plot_svi_subtype and svi_col in df_plot_svi_subtype:
        # Get unique subtypes present in the filtered data, maintaining overall order if possible
        ordered_subtypes_in_plot = [s for s in subtype_order if s in df_plot_svi_subtype[subtype_col].unique()]
        if not ordered_subtypes_in_plot:
            ordered_subtypes_in_plot = df_plot_svi_subtype[subtype_col].unique().tolist()
        
        try:
            plt.figure(figsize=(14, 8))
            # sns.boxplot(x=subtype_col, y=svi_col, data=df_plot_svi_subtype, order=ordered_subtypes_in_plot, palette='viridis')
            sns.violinplot(x=subtype_col, y=svi_col, data=df_plot_svi_subtype, order=ordered_subtypes_in_plot, palette='viridis', inner='quartile', cut=0)
            plt.title(f'SVI Total Score Distribution by JIA Subtype ({group_name}, N >= {min_subtype_samples} per subtype)')
            plt.xlabel('JIA Subtype')
            plt.ylabel(f'{SVI_TOTAL_COL} Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{group_name}_SVI_by_Subtype_Violin.png")
            plt.savefig(plot_file, dpi=DPI)
            plt.close()
            print(f"Saved SVI by Subtype violin plot for {group_name} to {plot_file}")

            # Statistical test (ANOVA/Kruskal-Wallis) for SVI differences among subtypes
            subtype_groups_svi = [df_plot_svi_subtype[df_plot_svi_subtype[subtype_col] == s][svi_col].dropna() for s in ordered_subtypes_in_plot]
            subtype_groups_svi_valid = [g for g in subtype_groups_svi if len(g) > 1]
            if len(subtype_groups_svi_valid) >= 2:
                # Check for normality and variance homogeneity if being rigorous, else default to Kruskal for non-parametric
                # For simplicity, using Kruskal-Wallis as subtype distributions might not be normal
                try:
                    h_stat, p_kruskal = stats.kruskal(*subtype_groups_svi_valid)
                    print(f"Kruskal-Wallis test for SVI Total across JIA Subtypes ({group_name}): H={h_stat:.3f}, p={p_kruskal:.4f}")
                except Exception as e:
                    print(f"Kruskal-Wallis test error ({group_name}): {e}")
            else:
                print(f"Not enough JIA subtype groups with sufficient data for Kruskal-Wallis test ({group_name}).")

        except Exception as e:
            print(f"Error generating SVI by subtype violin plot for {group_name}: {e}")
    else:
        print(f"Skipping SVI by subtype violin plot for {group_name} due to insufficient data after filtering.")
        
    return subtype_counts, subtype_percentages

# Now let's update the create_dual_bar_chart function to add p-values and fix N value positioning
def create_dual_bar_chart(df, subtype_col, svi_col, group_col, subtype_order, output_dir):
    """Creates a dual bar chart comparing JIA-Only and Any Uveitis by JIA subtype with p-values."""
    print("\n--- Creating Dual Bar Chart (JIA-Only vs Any Uveitis) ---")
    output_prefix = os.path.join(output_dir, "Dual_Comparison")

    # Ensure required columns exist
    if df.empty or subtype_col not in df.columns or svi_col not in df.columns or group_col not in df.columns:
        print("Skipping Dual Bar Chart: Missing required columns.")
        return

    # Filter for valid SVI and required groups
    groups_to_include = ['JIA-Only', 'Any Uveitis']
    analysis_df = df.dropna(subset=[svi_col]).copy()
    analysis_df = analysis_df[analysis_df[group_col].isin(groups_to_include)].copy()

    if analysis_df.empty:
        print("Skipping Dual Bar Chart: No valid data after filtering.")
        return

    # Get subtypes present in either group
    plot_order = [s for s in subtype_order if s in analysis_df[subtype_col].unique()]
    if not plot_order:
        print("Skipping Dual Bar Chart: No common subtypes found.")
        return

    print(f"Creating dual bar chart for {len(plot_order)} JIA subtypes...")

    try:
        # Prepare data: calculate mean, std error for each group and subtype combination
        # Use a pivot table to get means
        means = pd.pivot_table(
            analysis_df,
            values=svi_col,
            index=subtype_col,
            columns=group_col,
            aggfunc='mean'
        ).reindex(plot_order)

        # Calculate standard errors
        counts = pd.pivot_table(
            analysis_df,
            values=svi_col,
            index=subtype_col,
            columns=group_col,
            aggfunc='count'
        ).reindex(plot_order)

        stds = pd.pivot_table(
            analysis_df,
            values=svi_col,
            index=subtype_col,
            columns=group_col,
            aggfunc='std'
        ).reindex(plot_order)

        # Standard error = std / sqrt(n)
        stderr = stds / np.sqrt(counts)

        # Create figure with appropriate size
        plt.figure(figsize=(max(12, len(plot_order)*1.5), 8))

        # Set up the bar positions
        bar_width = 0.35
        x = np.arange(len(plot_order))

        # Plot bars in the specified order: JIA-Only then Any Uveitis
        ax = plt.gca()
        bars1 = ax.bar(x - bar_width/2, means['JIA-Only'].fillna(0),
                       bar_width, yerr=stderr['JIA-Only'].fillna(0),
                       label='JIA-Only', capsize=4,
                       color='steelblue')

        bars2 = ax.bar(x + bar_width/2, means['Any Uveitis'].fillna(0),
                       bar_width, yerr=stderr['Any Uveitis'].fillna(0),
                       label='Any Uveitis', capsize=4,
                       color='darkorange')

        # Calculate p-values for each subtype comparison between JIA-Only and Any Uveitis
        p_values = {}
        # p_value_texts = {} # This variable was defined but not used, can be removed

        # Calculate y range for positioning
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min

        for i, subtype in enumerate(plot_order):
            # Get data for this subtype
            subtype_df = analysis_df[analysis_df[subtype_col] == subtype]
            jia_only_data = subtype_df[subtype_df[group_col] == 'JIA-Only'][svi_col].values
            uveitis_data = subtype_df[subtype_df[group_col] == 'Any Uveitis'][svi_col].values

            # Only calculate p-value if both groups have data
            if len(jia_only_data) > 0 and len(uveitis_data) > 0:
                try:
                    # Perform t-test
                    t_stat, p_val = stats.ttest_ind(jia_only_data, uveitis_data, equal_var=False, nan_policy='omit') # Added nan_policy
                    p_values[subtype] = p_val

                    # Format p-value text
                    if p_val < 0.0001:
                        p_text = "p<0.0001"
                    elif p_val < 0.001:
                        p_text = "p<0.001"
                    elif p_val < 0.01:
                        p_text = "p<0.01"
                    elif p_val < 0.05:
                        p_text = f"p={p_val:.3f}*"
                    else:
                        p_text = f"p={p_val:.3f}"

                    # p_value_texts[subtype] = p_text # Corresponding to unused variable

                    # Get max height between the two bars plus error for text positioning
                    jia_y_pos = 0
                    if 'JIA-Only' in means.columns and subtype in means.index and not pd.isna(means.loc[subtype, 'JIA-Only']):
                         jia_y_pos = means.loc[subtype, 'JIA-Only']
                         if 'JIA-Only' in stderr.columns and subtype in stderr.index and not pd.isna(stderr.loc[subtype, 'JIA-Only']):
                              jia_y_pos += stderr.loc[subtype, 'JIA-Only']
                    
                    uveitis_y_pos = 0
                    if 'Any Uveitis' in means.columns and subtype in means.index and not pd.isna(means.loc[subtype, 'Any Uveitis']):
                         uveitis_y_pos = means.loc[subtype, 'Any Uveitis']
                         if 'Any Uveitis' in stderr.columns and subtype in stderr.index and not pd.isna(stderr.loc[subtype, 'Any Uveitis']):
                              uveitis_y_pos += stderr.loc[subtype, 'Any Uveitis']

                    max_height = max(jia_y_pos, uveitis_y_pos)
                    
                    # Add p-value text above the bars
                    ax.text(i, max_height + y_range*0.05, p_text, ha='center', va='bottom', size='x-small', fontweight='bold')

                except Exception as e:
                    print(f"Warning: Could not calculate p-value for {subtype}: {e}")

        # Create custom x-tick labels with N values for each group
        custom_labels = []
        for i, subtype_val in enumerate(plot_order): # Renamed subtype to subtype_val to avoid conflict
            # Handle potential NaN values by checking first and using 0 as default
            jia_count = 0
            if subtype_val in counts.index and 'JIA-Only' in counts.columns and not pd.isna(counts.loc[subtype_val, 'JIA-Only']):
                jia_count = int(counts.loc[subtype_val, 'JIA-Only'])

            uveitis_count = 0
            if subtype_val in counts.index and 'Any Uveitis' in counts.columns and not pd.isna(counts.loc[subtype_val, 'Any Uveitis']):
                uveitis_count = int(counts.loc[subtype_val, 'Any Uveitis'])

            custom_labels.append(f"{subtype_val}\\n(JIA:N={jia_count}, UV:N={uveitis_count})") # Shortened Uveitis label

        # Add labels, title, legend, etc.
        plt.xlabel('JIA Subtype (Consolidated)')
        plt.ylabel('Mean SVI Score')
        plt.title('Mean SVI by JIA Subtype: JIA-Only vs Any Uveitis Comparison')

        # Set custom x-tick labels
        plt.xticks(x, custom_labels, rotation=45, ha='right')
        plt.legend()

        # Add mean values above each bar
        for idx, bar in enumerate(bars1): # Changed i to idx
            height = bar.get_height()
            # Ensure index is within bounds for plot_order and means/stderr
            if height > 0 and idx < len(plot_order) and plot_order[idx] in means.index and 'JIA-Only' in means.columns:
                mean_value = means.loc[plot_order[idx], 'JIA-Only']
                if not pd.isna(mean_value):
                     ax.text(bar.get_x() + bar.get_width()/2., height + y_range*0.01,
                             f"{mean_value:.2f}", ha='center', va='bottom', size='small')

        for idx, bar in enumerate(bars2): # Changed i to idx
            height = bar.get_height()
            if height > 0 and idx < len(plot_order) and plot_order[idx] in means.index and 'Any Uveitis' in means.columns:
                mean_value = means.loc[plot_order[idx], 'Any Uveitis']
                if not pd.isna(mean_value):
                    ax.text(bar.get_x() + bar.get_width()/2., height + y_range*0.01,
                            f"{mean_value:.2f}", ha='center', va='bottom', size='small')

        # Adjust layout - increase bottom padding to avoid overlap
        plt.subplots_adjust(bottom=0.35) # Increased bottom margin
        plt.savefig(f"{output_prefix}_JIA_vs_Uveitis_SVI_Comparison.png", dpi=DPI)
        plt.close()
        print("Saved Dual Bar Chart comparing JIA-Only vs Any Uveitis with p-values.")

        # Generate summary table including p-values
        summary_data = {
            'JIA Subtype': plot_order,
            'JIA-Only Mean SVI': [means.loc[sub, 'JIA-Only'] if sub in means.index and 'JIA-Only' in means.columns else np.nan for sub in plot_order],
            'JIA-Only Count': [counts.loc[sub, 'JIA-Only'] if sub in counts.index and 'JIA-Only' in counts.columns else 0 for sub in plot_order],
            'Any Uveitis Mean SVI': [means.loc[sub, 'Any Uveitis'] if sub in means.index and 'Any Uveitis' in means.columns else np.nan for sub in plot_order],
            'Any Uveitis Count': [counts.loc[sub, 'Any Uveitis'] if sub in counts.index and 'Any Uveitis' in counts.columns else 0 for sub in plot_order],
        }
        summary = pd.DataFrame(summary_data)
        
        # Add p-values to the summary
        summary['p-value'] = [p_values.get(sub, np.nan) for sub in plot_order]

        summary_file = f"{output_prefix}_JIA_vs_Uveitis_Summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Saved comparison summary with p-values to {summary_file}")

    except Exception as e:
        print(f"Error generating dual bar chart: {e}")
        import traceback
        traceback.print_exc()

# --- Main Execution ---
def main():
    print("--- SVI by JIA Subtype Analysis ---")

    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        # Read header separately to handle duplicates
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            original_header = next(reader)
            header_names = _deduplicate_columns(original_header)
        # Read the full data with potentially deduplicated names
        df = pd.read_csv(INPUT_FILE, low_memory=False, header=0, names=header_names, encoding='utf-8')
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for essential SVI columns
    essential_svi_cols = SVI_COLS
    missing_svi_cols = [col for col in essential_svi_cols if col not in df.columns]
    if missing_svi_cols:
        print(f"Error: Essential SVI columns missing: {missing_svi_cols}")
        return
        
    # Check if at least one JIA subtype source column exists
    jia_source_cols = [JIA_SUBTYPE_PRIMARY_COL, JIA_SUBTYPE_ALT1_COL, JIA_SUBTYPE_ALT2_COL]
    if not any(col in df.columns for col in jia_source_cols):
        print(f"Error: None of the JIA subtype source columns found: {jia_source_cols}")
        return
    
    # Also need DX_CODE_COL for JIA/Uveitis grouping
    if DX_CODE_COL not in df.columns:
        print(f"Error: DX Code column '{DX_CODE_COL}' needed for JIA/Uveitis grouping is missing.")
        return

    # 1. Calculate SVI Total and Quartiles
    df = calculate_svi_total_and_quartiles(df, SVI_COLS, SVI_TOTAL_COL, SVI_QUARTILE_COL)

    # 2. Combine JIA subtype source columns
    df = combine_jia_subtypes(df, JIA_SUBTYPE_PRIMARY_COL, JIA_SUBTYPE_ALT1_COL, JIA_SUBTYPE_ALT2_COL, COMBINED_JIA_SOURCE_COL)

    # Use the combined source column as the effective subtype column for analysis
    EFFECTIVE_JIA_SUBTYPE_COL = COMBINED_JIA_SOURCE_COL 
    print(f"Using combined source column '{EFFECTIVE_JIA_SUBTYPE_COL}' for subtype analysis.")
    print(f"Raw subtype distribution (overall):\n{df[EFFECTIVE_JIA_SUBTYPE_COL].value_counts()}")

    # 4. Apply Diagnosis Grouping (JIA-Only vs Any Uveitis)
    # Important: Use the ORIGINAL DX_CODE_COL for this grouping
    print("Applying diagnosis grouping using original DX codes...")
    # Re-apply check_codes based on original DX_CODE_COL as subtype mapping changed
    df[DIAGNOSIS_GROUP_COL] = df[DX_CODE_COL].apply(lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS))
    groups_to_analyze = ['JIA-Only', 'Any Uveitis']
    print(f"Diagnosis group distribution:\n{df[DIAGNOSIS_GROUP_COL].value_counts()}")

    # 5. Filter Data for Analysis
    # Keep rows that have a valid SVI score, a non-empty combined JIA source string, AND belong to one of the target diagnosis groups
    analysis_ready_df = df[
        df[SVI_TOTAL_COL].notna() &
        df[EFFECTIVE_JIA_SUBTYPE_COL].notna() & # Check if the combined source is not null
        (df[EFFECTIVE_JIA_SUBTYPE_COL] != '') &   # Check if the combined source is not empty string
        df[DIAGNOSIS_GROUP_COL].isin(groups_to_analyze)
    ].copy()
    
    print(f"\nInitial patient count for analysis (valid SVI, Non-empty Combined JIA Source, JIA-Only/Any Uveitis): {len(analysis_ready_df)}")
    if analysis_ready_df.empty:
        print("Error: No patients remaining after filtering. Cannot proceed with subtype analysis.")
        return

    # Apply the final consolidation mapping
    print(f"Applying final subtype consolidation mapping...")
    analysis_ready_df[JIA_SUBTYPE_COL] = analysis_ready_df[EFFECTIVE_JIA_SUBTYPE_COL].apply(apply_final_subtype_grouping)
    
    # Filter out any rows that mapped to 'Other' during final consolidation
    analysis_ready_df = analysis_ready_df[analysis_ready_df[JIA_SUBTYPE_COL] != 'Other'].copy()
    print(f"Final patient count after consolidation and removing 'Other': {len(analysis_ready_df)}")
    print(f"Final Consolidated Subtype Distribution:\n{analysis_ready_df[JIA_SUBTYPE_COL].value_counts()}")
    
    if analysis_ready_df.empty:
        print("Error: No patients remaining after final subtype consolidation. Cannot proceed.")
        return

    # 6. Run Analysis per Group
    for group in groups_to_analyze:
        df_subset = analysis_ready_df[analysis_ready_df[DIAGNOSIS_GROUP_COL] == group].copy()
        run_analysis_for_group(
            df_subset,
            group_name=group,
            subtype_col=JIA_SUBTYPE_COL, # Use the FINAL mapped subtype column
            svi_col=SVI_TOTAL_COL,
            svi_quartile_col=SVI_QUARTILE_COL,
            subtype_order=JIA_SUBTYPE_ORDER, # Use the order for consolidated categories
            output_dir=OUTPUT_DIR
        )

    # 7. Create a dual bar chart comparing JIA-Only vs Any Uveitis
    create_dual_bar_chart(
        analysis_ready_df,
        subtype_col=JIA_SUBTYPE_COL,
        svi_col=SVI_TOTAL_COL,
        group_col=DIAGNOSIS_GROUP_COL,
        subtype_order=JIA_SUBTYPE_ORDER,
        output_dir=OUTPUT_DIR
    )

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main() 