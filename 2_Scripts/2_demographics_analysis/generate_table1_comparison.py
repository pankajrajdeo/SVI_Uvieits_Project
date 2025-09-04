import pandas as pd
import numpy as np
from scipy import stats
import re
import ast
import traceback # Ensure traceback is imported for error handling
import sys # Import sys for output redirection
from collections import Counter
import warnings

# Suppress specific warnings if needed (e.g., Chi-squared approximation)
warnings.filterwarnings("ignore", message="Chi-squared approximation may be incorrect")

# --- Configuration ---
FILE_PATH = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv"
DX_CODE_COL = 'dx code (list distinct)' # Column with semicolon-separated ICD codes

# Diagnosis Code Patterns
JIA_CODE_PATTERN = r'M08' 
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']

# Columns for Analysis (Ensure exact names from your CSV)
GENDER_COL = 'gender'
JIA_SUBTYPE_COL = 'ilar_code_display_value (list distinct)' # Note the trailing space
JIA_SUBTYPE_COL_ALT1 = 'ILAR code' # Alternative column 1 for JIA subtype
JIA_SUBTYPE_COL_ALT2 = 'jia subtype' # Alternative column 2 for JIA subtype
RACE_COL = 'race'
ETHNICITY_COL = 'ethnicity'
AGE_CALC_COL = 'age calc' # Interpreted as age at visit/enrollment
ARTHRITIS_ONSET_YEAR_COL = 'date of arth onsety'
UVEITIS_ONSET_YEAR_COL = 'date of uv onsety'
UVEITIS_LATERALITY_COL = 'which eyes are involved '
MEDICATION_GENERIC_NAME_COL = 'simple_generic_name (list distinct)'
UVEITIS_DIAGNOSIS_YEAR_COL = 'date of uv diagnosisy'
LAB_COMPONENT_COL = 'lab component name (list distinct)'
LAB_VALUE_COL = 'measure value (list distinct)'
ANA_COL = 'ana_display_value (list distinct)'
# DOB_COL = 'DOB' # DOB column exists but not used for age calculation in this final version
# --- ADDED: Define measure name column --- 
MEASURE_NAME_COL = 'measure name (list distinct)'
# --- ADDED: Define measure unit column ---
MEASURE_UNIT_COL = 'measure unit (list distinct)'

BIOLOGICS_LIST = [ # List of common biologics (lowercase)
    'abatacept', 'adalimumab', 'etanercept', 'golimumab', 
    'infliximab', 'leflunomide', 'secukinumab', 'tocilizumab', 'ustekinumab'
] 

# Standard thresholds (used by interpret_value)
ESR_HIGH_THRESHOLD = 11 # mm/hr (elevated if > 11)
VITD_DEFICIENT_THRESHOLD = 19 # ng/mL (low if < 19)
VITD_INSUFFICIENT_THRESHOLD = 19 # ng/mL (set same as deficient since we only want one threshold)

# --- Helper Functions ---

def parse_and_group_by_dx(df, dx_col, jia_pattern, uveitis_patterns):
    """Parses DX codes and groups rows into JIA-U, JIA-Only, or Other."""
    if dx_col not in df.columns:
        print(f"Error: Diagnosis code column '{dx_col}' not found.")
        return None

    def check_codes(dx_string):
        if pd.isna(dx_string):
            return 'Other'
        codes = [code.strip() for code in str(dx_string).split(';')]
        has_jia = any(re.match(jia_pattern, code) for code in codes)
        has_uveitis = any(any(re.match(uv_pattern, code) for code in codes) for uv_pattern in uveitis_patterns)

        if has_jia and has_uveitis:
            return 'JIA-U'
        elif has_jia and not has_uveitis:
            return 'JIA-Only'
        elif has_uveitis:
            return 'Uveitis-Only'
        else:
            return 'Other'

    df['diagnosis_group'] = df[dx_col].apply(check_codes)
    # Filter to only keep the groups needed for the table
    df_filtered = df[df['diagnosis_group'].isin(['JIA-Only', 'JIA-U', 'Uveitis-Only'])].copy()
    print(f"Filtered to JIA-Only, JIA-U, and Uveitis-Only groups. Shape: {df_filtered.shape}")
    return df_filtered

def calculate_categorical_stats(df, group_col, value_col, metric_name):
    """Calculates counts, percentages, and p-value for a categorical variable between two groups."""
    print(f"\n--- Analyzing: {metric_name} ---")

    # Check if value column exists
    if value_col not in df.columns:
        print(f"Column '{value_col}' not found. Skipping analysis for {metric_name}.")
        return

    # Work on a copy for cleaning/analysis
    df_analysis = df[[group_col, value_col]].copy()
    df_analysis['original_value'] = df_analysis[value_col].astype(str) 
    df_analysis[value_col + '_lower'] = df_analysis[value_col].fillna('Missing').astype(str).str.strip().str.lower()
    df_analysis[value_col + '_lower'] = df_analysis[value_col + '_lower'].replace({'nan': 'Missing', '': 'Missing', 'unknown': 'Missing'})

    # Define groups
    group1_name = 'JIA-Only'
    group2_name = 'JIA-U'
    group3_name = 'Uveitis-Only'
    group1_df = df_analysis[df_analysis[group_col] == group1_name]
    group2_df = df_analysis[df_analysis[group_col] == group2_name]
    group3_df = df_analysis[df_analysis[group_col] == group3_name]
    
    total_group1 = len(group1_df)
    total_group2 = len(group2_df)
    total_group3 = len(group3_df)
    print(f"{group1_name} Total Patients: {total_group1}")
    print(f"{group2_name} Total Patients: {total_group2}")
    print(f"{group3_name} Total Patients: {total_group3}")

    # Special handling for ESR and Vitamin D
    if metric_name == 'ESR Status (Interpreted)':
        # Make filtering case-insensitive
        df_valid = df_analysis[df_analysis[value_col].astype(str).str.lower().isin(['high', 'not high'])].copy()
        n_g1 = len(df_valid[df_valid[group_col] == group1_name])
        n_g2 = len(df_valid[df_valid[group_col] == group2_name])
        n_g3 = len(df_valid[df_valid[group_col] == group3_name])
        total_n_metric = n_g1 + n_g2 + n_g3
        print(f"{metric_name} (N={total_n_metric} with valid data)")
        # --- ADDED: Print threshold info ---
        print(f" (Threshold for High: >= {ESR_HIGH_THRESHOLD} mm/hr)")
        
        # Use the filtered interpreted column for crosstab
        contingency_table = pd.crosstab(df_valid[group_col], df_valid[value_col]) 
        
    elif metric_name == 'Vitamin D Status (Interpreted)':
        # Make filtering case-insensitive
        df_valid = df_analysis[df_analysis[value_col].astype(str).str.lower().isin(['low', 'not low'])].copy()
        n_g1 = len(df_valid[df_valid[group_col] == group1_name])
        n_g2 = len(df_valid[df_valid[group_col] == group2_name])
        n_g3 = len(df_valid[df_valid[group_col] == group3_name])
        total_n_metric = n_g1 + n_g2 + n_g3
        print(f"{metric_name} (N={total_n_metric} with valid data)")
        # --- ADDED: Print threshold info ---
        print(f" (Threshold for Low: <= {VITD_DEFICIENT_THRESHOLD} ng/mL, assumes conversion from nmol/L if needed)")
        
        # Use the filtered interpreted column for crosstab
        contingency_table = pd.crosstab(df_valid[group_col], df_valid[value_col])
        
    else:
        # Regular handling for other metrics
        counts_all_g1 = group1_df[value_col + '_lower'].value_counts()
        valid_cats_g1_idx = counts_all_g1.index[~counts_all_g1.index.isin(['missing'])] 
        n_g1 = counts_all_g1.loc[valid_cats_g1_idx].sum()

        counts_all_g2 = group2_df[value_col + '_lower'].value_counts()
        valid_cats_g2_idx = counts_all_g2.index[~counts_all_g2.index.isin(['missing'])]
        n_g2 = counts_all_g2.loc[valid_cats_g2_idx].sum()

        counts_all_g3 = group3_df[value_col + '_lower'].value_counts()
        valid_cats_g3_idx = counts_all_g3.index[~counts_all_g3.index.isin(['missing'])]
        n_g3 = counts_all_g3.loc[valid_cats_g3_idx].sum()
        total_n_metric = n_g1 + n_g2 + n_g3

        print(f"{metric_name} (N={total_n_metric} with valid data)")
        df_valid = df_analysis[~df_analysis[value_col + '_lower'].isin(['missing'])]
        contingency_table = pd.crosstab(df_valid[group_col], df_valid['original_value'])

    # Special handling for JIA Subtype
    if metric_name == 'JIA Subtype':
        # Only compare JIA-Only and JIA-U groups
        contingency_table = contingency_table.loc[[group1_name, group2_name]]

    # Check if statistical testing is appropriate
    groups_with_data = sum([n_g1 > 0, n_g2 > 0, n_g3 > 0])
    if groups_with_data <= 1:
        p_value_str = 'N/A (insufficient data for comparison)'
        print("(Note: Statistical testing skipped - data available for only one group or no groups)")
    else:
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            p_value_str = f"{p:.4f}"
            if p < 0.0001: p_value_str = "<0.0001"
            p_value_str += f" (χ²={chi2:.2f}, dof={dof})"
            if np.any(expected < 5):
                print("(Warning: Expected cell count < 5 in Chi-squared test. P-value may be inaccurate.)")
        except Exception as e:
            print(f"(Error calculating p-value: {e})")
            p_value_str = 'N/A (error in calculation)'

    # Print results
    print(f"{'Category':<45} {group1_name:<20} {group2_name:<20} {group3_name:<20} P-value")
    if metric_name == 'JIA Subtype':
        print(f"{'':<45} {'(N=' + str(n_g1) + ')':<20} {'(N=' + str(n_g2) + ')':<20} {'(N=0)':<20} {p_value_str}")
    else:
        print(f"{'':<45} {'(N=' + str(n_g1) + ')':<20} {'(N=' + str(n_g2) + ')':<20} {'(N=' + str(n_g3) + ')':<20} {p_value_str}")

    # Calculate and display percentages
    for category in contingency_table.columns:
        g1_count = contingency_table.loc[group1_name, category] if group1_name in contingency_table.index else 0
        g2_count = contingency_table.loc[group2_name, category] if group2_name in contingency_table.index else 0
        g3_count = contingency_table.loc[group3_name, category] if group3_name in contingency_table.index else 0
        
        g1_perc = (g1_count / n_g1 * 100) if n_g1 > 0 else 0
        g2_perc = (g2_count / n_g2 * 100) if n_g2 > 0 else 0
        g3_perc = (g3_count / n_g3 * 100) if n_g3 > 0 else 0
        
        g1_str = f"{int(g1_count)} ({g1_perc:.1f}%)"
        g2_str = f"{int(g2_count)} ({g2_perc:.1f}%)"
        g3_str = f"{int(g3_count)} ({g3_perc:.1f}%)"
        
        print(f"  {category:<43} {g1_str:<20} {g2_str:<20} {g3_str:<20}")

    # Only show missing counts for metrics other than ESR and Vitamin D
    if metric_name not in ['ESR Status (Interpreted)', 'Vitamin D Status (Interpreted)']:
        missing_g1 = total_group1 - n_g1
        missing_g2 = total_group2 - n_g2
        missing_g3 = total_group3 - n_g3
        if missing_g1 > 0 or missing_g2 > 0 or missing_g3 > 0:
            print(f"  {'Missing/Unknown':<43} {missing_g1:<20} {missing_g2:<20} {missing_g3:<20}")

def calculate_continuous_stats(df, group_col, value_col, metric_name):
    """Calculates mean/std dev and p-value for a continuous variable between two groups."""
    print(f"\n--- Analyzing: {metric_name} ---")

    # Check if value column exists
    if value_col not in df.columns:
        print(f"Column '{value_col}' not found. Skipping analysis for {metric_name}.")
        return

    # Ensure column is numeric, coercing errors to NaN
    # Make a copy to avoid SettingWithCopyWarning
    df_copy = df[[group_col, value_col]].copy()
    df_copy[value_col] = pd.to_numeric(df_copy[value_col], errors='coerce')
    df_valid = df_copy.dropna(subset=[value_col])

    # Define groups
    group1_name = 'JIA-Only'
    group2_name = 'JIA-U'
    group3_name = 'Uveitis-Only'
    group1_data = df_valid[df_valid[group_col] == group1_name][value_col]
    group2_data = df_valid[df_valid[group_col] == group2_name][value_col]
    group3_data = df_valid[df_valid[group_col] == group3_name][value_col]

    n_g1 = len(group1_data)
    n_g2 = len(group2_data)
    n_g3 = len(group3_data)
    total_n_metric = n_g1 + n_g2 + n_g3
    
    # Get total patient counts for context
    total_group1 = len(df[df[group_col] == group1_name])
    total_group2 = len(df[df[group_col] == group2_name])
    total_group3 = len(df[df[group_col] == group3_name])
    print(f"{group1_name} Total Patients: {total_group1}")
    print(f"{group2_name} Total Patients: {total_group2}")
    print(f"{group3_name} Total Patients: {total_group3}")
    print(f"{metric_name} (N={total_n_metric} with valid numeric data)")

    # Calculate Mean/Std Dev
    mean_g1 = group1_data.mean() if n_g1 > 0 else np.nan
    std_g1 = group1_data.std() if n_g1 > 1 else 0 # Use 0 if only 1 point, NaN if 0 points
    mean_g2 = group2_data.mean() if n_g2 > 0 else np.nan
    std_g2 = group2_data.std() if n_g2 > 1 else 0
    mean_g3 = group3_data.mean() if n_g3 > 0 else np.nan
    std_g3 = group3_data.std() if n_g3 > 1 else 0

    # Calculate p-value (using Mann-Whitney U as default, robust to non-normality)
    p_value_str = 'N/A'
    if n_g1 > 0 and n_g2 > 0 and n_g3 > 0:
        try:
            stat, p = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            p_value_str = f"{p:.4f}"
            if p < 0.0001: p_value_str = "<0.0001"
        except ValueError as e: # Handles cases like zero variance if all values are identical
            print(f"(Mann-Whitney U error, likely due to identical values: {e}. Checking means...)")
            if mean_g1 == mean_g2 == mean_g3:
                 p_value_str = '1.0000' # If means are identical, no significant difference
                 print("(Means are identical, setting p=1)")
            else:
                 p_value_str = 'Error' # Genuine error
    else:
        print("(Skipping p-value: Not enough data in one or more groups)")

    # Format results
    g1_str = f"{mean_g1:.1f} ± {std_g1:.1f}" if n_g1 > 0 else "N/A"
    g2_str = f"{mean_g2:.1f} ± {std_g2:.1f}" if n_g2 > 0 else "N/A"
    g3_str = f"{mean_g3:.1f} ± {std_g3:.1f}" if n_g3 > 0 else "N/A"
    # Handle cases with only 1 data point (no std dev)
    if n_g1 == 1: g1_str = f"{mean_g1:.1f}"
    if n_g2 == 1: g2_str = f"{mean_g2:.1f}"
    if n_g3 == 1: g3_str = f"{mean_g3:.1f}"

    # Print formatted results
    print(f"{'Value':<35} {group1_name:<20} {group2_name:<20} {group3_name:<20} P-value")
    print(f"{'':<35} {'(N=' + str(n_g1) + ')':<20} {'(N=' + str(n_g2) + ')':<20} {'(N=' + str(n_g3) + ')':<20} {p_value_str}")
    print(f"  Mean ± SD                 {g1_str:<20} {g2_str:<20} {g3_str:<20}")

    # Report missing count based on total group size
    missing_g1 = total_group1 - n_g1
    missing_g2 = total_group2 - n_g2
    missing_g3 = total_group3 - n_g3
    if missing_g1 > 0 or missing_g2 > 0 or missing_g3 > 0:
         print(f"  {'Missing/Non-Numeric':<33} {missing_g1:<20} {missing_g2:<20} {missing_g3:<20}")


def combine_race_ethnicity(df, race_col, ethnicity_col):
    """Combines race and ethnicity into standardized categories with corrected logic."""
    print(f"\n--- Combining Race/Ethnicity (using '{race_col}' and '{ethnicity_col}') --- ")
    
    # Check if columns exist
    if race_col not in df.columns or ethnicity_col not in df.columns:
        print(f"Warning: Race ('{race_col}') or Ethnicity ('{ethnicity_col}') column not found. Cannot combine.")
        df['race_ethnicity_combined'] = 'Unknown/Missing' # Assign default if columns missing
        return df
        
    # --- REVISED: Standardization and Logic ---
    # Use .copy() to avoid modifying the original DataFrame slices implicitly
    race_series = df[race_col].copy().fillna('unknown').astype(str).str.lower().str.strip()
    ethnicity_series = df[ethnicity_col].copy().fillna('unknown').astype(str).str.lower().str.strip()
    
    # Standardize common variations more comprehensively
    race_map = {
        'black or african american': 'Black',
        'american indian or alaska native': 'Other/Multiethnic',
        'native hawaiian or other pacific islander': 'Other/Multiethnic',
        'middle eastern': 'White', # Mapping Middle Eastern to White based on common practice
        'patient refused': 'Unknown',
        'please update': 'Unknown',
        'preferred category not available': 'Unknown',
        'no parent/primary caregiver present': 'Unknown',
        'more than one race': 'Other/Multiethnic',
        'unknown': 'Unknown',
        'other': 'Other/Multiethnic', # Explicitly map 'other'
        'nan': 'Unknown', # Handle string 'nan'
        '': 'Unknown', # Handle empty strings
        'white': 'White', # Ensure case consistency
        'asian': 'Asian', # Ensure case consistency
        'black': 'Black' # Ensure case consistency
    }
    ethnicity_map = {
        'non-hispanic': 'Not Hispanic or Latino',
        'not hispanic or latino': 'Not Hispanic or Latino',
        'hispanic': 'Hispanic or Latino',
        'hispanic or latino': 'Hispanic or Latino',
        'patient refused': 'Unknown',
        'unknown': 'Unknown',
        'missing': 'Unknown', # Added mapping for 'missing'
        'nan': 'Unknown', # Handle string 'nan'
        '': 'Unknown' # Handle empty strings
    }

    race_series = race_series.map(lambda x: race_map.get(x, 'Unknown')) # Use map for safer default
    ethnicity_series = ethnicity_series.map(lambda x: ethnicity_map.get(x, 'Unknown'))

    # Initialize the combined column with a default
    df['race_ethnicity_combined'] = 'Unknown/Missing' # Default value

    # Apply logic:
    # 1. Hispanic/Latino takes precedence
    df.loc[ethnicity_series == 'Hispanic or Latino', 'race_ethnicity_combined'] = 'Hispanic or Latino'
    
    # 2. If Not Hispanic/Latino, use Race
    # Use .loc to ensure assignment works correctly
    df.loc[(ethnicity_series == 'Not Hispanic or Latino') & (race_series == 'White'), 'race_ethnicity_combined'] = 'White, Not Hispanic or Latino'
    df.loc[(ethnicity_series == 'Not Hispanic or Latino') & (race_series == 'Black'), 'race_ethnicity_combined'] = 'Black, Not Hispanic or Latino'
    df.loc[(ethnicity_series == 'Not Hispanic or Latino') & (race_series == 'Asian'), 'race_ethnicity_combined'] = 'Asian, Not Hispanic or Latino'
    df.loc[(ethnicity_series == 'Not Hispanic or Latino') & (race_series == 'Other/Multiethnic'), 'race_ethnicity_combined'] = 'Other/Multiethnic, Not Hispanic or Latino'
    
    # 3. If Ethnicity is Unknown, try to use Race (if known and not already Hispanic)
    df.loc[(df['race_ethnicity_combined'] == 'Unknown/Missing') & (ethnicity_series == 'Unknown') & (race_series == 'White'), 'race_ethnicity_combined'] = 'White, Unknown Ethnicity'
    df.loc[(df['race_ethnicity_combined'] == 'Unknown/Missing') & (ethnicity_series == 'Unknown') & (race_series == 'Black'), 'race_ethnicity_combined'] = 'Black, Unknown Ethnicity'
    df.loc[(df['race_ethnicity_combined'] == 'Unknown/Missing') & (ethnicity_series == 'Unknown') & (race_series == 'Asian'), 'race_ethnicity_combined'] = 'Asian, Unknown Ethnicity'
    df.loc[(df['race_ethnicity_combined'] == 'Unknown/Missing') & (ethnicity_series == 'Unknown') & (race_series == 'Other/Multiethnic'), 'race_ethnicity_combined'] = 'Other/Multiethnic, Unknown Ethnicity'
    
    # 4. Remaining unknowns (where both Race and Ethnicity were Unknown, or Race was Unknown and Ethnicity wasn't Hispanic) stay as 'Unknown/Missing'
    # This is covered by the initialization and the logic above.

    print(f"Created 'race_ethnicity_combined'. Value counts:\n{df['race_ethnicity_combined'].value_counts().to_string()}")
    return df

# --- Function to map Uveitis Laterality ---
def map_uveitis_laterality(df, laterality_col):
    """Maps the raw laterality text to standardized categories."""
    print(f"\n--- Mapping Uveitis Laterality from '{laterality_col}' --- ")
    if laterality_col not in df.columns:
        print(f"Warning: Laterality column '{laterality_col}' not found. Adding empty column.")
        df['uveitis_laterality'] = 'Unknown/Missing'
        return df
        
    laterality_series = df[laterality_col].fillna('Unknown').astype(str).str.lower().str.strip()
    conditions = [ 
        laterality_series == 'both', 
        laterality_series.isin(['right', 'left', 'od', 'os', 'unilateral']) # Added 'unilateral'
    ]
    choices = [ 'Bilateral uveitis', 'Unilateral uveitis' ]
    default_choice = 'Unknown/Missing' # Includes 'Unknown', empty strings, NaNs etc.
    
    df['uveitis_laterality'] = np.select(conditions, choices, default=default_choice)
    print(f"Created 'uveitis_laterality'. Value counts:\n{df['uveitis_laterality'].value_counts().to_string()}")
    return df

# --- Lab Parsing/Checking Functions ---
def parse_semicolon_string_column(df, col_name):
    """Helper to split semicolon-separated strings into lists, handling NaNs."""
    if col_name not in df.columns: 
        print(f"Warning: Column '{col_name}' not found for parsing.")
        # Return a Series of empty lists with the same index as df
        return pd.Series([[] for _ in range(len(df))], index=df.index) 
    
    def split_string(val):
        # Ensure input is treated as string, split, strip, handle NaNs
        if pd.isna(val): return []
        # Ensure val is string before split
        return [item.strip() for item in str(val).split(';') if item.strip()] # Skip empty items after split
        
    return df[col_name].apply(split_string)

def check_medication(med_list, med_name_lower):
    """Checks if a specific medication (lowercase) is in the list."""
    if not isinstance(med_list, list): return False
    # Check for variations
    return any(str(med).lower() == med_name_lower or 
               str(med).lower() == med_name_lower + ' sodium' # Handle common variations like 'methotrexate sodium'
               for med in med_list)

def check_biologics(med_list, biologics_lower_list):
    """Checks if any known biologic (lowercase) is in the list."""
    if not isinstance(med_list, list): return False
    return any(str(med).lower() in biologics_lower_list for med in med_list)

# --- NEW: Function to Calculate Time to First MTX --- 
def calculate_time_to_first_mtx(df, start_date_col, treatment_col, onset_year_col):
    """
    Calculates the time in years from arthritis onset to the first Methotrexate start date.
    Uses the year part of dates only.
    """
    print(f"\n--- Calculating Time from Arthritis Onset ('{onset_year_col}') to First Methotrexate ---")
    print(f"Using start dates from '{start_date_col}' and treatments from '{treatment_col}'")

    results = []
    if not all(c in df.columns for c in [start_date_col, treatment_col, onset_year_col]):
        print(f"Warning: Required columns ('{start_date_col}', '{treatment_col}', '{onset_year_col}') not found. Skipping calculation.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Parse columns once
    start_dates_parsed = parse_semicolon_string_column(df, start_date_col)
    treatments_parsed = parse_semicolon_string_column(df, treatment_col)
    onset_years = pd.to_numeric(df[onset_year_col], errors='coerce')

    for index, row in df.iterrows():
        dates = start_dates_parsed.loc[index]
        treatments = treatments_parsed.loc[index]
        onset_year = onset_years.loc[index]

        if not isinstance(dates, list) or not isinstance(treatments, list) or pd.isna(onset_year):
            results.append(np.nan)
            continue

        # Ensure lists are of same length conceptually, though data might be messy
        min_len = min(len(dates), len(treatments))
        mtx_start_years = []

        for i in range(min_len):
            treatment_name = str(treatments[i]).lower()
            date_str = str(dates[i])

            if 'methotrexate' in treatment_name:
                try:
                    # Extract year: handles YYYY-MM-DD and YYYY
                    year = pd.to_datetime(date_str, errors='coerce').year
                    if pd.notna(year):
                         mtx_start_years.append(int(year))
                    else: # Try parsing as just YYYY if full date failed
                         if re.fullmatch(r'\d{4}', date_str):
                             year_int = int(date_str)
                             if 1900 < year_int < 2100: # Basic sanity check
                                 mtx_start_years.append(year_int)
                except Exception: # Catch any other parsing errors
                    pass # Ignore dates that cannot be parsed

        if mtx_start_years:
            first_mtx_year = min(mtx_start_years)
            # Ensure onset_year is also valid year
            if 1900 < onset_year < 2100:
                 time_diff = first_mtx_year - onset_year
                 # Only record non-negative time differences
                 results.append(time_diff if time_diff >= 0 else np.nan)
            else:
                 results.append(np.nan) # Invalid onset year
        else:
            results.append(np.nan) # No valid MTX start date found

    print(f"Calculation complete. Found valid time differences for {pd.Series(results).notna().sum()} patients.")
    return pd.Series(results, index=df.index)

# --- NEW: Interpretation Helper Function ---
def interpret_value(lab_name, value_str, unit_str):
    """Interprets a lab value based on lab name, value content, and unit."""
    if pd.isna(value_str):
        return "Unknown"
    
    val_lower = str(value_str).lower().strip()
    unit_lower = str(unit_str).lower().strip() if pd.notna(unit_str) else '' # Handle potential NaN units

    # 1. Check for common explicit keywords (usually unit-independent)
    if lab_name in ['ANA', 'HLA-B27']:
        if 'positive' in val_lower or 'detected' in val_lower:
            return "Positive"
        if 'negative' in val_lower or 'not detected' in val_lower or 'non-reactive' in val_lower or 'neg' in val_lower:
            return "Negative"
        # Check for titer patterns (e.g., < 1:16, 1:80) - often implies negative/low positive
        if re.match(r'(<|>)?\s?1:\d+', val_lower):
             try:
                 ratio = int(val_lower.split(':')[-1])
                 if ratio <= 16:
                      return "Negative (Titer)"
                 else:
                      return "Positive (Titer)"
             except:
                 pass

    # Explicit keywords for ESR/VitD might still be useful as fallback
    if lab_name == 'ESR':
        if 'high' in val_lower or 'elevated' in val_lower:
            return "High"
        if 'normal' in val_lower or 'wnl' in val_lower:
            return "Not High"

    if lab_name == 'Vitamin D':
        if 'low' in val_lower or 'deficient' in val_lower:
            return "Low"
        if 'normal' in val_lower or 'sufficient' in val_lower or 'insufficient' in val_lower:
            return "Not Low"

    # 2. Attempt numeric interpretation if no keyword match
    numeric_val = pd.to_numeric(value_str, errors='coerce')

    if pd.isna(numeric_val):
        # Handle titers even if numeric conversion failed initially
        if lab_name == 'ANA' and ':' in val_lower and '1:' in val_lower:
             try:
                 ratio = int(val_lower.split(':')[-1])
                 if ratio <= 16: return "Negative (Titer)"
                 else: return "Positive (Titer)"
             except: return "Unknown"
        else:
            return "Unknown"

    # Apply numeric thresholds, considering units
    if lab_name == 'ESR':
        # Assuming threshold is for mm/hr
        if unit_lower == 'mm/hr' or not unit_lower: # Treat missing unit as mm/hr for now
            if numeric_val < 0:
                 return "Unknown"
            if numeric_val >= ESR_HIGH_THRESHOLD:
                return "High"
            else:
                return "Not High"
        else:
            # Found a different unit for ESR, need specific handling or treat as unknown
            print(f"Warning: Unhandled ESR unit '{unit_str}' for value {value_str}. Treating as Unknown.")
            return "Unknown"

    if lab_name == 'Vitamin D':
        # Threshold is for ng/mL
        val_to_compare = numeric_val
        if unit_lower == 'nmol/l':
            val_to_compare = numeric_val / 2.496 # Convert nmol/L to ng/mL (using 2.496 factor)
        elif unit_lower == 'ng/ml' or not unit_lower: # Treat missing unit as ng/mL
            pass # Already in correct unit or assuming correct unit
        else:
             # Found a different unit for VitD
            print(f"Warning: Unhandled Vitamin D unit '{unit_str}' for value {value_str}. Treating as Unknown.")
            return "Unknown"
            
        # Now compare the (potentially converted) value to the threshold
        if val_to_compare < 0:
             return "Unknown"
        if val_to_compare <= VITD_DEFICIENT_THRESHOLD:
            return "Low"
        else:
            return "Not Low"

    return "Unknown"
# --- END NEW HELPER ---

# --- Function to combine JIA subtypes from multiple columns ---
def combine_jia_subtypes(df, primary_col, alt_col1, alt_col2):
    """
    Combines JIA subtype data from three columns with priority:
    primary_col > alt_col1 > alt_col2
    """
    print(f"\n--- Combining JIA subtypes from multiple columns ---")
    
    # Check which columns exist in the dataframe
    primary_exists = primary_col in df.columns
    alt1_exists = alt_col1 in df.columns
    alt2_exists = alt_col2 in df.columns
    
    if not (primary_exists or alt1_exists or alt2_exists):
        print("Warning: None of the JIA subtype columns found. Cannot analyze JIA subtypes.")
        df['combined_jia_subtype'] = None
        return df
    
    # Initialize the combined column with None
    df['combined_jia_subtype'] = None
    
    # Parse semicolon-separated values if columns exist
    if primary_exists:
        df['primary_jia_parsed'] = parse_semicolon_string_column(df, primary_col)
        # Use the first non-empty value if it's a list
        df['primary_jia'] = df['primary_jia_parsed'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )
        # Assign primary column values first (highest priority)
        df.loc[df['primary_jia'].notna(), 'combined_jia_subtype'] = df.loc[df['primary_jia'].notna(), 'primary_jia']
        print(f"Used primary column '{primary_col}' where available")
    
    # Fill in from alt_col1 where primary is missing
    if alt1_exists:
        df['alt1_jia_parsed'] = parse_semicolon_string_column(df, alt_col1)
        df['alt1_jia'] = df['alt1_jia_parsed'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )
        # Only fill in where combined is still None
        df.loc[(df['combined_jia_subtype'].isna()) & (df['alt1_jia'].notna()), 'combined_jia_subtype'] = \
            df.loc[(df['combined_jia_subtype'].isna()) & (df['alt1_jia'].notna()), 'alt1_jia']
        print(f"Used alternative column 1 '{alt_col1}' where primary was missing")
    
    # Fill in from alt_col2 where both primary and alt1 are missing
    if alt2_exists:
        df['alt2_jia_parsed'] = parse_semicolon_string_column(df, alt_col2)
        df['alt2_jia'] = df['alt2_jia_parsed'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )
        # Only fill in where combined is still None
        df.loc[(df['combined_jia_subtype'].isna()) & (df['alt2_jia'].notna()), 'combined_jia_subtype'] = \
            df.loc[(df['combined_jia_subtype'].isna()) & (df['alt2_jia'].notna()), 'alt2_jia']
        print(f"Used alternative column 2 '{alt_col2}' where primary and alt1 were missing")
    
    # Count how many were filled from each source
    if primary_exists:
        primary_count = df['primary_jia'].notna().sum()
        print(f"Found {primary_count} subtypes from primary column")
    
    if alt1_exists:
        alt1_fill_count = ((df['combined_jia_subtype'].notna()) & (df['primary_jia'].isna()) & (df['alt1_jia'].notna())).sum()
        print(f"Added {alt1_fill_count} subtypes from alternative column 1")
    
    if alt2_exists:
        alt2_fill_count = ((df['combined_jia_subtype'].notna()) & (df['primary_jia'].isna()) & 
                          (df['alt1_jia'].isna() if alt1_exists else True) & (df['alt2_jia'].notna())).sum()
        print(f"Added {alt2_fill_count} subtypes from alternative column 2")
    
    # Report final coverage
    total_filled = df['combined_jia_subtype'].notna().sum()
    print(f"Combined JIA subtype column has {total_filled}/{len(df)} filled values ({total_filled/len(df)*100:.1f}%)")
    
    # Clean up temporary columns
    cols_to_drop = ['primary_jia_parsed', 'primary_jia', 'alt1_jia_parsed', 'alt1_jia', 'alt2_jia_parsed', 'alt2_jia']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df

# --- Main Script ---
def main():
    # Define output file path
    OUTPUT_FILE_PATH = "table1_comparison_output.txt"
    GROUPS_OUTPUT_FILE = "patient_groups.csv"
    # --- ADDED: Define new column names --- 
    MED_START_DATE_COL = 'medication start date (list distinct)'
    MED_TREATMENT_COL = 'cmtrt (list distinct)'
    # --- Define column to use for treatment checks --- 
    TREATMENT_CHECK_COL = MED_TREATMENT_COL # Use cmtrt for checks now

    # Redirect stdout to the output file
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(OUTPUT_FILE_PATH, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        
        # --- Start of Analysis Logic (all prints will go to the file) ---
        print(f"--- Final Table 1 Generation Script Output ---")
        print(f"Data Source: {FILE_PATH}\n")
        # --- ADDED: Note about N values ---
        print("NOTE: 'N' values in table headers refer to the number of patients within the ")
        print("      JIA-Only/JIA-U groups with valid, non-missing data for that specific variable.")
        print("      Total patient counts for each group are listed below the metric name.")
        # --- END ADDED ---
        
        print(f"Loading data...")
        try:
            df = pd.read_csv(FILE_PATH, low_memory=False)
            print(f"Successfully loaded CSV. Initial shape: {df.shape}\n")
            
            # 1. Define JIA-Only vs JIA-U groups based on DX Codes
            print(f"--- Grouping based on codes in '{DX_CODE_COL}' ---")
            df_analysis = parse_and_group_by_dx(df, DX_CODE_COL, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS)

            if df_analysis is None or df_analysis.empty:
                print("Error: Could not define JIA-Only and JIA-U groups. Exiting.")
                sys.stdout = original_stdout 
                print("Error occurred during grouping, check table1_comparison_output.txt for details.")
                return 
            
            if len(df_analysis['diagnosis_group'].unique()) < 3:
                print("Error: Only two groups found after filtering. Cannot perform comparison. Exiting.")
                sys.stdout = original_stdout
                print("Error: Only two groups found after filtering, check table1_comparison_output.txt for details.")
                return

            print("\n========= Table 1: JIA-Only vs JIA-U vs Uveitis-Only Comparison =========")

            # --- Demographics --- 
            calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', GENDER_COL, 'Gender')
            
            # Combine JIA subtypes from multiple columns before analysis
            df_analysis = combine_jia_subtypes(df_analysis, JIA_SUBTYPE_COL, JIA_SUBTYPE_COL_ALT1, JIA_SUBTYPE_COL_ALT2)
            
            # Analyze the combined JIA subtype column if available
            if 'combined_jia_subtype' in df_analysis.columns:
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'combined_jia_subtype', 'JIA Subtype (Combined)')
                print("(Note: P-value for multi-category JIA Subtype might require different handling/interpretation)") 
            else:
                # Fallback to original column if combination failed
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', JIA_SUBTYPE_COL, 'JIA Subtype')
                print("(Note: P-value for multi-category JIA Subtype might require different handling/interpretation)") 
            
            # Race/Ethnicity Analysis
            if RACE_COL in df_analysis.columns and ETHNICITY_COL in df_analysis.columns:
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', RACE_COL, 'Race (Raw Categories)')
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', ETHNICITY_COL, 'Ethnicity (Raw Categories)')
                df_analysis = combine_race_ethnicity(df_analysis, RACE_COL, ETHNICITY_COL) # Add combined col to main df
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'race_ethnicity_combined', 'Race/Ethnicity (Combined)')
            else: 
                print(f"\nColumns '{RACE_COL}' or '{ETHNICITY_COL}' not found. Skipping Race/Ethnicity analysis.")
                
            # --- Disease Onset/Characteristics --- 
            calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', ARTHRITIS_ONSET_YEAR_COL, 'Arthritis Onset Year')
            calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', UVEITIS_ONSET_YEAR_COL, 'Uveitis Onset Year')
            
            # --- Calculating Age at Onset --- 
            # Calculate Age at Arthritis Onset (from DOB)
            print("\n--- Calculating Age at Arthritis Onset (from DOB) ---")
            dob_col_name = 'DOB' 
            if dob_col_name in df_analysis.columns and ARTHRITIS_ONSET_YEAR_COL in df_analysis.columns:
                try: 
                    birth_year = pd.to_numeric(df_analysis[dob_col_name].astype(str).str[:4], errors='coerce') 
                    arth_onset_year = pd.to_numeric(df_analysis[ARTHRITIS_ONSET_YEAR_COL], errors='coerce')
                    arth_onset_year = arth_onset_year.where((arth_onset_year > 1900) & (arth_onset_year < 2100))
                    birth_year = birth_year.where((birth_year > 1900) & (birth_year < 2100))
                    df_analysis['age_at_arth_onset_dob'] = arth_onset_year - birth_year
                    df_analysis.loc[df_analysis['age_at_arth_onset_dob'] < 0, 'age_at_arth_onset_dob'] = np.nan 
                    df_analysis.loc[df_analysis['age_at_arth_onset_dob'] > 100, 'age_at_arth_onset_dob'] = np.nan 
                    calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', 'age_at_arth_onset_dob', 'Age at Arthritis Onset (from DOB)')
                except Exception as e:
                    print(f"Error calculating Age at Arthritis Onset from DOB: {e}")
                    print("Skipping this calculation.")
            else:
                print(f"Skipping Age at Arthritis Onset calculation - Column '{dob_col_name}' or '{ARTHRITIS_ONSET_YEAR_COL}' not found or not suitable.")
                
            # Calculate Age at Uveitis Onset (from DOB)
            print("\n--- Calculating Age at Uveitis Onset (from DOB) ---")
            if dob_col_name in df_analysis.columns and UVEITIS_ONSET_YEAR_COL in df_analysis.columns:
                try: 
                    birth_year = pd.to_numeric(df_analysis[dob_col_name].astype(str).str[:4], errors='coerce') 
                    uv_onset_year = pd.to_numeric(df_analysis[UVEITIS_ONSET_YEAR_COL], errors='coerce')
                    uv_onset_year = uv_onset_year.where((uv_onset_year > 1900) & (uv_onset_year < 2100))
                    birth_year = birth_year.where((birth_year > 1900) & (birth_year < 2100))
                    df_analysis['age_at_uv_onset_dob'] = uv_onset_year - birth_year
                    df_analysis.loc[df_analysis['age_at_uv_onset_dob'] < 0, 'age_at_uv_onset_dob'] = np.nan 
                    df_analysis.loc[df_analysis['age_at_uv_onset_dob'] > 100, 'age_at_uv_onset_dob'] = np.nan 
                    calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', 'age_at_uv_onset_dob', 'Age at Uveitis Onset (from DOB)')
                except Exception as e:
                     print(f"Error calculating Age at Uveitis Onset from DOB: {e}")
                     print("Skipping this calculation.")
            else:
                print(f"Skipping Age at Uveitis Onset calculation - Column '{dob_col_name}' or '{UVEITIS_ONSET_YEAR_COL}' not found or not suitable.")

            # Map and Analyze Uveitis Laterality
            if UVEITIS_LATERALITY_COL in df_analysis.columns:
                df_analysis = map_uveitis_laterality(df_analysis, UVEITIS_LATERALITY_COL) # Add mapped col
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'uveitis_laterality', 'Uveitis Laterality')
            else: 
                 print(f"\nColumn '{UVEITIS_LATERALITY_COL}' not found.")
                 
            # Calculate Time Intervals (ensure columns exist first)
            if ARTHRITIS_ONSET_YEAR_COL in df_analysis.columns and UVEITIS_ONSET_YEAR_COL in df_analysis.columns:
                df_analysis['years_jia_uv_onset'] = pd.to_numeric(df_analysis[UVEITIS_ONSET_YEAR_COL], errors='coerce') - \
                                                  pd.to_numeric(df_analysis[ARTHRITIS_ONSET_YEAR_COL], errors='coerce')
                calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', 'years_jia_uv_onset', 'Years between JIA and Uveitis Onset')
            else: 
                print(f"\nCannot calculate Years between JIA and Uveitis Onset - required columns ('{ARTHRITIS_ONSET_YEAR_COL}', '{UVEITIS_ONSET_YEAR_COL}') missing or non-numeric.")

            if ARTHRITIS_ONSET_YEAR_COL in df_analysis.columns and UVEITIS_DIAGNOSIS_YEAR_COL in df_analysis.columns:
                df_analysis['years_jia_onset_uv_dx'] = pd.to_numeric(df_analysis[UVEITIS_DIAGNOSIS_YEAR_COL], errors='coerce') - \
                                                     pd.to_numeric(df_analysis[ARTHRITIS_ONSET_YEAR_COL], errors='coerce')
                calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', 'years_jia_onset_uv_dx', 'Years between JIA Onset and Uveitis Diagnosis')
            else: 
                print(f"\nCannot calculate Years between JIA Onset and Uveitis Diagnosis - required columns ('{ARTHRITIS_ONSET_YEAR_COL}', '{UVEITIS_DIAGNOSIS_YEAR_COL}') missing or non-numeric.")

            # --- NEW: Calculate Time to First MTX --- 
            df_analysis['time_to_mtx_years'] = calculate_time_to_first_mtx(
                df_analysis,
                start_date_col=MED_START_DATE_COL,
                treatment_col=MED_TREATMENT_COL,
                onset_year_col=ARTHRITIS_ONSET_YEAR_COL
            )
            calculate_continuous_stats(df_analysis.copy(), 'diagnosis_group', 'time_to_mtx_years', 'Time to Start Methotrexate from JIA Onset (Years)')

            # --- Treatments ---
            if TREATMENT_CHECK_COL in df_analysis.columns:
                print(f"\n--- Analyzing Treatments based on '{TREATMENT_CHECK_COL}' ---")
                df_analysis['treatment_list_parsed'] = parse_semicolon_string_column(df_analysis, TREATMENT_CHECK_COL)
                
                # Methotrexate
                df_analysis['treated_mtx'] = df_analysis['treatment_list_parsed'].apply(lambda x: check_medication(x, 'methotrexate'))
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'treated_mtx', 'Treated with Methotrexate')
                
                # Biologics
                df_analysis['treated_biologic'] = df_analysis['treatment_list_parsed'].apply(lambda x: check_biologics(x, BIOLOGICS_LIST))
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'treated_biologic', 'Treated with Biologic')
                
                # Biologic + MTX
                if 'treated_mtx' in df_analysis.columns and 'treated_biologic' in df_analysis.columns:
                    df_analysis['treated_both_mtx_bio'] = df_analysis['treated_mtx'] & df_analysis['treated_biologic']
                    calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'treated_both_mtx_bio', 'Treated with Biologic + Methotrexate')
                else:
                     print("Could not calculate 'Treated with Biologic + Methotrexate' as prerequisite columns missing.")
                    
                biologics_str = ", ".join(BIOLOGICS_LIST)
                print(f"(Biologics considered: {biologics_str})") 
                print("(Note: Treatment analysis based on presence in semicolon-separated list.)")
            else: 
                print(f"\nColumn '{TREATMENT_CHECK_COL}' not found. Cannot analyze treatments.")

            # --- Lab Values (Selected) ---
            needs_lab_parsing = (ANA_COL not in df_analysis.columns or not df_analysis[ANA_COL].notna().any()) or \
                                (LAB_COMPONENT_COL in df_analysis.columns and LAB_VALUE_COL in df_analysis.columns) or \
                                (MEASURE_NAME_COL in df_analysis.columns and LAB_VALUE_COL in df_analysis.columns) # Added check for measure name col

            labs_parsed = False
            # Parse Lab/Measure name/value columns if needed
            if needs_lab_parsing and LAB_VALUE_COL in df_analysis.columns:
                print("\n--- Parsing Relevant Lab/Measure Name/Value Columns --- ")
                df_analysis['lab_names_parsed'] = parse_semicolon_string_column(df_analysis, LAB_COMPONENT_COL) if LAB_COMPONENT_COL in df_analysis.columns else pd.Series([[] for _ in range(len(df_analysis))], index=df_analysis.index)
                df_analysis['measure_names_parsed'] = parse_semicolon_string_column(df_analysis, MEASURE_NAME_COL) if MEASURE_NAME_COL in df_analysis.columns else pd.Series([[] for _ in range(len(df_analysis))], index=df_analysis.index)
                df_analysis['lab_values_parsed'] = parse_semicolon_string_column(df_analysis, LAB_VALUE_COL)
                # --- ADDED: Parse units column --- 
                df_analysis['units_parsed'] = parse_semicolon_string_column(df_analysis, MEASURE_UNIT_COL) if MEASURE_UNIT_COL in df_analysis.columns else pd.Series([[] for _ in range(len(df_analysis))], index=df_analysis.index)
                
                if LAB_COMPONENT_COL in df_analysis.columns: print(f"Parsed '{LAB_COMPONENT_COL}'.")
                if MEASURE_NAME_COL in df_analysis.columns: print(f"Parsed '{MEASURE_NAME_COL}'.")
                print(f"Parsed '{LAB_VALUE_COL}'.")
                # --- ADDED: Print unit parsing status --- 
                if MEASURE_UNIT_COL in df_analysis.columns: print(f"Parsed '{MEASURE_UNIT_COL}'.")
                else: print(f"Warning: Unit column '{MEASURE_UNIT_COL}' not found.")
                labs_parsed = True
            elif needs_lab_parsing:
                 print(f"\nWarning: Need to parse labs but required columns ('{LAB_VALUE_COL}' and at least one name column) are missing.")

            # --- Process Labs using interpret_value ---
            if labs_parsed:
                print("\n--- Interpreting Lab Values ---")
                
                # Initialize interpretation columns
                df_analysis['ana_interpretation'] = 'Unknown'
                df_analysis['hla_b27_interpretation'] = 'Unknown'
                df_analysis['esr_interpretation'] = 'Unknown'
                df_analysis['vitd_interpretation'] = 'Unknown'

                # Process ANA from dedicated column first
                if ANA_COL in df_analysis.columns:
                    def process_ana_values(ana_str):
                        if pd.isna(ana_str):
                            return 'Unknown'
                        # Split by semicolon and process each value
                        values = [v.strip().lower() for v in str(ana_str).split(';')]
                        # Return Positive if any value is positive
                        if any('positive' in v for v in values):
                            return 'Positive'
                        # Return Negative only if we have values and none are positive
                        elif len(values) > 0 and all('negative' in v for v in values):
                            return 'Negative'
                        return 'Unknown'
                    
                    df_analysis['ana_interpretation'] = df_analysis[ANA_COL].apply(process_ana_values)
                
                # Define patterns for matching lab names (lowercase)
                ana_patterns = ['ana', 'antinuclear']
                hla_patterns = ['hla-b27', 'hla b27']
                esr_patterns = ['esr', 'sed rate', 'sedimentation rate']
                vitd_patterns = ['vitamin d', 'vit d', '25-hydroxyvitamin', '25(oh)', 'vitd']

                # Process other lab values through the regular pipeline
                for index, row in df_analysis.iterrows():
                    lab_names = row.get('lab_names_parsed', [])
                    measure_names = row.get('measure_names_parsed', [])
                    values = row.get('lab_values_parsed', [])
                    units = row.get('units_parsed', []) # --- ADDED: Get units --- 
                    
                    combined_names = lab_names + measure_names # Combine for easier searching
                    max_len = max(len(lab_names), len(measure_names), len(values), len(units)) # Use max length across all lists

                    # Use existing interpretations as base, only update if unknown
                    current_hla = df_analysis.loc[index, 'hla_b27_interpretation']
                    current_esr = df_analysis.loc[index, 'esr_interpretation']
                    current_vitd = df_analysis.loc[index, 'vitd_interpretation']
                    
                    found_hla_name = False # Flag for HLA fallback check
                    found_positive_value = False # Flag for HLA fallback check

                    # --- Fallback Check Preparation ---
                    if any(any(p in str(name).lower() for p in hla_patterns) for name in combined_names if pd.notna(name)):
                        found_hla_name = True
                    if any('positive' in str(val).lower() or 'detected' in str(val).lower() for val in values if pd.notna(val)):
                        found_positive_value = True
                    # --- End Fallback Check Prep ---

                    # --- Iterate through aligned indices ---
                    for i in range(max_len):
                        # Get names, value, and unit at index i (handle potential shorter lists)
                        l_name = lab_names[i].lower() if i < len(lab_names) and pd.notna(lab_names[i]) else ''
                        m_name = measure_names[i].lower() if i < len(measure_names) and pd.notna(measure_names[i]) else ''
                        value_str = values[i] if i < len(values) else None
                        unit_str = units[i] if i < len(units) else None # --- ADDED: Get unit string --- 
                        
                        # Interpret HLA-B27 (doesn't need units for positive/negative)
                        if any(p in l_name or p in m_name for p in hla_patterns):
                             interp = interpret_value('HLA-B27', value_str, unit_str) # Pass unit_str
                             if interp == 'Positive': current_hla = 'Positive'
                             elif interp == 'Negative':
                                 if current_hla == 'Unknown': current_hla = 'Negative'

                        # Interpret ESR (pass unit)
                        if current_esr == 'Unknown' and any(p in l_name or p in m_name for p in esr_patterns):
                            interp = interpret_value('ESR', value_str, unit_str) # Pass unit_str
                            if interp != 'Unknown': current_esr = interp # Update if interpretation successful
                        
                        # Interpret Vitamin D (pass unit)
                        if current_vitd == 'Unknown' and any(p in l_name or p in m_name for p in vitd_patterns):
                            interp = interpret_value('Vitamin D', value_str, unit_str) # Pass unit_str
                            if interp != 'Unknown': current_vitd = interp # Update if interpretation successful
                            
                    # --- Apply HLA-B27 Fallback ---
                    if current_hla == 'Unknown' and found_hla_name and found_positive_value:
                         print(f"Applying HLA-B27 fallback for index {index}") # Add logging
                         current_hla = 'Positive (Fallback)'
                    # --- End Fallback ---

                    # Assign final interpretation for the row
                    df_analysis.loc[index, 'hla_b27_interpretation'] = current_hla
                    df_analysis.loc[index, 'esr_interpretation'] = current_esr
                    df_analysis.loc[index, 'vitd_interpretation'] = current_vitd

                # --- Calculate Stats for Interpreted Labs ---
                print("--- Calculating Stats for Interpreted Lab Values ---")
                # Convert interpretations to boolean for simple Positive/Negative stats if needed
                df_analysis['ana_positive'] = df_analysis['ana_interpretation'] == 'Positive'
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'ana_positive', 'ANA Positivity (Interpreted)')
                
                df_analysis['hla_b27_positive'] = df_analysis['hla_b27_interpretation'].isin(['Positive', 'Positive (Fallback)'])
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'hla_b27_positive', 'HLA-B27 Positivity (Interpreted)')
                
                # Calculate stats directly on the interpretation categories
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'esr_interpretation', 'ESR Status (Interpreted)')
                calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'vitd_interpretation', 'Vitamin D Status (Interpreted)')

            else:
                 print("\nSkipping detailed Lab interpretation as parsing did not occur.")
                 # Attempt analysis on dedicated ANA column if it exists but parsing didn't happen
                 if ANA_COL in df_analysis.columns and df_analysis[ANA_COL].notna().any():
                     print(f"\n--- Analyzing ANA from dedicated column '{ANA_COL}' only ---")
                     df_analysis['ana_parsed'] = parse_semicolon_string_column(df_analysis, ANA_COL)
                     # Apply interpret_value to the first element if list is not empty
                     df_analysis['ana_interpretation'] = df_analysis['ana_parsed'].apply(
                         lambda lst: interpret_value('ANA', lst[0], None) if isinstance(lst, list) and len(lst) > 0 else 'Unknown'
                     )
                     df_analysis['ana_positive'] = df_analysis['ana_interpretation'].isin(['Positive', 'Positive (Titer)'])
                     calculate_categorical_stats(df_analysis.copy(), 'diagnosis_group', 'ana_positive', 'ANA Positivity (from ANA column only)')
                 else:
                     print("\nCannot determine ANA Positivity - relevant columns missing or empty.")
                 print("Cannot determine HLA-B27, ESR, or Vitamin D Status - parsing failed or columns missing.")
            

            # Save patient groups to CSV
            patient_groups = df_analysis[['Global Subject ID', 'diagnosis_group']]
            patient_groups.to_csv(GROUPS_OUTPUT_FILE, index=False)
            print(f"Saved patient groups to {GROUPS_OUTPUT_FILE}")

        except FileNotFoundError:
            print(f"Error: File not found at {FILE_PATH}")
            sys.stdout = original_stdout
            print(f"Error: File not found at {FILE_PATH}")
            return
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}")
            print("Traceback:")
            import traceback
            print(traceback.format_exc())
            sys.stdout = original_stdout
            print(f"An unexpected error occurred, check table1_comparison_output.txt for details: {e}")
            return

        # --- End of Analysis Logic ---
        
        print("\n========= Analysis Complete =========")

    # Restore stdout
    sys.stdout = original_stdout 
    print(f"Analysis complete. Results saved to {OUTPUT_FILE_PATH}")

# --- Execute Main Function ---
if __name__ == "__main__":
    main() 