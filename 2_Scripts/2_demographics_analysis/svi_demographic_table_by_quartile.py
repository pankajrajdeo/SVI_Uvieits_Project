#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import os
import re
from collections import Counter
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

# Suppress warnings
warnings.filterwarnings("ignore", message="Chi-squared approximation may be incorrect")

# Output directory setup
OUTPUT_DIR = "svi_quartile_demographics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input file path
FILE_PATH = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged.csv"

# --- Configuration ---
# SVI-related columns
SVI_COLUMNS = [
    'svi_socioeconomic (list distinct)',
    'svi_household_comp (list distinct)',
    'svi_minority (list distinct)',
    'svi_housing_transportation (list distinct)'
]

# Demographic/Clinical columns
RACE_COL = 'race'
ETHNICITY_COL = 'ethnicity'
ANA_COL = 'ana_display_value (list distinct)'

# Visualization settings
COLORS = {
    'Race (% White)': '#3274A1',
    'Ethnicity (% Hisp.)': '#E1812C',
    'ANA Positive (%)': '#3A923A'
}
DPI = 300

# --- Helper Functions ---
def _deduplicate_columns(columns):
    """Appends .1, .2 etc to duplicate column names."""
    counts = Counter()
    new_columns = []
    for col in columns:
        counts[col] += 1
        if counts[col] > 1:
            new_columns.append(f"{col}.{counts[col]-1}")
        else:
            new_columns.append(col)
    return new_columns

def parse_svi_values(value):
    """Parse semicolon-separated SVI values and return their mean."""
    if pd.isna(value):
        return np.nan
    
    try:
        # Split by semicolon and convert to float
        values = [float(v.strip()) for v in str(value).split(';') if v.strip()]
        if values:
            return np.mean(values)
        return np.nan
    except Exception as e:
        print(f"Error parsing value '{value}': {e}")
        return np.nan

def calculate_halves(series):
    """Split a series into two halves based on the median."""
    if series.isna().all():
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Remove NaN values for calculation
    non_na = series.dropna()
    
    if len(non_na) < 2:  # Not enough data for halves
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Calculate median
    median_value = non_na.median()
    
    # Create result series (initialized with None)
    result = pd.Series([None] * len(series), index=series.index, dtype='object')
    
    # Assign Q1 for values <= median and Q2 for values > median
    result[non_na[non_na <= median_value].index] = "Q1 (Low)"
    result[non_na[non_na > median_value].index] = "Q2 (High)"
    
    return result

def format_p_value(p_value):
    """Format p-value for table display."""
    if pd.isna(p_value):
        return "N/A"
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.01:
        return "<0.01"
    elif p_value < 0.05:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.3f}"

def calculate_categorical_stats(df, group_col, value_col, metric_name, value_map=None):
    """Calculate statistics for categorical variables across SVI halves."""
    print(f"\n--- Analyzing: {metric_name} ---")
    
    # Check if value column exists
    if value_col not in df.columns:
        print(f"Column '{value_col}' not found. Skipping analysis for {metric_name}.")
        return {
            'Characteristic': metric_name,
            'SVI Q1 (N=0)': "N/A",
            'SVI Q2 (N=0)': "N/A",
            'p-value': "N/A"
        }
    
    # Clean and prepare data
    df_analysis = df[[group_col, value_col]].copy()
    
    # Apply value mapping if provided
    if value_map:
        df_analysis['clean_value'] = df_analysis[value_col].apply(
            lambda x: value_map(x) if pd.notna(x) else 'Missing'
        )
    else:
        # Basic cleaning: convert to string, lowercase, strip whitespace
        df_analysis['clean_value'] = df_analysis[value_col].fillna('Missing').astype(str)
        df_analysis['clean_value'] = df_analysis['clean_value'].str.strip().str.lower()
        df_analysis['clean_value'] = df_analysis['clean_value'].replace({'nan': 'Missing', '': 'Missing', 'unknown': 'Missing'})
    
    # Count patients in each half
    group_counts = df_analysis[group_col].value_counts().sort_index()
    
    # Initialize result dictionary
    result = {'Characteristic': metric_name}
    
    # For each characteristic we're interested in (e.g., "White" for race, "Positive" for ANA)
    if metric_name == "Race (% White)":
        target_value = 'white'
        df_analysis['is_target'] = df_analysis['clean_value'].str.contains('white', case=False, na=False)
    elif metric_name == "Ethnicity (% Hisp.)":
        target_value = 'hispanic'
        df_analysis['is_target'] = df_analysis['clean_value'].str.contains('hispanic', case=False, na=False) | df_analysis['clean_value'].str.contains('latino', case=False, na=False)
    elif metric_name == "ANA Positive (%)":
        target_value = 'positive'
        df_analysis['is_target'] = df_analysis['clean_value'].str.contains('positive', case=False, na=False)
    else:
        # Default behavior for other metrics
        df_analysis['is_target'] = df_analysis['clean_value'] != 'Missing'
        target_value = "Any non-missing"
    
    # Calculate statistics for each half
    group_stats = {}
    
    # Get a list of non-null group values and sort them
    groups = [g for g in df_analysis[group_col].unique() if pd.notna(g)]
    # Sort using the group order we want
    group_order = ["Q1 (Low)", "Q2 (High)"]
    groups.sort(key=lambda x: group_order.index(x) if x in group_order else float('inf'))
    
    for group in groups:
        # Get patients in this group
        group_patients = df_analysis[df_analysis[group_col] == group]
        total_count = len(group_patients)
        
        if total_count == 0:
            group_stats[group] = {
                'count': 0,
                'target_count': 0,
                'percentage': 0,
                'display': "0 (0.0%)"
            }
            continue
        
        # Count patients with target value
        target_count = group_patients['is_target'].sum()
        percentage = (target_count / total_count) * 100
        
        group_stats[group] = {
            'count': total_count,
            'target_count': target_count,
            'percentage': percentage,
            'display': f"{target_count} ({percentage:.1f}%)"
        }
        
        # Add to result dictionary
        result[f'SVI {group} (N={total_count})'] = f"{percentage:.1f}%"
    
    # Fill in missing groups
    for group in ["Q1 (Low)", "Q2 (High)"]:
        if f'SVI {group}' not in result:
            result[f'SVI {group} (N=0)'] = "N/A"
    
    # Calculate p-value if we have sufficient data
    # Create a contingency table for chi-square test
    contingency_data = []
    for group in group_stats:
        target_count = group_stats[group]['target_count']
        non_target_count = group_stats[group]['count'] - target_count
        contingency_data.append([target_count, non_target_count])
    
    # Only calculate p-value if we have data for both groups
    if len(contingency_data) >= 2:
        try:
            # Convert to numpy array for chi-square test
            contingency_array = np.array(contingency_data)
            
            # Run chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency_array)
            result['p-value'] = format_p_value(p)
            print(f"Chi-square p-value: {p:.4f}")
            
            # Check for expected cell count < 5
            if np.any(expected < 5):
                print("Warning: Some expected cell counts < 5, p-value may be inaccurate")
                
        except Exception as e:
            print(f"Error calculating p-value: {e}")
            result['p-value'] = "N/A"
    else:
        result['p-value'] = "N/A"
    
    return result

def process_ana_values(ana_str):
    """Process ANA value strings to categorize as positive or negative."""
    if pd.isna(ana_str):
        return 'Missing'
    
    ana_str = str(ana_str).lower()
    
    # Split multi-valued entries
    values = [v.strip() for v in ana_str.split(';')]
    
    # Check for any positive results
    for val in values:
        if 'positive' in val:
            return 'Positive'
    
    # Check for negative results
    for val in values:
        if 'negative' in val:
            return 'Negative'
    
    # If can't determine, mark as missing
    return 'Missing'

def main():
    print(f"Reading data from {FILE_PATH}...")
    
    # Try to load data with original column names
    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to load with custom header...")
        
        # Fallback: Load data with deduplicated column names
        try:
            with open(FILE_PATH, 'r', encoding='utf-8') as f:
                header = next(f).strip().split(',')
                fixed_header = _deduplicate_columns(header)
            
            df = pd.read_csv(FILE_PATH, header=0, names=fixed_header, low_memory=False)
            print(f"Successfully loaded data with deduplicated header, {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            print(f"Failed to load data: {e}")
            return
    
    # Check if required SVI columns exist
    missing_cols = [col for col in SVI_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"Warning: These SVI columns are missing: {missing_cols}")
        print("Available columns that might contain SVI data:")
        for col in df.columns:
            if 'svi' in col.lower():
                print(f"  - {col}")
        return
    
    # Calculate mean for each SVI component
    print("Calculating SVI component means...")
    for col in SVI_COLUMNS:
        mean_col = f"{col}_mean"
        df[mean_col] = df[col].apply(parse_svi_values)
    
    # Calculate total SVI score (mean of all components)
    mean_cols = [f"{col}_mean" for col in SVI_COLUMNS]
    df['SVI_total'] = df[mean_cols].mean(axis=1)
    
    # Create SVI halves (median split)
    print("Creating SVI median split (Q1/Q2)...")
    df['SVI_half'] = calculate_halves(df['SVI_total'])
    
    # Print counts per half
    half_counts = df['SVI_half'].value_counts().sort_index()
    print("Patients per SVI group:")
    print(half_counts)
    
    # Initialize table results
    table_results = []
    
    # Race (% White)
    race_stats = calculate_categorical_stats(
        df, 'SVI_half', RACE_COL, 'Race (% White)'
    )
    table_results.append(race_stats)
    
    # Ethnicity (% Hispanic)
    ethnicity_stats = calculate_categorical_stats(
        df, 'SVI_half', ETHNICITY_COL, 'Ethnicity (% Hisp.)'
    )
    table_results.append(ethnicity_stats)
    
    # ANA Positive (%)
    ana_stats = calculate_categorical_stats(
        df, 'SVI_half', ANA_COL, 'ANA Positive (%)', 
        value_map=process_ana_values
    )
    table_results.append(ana_stats)
    
    # Create DataFrame from results
    table_df = pd.DataFrame(table_results)
    
    # Ensure consistent column order
    all_columns = ['Characteristic']
    for group in ["Q1 (Low)", "Q2 (High)"]:
        count = half_counts.get(group, 0)
        all_columns.append(f'SVI {group} (N={count})')
    all_columns.append('p-value')
    
    # Reindex to get correct column order
    table_df = table_df.reindex(columns=[col for col in all_columns if col in table_df.columns])
    
    # Save to CSV
    output_csv = os.path.join(OUTPUT_DIR, "svi_halves_demographics_table.csv")
    table_df.to_csv(output_csv, index=False)
    print(f"Table saved to {output_csv}")
    
    # Print table
    print("\n=== Demographic Characteristics by SVI (Median Split) ===\n")
    print(table_df.to_string(index=False))
    
    # Also save as markdown for easy viewing
    output_md = os.path.join(OUTPUT_DIR, "svi_halves_demographics_table.md")
    with open(output_md, 'w') as f:
        f.write("# Demographic Characteristics by SVI (Median Split)\n\n")
        f.write(table_df.to_markdown(index=False))
    print(f"Markdown table saved to {output_md}")
    
    # Create visualization with bar chart and table
    create_visualization(table_df, OUTPUT_DIR)
    print("Analysis complete!")

def create_visualization(table_df, output_dir):
    """Creates a bar chart visualization with a table below it."""
    print("Generating visualization...")
    
    # Extract data for plotting
    characteristics = table_df['Characteristic'].tolist()
    groups = [col for col in table_df.columns if col.startswith('SVI') and 'N=' in col]
    
    # Convert percentage strings to float values
    plot_data = {}
    for char in characteristics:
        plot_data[char] = []
        for g in groups:
            val_str = table_df.loc[table_df['Characteristic'] == char, g].values[0]
            if val_str == 'N/A':
                plot_data[char].append(0)
            else:
                plot_data[char].append(float(val_str.rstrip('%')))
    
    # Extract group names without N counts for x-axis labels
    x_labels = []
    for g in groups:
        # Extract just the group part (Q1, Q2)
        group_name = g.split('(')[0].strip()
        x_labels.append(group_name)
    
    # Setup the figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate positions for grouped bars
    bar_width = 0.25  # Slightly wider since we only have 2 groups now
    x = np.arange(len(x_labels))
    offsets = np.linspace(-(len(characteristics)-1)/2 * bar_width, 
                          (len(characteristics)-1)/2 * bar_width, 
                          len(characteristics))
    
    # Plot bars for each characteristic
    bars = []
    for i, char in enumerate(characteristics):
        pos = x + offsets[i]
        bar = ax.bar(pos, plot_data[char], width=bar_width, 
                     label=char, color=COLORS.get(char, f'C{i}'),
                     edgecolor='black', linewidth=1)
        bars.append(bar)
        
        # Add value labels on top of bars
        for j, rect in enumerate(bar):
            value = plot_data[char][j]
            if value > 0:  # Only add labels for non-zero values
                text = ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 1,
                          f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                # Add outline to text for better visibility
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Set plot labels and styling
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Demographic Characteristics by SVI (Median Split)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    # Add gridlines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(characteristics))
    
    # Set y-axis limit with some padding
    ax.set_ylim(0, 105)  # Max percentage is 100%, add 5% padding
    
    # Add a table at the bottom of the chart
    table_data = []
    for char in characteristics:
        row = [char]
        for g in groups:
            row.append(table_df.loc[table_df['Characteristic'] == char, g].values[0])
        row.append(table_df.loc[table_df['Characteristic'] == char, 'p-value'].values[0])
        table_data.append(row)
    
    # Extract patient counts from column names
    col_labels = []
    for g in groups:
        # Extract group name with N count
        col_labels.append(g.replace('SVI ', ''))
    col_labels.append('p-value')
    
    # Position table
    plt.subplots_adjust(bottom=0.25)
    table = plt.table(
        cellText=table_data,
        colLabels=['Characteristic'] + col_labels,
        loc='bottom',
        cellLoc='center',
        bbox=[0.0, -0.45, 1.0, 0.25]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Make rows a bit taller
    
    # Save the visualization
    output_file = os.path.join(output_dir, "svi_demographics_halves_chart.png")
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    # Close the plot to free memory
    plt.close(fig)

if __name__ == "__main__":
    main()
