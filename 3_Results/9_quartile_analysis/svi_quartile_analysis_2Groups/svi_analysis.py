#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os
from scipy import stats
import csv
from collections import Counter
import re # Import re for diagnosis code matching

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# --- ADDED: Diagnosis Grouping Configuration ---
DX_CODE_COL = 'dx code (list distinct)' 
JIA_CODE_PATTERN = r'M08' 
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']
# --- END ADDED ---

# Output directory for figures
OUTPUT_DIR = "svi_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DPI for high quality figures
DPI = 300

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
    if len(new_columns) != len(set(new_columns)):
         print("Warning: De-duplication might not have fully resolved unique names. Check header.")
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

def calculate_quartiles(series):
    """Calculate quartile categories for a series."""
    if series.isna().all():
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Remove NaN values for quartile calculation
    non_na = series.dropna()
    
    if len(non_na) < 2:  # Changed from < 4: Not enough data for 2 quartiles
        print(f"Warning: Not enough non-NA values ({len(non_na)}) to compute 2 quartiles for series. Returning NaNs.") # Added warning
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Calculate quartile boundaries
    try: # Added try-except for qcut robustness
        quartiles = pd.qcut(non_na, 2, labels=["Q1", "Q2"]) # Changed from 4 to 2, and labels
        
        # Create a Series with NaN for missing values - use object dtype to avoid the warning
        result = pd.Series([None] * len(series), index=series.index, dtype='object')
        result.loc[non_na.index] = quartiles # Use .loc for safer assignment
        
        return result
    except ValueError as e: # Handle cases where qcut might fail (e.g. too few unique values for 2 quantiles)
        print(f"Warning: pd.qcut failed for series (length {len(non_na)} after NaN drop): {e}. Returning NaNs.")
        return pd.Series([np.nan] * len(series), index=series.index)

def add_value_labels(ax, precision=2, fontsize=9):
    """Add value labels to a bar plot."""
    for rect in ax.patches:
        height = rect.get_height()
        if height > 0:  # Only add label if bar has height
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.01,
                f'{height:.{precision}f}',
                ha='center',
                va='bottom',
                fontsize=fontsize
            )

def add_count_percent_table(ax, counts, percentages, title="Distribution"):
    """Add a count and percentage table below the plot."""
    # Create table data
    table_data = []
    for cat, count, pct in zip(counts.index, counts.values, percentages.values):
        table_data.append([cat, f"{count} ({pct:.1f}%)"])
    
    # Create the table
    tab = ax.table(
        cellText=table_data,
        colLabels=["Category", "Count (%)"],
        loc='bottom',
        cellLoc='center',
        bbox=[0.2, -0.3, 0.6, 0.2]  # [left, bottom, width, height]
    )
    
    # Style the table
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.2, 1.2)
    
    # Adjust plot to make room for table
    plt.subplots_adjust(bottom=0.25)

# --- ADDED: Diagnosis Grouping Function --- 
def check_codes(dx_string, jia_pattern, uveitis_patterns):
    """Classifies a row based on JIA and Uveitis codes in a string."""
    if pd.isna(dx_string):
        return 'Other'
    codes = [code.strip() for code in str(dx_string).split(';')]
    has_jia = any(re.match(jia_pattern, code) for code in codes)
    has_uveitis = any(any(re.match(uv_pattern, code) for code in codes) for uv_pattern in uveitis_patterns)

    if has_jia and not has_uveitis:
        return 'JIA-Only'
    elif has_uveitis:  # Both JIA-U and Uveitis-Only combined into "Any Uveitis"
        return 'Any Uveitis'
    else:
        return 'Other'
# --- END ADDED ---

# --- ADDED: Advanced SVI distribution plot by group and quartile ---
def plot_svi_distribution_by_group_and_quartile(df, svi_measure_col, group_col, quartile_col, measure_name, group_order, output_dir):
    """Generate a point plot of an SVI measure by diagnosis group and SVI quartile."""
    print(f"Generating point plot for {measure_name} by {group_col} and {quartile_col}...")

    # Define the order for quartiles
    quartile_order = ["Q1", "Q2"] # Changed from Q1, Q2, Q3, Q4

    # Ensure the group column is categorical with the specified order
    if group_col in df.columns and group_order:
        df[group_col] = pd.Categorical(df[group_col], categories=group_order, ordered=True)
    
    # Ensure the quartile column is categorical with the specified order
    if quartile_col in df.columns:
        df[quartile_col] = pd.Categorical(df[quartile_col], categories=quartile_order, ordered=True)

    # Filter out rows where SVI measure, group, or quartile is NaN
    plot_df = df.dropna(subset=[svi_measure_col, group_col, quartile_col])

    if plot_df.empty:
        print(f"No data available to plot {measure_name} after NaN removal. Skipping plot.")
        return

    # Increase figure size for better label spacing
    plt.figure(figsize=(12, 8))
    ax = sns.pointplot(
        data=plot_df,
        x=quartile_col,
        y=svi_measure_col,
        hue=group_col,
        hue_order=group_order, # Use the defined group order for consistency
        dodge=True, # Separate points for different groups
        errorbar=('ci', 95), # Show 95% confidence intervals
        capsize=0.1 # Add caps to error bars
    )
    
    # Calculate group means for each quartile
    grouped_means = plot_df.groupby([quartile_col, group_col])[svi_measure_col].mean().unstack()
    
    # Store p-values in a dictionary for the legend
    p_values = {}
    
    # --- CALCULATE P-VALUES ---
    # 1. P-values comparing JIA-Only vs Any Uveitis within each quartile
    for i, quartile in enumerate(quartile_order):
        quartile_data = plot_df[plot_df[quartile_col] == quartile]
        
        # Skip if a quartile doesn't have data in both groups
        if not all(group in quartile_data[group_col].unique() for group in group_order):
            continue
            
        # Get data for each group within this quartile
        group_data = [quartile_data[quartile_data[group_col] == group][svi_measure_col].values 
                     for group in group_order]
                      
        # Make sure we have enough data for t-test
        if all(len(data) > 1 for data in group_data):
            t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
            p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            
            # Add significance indicators
            if p_val < 0.05:
                if p_val < 0.001:
                    p_text += "***"
                elif p_val < 0.01:
                    p_text += "**"
                else:
                    p_text += "*"
            
            # Store p-value for legend
            p_values[f"{quartile}: JIA-Only vs Any Uveitis"] = p_text
    
    # 2. P-values comparing Q1 vs Q2 for each diagnosis group
    for i, group in enumerate(group_order):
        group_data = plot_df[plot_df[group_col] == group]
        
        # Skip if group doesn't have data in both quartiles
        if not all(quartile in group_data[quartile_col].unique() for quartile in quartile_order):
            continue
            
        # Get data for each quartile within this group
        quartile_data = [group_data[group_data[quartile_col] == quartile][svi_measure_col].values 
                        for quartile in quartile_order]
                        
        # Make sure we have enough data for t-test
        if all(len(data) > 1 for data in quartile_data):
            t_stat, p_val = stats.ttest_ind(quartile_data[0], quartile_data[1], equal_var=False)
            p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            
            # Add significance indicators
            if p_val < 0.05:
                if p_val < 0.001:
                    p_text += "***"
                elif p_val < 0.01:
                    p_text += "**"
                else:
                    p_text += "*"
            
            # Store p-value for legend
            p_values[f"{group}: Q1 vs Q2"] = p_text
    
    # Calculate proper y-axis limits based on data
    y_data_min = plot_df[svi_measure_col].min()
    y_data_max = plot_df[svi_measure_col].max()
    y_range = y_data_max - y_data_min
    
    # Set limits with padding
    proper_y_min = max(0, y_data_min - y_range * 0.1)  # Don't go below 0 for SVI scores
    proper_y_max = y_data_max + y_range * 0.3  # Less space needed now without annotations
    plt.ylim(proper_y_min, proper_y_max)
    
    # Fix axis titles and labels
    plt.title(f"Mean {measure_name} by SVI Quartile and Diagnosis Group", fontsize=14, fontweight='bold')
    plt.xlabel("Quartile", fontsize=12)  # Remove redundant "SVI" in label
    plt.ylabel(f"Mean {measure_name} Score", fontsize=12)
    
    # Improve legend for diagnosis groups
    plt.legend(title="Diagnosis Group", loc='upper right', framealpha=0.9, fontsize=11)
    
    # Add p-value legend in top left corner
    if p_values:
        # Create the legend text
        legend_text = "Statistical Tests:\n"
        for key, value in p_values.items():
            legend_text += f"{key}: {value}\n"
        
        # Add text box with white background
        plt.figtext(0.01, 0.99, legend_text, 
                  ha='left', va='top', fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    # Improve grid appearance
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"{measure_name.replace(' ', '_')}_by_Group_and_Quartile_PointPlot.png")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')  # Added bbox_inches to avoid label cutoff
    plt.close()
    print(f"Saved plot to {filename}")

def plot_svi_component_correlation_heatmap(df, mean_cols, quartile_col, output_dir):
    """Plot correlation heatmaps of SVI components for each SVI Total Quartile."""
    print(f"Generating SVI component correlation heatmaps by {quartile_col}...")
    
    # Define the quartiles to iterate over
    quartiles = ["Q1", "Q2"] # Changed from Q1, Q2, Q3, Q4
    num_quartiles = len(quartiles)

    if num_quartiles == 0:
        print("No quartiles found to plot heatmaps.")
        return

    # Adjust subplot layout based on number of quartiles
    # For 2 quartiles, use a 1x2 layout
    if num_quartiles <= 2:
        fig, axes = plt.subplots(1, num_quartiles, figsize=(6 * num_quartiles, 5), squeeze=False) 
        axes = axes.flatten() # Ensure axes is always a 1D array
    else: # Fallback for more quartiles if logic changes later, though current target is 2
        # This part may need adjustment if more than 2 quartiles are ever used again
        nrows = (num_quartiles + 1) // 2
        ncols = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
        axes = axes.flatten()

    # Rename columns for better readability on the heatmap
    component_names = {
        f"svi_socioeconomic (list distinct)_mean": "SES",
        f"svi_household_comp (list distinct)_mean": "HHD",
        f"svi_housing_transportation (list distinct)_mean": "H&T",
        f"svi_minority (list distinct)_mean": "MIN"
    }

    for i, quartile_value in enumerate(quartiles):
        ax = axes[i]
        quartile_df = df[df[quartile_col] == quartile_value][mean_cols]
        
        if quartile_df.empty or quartile_df.shape[0] < 2: # Need at least 2 samples for correlation
            ax.set_title(f"{quartile_value} - No/Insufficient Data")
            ax.axis('off') # Turn off axis if no data
            continue
            
        # Rename columns for the current quartile's data
        plot_data = quartile_df.rename(columns=component_names)
        
        # Calculate correlation matrix
        corr_matrix = plot_data.corr()
        
        # Generate heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title(f"SVI Component Correlation ({quartile_value}, N={len(quartile_df)})")

    # Hide any unused subplots if num_quartiles < nrows*ncols (e.g. if 3 quartiles with 2x2 grid)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    filename = os.path.join(output_dir, f"SVI_Component_Correlations_by_{quartile_col}_Heatmap.png")
    plt.savefig(filename, dpi=DPI)
    plt.close()
    print(f"Saved SVI component correlation heatmaps to {filename}")

def main():
    # --- ADDED: Print start message ---
    print("--- main() function started ---") 
    
    # Read the CSV file
    input_file = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv" # Updated filename
    print(f"Reading data from {input_file}...")
    
    # Step 1: Read header and deduplicate column names
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        original_header = next(reader) # Read the first row
        header_names = _deduplicate_columns(original_header)
        print(f"Read header with {len(header_names)} columns.")
    
    # Step 2: Read full data with deduplicated header
    df = pd.read_csv(
        input_file,
        low_memory=False,
        header=0,
        names=header_names
    )
    
    # Display dataset info
    print(f"Dataset shape (original): {df.shape}")

    # --- ADDED: Apply Diagnosis Grouping ---
    if DX_CODE_COL in df.columns:
        print(f"Applying diagnosis grouping based on '{DX_CODE_COL}'...")
        df['diagnosis_group'] = df[DX_CODE_COL].apply(
            lambda x: check_codes(x, JIA_CODE_PATTERN, UVEITIS_CODE_PATTERNS)
        )
        
        # Filter to keep only the groups of interest
        groups_to_keep = ['JIA-Only', 'Any Uveitis']
        # --- ADDED: Define consistent order for diagnosis groups ---
        diagnosis_group_order = ['JIA-Only', 'Any Uveitis']
        # --- END ADDED ---
        original_count = len(df)
        df = df[df['diagnosis_group'].isin(groups_to_keep)].copy()
        print(f"Filtered dataframe to diagnosis groups: {groups_to_keep}. Shape: {df.shape}")
        print(f"Retained {len(df)} out of {original_count} patients.")
        print(f"Group counts:\n{df['diagnosis_group'].value_counts().to_string()}")
    else:
        print(f"Warning: Diagnosis code column '{DX_CODE_COL}' not found. Cannot apply grouping.")
        # Optionally create a dummy group if needed for plots to run
        # df['diagnosis_group'] = 'Unknown' 
        # --- ADDED: Define default order even if grouping fails ---
        diagnosis_group_order = ['JIA-Only', 'Any Uveitis']
        # --- END ADDED ---
    # --- END ADDED ---

    # Define SVI component columns
    svi_columns = [
        'svi_socioeconomic (list distinct)',
        'svi_household_comp (list distinct)',
        'svi_housing_transportation (list distinct)',
        'svi_minority (list distinct)'
    ]
    
    # Check which columns exist
    print("Checking for required SVI columns...") # Add check notification
    missing_cols = [col for col in svi_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: These SVI columns are missing: {missing_cols}") 
        print("Available columns that might contain SVI data:")
        for col in df.columns:
            if 'svi' in col.lower():
                print(f"  - {col}")
        # --- ADDED: Explicit exit message --- 
        print("--- Exiting main() because required SVI columns were not found. ---")
        return # Script exits here 
    else:
        print("All required SVI columns found.") # Add success notification

    # Calculate mean for each SVI component
    print("Calculating SVI component means...")
    for col in svi_columns:
        mean_col = f"{col}_mean"
        df[mean_col] = df[col].apply(parse_svi_values)
    
    # Calculate total SVI score (mean of all components)
    mean_cols = [f"{col}_mean" for col in svi_columns]
    df['SVI_total'] = df[mean_cols].mean(axis=1)
    
    # Create SVI_null category
    print("Creating SVI category for null values...")
    df['has_svi_data'] = df['SVI_total'].notna()
    print(f"Patients with SVI data: {df['has_svi_data'].sum()} out of {len(df)} ({df['has_svi_data'].mean()*100:.1f}%)")
    
    # Calculate quartiles for total SVI and components
    print("Calculating SVI quartiles...")
    df['SVI_quartile'] = calculate_quartiles(df['SVI_total'])
    
    for col in svi_columns:
        mean_col = f"{col}_mean"
        quartile_col = f"{mean_col}_quartile"
        df[quartile_col] = calculate_quartiles(df[mean_col])
    
    # Save processed data to CSV (now includes diagnosis_group)
    output_file = "SVI_processed_data_with_groups.csv" # Changed filename
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    print("Generating summary statistics...")
    
    # Summary table for SVI components (Overall)
    svi_summary = df[mean_cols + ['SVI_total']].describe().T
    svi_summary = svi_summary.rename(index={
        f"svi_socioeconomic (list distinct)_mean": "Socioeconomic Status",
        f"svi_household_comp (list distinct)_mean": "Household Composition & Disability",
        f"svi_housing_transportation (list distinct)_mean": "Housing & Transportation",
        f"svi_minority (list distinct)_mean": "Minority Status & Language",
        "SVI_total": "Overall SVI"
    })
    print("\nOverall SVI Component Summary Statistics:") # Clarified title
    print(svi_summary.round(3))
    svi_summary.to_csv(f"{OUTPUT_DIR}/svi_summary_statistics_overall.csv") # Changed filename

    # --- ADDED: Summary statistics by diagnosis group --- 
    print("\nGenerating SVI summary statistics by diagnosis group...")
    group_summary = df.groupby('diagnosis_group')[mean_cols + ['SVI_total']].describe()
    
    # Improve multi-index display for printing/saving if needed
    # Example: group_summary.unstack() or adjust column names
    print(group_summary.round(3))
    
    # Save grouped summary to CSV
    group_summary_file = f"{OUTPUT_DIR}/SVI_summary_by_group.csv"
    group_summary.to_csv(group_summary_file)
    print(f"Saved grouped summary statistics to {group_summary_file}")
    # --- END ADDED ---
    
    # Summary of quartile distribution (Overall)
    quartile_cols = ['SVI_quartile'] + [f"{col}_mean_quartile" for col in svi_columns]
    
    print("\nSVI Quartile Distribution:")
    for col in quartile_cols:
        counts = df[col].value_counts().sort_index()
        print(f"\n{col}:")
        print(counts)
        percentages = 100 * counts / counts.sum()
        print("Percentages:")
        for idx, pct in percentages.items():
            print(f"  {idx}: {pct:.1f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Figure 1: Box plots for all SVI components
    plt.figure(figsize=(16, 9)) # Adjusted size for annotations
    
    # Create more interpretable column names for plotting
    plot_df = df[mean_cols + ['SVI_total', 'diagnosis_group']].copy() # Include diagnosis_group
    plot_df.columns = [
        "Socioeconomic Status",
        "Household & Disability",
        "Housing & Transportation",
        "Minority Status",
        "Overall SVI",
        "diagnosis_group" # Add group name
    ]
    
    # Melt the data BEFORE plotting
    plot_df_melted = pd.melt(plot_df, id_vars=['diagnosis_group'], 
                             value_vars=["Socioeconomic Status", "Household & Disability", 
                                         "Housing & Transportation", "Minority Status", "Overall SVI"],
                             var_name='Component', 
                             value_name='Score')

    # Create box plot using MELTED data
    component_order = ["Socioeconomic Status", "Household & Disability", 
                       "Housing & Transportation", "Minority Status", "Overall SVI"]
    ax = sns.boxplot(data=plot_df_melted, 
                     x='Component', 
                     y='Score',      
                     hue='diagnosis_group', 
                     hue_order=diagnosis_group_order, # Explicitly set hue order
                     palette="Set2", 
                     medianprops=dict(color="black", alpha=0.9),
                     boxprops=dict(alpha=0.8), 
                     showfliers=False, 
                     order=component_order 
                    )

    # Add individual data points using MELTED data
    sns.stripplot(data=plot_df_melted, 
                  x='Component', 
                  y='Score',     
                  hue='diagnosis_group',
                  hue_order=diagnosis_group_order, # Explicitly set hue order
                  color='black', alpha=0.15, size=3, dodge=True, ax=ax, # Reduced alpha
                  order=component_order 
                 )
                 
    # --- Calculate max value FIRST --- 
    y_max = plot_df_melted['Score'].max() # Get overall max y for positioning
    y_offset = y_max * 0.05 # Offset for text above plots
    
    # --- ADDED: Mean Value Annotations for Each Box --- 
    # Calculate group means for each component
    group_means = plot_df_melted.groupby(['Component', 'diagnosis_group'])['Score'].mean()
    
    # Get positional information about boxes for text placement
    # The number of hue levels and components determines box positions
    num_groups = len(df['diagnosis_group'].unique())
    num_components = len(component_order)
    width = 0.8  # Approximate width of the group of boxes for each component
    group_width = width / num_groups  # Width allocated to each diagnosis group
    
    # Get unique diagnosis groups in the specified order
    groups = diagnosis_group_order # Use the defined order
    
    # Place mean values on top of each box
    for i, component in enumerate(component_order):
        for j, group in enumerate(groups):
            if (component, group) in group_means.index:
                mean_val = group_means.loc[(component, group)]
                
                # Calculate x position (centered over each box)
                # For each component index, adjust by group offset
                # The formula creates a position that's centered over each boxplot element
                x_pos = i + (j - (num_groups-1)/2) * group_width
                
                # Get the top of the box (you could also use the median or other metric)
                # Find the 75th percentile (top of box) for this component-group
                q3 = plot_df_melted[(plot_df_melted['Component'] == component) & 
                                   (plot_df_melted['diagnosis_group'] == group)]['Score'].quantile(0.75)
                
                # Add some space above the box
                y_pos = q3 + (y_max - q3) * 0.10
                
                # Add the text annotation
                ax.text(x_pos, y_pos, 
                        f"{mean_val:.3f}", 
                        ha='center', va='bottom', fontsize=8, 
                        fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    # --- END ADDED --- 
    
    # --- ADDED: Statistical Annotations (Kruskal-Wallis) --- 
    for i, component in enumerate(component_order):
        # Get data for the current component
        component_data = plot_df_melted[plot_df_melted['Component'] == component]
        
        # Prepare data for Kruskal-Wallis (list of arrays, one per group)
        groups = [group_data['Score'].dropna().values 
                  for name, group_data in component_data.groupby('diagnosis_group')]
        
        # Ensure we have enough groups with data
        groups_with_data = [g for g in groups if len(g) > 0]
        if len(groups_with_data) >= 2: # Need at least 2 groups to compare
            try:
                stat, p_val = stats.kruskal(*groups_with_data)
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                # Add text annotation above the component group
                ax.text(i, y_max + y_offset, p_text, ha='center', va='bottom', fontsize=9)
            except ValueError as e:
                 print(f"Skipping Kruskal-Wallis for {component} due to error: {e}")
                 ax.text(i, y_max + y_offset, "p=N/A", ha='center', va='bottom', fontsize=9)
        else:
            ax.text(i, y_max + y_offset, "p=N/A", ha='center', va='bottom', fontsize=9)

    plt.title('SVI Component Distributions by Diagnosis Group', fontsize=14, fontweight='bold') # Updated title
    plt.ylabel('SVI Score', fontsize=12)
    plt.xlabel('') # Remove x-axis label as categories are clear
    plt.xticks(rotation=15, ha='right') 
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust y-limit to make space for annotations
    ax.set_ylim(top=y_max + y_offset * 2) 
    
    # Ensure legend handles dodged stripplot points correctly
    handles, labels = ax.get_legend_handles_labels()
    num_groups = len(df['diagnosis_group'].unique())
    ax.legend(handles[:num_groups], labels[:num_groups], title='Diagnosis Group', bbox_to_anchor=(1.02, 1), loc='upper left') # Move legend outside
     
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout further for legend
    plt.savefig(f"{OUTPUT_DIR}/svi_components_boxplot_by_group.png", dpi=DPI, bbox_inches='tight') 
    plt.close() 
    
    # Figure 2: Quartile distribution bar plot for total SVI
    # --- MODIFIED: Create grouped bar chart for quartiles --- 
    plt.figure(figsize=(12, 8))
    # Calculate counts per group and quartile
    quartile_group_counts = df.groupby('diagnosis_group')['SVI_quartile'].value_counts().unstack(fill_value=0)
    
    # Reorder the index to match our desired order
    quartile_group_counts = quartile_group_counts.reindex(diagnosis_group_order)
    # --- Added: Get raw counts for labels --- 
    raw_counts = quartile_group_counts.copy()
    # --- End Added ---
    
    # Normalize to percentages within each group
    quartile_group_percentages = quartile_group_counts.apply(lambda x: 100 * x / float(x.sum()), axis=1)
    
    ax = quartile_group_percentages.plot(kind='bar', 
                                       figsize=(12, 8), 
                                       rot=0, 
                                       colormap="viridis", 
                                       width=0.8)

    plt.title('Total SVI Quartile Distribution by Diagnosis Group', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Patients within Group (%)', fontsize=12)
    plt.xlabel('Diagnosis Group', fontsize=12)
    plt.legend(title='SVI Quartile', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels AND N counts below bars
    # Determine positioning based on number of groups and bars
    num_groups = len(raw_counts.index)
    num_categories = len(raw_counts.columns)
    group_width = 0.8 / num_categories # Width allocated to each SVI quartile within a group
    bar_offset = -0.4 # Starting offset for the first bar group
    
    # Add N counts below the x-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_pos_n_count = y_min - y_range * 0.08 # Position N counts below the axis

    for i, group in enumerate(raw_counts.index):
        for j, quartile in enumerate(raw_counts.columns):
            count = raw_counts.loc[group, quartile]
            # Calculate x position for the N count text, centered under the bar group
            x_pos = i + bar_offset + (j + 0.5) * group_width
            ax.text(x_pos, y_pos_n_count, f"N={count}", 
                    ha='center', va='top', fontsize=8, color='black')

    # Add percentage labels inside bars
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        if height > 1: # Only label significant bars
            ax.text(x + width/2, 
                    y + height/2, 
                    f'{height:.1f}%', 
                    ha='center', 
                    va='center', 
                    fontsize=9,
                    color='white', # Use white for visibility on dark bars
                    fontweight='bold')
    
    # --- ADD: P-value calculations and annotations ---
    # 1. Compare distribution of Q1 vs Q2 within each diagnosis group (chi-square test)
    for i, group in enumerate(diagnosis_group_order):
        group_data = df[df['diagnosis_group'] == group]
        q1_count = len(group_data[group_data['SVI_quartile'] == 'Q1'])
        q2_count = len(group_data[group_data['SVI_quartile'] == 'Q2'])
        
        # Create contingency table for this group [Q1, Q2]
        contingency_table = np.array([[q1_count, q2_count]])
        
        # Expected equal distribution would be 50/50
        expected_counts = np.array([[sum(contingency_table[0])/2, sum(contingency_table[0])/2]])
        
        # Chi-square test (if we have sufficient data)
        if q1_count >= 5 and q2_count >= 5:  # Minimum expected cell count for reliable chi-square
            chi2, p_val = stats.chisquare(contingency_table[0], expected_counts[0])
            p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            
            # Format p-text with asterisks for significance
            if p_val < 0.05:
                if p_val < 0.001:
                    p_text += "***"
                elif p_val < 0.01:
                    p_text += "**"
                else:
                    p_text += "*"
            
            # Get max height for this group's bars
            max_height = quartile_group_percentages.loc[group].max()
            
            # Add p-value above the bars
            y_pos = max_height + y_range * 0.05
            ax.text(i, y_pos, f"Q1 vs Q2: {p_text}", 
                  ha='center', va='bottom', fontsize=9)
    
    # 2. Compare each quartile proportion between JIA-Only and Any Uveitis (proportion z-test)
    if len(diagnosis_group_order) == 2:
        # For each quartile, compare proportion between groups
        for j, quartile in enumerate(["Q1", "Q2"]):
            group1 = diagnosis_group_order[0]  # JIA-Only
            group2 = diagnosis_group_order[1]  # Any Uveitis
            
            # Get counts and total
            count1 = raw_counts.loc[group1, quartile]
            total1 = raw_counts.loc[group1].sum()
            prop1 = count1 / total1
            
            count2 = raw_counts.loc[group2, quartile]
            total2 = raw_counts.loc[group2].sum()
            prop2 = count2 / total2
            
            # Calculate z-score for proportion comparison
            if count1 >= 5 and count2 >= 5 and (total1-count1) >= 5 and (total2-count2) >= 5:
                # Two-sample z-test for proportions
                # z = (p1 - p2) / sqrt(p*(1-p)*(1/n1 + 1/n2)) where p = (count1 + count2)/(total1 + total2)
                p_pooled = (count1 + count2) / (total1 + total2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total1 + 1/total2))
                z_score = (prop1 - prop2) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed p-value
                
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                
                # Format p-text with asterisks for significance
                if p_val < 0.05:
                    if p_val < 0.001:
                        p_text += "***"
                    elif p_val < 0.01:
                        p_text += "**"
                    else:
                        p_text += "*"
                
                # Position the annotation centered between groups but offset toward quartile
                x_pos = 0.5  # Center between diagnosis groups
                
                # Y position should be above both bars
                y_position = y_max + y_range * (0.1 + j * 0.05)  # Staggered for Q1 vs Q2
                
                # Add an arrow pointing to the quartile
                ax.annotate(f"{quartile} (groups): {p_text}", 
                            xy=(x_pos, y_position - y_range*0.03), 
                            xytext=(x_pos, y_position),
                            ha='center', va='center', fontsize=9,
                            arrowprops=dict(arrowstyle='-', color='black', lw=1))
    
    # Adjust y-limit to make room for all annotations
    plt.ylim(top=y_max + y_range * 0.3)
    
    plt.subplots_adjust(bottom=0.15, right=0.85) # Adjust layout for N counts and legend
    # plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(f"{OUTPUT_DIR}/svi_total_quartile_bars_by_group.png", dpi=DPI, bbox_inches='tight') # New filename
    plt.close() # Close figure
    # --- END MODIFIED ---

    # --- Consider modifying other plots similarly if needed ---
    # Example: Quartile distribution for other components 
    for svi_col in svi_columns:
        quartile_col = f"{svi_col}_mean_quartile"
        mean_col = f"{svi_col}_mean"
        short_name = svi_col.replace('svi_','').replace(' (list distinct)','').replace('_',' ').title()
        
        if quartile_col in df.columns:
            plt.figure(figsize=(12, 8))
            group_counts = df.groupby('diagnosis_group')[quartile_col].value_counts().unstack(fill_value=0)
            # Reorder the index to match our desired order
            group_counts = group_counts.reindex(diagnosis_group_order)
            # --- Added: Get raw counts for labels --- 
            raw_counts_comp = group_counts.copy()
            # --- End Added ---
            group_percentages = group_counts.apply(lambda x: 100 * x / float(x.sum()), axis=1)
            ax = group_percentages.plot(kind='bar', figsize=(12, 8), rot=0, colormap="viridis", width=0.8)

            plt.title(f'{short_name} SVI Quartile Distribution by Diagnosis Group', fontsize=14, fontweight='bold')
            plt.ylabel('Percentage of Patients within Group (%)', fontsize=12)
            plt.xlabel('Diagnosis Group', fontsize=12)
            plt.legend(title=f'{short_name} Quartile', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            
            # Add N counts below the x-axis
            y_min_comp, y_max_comp = ax.get_ylim()
            y_range_comp = y_max_comp - y_min_comp
            y_pos_n_count_comp = y_min_comp - y_range_comp * 0.08
            
            num_groups_comp = len(raw_counts_comp.index)
            num_categories_comp = len(raw_counts_comp.columns)
            group_width_comp = 0.8 / num_categories_comp 
            bar_offset_comp = -0.4
            
            for i, group in enumerate(raw_counts_comp.index):
                 for j, quartile in enumerate(raw_counts_comp.columns):
                     count = raw_counts_comp.loc[group, quartile]
                     x_pos = i + bar_offset_comp + (j + 0.5) * group_width_comp
                     ax.text(x_pos, y_pos_n_count_comp, f"N={count}", 
                             ha='center', va='top', fontsize=8, color='black')

            # Add percentage labels inside bars
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                if height > 1:
                    ax.text(x + width/2, y + height/2, f'{height:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
            # --- ADD: P-value calculations and annotations ---
            # 1. Compare distribution of Q1 vs Q2 within each diagnosis group (chi-square test)
            for i, group in enumerate(diagnosis_group_order):
                group_data = df[df['diagnosis_group'] == group]
                q1_count = len(group_data[group_data[quartile_col] == 'Q1'])
                q2_count = len(group_data[group_data[quartile_col] == 'Q2'])
                
                # Create contingency table for this group [Q1, Q2]
                contingency_table = np.array([[q1_count, q2_count]])
                
                # Expected equal distribution would be 50/50
                expected_counts = np.array([[sum(contingency_table[0])/2, sum(contingency_table[0])/2]])
                
                # Chi-square test (if we have sufficient data)
                if q1_count >= 5 and q2_count >= 5:  # Minimum expected cell count for reliable chi-square
                    chi2, p_val = stats.chisquare(contingency_table[0], expected_counts[0])
                    p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                    
                    # Format p-text with asterisks for significance
                    if p_val < 0.05:
                        if p_val < 0.001:
                            p_text += "***"
                        elif p_val < 0.01:
                            p_text += "**"
                        else:
                            p_text += "*"
                    
                    # Get max height for this group's bars
                    max_height = group_percentages.loc[group].max()
                    
                    # Add p-value above the bars
                    y_pos = max_height + y_range_comp * 0.05
                    ax.text(i, y_pos, f"Q1 vs Q2: {p_text}", 
                          ha='center', va='bottom', fontsize=9)
            
            # 2. Compare each quartile proportion between JIA-Only and Any Uveitis (proportion z-test)
            if len(diagnosis_group_order) == 2:
                # For each quartile, compare proportion between groups
                for j, quartile in enumerate(["Q1", "Q2"]):
                    group1 = diagnosis_group_order[0]  # JIA-Only
                    group2 = diagnosis_group_order[1]  # Any Uveitis
                    
                    # Get counts and total
                    count1 = raw_counts_comp.loc[group1, quartile] if quartile in raw_counts_comp.columns else 0
                    total1 = raw_counts_comp.loc[group1].sum()
                    prop1 = count1 / total1 if total1 > 0 else 0
                    
                    count2 = raw_counts_comp.loc[group2, quartile] if quartile in raw_counts_comp.columns else 0
                    total2 = raw_counts_comp.loc[group2].sum()
                    prop2 = count2 / total2 if total2 > 0 else 0
                    
                    # Calculate z-score for proportion comparison
                    if count1 >= 5 and count2 >= 5 and (total1-count1) >= 5 and (total2-count2) >= 5:
                        # Two-sample z-test for proportions
                        p_pooled = (count1 + count2) / (total1 + total2)
                        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total1 + 1/total2))
                        z_score = (prop1 - prop2) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed p-value
                        
                        p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                        
                        # Format p-text with asterisks for significance
                        if p_val < 0.05:
                            if p_val < 0.001:
                                p_text += "***"
                            elif p_val < 0.01:
                                p_text += "**"
                            else:
                                p_text += "*"
                        
                        # Position the annotation centered between groups but offset toward quartile
                        x_pos = 0.5  # Center between diagnosis groups
                        
                        # Y position should be above both bars
                        y_position = y_max_comp + y_range_comp * (0.1 + j * 0.05)  # Staggered for Q1 vs Q2
                        
                        # Add an arrow pointing to the quartile
                        ax.annotate(f"{quartile} (groups): {p_text}", 
                                    xy=(x_pos, y_position - y_range_comp*0.03), 
                                    xytext=(x_pos, y_position),
                                    ha='center', va='center', fontsize=9,
                                    arrowprops=dict(arrowstyle='-', color='black', lw=1))
            
            # Adjust y-limit to make room for all annotations
            plt.ylim(top=y_max_comp + y_range_comp * 0.3)
            
            plt.subplots_adjust(bottom=0.15, right=0.85) # Adjust layout for N counts and legend
            # plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(f"{OUTPUT_DIR}/{short_name.replace(' ','_')}_quartile_bars_by_group.png", dpi=DPI, bbox_inches='tight')
            plt.close()

    print("\nVisualization generation complete.")

    # Define component_name_map for readable names in t-tests
    # Ensure mean_cols is defined before this point in main()
    component_name_map = {
        mean_cols[0]: "Socioeconomic Status",
        mean_cols[1]: "Household Composition & Disability",
        mean_cols[2]: "Housing & Transportation",
        mean_cols[3]: "Minority Status & Language"
    }
    
    # --- ADDED: Statistical tests for SVI differences between quartiles (Q1 vs Q2) and groups ---
    print("\n--- Statistical Tests ---")
    quartile_order_for_tests = ["Q1", "Q2"] # Explicitly for 2 quartiles

    # T-test for SVI Total: Q1 vs Q2 (overall)
    if 'SVI_quartile' in df.columns:
        q1_total_svi = df[df['SVI_quartile'] == 'Q1']['SVI_total'].dropna()
        q2_total_svi = df[df['SVI_quartile'] == 'Q2']['SVI_total'].dropna()
        if len(q1_total_svi) > 1 and len(q2_total_svi) > 1:
            t_stat, p_val = stats.ttest_ind(q1_total_svi, q2_total_svi, equal_var=False)
            print(f"T-test SVI Total (Q1 vs Q2, Overall): t={t_stat:.3f}, p={p_val:.4f}")
        else:
            print("T-test SVI Total (Q1 vs Q2, Overall): Not enough data in Q1 and/or Q2.")

    # T-tests for SVI components: Q1 vs Q2 (overall)
    for col_name, readable_name in component_name_map.items():
        quartile_col_for_component = f"{col_name}_quartile"
        if quartile_col_for_component in df.columns:
            q1_comp_svi = df[df[quartile_col_for_component] == 'Q1'][col_name].dropna()
            q2_comp_svi = df[df[quartile_col_for_component] == 'Q2'][col_name].dropna()
            if len(q1_comp_svi) > 1 and len(q2_comp_svi) > 1:
                t_stat, p_val = stats.ttest_ind(q1_comp_svi, q2_comp_svi, equal_var=False)
                print(f"T-test {readable_name} (Q1 vs Q2, Overall): t={t_stat:.3f}, p={p_val:.4f}")
            else:
                print(f"T-test {readable_name} (Q1 vs Q2, Overall): Not enough data in Q1 and/or Q2 for component.")
        else:
            print(f"Warning: Quartile column {quartile_col_for_component} not found for t-test.")


    # T-tests for SVI Total and Components between Diagnosis Groups (JIA-Only vs Any Uveitis)
    if 'diagnosis_group' in df.columns and len(df['diagnosis_group'].unique()) > 1:
        jia_only_df = df[df['diagnosis_group'] == 'JIA-Only']
        any_uveitis_df = df[df['diagnosis_group'] == 'Any Uveitis']

        if not jia_only_df.empty and not any_uveitis_df.empty:
            # SVI Total
            jia_svi_total = jia_only_df['SVI_total'].dropna()
            uveitis_svi_total = any_uveitis_df['SVI_total'].dropna()
            if len(jia_svi_total) > 1 and len(uveitis_svi_total) > 1:
                t_stat, p_val = stats.ttest_ind(jia_svi_total, uveitis_svi_total, equal_var=False)
                print(f"T-test SVI Total (JIA-Only vs Any Uveitis): t={t_stat:.3f}, p={p_val:.4f}")
            else:
                print("T-test SVI Total (JIA-Only vs Any Uveitis): Insufficient data.")

            # SVI Components
            for col_name, readable_name in component_name_map.items():
                jia_comp_data = jia_only_df[col_name].dropna()
                uveitis_comp_data = any_uveitis_df[col_name].dropna()
                if len(jia_comp_data) > 1 and len(uveitis_comp_data) > 1:
                    t_stat, p_val = stats.ttest_ind(jia_comp_data, uveitis_comp_data, equal_var=False)
                    print(f"T-test {readable_name} (JIA-Only vs Any Uveitis): t={t_stat:.3f}, p={p_val:.4f}")
                else:
                    print(f"T-test {readable_name} (JIA-Only vs Any Uveitis): Insufficient data for component.")
        else:
            print("Comparison between JIA-Only and Any Uveitis not possible: one or both groups missing/empty.")
    else:
        print("Diagnosis group comparisons not performed: 'diagnosis_group' column missing or only one group present.")
    # --- END ADDED ---

    # --- Call the advanced plot function for Overall SVI ---
    if 'SVI_total' in df.columns and 'diagnosis_group' in df.columns and 'SVI_quartile' in df.columns:
         plot_svi_distribution_by_group_and_quartile(
             df, 
             svi_measure_col='SVI_total', 
             group_col='diagnosis_group', 
             quartile_col='SVI_quartile', # Use the SVI Total quartile
             measure_name="Overall SVI", 
             group_order=diagnosis_group_order, 
             output_dir=OUTPUT_DIR
         )
    else:
        print("Skipping Overall SVI distribution plot by group and quartile due to missing columns.")
    # --- END Plot Call ---

    # --- Call the heatmap function ---
    if all(col in df.columns for col in mean_cols) and 'SVI_quartile' in df.columns:
        plot_svi_component_correlation_heatmap(df, mean_cols, 'SVI_quartile', OUTPUT_DIR)
    else:
        print("Skipping SVI component correlation heatmap due to missing SVI mean columns or SVI_quartile column.")
    # --- END Heatmap Call ---
    
    print("\n--- main() function finished successfully ---")

if __name__ == "__main__":
    main() 