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
    
    if len(non_na) < 4:  # Not enough data for quartiles
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Calculate quartile boundaries
    quartiles = pd.qcut(non_na, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    
    # Create a Series with NaN for missing values - use object dtype to avoid the warning
    result = pd.Series([None] * len(series), index=series.index, dtype='object')
    result[non_na.index] = quartiles
    
    return result

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
    elif has_jia and has_uveitis:
        return 'JIA-U'
    elif has_uveitis and not has_jia: # Explicitly check for not having JIA
        return 'Uveitis-Only'
    else:
        return 'Other'
# --- END ADDED ---

def main():
    # --- ADDED: Print start message ---
    print("--- main() function started ---") 
    
    # Read the CSV file
    input_file = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated.csv"
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
        groups_to_keep = ['JIA-Only', 'JIA-U', 'Uveitis-Only']
        # --- ADDED: Define consistent order for diagnosis groups ---
        diagnosis_group_order = ['JIA-Only', 'JIA-U', 'Uveitis-Only']
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
        diagnosis_group_order = ['JIA-Only', 'JIA-U', 'Uveitis-Only']
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
    quartile_group_percentages = quartile_group_counts.apply(lambda x: 100 * x / float(x.sum()) if x.sum() > 0 else 0, axis=1)
    
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
    num_categories = len(raw_counts.columns) # Should be 4 quartiles
    group_width = 0.8 / num_categories # Width allocated to each SVI quartile within a diagnosis group bar
    bar_offset = -0.4 # Starting offset for the first bar group (Q1)
    
    # Add N counts below the x-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_pos_n_count = y_min - y_range * 0.08 # Position N counts below the axis

    for i, group in enumerate(raw_counts.index): # Loop through diagnosis groups (JIA-Only, JIA-U, Uveitis-Only)
        for j, quartile in enumerate(raw_counts.columns): # Loop through quartiles (Q1, Q2, Q3, Q4)
            count = raw_counts.loc[group, quartile]
            # Calculate x position for the N count text, centered under the specific bar
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
            group_percentages = group_counts.apply(lambda x: 100 * x / float(x.sum()) if x.sum() > 0 else 0, axis=1)
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
            
            num_groups_comp = len(raw_counts_comp.index) # Should be 3
            num_categories_comp = len(raw_counts_comp.columns) # Should be 4
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
            
            plt.subplots_adjust(bottom=0.15, right=0.85) # Adjust layout for N counts and legend
            # plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(f"{OUTPUT_DIR}/{short_name.replace(' ','_')}_quartile_bars_by_group.png", dpi=DPI, bbox_inches='tight')
            plt.close()

    print("\nVisualization generation complete.")
    
    # --- ADDED: Print end message ---
    print("--- main() function finished successfully ---")

if __name__ == "__main__":
    main() 