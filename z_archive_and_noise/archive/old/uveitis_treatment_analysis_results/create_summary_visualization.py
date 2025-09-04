import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

# Set styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create results directory
results_dir = 'uveitis_treatment_analysis_results'
os.makedirs(results_dir, exist_ok=True)

# Load the dataset and treatment timeline data
print("Loading datasets...")
df = pd.read_csv('SVI_filtered_495_patients.csv')
try:
    timeline_df = pd.read_csv(f'{results_dir}/uveitis_treatment_timeline_simplified.csv')
    has_timeline_data = True
except FileNotFoundError:
    has_timeline_data = False
    print("No treatment timeline data found. Will skip treatment time analysis.")

# Create uveitis indicator column
uveitis_indicators = ['diagnosis of uveitis', 'uveitis curr', 'uveitis curr fup']
df['has_uveitis'] = df[uveitis_indicators].notna().any(axis=1)

# Create a comprehensive dashboard
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig)

# 1. Uveitis prevalence by SVI quartile
ax1 = fig.add_subplot(gs[0, 0])
prevalence_by_quartile = df.groupby('SVI_quartile')['has_uveitis'].mean() * 100
prevalence_by_quartile = prevalence_by_quartile.reindex(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
bars = sns.barplot(x=prevalence_by_quartile.index, y=prevalence_by_quartile.values, ax=ax1)

# Add value labels on top of bars
for i, v in enumerate(prevalence_by_quartile.values):
    ax1.text(i, v + 1, f"{v:.1f}%", ha='center')

ax1.set_title('Uveitis Prevalence by SVI Quartile', fontsize=14)
ax1.set_xlabel('SVI Quartile', fontsize=12)
ax1.set_ylabel('Prevalence (%)', fontsize=12)
ax1.set_ylim(0, max(prevalence_by_quartile.values) * 1.2)

# 2. Time to treatment by SVI quartile (if data available)
ax2 = fig.add_subplot(gs[0, 1])
if has_timeline_data:
    sns.boxplot(x='SVI_quartile', y='years_to_treatment', data=timeline_df, 
                order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], ax=ax2)
    
    # Add value labels for median
    medians = timeline_df.groupby('SVI_quartile')['years_to_treatment'].median()
    for i, quartile in enumerate(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']):
        if quartile in medians.index:
            ax2.text(i, medians[quartile] + 0.05, f"Median: {medians[quartile]:.1f}", ha='center')
    
    ax2.set_title('Time to Treatment by SVI Quartile', fontsize=14)
    ax2.set_xlabel('SVI Quartile', fontsize=12)
    ax2.set_ylabel('Years from Diagnosis to Treatment', fontsize=12)
else:
    ax2.text(0.5, 0.5, 'No treatment timeline data available', 
             ha='center', va='center', fontsize=14)
    ax2.axis('off')

# 3. Patient counts
ax3 = fig.add_subplot(gs[1, 0])
uveitis_counts = df.groupby('SVI_quartile')['has_uveitis'].sum().astype(int)
total_counts = df.groupby('SVI_quartile').size()
non_uveitis_counts = total_counts - uveitis_counts

# Reorder for consistent display
order = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
uveitis_counts = uveitis_counts.reindex(order)
non_uveitis_counts = non_uveitis_counts.reindex(order)

# Create stacked bar chart
width = 0.6
ax3.bar(order, uveitis_counts, width, label='Uveitis')
ax3.bar(order, non_uveitis_counts, width, bottom=uveitis_counts, label='No Uveitis')

# Add count labels
for i, (u, t) in enumerate(zip(uveitis_counts, total_counts)):
    ax3.text(i, u/2, f"{u}", ha='center', va='center', color='white', fontweight='bold')
    ax3.text(i, u + (t-u)/2, f"{t-u}", ha='center', va='center', color='white', fontweight='bold')
    ax3.text(i, t + 2, f"Total: {t}", ha='center', fontweight='bold')

ax3.set_title('Patient Counts by SVI Quartile', fontsize=14)
ax3.set_xlabel('SVI Quartile', fontsize=12)
ax3.set_ylabel('Number of Patients', fontsize=12)
ax3.legend()

# 4. Treatment distribution
ax4 = fig.add_subplot(gs[1, 1])
if has_timeline_data:
    # Distribution of time to treatment
    years_counts = timeline_df['years_to_treatment'].value_counts().sort_index()
    bars = sns.barplot(x=years_counts.index, y=years_counts.values, ax=ax4)
    
    # Add value labels on top of bars
    for i, v in enumerate(years_counts.values):
        ax4.text(i, v + 1, str(v), ha='center')
    
    ax4.set_title('Distribution of Time to Treatment', fontsize=14)
    ax4.set_xlabel('Years from Diagnosis to Treatment', fontsize=12)
    ax4.set_ylabel('Number of Patients', fontsize=12)
    ax4.set_ylim(0, max(years_counts.values) * 1.2)
else:
    ax4.text(0.5, 0.5, 'No treatment timeline data available', 
             ha='center', va='center', fontsize=14)
    ax4.axis('off')

# Add overall title
fig.suptitle('Uveitis and Social Vulnerability Analysis', fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the dashboard
output_path = f"{results_dir}/uveitis_svi_dashboard.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Dashboard saved to {output_path}")

# Create a more detailed visualization of treatment times if data is available
if has_timeline_data:
    plt.figure(figsize=(14, 8))
    
    # Create a violin plot with boxplot inside
    ax = sns.violinplot(x='SVI_quartile', y='years_to_treatment', data=timeline_df, 
                     order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], inner='box', 
                     palette='viridis')
    
    # Add individual points for better visibility
    sns.stripplot(x='SVI_quartile', y='years_to_treatment', data=timeline_df,
               order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], 
               color='black', alpha=0.5, jitter=True)
    
    # Add statistics to the plot
    quartile_stats = timeline_df.groupby('SVI_quartile')['years_to_treatment'].agg(['mean', 'median', 'count'])
    
    for i, quartile in enumerate(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']):
        if quartile in quartile_stats.index:
            stats = quartile_stats.loc[quartile]
            plt.text(i, -0.15, 
                    f"Mean: {stats['mean']:.2f}\nMedian: {stats['median']:.1f}\nn: {stats['count']}", 
                    ha='center', va='top', fontsize=10)
    
    plt.title('Treatment Timing by SVI Quartile', fontsize=16)
    plt.xlabel('SVI Quartile', fontsize=14)
    plt.ylabel('Years from Diagnosis to Treatment', fontsize=14)
    plt.tight_layout()
    
    # Save the detailed plot
    output_path = f"{results_dir}/uveitis_treatment_timing_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed treatment timing plot saved to {output_path}")

print("Visualizations complete!") 