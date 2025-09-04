import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

# Load the data
df = pd.read_csv('JIA_onset_analysis.csv')

# Filter data for valid ages (between 0 and 20)
df = df[(df['combined_onset_age'] >= 0) & (df['combined_onset_age'] <= 20)]

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# SUBPLOT 1: Original arth_onset_age data
sns.boxplot(x='SVI_quartile', y='arth_onset_age', data=df, ax=ax1, 
            order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
ax1.set_title('Age at JIA Onset by SVI Quartile \n(Using Arthritis Onset Date)', fontsize=12)
ax1.set_xlabel('Social Vulnerability Index Quartile', fontsize=10)
ax1.set_ylabel('Age at JIA Onset (years)', fontsize=10)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

# Add sample sizes to x-axis labels
counts = df.groupby('SVI_quartile')['arth_onset_age'].count()
xlabels = [f'{q}\n(n={counts[q]})' if q in counts else q for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]
ax1.set_xticklabels(xlabels)

# SUBPLOT 2: Combined onset age data
sns.boxplot(x='SVI_quartile', y='combined_onset_age', data=df, ax=ax2, 
            order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
ax2.set_title('Age at JIA Onset by SVI Quartile \n(Combined Onset Data)', fontsize=12)
ax2.set_xlabel('Social Vulnerability Index Quartile', fontsize=10)
ax2.set_ylabel('Age at JIA Onset (years)', fontsize=10)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

# Add sample sizes to x-axis labels
counts = df.groupby('SVI_quartile')['combined_onset_age'].count()
xlabels = [f'{q}\n(n={counts[q]})' if q in counts else q for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]
ax2.set_xticklabels(xlabels)

# Add statistical test results
# Perform ANOVA
groups = [df[df['SVI_quartile'] == q]['combined_onset_age'].dropna() for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]
f_stat, p_value = stats.f_oneway(*groups)
anova_text = f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}'

# Perform t-test between Q1 and Q4
q1 = df[df['SVI_quartile'] == 'Q1 (Low)']['combined_onset_age'].dropna()
q4 = df[df['SVI_quartile'] == 'Q4 (High)']['combined_onset_age'].dropna()
t_stat, p_value_t = stats.ttest_ind(q1, q4)
ttest_text = f'Q1 vs Q4 t-test: t={t_stat:.2f}, p={p_value_t:.4f}'

# Add text to the figure
plt.figtext(0.5, 0.01, f'{anova_text}\n{ttest_text}', ha='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.5))

# Add overall title
plt.suptitle('Relationship Between Social Vulnerability and Age at JIA Onset', fontsize=14)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig('JIA_onset_by_SVI_quartile.png', dpi=300, bbox_inches='tight')
print("Boxplot saved as 'JIA_onset_by_SVI_quartile.png'")

# Create an additional figure showing mean ages with error bars
plt.figure(figsize=(10, 6))

# Calculate mean and standard error for each quartile
stats_df = df.groupby('SVI_quartile')['combined_onset_age'].agg(['mean', 'std', 'count'])
stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
stats_df = stats_df.reindex(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

# Plot mean ages with error bars
plt.errorbar(stats_df.index, stats_df['mean'], yerr=stats_df['se'], fmt='o-', capsize=5, markersize=8)
plt.xlabel('Social Vulnerability Index Quartile', fontsize=12)
plt.ylabel('Mean Age at JIA Onset (years)', fontsize=12)
plt.title('Mean Age at JIA Onset by Social Vulnerability Quartile', fontsize=14)
plt.grid(True, alpha=0.3)

# Add text annotations with mean values
for i, q in enumerate(stats_df.index):
    plt.text(i, stats_df.loc[q, 'mean'] + 0.2, f"{stats_df.loc[q, 'mean']:.2f} yrs", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.text(i, stats_df.loc[q, 'mean'] - 0.6, f"n={stats_df.loc[q, 'count']}", 
             ha='center', va='top', fontsize=9)

# Add statistical test results
plt.figtext(0.5, 0.01, f'{anova_text}\n{ttest_text}', ha='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('JIA_onset_mean_by_SVI_quartile.png', dpi=300, bbox_inches='tight')
print("Mean plot saved as 'JIA_onset_mean_by_SVI_quartile.png'")

# Display basic statistics
print("\nStatistics for Age at JIA Onset (Combined Data) by SVI Quartile:")
stats_table = df.groupby('SVI_quartile')['combined_onset_age'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).round(2)
stats_table = stats_table.reindex(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
print(stats_table)

print("\nStatistical Test Results:")
print(anova_text)
print(ttest_text) 