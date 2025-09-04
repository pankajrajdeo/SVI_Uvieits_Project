import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

# Load the data
print("Loading the dataset...")
df = pd.read_csv('SVI_filtered_495_patients.csv')

# Create JIA status using all available data
# First, identify arthritis diagnosis
print("Creating combined JIA status variable...")
df['has_jia'] = np.where(df['diagnosis of arthritis'] == True, 'JIA', 'No JIA')

# Check for additional JIA data
if 'jia subtype ' in df.columns and df['jia subtype '].notna().any():
    print("Including JIA subtype information...")
    df.loc[df['jia subtype '].notna(), 'has_jia'] = 'JIA'

# Check for JIA symptom onset data
if 'date of JIA symptom onset' in df.columns and df['date of JIA symptom onset'].notna().any():
    print("Including JIA symptom onset information...")
    df.loc[df['date of JIA symptom onset'].notna(), 'has_jia'] = 'JIA'

# Count patients
jia_pts = df[df['has_jia'] == 'JIA']
non_jia_pts = df[df['has_jia'] == 'No JIA']
print(f"Total patients: {len(df)}")
print(f"JIA patients (combined definition): {len(jia_pts)}")
print(f"Non-JIA patients: {len(non_jia_pts)}")

# Overall SVI statistics by JIA diagnosis
print("\n--- SVI scores by JIA diagnosis (Combined Definition) ---")
print(f"JIA patients (n={len(jia_pts)}):")
print(f"  Mean SVI = {jia_pts['SVI_total'].mean():.3f}")
print(f"  Median SVI = {jia_pts['SVI_total'].median():.3f}")
print(f"  SD = {jia_pts['SVI_total'].std():.3f}")

print(f"\nNon-JIA patients (n={len(non_jia_pts)}):")
print(f"  Mean SVI = {non_jia_pts['SVI_total'].mean():.3f}")
print(f"  Median SVI = {non_jia_pts['SVI_total'].median():.3f}")
print(f"  SD = {non_jia_pts['SVI_total'].std():.3f}")

# T-test
t_stat, p_value = stats.ttest_ind(jia_pts['SVI_total'], non_jia_pts['SVI_total'], equal_var=False)
print("\n--- Statistical Test ---")
print(f"T-test: t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("The difference in SVI scores is statistically significant.")
else:
    print("The difference in SVI scores is not statistically significant.")

# Quartile analysis
print("\n--- SVI quartile distribution ---")
# Initialize counts
counts = {
    'Q1 (Low)': {'JIA': 0, 'Non-JIA': 0, 'Total': 0},
    'Q2': {'JIA': 0, 'Non-JIA': 0, 'Total': 0},
    'Q3': {'JIA': 0, 'Non-JIA': 0, 'Total': 0},
    'Q4 (High)': {'JIA': 0, 'Non-JIA': 0, 'Total': 0},
    'Total': {'JIA': 0, 'Non-JIA': 0, 'Total': 0}
}

# Count patients in each category
for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
    counts[quartile]['JIA'] = sum((df['SVI_quartile'] == quartile) & (df['has_jia'] == 'JIA'))
    counts[quartile]['Non-JIA'] = sum((df['SVI_quartile'] == quartile) & (df['has_jia'] == 'No JIA'))
    counts[quartile]['Total'] = counts[quartile]['JIA'] + counts[quartile]['Non-JIA']
    counts['Total']['JIA'] += counts[quartile]['JIA']
    counts['Total']['Non-JIA'] += counts[quartile]['Non-JIA']

counts['Total']['Total'] = counts['Total']['JIA'] + counts['Total']['Non-JIA']

# Print counts
print("Absolute counts:")
print(f"{'Quartile':<12} {'JIA':<10} {'Non-JIA':<10} {'Total':<10}")
print("-" * 42)
for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total']:
    print(f"{quartile:<12} {counts[quartile]['JIA']:<10} {counts[quartile]['Non-JIA']:<10} {counts[quartile]['Total']:<10}")

# Print percentages (column-wise)
print("\nPercentages (by column):")
print(f"{'Quartile':<12} {'JIA %':<10} {'Non-JIA %':<10} {'Total %':<10}")
print("-" * 42)
for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
    jia_pct = counts[quartile]['JIA'] / counts['Total']['JIA'] * 100
    non_jia_pct = counts[quartile]['Non-JIA'] / counts['Total']['Non-JIA'] * 100
    total_pct = counts[quartile]['Total'] / counts['Total']['Total'] * 100
    print(f"{quartile:<12} {jia_pct:.1f}%{'':<5} {non_jia_pct:.1f}%{'':<5} {total_pct:.1f}%{'':<5}")

# Chi-square test
contingency_table = np.array([
    [counts['Q1 (Low)']['JIA'], counts['Q1 (Low)']['Non-JIA']],
    [counts['Q2']['JIA'], counts['Q2']['Non-JIA']],
    [counts['Q3']['JIA'], counts['Q3']['Non-JIA']],
    [counts['Q4 (High)']['JIA'], counts['Q4 (High)']['Non-JIA']]
])

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\n--- Chi-square test for SVI quartile distribution ---")
print(f"Chi-square value: {chi2:.3f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of freedom: {dof}")
if p < 0.05:
    print("The distribution of JIA across SVI quartiles is statistically significant.")
else:
    print("The distribution of JIA across SVI quartiles is not statistically significant.")

# Save results to CSV
results_df = pd.DataFrame({
    'Quartile': ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total'],
    'JIA_Count': [counts[q]['JIA'] for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total']],
    'NonJIA_Count': [counts[q]['Non-JIA'] for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total']],
    'Total_Count': [counts[q]['Total'] for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total']],
    'JIA_Percent': [counts[q]['JIA'] / counts['Total']['JIA'] * 100 if q != 'Total' else 100 for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total']],
    'NonJIA_Percent': [counts[q]['Non-JIA'] / counts['Total']['Non-JIA'] * 100 if q != 'Total' else 100 for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)', 'Total']]
})

results_df.to_csv('svi_jia_combined_results.csv', index=False)
print("\nAnalysis complete. Results saved to svi_jia_combined_results.csv")

# Create enhanced visualizations
plt.figure(figsize=(15, 10))

# 1. Bar chart for distribution percentages
plt.subplot(2, 2, 1)
quartiles = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
jia_pcts = [counts[q]['JIA'] / counts['Total']['JIA'] * 100 for q in quartiles]
non_jia_pcts = [counts[q]['Non-JIA'] / counts['Total']['Non-JIA'] * 100 for q in quartiles]

x = np.arange(len(quartiles))
width = 0.35

bars1 = plt.bar(x - width/2, jia_pcts, width, label='JIA', color='#3182bd')
bars2 = plt.bar(x + width/2, non_jia_pcts, width, label='Non-JIA', color='#9ecae1')
plt.xlabel('SVI Quartile', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Distribution by SVI Quartile', fontsize=14)
plt.xticks(x, quartiles, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)

# Add data labels on bars
for i, bars in enumerate([bars1, bars2]):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 2. Box plot for SVI scores
plt.subplot(2, 2, 2)
sns.boxplot(x='has_jia', y='SVI_total', data=df)
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('SVI Score', fontsize=12)
plt.title('SVI Score Distribution by JIA Status', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add p-value annotation
plt.annotate(f'T-test p-value: {p_value:.4f}', xy=(0.5, 0.05), xycoords='axes fraction', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

# 3. Stacked bar chart showing JIA proportion within each quartile
plt.subplot(2, 2, 3)
quartile_totals = [counts[q]['Total'] for q in quartiles]
jia_counts = [counts[q]['JIA'] for q in quartiles]
non_jia_counts = [counts[q]['Non-JIA'] for q in quartiles]

# Calculate percentages within each quartile
jia_within_q = [counts[q]['JIA'] / counts[q]['Total'] * 100 for q in quartiles]
non_jia_within_q = [counts[q]['Non-JIA'] / counts[q]['Total'] * 100 for q in quartiles]

plt.bar(quartiles, non_jia_within_q, color='#9ecae1', label='Non-JIA')
plt.bar(quartiles, jia_within_q, bottom=non_jia_within_q, color='#3182bd', label='JIA')
plt.xlabel('SVI Quartile', fontsize=12)
plt.ylabel('Percentage within Quartile (%)', fontsize=12)
plt.title('JIA Proportion within each SVI Quartile', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)

# Add percentage labels
for i, q in enumerate(quartiles):
    # JIA percentage label
    plt.text(i, non_jia_within_q[i] + jia_within_q[i]/2, 
             f'{jia_within_q[i]:.1f}%', ha='center', va='center', fontsize=9, color='white')
    # Non-JIA percentage label  
    plt.text(i, non_jia_within_q[i]/2, 
             f'{non_jia_within_q[i]:.1f}%', ha='center', va='center', fontsize=9)
    # Total count label
    plt.text(i, 101, f'n={counts[q]["Total"]}', ha='center', va='bottom', fontsize=9)

# 4. Line plot showing distribution pattern
plt.subplot(2, 2, 4)
plt.plot(quartiles, jia_pcts, 'o-', linewidth=2, markersize=8, label='JIA', color='#3182bd')
plt.plot(quartiles, non_jia_pcts, 'o-', linewidth=2, markersize=8, label='Non-JIA', color='#9ecae1')
plt.xlabel('SVI Quartile', fontsize=12)
plt.ylabel('Percentage of Group (%)', fontsize=12)
plt.title('Distribution Pattern Across SVI Quartiles', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)

# Add chi-square result annotation
plt.annotate(f'Chi-square p-value: {p:.4f}', xy=(0.5, 0.05), xycoords='axes fraction', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

# Add data labels on lines
for i, quartile in enumerate(quartiles):
    plt.annotate(f'{jia_pcts[i]:.1f}%', 
                xy=(quartile, jia_pcts[i]), 
                xytext=(0, 7),
                textcoords='offset points',
                ha='center', 
                fontsize=9)
    plt.annotate(f'{non_jia_pcts[i]:.1f}%', 
                xy=(quartile, non_jia_pcts[i]), 
                xytext=(0, -14),
                textcoords='offset points',
                ha='center', 
                fontsize=9)

# Add overall title and adjust layout
plt.suptitle('Relationship Between Social Vulnerability Index and JIA Diagnosis\n(Combined Data Approach)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save high-quality figure
plt.savefig('svi_jia_combined_analysis.png', dpi=300, bbox_inches='tight')
print("Enhanced visualization saved as svi_jia_combined_analysis.png") 