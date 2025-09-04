import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Load data from CSVs
onset_csv = "svi_onset_analysis/Age_Onset_by_SVI_Quartile_Summary.csv"
treat_csv = "svi_treatment_delay_analysis/treatment_delay_summary_stats_all.csv"

# Panel A - Mean Age at Onset
onset_df = pd.read_csv(onset_csv)
svi_quartiles = ['Q1', 'Q2', 'Q3', 'Q4']

# JIA
jia_onset = onset_df[(onset_df['Onset Type'] == 'JIA') & (onset_df['SVI_quartile'].isin(svi_quartiles))]
jia_mean_age = [jia_onset[jia_onset['SVI_quartile'] == q]['mean'].values[0] if q in jia_onset['SVI_quartile'].values else np.nan for q in svi_quartiles]
jia_age_sd = [jia_onset[jia_onset['SVI_quartile'] == q]['std'].values[0] if q in jia_onset['SVI_quartile'].values else np.nan for q in svi_quartiles]

# Uveitis
uv_onset = onset_df[(onset_df['Onset Type'] == 'Uveitis') & (onset_df['SVI_quartile'].isin(svi_quartiles))]
uveitis_mean_age = [uv_onset[uv_onset['SVI_quartile'] == q]['mean'].values[0] if q in uv_onset['SVI_quartile'].values else np.nan for q in svi_quartiles]
uveitis_age_sd = [uv_onset[uv_onset['SVI_quartile'] == q]['std'].values[0] if q in uv_onset['SVI_quartile'].values else np.nan for q in svi_quartiles]

# Remove negative values (set to np.nan)
def remove_neg(arr):
    return [np.nan if (pd.isna(x) or x < 0) else x for x in arr]

# Sanitize all data arrays right before plotting
jia_mean_age = remove_neg(jia_mean_age)
jia_age_sd = remove_neg(jia_age_sd)
uveitis_mean_age = remove_neg(uveitis_mean_age)
uveitis_age_sd = remove_neg(uveitis_age_sd)

# Panel B - Mean Time to Medication
treat_df = pd.read_csv(treat_csv)

# JIA-Only
jia_treat = treat_df[(treat_df['diagnosis_group'] == 'JIA-Only') & (treat_df['svi_quartile'].isin(svi_quartiles))]
jia_mean_time = [jia_treat[jia_treat['svi_quartile'] == q]['mean'].values[0] if q in jia_treat['svi_quartile'].values else np.nan for q in svi_quartiles]
jia_time_sd = [jia_treat[jia_treat['svi_quartile'] == q]['std'].values[0] if q in jia_treat['svi_quartile'].values else np.nan for q in svi_quartiles]

# Uveitis
uv_treat = treat_df[(treat_df['diagnosis_group'] == 'Uveitis') & (treat_df['svi_quartile'].isin(svi_quartiles))]
uveitis_mean_time = [uv_treat[uv_treat['svi_quartile'] == q]['mean'].values[0] if q in uv_treat['svi_quartile'].values else np.nan for q in svi_quartiles]
uveitis_time_sd = [uv_treat[uv_treat['svi_quartile'] == q]['std'].values[0] if q in uv_treat['svi_quartile'].values else np.nan for q in svi_quartiles]

# Sanitize all data arrays right before plotting
jia_mean_time = remove_neg(jia_mean_time)
jia_time_sd = remove_neg(jia_time_sd)
uveitis_mean_time = remove_neg(uveitis_mean_time)
uveitis_time_sd = remove_neg(uveitis_time_sd)

x = np.arange(len(svi_quartiles))
width = 0.35

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, gridspec_kw={'wspace': 0.25})

# Panel A: error bars clipped at zero
for idx, (mean, sd, color, label) in enumerate([
    (jia_mean_age, jia_age_sd, '#1f77b4', 'JIA-Only'),
    (uveitis_mean_age, uveitis_age_sd, '#d62728', 'Uveitis')
]):
    means = np.array(mean)
    sds = np.array(sd)
    # Calculate lower and upper error bars
    lower = np.where(means - sds < 0, means, sds)
    lower = np.where(means - lower < 0, means, lower)  # Ensure lower error bar doesn't go below zero
    lower = np.where(np.isnan(means) | np.isnan(sds), 0, lower)
    upper = np.where(np.isnan(means) | np.isnan(sds), 0, sds)
    axs[0].errorbar(x, means, yerr=[lower, upper], fmt='o-', capsize=5, label=label, color=color)

axs[0].set_title('[A] Mean Age at Onset by SVI Quartile')
axs[0].set_ylabel('Mean Age at Onset (Years)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(svi_quartiles)
axs[0].grid(True, linestyle='--', alpha=0.5)

# Panel B: error bars clipped at zero
bar_means = [jia_mean_time, uveitis_mean_time]
bar_sds = [jia_time_sd, uveitis_time_sd]
bar_colors = ['#1f77b4', '#d62728']
bar_labels = ['JIA-Only', 'Uveitis']

for i, (means, sds, color, label) in enumerate(zip(bar_means, bar_sds, bar_colors, bar_labels)):
    means = np.array(means)
    sds = np.array(sds)
    lower = np.where(means - sds < 0, means, sds)
    lower = np.where(means - lower < 0, means, lower)
    lower = np.where(np.isnan(means) | np.isnan(sds), 0, lower)
    upper = np.where(np.isnan(means) | np.isnan(sds), 0, sds)
    yerr = np.vstack([lower, upper])
    axs[1].bar(x + (i - 0.5) * width, means, width, yerr=yerr, capsize=5, label=label, color=color)

axs[1].set_title('[B] Mean Time to First Systemic Medication by SVI Quartile')
axs[1].set_ylabel('Mean Time to Systemic Immunosuppression (Weeks)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(svi_quartiles)
axs[1].grid(True, linestyle='--', alpha=0.5)

# Custom Legend (top center, colored boxes)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor='#1f77b4', edgecolor='none', label='JIA-Only'),
    Patch(facecolor='#d62728', edgecolor='none', label='Uveitis')
]
# Move legend to bottom with more white space
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.15))

fig.tight_layout(rect=[0, 0.1, 1, 0.95])  # leave more space at bottom
fig.savefig("SVI_Onset_Treatment_Figure.png", dpi=400, bbox_inches='tight')
plt.show()
