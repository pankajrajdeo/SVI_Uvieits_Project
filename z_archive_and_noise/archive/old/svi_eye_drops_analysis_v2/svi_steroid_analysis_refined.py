import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os
from datetime import datetime
import re

# Create output directory
output_dir = 'svi_eye_drops_analysis_v2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading dataset...")
df = pd.read_csv('/Users/rajlq7/Desktop/SVI/SVI_filtered_495_patients.csv')
print(f"Total patients: {len(df)}")

# --- SVI Calculation (as before) ---
print("Checking and calculating SVI scores...")
if 'SVI_total' in df.columns and df['SVI_total'].notna().sum() > 0:
    print("Using existing SVI_total column")
else:
    print("Calculating SVI_total from components")
    def calculate_mean_svi(val):
        if pd.isna(val): return np.nan
        try:
            values = [float(x.strip()) for x in str(val).split(';') if x.strip() and x.strip().lower() != 'nan']
            return np.mean(values) if values else np.nan
        except: return np.nan
    
    svi_components = [
        'svi_socioeconomic (list distinct)', 'svi_household_comp (list distinct)',
        'svi_housing_transportation (list distinct)', 'svi_minority (list distinct)'
    ]
    for component in svi_components:
        mean_col = f"{component}_mean"
        if mean_col not in df.columns or df[mean_col].isnull().all():
             df[mean_col] = df[component].apply(calculate_mean_svi)
    
    mean_cols = [f"{component}_mean" for component in svi_components]
    df['SVI_total'] = df[mean_cols].mean(axis=1, skipna=True)
    
    print(f"SVI total calculated/present for {df['SVI_total'].notna().sum()} patients")

# Ensure SVI Quartile exists
if 'SVI_quartile' not in df.columns or df['SVI_quartile'].isnull().all():
     if df['SVI_total'].notna().sum() > 4:
         df['SVI_quartile'] = pd.qcut(df['SVI_total'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
         print(f"SVI quartiles created: {df['SVI_quartile'].value_counts().sort_index().to_dict()}")
     else:
         print("Not enough valid SVI_total scores to create quartiles.")
         df['SVI_quartile'] = 'Unknown'
elif df['SVI_quartile'].isnull().any():
    # Recalculate if some are missing but SVI_total exists
    mask = df['SVI_quartile'].isnull() & df['SVI_total'].notna()
    if mask.sum() > 0 and df['SVI_total'].notna().sum() > 4:
        df.loc[mask, 'SVI_quartile'] = pd.qcut(df.loc[mask, 'SVI_total'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
        print("Filled missing SVI quartiles.")
    df['SVI_quartile'] = df['SVI_quartile'].fillna('Unknown')

# --- Steroid Identification (Keyword Method) ---
print("\nIdentifying steroid eye drops using keyword method...")
steroid_indicators = ['steroi', 'prednisolone', 'difluprednate', 'fluorometholone', 'dexamethasone', 'loteprednol']
medication_col = 'medication name (list distinct)' # Using this, but could switch to simple_generic_name
treatment_col = 'cmeyetrt (list distinct)'

df['has_steroid_drops_kw'] = False

for i, row in df.iterrows():
    has_steroid = False
    # Check dedicated treatment column
    if pd.notna(row[treatment_col]):
        treatments = str(row[treatment_col]).lower()
        if any(steroid in treatments for steroid in steroid_indicators):
            has_steroid = True
            
    # Check general medication column, requiring ophthalmic context
    if not has_steroid and pd.notna(row[medication_col]):
        meds = str(row[medication_col]).lower()
        eye_indicators = ['ophth', 'eye', ' op ', 'ocul', ' oc ', 'opti'] # Keywords suggesting eye use
        if any(steroid in meds for steroid in steroid_indicators) and any(eye in meds for eye in eye_indicators):
             # Check individual meds in the list for steroid + eye context
             for med in meds.split(';'):
                 med_lower = med.strip().lower()
                 if any(s in med_lower for s in steroid_indicators) and any(eye in med_lower for eye in eye_indicators):
                     has_steroid = True
                     break # Found one, no need to check others for this patient
                     
    if has_steroid:
        df.loc[i, 'has_steroid_drops_kw'] = True

steroid_patient_count = df['has_steroid_drops_kw'].sum()
print(f"Identified {steroid_patient_count} patients with potential steroid eye drops (Keyword Method)")

# --- Duration Calculation (Year-Based) ---
def extract_years(date_str):
    if pd.isna(date_str): return []
    try:
        years = [int(y.strip()) for y in str(date_str).split(';') if y.strip().isdigit()]
        return sorted(list(set(years))) # Unique sorted years
    except: return []

def calculate_duration_years(start_dates, end_dates):
    if not start_dates or not end_dates: return np.nan
    earliest_start = min(start_dates)
    latest_end = max(end_dates)
    # Duration is max end - min start. Add 1 if start/end in same year but non-empty?
    # Let's stick to difference for now, acknowledging limitation.
    duration = latest_end - earliest_start 
    return max(0, duration) # Duration cannot be negative

print("\nCalculating treatment duration (based on years)...")
df['eye_drop_start_years'] = df['eye drop start date (list distinct)'].apply(extract_years)
df['eye_drop_end_years'] = df['eye drop end date (list distinct)'].apply(extract_years)

df['duration_years'] = df.apply(
    lambda row: calculate_duration_years(row['eye_drop_start_years'], row['eye_drop_end_years'])
    if row['has_steroid_drops_kw'] else np.nan, 
    axis=1
)

# Filter data for analysis (patients on steroids with valid SVI and duration)
analysis_data = df[df['has_steroid_drops_kw'] & df['duration_years'].notna() & (df['SVI_quartile'] != 'Unknown')].copy()
print(f"Analyzing {len(analysis_data)} patients with steroid drops, duration, and SVI quartile.")

if len(analysis_data) > 0:
    # --- Descriptive Statistics ---
    print("\nDescriptive Statistics for Treatment Duration (Years):")
    duration_stats = analysis_data['duration_years'].describe()
    print(duration_stats)

    # --- Analysis by SVI Quartile ---
    print("\nAnalyzing treatment duration by SVI quartile...")
    svi_duration_stats = analysis_data.groupby('SVI_quartile', observed=True)['duration_years'].agg(['count', 'mean', 'median', 'std']).reset_index()
    print(svi_duration_stats)

    # --- Visualization: Box Plot & Bar Plot ---
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    quartile_order = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    present_quartiles = [q for q in quartile_order if q in analysis_data['SVI_quartile'].unique()]
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='SVI_quartile', y='duration_years', data=analysis_data, order=present_quartiles)
    plt.title('Steroid Treatment Duration by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Duration (Years)', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='SVI_quartile', y='mean', data=svi_duration_stats, order=present_quartiles, palette='viridis', ci=None)
    # Add error bars manually if needed: plt.errorbar(...)
    plt.title('Mean Treatment Duration by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Mean Duration (Years)', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/duration_by_svi_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Statistical Testing: ANOVA --- 
    print("\nPerforming statistical tests (ANOVA)...")
    groups = [analysis_data['duration_years'][analysis_data['SVI_quartile'] == q] for q in present_quartiles]
    groups = [g for g in groups if len(g) > 0] # Remove empty groups

    anova_f_stat, anova_p_value = np.nan, np.nan
    if len(groups) >= 2:
        try:
            anova_f_stat, anova_p_value = stats.f_oneway(*groups)
            print(f"ANOVA for treatment duration across SVI quartiles: F={anova_f_stat:.2f}, p={anova_p_value:.4f}")
        except Exception as e:
            print(f"Could not perform ANOVA: {e}")
    else:
        print("Insufficient groups with data for ANOVA.")

    # --- Statistical Testing: Regression --- 
    print("\nPerforming statistical tests (Linear Regression)...")
    regression_data = analysis_data[['SVI_total', 'duration_years']].dropna()
    
    reg_slope, reg_intercept, reg_r_value, reg_p_value, reg_std_err = np.nan, np.nan, np.nan, np.nan, np.nan
    if len(regression_data) >= 10:
        X = regression_data['SVI_total']
        y = regression_data['duration_years']
        X = sm.add_constant(X) # Add intercept
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
        # Extract key results for report
        reg_slope = model.params['SVI_total']
        reg_intercept = model.params['const']
        reg_r_value = np.sqrt(model.rsquared)
        reg_p_value = model.pvalues['SVI_total']
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.regplot(x='SVI_total', y='duration_years', data=regression_data, scatter_kws={'alpha':0.5})
        plt.title(f'SVI Total vs Treatment Duration (R² = {model.rsquared:.2f}, p = {reg_p_value:.4f})', fontsize=14)
        plt.xlabel('SVI Total Score', fontsize=12)
        plt.ylabel('Treatment Duration (Years)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/svi_duration_regression_v2.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Insufficient data for regression analysis (need >= 10 points).")

    # --- Prevalence Analysis ---
    print("\nPrevalence of steroid eye drop use by SVI quartile:")
    # Use original df for prevalence calculation, group by SVI_quartile including 'Unknown'
    prevalence_table = df.groupby('SVI_quartile', observed=True)['has_steroid_drops_kw'].agg(['count', 'sum'])
    prevalence_table['percentage'] = (prevalence_table['sum'] / prevalence_table['count'] * 100).round(1)
    prevalence_table.columns = ['Total Patients', 'Patients on Steroid Drops', 'Percentage (%)']
    print(prevalence_table)

    # Visualization of prevalence
    plt.figure(figsize=(10, 6))
    prevalence_data_for_plot = prevalence_table.reset_index()
    # Exclude 'Unknown' SVI for plot clarity if desired
    # prevalence_data_for_plot = prevalence_data_for_plot[prevalence_data_for_plot['SVI_quartile'] != 'Unknown']
    sns.barplot(x='SVI_quartile', y='Percentage (%)', data=prevalence_data_for_plot, palette='viridis', ci=None, order=quartile_order + ['Unknown'])
    plt.title('Percentage of Patients on Steroid Eye Drops by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, max(prevalence_data_for_plot['Percentage (%)'].fillna(0)) * 1.2 + 5) # Add padding
    plt.xticks(rotation=45)
    for i, row in prevalence_data_for_plot.iterrows():
        plt.text(i, row['Percentage (%)'] + 1, f"{row['Percentage (%)']}%", ha='center', fontsize=9)
    plt.savefig(f'{output_dir}/steroid_prevalence_by_svi_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save Results ---
    print("\nSaving results...")
    svi_duration_stats.to_csv(f'{output_dir}/svi_duration_stats_v2.csv', index=False)
    prevalence_table.reset_index().to_csv(f'{output_dir}/steroid_prevalence_by_svi_v2.csv', index=False)

    # --- Generate Summary Report ---
    print("Generating summary report...")
    with open(f'{output_dir}/summary_report_v2.md', 'w') as f:
        f.write("# SVI and Steroid Eye Drop Treatment Analysis (Refined Methodology)\n\n")
        f.write("## Overview\n")
        f.write("This report examines the relationship between Social Vulnerability Index (SVI) and the duration of *presumed* topical steroid eye drop treatment. Steroid use was identified using keywords.")
        f.write(" Treatment duration was calculated based on the earliest start year and latest end year recorded.\n\n")
        f.write("**Limitations Acknowledged:**\n")
        f.write("- Steroid identification relies on keyword matching and may not be perfectly accurate.\n")
        f.write("- Duration calculation uses year-level data and assumes continuous treatment, potentially over/underestimating true duration.\n")
        f.write("- Analysis does not control for disease severity due to data limitations.\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"- **Total patients**: {len(df)}\n")
        f.write(f"- **Patients presumed on steroid eye drops (Keyword Method)**: {steroid_patient_count} ({steroid_patient_count/len(df)*100:.1f}%)\n")
        f.write(f"- **Patients included in duration analysis**: {len(analysis_data)}\n\n")
        
        f.write("### Treatment Duration Statistics (Years)\n")
        f.write(duration_stats.to_markdown())
        f.write("\n\n")
        
        f.write("### Treatment Duration by SVI Quartile\n")
        f.write(svi_duration_stats.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### Statistical Analysis\n")
        if not np.isnan(anova_p_value):
            f.write(f"- **ANOVA (Duration vs SVI Quartile)**: F={anova_f_stat:.2f}, p={anova_p_value:.4f}")
            f.write(" (Significant)" if anova_p_value < 0.05 else " (Not Significant)")
            f.write("\n")
        else:
            f.write("- ANOVA could not be performed.\n")
            
        if not np.isnan(reg_p_value):
            f.write(f"- **Linear Regression (Duration vs SVI Total)**:\n")
            f.write(f"  - Slope: {reg_slope:.4f}\n")
            f.write(f"  - R-squared: {reg_r_value**2:.4f}\n")
            f.write(f"  - P-value: {reg_p_value:.4f}")
            f.write(" (Significant)" if reg_p_value < 0.05 else " (Not Significant)")
            f.write("\n")
        else:
            f.write("- Linear Regression could not be performed.\n")
        f.write("\n")
        
        f.write("### Prevalence of Steroid Eye Drop Use by SVI Quartile\n")
        f.write(prevalence_table.reset_index().to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Interpretation\n")
        f.write("Acknowledging the methodological limitations, the analysis suggests: \n")
        if not np.isnan(reg_p_value) and reg_p_value < 0.05:
             f.write("- A statistically significant relationship between SVI and treatment duration. ")
             if reg_slope < 0:
                 f.write("Higher SVI is associated with **shorter** treatment durations. This *could* indicate issues with access, adherence, or follow-up, but might also reflect appropriate clinical adaptation or unmeasured confounding (like disease severity).\n")
             else:
                 f.write("Higher SVI is associated with **longer** treatment durations. This *could* indicate delayed inflammation control or potentially more severe disease requiring longer therapy, though severity could not be directly assessed.\n")
        elif not np.isnan(anova_p_value) and anova_p_value < 0.05:
            f.write("- Significant differences in treatment duration exist between SVI quartiles, but the relationship is not strictly linear with the continuous SVI score. Further investigation into quartile-specific factors is needed.\n")
        else:
            f.write("- No statistically significant relationship was detected between SVI (either quartile or total score) and the estimated treatment duration based on this data and methodology.\n")
        f.write("\nThese findings should be interpreted cautiously due to data constraints.")

else:
    print("Insufficient data after filtering for steroid use, duration, and SVI. Analysis cannot proceed.")

print(f"\nAnalysis complete. Results saved to '{output_dir}' directory.") 
