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
output_dir = 'uveitis_steroid_analysis_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading dataset...")
df = pd.read_csv('SVI_filtered_495_patients.csv')
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
            # Remove negative values
            values = [v for v in values if v >= 0]
            return np.mean(values) if values else np.nan
        except: return np.nan
    
    svi_components = [
        'svi_socioeconomic (list distinct)', 'svi_household_comp (list distinct)',
        'svi_housing_transportation (list distinct)', 'svi_minority (list distinct)'
    ]
    for component in svi_components:
        mean_col = f"{component.split(' ')[0]}_mean"
        if mean_col not in df.columns or df[mean_col].isnull().all():
             df[mean_col] = df[component].apply(calculate_mean_svi)
    
    mean_cols = [f"{component.split(' ')[0]}_mean" for component in svi_components]
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

# --- Identify Uveitis Patients ---
print("\nIdentifying uveitis patients...")
df['has_uveitis'] = 0

# Check uveitis-related columns for direct diagnosis
uveitis_cols = ['diagnosis of uveitis', 'uveitis curr', 'uveitis curr fup']
for col in uveitis_cols:
    if col in df.columns:
        print(f"Checking column: {col}")
        uveitis_mask = df[col].notna()
        print(f"Found {uveitis_mask.sum()} patients with uveitis in {col}")
        df.loc[uveitis_mask, 'has_uveitis'] = 1

# Check 'uveitis location' column to confirm diagnosis
if 'uveitis location ' in df.columns:
    print(f"Checking column: uveitis location")
    has_location_mask = df['uveitis location '].notna()
    print(f"Found {has_location_mask.sum()} patients with uveitis location specified")
    df.loc[has_location_mask, 'has_uveitis'] = 1

# Filter to only uveitis patients
uveitis_patients = df[df['has_uveitis'] == 1].copy()
print(f"\nTotal uveitis patients identified: {len(uveitis_patients)}")

# --- Steroid Identification (Keyword Method) ---
print("\nIdentifying steroid eye drops in uveitis patients using keyword method...")
steroid_indicators = ['steroi', 'prednisolone', 'difluprednate', 'fluorometholone', 'dexamethasone', 'loteprednol']
medication_col = 'medication name (list distinct)' # Using this, but could switch to simple_generic_name
treatment_col = 'cmeyetrt (list distinct)'

uveitis_patients['has_steroid_drops_kw'] = False

for i, row in uveitis_patients.iterrows():
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
        uveitis_patients.loc[i, 'has_steroid_drops_kw'] = True

steroid_patient_count = uveitis_patients['has_steroid_drops_kw'].sum()
print(f"Identified {steroid_patient_count} patients with uveitis and steroid eye drops (Keyword Method)")

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

print("\nCalculating treatment duration for uveitis patients (based on years)...")
uveitis_patients['eye_drop_start_years'] = uveitis_patients['eye drop start date (list distinct)'].apply(extract_years)
uveitis_patients['eye_drop_end_years'] = uveitis_patients['eye drop end date (list distinct)'].apply(extract_years)

uveitis_patients['duration_years'] = uveitis_patients.apply(
    lambda row: calculate_duration_years(row['eye_drop_start_years'], row['eye_drop_end_years'])
    if row['has_steroid_drops_kw'] else np.nan, 
    axis=1
)

# --- Analyze Uveitis Activity and Steroid Use ---
print("\nAnalyzing uveitis activity status and steroid use...")
if 'uveitis curr' in uveitis_patients.columns and 'uveitis curr fup' in uveitis_patients.columns:
    uveitis_patients['active_current'] = uveitis_patients['uveitis curr'].notna().astype(int)
    uveitis_patients['active_followup'] = uveitis_patients['uveitis curr fup'].notna().astype(int)
    uveitis_patients['active_any'] = ((uveitis_patients['active_current'] == 1) | 
                                     (uveitis_patients['active_followup'] == 1)).astype(int)
    
    print(f"Patients with currently active uveitis: {uveitis_patients['active_current'].sum()}")
    print(f"Patients with active uveitis at follow-up: {uveitis_patients['active_followup'].sum()}")
    print(f"Patients with active uveitis at any point: {uveitis_patients['active_any'].sum()}")
    
    # Cross-tabulation of activity and steroid use
    activity_steroid_crosstab = pd.crosstab(
        uveitis_patients['active_any'],
        uveitis_patients['has_steroid_drops_kw'],
        rownames=['Active Uveitis'],
        colnames=['Steroid Drops'],
        margins=True
    )
    print("\nCross-tabulation of uveitis activity and steroid use:")
    print(activity_steroid_crosstab)
    
    # Chi-square test
    table = activity_steroid_crosstab.iloc[:-1, :-1]
    if table.shape == (2, 2) and table.values.min() >= 5:
        chi2, p, dof, expected = stats.chi2_contingency(table)
        print(f"Chi-square test: chi2={chi2:.2f}, p={p:.4f}")
        
# Filter data for analysis (patients on steroids with valid SVI and duration)
analysis_data = uveitis_patients[uveitis_patients['has_steroid_drops_kw'] & 
                              uveitis_patients['duration_years'].notna() & 
                              (uveitis_patients['SVI_quartile'] != 'Unknown')].copy()
print(f"Analyzing {len(analysis_data)} uveitis patients with steroid drops, duration, and SVI quartile.")

if len(analysis_data) > 0:
    # --- Descriptive Statistics ---
    print("\nDescriptive Statistics for Steroid Treatment Duration in Uveitis Patients (Years):")
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
    
    # Use matplotlib boxplot with strict control over negative values
    plt.subplot(1, 2, 1)
    # Extract data by quartile and ensure no negative values
    duration_by_quartile = [analysis_data[analysis_data['SVI_quartile'] == q]['duration_years'].clip(lower=0).values 
                           for q in present_quartiles]
    
    # Custom boxplot with strict minimum at 0
    bp1 = plt.boxplot(duration_by_quartile, positions=range(len(present_quartiles)), patch_artist=True)
    plt.title('Steroid Treatment Duration for Uveitis by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Duration (Years)', fontsize=12)
    plt.xticks(range(len(present_quartiles)), present_quartiles, rotation=45)
    plt.ylim(0, None)  # Set explicit minimum of 0
    
    # Color the boxes
    for box in bp1['boxes']:
        box.set(facecolor='#1f77b4')  # Match seaborn blue color
    
    # Ensure no lines go below 0
    for key in ['whiskers', 'caps', 'fliers']:
        for line in bp1[key]:
            ydata = line.get_ydata()
            line.set_ydata(np.clip(ydata, 0, None))
    
    plt.subplot(1, 2, 2)
    # For bar plot we don't need the same treatment as it's just means
    mean_data = svi_duration_stats.copy()
    mean_data['mean'] = mean_data['mean'].clip(lower=0)
    sns.barplot(x='SVI_quartile', y='mean', data=mean_data, order=present_quartiles, palette='viridis', errorbar=None)
    plt.title('Mean Treatment Duration for Uveitis by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Mean Duration (Years)', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uveitis_steroid_duration_by_svi.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Statistical Testing: ANOVA --- 
    print("\nPerforming statistical tests (ANOVA)...")
    groups = [analysis_data['duration_years'][analysis_data['SVI_quartile'] == q].clip(lower=0) 
              for q in present_quartiles]
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
    # Ensure no negative durations
    regression_data['duration_years'] = regression_data['duration_years'].clip(lower=0)
    
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
        plt.title(f'SVI Total vs Uveitis Steroid Treatment Duration (R² = {model.rsquared:.2f}, p = {reg_p_value:.4f})', fontsize=14)
        plt.xlabel('SVI Total Score', fontsize=12)
        plt.ylabel('Treatment Duration (Years)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        plt.savefig(f'{output_dir}/uveitis_svi_duration_regression.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Insufficient data for regression analysis (need >= 10 points).")

    # --- Prevalence Analysis ---
    print("\nPrevalence of steroid eye drop use in uveitis patients by SVI quartile:")
    # Use uveitis_patients for prevalence calculation, group by SVI_quartile including 'Unknown'
    prevalence_table = uveitis_patients.groupby('SVI_quartile', observed=True)['has_steroid_drops_kw'].agg(['count', 'sum'])
    prevalence_table['percentage'] = (prevalence_table['sum'] / prevalence_table['count'] * 100).round(1)
    prevalence_table.columns = ['Total Uveitis Patients', 'Patients on Steroid Drops', 'Percentage (%)']
    print(prevalence_table)

    # Visualization of prevalence
    plt.figure(figsize=(10, 6))
    prevalence_data_for_plot = prevalence_table.reset_index()
    sns.barplot(x='SVI_quartile', y='Percentage (%)', data=prevalence_data_for_plot, palette='viridis', errorbar=None, order=quartile_order + ['Unknown'] if 'Unknown' in prevalence_data_for_plot['SVI_quartile'].values else quartile_order)
    plt.title('Percentage of Uveitis Patients on Steroid Eye Drops by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, max(prevalence_data_for_plot['Percentage (%)'].fillna(0)) * 1.2 + 5) # Add padding
    plt.xticks(rotation=45)
    for i, row in prevalence_data_for_plot.iterrows():
        plt.text(i, row['Percentage (%)'] + 1, f"{row['Percentage (%)']}%", ha='center', fontsize=9)
    plt.savefig(f'{output_dir}/uveitis_steroid_prevalence_by_svi.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save Results ---
    print("\nSaving results...")
    svi_duration_stats.to_csv(f'{output_dir}/uveitis_steroid_duration_stats.csv', index=False)
    prevalence_table.reset_index().to_csv(f'{output_dir}/uveitis_steroid_prevalence_by_svi.csv', index=False)

    # --- Activity and Steroid Use Analysis ---
    if 'active_any' in uveitis_patients.columns:
        # Analysis of steroid treatment duration by uveitis activity
        activity_duration = analysis_data.groupby('active_any', observed=True)['duration_years'].agg(['count', 'mean', 'median', 'std']).reset_index()
        activity_duration['active_any'] = activity_duration['active_any'].map({0: 'Inactive', 1: 'Active'})
        activity_duration.to_csv(f'{output_dir}/uveitis_activity_duration_stats.csv', index=False)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        activity_data = activity_duration.copy()
        activity_data['mean'] = activity_data['mean'].clip(lower=0)
        sns.barplot(x='active_any', y='mean', data=activity_data, palette='viridis', errorbar=None)
        plt.title('Mean Steroid Treatment Duration by Uveitis Activity Status', fontsize=14)
        plt.xlabel('Uveitis Activity', fontsize=12)
        plt.ylabel('Mean Duration (Years)', fontsize=12)
        plt.ylim(0, None)
        
        # Add value labels
        for i, row in activity_data.iterrows():
            plt.text(i, row['mean'] + 0.1, f"{row['mean']:.2f}", ha='center', fontsize=9)
            
        plt.savefig(f'{output_dir}/uveitis_activity_duration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # T-test if enough data
        active_duration = analysis_data[analysis_data['active_any'] == 1]['duration_years'].clip(lower=0)
        inactive_duration = analysis_data[analysis_data['active_any'] == 0]['duration_years'].clip(lower=0)
        
        if len(active_duration) >= 5 and len(inactive_duration) >= 5:
            t_stat, t_p_value = stats.ttest_ind(active_duration, inactive_duration, equal_var=False)
            print(f"T-test for treatment duration by activity status: t={t_stat:.2f}, p={t_p_value:.4f}")
        
    # --- Generate Summary Report ---
    print("Generating summary report...")
    with open(f'{output_dir}/uveitis_steroid_analysis_report.md', 'w') as f:
        f.write("# Uveitis and Steroid Eye Drop Treatment Analysis\n\n")
        f.write("## Overview\n")
        f.write("This report examines the relationship between Social Vulnerability Index (SVI) and the duration of topical steroid eye drop treatment specifically in patients with uveitis.")
        f.write(" Treatment duration was calculated based on the earliest start year and latest end year recorded.\n\n")
        f.write("**Limitations Acknowledged:**\n")
        f.write("- Steroid identification relies on keyword matching and may not be perfectly accurate.\n")
        f.write("- Duration calculation uses year-level data and assumes continuous treatment, potentially over/underestimating true duration.\n")
        f.write("- Analysis does not fully control for disease severity due to data limitations.\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"- **Total patients in dataset**: {len(df)}\n")
        f.write(f"- **Patients with uveitis**: {len(uveitis_patients)} ({len(uveitis_patients)/len(df)*100:.1f}%)\n")
        f.write(f"- **Uveitis patients on steroid eye drops**: {steroid_patient_count} ({steroid_patient_count/len(uveitis_patients)*100:.1f}%)\n")
        f.write(f"- **Patients included in duration analysis**: {len(analysis_data)}\n\n")
        
        # Add activity stats if available
        if 'active_any' in uveitis_patients.columns:
            active_count = uveitis_patients['active_any'].sum()
            f.write(f"- **Uveitis patients with active inflammation**: {active_count} ({active_count/len(uveitis_patients)*100:.1f}%)\n")
        
        f.write("\n### Treatment Duration Statistics for Uveitis Patients (Years)\n")
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
        
        f.write("### Prevalence of Steroid Eye Drop Use in Uveitis Patients by SVI Quartile\n")
        f.write(prevalence_table.reset_index().to_markdown(index=False))
        f.write("\n\n")
        
        # Add activity analysis if available
        if 'active_any' in uveitis_patients.columns and 'activity_duration' in locals():
            f.write("### Steroid Treatment Duration by Uveitis Activity Status\n")
            f.write(activity_duration.to_markdown(index=False))
            f.write("\n\n")
            
            if 't_p_value' in locals():
                f.write(f"- **T-test (Active vs Inactive Uveitis)**: t={t_stat:.2f}, p={t_p_value:.4f}")
                f.write(" (Significant)" if t_p_value < 0.05 else " (Not Significant)")
                f.write("\n\n")
        
        f.write("## Interpretation\n")
        f.write("Acknowledging the methodological limitations, the analysis of steroid treatment in uveitis patients suggests: \n")
        if not np.isnan(reg_p_value) and reg_p_value < 0.05:
             f.write("- A statistically significant relationship between SVI and treatment duration in uveitis patients. ")
             if reg_slope < 0:
                 f.write("Higher SVI is associated with **shorter** steroid treatment durations. This may indicate issues with access, adherence, or follow-up for uveitis patients from more vulnerable areas, but could also reflect other clinical factors.\n")
             else:
                 f.write("Higher SVI is associated with **longer** steroid treatment durations. This could suggest that patients from more vulnerable areas may have more persistent inflammation requiring extended therapy, or there may be barriers to transitioning to steroid-sparing immunosuppressive therapy.\n")
        elif not np.isnan(anova_p_value) and anova_p_value < 0.05:
            f.write("- Significant differences in steroid treatment duration exist between SVI quartiles among uveitis patients, but the relationship is not strictly linear with the continuous SVI score. This may reflect complex social determinants affecting treatment patterns.\n")
        else:
            f.write("- No statistically significant relationship was detected between SVI (either quartile or total score) and the estimated steroid treatment duration for uveitis patients based on this data and methodology.\n")
        
        if 'active_any' in uveitis_patients.columns and 't_p_value' in locals():
            f.write("\n- ")
            if t_p_value < 0.05:
                f.write("There is a significant difference in steroid treatment duration between patients with active versus inactive uveitis. ")
                if active_duration.mean() > inactive_duration.mean():
                    f.write("Patients with active uveitis have longer treatment durations, which may reflect appropriate clinical care for persistent inflammation.")
                else:
                    f.write("Patients with inactive uveitis have longer treatment durations, which may reflect successful inflammation control with maintained therapy.")
            else:
                f.write("No significant difference was found in steroid treatment duration between patients with active versus inactive uveitis. This could suggest that treatment duration is determined by factors other than current disease activity.")
                
        f.write("\n\nThese findings should be interpreted cautiously due to data constraints and the difficulty in fully accounting for disease severity and other clinical factors.")

else:
    print("Insufficient data after filtering for uveitis, steroid use, duration, and SVI. Analysis cannot proceed.")

print(f"\nAnalysis complete. Results saved to '{output_dir}' directory.") 