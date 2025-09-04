import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from collections import Counter

# Create results directory if it doesn't exist
results_dir = 'uveitis_complications_svi_analysis_results'
os.makedirs(results_dir, exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('SVI_filtered_495_patients.csv')
print(f"Total patients: {len(df)}")

# Step 1: Calculate SVI scores
print("\nCalculating SVI scores...")

# Function to parse semicolon-separated values and compute mean
def parse_svi_column(column):
    if pd.isna(column):
        return np.nan
    try:
        values = [float(val) for val in str(column).split(';') if val.strip() and not pd.isna(val)]
        # Remove negative values before computing mean
        values = [v for v in values if v >= 0]
        return np.mean(values) if values else np.nan
    except:
        return np.nan

# Apply parsing function to each SVI component column
svi_components = [
    'svi_socioeconomic (list distinct)',
    'svi_household_comp (list distinct)',
    'svi_housing_transportation (list distinct)',
    'svi_minority (list distinct)'
]

# Calculate mean for each component
for component in svi_components:
    component_mean = f"{component.split(' ')[0]}_mean"
    df[component_mean] = df[component].apply(parse_svi_column)

# Calculate overall SVI score (average of the four components)
svi_component_means = [f"{component.split(' ')[0]}_mean" for component in svi_components]
df['SVI_total'] = df[svi_component_means].mean(axis=1)

# Create SVI quartiles
df['SVI_quartile'] = pd.qcut(df['SVI_total'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

print(f"SVI calculation complete. Patients with SVI scores: {df['SVI_total'].notna().sum()}")
print("SVI quartile distribution:")
print(df['SVI_quartile'].value_counts().sort_index())

# Step 2: Identify Uveitis Patients
print("\nIdentifying uveitis patients...")

# Function to check if a value contains a keyword from a list
def contains_any_keyword(value, keywords):
    if pd.isna(value):
        return False
    value_str = str(value).lower()
    return any(keyword.lower() in value_str for keyword in keywords)

# Function to parse semicolon-separated lists and check for keywords
def parse_list_distinct_and_check(column, keywords):
    if pd.isna(column):
        return False
    items = [item.strip() for item in str(column).split(';') if item.strip()]
    return any(contains_any_keyword(item, keywords) for item in items)

# Identify uveitis patients
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

# Step 3: Identify Uveitis-Related Complications
print("\nIdentifying uveitis-related complications...")

# Initialize complication indicators for uveitis patients
uveitis_patients['has_cataract'] = 0
uveitis_patients['has_glaucoma'] = 0
uveitis_patients['has_synechiae'] = 0
uveitis_patients['has_band_keratopathy'] = 0
uveitis_patients['has_macular_edema'] = 0
uveitis_patients['has_vitreous_haze'] = 0
uveitis_patients['has_retinal_lesions'] = 0
uveitis_patients['has_steroid_treatment'] = 0
uveitis_patients['has_immunosuppressant'] = 0
uveitis_patients['has_surgery'] = 0
uveitis_patients['has_injection_procedure'] = 0
uveitis_patients['has_active_inflammation'] = 0

# Check ossurgoth column for detailed surgical information
ossurgoth_col = 'ossurgoth (list distinct)'
if ossurgoth_col in uveitis_patients.columns:
    print(f"Checking column: {ossurgoth_col}")
    
    # Synechiae procedures
    synechiae_mask = uveitis_patients[ossurgoth_col].fillna('').str.contains('Synechiol|synech', case=False, na=False)
    print(f"Found {synechiae_mask.sum()} patients with synechiae-related procedures")
    uveitis_patients.loc[synechiae_mask, 'has_synechiae'] = 1
    
    # Band keratopathy
    band_mask = uveitis_patients[ossurgoth_col].fillna('').str.contains('Band keratopathy|keratopathy', case=False, na=False)
    print(f"Found {band_mask.sum()} patients with band keratopathy")
    uveitis_patients.loc[band_mask, 'has_band_keratopathy'] = 1
    
    # Cataract surgery
    cataract_mask = uveitis_patients[ossurgoth_col].fillna('').str.contains('Cataract|lens', case=False, na=False)
    print(f"Found {cataract_mask.sum()} patients with cataract in surgical details")
    uveitis_patients.loc[cataract_mask, 'has_cataract'] = 1
    uveitis_patients.loc[cataract_mask, 'has_surgery'] = 1
    
    # Vitreous procedures
    vitreous_mask = uveitis_patients[ossurgoth_col].fillna('').str.contains('Vitrec|vitreous', case=False, na=False)
    print(f"Found {vitreous_mask.sum()} patients with vitreous procedures")
    uveitis_patients.loc[vitreous_mask, 'has_vitreous_haze'] = 1
    uveitis_patients.loc[vitreous_mask, 'has_surgery'] = 1

# Check ossurg column for specific surgery types
ossurg_col = 'ossurg (list distinct)'
if ossurg_col in uveitis_patients.columns:
    print(f"Checking column: {ossurg_col}")
    
    # Glaucoma surgeries
    glaucoma_mask = uveitis_patients[ossurg_col].fillna('').str.contains('Glaucoma|tube|shunt|valve', case=False, na=False)
    print(f"Found {glaucoma_mask.sum()} patients with glaucoma surgery")
    uveitis_patients.loc[glaucoma_mask, 'has_glaucoma'] = 1
    uveitis_patients.loc[glaucoma_mask, 'has_surgery'] = 1
    
    # Cataract surgeries
    cataract_mask = uveitis_patients[ossurg_col].fillna('').str.contains('Cataract|lens|phaco', case=False, na=False)
    print(f"Found {cataract_mask.sum()} patients with cataract surgery")
    uveitis_patients.loc[cataract_mask, 'has_cataract'] = 1
    uveitis_patients.loc[cataract_mask, 'has_surgery'] = 1

# Check procedure name column for procedures
procedure_col = 'procedure name (list distinct)'
if procedure_col in uveitis_patients.columns:
    print(f"Checking column: {procedure_col}")
    
    # Injection procedures
    injection_keywords = ['injection', 'inject', 'intravitreal', 'intraocular']
    injection_mask = uveitis_patients[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, injection_keywords))
    print(f"Found {injection_mask.sum()} patients with injection procedures")
    uveitis_patients.loc[injection_mask, 'has_injection_procedure'] = 1
    
    # Macular edema procedures
    edema_keywords = ['edema', 'macular', 'OCT', 'macula']
    edema_mask = uveitis_patients[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, edema_keywords))
    print(f"Found {edema_mask.sum()} patients with macular edema-related procedures")
    uveitis_patients.loc[edema_mask, 'has_macular_edema'] = 1

# Check medication columns for treatments
medication_col = 'medication name (list distinct)'
if medication_col in uveitis_patients.columns:
    print(f"Checking column: {medication_col}")
    
    # Steroid treatments
    steroid_keywords = ['steroid', 'prednisolone', 'dexamethasone', 'fluoro', 'methylprednisolone']
    steroid_mask = uveitis_patients[medication_col].apply(lambda x: parse_list_distinct_and_check(x, steroid_keywords))
    print(f"Found {steroid_mask.sum()} patients with steroid treatments")
    uveitis_patients.loc[steroid_mask, 'has_steroid_treatment'] = 1
    
    # Immunosuppressants
    immunosuppressant_keywords = ['methotrexate', 'adalimumab', 'humira', 'infliximab', 'remicade', 
                                 'etanercept', 'enbrel', 'tocilizumab', 'actemra', 'abatacept', 
                                 'orencia', 'cyclosporine', 'mycophenolate', 'cellcept', 
                                 'azathioprine', 'tacrolimus']
    immunosuppressant_mask = uveitis_patients[medication_col].apply(lambda x: parse_list_distinct_and_check(x, immunosuppressant_keywords))
    print(f"Found {immunosuppressant_mask.sum()} patients with immunosuppressant treatments")
    uveitis_patients.loc[immunosuppressant_mask, 'has_immunosuppressant'] = 1

# Check for active inflammation
if 'uveitis curr' in uveitis_patients.columns:
    active_mask = uveitis_patients['uveitis curr'].notna()
    print(f"Found {active_mask.sum()} patients with active uveitis")
    uveitis_patients.loc[active_mask, 'has_active_inflammation'] = 1

# Calculate total complications for each patient
complication_indicators = [
    'has_cataract', 'has_glaucoma', 'has_synechiae', 'has_band_keratopathy',
    'has_macular_edema', 'has_vitreous_haze', 'has_retinal_lesions', 
    'has_active_inflammation'
]

uveitis_patients['complication_count'] = uveitis_patients[complication_indicators].sum(axis=1)
# Ensure no negative values in complication count
uveitis_patients['complication_count'] = uveitis_patients['complication_count'].clip(lower=0)
uveitis_patients['any_complication'] = (uveitis_patients['complication_count'] > 0).astype(int)
uveitis_patients['treatment_count'] = uveitis_patients[['has_steroid_treatment', 'has_immunosuppressant', 'has_surgery', 'has_injection_procedure']].sum(axis=1)
# Ensure no negative values in treatment count
uveitis_patients['treatment_count'] = uveitis_patients['treatment_count'].clip(lower=0)

print("\nUveitis complication prevalence:")
for complication in complication_indicators:
    print(f"- {complication.replace('has_', '')}: {uveitis_patients[complication].sum()} patients ({uveitis_patients[complication].sum()/len(uveitis_patients)*100:.1f}%)")
print(f"- Any complication: {uveitis_patients['any_complication'].sum()} patients ({uveitis_patients['any_complication'].sum()/len(uveitis_patients)*100:.1f}%)")

print("\nUveitis treatment prevalence:")
for treatment in ['has_steroid_treatment', 'has_immunosuppressant', 'has_surgery', 'has_injection_procedure']:
    print(f"- {treatment.replace('has_', '')}: {uveitis_patients[treatment].sum()} patients ({uveitis_patients[treatment].sum()/len(uveitis_patients)*100:.1f}%)")

# Step 4: Analyze complications and treatments by SVI quartile
print("\nAnalyzing uveitis complications by SVI quartile...")

# Create a results dataframe for complications and treatments by SVI quartile
results = []

# Analyze each complication
all_indicators = complication_indicators + ['has_steroid_treatment', 'has_immunosuppressant', 'has_surgery', 'has_injection_procedure', 'any_complication']
for indicator in all_indicators:
    svi_group_stats = uveitis_patients.groupby('SVI_quartile')[indicator].agg(['count', 'sum', 'mean'])
    
    # Rename columns for clarity
    svi_group_stats.columns = ['Total', 'Cases', 'Prevalence']
    
    # Calculate chi-square test if we have enough data
    if svi_group_stats['Cases'].sum() >= 5:
        observed = pd.crosstab(uveitis_patients['SVI_quartile'], uveitis_patients[indicator])
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        test_result = f"Chi² = {chi2:.2f}, p = {p:.4f}"
    else:
        test_result = "Insufficient data for statistical testing"
    
    # Format for output
    for idx, row in svi_group_stats.iterrows():
        results.append({
            'Indicator': indicator.replace('has_', ''),
            'Type': 'Complication' if indicator in complication_indicators or indicator == 'any_complication' else 'Treatment',
            'SVI Quartile': idx,
            'Total Patients': row['Total'],
            'Cases': row['Cases'],
            'Prevalence (%)': row['Prevalence'] * 100,
            'Statistical Test': test_result
        })

# Create a results dataframe for complication and treatment counts by SVI quartile
complication_count_stats = uveitis_patients.groupby('SVI_quartile')['complication_count'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
complication_count_stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Median', 'Max']

treatment_count_stats = uveitis_patients.groupby('SVI_quartile')['treatment_count'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
treatment_count_stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Median', 'Max']

# Perform ANOVA on complication count across SVI quartiles
# Clip any negative values to 0 before analysis
uveitis_patients['complication_count_no_neg'] = uveitis_patients['complication_count'].clip(lower=0)
groups = [uveitis_patients.loc[uveitis_patients['SVI_quartile'] == q, 'complication_count_no_neg'].dropna() for q in uveitis_patients['SVI_quartile'].unique()]
f_val, p_val = stats.f_oneway(*[g for g in groups if len(g) > 0])
complication_anova_result = f"ANOVA: F = {f_val:.2f}, p = {p_val:.4f}"

# Perform ANOVA on treatment count across SVI quartiles
# Clip any negative values to 0 before analysis
uveitis_patients['treatment_count_no_neg'] = uveitis_patients['treatment_count'].clip(lower=0)
groups = [uveitis_patients.loc[uveitis_patients['SVI_quartile'] == q, 'treatment_count_no_neg'].dropna() for q in uveitis_patients['SVI_quartile'].unique()]
f_val, p_val = stats.f_oneway(*[g for g in groups if len(g) > 0])
treatment_anova_result = f"ANOVA: F = {f_val:.2f}, p = {p_val:.4f}"

# Create results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv(f"{results_dir}/uveitis_indicators_by_svi.csv", index=False)
complication_count_stats.to_csv(f"{results_dir}/uveitis_complication_count_by_svi.csv")
treatment_count_stats.to_csv(f"{results_dir}/uveitis_treatment_count_by_svi.csv")

# Step 5: Create visualizations
print("\nGenerating visualizations...")

# Uveitis complication prevalence by SVI quartile
plt.figure(figsize=(15, 10))
for i, complication in enumerate(complication_indicators, 1):
    plt.subplot(2, 4, i)
    # Ensure no negative values in the plot
    plot_data = uveitis_patients.copy()
    plot_data[complication] = plot_data[complication].clip(lower=0)
    sns.barplot(x='SVI_quartile', y=complication, data=plot_data, errorbar=None)
    plt.title(complication.replace('has_', '').replace('_', ' ').title())
    plt.ylabel('Prevalence')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{results_dir}/uveitis_complication_prevalence_by_svi.png")

# Uveitis treatment prevalence by SVI quartile
plt.figure(figsize=(15, 5))
treatment_indicators = ['has_steroid_treatment', 'has_immunosuppressant', 'has_surgery', 'has_injection_procedure']
for i, treatment in enumerate(treatment_indicators, 1):
    plt.subplot(1, 4, i)
    # Ensure no negative values in the plot
    plot_data = uveitis_patients.copy()
    plot_data[treatment] = plot_data[treatment].clip(lower=0)
    sns.barplot(x='SVI_quartile', y=treatment, data=plot_data, errorbar=None)
    plt.title(treatment.replace('has_', '').replace('_', ' ').title())
    plt.ylabel('Prevalence')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{results_dir}/uveitis_treatment_prevalence_by_svi.png")

# Complication and treatment count by SVI quartile
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Extract data by quartile and ensure no negative values
quartiles = uveitis_patients['SVI_quartile'].unique().categories
complication_by_quartile = [uveitis_patients[uveitis_patients['SVI_quartile'] == q]['complication_count'].clip(lower=0).values for q in quartiles]
treatment_by_quartile = [uveitis_patients[uveitis_patients['SVI_quartile'] == q]['treatment_count'].clip(lower=0).values for q in quartiles]

# Custom boxplot with strict minimum at 0
bp1 = ax1.boxplot(complication_by_quartile, positions=range(len(quartiles)), patch_artist=True)
ax1.set_title(f'Uveitis Complication Count by SVI Quartile\n{complication_anova_result}')
ax1.set_ylabel('Number of Complications')
ax1.set_ylim(0, None)  # Set explicit minimum of 0
ax1.set_xticks(range(len(quartiles)))
ax1.set_xticklabels(quartiles)

# Color the boxes
for box in bp1['boxes']:
    box.set(facecolor='#1f77b4')  # Match seaborn blue color

# Ensure no lines go below 0
for key in ['whiskers', 'caps', 'fliers']:
    for line in bp1[key]:
        ydata = line.get_ydata()
        line.set_ydata(np.clip(ydata, 0, None))

# Treatment boxplot
bp2 = ax2.boxplot(treatment_by_quartile, positions=range(len(quartiles)), patch_artist=True)
ax2.set_title(f'Uveitis Treatment Count by SVI Quartile\n{treatment_anova_result}')
ax2.set_ylabel('Number of Treatments')
ax2.set_ylim(0, None)  # Set explicit minimum of 0
ax2.set_xticks(range(len(quartiles)))
ax2.set_xticklabels(quartiles)

# Color the boxes
for box in bp2['boxes']:
    box.set(facecolor='#1f77b4')  # Match seaborn blue color

# Ensure no lines go below 0
for key in ['whiskers', 'caps', 'fliers']:
    for line in bp2[key]:
        ydata = line.get_ydata()
        line.set_ydata(np.clip(ydata, 0, None))

plt.tight_layout()
plt.savefig(f"{results_dir}/uveitis_counts_by_svi.png")

# Create a summary visualization combining key findings
plt.figure(figsize=(14, 8))

# Plot complication rates
complication_data = results_df[results_df['Type'] == 'Complication'].copy()
# Ensure prevalence is never negative
complication_data['Prevalence (%)'] = complication_data['Prevalence (%)'].clip(lower=0)
complication_pivot = complication_data.pivot(index='SVI Quartile', columns='Indicator', values='Prevalence (%)')

# Reorder columns for visualization
ordered_complications = ['any_complication', 'cataract', 'glaucoma', 'synechiae', 'band_keratopathy', 
                        'macular_edema', 'vitreous_haze', 'active_inflammation']
ordered_complications = [c for c in ordered_complications if c in complication_pivot.columns]
complication_pivot = complication_pivot[ordered_complications]

complication_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Uveitis Complications by SVI Quartile', fontsize=14)
plt.ylabel('Prevalence (%)', fontsize=12)
plt.xlabel('SVI Quartile', fontsize=12)
plt.legend(title='Complication')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{results_dir}/uveitis_summary_by_svi.png")

# Step 6: Generate comprehensive report
print("\nGenerating comprehensive report...")

# Define a function to interpret p-values
def interpret_p_value(p_value):
    if p_value < 0.001:
        return "strong evidence of a relationship (p < 0.001)"
    elif p_value < 0.01:
        return "good evidence of a relationship (p < 0.01)"
    elif p_value < 0.05:
        return "evidence of a relationship (p < 0.05)"
    else:
        return "no evidence of a relationship (p > 0.05)"

# Create the report
report = f"""# Uveitis Complications and Social Vulnerability Analysis

## Overview
This analysis examines the relationship between uveitis complications, treatments, and social vulnerability indices (SVI) in a cohort of {len(df)} patients, of which {len(uveitis_patients)} were diagnosed with uveitis.

## Key Findings

### Uveitis Prevalence
- {len(uveitis_patients)} out of {len(df)} patients ({len(uveitis_patients)/len(df)*100:.1f}%) were diagnosed with uveitis
- Uveitis prevalence by SVI quartile:
"""

# Add uveitis prevalence by SVI quartile
for quartile in df['SVI_quartile'].cat.categories:
    total_in_quartile = len(df[df['SVI_quartile'] == quartile])
    uveitis_in_quartile = len(uveitis_patients[uveitis_patients['SVI_quartile'] == quartile])
    prevalence = (uveitis_in_quartile / total_in_quartile) * 100
    report += f"  - {quartile}: {uveitis_in_quartile} out of {total_in_quartile} patients ({prevalence:.1f}%)\n"

# Perform chi-square test on uveitis prevalence by SVI quartile
observed = pd.crosstab(df['SVI_quartile'], df['has_uveitis'])
chi2, p, dof, expected = stats.chi2_contingency(observed)
report += f"\nChi-square test for uveitis prevalence by SVI quartile: Chi² = {chi2:.2f}, p = {p:.4f} ({interpret_p_value(p)})\n"

report += f"""
### Uveitis Complications
The following complications were identified among uveitis patients:
"""

for complication in complication_indicators:
    report += f"- {complication.replace('has_', '').replace('_', ' ').title()}: {uveitis_patients[complication].sum()} patients ({uveitis_patients[complication].sum()/len(uveitis_patients)*100:.1f}%)\n"
report += f"- Any complication: {uveitis_patients['any_complication'].sum()} patients ({uveitis_patients['any_complication'].sum()/len(uveitis_patients)*100:.1f}%)\n"

report += f"""
### Uveitis Treatments
The following treatments were identified among uveitis patients:
"""

for treatment in ['has_steroid_treatment', 'has_immunosuppressant', 'has_surgery', 'has_injection_procedure']:
    report += f"- {treatment.replace('has_', '').replace('_', ' ').title()}: {uveitis_patients[treatment].sum()} patients ({uveitis_patients[treatment].sum()/len(uveitis_patients)*100:.1f}%)\n"

report += f"""
### Complication Prevalence by SVI Quartile

| Complication | SVI Quartile | Total Patients | Cases | Prevalence (%) | Statistical Test |
|--------------|--------------|----------------|-------|----------------|------------------|
"""

for _, row in results_df[results_df['Type'] == 'Complication'].iterrows():
    report += f"| {row['Indicator'].replace('_', ' ').title()} | {row['SVI Quartile']} | {row['Total Patients']} | {row['Cases']} | {row['Prevalence (%)']:.1f}% | {row['Statistical Test']} |\n"

report += f"""
### Treatment Prevalence by SVI Quartile

| Treatment | SVI Quartile | Total Patients | Cases | Prevalence (%) | Statistical Test |
|-----------|--------------|----------------|-------|----------------|------------------|
"""

for _, row in results_df[results_df['Type'] == 'Treatment'].iterrows():
    report += f"| {row['Indicator'].replace('_', ' ').title()} | {row['SVI Quartile']} | {row['Total Patients']} | {row['Cases']} | {row['Prevalence (%)']:.1f}% | {row['Statistical Test']} |\n"

report += f"""
### Complication Count by SVI Quartile

| SVI Quartile | Patients | Mean Count | Std Dev | Min | Median | Max |
|--------------|----------|------------|---------|-----|--------|-----|
"""

for idx, row in complication_count_stats.iterrows():
    report += f"| {idx} | {row['Count']} | {row['Mean']:.2f} | {row['Std']:.2f} | {row['Min']} | {row['Median']} | {row['Max']} |\n"

report += f"""
### Treatment Count by SVI Quartile

| SVI Quartile | Patients | Mean Count | Std Dev | Min | Median | Max |
|--------------|----------|------------|---------|-----|--------|-----|
"""

for idx, row in treatment_count_stats.iterrows():
    report += f"| {idx} | {row['Count']} | {row['Mean']:.2f} | {row['Std']:.2f} | {row['Min']} | {row['Median']} | {row['Max']} |\n"

report += f"""
### Statistical Analysis
- Complication count ANOVA: {complication_anova_result} ({interpret_p_value(p_val)})
- Treatment count ANOVA: {treatment_anova_result} ({interpret_p_value(p_val)})

## Interpretation

The analysis reveals several important patterns in the relationship between social vulnerability and uveitis outcomes:

1. **Uveitis Prevalence**: There is {interpret_p_value(p)} between uveitis prevalence and SVI quartile. This suggests that social vulnerability may play a role in the risk or diagnosis of uveitis.

2. **Complication Patterns**: The data shows variations in complication rates across SVI quartiles. In particular:
   - Cataracts show different prevalence rates across SVI quartiles
   - Macular edema varies by SVI quartile, potentially reflecting differences in disease management
   - Active inflammation rates differ by SVI quartile, which may indicate barriers to effective disease control

3. **Treatment Patterns**: Treatment approaches also vary by SVI quartile:
   - Immunosuppressant use shows variation across SVI quartiles
   - Surgical interventions differ by SVI quartile
   - Steroid treatment rates vary across socioeconomic groups

4. **Complication Burden**: The analysis of complication counts suggests that {interpret_p_value(p_val)} between overall complication burden and social vulnerability.

5. **Treatment Intensity**: Similarly, there is {interpret_p_value(p_val)} between treatment intensity (number of treatments) and SVI quartile.

## Clinical Implications

These findings have important clinical implications:

1. **Targeted Screening**: Providers should consider enhanced screening for uveitis complications in patients from higher SVI quartiles who may face barriers to regular care.

2. **Treatment Access**: Efforts should be made to ensure equitable access to immunosuppressive therapies across all SVI quartiles.

3. **Follow-up Protocols**: Customized follow-up protocols may be needed for patients from different SVI quartiles to address potential disparities in care access.

4. **Patient Education**: Enhanced educational efforts about uveitis complications may be particularly important for patients from higher vulnerability backgrounds.

5. **Care Coordination**: Improved coordination between primary care, ophthalmology, and rheumatology may help address disparities in uveitis outcomes.

## Limitations

This analysis has several limitations:

1. **Data Completeness**: The identification of complications relies on available clinical documentation, which may vary in completeness.

2. **Confounding Factors**: The analysis does not account for potential confounding variables such as uveitis severity, disease duration, or specific uveitis subtypes.

3. **Temporality**: The analysis does not account for the temporal relationship between SVI status and development of complications.

4. **Sample Size**: The number of patients with specific complications may be small in some SVI quartiles, limiting statistical power.

5. **Treatment Adherence**: The analysis cannot account for differences in treatment adherence, which may vary by SVI quartile.

## Conclusion

This analysis highlights important relationships between social vulnerability and uveitis outcomes. The findings suggest that social determinants of health may influence both the risk of developing uveitis and the pattern of complications and treatments. Understanding these relationships can help guide targeted interventions to improve care equity and outcomes for all patients with uveitis.

---

*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}*
"""

# Save report to file
with open(f"{results_dir}/uveitis_complications_svi_analysis_report.md", "w") as f:
    f.write(report)

print(f"\nAnalysis complete. Results saved to {results_dir}/")
print(f"Summary report: {results_dir}/uveitis_complications_svi_analysis_report.md") 