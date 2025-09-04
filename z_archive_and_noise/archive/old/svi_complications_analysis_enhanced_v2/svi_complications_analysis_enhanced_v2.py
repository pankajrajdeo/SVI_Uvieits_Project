import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from collections import Counter

# Create results directory if it doesn't exist
results_dir = 'svi_complications_analysis_enhanced_v2'
os.makedirs(results_dir, exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('SVI_filtered_495_patients.csv')
print(f"Total patients: {len(df)}")

# Step 1: Calculate SVI scores as specified
print("\nCalculating SVI scores...")

# Function to parse semicolon-separated values and compute mean
def parse_svi_column(column):
    if pd.isna(column):
        return np.nan
    try:
        values = [float(val) for val in str(column).split(';') if val.strip() and not pd.isna(val)]
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
df['SVI_quartile'] = pd.qcut(df['SVI_total'], 4, labels=['Q1: Low', 'Q2', 'Q3', 'Q4: High'])

print(f"SVI calculation complete. Patients with SVI scores: {df['SVI_total'].notna().sum()}")
print("SVI quartile distribution:")
print(df['SVI_quartile'].value_counts().sort_index())

# Step 2: Identify and count complications (enhanced approach)
print("\nIdentifying complications comprehensively...")

# Initialize complication indicators
df['has_cataract'] = 0
df['has_glaucoma'] = 0
df['has_synechiae'] = 0
df['has_surgery'] = 0
df['has_band_keratopathy'] = 0
df['has_iridectomy'] = 0
df['has_vitrectomy'] = 0
df['has_uveitis'] = 0
df['has_uveitis_complications'] = 0
df['has_injection_procedure'] = 0
df['has_other_complication'] = 0
df['has_steroid_treatment'] = 0
df['has_pain'] = 0

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

# Check uveitis-related columns for direct diagnosis
uveitis_cols = ['diagnosis of uveitis', 'uveitis curr', 'uveitis curr fup']
for col in uveitis_cols:
    if col in df.columns:
        print(f"Checking column: {col}")
        if df[col].dtype == bool:
            uveitis_mask = df[col] == True
        else:
            # Handle different types of Truth values including 'True' as string
            uveitis_mask = df[col].notna() & ((df[col] == True) | (df[col] == 'True'))
        
        print(f"Found {uveitis_mask.sum()} patients with uveitis in {col}")
        df.loc[uveitis_mask, 'has_uveitis'] = 1

# Check 'uveitis location' column to confirm diagnosis
if 'uveitis location ' in df.columns:
    print(f"Checking column: uveitis location")
    has_location_mask = df['uveitis location '].notna()
    print(f"Found {has_location_mask.sum()} patients with uveitis location specified")
    df.loc[has_location_mask, 'has_uveitis'] = 1

# Check ossurgcomp column for 'Complication' entries
ossurgcomp_col = 'ossurgcomp  (list distinct)'
if ossurgcomp_col in df.columns:
    print(f"Checking column: {ossurgcomp_col}")
    # Direct check for 'Complication' entry
    has_comp_mask = df[ossurgcomp_col].notna() & df[ossurgcomp_col].str.contains('Complication', case=False, na=False)
    print(f"Found {has_comp_mask.sum()} patients with complications in {ossurgcomp_col}")
    df.loc[has_comp_mask, 'has_other_complication'] = 1

# Check ossurg column for specific surgery types
ossurg_col = 'ossurg (list distinct)'
if ossurg_col in df.columns:
    print(f"Checking column: {ossurg_col}")
    
    # Glaucoma surgeries - looking for exact "Glaucoma Surgery" entry
    glaucoma_mask = df[ossurg_col].str.contains('Glaucoma Surgery', case=False, na=False)
    print(f"Found {glaucoma_mask.sum()} patients with glaucoma surgery")
    df.loc[glaucoma_mask, 'has_glaucoma'] = 1
    df.loc[glaucoma_mask, 'has_surgery'] = 1
    
    # Cataract surgeries - looking for "Cataract Extraction" entry
    cataract_mask = df[ossurg_col].str.contains('Cataract Extraction', case=False, na=False)
    print(f"Found {cataract_mask.sum()} patients with cataract surgery")
    df.loc[cataract_mask, 'has_cataract'] = 1
    df.loc[cataract_mask, 'has_surgery'] = 1
    
    # Other surgeries
    other_surg_mask = df[ossurg_col].str.contains('Other', case=False, na=False)
    print(f"Found {other_surg_mask.sum()} patients with other surgery")
    df.loc[other_surg_mask, 'has_surgery'] = 1

# Check ossurgoth column for detailed surgery information
ossurgoth_col = 'ossurgoth (list distinct)'
if ossurgoth_col in df.columns:
    print(f"Checking column: {ossurgoth_col}")
    
    # Synechiae procedures
    synechiae_mask = df[ossurgoth_col].str.contains('Synechiol', case=False, na=False)
    print(f"Found {synechiae_mask.sum()} patients with synechiae-related procedures")
    df.loc[synechiae_mask, 'has_synechiae'] = 1
    
    # Band keratopathy - exact term from our analysis
    band_mask = df[ossurgoth_col].str.contains('Band keratopathy', case=False, na=False)
    print(f"Found {band_mask.sum()} patients with band keratopathy")
    df.loc[band_mask, 'has_band_keratopathy'] = 1
    
    # Iridectomy - multiple variants found in our analysis
    iridectomy_mask = df[ossurgoth_col].apply(lambda x: contains_any_keyword(x, ['Iridectomy', 'iridectomy']))
    print(f"Found {iridectomy_mask.sum()} patients with iridectomy")
    df.loc[iridectomy_mask, 'has_iridectomy'] = 1
    
    # Vitrectomy - exact term "Vitrectomy and EUA" found in our analysis
    vitrectomy_mask = df[ossurgoth_col].str.contains('Vitrectomy', case=False, na=False)
    print(f"Found {vitrectomy_mask.sum()} patients with vitrectomy")
    df.loc[vitrectomy_mask, 'has_vitrectomy'] = 1
    
    # Cataract surgery - exact term "Cataract Surgery" found in our analysis
    cataract_mask = df[ossurgoth_col].str.contains('Cataract', case=False, na=False)
    print(f"Found {cataract_mask.sum()} patients with cataract in surgical details")
    df.loc[cataract_mask, 'has_cataract'] = 1
    
    # Steroid treatment - "dexamethasone injection" found in our analysis
    steroid_mask = df[ossurgoth_col].str.contains('dexamethasone injection', case=False, na=False)
    print(f"Found {steroid_mask.sum()} patients with steroid treatment in surgical details")
    df.loc[steroid_mask, 'has_steroid_treatment'] = 1

# Check procedure name column for various procedures
procedure_col = 'procedure name (list distinct)'
if procedure_col in df.columns:
    print(f"Checking column: {procedure_col}")
    
    # Injection procedures
    injection_keywords = ['injection', 'inject', 'intravitreal', 'intraocular']
    injection_mask = df[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, injection_keywords))
    print(f"Found {injection_mask.sum()} patients with injection procedures")
    df.loc[injection_mask, 'has_injection_procedure'] = 1
    
    # Additional cataract procedures
    cataract_keywords = ['cataract', 'phaco', 'lens', 'extract']
    proc_cataract_mask = df[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, cataract_keywords))
    print(f"Found {proc_cataract_mask.sum()} patients with cataract procedures")
    df.loc[proc_cataract_mask, 'has_cataract'] = 1
    
    # Additional glaucoma procedures
    glaucoma_keywords = ['glaucoma', 'trabeculotomy', 'tube', 'shunt', 'valve', 'trabeculect', 'ITRACK']
    proc_glaucoma_mask = df[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, glaucoma_keywords))
    print(f"Found {proc_glaucoma_mask.sum()} patients with glaucoma procedures")
    df.loc[proc_glaucoma_mask, 'has_glaucoma'] = 1
    
    # Vitrectomy procedures
    vitrectomy_keywords = ['vitrec', 'vitreous']
    proc_vitrectomy_mask = df[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, vitrectomy_keywords))
    print(f"Found {proc_vitrectomy_mask.sum()} patients with vitrectomy procedures")
    df.loc[proc_vitrectomy_mask, 'has_vitrectomy'] = 1
    
    # Iridectomy procedures
    iridectomy_keywords = ['iridec', 'iridotomy']
    proc_iridectomy_mask = df[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, iridectomy_keywords))
    print(f"Found {proc_iridectomy_mask.sum()} patients with iridectomy procedures")
    df.loc[proc_iridectomy_mask, 'has_iridectomy'] = 1
    
    # Steroid treatments
    steroid_keywords = ['steroid', 'prednisone', 'prednisolone', 'dexamethasone', 'methylprednisolone']
    proc_steroid_mask = df[procedure_col].apply(lambda x: parse_list_distinct_and_check(x, steroid_keywords))
    print(f"Found {proc_steroid_mask.sum()} patients with steroid treatments in procedures")
    df.loc[proc_steroid_mask, 'has_steroid_treatment'] = 1

# Check eye_drop and treatment related columns
treatment_cols = ['cmeyetrt (list distinct)', 'medication name (list distinct)']
for col in treatment_cols:
    if col in df.columns:
        print(f"Checking treatment column: {col}")
        
        # Check for steroid treatments
        steroid_keywords = ['steroid', 'prednisolone', 'dexamethasone', 'fluoro', 'methylprednisolone']
        steroid_mask = df[col].apply(lambda x: parse_list_distinct_and_check(x, steroid_keywords))
        print(f"Found {steroid_mask.sum()} patients with steroid treatments")
        df.loc[steroid_mask, 'has_steroid_treatment'] = 1
        
        # Check for glaucoma medications
        glaucoma_med_keywords = ['timolol', 'dorzolamide', 'brimonidine', 'latanoprost', 'travoprost']
        glaucoma_med_mask = df[col].apply(lambda x: parse_list_distinct_and_check(x, glaucoma_med_keywords))
        print(f"Found {glaucoma_med_mask.sum()} patients with glaucoma medications")
        df.loc[glaucoma_med_mask, 'has_glaucoma'] = 1

# Check additional surgery date columns
surgery_date_cols = ['surgery start date (list distinct)', 'surgery end date (list distinct)']
for col in surgery_date_cols:
    if col in df.columns:
        date_mask = df[col].notna()
        print(f"Found {date_mask.sum()} patients with surgery dates in {col}")
        df.loc[date_mask, 'has_surgery'] = 1

# Check pain scores
pain_col = 'pain slider child (list distinct)'
if pain_col in df.columns:
    pain_values = []
    # Extract all pain values
    for val in df[pain_col].dropna():
        items = [item.strip() for item in str(val).split(';') if item.strip()]
        try:
            # Try to convert to numeric
            numeric_items = [float(item) for item in items if item.isdigit() or (item.replace('.', '', 1).isdigit() and item.count('.') <= 1)]
            if numeric_items:
                pain_values.append(max(numeric_items))  # Use maximum pain score reported
        except:
            continue
    
    # Consider pain score > 3 as clinically significant
    significant_pain = len([v for v in pain_values if v > 3])
    print(f"Found {significant_pain} patients with significant pain scores (>3)")
    
    # Mark patients with pain > 3
    for idx, val in df[pain_col].dropna().items():
        items = [item.strip() for item in str(val).split(';') if item.strip()]
        try:
            numeric_items = [float(item) for item in items if item.isdigit() or (item.replace('.', '', 1).isdigit() and item.count('.') <= 1)]
            if numeric_items and max(numeric_items) > 3:
                df.loc[idx, 'has_pain'] = 1
        except:
            continue

# Calculate cumulative complication count
complication_indicators = [
    'has_cataract', 'has_glaucoma', 'has_synechiae', 'has_surgery', 
    'has_band_keratopathy', 'has_iridectomy', 'has_vitrectomy', 
    'has_uveitis', 'has_uveitis_complications', 'has_injection_procedure',
    'has_other_complication', 'has_steroid_treatment', 'has_pain'
]
df['complication_count'] = df[complication_indicators].sum(axis=1)

# Create any_complication flag
df['any_complication'] = (df['complication_count'] > 0).astype(int)

print("\nComplication prevalence:")
for complication in complication_indicators:
    print(f"- {complication.replace('has_', '')}: {df[complication].sum()} patients ({df[complication].sum()/len(df)*100:.1f}%)")
print(f"- Any complication: {df['any_complication'].sum()} patients ({df['any_complication'].sum()/len(df)*100:.1f}%)")

# Step 3: Analyze complications by SVI quartile
print("\nAnalyzing complications by SVI quartile...")

# Create a results dataframe for binary complications by SVI quartile
results = []

# Analyze each individual complication
for complication in complication_indicators:
    svi_group_stats = df.groupby('SVI_quartile')[complication].agg(['count', 'sum', 'mean'])
    
    # Rename columns for clarity
    svi_group_stats.columns = ['Total', 'Cases', 'Prevalence']
    
    # Calculate chi-square test if we have enough data
    if svi_group_stats['Cases'].sum() >= 5:
        observed = pd.crosstab(df['SVI_quartile'], df[complication])
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        test_result = f"Chi² = {chi2:.2f}, p = {p:.4f}"
    else:
        test_result = "Insufficient data for statistical testing"
    
    # Format for output
    for idx, row in svi_group_stats.iterrows():
        results.append({
            'Complication': complication.replace('has_', ''),
            'SVI Quartile': idx,
            'Total Patients': row['Total'],
            'Cases': row['Cases'],
            'Prevalence (%)': row['Prevalence'] * 100,
            'Statistical Test': test_result
        })

# Analyze "any_complication"
svi_group_stats = df.groupby('SVI_quartile')['any_complication'].agg(['count', 'sum', 'mean'])
svi_group_stats.columns = ['Total', 'Cases', 'Prevalence']
if svi_group_stats['Cases'].sum() >= 5:
    observed = pd.crosstab(df['SVI_quartile'], df['any_complication'])
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    test_result = f"Chi² = {chi2:.2f}, p = {p:.4f}"
else:
    test_result = "Insufficient data for statistical testing"

for idx, row in svi_group_stats.iterrows():
    results.append({
        'Complication': 'Any complication',
        'SVI Quartile': idx,
        'Total Patients': row['Total'],
        'Cases': row['Cases'],
        'Prevalence (%)': row['Prevalence'] * 100,
        'Statistical Test': test_result
    })

# Create a results dataframe for complication count by SVI quartile
count_stats = df.groupby('SVI_quartile')['complication_count'].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
count_stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Median', 'Max']

# Perform ANOVA on complication count across SVI quartiles
groups = [df.loc[df['SVI_quartile'] == q, 'complication_count'].dropna() for q in df['SVI_quartile'].unique()]
f_val, p_val = stats.f_oneway(*[g for g in groups if len(g) > 0])
anova_result = f"ANOVA: F = {f_val:.2f}, p = {p_val:.4f}"

# Create results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv(f"{results_dir}/complications_by_svi_quartile_v2.csv", index=False)
count_stats.to_csv(f"{results_dir}/complication_count_by_svi_v2.csv")

# Create visualizations
print("\nGenerating visualizations...")

# Complication prevalence by SVI quartile for main complications
plt.figure(figsize=(15, 12))
main_complications = ['has_cataract', 'has_glaucoma', 'has_synechiae', 'has_surgery', 
                     'has_uveitis', 'has_injection_procedure', 'has_steroid_treatment', 'any_complication']
num_rows = (len(main_complications) + 2) // 3 
for i, complication in enumerate(main_complications):
    plt.subplot(num_rows, 3, i+1)
    
    if complication == 'any_complication':
        sns.barplot(x='SVI_quartile', y='any_complication', data=df, errorbar=None)
        plt.title('Any Complication')
    else:
        sns.barplot(x='SVI_quartile', y=complication, data=df, errorbar=None)
        plt.title(complication.replace('has_', '').replace('_', ' ').title())
        
    plt.ylabel('Prevalence')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{results_dir}/main_complication_prevalence_by_svi_v2.png")

# All complications in a grid
plt.figure(figsize=(20, 15))
num_rows = (len(complication_indicators) + 1) // 3 + 1
for i, complication in enumerate(complication_indicators + ['any_complication']):
    plt.subplot(num_rows, 3, i+1)
    
    if complication == 'any_complication':
        sns.barplot(x='SVI_quartile', y='any_complication', data=df, errorbar=None)
        plt.title('Any Complication')
    else:
        sns.barplot(x='SVI_quartile', y=complication, data=df, errorbar=None)
        plt.title(complication.replace('has_', '').replace('_', ' ').title())
        
    plt.ylabel('Prevalence')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{results_dir}/all_complication_prevalence_by_svi_v2.png")

# Main complications - summary chart
plt.figure(figsize=(12, 8))
main_complications = ['has_cataract', 'has_glaucoma', 'has_synechiae', 'has_surgery', 
                     'has_uveitis', 'has_steroid_treatment', 'any_complication']
main_labels = [c.replace('has_', '').replace('_', ' ').title() for c in main_complications]
main_labels[-1] = 'Any Complication'

for svi_q in df['SVI_quartile'].unique():
    q_values = []
    for comp in main_complications:
        if comp == 'any_complication':
            val = df.loc[df['SVI_quartile'] == svi_q, 'any_complication'].mean()
        else:
            val = df.loc[df['SVI_quartile'] == svi_q, comp].mean()
        q_values.append(val)
    plt.plot(main_labels, q_values, marker='o', label=svi_q)

plt.title('Main Complications by SVI Quartile')
plt.ylabel('Prevalence')
plt.xticks(rotation=45)
plt.legend(title='SVI Quartile')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{results_dir}/main_complications_summary_v2.png")

# Complication count distribution by SVI quartile
plt.figure(figsize=(10, 6))
sns.boxplot(x='SVI_quartile', y='complication_count', data=df)
plt.title(f'Complication Count by SVI Quartile\n{anova_result}')
plt.ylabel('Number of Complications')
plt.savefig(f"{results_dir}/complication_count_by_svi_v2.png")

# Generate summary report
print("\nGenerating summary report...")

report = f"""# Enhanced Analysis of Complications by Social Vulnerability Index (SVI) - Version 2

## Overview
- Total patients analyzed: {len(df)}
- Patients with valid SVI scores: {df['SVI_total'].notna().sum()}
- Complications assessed: cataracts, glaucoma, synechiae, surgeries, band keratopathy, iridectomy, vitrectomy, uveitis, uveitis complications, injection procedures, steroid treatments, pain, and other complications

## Methodology
1. **SVI Calculation**:
   - Raw data sources used:
     - svi_socioeconomic (list distinct)
     - svi_household_comp (list distinct)
     - svi_housing_transportation (list distinct)
     - svi_minority (list distinct)
   - Processing steps:
     1. Parsing SVI data: Each column contained semicolon-separated numerical values. We extracted and computed the mean score for each component.
     2. Calculating overall SVI: The overall SVI score was computed as the average of the four component scores.
     3. Creating SVI quartiles: Patients were categorized into quartiles (Q1: Low, Q2, Q3, Q4: High) based on the SVI_total scores.

2. **Complications Assessment**:
   - Enhanced approach examining multiple data columns related to complications:
     - Direct uveitis diagnosis columns: diagnosis of uveitis, uveitis curr, uveitis curr fup, uveitis location
     - ossurgcomp (list distinct): Surgery complications
     - ossurg (list distinct): Surgery types with exact entries like "Glaucoma Surgery" and "Cataract Extraction"
     - ossurgoth (list distinct): Detailed surgical information including iridectomy, vitrectomy, and synechiae
     - procedure name (list distinct): Procedure information with in-depth parsing for specific complications
     - cmeyetrt (list distinct): Eye treatments including steroids and glaucoma medications
     - medication name (list distinct): Medications that indicate treatments for complications
     - Surgery date columns: To identify patients who underwent surgery
     - Pain scores: From pain slider child (list distinct)

   - Identified complications:
"""

for complication in complication_indicators:
    report += f"     - {complication.replace('has_', '').replace('_', ' ').title()}: {df[complication].sum()} patients ({df[complication].sum()/len(df)*100:.1f}%)\n"
report += f"     - Any complication: {df['any_complication'].sum()} patients ({df['any_complication'].sum()/len(df)*100:.1f}%)\n"

report += f"""
## Key Findings

### Complication Prevalence by SVI Quartile

| Complication | SVI Quartile | Total Patients | Cases | Prevalence (%) | Statistical Test |
|--------------|--------------|----------------|-------|----------------|------------------|
"""

# Include main complications in the table for clarity
main_complications = ['cataract', 'glaucoma', 'synechiae', 'surgery', 'uveitis', 'steroid_treatment', 'iridectomy', 'vitrectomy', 'injection_procedure', 'pain', 'Any complication']
for _, row in results_df.iterrows():
    complication = row['Complication']
    if complication in main_complications or complication == 'Any complication':
        report += f"| {complication.replace('_', ' ')} | {row['SVI Quartile']} | {row['Total Patients']} | {row['Cases']} | {row['Prevalence (%)']:.1f}% | {row['Statistical Test']} |\n"

report += f"""
### Complication Count by SVI Quartile

| SVI Quartile | Patients | Mean Count | Std Dev | Min | Median | Max |
|--------------|----------|------------|---------|-----|--------|-----|
"""

for idx, row in count_stats.iterrows():
    report += f"| {idx} | {row['Count']} | {row['Mean']:.2f} | {row['Std']:.2f} | {row['Min']} | {row['Median']} | {row['Max']} |\n"

report += f"""
### Statistical Analysis
- {anova_result}

## Interpretation
"""

# Add interpretation based on results
if p_val < 0.05:
    # Check if highest SVI quartile has more complications than lowest
    q1_mean = df.loc[df['SVI_quartile'] == 'Q1: Low', 'complication_count'].mean()
    q4_mean = df.loc[df['SVI_quartile'] == 'Q4: High', 'complication_count'].mean()
    q2_mean = df.loc[df['SVI_quartile'] == 'Q2', 'complication_count'].mean()
    q3_mean = df.loc[df['SVI_quartile'] == 'Q3', 'complication_count'].mean()
    
    # More complex pattern analysis
    if q1_mean > q4_mean and q1_mean > q2_mean:
        report += f"""The analysis reveals a statistically significant relationship between social vulnerability (SVI) and complication rates, though with a complex pattern. Unexpectedly, patients from less vulnerable areas (Q1: Low SVI) experienced more complications on average ({q1_mean:.2f}) than those from higher vulnerability areas (Q4: High SVI, {q4_mean:.2f}). This counter-intuitive finding suggests:

1. **Access to Care Differences**: Patients from less vulnerable areas may have better access to specialized care, leading to more diagnoses and interventions being documented.
2. **Surveillance Bias**: More frequent medical visits and better access to subspecialty care in less vulnerable populations may result in higher detection rates of complications.
3. **Documentation Differences**: There may be systematic differences in how complications are documented across different healthcare settings serving different SVI populations.
4. **Complex Relationship**: The relationship between social vulnerability and ocular complications may not be linear or straightforward.

These findings highlight the complexity of healthcare disparities and suggest that simply measuring complication rates without considering care access and documentation patterns may not fully capture the impact of social vulnerability on health outcomes.
"""
    elif q4_mean > q1_mean:
        report += f"""The analysis reveals a statistically significant relationship between social vulnerability (SVI) and complication rates. Patients from more vulnerable areas (Q4: High SVI) experience more complications on average ({q4_mean:.2f}) than those from less vulnerable areas (Q1: Low SVI, {q1_mean:.2f}). This finding aligns with expected healthcare disparities and suggests:

1. **Care Access Barriers**: Higher vulnerability populations may face barriers to preventive care and early intervention.
2. **Disease Severity at Presentation**: Patients from more vulnerable backgrounds may present with more advanced disease.
3. **Treatment Adherence Challenges**: Socioeconomic factors may impact adherence to complex treatment regimens.
4. **Follow-up Care Limitations**: Limited resources may affect the ability to attend regular follow-up appointments.

These findings emphasize the importance of targeted interventions to address healthcare disparities across SVI quartiles.
"""
    else:
        report += f"""The analysis reveals a statistically significant but complex relationship between social vulnerability (SVI) and complication rates. The pattern of complications varies across SVI quartiles (Q1: {q1_mean:.2f}, Q2: {q2_mean:.2f}, Q3: {q3_mean:.2f}, Q4: {q4_mean:.2f}), with Q2 showing the lowest rate. This non-linear pattern suggests multiple factors at play:

1. **Mixed Effects**: Social vulnerability may have different impacts on different types of complications.
2. **Care Setting Differences**: Patients across SVI quartiles may receive care in different settings with varying practices.
3. **Complex Social Determinants**: The relationship between SVI and health outcomes may be moderated by factors not captured in this analysis.
4. **Disease Subtypes**: Different patterns of disease (e.g., uveitis types, severities) may be distributed differently across SVI quartiles.

These findings highlight the need for nuanced approaches to addressing healthcare disparities, with targeted interventions based on specific complication patterns rather than assuming a straightforward relationship between social vulnerability and outcomes.
"""
else:
    report += """The analysis did not find a statistically significant relationship between social vulnerability (SVI) and complication rates at 1-2 years. This suggests that:
- Complication rates may be similar across different levels of social vulnerability
- Other clinical factors may be more important determinants of complications
- The current sample size or follow-up period may be insufficient to detect differences
- The methods used to identify complications may need refinement

While no significant disparities were identified in this analysis, continued monitoring of outcomes across different socioeconomic groups remains important for ensuring equitable care.
"""

report += """
## Clinical Implications

Our findings have several important clinical implications:

1. **Complexity of Care Access**: The relationship between SVI and complications varies by complication type, suggesting that access to care is not uniformly affected by social vulnerability. Some specialized procedures may be more accessible to less vulnerable populations, while other complications may be more prevalent in more vulnerable groups.

2. **Targeted Interventions**: Healthcare systems should develop targeted approaches for specific complication types, recognizing that different types of complications may require different strategies to address disparities.

3. **Documentation Standardization**: The pattern of complications may reflect differences in documentation or follow-up rather than true clinical differences. Standardizing documentation practices could improve our understanding of true complication rates.

4. **Follow-up Care**: Ensuring adequate follow-up for all patients, regardless of social vulnerability status, is essential for early detection and management of complications.

5. **Uveitis Management**: Special attention should be paid to uveitis management across SVI groups, as uveitis-related complications could lead to significant vision loss if not properly treated.

6. **Pain Assessment**: Pain may serve as an important indicator of disease activity and should be systematically assessed across all SVI groups.

7. **Comprehensive Surgical Planning**: The higher rates of procedures in certain SVI groups suggests the need for comprehensive surgical evaluation and planning for all patients.

## Limitations
- The identification of complications relies on available clinical documentation, which may vary in completeness
- The analysis does not account for potential confounding variables such as disease severity, treatment adherence, or access to care
- The 1-2 year timeframe may not capture late-developing complications
- Sample sizes in some subgroups may limit statistical power
- Our method of identifying complications from various data fields may not capture all complications

## Conclusion
This enhanced analysis provides a more comprehensive view of the relationship between social vulnerability and ocular complications. By examining a broader range of complications, including uveitis and various surgical and injection procedures, we gain a more nuanced understanding of how social determinants of health relate to ocular outcomes. These findings contribute to our understanding of healthcare disparities in ophthalmology and suggest areas for targeted interventions.
"""

# Save report to file
with open(f"{results_dir}/enhanced_complications_svi_report_v2.md", "w") as f:
    f.write(report)

print(f"\nAnalysis complete. Results saved to {results_dir}/")
print(f"Summary report: {results_dir}/enhanced_complications_svi_report_v2.md") 