import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Create results directory
results_dir = 'uveitis_treatment_analysis_results'
os.makedirs(results_dir, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('SVI_filtered_495_patients.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Identify uveitis columns
print("\nLooking for uveitis-related columns...")
uveitis_cols = [col for col in df.columns if 'uveitis' in col.lower() or 'uv ' in col.lower() or 'uv_' in col.lower()]
print(f"Found {len(uveitis_cols)} uveitis-related columns:")
for i, col in enumerate(uveitis_cols, 1):
    non_null = df[col].count()
    percent = (non_null / len(df)) * 100
    print(f"{i}. {col}: {non_null} non-null values ({percent:.1f}%)")

# Identify medication columns
print("\nLooking for medication columns...")
med_cols = [col for col in df.columns if 'medication' in col.lower() or 'med ' in col.lower()]
print(f"Found {len(med_cols)} medication columns:")
for i, col in enumerate(med_cols, 1):
    non_null = df[col].count()
    percent = (non_null / len(df)) * 100
    print(f"{i}. {col}: {non_null} non-null values ({percent:.1f}%)")

# Define key immunosuppressants to look for
immunosuppressants = [
    'methotrexate', 'adalimumab', 'humira', 'infliximab', 'remicade', 
    'etanercept', 'enbrel', 'tocilizumab', 'actemra', 'abatacept', 
    'orencia', 'cyclosporine', 'mycophenolate', 'cellcept', 
    'azathioprine', 'tacrolimus'
]

# Filter for patients with confirmed uveitis
print("\nIdentifying patients with uveitis...")
uveitis_indicators = ['diagnosis of uveitis', 'uveitis curr', 'uveitis curr fup']
uveitis_mask = df[uveitis_indicators].notna().any(axis=1)
uveitis_patients = df[uveitis_mask].copy()
print(f"Found {len(uveitis_patients)} patients with confirmed uveitis")

# Extract uveitis diagnosis year
print("\nExtracting uveitis diagnosis years...")
uveitis_patients['uveitis_diagnosis_year'] = uveitis_patients['date of uv diagnosisy']
print(f"Found {uveitis_patients['uveitis_diagnosis_year'].count()} patients with diagnosis year")

# Identify immunosuppressive medication years
print("\nIdentifying immunosuppressive medication years...")
main_med_col = 'medication name (list distinct)'
main_med_start_col = 'medication start date (list distinct)'

if main_med_col in df.columns and main_med_start_col in df.columns:
    # Prepare data for time-to-treatment analysis
    results = []
    
    for idx, patient in uveitis_patients.iterrows():
        patient_id = patient.get('patid1', f"Patient_{idx}")
        diagnosis_year = patient.get('uveitis_diagnosis_year')
        
        # Skip patients without diagnosis year
        if pd.isna(diagnosis_year):
            continue
        
        diagnosis_year = int(float(diagnosis_year))
        
        # Extract medication info
        med_names = str(patient[main_med_col]).split(';') if pd.notna(patient[main_med_col]) else []
        med_years = str(patient[main_med_start_col]).split(';') if pd.notna(patient[main_med_start_col]) else []
        
        # Clean up the years
        year_pattern = r'\b\d{4}\b'
        clean_years = []
        for year_str in med_years:
            matches = re.findall(year_pattern, year_str)
            if matches:
                clean_years.append(int(matches[0]))
            elif year_str.strip().isdigit():
                clean_years.append(int(year_str.strip()))
        
        # Find if any medications are immunosuppressants
        has_immunosuppressant = False
        for med_name in med_names:
            med_name = med_name.strip().lower()
            if any(drug in med_name for drug in immunosuppressants):
                has_immunosuppressant = True
                break
        
        # If the patient has immunosuppressants and we have years, calculate time to treatment
        if has_immunosuppressant and clean_years:
            earliest_med_year = min(clean_years)
            
            if earliest_med_year >= diagnosis_year:
                years_to_treatment = earliest_med_year - diagnosis_year
                
                results.append({
                    'patient_id': patient_id,
                    'uveitis_diagnosis_year': diagnosis_year,
                    'earliest_medication_year': earliest_med_year,
                    'years_to_treatment': years_to_treatment,
                    'SVI_quartile': patient.get('SVI_quartile', 'Unknown')
                })
    
    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        print(f"\nAnalyzed time-to-treatment for {len(results_df)} patients with uveitis")
        
        # Summary statistics for time to treatment
        print("\nTime from uveitis diagnosis year to first immunosuppressive treatment year:")
        print(f"Number of patients: {len(results_df)}")
        print(f"Mean years: {results_df['years_to_treatment'].mean():.1f}")
        print(f"Median years: {results_df['years_to_treatment'].median():.1f}")
        print(f"Range: {results_df['years_to_treatment'].min()} to {results_df['years_to_treatment'].max()} years")
        
        # Count by years to treatment
        years_counts = results_df['years_to_treatment'].value_counts().sort_index()
        print("\nDistribution of years from diagnosis to treatment:")
        for years, count in years_counts.items():
            print(f"{years} years: {count} patients ({count/len(results_df)*100:.1f}%)")
        
        # Create visualizations
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['years_to_treatment'], bins=range(0, int(results_df['years_to_treatment'].max()) + 2))
        plt.title('Distribution of Years from Uveitis Diagnosis to First Immunosuppressive Treatment')
        plt.xlabel('Years')
        plt.ylabel('Count')
        plt.xticks(range(0, int(results_df['years_to_treatment'].max()) + 2))
        plt.tight_layout()
        plt.savefig(f"{results_dir}/time_to_treatment_histogram.png")
        
        # Analyze time to treatment by SVI quartile
        if 'SVI_quartile' in results_df.columns:
            print("\nAnalyzing time to treatment by SVI quartile...")
            quartile_stats = results_df.groupby('SVI_quartile')['years_to_treatment'].agg(['mean', 'median', 'count'])
            print(quartile_stats)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='SVI_quartile', y='years_to_treatment', data=results_df, order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
            plt.title('Time to Treatment by SVI Quartile')
            plt.xlabel('SVI Quartile')
            plt.ylabel('Years from Diagnosis to Treatment')
            plt.tight_layout()
            plt.savefig(f"{results_dir}/time_to_treatment_by_svi.png")
            
            # Statistical test
            from scipy.stats import kruskal
            # Only run test if we have data in multiple quartiles
            svi_groups = [results_df[results_df['SVI_quartile'] == q]['years_to_treatment'] 
                          for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'] 
                          if q in results_df['SVI_quartile'].unique()]
            
            if len(svi_groups) > 1 and all(len(g) > 0 for g in svi_groups):
                stat, p = kruskal(*svi_groups)
                print(f"\nKruskal-Wallis test for time to treatment by SVI quartile:")
                print(f"H statistic: {stat:.2f}")
                print(f"p-value: {p:.4f}")
                print(f"Statistically significant difference: {'Yes' if p < 0.05 else 'No'}")
        
        # Save results to CSV
        results_df.to_csv(f"{results_dir}/uveitis_treatment_timeline_simplified.csv", index=False)
        
        print(f"\nResults saved to {results_dir}/")
    else:
        print("No valid treatment timeline data found.")
else:
    print(f"Required medication columns not found.")

# Additional analysis: Count patients with uveitis by SVI quartile
print("\nAnalyzing uveitis prevalence by SVI quartile...")

if 'SVI_quartile' in df.columns:
    # Create a uveitis indicator column
    df['has_uveitis'] = df[uveitis_indicators].notna().any(axis=1)
    
    # Calculate uveitis prevalence by SVI quartile
    uveitis_by_quartile = df.groupby('SVI_quartile')['has_uveitis'].sum()
    total_by_quartile = df.groupby('SVI_quartile').size()
    prevalence_by_quartile = (uveitis_by_quartile / total_by_quartile) * 100
    
    print("\nUveitis prevalence by SVI quartile:")
    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        if quartile in prevalence_by_quartile.index:
            print(f"{quartile}: {prevalence_by_quartile[quartile]:.1f}% ({uveitis_by_quartile[quartile]} of {total_by_quartile[quartile]} patients)")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=prevalence_by_quartile.index, y=prevalence_by_quartile.values, order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    plt.title('Uveitis Prevalence by SVI Quartile')
    plt.xlabel('SVI Quartile')
    plt.ylabel('Prevalence (%)')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/uveitis_prevalence_by_svi.png")
    
    # Statistical test (Chi-square)
    from scipy.stats import chi2_contingency
    
    # Create contingency table
    contingency = pd.crosstab(df['SVI_quartile'], df['has_uveitis'])
    
    # Run chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square test for uveitis prevalence by SVI quartile:")
    print(f"Chi2 value: {chi2:.2f}")
    print(f"p-value: {p:.4f}")
    print(f"Statistically significant difference: {'Yes' if p < 0.05 else 'No'}")
    
    # Create a summary
    with open(f"{results_dir}/uveitis_svi_analysis_summary.md", "w") as f:
        f.write("# Uveitis and Social Vulnerability Analysis\n\n")
        
        f.write("## Uveitis Prevalence\n")
        f.write(f"- Total patients in dataset: {len(df)}\n")
        f.write(f"- Patients with uveitis: {len(uveitis_patients)} ({len(uveitis_patients)/len(df)*100:.1f}%)\n\n")
        
        f.write("## Uveitis Prevalence by SVI Quartile\n")
        for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
            if quartile in prevalence_by_quartile.index:
                f.write(f"- {quartile}: {prevalence_by_quartile[quartile]:.1f}% ({uveitis_by_quartile[quartile]} of {total_by_quartile[quartile]} patients)\n")
        
        f.write(f"\nChi-square test: p-value = {p:.4f} ({'Significant' if p < 0.05 else 'Not significant'})\n\n")
        
        if results:
            f.write("## Time to Treatment Analysis\n")
            f.write(f"- Patients with diagnosis year and treatment data: {len(results_df)}\n")
            f.write(f"- Mean years to treatment: {results_df['years_to_treatment'].mean():.1f}\n")
            f.write(f"- Median years to treatment: {results_df['years_to_treatment'].median():.1f}\n")
            f.write(f"- Range: {results_df['years_to_treatment'].min()} to {results_df['years_to_treatment'].max()} years\n\n")
            
            f.write("### Distribution of time to treatment\n")
            for years, count in years_counts.items():
                f.write(f"- {years} years: {count} patients ({count/len(results_df)*100:.1f}%)\n")
            
            if 'SVI_quartile' in results_df.columns:
                f.write("\n### Time to Treatment by SVI Quartile\n")
                quartile_stats = results_df.groupby('SVI_quartile')['years_to_treatment'].agg(['mean', 'median', 'count'])
                for quartile, stats in quartile_stats.iterrows():
                    f.write(f"- {quartile}: Mean = {stats['mean']:.1f} years, Median = {stats['median']:.1f} years, n = {stats['count']}\n")
                
                if 'p' in locals():
                    f.write(f"\nKruskal-Wallis test: p-value = {p:.4f} ({'Significant' if p < 0.05 else 'Not significant'})\n")
    
    print(f"\nSummary report saved to {results_dir}/uveitis_svi_analysis_summary.md")
else:
    print("SVI quartile column not found. Cannot analyze by quartile.")

print("\nAnalysis complete!") 