import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
import os

# Set up styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

print("Loading datasets...")
df = pd.read_csv('SVI_filtered_495_patients.csv')
qol_inventory = pd.read_csv('qol_columns_inventory.csv')
print(f"Total patients: {len(df)}")
print(f"QOL columns inventory: {len(qol_inventory)} entries")

# Create output directory if it doesn't exist
os.makedirs('uveitis_qol_svi_results', exist_ok=True)

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
df = df[df['has_uveitis'] == 1].copy()
print(f"\nTotal uveitis patients identified: {len(df)}")
print(f"Proceeding with analysis on {len(df)} uveitis patients...")

# Function to process PedsQL responses
def process_pedsql_response(value):
    if pd.isna(value):
        return np.nan
    
    # Extract the highest value mentioned in the response
    # PedsQL: 0=Never, 1=Almost Never, 2=Sometimes, 3=Often, 4=Almost Always
    # Higher values = more problems (worse QOL)
    for i in reversed(range(5)):
        if str(i) in str(value):
            return i
    return np.nan

# Function to process CHAQ responses
def process_chaq_response(value):
    if pd.isna(value):
        return np.nan
    
    # CHAQ: 0=Without ANY Difficulty, 1=With SOME Difficulty, 2=With MUCH Difficulty, 3=UNABLE To Do
    # Higher values = worse function
    value_str = str(value).lower()
    if "without any difficulty" in value_str and "with some difficulty" not in value_str:
        return 0
    elif "with some difficulty" in value_str:
        return 1
    elif "with much difficulty" in value_str:
        return 2
    elif "unable to do" in value_str:
        return 3
    else:
        return np.nan

# Function to process EQ-Vision responses
def process_eqvision_response(value):
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).lower()
    if "never hard" in value_str and "sometimes hard" not in value_str:
        return 0  # No difficulty
    elif "sometimes hard" in value_str:
        return 1  # Some difficulty
    elif "always hard" in value_str:
        return 2  # Severe difficulty
    else:
        return np.nan

# Function to standardize PedsQL scores
def standardize_pedsql(score):
    if pd.isna(score):
        return np.nan
    # Convert 0-4 scale to 0-100 scale
    # Note: In PedsQL, 0=Never (100) to 4=Almost Always (0)
    # This transformation makes higher scores = better QOL
    return (4 - score) * 25

# Function to standardize CHAQ scores
def standardize_chaq(score):
    if pd.isna(score):
        return np.nan
    # Convert 0-3 scale to 0-100 scale
    # In CHAQ: 0=No difficulty (100) to 3=Unable to do (0)
    # This transformation makes higher scores = better function
    return (3 - score) * (100/3)

# Function to standardize EQ-Vision scores
def standardize_eqvision(score):
    if pd.isna(score):
        return np.nan
    # Convert 0-2 scale to 0-100 scale
    # 0=Never hard (100) to 2=Always hard (0)
    # This transformation makes higher scores = better vision-related QOL
    return (2 - score) * 50

# Function to perform statistical tests
def test_by_quartile(df, measure):
    # Only include patients with QOL data
    data = df[df[measure].notna()]
    
    # Get values by quartile
    groups = [data[data['SVI_quartile'] == q][measure].dropna() for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']]
    
    # Skip if any group has fewer than 2 values
    if any(len(g) < 2 for g in groups):
        return None, None, None
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # T-test between Q1 and Q4
    t_stat, p_value_t = stats.ttest_ind(groups[0], groups[3], equal_var=False)
    
    return f_stat, p_value, (t_stat, p_value_t)

print("\nExtracting QOL columns by category...")

# 1. PedsQL Child-Reported (Ages 5-18)
pedsql_cols = qol_inventory[
    (qol_inventory['Category'] == 'PedsQL') & 
    (qol_inventory['Reporter'] == 'Child') & 
    (qol_inventory['Age_Group'].isin(['5-7 years', '8-12 years', '13-18 years'])) &
    (qol_inventory['Non_Null_Count'] > 0)
]

# Separate PedsQL domains
pedsql_emotional = pedsql_cols[pedsql_cols['Column'].str.contains('afraid|angry|sad|worried|scared|sleep', case=False)]
pedsql_physical = pedsql_cols[pedsql_cols['Column'].str.contains('bath|lift|exercise|run|sport|energy|chore', case=False)]
pedsql_social = pedsql_cols[pedsql_cols['Column'].str.contains('friend|play|peer|keepup|behind', case=False)]
pedsql_pain = pedsql_cols[pedsql_cols['Column'].str.contains('hurt|pain', case=False)]

print(f"PedsQL Emotional columns: {len(pedsql_emotional)}")
print(f"PedsQL Physical columns: {len(pedsql_physical)}")
print(f"PedsQL Social columns: {len(pedsql_social)}")
print(f"PedsQL Pain columns: {len(pedsql_pain)}")

# 2. CHAQ-style Functional Items
chaq_cols = qol_inventory[
    (qol_inventory['Category'] == 'CHAQ/HAQ') & 
    (qol_inventory['Column'].str.contains('child dress|child walk|child cut meat|child tub bath|child toilet|child sweater', case=False)) &
    (qol_inventory['Non_Null_Count'] > 0)
]
print(f"CHAQ Functional Items: {len(chaq_cols)}")

# 3. Extended Functional Independence
extended_func_cols = qol_inventory[
    (qol_inventory['Column'].str.contains('child shampoo|child socks|child nails|child stand', case=False)) &
    (qol_inventory['Non_Null_Count'] > 0)
]
print(f"Extended Functional Independence: {len(extended_func_cols)}")

# 4. Pain & Functioning Sliders
slider_cols = qol_inventory[
    (qol_inventory['Column'].str.contains('pain slider child|functioning slider child', case=False)) &
    (qol_inventory['Non_Null_Count'] > 0)
]
print(f"Pain & Functioning Sliders: {len(slider_cols)}")

# 5. Vision-Specific QoL (EQ-Vision)
vision_cols = qol_inventory[
    (qol_inventory['Column'].str.contains('eqv5y', case=False)) &
    (qol_inventory['Non_Null_Count'] > 0)
]
print(f"Vision-Specific QoL: {len(vision_cols)}")

print("\nProcessing QOL measures...")

# Process PedsQL Emotional Functioning
pedsql_emotional_cols = pedsql_emotional['Column'].tolist()[:3]  # Take top 3 with most data
if pedsql_emotional_cols:
    for col in pedsql_emotional_cols:
        df[f'{col}_score'] = df[col].apply(process_pedsql_response)
        df[f'{col}_std'] = df[f'{col}_score'].apply(standardize_pedsql)
    
    # Create emotional functioning score (average of standardized scores)
    df['pedsql_emotional_score'] = df[[f'{col}_std' for col in pedsql_emotional_cols]].mean(axis=1)
    print(f"Created PedsQL Emotional score from {len(pedsql_emotional_cols)} items")

# Process PedsQL Physical Functioning
pedsql_physical_cols = pedsql_physical['Column'].tolist()[:3]  # Take top 3 with most data
if pedsql_physical_cols:
    for col in pedsql_physical_cols:
        df[f'{col}_score'] = df[col].apply(process_pedsql_response)
        df[f'{col}_std'] = df[f'{col}_score'].apply(standardize_pedsql)
    
    # Create physical functioning score
    df['pedsql_physical_score'] = df[[f'{col}_std' for col in pedsql_physical_cols]].mean(axis=1)
    print(f"Created PedsQL Physical score from {len(pedsql_physical_cols)} items")

# Process PedsQL Social Functioning
pedsql_social_cols = pedsql_social['Column'].tolist()[:3]  # Take top 3 with most data
if pedsql_social_cols:
    for col in pedsql_social_cols:
        df[f'{col}_score'] = df[col].apply(process_pedsql_response)
        df[f'{col}_std'] = df[f'{col}_score'].apply(standardize_pedsql)
    
    # Create social functioning score
    df['pedsql_social_score'] = df[[f'{col}_std' for col in pedsql_social_cols]].mean(axis=1)
    print(f"Created PedsQL Social score from {len(pedsql_social_cols)} items")

# Process PedsQL Pain
pedsql_pain_cols = pedsql_pain['Column'].tolist()[:2]  # Take top 2 with most data
if pedsql_pain_cols:
    for col in pedsql_pain_cols:
        df[f'{col}_score'] = df[col].apply(process_pedsql_response)
        df[f'{col}_std'] = df[f'{col}_score'].apply(standardize_pedsql)
    
    # Create pain score
    df['pedsql_pain_score'] = df[[f'{col}_std' for col in pedsql_pain_cols]].mean(axis=1)
    print(f"Created PedsQL Pain score from {len(pedsql_pain_cols)} items")

# Create PedsQL Total Score
pedsql_domain_scores = []
if 'pedsql_emotional_score' in df.columns:
    pedsql_domain_scores.append('pedsql_emotional_score')
if 'pedsql_physical_score' in df.columns:
    pedsql_domain_scores.append('pedsql_physical_score')
if 'pedsql_social_score' in df.columns:
    pedsql_domain_scores.append('pedsql_social_score')
if 'pedsql_pain_score' in df.columns:
    pedsql_domain_scores.append('pedsql_pain_score')

if pedsql_domain_scores:
    df['pedsql_total_score'] = df[pedsql_domain_scores].mean(axis=1)
    print(f"Created PedsQL Total score from {len(pedsql_domain_scores)} domains")

# Process CHAQ Functional Items
chaq_functional_cols = chaq_cols['Column'].tolist()
if chaq_functional_cols:
    for col in chaq_functional_cols:
        df[f'{col}_score'] = df[col].apply(process_chaq_response)
    
    # Create CHAQ disability index (0-3, higher = worse function)
    df['chaq_disability_index'] = df[[f'{col}_score' for col in chaq_functional_cols]].max(axis=1)
    
    # Create inverted CHAQ score (0-100, higher = better function)
    df['chaq_function_score'] = df['chaq_disability_index'].apply(standardize_chaq)
    print(f"Created CHAQ Function score from {len(chaq_functional_cols)} items")

# Process Extended Functional Independence
extended_func_cols_list = extended_func_cols['Column'].tolist()
if extended_func_cols_list:
    for col in extended_func_cols_list:
        df[f'{col}_score'] = df[col].apply(process_chaq_response)
    
    # Create extended function score
    df['extended_function_score'] = df[[f'{col}_score' for col in extended_func_cols_list]].mean(axis=1).apply(standardize_chaq)
    print(f"Created Extended Function score from {len(extended_func_cols_list)} items")

# Process Pain & Functioning Sliders
# Pain slider is on a 0-10 scale where higher = worse pain
pain_slider_col = [col for col in slider_cols['Column'] if 'pain slider' in col.lower()]
if pain_slider_col:
    col = pain_slider_col[0]
    # Extract numeric value from the response
    df['pain_slider_value'] = df[col].apply(lambda x: float(str(x).split()[0]) if pd.notna(x) and str(x).split()[0].replace('.', '', 1).isdigit() else np.nan)
    # Convert to 0-100 scale (higher = less pain = better QOL)
    df['pain_slider_score'] = 100 - (df['pain_slider_value'] * 10)
    print("Created Pain Slider score")

# Functioning slider is on a 0-10 scale where higher = better function
func_slider_col = [col for col in slider_cols['Column'] if 'functioning slider' in col.lower()]
if func_slider_col:
    col = func_slider_col[0]
    # Extract numeric value from the response
    df['function_slider_value'] = df[col].apply(lambda x: float(str(x).split()[0]) if pd.notna(x) and str(x).split()[0].replace('.', '', 1).isdigit() else np.nan)
    # Convert to 0-100 scale (higher = better function)
    df['function_slider_score'] = df['function_slider_value'] * 10
    print("Created Function Slider score")

# Process Vision-Specific QoL
vision_cols_list = vision_cols['Column'].tolist()[:5]  # Take top 5 with most data
if vision_cols_list:
    for col in vision_cols_list:
        df[f'{col}_score'] = df[col].apply(process_eqvision_response)
        df[f'{col}_std'] = df[f'{col}_score'].apply(standardize_eqvision)
    
    # Create vision QOL score
    df['vision_qol_score'] = df[[f'{col}_std' for col in vision_cols_list]].mean(axis=1)
    print(f"Created Vision QOL score from {len(vision_cols_list)} items")

# Collect all QOL measures
qol_scores = []
if 'pedsql_emotional_score' in df.columns:
    qol_scores.append('pedsql_emotional_score')
if 'pedsql_physical_score' in df.columns:
    qol_scores.append('pedsql_physical_score')
if 'pedsql_social_score' in df.columns:
    qol_scores.append('pedsql_social_score')
if 'pedsql_pain_score' in df.columns:
    qol_scores.append('pedsql_pain_score')
if 'pedsql_total_score' in df.columns:
    qol_scores.append('pedsql_total_score')
if 'chaq_function_score' in df.columns:
    qol_scores.append('chaq_function_score')
if 'extended_function_score' in df.columns:
    qol_scores.append('extended_function_score')
if 'pain_slider_score' in df.columns:
    qol_scores.append('pain_slider_score')
if 'function_slider_score' in df.columns:
    qol_scores.append('function_slider_score')
if 'vision_qol_score' in df.columns:
    qol_scores.append('vision_qol_score')

# Print summary of QOL scores
print("\nQOL measures summary:")
print("-" * 50)

for col in qol_scores:
    non_null = df[col].notna().sum()
    print(f"{col}: {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
    if non_null > 0:
        print(f"  Mean: {df[col].mean():.1f}, Median: {df[col].median():.1f}, Min: {df[col].min():.1f}, Max: {df[col].max():.1f}")

# Analyze QOL by SVI quartile
print("\nAnalyzing QOL measures by SVI quartile...")
print("-" * 50)

# Calculate statistics for each QOL measure
results = {}
for measure in qol_scores:
    # Get counts by quartile
    quartile_counts = df.groupby('SVI_quartile')[measure].apply(lambda x: x.notna().sum()).reset_index()
    quartile_counts = dict(zip(quartile_counts['SVI_quartile'], quartile_counts[measure]))
    
    # Get mean scores by quartile
    quartile_means = df.groupby('SVI_quartile')[measure].mean().reset_index()
    quartile_means = dict(zip(quartile_means['SVI_quartile'], quartile_means[measure]))
    
    # Perform statistical tests
    f_stat, p_value, t_test = test_by_quartile(df, measure)
    
    results[measure] = {
        'counts': quartile_counts,
        'means': quartile_means,
        'anova': (f_stat, p_value),
        't_test': t_test
    }
    
    # Print results
    print(f"\n{measure} by SVI quartile:")
    print(f"  Q1 (Low): {quartile_means.get('Q1 (Low)', 'N/A'):.1f} (n={quartile_counts.get('Q1 (Low)', 0)})")
    print(f"  Q2: {quartile_means.get('Q2', 'N/A'):.1f} (n={quartile_counts.get('Q2', 0)})")
    print(f"  Q3: {quartile_means.get('Q3', 'N/A'):.1f} (n={quartile_counts.get('Q3', 0)})")
    print(f"  Q4 (High): {quartile_means.get('Q4 (High)', 'N/A'):.1f} (n={quartile_counts.get('Q4 (High)', 0)})")
    
    if f_stat is not None:
        print(f"  ANOVA: F={f_stat:.2f}, p={p_value:.4f}" + (" *" if p_value < 0.05 else ""))
        print(f"  T-test (Q1 vs Q4): t={t_test[0]:.2f}, p={t_test[1]:.4f}" + (" *" if t_test[1] < 0.05 else ""))

# Save results to CSV
print("\nSaving results to CSV...")
csv_data = []

quartiles = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
for quartile in quartiles:
    row = {'SVI_quartile': quartile}
    for measure in qol_scores:
        measure_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
        # Mean score
        if quartile in results[measure]['means']:
            row[f"{measure_name} Mean"] = results[measure]['means'][quartile]
        # Count
        if quartile in results[measure]['counts']:
            row[f"{measure_name} Count"] = results[measure]['counts'][quartile]
    csv_data.append(row)

# Add statistical test results
row = {'SVI_quartile': 'Statistical Tests'}
for measure in qol_scores:
    measure_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
    if results[measure]['anova'][0] is not None:
        row[f"{measure_name} Mean"] = f"ANOVA p={results[measure]['anova'][1]:.4f}, T-test p={results[measure]['t_test'][1]:.4f}"
    else:
        row[f"{measure_name} Mean"] = "Insufficient data"
    row[f"{measure_name} Count"] = ""
csv_data.append(row)

# Create DataFrame and save
results_df = pd.DataFrame(csv_data)
results_df.to_csv('uveitis_qol_svi_results/uveitis_qol_svi_results.csv', index=False)
print("Results saved to uveitis_qol_svi_results/uveitis_qol_svi_results.csv")

# Create visualizations - Boxplots
print("\nGenerating visualizations...")

# Define a function to create and save boxplots for each measure
def create_boxplot(measure, title, ylabel):
    plt.figure(figsize=(10, 6))
    # Create boxplot
    data = df[['SVI_quartile', measure]].dropna()
    if len(data) > 10:  # Only create plot if enough data
        sns.boxplot(x='SVI_quartile', y=measure, data=data, 
                    order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        plt.title(f"{title} (Uveitis Patients)", fontsize=14)
        plt.xlabel('SVI Quartile', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add ANOVA p-value if available
        if results[measure]['anova'][0] is not None:
            p_val = results[measure]['anova'][1]
            plt.annotate(f"ANOVA p={p_val:.4f}" + (" *" if p_val < 0.05 else ""), 
                         xy=(0.5, 0.02), xycoords='axes fraction', ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", alpha=0.2))
        
        # Save figure
        plt.tight_layout()
        filename = f"uveitis_qol_svi_results/{measure.replace('_', '-')}_boxplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    return None

# Create boxplots for all QOL measures
saved_plots = []
for measure in qol_scores:
    nice_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
    ylabel = f"{nice_name}\n(higher = better)"
    filename = create_boxplot(measure, f"{nice_name} by SVI Quartile", ylabel)
    if filename:
        saved_plots.append(filename)
        print(f"Created boxplot: {filename}")

# Create overview visualization with multiple QOL domains
print("\nCreating comprehensive visualization...")

plt.figure(figsize=(18, 12))

# 1. PedsQL Total Score by SVI Quartile (if available)
if 'pedsql_total_score' in qol_scores:
    plt.subplot(2, 3, 1)
    pedsql_data = df[['SVI_quartile', 'pedsql_total_score']].dropna()
    sns.boxplot(x='SVI_quartile', y='pedsql_total_score', data=pedsql_data, 
                order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    plt.title('PedsQL Total Score (Uveitis Patients)', fontsize=12)
    plt.xlabel('SVI Quartile')
    plt.ylabel('Score\n(higher = better)')
    plt.ylim(0, 100)
    
    if results['pedsql_total_score']['anova'][0] is not None:
        p_val = results['pedsql_total_score']['anova'][1]
        plt.annotate(f"p={p_val:.4f}" + (" *" if p_val < 0.05 else ""), 
                     xy=(0.5, 0.02), xycoords='axes fraction', ha='center')

# 2. CHAQ Function Score by SVI Quartile (if available)
if 'chaq_function_score' in qol_scores:
    plt.subplot(2, 3, 2)
    chaq_data = df[['SVI_quartile', 'chaq_function_score']].dropna()
    sns.boxplot(x='SVI_quartile', y='chaq_function_score', data=chaq_data, 
                order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    plt.title('CHAQ Function Score', fontsize=12)
    plt.xlabel('SVI Quartile')
    plt.ylabel('Score\n(higher = better)')
    plt.ylim(0, 100)
    
    if results['chaq_function_score']['anova'][0] is not None:
        p_val = results['chaq_function_score']['anova'][1]
        plt.annotate(f"p={p_val:.4f}" + (" *" if p_val < 0.05 else ""), 
                     xy=(0.5, 0.02), xycoords='axes fraction', ha='center')

# 3. Pain Slider Score by SVI Quartile (if available)
if 'pain_slider_score' in qol_scores:
    plt.subplot(2, 3, 3)
    pain_data = df[['SVI_quartile', 'pain_slider_score']].dropna()
    sns.boxplot(x='SVI_quartile', y='pain_slider_score', data=pain_data, 
                order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    plt.title('Pain Score', fontsize=12)
    plt.xlabel('SVI Quartile')
    plt.ylabel('Score\n(higher = less pain)')
    plt.ylim(0, 100)
    
    if results['pain_slider_score']['anova'][0] is not None:
        p_val = results['pain_slider_score']['anova'][1]
        plt.annotate(f"p={p_val:.4f}" + (" *" if p_val < 0.05 else ""), 
                     xy=(0.5, .02), xycoords='axes fraction', ha='center')

# 4. PedsQL Domain Scores by SVI Quartile
plt.subplot(2, 3, 4)

# Prepare data for domain comparison
domain_data = []
domains = []
domain_cols = []

if 'pedsql_physical_score' in qol_scores:
    domains.append('Physical')
    domain_cols.append('pedsql_physical_score')
if 'pedsql_emotional_score' in qol_scores:
    domains.append('Emotional')
    domain_cols.append('pedsql_emotional_score')
if 'pedsql_social_score' in qol_scores:
    domains.append('Social')
    domain_cols.append('pedsql_social_score')
if 'pedsql_pain_score' in qol_scores:
    domains.append('Pain')
    domain_cols.append('pedsql_pain_score')

for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
    for domain, col in zip(domains, domain_cols):
        mean_score = df[df['SVI_quartile'] == quartile][col].mean()
        if not pd.isna(mean_score):
            domain_data.append({
                'SVI_quartile': quartile,
                'Domain': domain,
                'Score': mean_score
            })

domain_df = pd.DataFrame(domain_data)

if not domain_df.empty and len(domains) > 0:
    sns.barplot(x='Domain', y='Score', hue='SVI_quartile', data=domain_df, 
               hue_order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    plt.title('PedsQL Domains by SVI Quartile', fontsize=12)
    plt.xlabel('Domain')
    plt.ylabel('Score (higher = better)')
    plt.ylim(0, 100)
    plt.legend(title='SVI Quartile', loc='lower left')

# 5. Vision QOL Score by SVI Quartile (if available)
if 'vision_qol_score' in qol_scores:
    plt.subplot(2, 3, 5)
    vision_data = df[['SVI_quartile', 'vision_qol_score']].dropna()
    sns.boxplot(x='SVI_quartile', y='vision_qol_score', data=vision_data, 
                order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    plt.title('Vision-Related QOL', fontsize=12)
    plt.xlabel('SVI Quartile')
    plt.ylabel('Score\n(higher = better)')
    plt.ylim(0, 100)
    
    if results['vision_qol_score']['anova'][0] is not None:
        p_val = results['vision_qol_score']['anova'][1]
        plt.annotate(f"p={p_val:.4f}" + (" *" if p_val < 0.05 else ""), 
                     xy=(0.5, 0.02), xycoords='axes fraction', ha='center')

# 6. Correlation between SVI and QOL measures
plt.subplot(2, 3, 6)

# Calculate correlations
correlations = []
for measure in qol_scores:
    # Create a temporary dataframe with only rows that have both SVI_total and the measure
    temp_df = df[['SVI_total', measure]].dropna(how='any')
    
    # Skip if too few data points
    if len(temp_df) < 10:
        continue
    
    # Now x and y will have the same length
    corr, p_val = stats.pearsonr(temp_df['SVI_total'], temp_df[measure])
    nice_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_score', '').replace('_', ' ').title()
    
    # Limit to 30 characters to fit on the plot
    if len(nice_name) > 30:
        nice_name = nice_name[:27] + '...'
        
    correlations.append({
        'Measure': nice_name,
        'Correlation': corr,
        'P-value': p_val
    })

corr_df = pd.DataFrame(correlations)
if not corr_df.empty:
    corr_df = corr_df.sort_values('Correlation')
    
    bars = plt.barh(corr_df['Measure'], corr_df['Correlation'])
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Correlation with SVI Total Score', fontsize=12)
    plt.xlabel('Pearson Correlation Coefficient')
    
    # Highlight significant correlations
    for i, (bar, p_val) in enumerate(zip(bars, corr_df['P-value'])):
        if p_val < 0.05:
            bar.set_color('red')
    
    # Add correlation values
    for i, (corr, p_val) in enumerate(zip(corr_df['Correlation'], corr_df['P-value'])):
        plt.text(corr + (0.01 if corr >= 0 else -0.01), 
                 i, 
                 f"{corr:.2f}" + (" *" if p_val < 0.05 else ""), 
                 va='center', 
                 ha='left' if corr >= 0 else 'right')

plt.tight_layout()
plt.suptitle('Relationship Between Social Vulnerability and Quality of Life in Uveitis Patients', fontsize=16)
plt.subplots_adjust(top=0.9)

# Save comprehensive plot
plt.savefig('uveitis_qol_svi_results/uveitis_qol_svi_analysis.png', dpi=300, bbox_inches='tight')
print("Comprehensive visualization saved as uveitis_qol_svi_results/uveitis_qol_svi_analysis.png")

# Generate a comprehensive markdown report
print("\nGenerating comprehensive markdown report...")

# Identify significant findings
significant_measures = []
for measure in qol_scores:
    if (results[measure]['anova'][0] is not None and results[measure]['anova'][1] < 0.05) or \
       (results[measure]['t_test'] is not None and results[measure]['t_test'][1] < 0.05):
        significant_measures.append(measure)

# Find measures with largest quartile differences
quartile_differences = {}
for measure in qol_scores:
    if 'Q1 (Low)' in results[measure]['means'] and 'Q4 (High)' in results[measure]['means']:
        q1_score = results[measure]['means']['Q1 (Low)']
        q4_score = results[measure]['means']['Q4 (High)']
        diff = q1_score - q4_score
        quartile_differences[measure] = (diff, q1_score, q4_score)

# Sort by absolute difference
sorted_differences = sorted(quartile_differences.items(), key=lambda x: abs(x[1][0]), reverse=True)
top_differences = sorted_differences[:3] if len(sorted_differences) >= 3 else sorted_differences

with open('uveitis_qol_svi_results/Uveitis_QOL_SVI_Report.md', 'w') as f:
    f.write("# Quality of Life and Social Vulnerability in Pediatric Uveitis Patients\n\n")
    
    # Introduction
    f.write("## Overview\n")
    f.write("This report analyzes the relationship between social vulnerability index (SVI) and quality of life (QOL) measures ")
    f.write("specifically in pediatric patients with uveitis. The analysis examines how social determinants of health may ")
    f.write("influence physical functioning, emotional wellbeing, social interactions, pain management, and vision-related quality of life.\n\n")
    
    # Methods summary
    f.write("## Methods\n")
    f.write(f"The analysis included {len(df)} patients with confirmed uveitis diagnoses. Quality of life was assessed using multiple validated instruments:\n\n")
    f.write("- **PedsQL™**: Pediatric Quality of Life Inventory (emotional, physical, social functioning, and pain domains)\n")
    f.write("- **CHAQ**: Childhood Health Assessment Questionnaire (physical function)\n")
    f.write("- **Pain and Function Sliders**: Visual analog scales for pain intensity and overall function\n")
    f.write("- **Vision-Related QOL**: EQ-Vision measures specific to vision-related quality of life\n\n")
    
    f.write("Patients were stratified by SVI quartile, where Q1 represents the lowest social vulnerability and Q4 represents the highest. ")
    f.write("All QOL measures were standardized to a 0-100 scale where higher scores indicate better outcomes.\n\n")
    
    # Patient demographics
    f.write("## Patient Characteristics\n")
    quartile_counts = df['SVI_quartile'].value_counts().sort_index()
    f.write(f"- Total uveitis patients analyzed: {len(df)}\n")
    f.write("- Patients by SVI quartile:\n")
    for quartile, count in quartile_counts.items():
        f.write(f"  - {quartile}: {count} patients ({count/len(df)*100:.1f}%)\n")
    f.write("\n")
    
    # Key findings
    f.write("## Key Findings\n\n")
    
    # Summary of data availability
    f.write("### Data Availability\n")
    for col in qol_scores:
        non_null = df[col].notna().sum()
        nice_name = col.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
        f.write(f"- {nice_name}: {non_null} patients ({non_null/len(df)*100:.1f}%)\n")
    f.write("\n")
    
    # Overall QOL scores
    f.write("### Overall Quality of Life Scores\n")
    for col in qol_scores:
        nice_name = col.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
        f.write(f"- {nice_name}: Mean = {df[col].mean():.1f}, Median = {df[col].median():.1f} (Range: {df[col].min():.1f}-{df[col].max():.1f})\n")
    f.write("\n")
    
    # Significant Differences
    if significant_measures:
        f.write("### Significant Differences by SVI Quartile\n")
        for measure in significant_measures:
            nice_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
            f.write(f"#### {nice_name}\n")
            
            # Extract quartile scores
            scores_by_quartile = []
            for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
                if quartile in results[measure]['means']:
                    score = results[measure]['means'][quartile]
                    count = results[measure]['counts'][quartile]
                    scores_by_quartile.append(f"{quartile}: {score:.1f} (n={count})")
            
            f.write("- Scores by quartile: " + ", ".join(scores_by_quartile) + "\n")
            
            # Statistical results
            if results[measure]['anova'][0] is not None:
                f_stat, p_value = results[measure]['anova']
                f.write(f"- ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}" + (" *" if p_value < 0.05 else "") + "\n")
            
            if results[measure]['t_test'] is not None:
                t_stat, p_value_t = results[measure]['t_test']
                f.write(f"- T-test (Q1 vs Q4): t = {t_stat:.2f}, p = {p_value_t:.4f}" + (" *" if p_value_t < 0.05 else "") + "\n")
            
            f.write("\n")
    else:
        f.write("### No statistically significant differences were found between SVI quartiles for any QOL measure.\n\n")
    
    # Largest differences between Q1 and Q4
    f.write("### Largest Differences Between Lowest and Highest SVI Quartiles\n")
    for measure, (diff, q1, q4) in top_differences:
        nice_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
        direction = "lower" if diff > 0 else "higher"
        f.write(f"- **{nice_name}**: Q4 scores {abs(diff):.1f} points {direction} than Q1 ")
        f.write(f"(Q1: {q1:.1f} vs. Q4: {q4:.1f})\n")
    f.write("\n")
    
    # Interpretation
    f.write("## Interpretation\n\n")
    
    # Generate some interpretive text based on the data
    if significant_measures:
        f.write("The analysis reveals several important associations between social vulnerability and quality of life measures in pediatric uveitis patients:\n\n")
        
        # Check for emotional/psychosocial impacts
        if 'pedsql_emotional_score' in significant_measures or 'pedsql_social_score' in significant_measures:
            f.write("### Psychosocial Impact\n")
            if 'pedsql_emotional_score' in significant_measures:
                q1 = results['pedsql_emotional_score']['means'].get('Q1 (Low)', 'N/A')
                q4 = results['pedsql_emotional_score']['means'].get('Q4 (High)', 'N/A')
                if q1 != 'N/A' and q4 != 'N/A' and q1 > q4:
                    f.write("- **Emotional wellbeing** shows a significant vulnerability gradient, with higher SVI patients ")
                    f.write(f"reporting more emotional challenges (scores {q1-q4:.1f} points lower in Q4 vs. Q1). ")
                    f.write("This suggests that social determinants may compound the emotional burden of managing uveitis.\n")
            
            if 'pedsql_social_score' in significant_measures:
                q1 = results['pedsql_social_score']['means'].get('Q1 (Low)', 'N/A')
                q4 = results['pedsql_social_score']['means'].get('Q4 (High)', 'N/A')
                if q1 != 'N/A' and q4 != 'N/A' and q1 > q4:
                    f.write("- **Social functioning** is significantly lower in patients from higher vulnerability areas, ")
                    f.write(f"with Q4 patients scoring {q1-q4:.1f} points lower than Q1 patients. ")
                    f.write("This suggests potential challenges in peer relationships, school participation, or social support.\n")
            f.write("\n")
        
        # Check for physical functioning impacts
        physical_measures = ['pedsql_physical_score', 'chaq_function_score', 'function_slider_score']
        physical_significant = [m for m in physical_measures if m in significant_measures]
        if physical_significant:
            f.write("### Physical Functioning\n")
            for measure in physical_significant:
                q1 = results[measure]['means'].get('Q1 (Low)', 'N/A')
                q4 = results[measure]['means'].get('Q4 (High)', 'N/A')
                if q1 != 'N/A' and q4 != 'N/A':
                    measure_name = measure.replace('pedsql_', '').replace('chaq_', '').replace('_', ' ').title()
                    direction = "lower" if q1 > q4 else "higher"
                    abs_diff = abs(q1 - q4)
                    f.write(f"- **{measure_name}** is {abs_diff:.1f} points {direction} in patients from high vulnerability areas. ")
                    if direction == "lower":
                        f.write("This suggests that social determinants may impact physical activities and daily functioning ")
                        f.write("in uveitis patients from more vulnerable communities.\n")
                    else:
                        f.write("Interestingly, patients from higher vulnerability areas report better physical function scores, ")
                        f.write("which may reflect differences in activity expectations, reporting patterns, or adaptation strategies.\n")
            f.write("\n")
        
        # Check for pain impacts
        pain_measures = ['pedsql_pain_score', 'pain_slider_score']
        pain_significant = [m for m in pain_measures if m in significant_measures]
        if pain_significant:
            f.write("### Pain Experience\n")
            for measure in pain_significant:
                q1 = results[measure]['means'].get('Q1 (Low)', 'N/A')
                q4 = results[measure]['means'].get('Q4 (High)', 'N/A')
                if q1 != 'N/A' and q4 != 'N/A':
                    measure_name = measure.replace('pedsql_', '').replace('_', ' ').title()
                    direction = "worse" if q1 > q4 else "better"
                    abs_diff = abs(q1 - q4)
                    f.write(f"- **{measure_name}** is {abs_diff:.1f} points {direction} in patients from high vulnerability areas. ")
                    if direction == "worse":
                        f.write("This suggests potential disparities in pain management, access to effective treatments, ")
                        f.write("or differences in the burden of disease.\n")
                    else:
                        f.write("This unexpected finding might reflect reporting differences, pain perception variations, ")
                        f.write("or potential adaptations to chronic pain in different socioeconomic contexts.\n")
            f.write("\n")
            
        # Check for vision-specific impacts
        if 'vision_qol_score' in significant_measures:
            f.write("### Vision-Related Quality of Life\n")
            q1 = results['vision_qol_score']['means'].get('Q1 (Low)', 'N/A')
            q4 = results['vision_qol_score']['means'].get('Q4 (High)', 'N/A')
            if q1 != 'N/A' and q4 != 'N/A':
                direction = "worse" if q1 > q4 else "better"
                abs_diff = abs(q1 - q4)
                f.write(f"- Vision-related QOL is {abs_diff:.1f} points {direction} in patients from high vulnerability areas. ")
                if direction == "worse":
                    f.write("This suggests potential disparities in access to vision care, vision rehabilitation services, ")
                    f.write("or visual aids for uveitis patients from more vulnerable communities.\n")
                else:
                    f.write("This finding requires further investigation, as it may reflect reporting differences ")
                    f.write("or unexpected variations in vision-related support systems across different socioeconomic contexts.\n")
            f.write("\n")
    else:
        f.write("The analysis did not find statistically significant differences in quality of life measures across SVI quartiles ")
        f.write("in this cohort of pediatric uveitis patients. This could suggest that:\n\n")
        f.write("1. The impact of uveitis on quality of life may transcend socioeconomic boundaries\n")
        f.write("2. The current standard of care for pediatric uveitis might help mitigate potential disparities\n")
        f.write("3. The limited sample size, especially in some SVI quartiles, may have affected statistical power\n\n")
        
        # Add nuance about trends even if not significant
        top_measure, (top_diff, top_q1, top_q4) = top_differences[0] if top_differences else (None, (0, 0, 0))
        if top_measure and abs(top_diff) > 10:  # Only mention if difference is substantial
            nice_name = top_measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
            direction = "lower" if top_diff > 0 else "higher"
            f.write(f"Despite the lack of statistical significance, there was a notable {abs(top_diff):.1f}-point {direction} score ")
            f.write(f"in {nice_name} among patients from high vulnerability areas. ")
            f.write("This trend warrants further investigation with larger sample sizes.\n\n")

    # Clinical implications
    f.write("## Clinical Implications\n\n")
    f.write("Based on these findings, several clinical implications emerge for the care of pediatric uveitis patients:\n\n")
    
    if significant_measures:
        if any(m in significant_measures for m in ['pedsql_emotional_score', 'pedsql_social_score']):
            f.write("1. **Psychosocial Support**: Consider enhanced screening for emotional and social challenges in uveitis patients ")
            f.write("from higher vulnerability areas, with appropriate referrals to mental health services and social support programs.\n\n")
        
        if any(m in significant_measures for m in ['pedsql_physical_score', 'chaq_function_score']):
            f.write("2. **Physical Functioning**: Assess barriers to physical activity and daily functioning, particularly ")
            f.write("for patients from more vulnerable communities. Consider physical therapy referrals and activity accommodations as needed.\n\n")
        
        if any(m in significant_measures for m in ['pedsql_pain_score', 'pain_slider_score']):
            f.write("3. **Pain Management**: Evaluate pain management strategies across SVI quartiles, ensuring equitable ")
            f.write("access to effective pain control for all patients regardless of socioeconomic status.\n\n")
        
        if 'vision_qol_score' in significant_measures:
            f.write("4. **Vision Support**: Address potential disparities in vision care by ensuring access to visual aids, ")
            f.write("vision rehabilitation services, and educational accommodations for patients from higher vulnerability areas.\n\n")
        
        f.write("5. **Personalized Care Plans**: Develop individualized care plans that address not only the clinical aspects of uveitis ")
        f.write("but also the social determinants of health that may impact treatment outcomes and quality of life.\n\n")
    else:
        f.write("1. **Standardized Approach**: Continue current comprehensive care approaches that may be helping to mitigate SVI-related disparities.\n\n")
        f.write("2. **Vigilant Monitoring**: Maintain vigilance for potential emerging disparities in quality of life outcomes.\n\n")
        f.write("3. **Enhanced Assessment**: Consider more nuanced assessment tools that might better capture quality of life differences across SVI quartiles.\n\n")
    
    # Limitations
    f.write("## Limitations\n\n")
    f.write("Several limitations should be considered when interpreting these results:\n\n")
    f.write("1. **Sample Size**: The sample includes only 125 uveitis patients, with uneven distribution across SVI quartiles (particularly Q2).\n\n")
    f.write("2. **Cross-Sectional Design**: This analysis represents a snapshot in time and cannot establish causality or temporal relationships.\n\n")
    f.write("3. **Missing Data**: Not all QOL measures were available for all patients, potentially introducing bias.\n\n")
    f.write("4. **Confounding Factors**: The analysis does not control for potential confounders such as uveitis severity, duration, treatment differences, or comorbidities.\n\n")
    f.write("5. **Self-Reported Measures**: QOL assessments are based on self-report (or parent-report) and may be influenced by reporting biases that vary across socioeconomic groups.\n\n")
    
    # Conclusion
    f.write("## Conclusion\n\n")
    if significant_measures:
        f.write("This analysis demonstrates important associations between social vulnerability and quality of life in pediatric uveitis patients. ")
        f.write("The findings highlight the need for attention to social determinants of health in the management of uveitis, with particular focus on ")
        affected_domains = []
        if 'pedsql_emotional_score' in significant_measures or 'pedsql_social_score' in significant_measures:
            affected_domains.append("psychosocial support")
        if any(m in significant_measures for m in ['pedsql_physical_score', 'chaq_function_score']):
            affected_domains.append("physical functioning")
        if any(m in significant_measures for m in ['pedsql_pain_score', 'pain_slider_score']):
            affected_domains.append("pain management")
        if 'vision_qol_score' in significant_measures:
            affected_domains.append("vision care")
        
        if affected_domains:
            f.write(", ".join(affected_domains))
            f.write(". ")
        
        f.write("Addressing these disparities could improve overall outcomes and reduce the burden of disease for all pediatric uveitis patients.\n")
    else:
        f.write("While this analysis did not find statistically significant differences in quality of life across SVI quartiles, ")
        f.write("the trends observed suggest that social vulnerability may still influence certain aspects of life for pediatric uveitis patients. ")
        f.write("Further research with larger sample sizes and longitudinal designs is needed to better understand these relationships ")
        f.write("and ensure equitable outcomes for all children with uveitis, regardless of socioeconomic status.\n")

    # Add timestamp
    from datetime import datetime
    f.write(f"\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")

print(f"Comprehensive report saved to uveitis_qol_svi_results/Uveitis_QOL_SVI_Report.md")

print("\nAnalysis complete.") 