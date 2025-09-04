import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os
from matplotlib.ticker import MaxNLocator

# Set up styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")
plt.rcParams['figure.figsize'] = (12, 8)

print("Loading datasets...")
df = pd.read_csv('SVI_filtered_495_patients_with_qol_scores.csv')
qol_inventory = pd.read_csv('/Users/rajlq7/Desktop/SVI/qol_svi_results/qol_columns_inventory.csv')
print(f"Total patients: {len(df)}")
print(f"QOL columns inventory: {len(qol_inventory)} entries")

# Create output directory if it doesn't exist
os.makedirs('advanced_analyses', exist_ok=True)

# Identify relevant columns that we'll need for analysis
# Look for columns related to disease severity, visual acuity, treatments, etc.
print("\nIdentifying relevant clinical columns...")

# Function to find columns matching patterns
def find_columns(pattern_list):
    matching_cols = []
    for pattern in pattern_list:
        matches = [col for col in df.columns if pattern.lower() in col.lower()]
        matching_cols.extend(matches)
    return list(set(matching_cols))  # Remove duplicates

# Find disease severity indicators
severity_cols = find_columns(['severity', 'complication', 'surgery', 'hospitalization', 'flare'])
print(f"Potential disease severity columns: {len(severity_cols)}")
if severity_cols:
    print("Examples:", severity_cols[:3])

# Find visual acuity indicators
visual_acuity_cols = find_columns(['visual acuity', 'vision test', 'eyesight', 'eye exam'])
print(f"Potential visual acuity columns: {len(visual_acuity_cols)}")
if visual_acuity_cols:
    print("Examples:", visual_acuity_cols[:3])

# Find treatment indicators
treatment_cols = find_columns(['treatment', 'medication', 'drug', 'therapy', 'methotrexate', 'steroid', 'biologic', 'eyedrop'])
print(f"Potential treatment columns: {len(treatment_cols)}")
if treatment_cols:
    print("Examples:", treatment_cols[:3])

# Find SVI subdomain columns
svi_subdomain_cols = find_columns(['SVI_theme'])
print(f"SVI subdomain columns: {len(svi_subdomain_cols)}")
if svi_subdomain_cols:
    print("Examples:", svi_subdomain_cols[:3])

# Verify we have QOL measures and age groups
print("\nChecking QOL measures and age groups...")

# Load QOL scores from existing analyses or recalculate them
# First, check if the scores already exist from the previous analysis
qol_scores = []
for score_name in ['pedsql_emotional_score', 'pedsql_physical_score', 
                  'pedsql_social_score', 'pedsql_pain_score', 'pedsql_total_score',
                  'chaq_function_score', 'vision_qol_score', 'pain_slider_score',
                  'function_slider_score']:
    if score_name in df.columns:
        qol_scores.append(score_name)
        print(f"Found existing QOL score: {score_name}")

# Check for age group indicators
age_cols = find_columns(['age', 'year'])
print(f"Potential age columns: {len(age_cols)}")
if age_cols:
    print("Examples:", age_cols[:3])

# Create age groups if needed
if 'age' in df.columns:
    # Create age groups based on years
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 7, 12, 18, 100], 
                            labels=['5-7 years', '8-12 years', '13-18 years', 'Adult'])
    print("Created age groups from age column")
elif 'age_group' not in df.columns:
    # Try to infer age group from PedsQL columns
    age_indicators = []
    for col in df.columns:
        if 'pedsql' in col.lower():
            if '5 7' in col.lower() or '5-7' in col.lower():
                age_indicators.append(('5-7 years', col))
            elif '8 12' in col.lower() or '8-12' in col.lower():
                age_indicators.append(('8-12 years', col))
            elif '13 18' in col.lower() or '13-18' in col.lower():
                age_indicators.append(('13-18 years', col))
    
    if age_indicators:
        # Assign age groups based on which age-specific PedsQL has non-null values
        for age_group, col in age_indicators:
            mask = df[col].notna()
            # Only assign if not already assigned
            if 'age_group' not in df.columns:
                df['age_group'] = np.nan
            # Only fill where age_group is null and this column has a value
            df.loc[mask & df['age_group'].isna(), 'age_group'] = age_group
        
        print(f"Inferred age groups from PedsQL columns for {df['age_group'].notna().sum()} patients")

# ================ ANALYSIS 1: QOL by AGE GROUP and SVI ================
print("\n\n========== ANALYSIS 1: QOL by AGE GROUP and SVI ==========")
if 'age_group' in df.columns and qol_scores:
    print(f"Age group distribution:\n{df['age_group'].value_counts()}")
    
    # Create function to analyze QOL by age group and SVI quartile
    def analyze_qol_by_age_svi(qol_measure):
        # Create result dataframe for this measure
        result_data = []
        
        for age_group in df['age_group'].dropna().unique():
            age_df = df[df['age_group'] == age_group]
            print(f"\nAnalyzing {qol_measure} for age group {age_group} (n={len(age_df)})")
            
            # Get mean scores by SVI quartile for this age group
            quartile_means = age_df.groupby('SVI_quartile')[qol_measure].agg(['mean', 'count']).reset_index()
            
            for _, row in quartile_means.iterrows():
                result_data.append({
                    'Age Group': age_group,
                    'SVI Quartile': row['SVI_quartile'],
                    'QOL Measure': qol_measure,
                    'Mean Score': row['mean'],
                    'Count': row['count']
                })
            
            # Perform ANOVA if enough data
            quartiles_with_data = quartile_means[quartile_means['count'] >= 5]['SVI_quartile'].tolist()
            if len(quartiles_with_data) >= 2:
                groups = [age_df[age_df['SVI_quartile'] == q][qol_measure].dropna() for q in quartiles_with_data]
                if all(len(g) >= 5 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    print(f"  ANOVA for {qol_measure} across SVI quartiles: F={f_stat:.2f}, p={p_value:.4f}" +
                         (" *" if p_value < 0.05 else ""))
                    
                    # Add to the result data
                    result_data.append({
                        'Age Group': age_group,
                        'SVI Quartile': 'ANOVA',
                        'QOL Measure': qol_measure,
                        'Mean Score': f_stat,
                        'Count': p_value
                    })
        
        return pd.DataFrame(result_data)
    
    # Analyze each QOL measure by age and SVI
    age_results = pd.DataFrame()
    for measure in qol_scores:
        measure_results = analyze_qol_by_age_svi(measure)
        age_results = pd.concat([age_results, measure_results])
    
    # Save results to CSV
    age_results.to_csv('advanced_analyses/qol_by_age_svi.csv', index=False)
    print(f"Saved age group analysis to advanced_analyses/qol_by_age_svi.csv")
    
    # Create visualizations for age group differences
    for measure in qol_scores:
        plt.figure(figsize=(12, 7))
        
        # Filter to only include rows with mean scores (not ANOVA results)
        plot_data = age_results[(age_results['QOL Measure'] == measure) & 
                               (age_results['SVI Quartile'] != 'ANOVA')]
        
        if len(plot_data) > 0:
            # Convert to numeric for plotting
            plot_data['Mean Score'] = pd.to_numeric(plot_data['Mean Score'], errors='coerce')
            
            # Create the plot
            ax = sns.barplot(x='SVI Quartile', y='Mean Score', hue='Age Group', data=plot_data,
                           order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
            
            # Customize plot
            measure_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
            plt.title(f'{measure_name} by Age Group and SVI Quartile', fontsize=14)
            plt.xlabel('SVI Quartile', fontsize=12)
            plt.ylabel('Mean Score (higher = better)', fontsize=12)
            plt.ylim(0, 105)  # Slightly above 100 to show full bars
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=9)
            
            # Add sample size annotations
            for i, quartile in enumerate(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']):
                for j, age in enumerate(plot_data['Age Group'].unique()):
                    count = plot_data[(plot_data['SVI Quartile'] == quartile) & 
                                    (plot_data['Age Group'] == age)]['Count'].values
                    if len(count) > 0 and not np.isnan(count[0]):
                        x_pos = i + (j - len(plot_data['Age Group'].unique())/2 + 0.5) * 0.8/len(plot_data['Age Group'].unique())
                        plt.annotate(f'n={int(count[0])}', xy=(x_pos, 5), ha='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'advanced_analyses/{measure}_by_age_svi.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created {measure} by age group visualization")
else:
    print("Cannot perform age group analysis - missing age groups or QOL scores")

# ================ ANALYSIS 2: DISEASE BURDEN & FUNCTIONAL IMPACT ================
print("\n\n========== ANALYSIS 2: DISEASE BURDEN & FUNCTIONAL IMPACT ==========")

# Check for uveitis diagnosis 
if 'diagnosis of uveitis' in df.columns:
    print(f"Found uveitis diagnosis column. Distribution:\n{df['diagnosis of uveitis'].value_counts()}")
    
    # Analyze QOL by uveitis diagnosis and SVI quartile
    if qol_scores:
        print("\nAnalyzing QOL by uveitis diagnosis and SVI quartile...")
        for measure in qol_scores:
            # Create crosstab of mean scores
            crosstab = pd.crosstab(df['SVI_quartile'], df['diagnosis of uveitis'], 
                                  values=df[measure], aggfunc='mean').round(1)
            print(f"\n{measure} mean scores by uveitis diagnosis and SVI quartile:")
            print(crosstab)
            
            # Save crosstab
            crosstab.to_csv(f'advanced_analyses/{measure}_by_uveitis_svi.csv')
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            crosstab.plot(kind='bar', rot=0)
            
            measure_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
            plt.title(f'{measure_name} by Uveitis Diagnosis and SVI Quartile')
            plt.xlabel('SVI Quartile')
            plt.ylabel('Mean Score (higher = better)')
            plt.ylim(0, 105)
            plt.legend(title='Uveitis Diagnosis')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for i, container in enumerate(plt.gca().containers):
                plt.bar_label(container, fmt='%.1f', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'advanced_analyses/{measure}_by_uveitis_svi.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created {measure} by uveitis visualization")
            
            # For vision QOL, do more in-depth analysis if available
            if measure == 'vision_qol_score':
                print("\nDetailed analysis of vision QOL by uveitis diagnosis...")
                
                # Compare vision QOL by uveitis diagnosis with t-test
                uveitis_vision = df[df['diagnosis of uveitis'] == True]['vision_qol_score'].dropna()
                no_uveitis_vision = df[df['diagnosis of uveitis'] == False]['vision_qol_score'].dropna()
                
                if len(uveitis_vision) > 5 and len(no_uveitis_vision) > 5:
                    t_stat, p_val = stats.ttest_ind(uveitis_vision, no_uveitis_vision, equal_var=False)
                    print(f"Vision QOL t-test (uveitis vs no uveitis): t={t_stat:.2f}, p={p_val:.4f}")
                    
                    # Create boxplot
                    plt.figure(figsize=(8, 6))
                    box_data = df[['diagnosis of uveitis', 'vision_qol_score']].dropna()
                    sns.boxplot(x='diagnosis of uveitis', y='vision_qol_score', data=box_data)
                    
                    plt.title(f'Vision QOL by Uveitis Diagnosis\n(t={t_stat:.2f}, p={p_val:.4f})')
                    plt.xlabel('Uveitis Diagnosis')
                    plt.ylabel('Vision QOL Score (higher = better)')
                    plt.xticks([0, 1], ['No', 'Yes'])
                    
                    # Add sample size
                    plt.annotate(f'n={len(no_uveitis_vision)}', xy=(0, 105), ha='center')
                    plt.annotate(f'n={len(uveitis_vision)}', xy=(1, 105), ha='center')
                    
                    plt.tight_layout()
                    plt.savefig('advanced_analyses/vision_qol_by_uveitis.png', dpi=300, bbox_inches='tight')
                    plt.close()
else:
    print("No uveitis diagnosis column found for disease burden analysis")

# Check for JIA diagnosis 
jia_col = None
for col in df.columns:
    if 'jia' in col.lower() and 'diagnosis' in col.lower():
        jia_col = col
        break

if jia_col:
    print(f"\nFound JIA diagnosis column: {jia_col}. Distribution:\n{df[jia_col].value_counts()}")
    
    # Analyze QOL by JIA diagnosis and SVI quartile
    if qol_scores:
        print("\nAnalyzing QOL by JIA diagnosis and SVI quartile...")
        for measure in qol_scores:
            # Create crosstab of mean scores
            crosstab = pd.crosstab(df['SVI_quartile'], df[jia_col], 
                                  values=df[measure], aggfunc='mean').round(1)
            print(f"\n{measure} mean scores by JIA diagnosis and SVI quartile:")
            print(crosstab)
            
            # Save crosstab
            crosstab.to_csv(f'advanced_analyses/{measure}_by_jia_svi.csv')
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            crosstab.plot(kind='bar', rot=0)
            
            measure_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
            plt.title(f'{measure_name} by JIA Diagnosis and SVI Quartile')
            plt.xlabel('SVI Quartile')
            plt.ylabel('Mean Score (higher = better)')
            plt.ylim(0, 105)
            plt.legend(title='JIA Diagnosis')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for i, container in enumerate(plt.gca().containers):
                plt.bar_label(container, fmt='%.1f', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'advanced_analyses/{measure}_by_jia_svi.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created {measure} by JIA visualization")
else:
    print("No JIA diagnosis column found for disease burden analysis")

# ================ ANALYSIS 3: SVI SUBDOMAINS IMPACT ================
print("\n\n========== ANALYSIS 3: SVI SUBDOMAINS IMPACT ==========")

# Check if we have SVI theme columns for subdomain analysis
svi_themes = [col for col in df.columns if 'SVI_theme' in col]
print(f"Found {len(svi_themes)} SVI theme columns: {svi_themes}")

if svi_themes and qol_scores:
    print("\nAnalyzing impact of SVI subdomains on QOL...")
    
    # Create correlation heatmap between SVI themes and QOL measures
    corr_data = df[svi_themes + qol_scores].copy()
    
    # Rename columns for better readability
    renamed_cols = {}
    for col in corr_data.columns:
        if 'SVI_theme' in col:
            theme_name = col.replace('SVI_theme', '').strip('_')
            renamed_cols[col] = f"Theme {theme_name}"
        elif 'pedsql_' in col:
            renamed_cols[col] = col.replace('pedsql_', 'PedsQL ').replace('_score', '')
        elif 'chaq_' in col:
            renamed_cols[col] = col.replace('chaq_', 'CHAQ ').replace('_score', '')
        elif '_score' in col:
            renamed_cols[col] = col.replace('_score', '')
    
    corr_data.rename(columns=renamed_cols, inplace=True)
    
    # Calculate correlations
    correlations = corr_data.corr().round(2)
    
    # Filter to only show correlations between themes and QOL measures
    theme_cols = [renamed_cols[col] for col in svi_themes]
    qol_cols = [renamed_cols[col] for col in qol_scores]
    theme_qol_corr = correlations.loc[theme_cols, qol_cols]
    
    # Save correlation table
    theme_qol_corr.to_csv('advanced_analyses/svi_themes_qol_correlations.csv')
    print(f"Saved SVI theme correlations to advanced_analyses/svi_themes_qol_correlations.csv")
    
    # Create heatmap visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(theme_qol_corr, annot=True, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5)
    plt.title('Correlation between SVI Themes and QOL Measures', fontsize=14)
    plt.tight_layout()
    plt.savefig('advanced_analyses/svi_themes_qol_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created SVI themes correlation heatmap")
    
    # For each QOL measure, find which SVI theme has strongest correlation
    for qol_col in qol_cols:
        qol_theme_corr = theme_qol_corr[qol_col].sort_values(ascending=False)
        print(f"\nSVI themes most correlated with {qol_col}:")
        print(qol_theme_corr)
        
        # Use multivariate regression to see which themes are independently predictive
        y = df[qol_scores[qol_cols.index(qol_col)]].dropna()
        X_cols = [col for col in svi_themes if df[col].notna().any()]
        X = df.loc[y.index, X_cols].copy()
        
        # Only proceed if we have enough data
        if len(y) > 10 and not X.empty and not X.isna().all().all():
            # Add constant for statsmodels
            X = sm.add_constant(X)
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Print summary of significant predictors
            predictors = []
            for var, coef, pval in zip(model.params.index, model.params, model.pvalues):
                if var != 'const' and pval < 0.1:  # Use 0.1 threshold to show marginally significant results
                    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else '.'))
                    predictors.append(f"{var}: β={coef:.3f} {sig} (p={pval:.3f})")
            
            if predictors:
                print(f"Significant SVI theme predictors of {qol_col}:")
                for pred in predictors:
                    print(f"  {pred}")
            else:
                print(f"No significant SVI theme predictors for {qol_col}")
        else:
            print(f"Insufficient data for regression analysis of {qol_col}")
else:
    print("Cannot perform SVI subdomain analysis - missing SVI themes or QOL scores")

# ================ ANALYSIS 4: THRESHOLDS FOR INTERVENTIONS ================
print("\n\n========== ANALYSIS 4: THRESHOLDS FOR INTERVENTIONS ==========")

# Identify potential thresholds for interventions based on SVI and QOL scores
if qol_scores:
    print("\nExploring potential QOL thresholds for interventions...")
    
    # For each QOL measure, analyze distribution and identify potential cutoffs
    for measure in qol_scores:
        data = df[measure].dropna()
        if len(data) < 10:
            continue
            
        # Calculate percentiles
        p25, p33, median, p66, p75 = data.quantile([0.25, 0.33, 0.5, 0.66, 0.75])
        
        print(f"\n{measure} distribution:")
        print(f"  Mean: {data.mean():.1f}, Median: {median:.1f}")
        print(f"  25th percentile: {p25:.1f}, 33rd percentile: {p33:.1f}")
        print(f"  66th percentile: {p66:.1f}, 75th percentile: {p75:.1f}")
        
        # Analyze proportion of patients in Q4 (high SVI) under different thresholds
        q4_patients = df[df['SVI_quartile'] == 'Q4 (High)'][measure].dropna()
        if len(q4_patients) < 5:
            continue
            
        print(f"Proportion of Q4 (high SVI) patients below thresholds:")
        for threshold in [p25, p33, median, p66, p75]:
            pct_below = (q4_patients < threshold).mean() * 100
            print(f"  Below {threshold:.1f}: {pct_below:.1f}%")
        
        # Create visualization of distribution with potential thresholds
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(data, kde=True, bins=15)
        
        # Add vertical lines for thresholds
        thresholds = [p25, p33, median, p66, p75]
        threshold_labels = ["25th", "33rd", "50th", "66th", "75th"]
        colors = ['darkred', 'firebrick', 'black', 'royalblue', 'darkblue']
        
        for i, (threshold, label, color) in enumerate(zip(thresholds, threshold_labels, colors)):
            plt.axvline(x=threshold, color=color, linestyle='--', 
                       label=f"{label} percentile: {threshold:.1f}")
        
        # Customize plot
        measure_name = measure.replace('pedsql_', 'PedsQL ').replace('chaq_', 'CHAQ ').replace('_', ' ').title()
        plt.title(f'Distribution of {measure_name} with Potential Thresholds', fontsize=14)
        plt.xlabel(f'{measure_name} (higher = better)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(title="Potential thresholds", loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'advanced_analyses/{measure}_thresholds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created threshold visualization for {measure}")
        
        # Create box plot showing distribution by SVI quartile
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='SVI_quartile', y=measure, data=df, order=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        
        # Add horizontal lines for thresholds
        for threshold, label, color in zip(thresholds, threshold_labels, colors):
            plt.axhline(y=threshold, color=color, linestyle='--', 
                       label=f"{label} percentile: {threshold:.1f}")
        
        # Customize plot
        plt.title(f'{measure_name} by SVI Quartile with Potential Thresholds', fontsize=14)
        plt.xlabel('SVI Quartile', fontsize=12)
        plt.ylabel(f'{measure_name} (higher = better)', fontsize=12)
        plt.legend(title="Potential thresholds", loc='lower left')
        
        # Add SVI quartile distribution information
        for i, quartile in enumerate(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']):
            quartile_data = df[df['SVI_quartile'] == quartile][measure].dropna()
            if len(quartile_data) > 0:
                plt.annotate(f'n={len(quartile_data)}', xy=(i, plt.ylim()[0] + 5), ha='center')
                for j, threshold in enumerate(thresholds):
                    pct_below = (quartile_data < threshold).mean() * 100
                    plt.annotate(f'{pct_below:.0f}% < {threshold:.0f}', 
                                xy=(i, threshold + 2 + j*3), ha='center', fontsize=8, color=colors[j])
        
        plt.tight_layout()
        plt.savefig(f'advanced_analyses/{measure}_by_svi_thresholds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created SVI quartile threshold visualization for {measure}")
else:
    print("Cannot perform threshold analysis - missing QOL scores")

print("\nAdvanced analyses complete. Results saved in 'advanced_analyses' directory.") 