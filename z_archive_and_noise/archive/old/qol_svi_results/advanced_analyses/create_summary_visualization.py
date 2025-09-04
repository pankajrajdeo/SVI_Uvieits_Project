import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create directory for summary visualizations if it doesn't exist
if not os.path.exists('summary_visualizations'):
    os.makedirs('summary_visualizations')

# Load the age group analysis data
age_data = pd.read_csv('advanced_analyses/qol_by_age_svi.csv')

# Filter out ANOVA rows and only keep mean scores for visualization
age_viz_data = age_data[~age_data['SVI Quartile'].str.contains('ANOVA')]

# Create a comprehensive comparison plot
plt.figure(figsize=(20, 12))
sns.set_style("whitegrid")

# Plot 1: Age group comparison for key measures
plt.subplot(2, 2, 1)
key_measures = ['pedsql_emotional_score', 'pedsql_social_score', 'vision_qol_score']
age_plot_data = age_viz_data[age_viz_data['QOL Measure'].isin(key_measures) & 
                            (age_viz_data['SVI Quartile'] == 'Q4 (High)')]

pivot_data = age_plot_data.pivot(index='Age Group', columns='QOL Measure', values='Mean Score')
pivot_data.plot(kind='bar', ax=plt.gca(), colormap='viridis')
plt.title('Impact of High SVI (Q4) on QOL by Age Group', fontsize=14)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Mean Score', fontsize=12)
plt.ylim(0, 100)
plt.legend(title='QOL Measure', fontsize=10)
plt.xticks(rotation=45)

# Plot 2: SVI quartile comparison across all ages for key measures
plt.subplot(2, 2, 2)
quartile_data = age_viz_data[age_viz_data['QOL Measure'].isin(key_measures)]
quartile_pivot = quartile_data.groupby(['SVI Quartile', 'QOL Measure'])['Mean Score'].mean().reset_index()
quartile_pivot = quartile_pivot.pivot(index='SVI Quartile', columns='QOL Measure', values='Mean Score')

# Ensure consistent order of quartiles
quartile_order = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
quartile_pivot = quartile_pivot.reindex(quartile_order)

quartile_pivot.plot(kind='bar', ax=plt.gca(), colormap='viridis')
plt.title('QOL Measures by SVI Quartile (All Ages)', fontsize=14)
plt.xlabel('SVI Quartile', fontsize=12)
plt.ylabel('Mean Score', fontsize=12)
plt.ylim(0, 100)
plt.legend(title='QOL Measure', fontsize=10)
plt.xticks(rotation=45)

# Plot 3: Uveitis patients by SVI quartile
plt.subplot(2, 2, 3)

# Directly read and process the uveitis data for key measures
uveitis_data = {}
for measure in key_measures:
    try:
        file_path = f'advanced_analyses/{measure}_by_uveitis_svi.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                # The CSV has SVI_quartile as index and 'True' as column name for uveitis patients
                uveitis_data[measure] = df.set_index('SVI_quartile')['True']
    except Exception as e:
        print(f"Error processing {measure}: {e}")

if uveitis_data:
    # Convert to DataFrame for plotting
    uveitis_df = pd.DataFrame(uveitis_data)
    
    # Ensure consistent order of quartiles
    quartile_order = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    uveitis_df = uveitis_df.reindex(quartile_order)
    
    uveitis_df.plot(kind='bar', ax=plt.gca(), colormap='viridis')
    plt.title('QOL in Uveitis Patients by SVI Quartile', fontsize=14)
    plt.xlabel('SVI Quartile', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(title='QOL Measure', fontsize=10)
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, "No uveitis data available", ha='center', va='center', fontsize=14)

# Plot 4: Threshold analysis for high SVI patients
plt.subplot(2, 2, 4)

# Manually create data based on the threshold analysis
measures = ['Emotional', 'Social', 'Vision QOL', 'PedsQL Total']
below_median = [65.6, 53.1, 75.0, 56.2]
below_threshold = [25.0, 21.9, 48.1, 29.7]

x = np.arange(len(measures))
width = 0.35

ax = plt.gca()
rects1 = ax.bar(x - width/2, below_median, width, label='Below Median')
rects2 = ax.bar(x + width/2, below_threshold, width, label='Below Concerning Threshold')

ax.set_ylabel('Percentage of High SVI Patients', fontsize=12)
ax.set_title('High SVI Patients Below QOL Thresholds', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(measures)
ax.set_ylim(0, 100)
ax.legend()

# Add some text for labels, title and custom x-axis tick labels, etc.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.suptitle('Summary of SVI Impact on Quality of Life', fontsize=16, y=1.02)
plt.savefig('summary_visualizations/svi_qol_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Summary visualization created at 'summary_visualizations/svi_qol_summary.png'") 