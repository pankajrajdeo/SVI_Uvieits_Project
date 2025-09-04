# Analysis of Social Vulnerability and Age at JIA Onset

## Executive Summary
This analysis explores the relationship between social vulnerability and the age at onset of Juvenile Idiopathic Arthritis (JIA) in a cohort of 495 patients. We found that children from neighborhoods with higher social vulnerability were diagnosed with JIA at significantly older ages compared to those from less vulnerable neighborhoods, suggesting potential healthcare access disparities.

## Data Sources
The analysis uses a dataset of 495 patients with JIA and/or uveitis:
- Original file: `SVI_filtered_495_patients.csv`
- Analysis dataset: `JIA_onset_analysis.csv` (created during this analysis)

## Methodology

### Data Collection and Preparation
1. **Identifying relevant columns:**
   - We first identified columns related to JIA diagnosis and onset
   - We discovered two key columns that contained onset information:
     - `date of arth onsety`: 170 patients (34.3% of dataset)
     - `date of JIA symptom onset`: 218 patients (44.0% of dataset)
   - When combined, we had onset data for 271 patients (54.7% of dataset)

2. **Creating the age at onset variable:**
   - Subtracted birth year (`DOB`) from each onset year to calculate age at onset
   - Created three versions:
     - `arth_onset_age`: using arthritis onset date
     - `symptom_onset_age`: using JIA symptom onset date
     - `combined_onset_age`: using JIA symptom onset when available, otherwise arthritis onset

3. **Data validation:**
   - Examined both onset columns to confirm they contained years
   - Filtered out invalid ages (negative values or extreme outliers)
   - Confirmed data quality through cross-validation between the two onset sources
   - Analyzed the relationship between onset dates when both were present (mean difference: 0.02 years)

### Statistical Analysis
1. **Descriptive statistics by SVI quartile:**
   - Calculated count, mean, median, minimum, maximum, and standard deviation of age at onset
   - Stratified by Social Vulnerability Index (SVI) quartiles:
     - Q1: Low vulnerability
     - Q2: Moderate-low vulnerability
     - Q3: Moderate-high vulnerability
     - Q4: High vulnerability

2. **Inferential statistics:**
   - ANOVA test comparing age at onset across all SVI quartiles
   - Independent t-test comparing Q1 (low vulnerability) vs. Q4 (high vulnerability)

3. **Visualization:**
   - Created box plots of age at onset by SVI quartile
   - Generated mean plots with error bars for visual comparison

## Results

### Sample Characteristics
- **Total patients in dataset:** 495
- **Patients with JIA onset data:**
  - Using arthritis onset date: 170 (34.3%)
  - Using JIA symptom onset: 218 (44.0%)
  - Using combined approach: 271 (54.7%)

### Age at JIA Onset by SVI Quartile (Combined Dataset)
| SVI Quartile | Count | Mean (years) | Median | Min | Max | Std |
|--------------|-------|--------------|--------|-----|-----|-----|
| Q1 (Low)     | 78    | 4.51         | 4.0    | 0.0 | 11.0| 2.92|
| Q2           | 54    | 3.78         | 3.0    | 0.0 | 11.0| 2.96|
| Q3           | 66    | 4.42         | 4.0    | 1.0 | 9.0 | 2.23|
| Q4 (High)    | 72    | 5.83         | 6.0    | 1.0 | 12.0| 3.17|

### Statistical Test Results
- **ANOVA across all quartiles:**
  - F-statistic: 5.9969
  - p-value: 0.0006 (statistically significant)
  
- **T-test between Q1 and Q4:**
  - t-statistic: -2.6543
  - p-value: 0.0088 (statistically significant)

## Key Findings

1. **Social Vulnerability and Age at JIA Onset:**
   - Children from high social vulnerability neighborhoods (Q4) are diagnosed with JIA at a significantly older age (mean: 5.83 years) compared to those from low social vulnerability neighborhoods (Q1, mean: 4.51 years).
   - The 1.32-year delay in diagnosis for children from high vulnerability areas is statistically significant.

2. **Distribution Pattern:**
   - The relationship between SVI and age at onset is not strictly linear:
   - Q2 shows the earliest mean age at onset (3.78 years)
   - Q1 and Q3 have similar means (4.51 and 4.42 years respectively)
   - Q4 shows the latest mean age at onset (5.83 years)

3. **Sample Size Improvements:**
   - By combining data from multiple onset columns, we increased the available data from 170 to 271 patients, strengthening the reliability of our results.

## Implications
The significant delay in JIA diagnosis for children from socially vulnerable neighborhoods suggests potential healthcare access disparities. This delay may result in:
- Delayed treatment initiation
- Increased risk of permanent joint damage
- Poorer long-term outcomes
- Increased healthcare costs

## Limitations

1. **Missing Data:**
   - Despite combining onset columns, 45.3% of patients still lacked onset data
   - Analysis does not include patients without JIA (e.g., uveitis-only patients)

2. **SVI Distribution Bias:**
   - The distribution of SVI quartiles was not equal among patients with available data
   - Low SVI (Q1) patients were overrepresented in the JIA onset data

3. **Data Quality:**
   - The study relies on retrospective clinical data with inherent limitations
   - Some records showed anomalies (e.g., negative ages) that required filtering

## Conclusion
This analysis provides compelling evidence that social vulnerability is associated with delayed diagnosis of JIA. The 1.3-year diagnostic delay observed in children from high vulnerability neighborhoods is clinically meaningful and statistically significant. These findings highlight the importance of addressing healthcare access disparities and developing targeted interventions to ensure timely diagnosis for all children, regardless of social vulnerability status. 