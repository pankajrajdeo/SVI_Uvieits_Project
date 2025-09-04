# Comprehensive Uveitis and Social Vulnerability Analysis

## Overview
This report analyzes the relationship between uveitis, treatment timing, and social vulnerability indices (SVI) in a cohort of 495 pediatric patients. The analysis investigates:
1. The prevalence of uveitis across different SVI quartiles
2. The time between uveitis diagnosis and immunosuppressive treatment initiation
3. Patterns of treatment timing across SVI quartiles

## Uveitis Prevalence

Among the 495 patients analyzed:
- 125 patients (25.3%) were diagnosed with uveitis
- The prevalence varied significantly by SVI quartile:
  - Q1 (Low SVI): 45.6% (57 of 125 patients)
  - Q2: 0.8% (1 of 123 patients)
  - Q3: 26.2% (33 of 126 patients)
  - Q4 (High SVI): 28.1% (34 of 121 patients)

A chi-square test revealed a statistically significant difference in uveitis prevalence across SVI quartiles (p < 0.0001), indicating a strong association between social vulnerability and uveitis diagnosis.

![Uveitis Prevalence by SVI Quartile](uveitis_prevalence_by_svi.png)

## Treatment Timing Analysis

Of the 125 patients with uveitis, 106 (84.8%) had both diagnosis year and treatment data available for analysis:

### Overall Treatment Timing
- Mean time to treatment: 0.2 years (approximately 2.4 months)
- Median time to treatment: 0.0 years (most patients received treatment in the same year as diagnosis)
- Range: 0 to 1 years

### Distribution of Time to Treatment
- Same year as diagnosis (0 years): 85 patients (80.2%)
- One year after diagnosis (1 year): 21 patients (19.8%)

### Treatment Timing by SVI Quartile
- Q1 (Low SVI): Mean = 0.2 years, Median = 0.0 years, n = 40
- Q3: Mean = 0.0 years, Median = 0.0 years, n = 33
- Q4 (High SVI): Mean = 0.4 years, Median = 0.0 years, n = 33

A Kruskal-Wallis test showed a statistically significant difference in treatment timing across SVI quartiles (p = 0.001). This suggests that social vulnerability may influence how quickly patients receive immunosuppressive treatment after uveitis diagnosis.

![Treatment Timing by SVI Quartile](time_to_treatment_by_svi.png)

## Detailed Visualization

The following dashboard provides a comprehensive view of the relationship between uveitis, treatment timing, and social vulnerability:

![Comprehensive Dashboard](uveitis_svi_dashboard.png)

A more detailed visualization of treatment timing by SVI quartile, including individual patient data points:

![Detailed Treatment Timing](uveitis_treatment_timing_detailed.png)

## Key Findings

1. **Uveitis Prevalence and SVI**:
   - Uveitis shows a complex relationship with social vulnerability
   - The lowest SVI quartile (Q1) has the highest uveitis prevalence (45.6%)
   - Q2 shows virtually no uveitis cases (0.8%)
   - Q3 and Q4 (higher vulnerability) show moderate uveitis prevalence (26-28%)

2. **Treatment Timing and SVI**:
   - Most patients (80.2%) receive immunosuppressive treatment in the same year as diagnosis
   - Patients in the highest vulnerability quartile (Q4) show the longest average time to treatment (0.4 years)
   - Patients in Q3 show the shortest time to treatment (all treated within the same year)

3. **Statistical Significance**:
   - Both the relationship between uveitis prevalence and SVI quartile (p < 0.0001)
   - The relationship between treatment timing and SVI quartile (p = 0.001)
   - These are statistically significant, indicating true associations rather than random chance

## Implications

These findings suggest:

1. **Social determinants matter in uveitis care**: There is clear evidence that social vulnerability affects both the prevalence of diagnosed uveitis and the timing of treatment.

2. **Non-linear relationship**: The relationship between SVI and uveitis is not simply linear, with the highest prevalence in the lowest vulnerability group (Q1). This suggests complex factors at play that require further investigation.

3. **Treatment disparities**: While most patients receive timely treatment, those in the highest vulnerability quartile (Q4) experience longer times to treatment on average, suggesting potential barriers to care.

4. **Targeted interventions**: These findings can inform targeted interventions to improve timely diagnosis and treatment of uveitis, particularly focusing on higher vulnerability populations.

## Limitations

This analysis has several limitations:

1. **Data quality**: The dataset had limitations in the format of dates, with many patients having only year information without specific month and day.

2. **Missing treatment data**: Not all patients with uveitis had complete treatment timing information.

3. **Limited timeframe**: The maximum observed treatment delay was only 1 year, which may reflect data limitations rather than the full range of clinical reality.

4. **Potential confounders**: The analysis does not account for potential confounding factors such as disease severity, access to care, insurance status, etc.

## Conclusion

This analysis reveals significant associations between social vulnerability, uveitis prevalence, and treatment timing. The findings suggest that social determinants of health play an important role in pediatric uveitis care, with both diagnosis patterns and treatment timing showing variations across SVI quartiles.

Further research is needed to better understand the complex relationship between social vulnerability and uveitis, particularly the unexpectedly high prevalence in the lowest vulnerability quartile and the near absence of cases in Q2.

These insights can inform clinical practice and health policy to address disparities in care and improve outcomes for all pediatric patients with uveitis, regardless of social vulnerability status. 