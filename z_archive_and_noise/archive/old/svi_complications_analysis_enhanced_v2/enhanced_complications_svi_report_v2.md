# Enhanced Analysis of Complications by Social Vulnerability Index (SVI) - Version 2

## Overview
- Total patients analyzed: 495
- Patients with valid SVI scores: 495
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
     - Cataract: 211 patients (42.6%)
     - Glaucoma: 315 patients (63.6%)
     - Synechiae: 1 patients (0.2%)
     - Surgery: 119 patients (24.0%)
     - Band Keratopathy: 4 patients (0.8%)
     - Iridectomy: 19 patients (3.8%)
     - Vitrectomy: 13 patients (2.6%)
     - Uveitis: 125 patients (25.3%)
     - Uveitis Complications: 0 patients (0.0%)
     - Injection Procedure: 174 patients (35.2%)
     - Other Complication: 119 patients (24.0%)
     - Steroid Treatment: 356 patients (71.9%)
     - Pain: 113 patients (22.8%)
     - Any complication: 422 patients (85.3%)

## Key Findings

### Complication Prevalence by SVI Quartile

| Complication | SVI Quartile | Total Patients | Cases | Prevalence (%) | Statistical Test |
|--------------|--------------|----------------|-------|----------------|------------------|
| cataract | Q1: Low | 125.0 | 70.0 | 56.0% | Chi² = 38.70, p = 0.0000 |
| cataract | Q2 | 123.0 | 30.0 | 24.4% | Chi² = 38.70, p = 0.0000 |
| cataract | Q3 | 126.0 | 43.0 | 34.1% | Chi² = 38.70, p = 0.0000 |
| cataract | Q4: High | 121.0 | 68.0 | 56.2% | Chi² = 38.70, p = 0.0000 |
| glaucoma | Q1: Low | 125.0 | 87.0 | 69.6% | Chi² = 7.79, p = 0.0505 |
| glaucoma | Q2 | 123.0 | 68.0 | 55.3% | Chi² = 7.79, p = 0.0505 |
| glaucoma | Q3 | 126.0 | 87.0 | 69.0% | Chi² = 7.79, p = 0.0505 |
| glaucoma | Q4: High | 121.0 | 73.0 | 60.3% | Chi² = 7.79, p = 0.0505 |
| synechiae | Q1: Low | 125.0 | 1.0 | 0.8% | Insufficient data for statistical testing |
| synechiae | Q2 | 123.0 | 0.0 | 0.0% | Insufficient data for statistical testing |
| synechiae | Q3 | 126.0 | 0.0 | 0.0% | Insufficient data for statistical testing |
| synechiae | Q4: High | 121.0 | 0.0 | 0.0% | Insufficient data for statistical testing |
| surgery | Q1: Low | 125.0 | 57.0 | 45.6% | Chi² = 69.72, p = 0.0000 |
| surgery | Q2 | 123.0 | 1.0 | 0.8% | Chi² = 69.72, p = 0.0000 |
| surgery | Q3 | 126.0 | 27.0 | 21.4% | Chi² = 69.72, p = 0.0000 |
| surgery | Q4: High | 121.0 | 34.0 | 28.1% | Chi² = 69.72, p = 0.0000 |
| iridectomy | Q1: Low | 125.0 | 1.0 | 0.8% | Chi² = 20.44, p = 0.0001 |
| iridectomy | Q2 | 123.0 | 0.0 | 0.0% | Chi² = 20.44, p = 0.0001 |
| iridectomy | Q3 | 126.0 | 6.0 | 4.8% | Chi² = 20.44, p = 0.0001 |
| iridectomy | Q4: High | 121.0 | 12.0 | 9.9% | Chi² = 20.44, p = 0.0001 |
| vitrectomy | Q1: Low | 125.0 | 1.0 | 0.8% | Chi² = 33.50, p = 0.0000 |
| vitrectomy | Q2 | 123.0 | 0.0 | 0.0% | Chi² = 33.50, p = 0.0000 |
| vitrectomy | Q3 | 126.0 | 0.0 | 0.0% | Chi² = 33.50, p = 0.0000 |
| vitrectomy | Q4: High | 121.0 | 12.0 | 9.9% | Chi² = 33.50, p = 0.0000 |
| uveitis | Q1: Low | 125.0 | 57.0 | 45.6% | Chi² = 66.92, p = 0.0000 |
| uveitis | Q2 | 123.0 | 1.0 | 0.8% | Chi² = 66.92, p = 0.0000 |
| uveitis | Q3 | 126.0 | 33.0 | 26.2% | Chi² = 66.92, p = 0.0000 |
| uveitis | Q4: High | 121.0 | 34.0 | 28.1% | Chi² = 66.92, p = 0.0000 |
| injection procedure | Q1: Low | 125.0 | 68.0 | 54.4% | Chi² = 27.49, p = 0.0000 |
| injection procedure | Q2 | 123.0 | 33.0 | 26.8% | Chi² = 27.49, p = 0.0000 |
| injection procedure | Q3 | 126.0 | 38.0 | 30.2% | Chi² = 27.49, p = 0.0000 |
| injection procedure | Q4: High | 121.0 | 35.0 | 28.9% | Chi² = 27.49, p = 0.0000 |
| steroid treatment | Q1: Low | 125.0 | 88.0 | 70.4% | Chi² = 9.21, p = 0.0267 |
| steroid treatment | Q2 | 123.0 | 77.0 | 62.6% | Chi² = 9.21, p = 0.0267 |
| steroid treatment | Q3 | 126.0 | 99.0 | 78.6% | Chi² = 9.21, p = 0.0267 |
| steroid treatment | Q4: High | 121.0 | 92.0 | 76.0% | Chi² = 9.21, p = 0.0267 |
| pain | Q1: Low | 125.0 | 37.0 | 29.6% | Chi² = 40.56, p = 0.0000 |
| pain | Q2 | 123.0 | 5.0 | 4.1% | Chi² = 40.56, p = 0.0000 |
| pain | Q3 | 126.0 | 27.0 | 21.4% | Chi² = 40.56, p = 0.0000 |
| pain | Q4: High | 121.0 | 44.0 | 36.4% | Chi² = 40.56, p = 0.0000 |
| Any complication | Q1: Low | 125.0 | 106.0 | 84.8% | Chi² = 7.69, p = 0.0528 |
| Any complication | Q2 | 123.0 | 97.0 | 78.9% | Chi² = 7.69, p = 0.0528 |
| Any complication | Q3 | 126.0 | 115.0 | 91.3% | Chi² = 7.69, p = 0.0528 |
| Any complication | Q4: High | 121.0 | 104.0 | 86.0% | Chi² = 7.69, p = 0.0528 |

### Complication Count by SVI Quartile

| SVI Quartile | Patients | Mean Count | Std Dev | Min | Median | Max |
|--------------|----------|------------|---------|-----|--------|-----|
| Q1: Low | 125.0 | 4.22 | 2.97 | 0.0 | 4.0 | 10.0 |
| Q2 | 123.0 | 1.76 | 1.36 | 0.0 | 2.0 | 5.0 |
| Q3 | 126.0 | 3.07 | 2.31 | 0.0 | 2.0 | 8.0 |
| Q4: High | 121.0 | 3.62 | 2.94 | 0.0 | 3.0 | 10.0 |

### Statistical Analysis
- ANOVA: F = 22.26, p = 0.0000

## Interpretation
The analysis reveals a statistically significant relationship between social vulnerability (SVI) and complication rates, though with a complex pattern. Unexpectedly, patients from less vulnerable areas (Q1: Low SVI) experienced more complications on average (4.22) than those from higher vulnerability areas (Q4: High SVI, 3.62). This counter-intuitive finding suggests:

1. **Access to Care Differences**: Patients from less vulnerable areas may have better access to specialized care, leading to more diagnoses and interventions being documented.
2. **Surveillance Bias**: More frequent medical visits and better access to subspecialty care in less vulnerable populations may result in higher detection rates of complications.
3. **Documentation Differences**: There may be systematic differences in how complications are documented across different healthcare settings serving different SVI populations.
4. **Complex Relationship**: The relationship between social vulnerability and ocular complications may not be linear or straightforward.

These findings highlight the complexity of healthcare disparities and suggest that simply measuring complication rates without considering care access and documentation patterns may not fully capture the impact of social vulnerability on health outcomes.

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
