# SVI and Quality of Life Analysis

## Project Overview
This project analyzes the relationship between Social Vulnerability Index (SVI) and Quality of Life (QOL) measures in pediatric ophthalmology patients. The analysis focuses on how social determinants of health impact quality of life across different age groups and clinical subpopulations.

## Key Files and Directories

### Data Files
- `SVI_filtered_495_patients.csv`: Original dataset with 495 patients
- `SVI_filtered_495_patients_with_qol_scores.csv`: Dataset with calculated QOL scores
- `qol_svi_results/qol_columns_inventory.csv`: Inventory of QOL-related columns

### Analysis Scripts
- `comprehensive_qol_svi_analysis.py`: Initial analysis script that calculates QOL scores and examines their relationship with SVI
- `extended_svi_qol_analysis.py`: Advanced analysis script that examines age-specific patterns, disease burden, SVI subdomain impacts, and intervention thresholds
- `save_processed_data.py`: Script to save processed data with calculated QOL scores
- `check_qol_scores.py`: Utility script to check for QOL-related columns in the dataset
- `create_summary_visualization.py`: Script to create a summary visualization of key findings

### Results and Reports
- `qol_svi_results/`: Directory containing initial analysis results
  - `comprehensive_qol_svi_results.csv`: Detailed results from initial analysis
  - `Comprehensive_SVI_QOL_Report.md`: Report on initial findings
  - Various boxplot visualizations for each QOL measure
  
- `advanced_analyses/`: Directory containing advanced analysis results
  - `qol_by_age_svi.csv`: Results of QOL analysis by age group and SVI quartile
  - Various visualizations for QOL measures by age, SVI quartile, and uveitis status
  
- `summary_visualizations/`: Summary visualizations combining key findings
  - `svi_qol_summary.png`: Comprehensive summary of SVI impact on QOL

- `qol_analysis_recommendations.md`: Recommendations for QOL analysis
- `Advanced_SVI_QOL_Analysis_Report.md`: Detailed report on advanced analysis findings

## Key Findings

1. **Age-Specific SVI Impact**: Different age groups show varying patterns of SVI impact on QOL:
   - Adolescents (13-18): Strongest impact on emotional functioning
   - School-age (8-12): Significant impact across multiple domains
   - Young children (5-7): Most significant in physical and social domains

2. **Disease Burden**: Patients with uveitis from high SVI areas show substantially lower QOL scores across multiple domains.

3. **Intervention Thresholds**: Identified specific QOL thresholds that could trigger clinical interventions, with a significant percentage of high SVI patients falling below these thresholds.

## How to Use This Repository

1. To calculate QOL scores:
   ```
   python comprehensive_qol_svi_analysis.py
   ```

2. To run the extended analysis:
   ```
   python extended_svi_qol_analysis.py
   ```

3. To generate the summary visualization:
   ```
   python create_summary_visualization.py
   ```

## Clinical Implications

The findings suggest that:
1. SVI should be considered when assessing QOL in pediatric ophthalmology patients
2. Age-specific interventions may be needed to address QOL concerns
3. Patients with both high SVI and certain conditions (e.g., uveitis) may require additional support
4. Specific QOL thresholds can be used to identify patients at highest risk

## Future Directions

1. Investigate specific SVI components (housing, transportation, etc.) that most impact QOL
2. Develop and test targeted interventions for high SVI patients
3. Conduct longitudinal analysis to determine how SVI impacts QOL over time
4. Explore mediating factors between SVI and QOL outcomes 