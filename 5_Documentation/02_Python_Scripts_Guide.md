
# **02: Python Scripts Guide**

---

This document provides a detailed reference for every Python script in the project. The scripts are organized by their location and purpose.

---

## **Core Analysis Scripts**

These scripts perform the primary analyses of the project and are located in their respective subdirectories within `2_Scripts/`.

### **`2_Scripts/1_complications_analysis/`**

*   **File**: `svi_complications_analysis.py`
*   **Purpose**: To determine if higher SVI is associated with a greater lifetime prevalence of ocular complications.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `svi_vs_comp_count_stats.csv`: CSV with mean complication counts by SVI quartile.
    *   `svi_vs_any_comp_proportion_stats.csv`: CSV with the proportion of patients having any complication.
    *   `svi_vs_mean_comp_count_plot.png`: Bar chart of mean complication counts.
    *   `svi_vs_any_comp_proportion_plot.png`: Bar chart of the proportion of patients with any complication.
    *   `grouped_complications_*.png`: Grouped bar charts showing the prevalence of specific complications.
*   **Key Logic**:
    1.  Loads the main dataset.
    2.  Calculates `SVI_total` and `SVI_quartile` for each patient.
    3.  Scans multiple free-text columns (e.g., `dx name`, `procedure name`) for keywords like `'cataract'`, `'glaucoma'`, `'surgery'` to create binary flags for each complication.
    4.  Groups patients by `Diagnosis_Group` and `SVI_quartile`.
    5.  Performs statistical tests (Chi-squared) to compare complication rates across SVI quartiles.
    6.  Generates and saves plots and summary tables.

---

### **`2_Scripts/2_demographics_analysis/`**

*   **File**: `generate_table1_comparison_2Groups.py`
*   **Purpose**: To generate a classic "Table 1" of demographic and clinical characteristics, comparing the `JIA-Only` and `Any Uveitis` cohorts.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `table1_comparison_output.txt`: A detailed text file containing the formatted table with counts, percentages, and p-values.
    *   `patient_groups.csv`: An intermediate file mapping patient IDs to their assigned `Diagnosis_Group`.
*   **Key Logic**:
    1.  Loads the main dataset.
    2.  Groups patients into `JIA-Only` and `Any Uveitis` based on ICD codes in `dx code (list distinct)`.
    3.  For each variable (e.g., `gender`, `race`, `JIA_Subtype`), it calculates counts and percentages for each group.
    4.  It performs the appropriate statistical test (Chi-squared for categorical, Mann-Whitney U for continuous) to get a p-value comparing the two groups.
    5.  Formats and prints all results into a text file.

*   **File**: `svi_demographic_table_by_quartile.py`
*   **Purpose**: To generate demographic tables stratified by SVI quartiles.
*   **Outputs**:
    *   `svi_halves_demographics_table.csv`: Demographic comparison by SVI halves.
    *   `svi_quartile_demographics_table.csv`: Demographic comparison by SVI quartiles.

---

### **`2_Scripts/3_jia_subtype_analysis/`**

*   **File**: `svi_jia_subtype_analysis.py`
*   **Purpose**: To investigate if the mean SVI score differs across various JIA subtypes.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `*_SVI_Summary_Stats.csv`: CSVs with SVI summary statistics for each JIA subtype.
    *   `*_Subtype_Counts_by_SVI.csv`: CSVs with patient counts for each subtype within each SVI quartile.
    *   `*_SVI_by_Subtype_Violin.png`: Violin plots showing the distribution of SVI scores for each JIA subtype.
    *   `Dual_Comparison_*.png` and `.csv`: A bar chart and summary table directly comparing the mean SVI for each subtype between the `JIA-Only` and `Any Uveitis` groups.
*   **Key Logic**:
    1.  Loads the main dataset and calculates SVI.
    2.  Identifies the JIA subtype for each patient from columns like `ilar_code_display_value (list distinct)`.
    3.  Groups patients by their JIA subtype.
    4.  Calculates the mean SVI for each subtype.
    5.  Performs ANOVA to test if the mean SVI is significantly different across the subtypes.
    6.  Generates plots to visualize these distributions and comparisons.

---

### **`2_Scripts/4_onset_analysis/`**

*   **File**: `svi_age_onset_analysis.py`
*   **Purpose**: To determine if SVI is associated with the age of disease onset for JIA and Uveitis.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `Age_Onset_by_SVI_Quartile_Summary.csv`: A summary table of mean/median onset ages.
    *   `Combined_Onset_Age_vs_SVI_Lineplot.png`: A line plot showing the mean age of onset for both JIA and Uveitis across SVI quartiles.
*   **Key Logic**:
    1.  Loads data and calculates SVI.
    2.  Calculates patient age at disease onset by subtracting birth year (`DOB`) from the year of onset (`date of arth onsety`, `date of uv onsety`).
    3.  Compares the mean age of onset across the four SVI quartiles using ANOVA.
    4.  Generates and saves the summary plot.

---

### **`2_Scripts/5_qol_analysis/`**

*   **File**: `svi_qol_analysis.py`
*   **Purpose**: To analyze the relationship between SVI and multiple Quality of Life (QOL) measures.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**: This script generates a large number of files, organized into subdirectories for each QOL measure (e.g., `pedsql_total_score_child/`). Key outputs include:
    *   `*_descriptive_stats.csv`: Basic statistics for the QOL score.
    *   `*_by_svi_stats.csv`: The mean QOL score for each SVI quartile.
    *   `*_by_svi_plots.png`: Box plots showing the QOL score distribution across SVI quartiles.
    *   `*_svi_regression.png`: A scatter plot with a regression line showing the trend between SVI and the QOL score.
    *   `Figure3_SVI_QoL_MultiPanel.png`: A final summary figure combining the key QOL findings.
*   **Key Logic**:
    1.  Loads data and calculates SVI.
    2.  For each QOL instrument (PedsQL, CHAQ, etc.), it finds the relevant columns.
    3.  It processes the raw text-based answers (e.g., "Often a problem") into numerical scores and then transforms them into a standardized 0-100 scale (higher = better QOL).
    4.  It runs the full analysis suite (ANOVA, regression) for each QOL score against SVI for both the `JIA-Only` and `Any Uveitis` groups.
    5.  Generates and saves all plots and summary tables.

---

### **`2_Scripts/6_steroid_duration_analysis/`**

*   **File**: `svi_steroid_duration_analysis.py`
*   **Purpose**: To assess if SVI is associated with the duration of topical steroid eye drop use.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `*_duration_by_svi_stats.csv`: CSV with mean steroid duration by SVI quartile.
    *   `*_duration_by_svi_boxplot.png`: Box plots showing duration distribution across SVI quartiles.
    *   `*_svi_duration_regression.png`: Regression plot of SVI vs. duration.
*   **Key Logic**:
    1.  Loads data and calculates SVI.
    2.  Identifies patients on steroid eye drops by searching for keywords in `cmeyetrt (list distinct)`.
    3.  For those patients, it calculates the duration of use by finding the difference between the `eye drop end date` and `eye drop start date`.
    4.  Compares the mean duration across SVI quartiles.

---

### **`2_Scripts/7_va_analysis/`**

*   **File**: `svi_va_analysis.py`
*   **Purpose**: To analyze the relationship between SVI and worst-recorded visual acuity (VA).
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `svi_vs_worse_va_proportions_FINAL.csv`: CSV with the proportion of patients with poor vision (≥20/50) in each SVI quartile.
    *   `svi_vs_worse_va_proportion_plot_FINAL.png`: Bar chart visualizing these proportions.
    *   `logmar_analysis_by_group/`: A subdirectory containing box plots of LogMAR scores by SVI quartile.
*   **Key Logic**:
    1.  Loads data and calculates SVI.
    2.  Finds all VA columns (e.g., `exam left vadcc`).
    3.  Converts the various VA notations (e.g., "20/40", "CF", "HM") into the standardized **LogMAR** scale (higher score = worse vision).
    4.  Determines the single worst LogMAR score for each patient.
    5.  Compares both the mean LogMAR score and the proportion of patients with poor vision across SVI quartiles.

---

### **`2_Scripts/8_treatment_delay_analysis/`**

*   **File**: `svi_treatment_delay_analysis.py`
*   **Purpose**: To analyze the relationship between SVI and treatment delay patterns.
*   **Inputs**: `SVI_Pankaj_data_1_updated_merged_new.csv`
*   **Outputs**:
    *   `treatment_delay_summary_stats_all.csv`: Summary statistics for treatment delays.
    *   `treatment_delay_summary_stats_with_svi.csv`: Treatment delay statistics by SVI quartile.
    *   Various plots showing treatment delay patterns by group and SVI quartile.
*   **Key Logic**:
    1.  Loads data and calculates SVI.
    2.  Identifies treatment initiation dates and calculates delays.
    3.  Compares treatment timing across SVI quartiles and patient groups.

---

### **`2_Scripts/9_quartile_analysis/`**

*   **File**: `svi_quartile_analysis_2Groups.py`
*   **Purpose**: To perform comprehensive SVI quartile analyses for 2-group comparisons.
*   **Outputs**: Various quartile-based analyses and visualizations.

*   **File**: `svi_quartile_analysis_3Groups.py`
*   **Purpose**: To perform comprehensive SVI quartile analyses for 3-group comparisons.
*   **Outputs**: Various quartile-based analyses and visualizations.

---

## **Utility Scripts**

These scripts are located in `2_Scripts/utils/` and serve general purposes.

*   **File**: `update_csv.py`
    *   **Purpose**: A data management script used to merge or update the main CSV file from another CSV file. It matches rows based on patient identifiers.
    *   **Note**: You should not need to run this unless you are given a new, separate data file to merge into the main dataset.

*   **File**: `plot.py`
    *   **Purpose**: To generate a specific, multi-panel summary figure named `SVI_Onset_Treatment_Figure.png`.
    *   **Note**: This script appears to be for creating a custom figure for a presentation or paper, combining results from the onset and treatment delay analyses.

---

## **How to Run Scripts**

To run any analysis script:

1. Navigate to the appropriate directory in `2_Scripts/`
2. Ensure the data file is in the correct location (`1_Data/1_Raw/`)
3. Run the script with Python:
   ```bash
   python script_name.py
   ```
4. Check the corresponding directory in `3_Results/` for outputs

All scripts are designed to be self-contained and will generate their outputs in the appropriate `3_Results/` subdirectory.
