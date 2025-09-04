# The Impact of Social Vulnerability on Clinical Outcomes in Pediatric JIA and Uveitis

---

## **Project Overview**

This project investigates the association between community-level social vulnerability and clinical outcomes in a cohort of pediatric patients with Juvenile Idiopathic Arthritis (JIA) and associated Uveitis. The primary goal is to determine if children living in more socially vulnerable communities experience a greater burden of disease, including a higher rate of vision-threatening complications, poorer quality of life, and worse visual acuity.

The core hypothesis is that higher social vulnerability, as measured by the CDC's Social Vulnerability Index (SVI), is associated with adverse clinical outcomes in this patient population.

---

## **Project Structure**

The project is organized into the following main directories:

### **Core Directories**

*   **`1_Data/`**: Contains the raw and processed datasets used in the analyses.
    *   `1_Raw/`: The original, untouched data files
    *   `2_Processed/`: Processed and cleaned data files with patient groupings

*   **`2_Scripts/`**: Contains all Python scripts for the project, organized by analysis type.
    *   `1_complications_analysis/`: Analysis of uveitis complications by SVI quartile
    *   `2_demographics_analysis/`: Demographic comparisons and Table 1 generation
    *   `3_jia_subtype_analysis/`: JIA subtype analysis by SVI quartiles
    *   `4_onset_analysis/`: Age of onset analysis for JIA and uveitis
    *   `5_qol_analysis/`: Quality of life measures analysis
    *   `6_steroid_duration_analysis/`: Steroid drops duration analysis
    *   `7_va_analysis/`: Visual acuity analysis
    *   `8_treatment_delay_analysis/`: Treatment delay and timing analysis
    *   `9_quartile_analysis/`: SVI quartile-based analyses
    *   `utils/`: Utility scripts for plotting and data updates

*   **`3_Results/`**: Contains all output files from the analysis scripts, organized by analysis type.
    *   Each sub-directory corresponds to the analysis scripts and contains figures, tables, and statistical outputs
    *   Results include demographic tables, complication analyses, QoL measures, visual acuity data, and more

*   **`4_Reports/`**: Contains final reports, presentations, and abstracts related to the project.

*   **`5_Documentation/`**: Contains detailed documentation for the project, including:
    *   Data dictionary
    *   Python scripts guide
    *   Figures and results interpretation
    *   Project analysis report

*   **`z_archive_and_noise/`**: Contains archived files and older versions of analyses that are not part of the main project.

### **Project Management**

*   **`tasks.md`**: Contains the specific data analyses and tables requested by mentors, including:
    *   SVI domains analysis for JIA groups
    *   Uveitis complications by SVI quartile
    *   Visual acuity analysis
    *   Steroid drops duration analysis
    *   Quality of life measures (EQ-5D, CHAQ)
    *   JIA severity analysis
    *   Summary tables and inclusion/exclusion criteria

---

## **Key Analyses Performed**

Based on the tasks outlined in `tasks.md`, this project includes:

1. **Demographic Analysis**: SVI domains comparison between JIA-no-U vs JIA-U groups
2. **Complications Analysis**: Uveitis complications (synechiae, band keratopathy, cystoid macular edema) by SVI quartile
3. **Visual Acuity Analysis**: Initial visit VA and SVI quartile relationships
4. **Treatment Analysis**: Steroid drops duration and treatment timing by SVI quartile
5. **Quality of Life Analysis**: EQ-5D and CHAQ scores vs SVI quartiles
6. **Disease Severity Analysis**: JIA severity at presentation vs SVI quartile
7. **Quartile Analysis**: Comprehensive SVI quartile-based analyses across all outcomes

---

## **How to Run the Analysis**

The analysis scripts are located in the `2_Scripts/` directory. Each analysis is self-contained in its respective folder.

To run an analysis, navigate to the script's directory and execute it using Python. For example, to run the Quality of Life analysis:

```bash
cd 2_Scripts/5_qol_analysis/
python svi_qol_analysis.py
```

The script will read the necessary data from the `1_Data/` directory and save its output (figures and tables) to the corresponding directory in `3_Results/`.

---

## **Data**

The primary dataset for this project is located at: `1_Data/1_Raw/SVI_Pankaj_data_1_updated_merged_new.csv`.

A processed version of the data with patient groups is also available: `1_Data/2_Processed/SVI_processed_data_with_groups.csv`.

For a detailed explanation of the data columns, please refer to the **[Data Dictionary](./5_Documentation/01_Data_Dictionary.md)**.

---

## **Results Summary**

The project has generated comprehensive analyses across multiple domains:

- **Demographics**: Detailed demographic comparisons by SVI quartiles and patient groups
- **Complications**: Analysis of uveitis complications stratified by social vulnerability
- **Quality of Life**: Multi-dimensional QoL analysis including CHAQ, PedsQL, and vision-specific measures
- **Treatment Patterns**: Analysis of treatment timing, duration, and patterns by SVI quartiles
- **Visual Outcomes**: Visual acuity analysis and correlation with social vulnerability
- **Disease Severity**: JIA severity measures and their relationship to social vulnerability

All results are available in the `3_Results/` directory, organized by analysis type with corresponding statistical summaries and visualizations.

---

## **Contact**

For questions about this project, please refer to the documentation in the `5_Documentation/` directory or contact the research team.
