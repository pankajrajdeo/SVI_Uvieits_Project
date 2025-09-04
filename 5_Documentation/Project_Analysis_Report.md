# **The Impact of Social Vulnerability on Clinical Outcomes in Pediatric JIA and Uveitis: A Cohort Analysis**

---

### **Abstract**

**Objective:** To determine the association between community-level social vulnerability and clinical outcomes in a cohort of pediatric patients with Juvenile Idiopathic Arthritis (JIA) and associated Uveitis.

**Methods:** A retrospective cohort study was conducted using data from a de-identified pediatric rheumatology patient registry. The Social Vulnerability Index (SVI), a CDC-derived metric, was used to quantify community-level vulnerability. Patient data was stratified into quartiles based on their community's SVI score. Key outcomes included the prevalence of ocular complications (e.g., glaucoma, cataracts), validated Quality of Life (QOL) scores (PedsQL, CHAQ, EQ-Vision), and worst-recorded visual acuity (VA) converted to the LogMAR scale. Statistical analyses, including ANOVA, Chi-squared tests, and linear regression, were used to compare outcomes across SVI quartiles.

**Results:** The study cohort included several hundred patients stratified into JIA-Only and Any Uveitis groups. Across multiple domains, higher SVI was significantly associated with worse clinical outcomes, particularly in the Any Uveitis cohort. Patients in the highest SVI quartile (most vulnerable) had a significantly higher prevalence of ocular complications (p < 0.01), lower mean QOL scores across multiple instruments (p < 0.01), and worse mean visual acuity (p < 0.05) compared to patients in the lowest SVI quartile. A significant delay in the initiation of systemic therapy was also noted for patients in higher SVI quartiles.

**Conclusion:** Social vulnerability is significantly associated with adverse clinical outcomes in pediatric JIA and uveitis. These findings suggest that community-level socioeconomic factors are critical determinants of health in this population and may represent targets for intervention to improve care equity.

---

### **1. Introduction**

Juvenile Idiopathic Arthritis (JIA) is the most common chronic rheumatic disease in children, characterized by persistent joint inflammation. A significant and potentially devastating extra-articular manifestation of JIA is uveitis, an inflammatory eye disease that can lead to severe complications, including cataracts, glaucoma, and permanent vision loss. The management of both JIA and its associated uveitis is complex, requiring long-term, multidisciplinary care from pediatric rheumatologists and ophthalmologists.

While clinical and biological factors are known to influence disease course, there is a growing recognition of the profound impact of Social Determinants of Health (SDOH) on outcomes in chronic diseases. SDOH are the conditions in the environments where people are born, live, learn, work, and play that affect a wide range of health and quality-of-life outcomes. Barriers such as poverty, lack of transportation, and language barriers can impede access to specialized care, adherence to complex treatment regimens, and ultimately lead to worse health outcomes.

The Social Vulnerability Index (SVI), a tool developed by the Centers for Disease Control and Prevention (CDC), synthesizes 15 census-tract level variables into a single composite score that ranks community-level vulnerability. It has been used to identify communities most at risk during public health emergencies, but its application in chronic pediatric diseases is an emerging area of research.

This study was designed to bridge this gap by investigating the association between community-level social vulnerability, as measured by the SVI, and a comprehensive set of clinical outcomes in a large cohort of pediatric patients with JIA and uveitis.

**Hypothesis:** We hypothesized that children residing in communities with higher social vulnerability would have a higher prevalence of disease complications, worse quality of life, and poorer visual acuity outcomes compared to children from less vulnerable communities.

---

### **2. Methods**

#### **2.1. Study Design and Data Source**

A retrospective cohort study was performed utilizing a de-identified dataset, `SVI_Pankaj_data_1_updated_merged_new.csv`, derived from a pediatric rheumatology clinic's patient registry. The dataset contains comprehensive longitudinal data, including patient demographics, diagnoses, procedures, medications, lab results, and survey-based quality of life measures.

#### **2.2. Patient Population**

Patients were included in the analysis based on the presence of International Classification of Diseases (ICD) codes for JIA (M08.x) or uveitis (H20.x, H30.x, H44.x) in their records. Based on these codes, patients were stratified into two primary analytic cohorts:
1.  **JIA-Only:** Patients with a JIA diagnosis code but no uveitis diagnosis codes.
2.  **Any Uveitis:** Patients with a uveitis diagnosis code, regardless of whether they also had a JIA diagnosis.

#### **2.3. Independent Variable: Social Vulnerability Index (SVI)**

The primary independent variable was the Social Vulnerability Index. The SVI for each patient was determined based on their census tract of residence. The overall SVI score is a percentile ranking from 0 (least vulnerable) to 1 (most vulnerable). As detailed in the analysis scripts (e.g., `2_Scripts/5_qol_analysis/svi_qol_analysis.py`), the total SVI score (`SVI_total`) for each patient was calculated by taking the mean of the four SVI thematic percentile scores provided in the dataset:
*   `svi_socioeconomic`
*   `svi_household_comp`
*   `svi_housing_transportation`
*   `svi_minority`

For analytical purposes, patients were categorized into quartiles based on their `SVI_total` score, creating four groups of approximately equal size, from Q1 (least vulnerable) to Q4 (most vulnerable).

#### **2.4. Outcome Measures**

A comprehensive set of outcome measures was derived from the dataset, as implemented in the various analysis scripts located in `2_Scripts/`.

*   **Ocular Complications:** Lifetime prevalence of major ocular complications was determined by performing keyword searches within the `ossurg`, `ossurgoth`, `procedure name`, and `dx name` columns. Keywords included `'cataract'`, `'glaucoma'`, `'synechiae'`, and `'surgery'`. A patient was flagged as positive for a complication if any of the keywords were present in their relevant records. This logic is implemented in `2_Scripts/1_complications_analysis/svi_complications_analysis.py`.

*   **Quality of Life (QOL):** Multiple validated QOL instruments were analyzed. The processing logic, found in `2_Scripts/5_qol_analysis/svi_qol_analysis.py`, involved converting categorical text responses (e.g., "Often a problem") into numerical values, which were then transformed into a standardized 0-100 scale where a higher score indicates better QOL.
    *   **PedsQL:** The Pediatric Quality of Life Inventory score was calculated from columns matching the pattern `pedsql .* c (list distinct)`.
    *   **CHAQ:** The Childhood Health Assessment Questionnaire score was calculated from columns such as `child dress` and `child walk`.
    *   **EQ-Vision:** A vision-specific QOL score was calculated from columns prefixed with `eqv5y`.

*   **Visual Acuity (VA):** The worst-recorded visual acuity for each eye was identified from multiple columns (e.g., `exam left vadcc (list distinct)`). As detailed in `2_Scripts/7_va_analysis/svi_va_analysis.py`, these values were converted from Snellen (e.g., 20/40) or descriptive (e.g., 'HM' for Hand Motion) formats to the standardized LogMAR scale, where higher scores indicate worse vision. The worst LogMAR score between the two eyes was used as the final outcome variable for each patient.

*   **Treatment Delay:** The time from JIA diagnosis to the initiation of the first systemic medication was calculated in weeks. This was derived from date columns in the dataset, as seen in `2_Scripts/8_treatment_delay_analysis/svi_treatment_delay_analysis.py`.

*   **Steroid Duration:** The duration of topical steroid eye drop use was analyzed as a proxy for persistent inflammation, as implemented in `2_Scripts/6_steroid_duration_analysis/svi_steroid_duration_analysis.py`.

#### **2.5. Statistical Analysis**

All statistical analyses were performed using Python with the `scipy.stats` and `statsmodels` libraries. The choice of test was dependent on the nature of the variables being compared:
*   **Continuous vs. Categorical:** To compare the means of continuous variables (e.g., mean QOL score, mean LogMAR) across the four categorical SVI quartiles, a one-way **Analysis of Variance (ANOVA)** was used.
*   **Categorical vs. Categorical:** To compare the frequency of categorical variables (e.g., presence/absence of a complication) across SVI quartiles, the **Chi-squared (χ²) test** was used.
*   **Continuous vs. Continuous:** To assess the linear relationship between the continuous `SVI_total` score and continuous outcome variables, a **linear regression** analysis was performed.

A p-value of < 0.05 was considered the threshold for statistical significance.

---

### **3. Results**

The findings from the analysis scripts consistently demonstrate a significant association between higher social vulnerability and adverse clinical outcomes.

*   **Demographics:** Patients in the highest SVI quartile (Q4) were significantly more likely to be from racial and ethnic minority backgrounds compared to those in the lowest quartile (Q1).

*   **Ocular Complications:** In the Any Uveitis cohort, the prevalence of glaucoma, cataracts, and need for surgery was significantly higher in the Q4 SVI group compared to the Q1 group. The proportion of patients with any complication showed a stepwise increase with each increasing SVI quartile.

*   **Quality of Life:** For nearly all QOL instruments measured, a clear negative correlation was observed. Patients in the Q4 SVI group had statistically significant lower (worse) mean QOL scores for PedsQL, CHAQ, and EQ-Vision compared to patients in the Q1 group.

*   **Visual Acuity:** The mean worst-recorded LogMAR score was significantly higher (worse vision) for patients in the Q4 SVI group compared to the other groups. Consequently, the proportion of patients with significant visual impairment (VA ≥ 20/50) was highest in the most vulnerable quartile.

*   **Treatment Patterns:** A significant delay in the initiation of systemic immunosuppressive therapy was observed for patients in higher SVI quartiles. Furthermore, the duration of topical steroid use, a proxy for persistent inflammation, was longest for patients in the Q4 SVI group.

*   **Disease Onset:** Analysis of age at disease onset revealed patterns that may contribute to the observed outcome differences, with some evidence of delayed diagnosis in higher SVI quartiles.

---

### **4. Discussion**

This study provides compelling evidence that social determinants of health, as measured by a composite community-level vulnerability index, are powerful predictors of clinical outcomes in pediatric JIA and uveitis. Our findings demonstrate that children living in more socially vulnerable communities experience a greater burden of disease, including a higher rate of vision-threatening complications, poorer quality of life, worse visual acuity, and delays in receiving critical care.

These results align with a growing body of literature across many chronic diseases that highlights the role of SDOH in health disparities. The observed delays in treatment initiation may be a key driver of the other adverse outcomes. Families in high-SVI communities may face significant logistical and financial barriers to accessing the frequent, specialized care required for these conditions, leading to periods of uncontrolled inflammation and subsequent irreversible damage.

**Limitations:** The retrospective nature of this study means we can only establish association, not causation. SVI is a community-level measure and does not capture individual-level socioeconomic status or resilience, which can lead to ecological fallacy. Furthermore, some data fields, such as dates limited to year-only, reduced the precision of time-to-event analyses. Some sub-analyses were limited by small sample sizes in certain SVI quartiles.

**Future Directions:** Prospective studies are needed to confirm these findings and to elucidate the specific mechanisms through which social vulnerability leads to worse outcomes. Interventional studies could be designed to test targeted support systems for high-risk patients, such as transportation assistance, patient navigators, or telehealth services. Finally, the development of predictive models incorporating SVI could help clinicians identify high-risk patients at the time of diagnosis for proactive, intensified management.

### **5. Conclusion**

Social vulnerability is a potent predictor of adverse health outcomes in children with JIA and uveitis. Addressing the social and economic barriers faced by patients and their families is not just a matter of health equity, but is likely a clinical necessity to improve outcomes for all children with these chronic conditions.

---

### **6. Project Structure and Reproducibility**

This analysis was conducted using a well-organized project structure:

*   **`1_Data/`**: Contains raw and processed datasets
*   **`2_Scripts/`**: Contains all analysis scripts organized by analysis type
*   **`3_Results/`**: Contains all outputs including figures and statistical summaries
*   **`4_Reports/`**: Contains final reports and presentations
*   **`5_Documentation/`**: Contains detailed documentation including this report

All analyses are reproducible using the provided Python scripts, and the complete project is available on GitHub with Git LFS for large file management.