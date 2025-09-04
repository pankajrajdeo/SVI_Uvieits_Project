# SVI and Steroid Eye Drop Treatment Analysis (Refined Methodology)

## Overview
This report examines the relationship between Social Vulnerability Index (SVI) and the duration of *presumed* topical steroid eye drop treatment. Steroid use was identified using keywords. Treatment duration was calculated based on the earliest start year and latest end year recorded.

**Limitations Acknowledged:**
- Steroid identification relies on keyword matching and may not be perfectly accurate.
- Duration calculation uses year-level data and assumes continuous treatment, potentially over/underestimating true duration.
- Analysis does not control for disease severity due to data limitations.

## Key Findings

- **Total patients**: 495
- **Patients presumed on steroid eye drops (Keyword Method)**: 157 (31.7%)
- **Patients included in duration analysis**: 133

### Treatment Duration Statistics (Years)
|       |   duration_years |
|:------|-----------------:|
| count |        133       |
| mean  |          4.89474 |
| std   |          3.82033 |
| min   |          0       |
| 25%   |          2       |
| 50%   |          3       |
| 75%   |          9       |
| max   |         11       |

### Treatment Duration by SVI Quartile
| SVI_quartile   |   count |    mean |   median |       std |
|:---------------|--------:|--------:|---------:|----------:|
| Q1 (Low)       |      57 | 5.91228 |        6 |   4.37232 |
| Q2             |       1 | 2       |        2 | nan       |
| Q3             |      41 | 3.78049 |        3 |   3.03737 |
| Q4 (High)      |      34 | 4.61765 |        3 |   3.33044 |

### Statistical Analysis
- **ANOVA (Duration vs SVI Quartile)**: F=2.88, p=0.0385 (Significant)
- **Linear Regression (Duration vs SVI Total)**:
  - Slope: -5.2328
  - R-squared: 0.0811
  - P-value: 0.0009 (Significant)

### Prevalence of Steroid Eye Drop Use by SVI Quartile
| SVI_quartile   |   Total Patients |   Patients on Steroid Drops |   Percentage (%) |
|:---------------|-----------------:|----------------------------:|-----------------:|
| Q1 (Low)       |              125 |                          59 |             47.2 |
| Q2             |              123 |                          14 |             11.4 |
| Q3             |              126 |                          46 |             36.5 |
| Q4 (High)      |              121 |                          38 |             31.4 |

## Interpretation
Acknowledging the methodological limitations, the analysis suggests: 
- A statistically significant relationship between SVI and treatment duration. Higher SVI is associated with **shorter** treatment durations. This *could* indicate issues with access, adherence, or follow-up, but might also reflect appropriate clinical adaptation or unmeasured confounding (like disease severity).

These findings should be interpreted cautiously due to data constraints.