# Uveitis and Steroid Eye Drop Treatment Analysis

## Overview
This report examines the relationship between Social Vulnerability Index (SVI) and the duration of topical steroid eye drop treatment specifically in patients with uveitis. Treatment duration was calculated based on the earliest start year and latest end year recorded.

**Limitations Acknowledged:**
- Steroid identification relies on keyword matching and may not be perfectly accurate.
- Duration calculation uses year-level data and assumes continuous treatment, potentially over/underestimating true duration.
- Analysis does not fully control for disease severity due to data limitations.

## Key Findings

- **Total patients in dataset**: 495
- **Patients with uveitis**: 125 (25.3%)
- **Uveitis patients on steroid eye drops**: 125 (100.0%)
- **Patients included in duration analysis**: 125

- **Uveitis patients with active inflammation**: 117 (93.6%)

### Treatment Duration Statistics for Uveitis Patients (Years)
|       |   duration_years |
|:------|-----------------:|
| count |        125       |
| mean  |          5.208   |
| std   |          3.72718 |
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
| Q3             |      33 | 4.69697 |        5 |   2.66323 |
| Q4 (High)      |      34 | 4.61765 |        3 |   3.33044 |

### Statistical Analysis
- **ANOVA (Duration vs SVI Quartile)**: F=1.43, p=0.2370 (Not Significant)
- **Linear Regression (Duration vs SVI Total)**:
  - Slope: -4.4578
  - R-squared: 0.0644
  - P-value: 0.0043 (Significant)

### Prevalence of Steroid Eye Drop Use in Uveitis Patients by SVI Quartile
| SVI_quartile   |   Total Uveitis Patients |   Patients on Steroid Drops |   Percentage (%) |
|:---------------|-------------------------:|----------------------------:|-----------------:|
| Q1 (Low)       |                       57 |                          57 |              100 |
| Q2             |                        1 |                           1 |              100 |
| Q3             |                       33 |                          33 |              100 |
| Q4 (High)      |                       34 |                          34 |              100 |

### Steroid Treatment Duration by Uveitis Activity Status
| active_any   |   count |    mean |   median |      std |
|:-------------|--------:|--------:|---------:|---------:|
| Inactive     |       8 | 2.625   |        3 | 0.744024 |
| Active       |     117 | 5.38462 |        5 | 3.78483  |

- **T-test (Active vs Inactive Uveitis)**: t=6.30, p=0.0000 (Significant)

## Interpretation
Acknowledging the methodological limitations, the analysis of steroid treatment in uveitis patients suggests: 
- A statistically significant relationship between SVI and treatment duration in uveitis patients. Higher SVI is associated with **shorter** steroid treatment durations. This may indicate issues with access, adherence, or follow-up for uveitis patients from more vulnerable areas, but could also reflect other clinical factors.

- There is a significant difference in steroid treatment duration between patients with active versus inactive uveitis. Patients with active uveitis have longer treatment durations, which may reflect appropriate clinical care for persistent inflammation.

These findings should be interpreted cautiously due to data constraints and the difficulty in fully accounting for disease severity and other clinical factors.