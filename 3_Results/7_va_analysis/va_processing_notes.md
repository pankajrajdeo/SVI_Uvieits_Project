# Visual Acuity Processing Notes for SVI Analysis

This document outlines the methods used to process the raw visual acuity (VA) data from the `exam left vadcc (list distinct)` (Column 539) and `exam right vadcc (list distinct)` (Column 556) columns in the `SVI_Pankaj_data_1_updated_merged.csv` file for the SVI vs. Visual Acuity analysis.

## Raw Data Format

The raw data in these columns consists of semicolon-separated strings, where each part represents a VA measurement recorded at potentially different time points for a patient. Examples:
- `"20/20; 20/25; 20/30"`
- `"20/100; Other; 20/80"`
- `"CF; HM; 20/400"`

## Processing Steps

1.  **Parsing:** Each string is split by the semicolon (`;`) into individual VA records.
2.  **Extraction:** For each record, valid VA notations are extracted. This includes:
    *   Snellen fractions (e.g., `20/20`, `20/400`)
    *   Count Fingers (`CF`)
    *   Hand Motion (`HM`)
    *   Light Perception (`LP`)
    *   No Light Perception (`NLP`)
    *   Pin Hole (`PH`) variants (e.g., `20/50 PH`) - the PH indication itself is noted but doesn't change the LogMAR value of the Snellen fraction itself for this analysis.
3.  **Normalization:** Text entries are converted to uppercase and stripped of whitespace.
4.  **LogMAR Conversion:** Each valid VA notation is converted to its corresponding LogMAR (Logarithm of the Minimum Angle of Resolution) value using the following conventions based on standard ophthalmic practice:
    *   **Snellen:** `LogMAR = log10(Denominator / Numerator)`. For example, `20/40` -> `log10(40/20) = log10(2) = 0.3`.
        *   Values like `20/20-1`, `20/40+2` are treated as `20/20` and `20/40` respectively.
    *   **Count Fingers (CF):** Assigned `1.9` LogMAR (equivalent to approx. 20/1600).
    *   **Hand Motion (HM):** Assigned `2.3` LogMAR (equivalent to approx. 20/4000).
    *   **Light Perception (LP):** Assigned `2.7` LogMAR.
    *   **No Light Perception (NLP):** Assigned `3.0` LogMAR (represents the upper limit).
    *   **Invalid/Unparseable:** Entries like "Other", "CSM", empty strings, or purely alphabetical strings that don't match the above are ignored (treated as missing data for that specific record).
5.  **Worst VA per Eye:** For each eye (left/right), the *maximum* LogMAR value found among all valid records in the string is taken as the worst recorded VA for that eye.
6.  **Worst Overall VA:** The patient's overall worst VA is the *maximum* LogMAR value between the worst left eye LogMAR and the worst right eye LogMAR. If one eye has no valid VA data, the worst VA from the other eye is used. If both eyes lack valid data, the patient has missing VA data for this analysis.

## Binary Outcome: "Worse VA"

For the primary analysis, a binary outcome variable (`Worse_VA_20_50`) is created based on the overall worst LogMAR:
*   **Worse VA (True/1):** If `Worst_Overall_LogMAR >= 0.4` (This corresponds to 20/50 or worse vision).
*   **Not Worse VA (False/0):** If `Worst_Overall_LogMAR < 0.4`.

## Assumptions & Limitations

*   The LogMAR values assigned to CF, HM, LP, NLP are standard approximations.
*   The analysis assumes the 'vadcc' columns represent best *corrected* distance acuity.
*   Entries marked "Other", "CSM", or uninterpretable strings are excluded from the LogMAR calculation for that record, potentially leading to missing data if no other valid VA is present in the string.
*   The temporal aspect of the VA measurements within the string is ignored; only the single worst value is used.
*   Pin Hole improvements are not factored into the LogMAR value itself, though their presence could be analyzed separately. 