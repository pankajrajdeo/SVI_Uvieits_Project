import pandas as pd
import numpy as np
import re
import sys

# --- Configuration ---
INPUT_CSV_PATH = "detailed_lab_info_output.csv"
OUTPUT_CSV_PATH = "processed_lab_info.csv"
PATIENT_ID_COL = 'Patient ID'
LAB_NAME_COL = 'Lab Name'
VALUE_COL = 'Value'
# UNIT_COL = 'Unit' # No longer used for interpretation
# REF_LOW_COL = 'Ref_low' # No longer used for interpretation
# REF_HIGH_COL = 'Ref_high' # No longer used for interpretation
DATE_COL = 'Date'
SOURCE_COL = 'Source'

# Standard thresholds (adjust if needed based on clinical context)
ESR_HIGH_THRESHOLD = 20 # mm/hr (example threshold)
VITD_DEFICIENT_THRESHOLD = 20 # ng/mL (example threshold)
VITD_INSUFFICIENT_THRESHOLD = 30 # ng/mL (example threshold)

# Labs to process
TARGET_LABS = ['ESR', 'Vitamin D', 'ANA', 'HLA-B27']

# --- Helper Functions ---
def safe_to_numeric(series, errors='coerce'):
    """Convert series to numeric, coercing errors, handling potential lists/objects."""
    # Ensure input is treated as string first if not already numeric
    series_str = series.astype(str)
    # Attempt conversion
    return pd.to_numeric(series_str, errors=errors)

def interpret_value(lab_name, value_str):
    """Interprets a lab value based on lab name and value content."""
    if pd.isna(value_str):
        return "Unknown"
    
    val_lower = str(value_str).lower().strip()

    # 1. Check for common explicit keywords
    if lab_name in ['ANA', 'HLA-B27']:
        if 'positive' in val_lower or 'detected' in val_lower:
            return "Positive"
        if 'negative' in val_lower or 'not detected' in val_lower or 'non-reactive' in val_lower or 'neg' in val_lower:
            return "Negative"
        # Check for titer patterns (e.g., < 1:16, 1:80) - often implies negative/low positive
        if re.match(r'(<|>)?\s?1:\d+', val_lower):
             # Simple check: assume low titers might be considered Negative for initial pass
             # More nuanced interpretation might be needed
             try:
                 ratio = int(val_lower.split(':')[-1])
                 if ratio <= 16: # Example threshold for negativity based on titer
                      return "Negative (Titer)"
                 else:
                      return "Positive (Titer)"
             except:
                 pass # Fall through if parsing fails

    if lab_name == 'ESR':
        if 'high' in val_lower or 'elevated' in val_lower:
            return "High (Text)"
        if 'normal' in val_lower or 'wnl' in val_lower:
            return "Normal (Text)"

    if lab_name == 'Vitamin D':
        if 'low' in val_lower or 'deficient' in val_lower:
            return "Low/Deficient (Text)"
        if 'insufficient' in val_lower:
            return "Insufficient (Text)"
        if 'normal' in val_lower or 'sufficient' in val_lower:
            return "Sufficient (Text)"

    # 2. Attempt numeric interpretation if no keyword match
    numeric_val = pd.to_numeric(value_str, errors='coerce')

    if pd.isna(numeric_val):
        # If it's not numeric and didn't match keywords, it's likely noise
        return "Ambiguous/Non-numeric"

    # Apply numeric thresholds
    if lab_name == 'ESR':
        # Basic validity check - ESR shouldn't be negative
        if numeric_val < 0:
             return "Ambiguous/Non-numeric" 
        if numeric_val > ESR_HIGH_THRESHOLD:
            return "High (Numeric)"
        else:
            return "Normal (Numeric)"

    if lab_name == 'Vitamin D':
         # Basic validity check - Vit D shouldn't be negative
        if numeric_val < 0:
             return "Ambiguous/Non-numeric"
        # Assume ng/mL - add check/conversion if units become reliable later
        if numeric_val < VITD_DEFICIENT_THRESHOLD:
            return "Deficient (Numeric)"
        elif numeric_val < VITD_INSUFFICIENT_THRESHOLD:
            return "Insufficient (Numeric)"
        else:
            return "Sufficient (Numeric)"

    # If numeric but not ESR or Vit D (e.g., ANA/HLA-B27 got a number)
    return "Ambiguous/Numeric"

# --- Main Logic ---
def main():
    print(f"--- Processing Detailed Lab Info ---")
    print(f"Input CSV: {INPUT_CSV_PATH}")
    print(f"Output CSV: {OUTPUT_CSV_PATH}")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Loaded data. Shape: {df.shape}")
        if df.empty:
            print("Input CSV is empty. Exiting.")
            return

        # --- Phase 1: Analysis (Keep for reference, but not used for interpretation) ---
        # ... (Analysis code omitted for brevity, as it's no longer driving interpretation) ...
        print("\n--- Phase 1: Analysis Complete (Results not used for interpretation) ---")

        # --- Phase 2: Apply Interpretation Logic ---
        print("\n--- Phase 2: Applying Interpretation Logic ---")
        
        df['Interpretation'] = df.apply(
            lambda row: interpret_value(row[LAB_NAME_COL], row[VALUE_COL]),
            axis=1
        )
        
        print("Interpretation applied. Value Counts:")
        print(df.groupby(LAB_NAME_COL)['Interpretation'].value_counts())

        # --- Output ---
        # Select and order columns for output
        output_cols = [PATIENT_ID_COL, LAB_NAME_COL, VALUE_COL, 'Interpretation', DATE_COL, SOURCE_COL]
        # Keep only columns that actually exist in the dataframe
        output_cols_present = [col for col in output_cols if col in df.columns]

        df_processed = df[output_cols_present]

        df_processed.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"\nProcessed data with interpretations saved to {OUTPUT_CSV_PATH}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 