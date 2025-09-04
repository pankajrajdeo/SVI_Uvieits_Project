import pandas as pd
import re
import numpy as np

# --- Configuration (match generate_table1_comparison.py) ---
FILE_PATH = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1.csv"
PATIENT_ID_COL = 'Global Subject ID'
DX_CODE_COL = 'dx code (list distinct)'

# Diagnosis Code Patterns (match generate_table1_comparison.py)
JIA_CODE_PATTERN = r'M08' 
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']

# Optional: Other columns to display for context
CONTEXT_COLS = ['uveitis onset year (first)', 'jia subtype '] # Add more if needed

# --- Helper Function (copied from generate_table1_comparison.py) ---
def check_codes(dx_string):
    """Assigns diagnosis group based on JIA and Uveitis codes."""
    if pd.isna(dx_string):
        return 'Unknown'
    codes = [code.strip() for code in str(dx_string).split(';')]
    has_jia = any(re.match(JIA_CODE_PATTERN, code) for code in codes)
    has_uveitis = any(any(re.match(uv_pattern, code) for code in codes) for uv_pattern in UVEITIS_CODE_PATTERNS)

    if has_jia and has_uveitis:
        return 'JIA-U'
    elif has_jia and not has_uveitis:
        return 'JIA-Only'
    else: # Group Uveitis-Only and Other/Unknown together as 'Other'
        return 'Other'

# --- Main Logic ---
def main():
    print(f"--- Finding Excluded Patients ---")
    print(f"Loading data from: {FILE_PATH}")

    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
        print(f"Loaded data. Shape: {df.shape}")

        if DX_CODE_COL not in df.columns or PATIENT_ID_COL not in df.columns:
            print(f"Error: Required columns '{DX_CODE_COL}' or '{PATIENT_ID_COL}' not found.")
            return

        # Apply the grouping logic
        df['temp_diagnosis_group'] = df[DX_CODE_COL].apply(check_codes)

        # Filter for the excluded group
        df_excluded = df[df['temp_diagnosis_group'] == 'Other'].copy()

        num_excluded = len(df_excluded)
        print(f"\nFound {num_excluded} patients assigned to 'Other' group (excluded from JIA-Only/JIA-U comparison).")

        if num_excluded > 0:
            print("Details of excluded patients:")
            # Select columns to display
            display_cols = [PATIENT_ID_COL, DX_CODE_COL]
            for col in CONTEXT_COLS:
                if col in df_excluded.columns:
                    display_cols.append(col)
                else:
                    print(f"(Context column '{col}' not found in data)")
            
            # Print details
            # Use to_string to avoid truncation
            print(df_excluded[display_cols].to_string(index=False))
        else:
            print("No patients were assigned to the 'Other' group.")

    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 