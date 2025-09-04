import pandas as pd
import re
import sys

# --- Configuration ---
FILE_PATH = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1.csv"

# Columns potentially containing the lab names or results
# Based on the analysis script and user input
COLUMNS_TO_SEARCH = [
    'lab component name (list distinct)',
    'measure name (list distinct)',
    'measure value (list distinct)',
    'ana_display_value (list distinct)'
]

# --- ADDED: Config for Diagnosis Grouping (from main script) ---
DX_CODE_COL = 'dx code (list distinct)'
JIA_CODE_PATTERN = r'M08'
UVEITIS_CODE_PATTERNS = [r'H20', r'H30', r'H44']
# --- END ADDED ---

# Regex pattern to find relevant terms (case-insensitive)
# Looks for HLA-B27 (with optional hyphen), ANA, Antinuclear, positive, negative, detected
SEARCH_PATTERN = r'(HLA-?B27|ANA|Antinuclear|positive|negative|detected)'

# --- ADDED: Diagnosis Grouping Function (from main script) ---
def check_codes(dx_string):
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
# --- END ADDED ---

def main():
    print(f"--- Searching for Lab Terms in {FILE_PATH} ---")
    print(f"Searching columns: {', '.join(COLUMNS_TO_SEARCH)}")
    print(f"Using regex pattern: {SEARCH_PATTERN} (case-insensitive)")

    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
        print(f"Successfully loaded CSV. Shape: {df.shape}")
        
        # --- ADDED: Calculate diagnosis group ---
        if DX_CODE_COL in df.columns:
            df['diagnosis_group'] = df[DX_CODE_COL].apply(check_codes)
            print("Calculated diagnosis groups.")
        else:
            print(f"Error: Diagnosis code column '{DX_CODE_COL}' not found. Cannot filter by group.")
            sys.exit(1)
        # --- END ADDED ---

    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV or calculating groups: {e}")
        sys.exit(1)

    found_matches = False
    matched_rows = set() # Keep track of rows already reported
    rows_in_target_groups = 0

    print("\n--- Matches Found (Only showing rows in JIA-Only or JIA-U groups) ---")

    for col_name in COLUMNS_TO_SEARCH:
        if col_name in df.columns:
            print(f"\nSearching column: '{col_name}'...")
            col_found = False
            # Iterate through rows, checking for the pattern
            for index, row_data in df.iterrows(): # Use iterrows to access diagnosis_group
                # --- ADDED: Filter by diagnosis group ---
                if row_data['diagnosis_group'] not in ['JIA-Only', 'JIA-U']:
                    continue # Skip rows not in the target groups
                # --- END ADDED ---
                
                value = row_data[col_name]
                # Ensure value is treated as string, handle NaN/None
                value_str = str(value)
                # Search for pattern case-insensitively
                match = re.search(SEARCH_PATTERN, value_str, re.IGNORECASE)
                if match:
                    if index not in matched_rows: # Only print first match per row for brevity
                         print(f"  Row {index} (Group: {row_data['diagnosis_group']}): Found term '{match.group(0)}' in '{col_name}': '{value_str}'")
                         matched_rows.add(index)
                         col_found = True
                         found_matches = True
                         rows_in_target_groups += 1
            if not col_found:
                 print(f"  No matches found in this column for target groups.")

        else:
            print(f"\nWarning: Column '{col_name}' not found in the CSV.")

    print("\n--- Search Summary ---")
    if found_matches:
        # Note: len(matched_rows) will count unique rows across all columns searched
        print(f"Found matches in a total of {len(matched_rows)} unique rows belonging to JIA-Only or JIA-U groups.")
        print("Review the output above.")
    else:
        print("No matches found for the specified patterns in the searched columns within the JIA-Only or JIA-U groups.")

    print("\n--- Search Complete ---")

if __name__ == "__main__":
    main() 