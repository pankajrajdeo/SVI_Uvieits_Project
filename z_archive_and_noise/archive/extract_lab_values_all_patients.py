import pandas as pd
import re
import sys

# --- Configuration ---
FILE_PATH = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1.csv"
OUTPUT_FILE_PATH = "all_patients_lab_values_output.txt"
PATIENT_ID_COL = 'Global Subject ID' # Updated based on user feedback

# Define Lab patterns (lowercase for case-insensitive matching)
LAB_PATTERNS = {
    'ANA': [r'ana', r'antinuclear'],
    'HLA-B27': [r'hla-?b27'], # Optional hyphen
    'ESR': [r'esr', r'sed rate', r'sedimentation rate'],
    'Vitamin D': [r'vitamin d', r'vit d', r'25-hydroxyvitamin', r'25\(oh\)', r'vitd']
}

# Define columns with parallel lists
PARALLEL_NAME_COLS = [
    'lab component name (list distinct)',
    'measure name (list distinct)'
]
PARALLEL_VALUE_COL = 'measure value (list distinct)'

# Define dedicated columns (and the lab they correspond to)
DEDICATED_COLS = {
    'ana_display_value (list distinct)': 'ANA'
}

# --- Helper Functions ---
def parse_semicolon_list(value):
    """Safely parses a potentially NaN or non-string value into a list of strings."""
    if pd.isna(value):
        return []
    try:
        # Split, strip whitespace, and filter out empty strings
        return [item.strip() for item in str(value).split(';') if item.strip()]
    except Exception:
        return [] # Return empty list on any parsing error

def find_pattern_in_list(item_list, patterns):
    """Checks if any pattern matches any item in the list (case-insensitive)."""
    if not isinstance(item_list, list):
        return False, None
    for item in item_list:
        if pd.isna(item): continue
        item_lower = str(item).lower()
        for pattern in patterns:
            if re.search(pattern, item_lower):
                return True, item # Return True and the matching item
    return False, None

def get_value_at_index(value_list, index):
    """Safely retrieves a value from a list at a specific index."""
    if isinstance(value_list, list) and 0 <= index < len(value_list):
        return value_list[index]
    return None # Return None if index is out of bounds or list is invalid

# --- Main Logic ---
def main():
    original_stdout = sys.stdout
    with open(OUTPUT_FILE_PATH, 'w') as f_out:
        sys.stdout = f_out # Redirect print to file
        
        print(f"--- Extracting Lab Values for All Patients ---")
        print(f"Data Source: {FILE_PATH}")
        print(f"Searching for: {', '.join(LAB_PATTERNS.keys())}")
        print(f"Primary Columns Searched (Name/Value pairs):")
        for name_col in PARALLEL_NAME_COLS:
            print(f"  - '{name_col}' / '{PARALLEL_VALUE_COL}'")
        print(f"Dedicated Columns Searched:")
        for col, lab in DEDICATED_COLS.items():
             print(f"  - '{col}' (for {lab})")
        print("-" * 40)

        results = [] # Store dictionaries: {'patient_id': pid, 'lab': lab_name, 'value': value, 'source': source_desc}

        try:
            df = pd.read_csv(FILE_PATH, low_memory=False)
            print(f"Loaded data. Shape: {df.shape}")

            if PATIENT_ID_COL not in df.columns:
                print(f"Error: Patient ID column '{PATIENT_ID_COL}' not found. Cannot proceed.")
                sys.stdout = original_stdout
                print(f"Error: Patient ID column '{PATIENT_ID_COL}' not found. Check script config.")
                return

            for index, row in df.iterrows():
                patient_id = row[PATIENT_ID_COL]
                found_in_row = set() # Track labs found for this patient to avoid duplicate reporting from different name columns

                # 1. Process Parallel List Columns
                if PARALLEL_VALUE_COL in row:
                    value_list = parse_semicolon_list(row[PARALLEL_VALUE_COL])
                    for name_col in PARALLEL_NAME_COLS:
                        if name_col in row:
                            name_list = parse_semicolon_list(row[name_col])
                            max_len = max(len(name_list), len(value_list)) # Iterate based on longer list

                            for i in range(max_len):
                                name = name_list[i] if i < len(name_list) else None
                                if pd.isna(name): continue
                                
                                name_lower = str(name).lower()
                                for lab_name, patterns in LAB_PATTERNS.items():
                                     # Check if this lab was already found via another name column for this patient
                                    #if lab_name in found_in_row: continue 
                                     
                                    # Check if name matches pattern
                                    matched_pattern = False
                                    for pattern in patterns:
                                        if re.search(pattern, name_lower):
                                            matched_pattern = True
                                            break
                                    
                                    if matched_pattern:
                                        value = get_value_at_index(value_list, i)
                                        if value is not None:
                                            source_desc = f"'{name_col}'[idx {i}] / '{PARALLEL_VALUE_COL}'[idx {i}]"
                                            results.append({PATIENT_ID_COL: patient_id, 'lab': lab_name, 'value': value, 'source': source_desc})
                                            found_in_row.add(lab_name + str(value)) # Add combo to allow multiple diff values per lab


                # 2. Process Dedicated Columns
                for col_name, lab_target in DEDICATED_COLS.items():
                    if col_name in row:
                        dedicated_values = parse_semicolon_list(row[col_name])
                        for val in dedicated_values:
                             # Check if this specific value was already reported from parallel lists
                            if lab_target + str(val) not in found_in_row:
                                results.append({PATIENT_ID_COL: patient_id, 'lab': lab_target, 'value': val, 'source': f"'{col_name}'"})
                                found_in_row.add(lab_target + str(val)) # Mark as found

            print(f"\n--- Found {len(results)} potential lab value entries ---")
            # Print results grouped by patient for slightly better readability
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                 # Sort for consistency
                results_df = results_df.sort_values(by=[PATIENT_ID_COL, 'lab', 'source']) 
                
                current_pid = None
                for _, res_row in results_df.iterrows():
                    pid = res_row[PATIENT_ID_COL]
                    if pid != current_pid:
                        if current_pid is not None: print("") # Add newline between patients
                        print(f"Patient ID: {pid}")
                        current_pid = pid
                    print(f"  Lab: {res_row['lab']:<10} | Value: {res_row['value']:<20} | Source: {res_row['source']}")
            else:
                 print("No matching lab values found in the specified columns.")


        except FileNotFoundError:
            print(f"Error: File not found at {FILE_PATH}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            print(traceback.format_exc())

    # Restore stdout and notify user
    sys.stdout = original_stdout
    print(f"Extraction complete. Output saved to: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main() 