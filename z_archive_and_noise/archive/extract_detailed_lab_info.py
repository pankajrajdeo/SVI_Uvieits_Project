import pandas as pd
import re
import sys
import csv

# --- Configuration ---
FILE_PATH = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1.csv"
OUTPUT_CSV_PATH = "detailed_lab_info_output.csv"
PATIENT_ID_COL = 'Global Subject ID'

# Define Lab patterns (lowercase for case-insensitive matching)
LAB_PATTERNS = {
    'ANA': [r'ana', r'antinuclear'],
    'HLA-B27': [r'hla-?b27'], # Optional hyphen
    'ESR': [r'esr', r'sed rate', r'sedimentation rate'],
    'Vitamin D': [r'vitamin d', r'vit d', r'25-hydroxyvitamin', r'25\(oh\)', r'vitd'] # Escaped parenthesis
}

# Define columns with parallel lists to extract
PARALLEL_NAME_COLS = [
    'lab component name (list distinct)',
    'measure name (list distinct)'
]
PARALLEL_DATA_COLS = {
    'value': 'measure value (list distinct)',
    'unit': 'measure unit (list distinct)',
    'ref_low': 'reference_low (list distinct)',
    'ref_high': 'reference_high (list distinct)',
    'date': 'result date (list distinct)'
}

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

def get_value_at_index(data_list, index):
    """Safely retrieves a value from a list at a specific index."""
    if isinstance(data_list, list) and 0 <= index < len(data_list):
        return data_list[index]
    return None # Return None if index is out of bounds or list is invalid

# --- Main Logic ---
def main():
    print(f"--- Extracting Detailed Lab Info ---")
    print(f"Data Source: {FILE_PATH}")
    print(f"Output CSV: {OUTPUT_CSV_PATH}")
    print(f"Searching for: {', '.join(LAB_PATTERNS.keys())}")

    results = [] # List of dictionaries to be written to CSV

    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
        print(f"Loaded data. Shape: {df.shape}")

        if PATIENT_ID_COL not in df.columns:
            print(f"Error: Patient ID column '{PATIENT_ID_COL}' not found. Cannot proceed.")
            return

        for index, row in df.iterrows():
            patient_id = row[PATIENT_ID_COL]
            found_lab_indices_in_row = {} # Track {lab_name: set(indices)} found via parallel lists

            # Pre-parse all relevant parallel lists for the row
            parsed_data_lists = {}
            for key, col_name in PARALLEL_DATA_COLS.items():
                 if col_name in row:
                    parsed_data_lists[key] = parse_semicolon_list(row[col_name])
                 else:
                    parsed_data_lists[key] = []


            # 1. Process Parallel List Columns
            for name_col in PARALLEL_NAME_COLS:
                if name_col in row:
                    name_list = parse_semicolon_list(row[name_col])

                    for i, name in enumerate(name_list):
                        if pd.isna(name): continue

                        name_lower = str(name).lower()
                        for lab_name, patterns in LAB_PATTERNS.items():
                            # Check if name matches pattern
                            matched_pattern = False
                            for pattern in patterns:
                                if re.search(pattern, name_lower):
                                    matched_pattern = True
                                    break

                            if matched_pattern:
                                # Track that we found this lab at this index from this name column
                                if lab_name not in found_lab_indices_in_row:
                                    found_lab_indices_in_row[lab_name] = set()
                                
                                # Avoid duplicate entries if the same lab name/index appears in multiple name columns
                                if i in found_lab_indices_in_row[lab_name]:
                                    continue 
                                found_lab_indices_in_row[lab_name].add(i)

                                # Extract corresponding data using the index 'i'
                                extracted_data = {
                                    'Patient ID': patient_id,
                                    'Lab Name': lab_name,
                                    'Source': f"{name_col}[{i}]" # Indicate source column and index
                                }
                                for key, data_list in parsed_data_lists.items():
                                    extracted_data[key.capitalize()] = get_value_at_index(data_list, i)

                                results.append(extracted_data)


            # 2. Process Dedicated Columns
            for col_name, lab_target in DEDICATED_COLS.items():
                if col_name in row:
                    dedicated_values = parse_semicolon_list(row[col_name])
                    for val in dedicated_values:
                        # For dedicated columns, we don't have parallel units/refs/dates
                         results.append({
                             'Patient ID': patient_id,
                             'Lab Name': lab_target,
                             'Value': val,
                             'Unit': None,
                             'Ref_low': None,
                             'Ref_high': None,
                             'Date': None,
                             'Source': col_name
                         })

        print(f"Found {len(results)} potential lab entries.")

        # Write results to CSV
        if results:
            # Define CSV header based on keys of the first result dictionary
            header = list(results[0].keys())
            # Ensure consistent order
            ordered_header = ['Patient ID', 'Lab Name', 'Value', 'Unit', 'Ref_low', 'Ref_high', 'Date', 'Source']
            # Filter header to only include keys actually present
            final_header = [h for h in ordered_header if h in header] 
            
            with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_header)
                writer.writeheader()
                writer.writerows(results)
            print(f"Successfully wrote detailed lab info to {OUTPUT_CSV_PATH}")
        else:
            print("No matching lab entries found to write to CSV.")


    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 