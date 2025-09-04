import pandas as pd
import sys
import csv
import re
from collections import Counter # Import Counter for de-duplication

def _deduplicate_columns(columns):
    """Appends .1, .2 etc to duplicate column names."""
    counts = Counter()
    new_columns = []
    for col in columns:
        counts[col] += 1
        if counts[col] > 1:
            new_columns.append(f"{col}.{counts[col]-1}")
        else:
            new_columns.append(col)
    if len(new_columns) != len(set(new_columns)):
         print("Warning: De-duplication might not have fully resolved unique names. Check header.")
    return new_columns

def filter_uveitis_patients(input_file, output_file):
    """Filters the dataset to include only patients with indicators of uveitis."""
    try:
        # Step 1: Read header and deduplicate column names
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            original_header = next(reader)
            print(f"Read header with {len(original_header)} columns.")
            deduplicated_header = _deduplicate_columns(original_header)
            if original_header != deduplicated_header:
                print("Duplicate columns found and handled.")

        # Step 2: Read the full CSV using the deduplicated header names
        print(f"Reading full dataset from {input_file}...")
        df = pd.read_csv(
            input_file,
            low_memory=False,
            header=0,             # Still skip the original header row in the file
            names=deduplicated_header # Use the unique names
            # Removed mangle_dupe_cols=True as it's handled manually now
        )
        print(f"Original dataset shape: {df.shape}")

        # Step 3: Define filtering conditions for Uveitis
        # Use ORIGINAL header names for defining conditions, as pandas uses the unique ones
        # (Need to map original -> deduplicated if duplicates existed in filter columns, but these look unique)
        
        # Condition 1: Direct diagnosis flag
        mask1 = pd.Series([False] * len(df)) # Default to False
        if 'diagnosis of uveitis' in df.columns:
             # Ensure boolean comparison, treat NaN as False
            mask1 = df['diagnosis of uveitis'].fillna(False).astype(bool) == True
        else:
            print("Warning: Column 'diagnosis of uveitis' not found.")

        # Condition 2: Presence of diagnosis date
        mask2 = pd.Series([False] * len(df))
        if 'date of uveitis diagnosis' in df.columns:
            mask2 = df['date of uveitis diagnosis'].notna()
        else:
            print("Warning: Column 'date of uveitis diagnosis' not found.")
        
        # Condition 3: Presence of uveitis type
        mask3 = pd.Series([False] * len(df))
        if 'uv typedx' in df.columns:
             # Check for non-null and non-empty string
            mask3 = df['uv typedx'].notna() & (df['uv typedx'].astype(str).str.strip() != '')
        else:
            print("Warning: Column 'uv typedx' not found.")

        # Condition 4: Presence of uveitis location
        mask4 = pd.Series([False] * len(df))
        if 'uveitis location ' in df.columns: # Note the trailing space in the header
            mask4 = df['uveitis location '].notna() & (df['uveitis location '].astype(str).str.strip() != '')
        else:
             print("Warning: Column 'uveitis location ' not found.")

        # Condition 5: Eyes involved specified
        mask5 = pd.Series([False] * len(df))
        if 'which eyes are involved ' in df.columns: # Note the trailing space
             mask5 = df['which eyes are involved '].notna() & (df['which eyes are involved '].astype(str).str.strip() != '')
        else:
             print("Warning: Column 'which eyes are involved ' not found.")

        # Condition 6: Presence of onset date
        mask6 = pd.Series([False] * len(df))
        if 'date of uveitis onset' in df.columns:
             mask6 = df['date of uveitis onset'].notna()
        else:
             print("Warning: Column 'date of uveitis onset' not found.")

        # Condition 7: Diagnosis name contains uveitis terms (case-insensitive)
        mask7 = pd.Series([False] * len(df))
        uveitis_terms = ['uveitis', 'iridocyclitis', 'pars planitis', 'panuveitis']
        if 'dx name (list distinct)' in df.columns:
            pattern = '|'.join(uveitis_terms)
            mask7 = df['dx name (list distinct)'].fillna('').str.contains(pattern, case=False, na=False)
        else:
             print("Warning: Column 'dx name (list distinct)' not found.")
        
        # Condition 8: Diagnosis code matches uveitis patterns (e.g., H20.*, H30.*)
        mask8 = pd.Series([False] * len(df))
        if 'dx code (list distinct)' in df.columns:
            # Regex to match codes starting with H20 or H30
            mask8 = df['dx code (list distinct)'].fillna('').str.contains(r'H20|H30', case=False, na=False, regex=True)
        else:
            print("Warning: Column 'dx code (list distinct)' not found.")

        # Condition 9: Status columns indicate current/previous diagnosis
        mask9 = pd.Series([False] * len(df))
        status_cols = ['uveitis curr', 'prev uvei diag', 'uveitis curr fup', 'uvei diag fup '] # Note space in last one
        for col in status_cols:
            if col in df.columns:
                mask9 |= (df[col].fillna(False).astype(bool) == True)
            else:
                print(f"Warning: Status column '{col}' not found.")

        # Combine all conditions with OR
        combined_mask = mask1 | mask2 | mask3 | mask4 | mask5 | mask6 | mask7 | mask8 | mask9

        # Step 4: Apply the filter
        df_filtered = df[combined_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        print(f"Filtered dataset shape: {df_filtered.shape}")

        # Step 5: Save the filtered data
        df_filtered.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Filtered dataset saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_uveitis.py <input_csv_path> <output_csv_path>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    filter_uveitis_patients(input_path, output_path) 