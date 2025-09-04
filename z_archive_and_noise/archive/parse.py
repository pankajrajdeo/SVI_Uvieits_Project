import pandas as pd
import io

# Read the uploaded CSV data
# Need to handle potential bad lines and the complex header
# Let's try reading it, skipping bad lines if necessary
try:
    # Use the actual filename provided by the system
    df = pd.read_csv("Lab.xlsx - Lab.csv", on_bad_lines='skip')
except Exception as e:
    print(f"Error reading CSV: {e}")
    # If initial read fails, try specifying low_memory=False or other options
    try:
        df = pd.read_csv("Lab.xlsx - Lab.csv", on_bad_lines='skip', low_memory=False)
    except Exception as e2:
        print(f"Second attempt to read CSV failed: {e2}")
        df = pd.DataFrame() # Create empty DataFrame if reading fails

# Check if DataFrame is empty
if df.empty:
    print("Could not read the CSV file or the file is empty.")
else:
    # --- Identify the correct columns based on the header analysis ---
    # It seems the header might be duplicated or malformed. We'll select based on expected names.
    # Using names identified previously, including potential suffixes like '.1'

    # Check which columns actually exist in the loaded dataframe
    available_cols = df.columns.tolist()

    # Define desired columns based on previous analysis, checking for existence
    col_mapping = {
        'Global Subject ID': 'Global Subject ID',
        'Test Name': 'lab component name (list distinct)',
        'Unit': 'measure unit (list distinct)',
        'Reference Value (High)': 'reference_high (list distinct)',
        'Reference Value (Low)': 'reference_low (list distinct)',
        'Measured Value': 'measure value (list distinct).1', # Note the .1
        'LOINC Code': 'LOINC code (list distinct)' # Correcting user typo 'LIONIC'
    }

    # Filter the mapping to only include columns present in the DataFrame
    cols_to_use = {}
    missing_cols = []
    for target_name, source_name in col_mapping.items():
        if source_name in available_cols:
            cols_to_use[target_name] = source_name
        else:
            # Handle potential variations if exact name isn't found (e.g., without '.1')
            base_name = source_name.split(' (list distinct)')[0]
            found_alternative = False
            for col in available_cols:
                 # Check if a column starts with the base name and contains '(list distinct)'
                 # This is less precise but might catch variations like different suffixes
                if col.startswith(base_name) and '(list distinct)' in col:
                    # Check if this alternative is already assigned or is the original target
                    if col != source_name and col not in cols_to_use.values():
                         print(f"Warning: Using column '{col}' for '{target_name}' instead of expected '{source_name}'.")
                         cols_to_use[target_name] = col
                         found_alternative = True
                         break # Use the first alternative found
            if not found_alternative:
                 missing_cols.append(f"{target_name} (expected: {source_name})")


    if missing_cols:
        print(f"Error: The following expected columns were not found in the CSV: {', '.join(missing_cols)}")
        print(f"Available columns are: {available_cols}")
    elif 'Global Subject ID' not in cols_to_use:
         print(f"Error: The crucial 'Global Subject ID' column is missing.")
    else:
        # Select only the necessary columns using the identified source names
        df_subset = df[[cols_to_use[target] for target in cols_to_use]].copy()
        # Rename columns to the desired target names
        rename_dict = {v: k for k, v in cols_to_use.items()}
        df_subset.rename(columns=rename_dict, inplace=True)

        # --- Data Transformation ---
        # Convert columns (except ID) to string type to handle potential non-string values before splitting
        for col in df_subset.columns:
            if col != 'Global Subject ID':
                df_subset[col] = df_subset[col].astype(str)

        # Function to split strings and handle potential variations in list lengths
        def split_and_align(row):
            data = {}
            list_lengths = set()
            max_len = 0

            # Split each relevant column into a list
            for col in row.index:
                if col != 'Global Subject ID':
                    # Replace 'nan' strings resulting from astype(str) back to None or empty list
                    if row[col].lower() == 'nan':
                         data[col] = []
                    else:
                        data[col] = str(row[col]).split(';')
                    current_len = len(data[col])
                    list_lengths.add(current_len)
                    if current_len > max_len:
                        max_len = current_len

            # Check if all lists had the same length (ignoring empty lists if 'nan' was the only content)
            consistent_length = len(list_lengths) <= 1 or (len(list_lengths) == 2 and 0 in list_lengths)


            # Create records for each item, padding shorter lists with None
            records = []
            for i in range(max_len):
                record = {'Global Subject ID': row['Global Subject ID']}
                # Add a warning if lengths were inconsistent
                record['Data Consistency Warning'] = None if consistent_length else "Inconsistent number of items across fields for this subject"

                for col in data: # Iterate through the split data dictionary
                    record[col] = data[col][i] if i < len(data[col]) else None # Pad with None if index is out of bounds
                records.append(record)

            return records

        # Apply the function row-wise and collect all results
        all_records = []
        for index, row in df_subset.iterrows():
            all_records.extend(split_and_align(row))

        # Create the final DataFrame
        final_df = pd.DataFrame(all_records)

        # Reorder columns to the desired sequence + add warning column at the end
        desired_order = [
            'Global Subject ID',
            'Test Name',
            'Unit',
            'Reference Value (High)',
            'Reference Value (Low)',
            'Measured Value',
            'LOINC Code',
            'Data Consistency Warning' # Keep warning column for clarity
        ]
        # Filter final_df columns to only include those in desired_order that actually exist
        final_df_ordered = final_df[[col for col in desired_order if col in final_df.columns]]


        # Display the resulting table (or its head)
        print("Generated Table (first 50 rows):")
        print(final_df_ordered.head(50).to_markdown(index=False, numalign="left", stralign="left"))

        # Display table info
        print("\nTable Info:")
        final_df_ordered.info()

        # Save to CSV file
        output_csv_file = 'lab_results_parsed.csv'
        final_df_ordered.to_csv(output_csv_file, index=False)
        print(f"\nFull table saved to {output_csv_file}")