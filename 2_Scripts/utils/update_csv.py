#!/usr/bin/env python3
import pandas as pd
import os
import numpy as np

# File paths
first_file = '/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged.csv'
second_file = '/Users/rajlq7/Desktop/SVI/results_2025-05-07.csv'
output_file = '/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated_merged_new.csv'

# Read the CSV files
print(f"Reading first file: {first_file}")
df1 = pd.read_csv(first_file, low_memory=False)
print(f"Reading second file: {second_file}")
df2 = pd.read_csv(second_file, low_memory=False)

print(f"First file shape: {df1.shape}")
print(f"Second file shape: {df2.shape}")

# Display the first few rows of each dataframe to understand structure
print("\nSample data from first file:")
print(df1.iloc[:1].T)
print("\nSample data from second file:")
print(df2.iloc[:1].T)

# Convert column names to lowercase for case-insensitive matching
df1_cols_lower = {col: col.lower() for col in df1.columns}
df2_cols_lower = {col: col.lower() for col in df2.columns}

# Check if key columns exist
key_cols = ['global subject id', 'locus study id', 'redcap repeat instance']
print("\nChecking for key columns:")
for key in key_cols:
    print(f"  '{key}' in first file: {key in df1_cols_lower.values()}")
    print(f"  '{key}' in second file: {key in df2_cols_lower.values()}")

# Identify matching columns (case insensitive)
print("\nIdentifying matching columns...")
matching_columns = []
for col2 in df2.columns:
    col2_lower = col2.lower()
    if col2_lower in df1_cols_lower.values():
        # Find the original column name in df1
        original_col1 = [orig_col for orig_col, lower_col in df1_cols_lower.items() if lower_col == col2_lower][0]
        matching_columns.append((original_col1, col2))

print(f"Found {len(matching_columns)} matching columns between the files")

# Print matching columns for verification
print("Matching columns:")
for col1, col2 in matching_columns:
    print(f"  {col1} <-> {col2}")

# Get the actual column names for the key fields
global_id_col1 = None
global_id_col2 = None
locus_id_col1 = None
locus_id_col2 = None
redcap_col1 = None
redcap_col2 = None

if 'global subject id' in df1_cols_lower.values():
    global_id_col1 = [col for col, lower_col in df1_cols_lower.items() if lower_col == 'global subject id'][0]
if 'global subject id' in df2_cols_lower.values():
    global_id_col2 = [col for col, lower_col in df2_cols_lower.items() if lower_col == 'global subject id'][0]
if 'locus study id' in df1_cols_lower.values():
    locus_id_col1 = [col for col, lower_col in df1_cols_lower.items() if lower_col == 'locus study id'][0]
if 'locus study id' in df2_cols_lower.values():
    locus_id_col2 = [col for col, lower_col in df2_cols_lower.items() if lower_col == 'locus study id'][0]
if 'redcap repeat instance' in df1_cols_lower.values():
    redcap_col1 = [col for col, lower_col in df1_cols_lower.items() if lower_col == 'redcap repeat instance'][0]
if 'redcap repeat instance (list distinct)' in df2_cols_lower.values():
    redcap_col2 = [col for col, lower_col in df2_cols_lower.items() if lower_col == 'redcap repeat instance (list distinct)'][0]

print("\nActual key column names:")
print(f"  Global Subject ID: {global_id_col1} <-> {global_id_col2}")
print(f"  LOCUS Study ID: {locus_id_col1} <-> {locus_id_col2}")
print(f"  Redcap Repeat Instance: {redcap_col1} <-> {redcap_col2}")

# Print sample values to verify format
print("\nSample values from key columns:")
if global_id_col1 and global_id_col2:
    print(f"  Global Subject ID in df1: {df1[global_id_col1].iloc[:5].tolist()}")
    print(f"  Global Subject ID in df2: {df2[global_id_col2].iloc[:5].tolist()}")
if locus_id_col1 and locus_id_col2:
    print(f"  LOCUS Study ID in df1: {df1[locus_id_col1].iloc[:5].tolist()}")
    print(f"  LOCUS Study ID in df2: {df2[locus_id_col2].iloc[:5].tolist()}")
if redcap_col1 and redcap_col2:
    print(f"  Redcap Repeat Instance in df1: {df1[redcap_col1].iloc[:5].tolist()}")
    print(f"  Redcap Repeat Instance in df2: {df2[redcap_col2].iloc[:5].tolist()}")

# Get datatypes of each column in df1 to ensure proper conversion
df1_dtypes = {col: df1[col].dtype for col in df1.columns}
print("\nDatatype samples:")
for col, dtype in list(df1_dtypes.items())[:5]:
    print(f"  {col}: {dtype}")

# Identify date columns based on name patterns
date_column_patterns = ['date', 'dob', 'birth']
date_columns = [col for col in df1.columns if any(pattern in col.lower() for pattern in date_column_patterns)]
print(f"\nIdentified potential date columns: {len(date_columns)}")
for i, col in enumerate(date_columns[:10]):
    print(f"  {i+1}. {col}: {df1_dtypes.get(col)}")

# Function to find matching row in df2 for a row in df1
def find_matching_row(row, debug=False):
    # Try to match by "Global Subject ID"
    global_id_match = None
    locus_id_match = None
    
    if debug:
        print(f"\nDebugging row with Global ID: {row.get(global_id_col1, 'N/A')}, LOCUS ID: {row.get(locus_id_col1, 'N/A')}")
    
    # Check for Global Subject ID match (case insensitive)
    if global_id_col1 and global_id_col2:
        if pd.notna(row[global_id_col1]) and row[global_id_col1]:
            row_global_id = str(row[global_id_col1]).lower()
            global_id_matches = df2[df2[global_id_col2].astype(str).str.lower() == row_global_id]
            
            if debug:
                print(f"  Global ID '{row_global_id}' matches: {len(global_id_matches)} rows")
                
            if len(global_id_matches) > 0:
                # Further filter by redcap repeat instance if available
                if redcap_col1 and redcap_col2:
                    if pd.notna(row[redcap_col1]) and row[redcap_col1]:
                        row_redcap = str(row[redcap_col1])
                        if debug:
                            print(f"  Redcap instance to match: {row_redcap}")
                            
                        # Check if the redcap repeat instance in df1 is in the list in df2
                        instance_matches = global_id_matches[global_id_matches[redcap_col2].apply(
                            lambda x: row_redcap in str(x).split('; ') if pd.notna(x) else False
                        )]
                        
                        if debug:
                            print(f"  Redcap instance matches: {len(instance_matches)} rows")
                            if len(instance_matches) > 0:
                                print(f"  Matching redcap values: {instance_matches[redcap_col2].iloc[0]}")
                                
                        if len(instance_matches) > 0:
                            return instance_matches.iloc[0]
                
                # If no redcap instance match or not available, return the global ID match
                global_id_match = global_id_matches.iloc[0]
    
    # Check for LOCUS Study ID match (case insensitive)
    if locus_id_col1 and locus_id_col2:
        if pd.notna(row[locus_id_col1]) and row[locus_id_col1]:
            row_locus_id = str(row[locus_id_col1]).lower()
            locus_id_matches = df2[df2[locus_id_col2].astype(str).str.lower() == row_locus_id]
            
            if debug:
                print(f"  LOCUS ID '{row_locus_id}' matches: {len(locus_id_matches)} rows")
                
            if len(locus_id_matches) > 0:
                # Further filter by redcap repeat instance if available
                if redcap_col1 and redcap_col2:
                    if pd.notna(row[redcap_col1]) and row[redcap_col1]:
                        row_redcap = str(row[redcap_col1])
                        if debug:
                            print(f"  Redcap instance to match: {row_redcap}")
                            
                        # Check if the redcap repeat instance in df1 is in the list in df2
                        instance_matches = locus_id_matches[locus_id_matches[redcap_col2].apply(
                            lambda x: row_redcap in str(x).split('; ') if pd.notna(x) else False
                        )]
                        
                        if debug:
                            print(f"  Redcap instance matches: {len(instance_matches)} rows")
                            if len(instance_matches) > 0:
                                print(f"  Matching redcap values: {instance_matches[redcap_col2].iloc[0]}")
                                
                        if len(instance_matches) > 0:
                            return instance_matches.iloc[0]
                
                # If no redcap instance match or not available, return the LOCUS ID match
                locus_id_match = locus_id_matches.iloc[0]
    
    # Return the best match found (prioritize Global Subject ID if both exist)
    return global_id_match if global_id_match is not None else locus_id_match

# Function to convert value to appropriate type for the column
def convert_value(value, target_col):
    target_dtype = df1_dtypes.get(target_col)
    
    if pd.isna(value):
        return value
    
    # Special handling for date columns
    if any(pattern in target_col.lower() for pattern in date_column_patterns):
        # If it's a date column, keep it as string
        return str(value)
    
    # Handle data type conversions
    if pd.api.types.is_float_dtype(target_dtype):
        try:
            # For float columns, try to convert to float
            return float(value)
        except (ValueError, TypeError):
            # If conversion fails, return as-is
            return value
    elif pd.api.types.is_integer_dtype(target_dtype):
        try:
            # For integer columns, try to convert to int
            return int(float(value))
        except (ValueError, TypeError):
            # If conversion fails, return as-is
            return value
    else:
        # For all other types (string, object, etc.), return as-is
        return value

# Debug a few rows to see the matching process
print("\nDebugging match process for first 5 rows:")
for idx, row in df1.iloc[:5].iterrows():
    match = find_matching_row(row, debug=True)
    if match is not None:
        print(f"  Row {idx}: MATCH FOUND")
    else:
        print(f"  Row {idx}: NO MATCH")

# Make a copy of df1 to avoid in-place modification warnings
df1_updated = df1.copy()

# Update the data in df1_updated using data from df2
print("\nUpdating data from matching rows...")
updated_count = 0
updated_columns = set()

for idx, row in df1.iterrows():
    matching_row = find_matching_row(row)
    if matching_row is not None:
        for col1, col2 in matching_columns:
            if pd.notna(matching_row[col2]):
                # Convert the value to the appropriate type for df1
                converted_value = convert_value(matching_row[col2], col1)
                df1_updated.at[idx, col1] = converted_value
                updated_columns.add(col1)
        updated_count += 1
    
    # Display progress every 100 rows
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} rows, updated {updated_count} rows")

print(f"Total rows processed: {len(df1)}, updated {updated_count} rows")
print(f"Updated columns: {', '.join(sorted(updated_columns))}")

# Save the updated dataframe to a new CSV file
print(f"Saving updated data to: {output_file}")
df1_updated.to_csv(output_file, index=False)
print("Done!") 