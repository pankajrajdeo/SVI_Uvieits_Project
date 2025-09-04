import pandas as pd
import numpy as np

# File paths
main_file = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1.csv"
eye_drops_file = "/Users/rajlq7/Desktop/SVI/Filtered_Results_Dataset.csv"
medications_file = "/Users/rajlq7/Desktop/SVI/results_2025-05-01.csv"
output_file = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1_updated.csv"

print("Reading main file...")
main_df = pd.read_csv(main_file, low_memory=False)
print(f"Main file shape: {main_df.shape}")

print("Reading eye drops file...")
eye_drops_df = pd.read_csv(eye_drops_file, low_memory=False)
print(f"Eye drops file shape: {eye_drops_df.shape}")

print("Reading medications file...")
medications_df = pd.read_csv(medications_file, low_memory=False)
print(f"Medications file shape: {medications_df.shape}")

# Make column names lowercase for case-insensitive matching
# Create mapping dictionaries for easier reference
main_cols_lower = {col.lower(): col for col in main_df.columns}
eye_drops_cols_lower = {col.lower(): col for col in eye_drops_df.columns}
medications_cols_lower = {col.lower(): col for col in medications_df.columns}

# Find the matching columns in each file
main_id_cols = []
eye_drops_id_cols = []
medications_id_cols = []

# For Global Subject ID
if 'global subject id' in main_cols_lower and 'global subject id' in eye_drops_cols_lower:
    main_id_cols.append(main_cols_lower['global subject id'])
    eye_drops_id_cols.append(eye_drops_cols_lower['global subject id'])
elif 'global subject id' in main_cols_lower and 'global subject id_x' in eye_drops_cols_lower:
    main_id_cols.append(main_cols_lower['global subject id'])
    eye_drops_id_cols.append(eye_drops_cols_lower['global subject id_x'])

if 'global subject id' in main_cols_lower and 'global subject id' in medications_cols_lower:
    if not main_id_cols:  # Only add if not already added
        main_id_cols.append(main_cols_lower['global subject id'])
    medications_id_cols.append(medications_cols_lower['global subject id'])

# For LOCUS study ID
if 'locus study id' in main_cols_lower and 'locus study id' in eye_drops_cols_lower:
    main_id_cols.append(main_cols_lower['locus study id'])
    eye_drops_id_cols.append(eye_drops_cols_lower['locus study id'])
elif 'locus study id' in main_cols_lower and 'locus study id_x' in eye_drops_cols_lower:
    main_id_cols.append(main_cols_lower['locus study id'])
    eye_drops_id_cols.append(eye_drops_cols_lower['locus study id_x'])

if 'locus study id' in main_cols_lower and 'locus study id' in medications_cols_lower:
    if len(main_id_cols) < 2:  # Only add if not already added
        main_id_cols.append(main_cols_lower['locus study id'])
    medications_id_cols.append(medications_cols_lower['locus study id'])

# For redcap repeat instance
main_repeat_col = main_cols_lower.get('redcap repeat instance')
eye_drops_repeat_col = eye_drops_cols_lower.get('redcap repeat instance')
medications_repeat_col = medications_cols_lower.get('redcap repeat instance')

print(f"Main ID columns: {main_id_cols}")
print(f"Main repeat column: {main_repeat_col}")
print(f"Eye drops ID columns: {eye_drops_id_cols}")
print(f"Eye drops repeat column: {eye_drops_repeat_col}")
print(f"Medications ID columns: {medications_id_cols}")
print(f"Medications repeat column: {medications_repeat_col}")

# Columns to replace from eye drops file
eye_drop_start_col_main = main_cols_lower.get('eye drop start date (list distinct)')
eye_drop_end_col_main = main_cols_lower.get('eye drop end date (list distinct)')
eye_drop_start_col_source = eye_drops_cols_lower.get('eye drop start date (list distinct)')
eye_drop_end_col_source = eye_drops_cols_lower.get('eye drop end date (list distinct)')

# Columns to replace from medications file
med_start_col_main = main_cols_lower.get('medication start date (list distinct)')
med_end_col_main = main_cols_lower.get('medication end date (list distinct)')
cmtrt_col_main = main_cols_lower.get('cmtrt (list distinct)')
med_start_col_source = medications_cols_lower.get('medication start date (list distinct)')
med_end_col_source = medications_cols_lower.get('medication end date (list distinct)')
cmtrt_col_source = medications_cols_lower.get('cmtrt (list distinct)')

print("\nColumns to replace:")
print(f"Main eye drop start: {eye_drop_start_col_main}, Source: {eye_drop_start_col_source}")
print(f"Main eye drop end: {eye_drop_end_col_main}, Source: {eye_drop_end_col_source}")
print(f"Main medication start: {med_start_col_main}, Source: {med_start_col_source}")
print(f"Main medication end: {med_end_col_main}, Source: {med_end_col_source}")
print(f"Main cmtrt: {cmtrt_col_main}, Source: {cmtrt_col_source}")

# Create new columns to hold updated values
main_df['eye_drop_start_updated'] = main_df[eye_drop_start_col_main].copy() if eye_drop_start_col_main else None
main_df['eye_drop_end_updated'] = main_df[eye_drop_end_col_main].copy() if eye_drop_end_col_main else None
main_df['medication_start_updated'] = main_df[med_start_col_main].copy() if med_start_col_main else None
main_df['medication_end_updated'] = main_df[med_end_col_main].copy() if med_end_col_main else None
main_df['cmtrt_updated'] = main_df[cmtrt_col_main].copy() if cmtrt_col_main else None

# Function to match and update values based on two possible ID columns
def match_and_update(main_df, source_df, main_id_cols, source_id_cols, main_repeat_col, source_repeat_col, 
                     main_target_cols, source_cols, updated_cols):
    updated_count = 0
    
    # Convert ID columns to string to ensure matching works correctly
    for col in main_id_cols:
        if col in main_df.columns:
            main_df[col] = main_df[col].astype(str)
    
    for col in source_id_cols:
        if col in source_df.columns:
            source_df[col] = source_df[col].astype(str)
    
    if main_repeat_col:
        main_df[main_repeat_col] = main_df[main_repeat_col].astype(str)
    
    if source_repeat_col:
        source_df[source_repeat_col] = source_df[source_repeat_col].astype(str)
    
    # Iterate through each row in the main dataframe
    for idx, main_row in main_df.iterrows():
        # Try to find a match in the source dataframe
        matched = False
        
        for main_id_col, source_id_col in zip(main_id_cols, source_id_cols):
            if main_id_col in main_df.columns and source_id_col in source_df.columns:
                main_id = main_row[main_id_col]
                
                # Filter source dataframe based on the ID
                filtered_source = source_df[source_df[source_id_col].str.lower() == main_id.lower()]
                
                # Further filter by redcap repeat instance if available
                if main_repeat_col and source_repeat_col and main_repeat_col in main_df.columns and source_repeat_col in source_df.columns:
                    if pd.notna(main_row[main_repeat_col]):
                        filtered_source = filtered_source[filtered_source[source_repeat_col].str.lower() == main_row[main_repeat_col].lower()]
                
                if not filtered_source.empty:
                    matched = True
                    # Update the values
                    for main_target_col, source_col, updated_col in zip(main_target_cols, source_cols, updated_cols):
                        if main_target_col and source_col and updated_col:
                            if source_col in filtered_source.columns:
                                # Use the first matching row's value
                                new_value = filtered_source.iloc[0][source_col]
                                if pd.notna(new_value):
                                    main_df.at[idx, updated_col] = new_value
                                    updated_count += 1
                    # Break after finding a match
                    break
    
    return updated_count

# Update eye drop data
print("\nUpdating eye drop data...")
eye_drop_updated = match_and_update(
    main_df, eye_drops_df, 
    main_id_cols, eye_drops_id_cols, 
    main_repeat_col, eye_drops_repeat_col,
    [eye_drop_start_col_main, eye_drop_end_col_main],
    [eye_drop_start_col_source, eye_drop_end_col_source],
    ['eye_drop_start_updated', 'eye_drop_end_updated']
)
print(f"Updated {eye_drop_updated} eye drop values")

# Update medication data
print("Updating medication data...")
medication_updated = match_and_update(
    main_df, medications_df, 
    main_id_cols, medications_id_cols, 
    main_repeat_col, medications_repeat_col,
    [med_start_col_main, med_end_col_main, cmtrt_col_main],
    [med_start_col_source, med_end_col_source, cmtrt_col_source],
    ['medication_start_updated', 'medication_end_updated', 'cmtrt_updated']
)
print(f"Updated {medication_updated} medication values")

# Replace the original columns with the updated values
if eye_drop_start_col_main:
    main_df[eye_drop_start_col_main] = main_df['eye_drop_start_updated']
if eye_drop_end_col_main:
    main_df[eye_drop_end_col_main] = main_df['eye_drop_end_updated']
if med_start_col_main:
    main_df[med_start_col_main] = main_df['medication_start_updated']
if med_end_col_main:
    main_df[med_end_col_main] = main_df['medication_end_updated']
if cmtrt_col_main:
    main_df[cmtrt_col_main] = main_df['cmtrt_updated']

# Drop the temporary columns
main_df = main_df.drop(columns=[col for col in ['eye_drop_start_updated', 'eye_drop_end_updated', 
                                                'medication_start_updated', 'medication_end_updated', 
                                                'cmtrt_updated'] 
                                 if col in main_df.columns])

# Save the updated dataframe
print(f"\nSaving updated data to {output_file}...")
main_df.to_csv(output_file, index=False)
print("Done!") 