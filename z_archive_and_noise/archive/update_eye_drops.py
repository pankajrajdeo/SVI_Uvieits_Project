#!/usr/bin/env python3
import pandas as pd
import numpy as np

def update_eye_drop_dates():
    print("Reading the datasets...")
    
    # Read the files
    uveitis_file = "/Users/rajlq7/Desktop/SVI/all_uveitis.csv"
    filtered_file = "/Users/rajlq7/Desktop/SVI/Filtered_Results_Dataset.csv"
    
    uveitis_df = pd.read_csv(uveitis_file)
    filtered_df = pd.read_csv(filtered_file)
    
    print(f"Uveitis file has {len(uveitis_df)} rows and {len(uveitis_df.columns)} columns")
    print(f"Filtered results file has {len(filtered_df)} rows and {len(filtered_df.columns)} columns")
    
    # Normalize column names for matching
    filtered_df.columns = [col.lower().strip() for col in filtered_df.columns]
    
    # Identify key columns for matching
    locus_id_cols_filtered = [col for col in filtered_df.columns if 'locus' in col and 'study' in col and 'id' in col]
    global_id_cols_filtered = [col for col in filtered_df.columns if 'global' in col and 'subject' in col and 'id' in col]
    
    print(f"Found these LOCUS ID columns in filtered file: {locus_id_cols_filtered}")
    print(f"Found these Global Subject ID columns in filtered file: {global_id_cols_filtered}")
    
    # Convert column types to facilitate matching
    for col in locus_id_cols_filtered:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].astype(str).replace('nan', '')
    
    for col in global_id_cols_filtered:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].astype(str).replace('nan', '')
    
    # Make a copy of the original data
    updated_df = uveitis_df.copy()
    
    # Track updates
    updates_made = 0
    patients_updated = set()
    
    # Define target columns
    start_date_col = 'eye drop start date (list distinct)'
    end_date_col = 'eye drop end date (list distinct)'
    
    # For each row in uveitis_df, try to find a match in filtered_df
    for idx, row in uveitis_df.iterrows():
        if idx % 10 == 0:
            print(f"Processing row {idx}...")
            
        # Get values for matching
        locus_id = str(row['LOCUS study ID']) if not pd.isna(row['LOCUS study ID']) else ''
        global_id = str(row['Global Subject ID']) if not pd.isna(row['Global Subject ID']) else ''
        redcap_instance = row['redcap repeat instance'] if not pd.isna(row['redcap repeat instance']) else None
        
        if not locus_id and not global_id:
            continue
            
        # Try to find match based on LOCUS study ID and redcap repeat instance
        matches = []
        if locus_id:
            for locus_col in locus_id_cols_filtered:
                if locus_col in filtered_df.columns:
                    instance_matches = filtered_df[
                        (filtered_df[locus_col] == locus_id) & 
                        (filtered_df['redcap repeat instance'] == redcap_instance)
                    ]
                    if not instance_matches.empty:
                        matches.append(instance_matches)
        
        # If no match found by LOCUS ID, try Global Subject ID
        if not matches and global_id:
            for global_col in global_id_cols_filtered:
                if global_col in filtered_df.columns:
                    instance_matches = filtered_df[
                        (filtered_df[global_col] == global_id) & 
                        (filtered_df['redcap repeat instance'] == redcap_instance)
                    ]
                    if not instance_matches.empty:
                        matches.append(instance_matches)
        
        # If matches found, update eye drop dates
        if matches:
            match_df = pd.concat(matches).drop_duplicates()
            
            # Make sure our target columns exist
            if 'eye drop start date (list distinct)' in match_df.columns and 'eye drop end date (list distinct)' in match_df.columns:
                start_date = match_df['eye drop start date (list distinct)'].iloc[0]
                end_date = match_df['eye drop end date (list distinct)'].iloc[0]
                
                if not pd.isna(start_date) or not pd.isna(end_date):
                    # Update the values in updated_df
                    if not pd.isna(start_date):
                        updated_df.at[idx, 'eye drop start date (list distinct)'] = start_date
                    
                    if not pd.isna(end_date):
                        updated_df.at[idx, 'eye drop end date (list distinct)'] = end_date
                    
                    updates_made += 1
                    patients_updated.add(locus_id if locus_id else global_id)
                    
                    print(f"Updated dates for patient {locus_id or global_id}")
    
    # Save the updated data
    output_file = "/Users/rajlq7/Desktop/SVI/all_uveitis_updated.csv"
    updated_df.to_csv(output_file, index=False)
    
    print(f"\nUpdate complete!")
    print(f"Made {updates_made} updates for {len(patients_updated)} unique patients")
    print(f"Updated data saved to {output_file}")

if __name__ == "__main__":
    update_eye_drop_dates() 