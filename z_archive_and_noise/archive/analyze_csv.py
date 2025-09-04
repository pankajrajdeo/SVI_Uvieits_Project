import pandas as pd
import sys

# Get the file path from command line argument or use a default
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    # Default path if no argument is provided
    file_path = "/Users/rajlq7/Desktop/SVI/SVI_Pankaj_data_1.csv" 

print(f"Analyzing CSV file: {file_path}")

try:
    # Attempt to read the CSV file
    # Using low_memory=False can help with mixed types but uses more memory
    df = pd.read_csv(file_path, low_memory=False)
    
    print("\nSuccessfully loaded CSV.")
    print(f"Shape: {df.shape}")
    
    print("\nColumns:")
    print(df.columns.tolist())
    
    print("\nData Types (inferred by pandas):")
    print(df.dtypes)
    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
    print("The file might be too large to load into memory directly, or there might be parsing issues.")
    print("Consider using chunking or sampling techniques for very large files.") 