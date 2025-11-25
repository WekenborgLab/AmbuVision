#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os


file_path = 'artifacts/visual_factors/llm_categorization.csv'
output_file = 'artifacts/visual_factors/removed_none.csv'


if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
else:
    try:

        df = pd.read_csv(file_path)
        

        initial_rows = len(df)
        print(f"Initial number of rows loaded: {initial_rows}")
        
    
        df_filtered = df.dropna(subset=['Category'])
        
        df_filtered = df_filtered[~df_filtered['Category'].astype(str).str.strip().str.lower().isin(['none'])]
        
        df_filtered = df_filtered[df_filtered['Category'].astype(str).str.strip() != '']
        
 
        final_rows = len(df_filtered)
        print(f"Number of rows remaining after filtering 'None'/'NaN'/'Empty' categories: {final_rows}")
        print(f"Number of rows filtered out: {initial_rows - final_rows}")
        

        df_filtered.to_csv(output_file, index=False)
        print(f"Filtered data saved to '{output_file}'")

    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except KeyError:
        print("Error: The column 'Category' was not found in the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
