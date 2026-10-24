import re
import pandas as pd

import numpy as np
import os
import csv
import json
import time

import re

from tkinter import filedialog

def reinflate():
    # Open a file dialog to select the CSV file
    file_path_unique = filedialog.askopenfilename(title="Select the unique output CSV file", filetypes=[("CSV files", "*.csv")])
    

    if not file_path_unique:
        print("No file selected.")
        return
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path_unique)

    file_path_input = filedialog.askopenfilename(title="Select the input CSV file", filetypes=[("CSV files", "*.csv")])

    if not file_path_input:
        print("No file selected.")
        return

    input_df = pd.read_csv(file_path_input)

    combined_df = pd.merge(input_df, df, left_on='655 - Local Param 04', right_on='original', how='left')   
  
    # # Check if the 'inflated' column exists
    # if 'inflated' not in df.columns:
    #     print("The 'inflated' column is missing in the CSV file.")
    #     return
    
    # Reinflate the values in the 'inflated' column
    # df['inflated'] = df['inflated'].apply(lambda x: x * 1.1)  # Example inflation factor
    
    # Save the reinflated DataFrame back to a new CSV file
    output_file_path = os.path.splitext(file_path_input)[0] + '_reinflated.csv'
    combined_df.to_csv(output_file_path, index=False)
    
    print(f"Reinflated data saved to {output_file_path}")   

reinflate()