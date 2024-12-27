import os
import pandas as pd

def read_and_sort_csv(file_path, required_columns):
    """
    Reads a CSV file, ensures it contains the required columns, retains only these columns,
    converts the 'date' column to datetime, sets it as the index, and sorts the data by date
    in descending order (newest at index=0).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the CSV: {missing_columns}")
    
    # Retain only the required columns
    df = df[required_columns]
    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set 'date' as the DataFrame index
    df.set_index('date', inplace=True)
    
    # Sort the DataFrame by the datetime index in DESCENDING order (newest first)
    df.sort_index(ascending=False, inplace=True)
    
    # Remove duplicate indices if any
    df = df[~df.index.duplicated(keep='first')]
    
    return df
