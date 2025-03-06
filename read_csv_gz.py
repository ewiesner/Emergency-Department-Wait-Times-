import pandas as pd

def read_csv_gz(file_path):
    """
    Reads a single .csv.gz file and returns a DataFrame.
    
    Parameters:
        file_path (str): The path to the .csv.gz file.
    
    Returns:
        DataFrame: The DataFrame created from the .csv.gz file.
    """
    return pd.read_csv(file_path, compression='gzip')