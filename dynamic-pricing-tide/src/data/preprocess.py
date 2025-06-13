import pandas as pd
import os

def load_data(file_path):
    """Load dataset from a specified file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset."""
    # Example preprocessing steps
    df.dropna(inplace=True)  # Remove missing values
    df.reset_index(drop=True, inplace=True)  # Reset index
    # Add more preprocessing steps as needed
    return df

def save_processed_data(df, output_path):
    """Save the processed dataset to the specified output path."""
    df.to_csv(output_path, index=False)