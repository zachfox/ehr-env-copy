import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from prefect import flow, task
from data.pm25_cher import download_pmdata_from_database

from prefect import flow, task

def convert_date_to_time(date):
    """Convert date string to a datetime ordinal representation."""
    return datetime.strptime(date, "%Y-%m-%d").toordinal()

@task
def load_and_preprocess_data(file_path, min_lat, max_lat, min_lon, max_lon):
    """Load data and process Lat, Lon, Elevation..."""
    data = pd.read_csv(file_path)
    data = data[
        (data["Lat"] > min_lat)
        & (data["Lat"] < max_lat)
        & (data["Lon"] > min_lon)
        & (data["Lon"] < max_lon)
    ]
    data["Date"] = data["Date"].apply(convert_date_to_time)
    data["Elevation"] = data["Elevation"].values
    data.loc[data["Elevation"] < 0, "Elevation"] = 0
    return data

@task
def normalize_positions(data):
    """Normalize the latitude, longitude, and elevation positions."""
    positions = data[["Lat", "Lon", "Elevation"]].values
    # Normalize positions
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)
    normalized_positions = (positions - min_vals) / (max_vals - min_vals)
    # Create a DataFrame with separate columns for "Lat", "Lon" and "Elevation"
    normalized_df = pd.DataFrame(
        normalized_positions, columns=["Lat", "Lon", "Elevation"]
    )
    return normalized_df

@flow(log_prints=True)
def process_and_save_data(data_dir):
    """Main function to process, split and save result."""
    file_path = download_pmdata_from_database(data_dir)  # Call CHER download function
    if file_path is None or not os.path.isfile(file_path):
        raise ValueError(f"Download failed or file not found at path: {file_path}")
    data = load_and_preprocess_data(file_path, 30, 60, -90, -80)
    positions = normalize_positions(data)
    targets = data["Conc"].values
    time = data["Date"].values
    # Create a DataFrame for the processed data
    df = pd.DataFrame(
        {
            "Time": time,
            "Lat": positions["Lat"],
            "Lon": positions["Lon"],
            "Elevation": positions["Elevation"],
            "Target": targets,
        }
    )

    # Split into training = 80%, testing = validation = 10%
    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Add split column
    train_data["Split"] = "train"
    test_data["Split"] = "test"
    val_data["Split"] = "val"

    # Concatenate the datasets and shuffle train, test and val
    split_data = pd.concat([train_data, test_data, val_data])

    # Reorder columns to have "Split", "Time", "Lat", "Lon", "Elevation", and "Target"
    final_data = split_data[["Split", "Time", "Lat", "Lon", "Elevation", "Target"]]
    save_path = os.path.join(data_dir, 'processed' )
    save_name = 'processed_pm25.csv'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    final_data.to_csv(os.path.join(save_path, save_name), index=False)
    print(f"Data successfully saved to {save_path}")


# Usage within the same script
if __name__ == "__main__":
    print("Starting process_and_save_data execution...")
    data_dir =  '../data'
    process_and_save_data(data_dir)
    print("Execution completed.")
