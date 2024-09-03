import os
import pandas as pd
import numpy as np
from datetime import datetime
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split

from data.pm25_cher import download_pmdata_from_database as pm
from data.gridmet_cher import download_gridmetpm25_from_database as gm

def convert_date_to_time(date):
    """Convert date string to a datetime ordinal representation."""
    return datetime.strptime(date, "%Y-%m-%d").toordinal()

@task
def load_and_preprocess_pm_data(file_path, min_lat, max_lat, min_lon, max_lon):
    """Load PM data and process Lat, Lon, Elevation..."""
    data = pd.read_csv(file_path)
    print(data.columns)
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
def load_and_preprocess_gridmet_data(file_path, min_lat, max_lat, min_lon, max_lon):
    """Load GridMET data and process Lat, Lon, other relevant gridmet columns..."""
    data = pd.read_csv(file_path)
    data = data[
        (data["pm_Lat"] > min_lat)
        & (data["pm_Lat"] < max_lat)
        & (data["pm_Lon"] > min_lon)
        & (data["pm_Lon"] < max_lon)
    ]

    data["pm_Date"] = data["pm_Date"].apply(convert_date_to_time)
    
    gridmet_columns = [
        "min_air_temperature_value", "max_air_temperature_value", 
        "min_relative_humidity_value", "max_relative_humidity_value", 
        "wind_speed_value", "precipitation_amount_value"
    ]
    for col in gridmet_columns:
        data[col] = data[col].values
        data.loc[data[col] < 0, col] = 0

    return data

@task
def normalize_columns(df, columns):
    """Normalize specified columns in the dataframe."""
    normalized_df = df.copy()
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
    return normalized_df

@flow
def process_and_save_data(data_dir, save_name="processed_pm25_gridmet.csv", download=True):
    """Main function to process, split and save result."""
    if download:
        pm_file_path = pm(data_dir)  # Call PM data download function
        gridmet_file_path = gm(data_dir)  # Call GridMET data download function
    pm_file_path = data_dir+'/raw/pmdata.csv'
    gridmet_file_path = data_dir+'/raw/gridmetpm25.csv'

    if pm_file_path is None or not os.path.isfile(pm_file_path):
        raise ValueError(f"Download failed or file not found at path: {pm_file_path}")

    if gridmet_file_path is None or not os.path.isfile(gridmet_file_path):
        raise ValueError(f"Download failed or file not found at path: {gridmet_file_path}")

    # filter data based on lat and lon; this roughly corresponds to the SE us. 
    # min_lat = 30
    # max_lat = 60
    # min_lon = -90
    # max_lon = -80
    # filter data based on lat and lon; this roughly corresponds to the eastern seaboard. 
    min_lat = 30
    max_lat = 60
    min_lon = -80
    max_lon = -70
    # filter data based on lat and lon; this roughly corresponds to the west coast. 
    # min_lat = 30
    # max_lat = 60
    # min_lon = -125
    # max_lon = -102
    pm_data = load_and_preprocess_pm_data(pm_file_path, min_lat, max_lat, min_lon, max_lon)
    gridmet_data = load_and_preprocess_gridmet_data(gridmet_file_path, min_lat, max_lat, min_lon, max_lon)

    # Renaming columns in gridmetdata to match pmdata columns
    pm_data.rename(columns = {'Site': 'site','Date': 'date','Lat':'lat','Lon':'lon','Elevation':'elevation', 'Conc':'conc'}, inplace=True)
    gridmet_data.rename(columns={"pm_Date": "date", "pm_Lat": "lat", "pm_Lon": "lon"}, inplace=True)

    pm_data = normalize_columns(pm_data, ["lat", "lon", "elevation"])
    gridmet_columns = [
        "min_air_temperature_value", "max_air_temperature_value", 
        "min_relative_humidity_value", "max_relative_humidity_value", 
        "wind_speed_value", "precipitation_amount_value"
    ]
    gridmet_data = normalize_columns(gridmet_data, gridmet_columns)

    # Ensuring to avoid column conflicts by dropping duplicates
    gridmet_data = gridmet_data.drop(columns=["lat", "lon"], errors='ignore')

    merged_data = pd.merge(pm_data, gridmet_data, left_on=["date", "site"], right_on=["date", "pm_Site"], how="inner")

    targets = merged_data["conc"].values
    time = merged_data["date"].values
    columns = ["time", "lat", "lon", "elevation"] + gridmet_columns

    df = pd.DataFrame(
        {
            "time": time,
            "lat": merged_data["lat"],
            "lon": merged_data["lon"],
            "elevation": merged_data["elevation"],
            "target": targets,
            "site": merged_data["site"]
        }
    )

    for col in gridmet_columns:
        df[col] = merged_data[col]

    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_data["split"] = "train"
    test_data["split"] = "test"
    val_data["split"] = "val"

    split_data = pd.concat([train_data, test_data, val_data])
    split_data = split_data.sample(frac=1, random_state=42).reset_index(drop=True)

    final_data = split_data[["split", "time", "lat", "lon", "elevation", "target", "site"] + gridmet_columns]
    print(final_data.columns)
    save_path = os.path.join(data_dir, 'processed')
    os.makedirs(save_path, exist_ok=True)

    final_data.to_csv(os.path.join(save_path, save_name), index=False)

    create_markdown_artifact(
        key="data-save-summary",
        markdown=f"## Data successfully saved\n\nFile Path: `{os.path.join(save_path, save_name)}`\n\n### Preview\n\n{final_data.head().to_markdown()}"
    )
    
    print(f"Data successfully saved to {os.path.join(save_path, save_name)}")

if __name__ == "__main__":
    data_dir = '../data'
    process_and_save_data(data_dir=data_dir, save_name="processed_pm25_gridmet_eastern.csv", download=False)
