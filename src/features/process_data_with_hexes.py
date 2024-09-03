import os
import polars as pl
from datetime import datetime, timedelta
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
import argparse

from data.pm25_cher import download_pmdata_from_database as pm
from data.gridmet_cher import download_gridmetpm25_from_database as gm
from data.hexes_centres import download_hexes_data_from_database as hm

def convert_date_to_time(date):
    """Convert date string to a datetime ordinal representation."""
    return datetime.strptime(date, "%Y-%m-%d").toordinal()

@task
def load_and_preprocess_pm_data(file_path, max_lags=10, chunk_size=111):
    """Load PM data and process Lat, Lon, Elevation..."""
    data = pl.read_csv(file_path)
    total_rows = data.shape[0]
    pm_data = pl.DataFrame()

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        pm_data_chunk = load_and_preprocess_pm_data_chunk(data, max_lags, start, end)
        pm_data = pl.concat([pm_data, pm_data_chunk], how="vertical")
    return pm_data

def get_prior_pm(row, data, max_lags):
    """Get previous PM values for a given site and time."""
    end_date = datetime.fromordinal(row["Date"])
    start_date = end_date - timedelta(days=10)
    site_data = data.filter(
        (pl.col("Site") == row["Site"]) & 
        (pl.col("Date") < end_date.toordinal()) & 
        (pl.col("Date") >= start_date.toordinal())
    ).sort("Date", descending=True)

    # For colocated conc, group the data by date and average the Conc values for each date
    site_data = site_data.group_by("Date").agg(pl.mean("Conc").alias("Conc"))
    site_data = site_data.sort("Date")  # Sort in ascending order to maintain chronological order

    pm_list = site_data["Conc"].to_list()[:max_lags]

    # Ensure the pm_list contains exactly max_lags elements
    while len(pm_list) < max_lags:
        pm_list.append(row["Conc"])
    return pm_list

def load_and_preprocess_pm_data_chunk(data, max_lags, start, end):
    """Load and preprocess a chunk of PM data."""
    print(f"Loading PM data chunk from row {start} to {end}...")
    data = data.slice(start, end - start)

    data = data.with_columns([
        pl.col("Date").map_elements(convert_date_to_time, return_dtype=pl.Int64).alias("Date"),
        pl.col("elevation").map_elements(lambda x: max(x, 0), return_dtype=pl.Float64).alias("Elevation")
    ])

    # Drop pmdata rows with NaNs
    data = data.drop_nulls()

    # Average colocated daily measurements (Conc values): Group by Site, Date, Lat, Lon.
    data = data.group_by(["Site", "Date", "Lat", "Lon"]).agg([
        pl.mean("Conc").alias("Conc"),
        pl.mean("Elevation").alias("Elevation"),
        pl.first("h3_08").alias("h3_08")
    ])

    print("Adding max_lags_pm_prior column to PM data...")
    data_dicts = data.to_dicts()
    for row in data_dicts:
        row["lags_pm_prior"] = get_prior_pm(row, data, max_lags)
    data = pl.DataFrame(data_dicts)
    print("PM data preprocessing completed for chunk.")

    return data

@task
def load_and_preprocess_gridmet_data(file_path):
    """Load and preprocess GridMET data."""
    data = pl.read_csv(file_path)
    
    data = data.with_columns([
        pl.col("pm_Date").map_elements(convert_date_to_time, return_dtype=pl.Int64).alias("Date")
    ])

    gridmet_columns = [
        "min_air_temperature_value", "max_air_temperature_value", 
        "min_relative_humidity_value", "max_relative_humidity_value", 
        "wind_speed_value", "precipitation_amount_value"
    ]
    for col in gridmet_columns:
        data = data.with_columns([
            pl.col(col).map_elements(lambda x: max(x, 0), return_dtype=pl.Float64).alias(col)
        ])

    # Drop gridmet rows with NaNs
    data = data.drop_nulls()

    # Rename columns to match pmdata columns
    data = data.rename({
        "pm_Site": "Site",
        "pm_Lat": "Lat",
        "pm_Lon": "Lon"
    })

    # Drop Lat and Lon columns to avoid duplication
    data = data.drop(["Lat", "Lon"])

    # Group by Site, Date and average the relevant columns
    data = data.group_by(["Site", "Date"]).agg([
        pl.mean("min_air_temperature_value").alias("min_air_temperature_value"),
        pl.mean("max_air_temperature_value").alias("max_air_temperature_value"),
        pl.mean("min_relative_humidity_value").alias("min_relative_humidity_value"),
        pl.mean("max_relative_humidity_value").alias("max_relative_humidity_value"),
        pl.mean("wind_speed_value").alias("wind_speed_value"),
        pl.mean("precipitation_amount_value").alias("precipitation_amount_value"),
        pl.first("id").alias("id")  # Include id from gridmet data
    ])

    return data

@task
def merge_and_normalize(pm_data, gridmet_data):
    """Merge PM and GridMET data and normalize."""
    data = pm_data.join(gridmet_data, on=["Site", "Date"], how="inner")

    # Normalize the necessary columns
    columns_to_normalize = ["Lat", "Lon", "Elevation", "min_air_temperature_value", 
                            "max_air_temperature_value", "min_relative_humidity_value", 
                            "max_relative_humidity_value", "wind_speed_value", "precipitation_amount_value"]
    data = normalize_columns(data, columns_to_normalize)

    # Drop rows with NaN values
    data = data.drop_nulls()

    # Add split column
    data = data.with_columns([
        pl.lit("train").alias("split")
    ])

    return data

def normalize_columns(df, columns):
    """Normalize specified columns in the dataframe."""
    print(f"Normalizing columns: {columns}")
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df = df.with_columns([
            ((pl.col(column) - min_val) / (max_val - min_val)).alias(column)
        ])
    return df

@task
def load_and_process_hexes_centres(file_path):
    """Load and preprocess hexes centres data."""
    data = pl.read_csv(file_path)
    # Normalize the necessary columns
    columns_to_normalize = ["lat", "lon", "elevation", "min_air_temperature_value", 
                            "max_air_temperature_value", "min_relative_humidity_value", 
                            "max_relative_humidity_value", "wind_speed_value", "precipitation_amount_value"]
    data = normalize_columns(data, columns_to_normalize)

    # Drop rows with NaN values
    data = data.drop_nulls()

    # Add split column
    data = data.with_columns([
        pl.lit("test").alias("split")
    ])

    return data

@task
def select_year_day(data, date):
    """Select data for the specified year and day."""
    ordinal_date = convert_date_to_time(date)
    data = data.filter(pl.col("Date") == ordinal_date)
    return data

@flow
def process_and_save_data(data_dir, date, pm_save_name="processed_pmgrid.parquet", hexes_save_name="processed_hexes.parquet", max_lags=20, chunk_size=100000):
    """Main function to process and save data."""
    print("Starting the data processing flow...")
    pm_file_path = os.path.join(data_dir, "raw", "pmdata.csv")
    gridmet_file_path = os.path.join(data_dir, "raw", "gridmetpm25.csv")
    hexes_centres_file_path = os.path.join(data_dir, "raw", "conus_hexes_data.csv")

    if not os.path.isfile(pm_file_path):
        print("PM data not found, downloading...")
        pm_file_path = pm(data_dir)
        if pm_file_path is None or not os.path.isfile(pm_file_path):
            raise ValueError(f"Download failed or file not found at path: {pm_file_path}")
    else:
        print("PM data already exists, skipping download.")

    if not os.path.isfile(gridmet_file_path):
        print("GridMET data not found, downloading...")
        gridmet_file_path = gm(data_dir)
        if gridmet_file_path is None or not os.path.isfile(gridmet_file_path):
            raise ValueError(f"Download failed or file not found at path: {gridmet_file_path}")
    else:
        print("GridMET data already exists, skipping download.")

    if not os.path.isfile(hexes_centres_file_path):
        print("HEXES data not found, downloading...")
        hexes_centres_file_path = hm(data_dir)
        if hexes_centres_file_path is None or not os.path.isfile(hexes_centres_file_path):
            raise ValueError(f"Download failed or file not found at path: {hexes_centres_file_path}")
    else:
        print("HEXES data already exists, skipping download.")

    pm_data = load_and_preprocess_pm_data(pm_file_path, chunk_size=chunk_size, max_lags=max_lags)
    gridmet_data = load_and_preprocess_gridmet_data(gridmet_file_path)

    merged_data = merge_and_normalize(pm_data, gridmet_data)
    merged_data = select_year_day(merged_data, date)

    hexes_centres_data = load_and_process_hexes_centres(hexes_centres_file_path)

    save_path = os.path.join(data_dir, 'processed')
    os.makedirs(save_path, exist_ok=True)
    
    pm_save_path = os.path.join(save_path, pm_save_name)
    merged_data.write_parquet(pm_save_path)
    create_markdown_artifact(
        key="pm-data-save-summary",
        markdown=f"## PM and GridMET Data successfully saved\n\nFile Path: `{pm_save_path}`\n\n### Preview\n\n{merged_data.head().to_pandas().to_markdown()}"
    )
    print(f"PM and GridMET Data successfully saved to {pm_save_path}")

    hexes_save_path = os.path.join(save_path, hexes_save_name)
    hexes_centres_data.write_parquet(hexes_save_path)
    create_markdown_artifact(
        key="hexes-data-save-summary",
        markdown=f"## Hexes Data successfully saved\n\nFile Path: `{hexes_save_path}`\n\n### Preview\n\n{hexes_centres_data.head().to_pandas().to_markdown()}"
    )
    print(f"Hexes Data successfully saved to {hexes_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--date',
        type=str,
        help='The date to process (YYYY-MM-DD).',
        default='2013-01-01'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to the data.',
        default='../data'
    )

    # Parse the arguments
    args = parser.parse_args()
    process_and_save_data(data_dir=args.data_dir, date=args.date, pm_save_name="processed_pmgrid.parquet", hexes_save_name="processed_hexes.parquet")
