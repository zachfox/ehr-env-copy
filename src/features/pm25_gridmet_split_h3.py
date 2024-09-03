import os
import pandas as pd
import h3
from sklearn.model_selection import train_test_split
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

@task
def split_data(file_path):
    """Split data into train, test, and validation sets."""
    data = pd.read_csv(file_path)

    # Split data based on unique sites
    sites = data["Site"].unique()
    train_sites, test_sites = train_test_split(sites, test_size=0.1, random_state=42)
    train_sites, val_sites = train_test_split(train_sites, test_size=0.1, random_state=42)

    train_data = data[data["Site"].isin(train_sites)].copy()
    val_data = data[data["Site"].isin(val_sites)].copy()
    test_data = data[data["Site"].isin(test_sites)].copy()

    train_data.loc[:, "Split"] = "train"
    val_data.loc[:, "Split"] = "val"
    test_data.loc[:, "Split"] = "test"

    split_data = pd.concat([train_data, val_data, test_data])
    split_data = split_data.sample(frac=1, random_state=42).reset_index(drop=True)

    final_data = split_data[["Split", "id", "Site"]]
    return final_data

def create_h3_from_h3_08(h3_08, target_resolution):
    """Function to create a lower resolution h3 index from a higher resolution h3_08 index."""
    return h3.h3_to_parent(h3_08, target_resolution)

@task
def process_and_join(file_path_processed, split_data, save_name="split_h3.csv", target_resolution=3):
    """Process h3_08 to h3 and join with split data."""
    processed_data = pd.read_csv(file_path_processed)

    processed_data["h3"] = processed_data["h3_08"].apply(create_h3_from_h3_08, target_resolution=target_resolution)

    final_data = pd.merge(split_data, processed_data, on="id", how="inner")

    final_data = final_data.rename(columns={"Site_x": "Site", "Site_y": "Original_Site"})

    final_data = final_data[['Split', 'id', 'Site', 'h3_08', 'h3']]

    save_path = os.path.dirname(file_path_processed)
    final_file_path = os.path.join(save_path, save_name)

    final_data.to_csv(final_file_path, index=False)

    create_markdown_artifact(f"Data successfully saved to `{final_file_path}`")
    print(f"Data successfully saved to {final_file_path}")
    return final_file_path

@flow
def data_splitting_and_processing():
    processed_file_path = '../data/processed/processed_pm25_gridmet.csv'

    split_df = split_data(file_path=processed_file_path)

    final_file_path = process_and_join(file_path_processed=processed_file_path, split_data=split_df)

if __name__ == "__main__":
    data_splitting_and_processing()
