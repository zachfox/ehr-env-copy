import os
import pickle

import pandas as pd
import numpy as np

from tqdm import tqdm
from datetime import datetime
from pyhigh import get_elevation_batch

"""
script to convert PM2.5 daily data into a friendly format for ML tasks. 
"""


def convert_date_to_time(date):
    return datetime.strptime(date, "%Y-%m-%d").toordinal()


def main(path_to_data, save_path, normalize_positions=True):
    year_start = 2002
    year_end = 2020

    # concatenate the data
    all_time, all_lat, all_lon, all_elevation, all_pm = [], [], [], [], []
    for year in tqdm(range(year_start, year_end)):
        try:
            data = pd.read_csv(os.path.join(path_to_data, f"pm_{year}.csv"))
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

            data = data[data["Lat"] > min_lat]
            data = data[data["Lat"] < max_lat]
            data = data[data["Lon"] > min_lon]
            data = data[data["Lon"] < max_lon]

            all_time += data["Date"].apply(convert_date_to_time).to_list()
            lats = data["Lat"].to_list()
            lons = data["Lon"].to_list()
            all_pm += data["Conc"].to_list()
            all_lat += lats
            all_lon += lons
            all_elevation += list(
                get_elevation_batch(list(zip(lats, lons)))
            )  # this is calling a usgs api; may need to be called more or less
        except pd.errors.EmptyDataError:
            print(f"file for year {year} is empty")
    all_elevation = np.array(all_elevation)
    all_elevation[all_elevation<0] = 0 
    positions = [all_lon, all_lat, all_elevation]
    print('number of data points: {0}'.format(len(all_lat)))
    # normalize the positions
    if normalize_positions:
        poses = []
        for position in positions:
            pos = np.array(position)
            position = list((pos - np.min(pos))/ (np.max(pos) - np.min(pos)))
            poses.append(position)
        positions = poses
        
    # write to disk
    pm_dataset = {"time": all_time, "position": positions, "target": all_pm}
    with open(save_path, "wb") as f:
        pickle.dump(pm_dataset, f)
    return

if __name__ == "__main__":
    base_dir = "../.."
    path_to_data = os.path.join(base_dir, "data", "raw", "daily_pm")
    save_path = os.path.join(base_dir, "data", "processed", "daily_pm_east.pickle")
    main(path_to_data, save_path)
