import torch
from torch.utils.data import Dataset, DataLoader

import polars as pl
import positional as ps

from spatial_samplers import SpatialSampler


def senseiver_dataloader(data_config, num_workers=0, mode="train"):
    if mode == "validation":
        return DataLoader(
            SenseiverLoader(data_config, mode),
            batch_size=32,
            pin_memory=True,
            shuffle=False
        )
    elif mode == "train":
        return DataLoader(
            SenseiverLoader(data_config, mode),
            batch_size=data_config["batch_size"],
            pin_memory=True,
            shuffle=True
        )
    elif mode == "test":
        return DataLoader(
            SenseiverLoader(data_config, mode),
            batch_size=128,
            pin_memory=True,
            shuffle=False,
            num_workers=7,
        )

    elif mode == "inference":
        return DataLoader(
            SenseiverLoader(data_config, mode),
            batch_size=128,
            pin_memory=True,
            shuffle=False,
            num_workers=7,
        )


class SenseiverLoader(Dataset):
    def __init__(self, data_config, mode):
        data_path = data_config["data_path"]
        split_path = data_config["split_path"]
        self.num_sensors = data_config["num_sensors"]
        self.num_queries = data_config["num_queries"]
        self.num_freqs = data_config["space_bands"]
        self.nlags = data_config["nlags"]
        self.locality = data_config["locality"]
        self.mode = mode
        seed = data_config["seed"]

        dataset = pl.read_parquet(data_path)
        splits = pl.read_csv(split_path)

        dataset = dataset.join(splits, on="id", how="left")
        dataset = dataset.drop_nulls()

        dataset = dataset.rename(
            {
                "Conc": "target",
                "Lat": "lat",
                "Lon": "lon",
                "Split": "split",
                "Site": "site",
                "Date": "time",
                "Elevation": "elevation",
            }
        )

        dataset = dataset.filter(pl.col("target") > 0)
        dataset = dataset.with_columns(
            [pl.col("target").log(base=10).alias("log_target")]
        )

        self.sensor_data, self.query_data = self.split_dataset(dataset)
        self.times = list(set(self.query_data["time"]))

        self.weather_columns = [
            "min_air_temperature_value",
            "max_air_temperature_value",
            "min_relative_humidity_value",
            "max_relative_humidity_value",
            "wind_speed_value",
            "precipitation_amount_value",
        ]
        self.target_col = "log_target"
        self.position_columns = ["lat", "lon", "elevation"]
        self.extra = [self.target_col, "time", "lags_pm_prior"]
        all_cols = self.weather_columns + self.position_columns + self.extra

        self.sensor_data = {
            col: torch.Tensor(self.sensor_data[col]) for col in all_cols
        }
        self.query_data = {col: torch.Tensor(self.query_data[col]) for col in all_cols}

        self.sensor_data["lags_pm_prior"][self.sensor_data["lags_pm_prior"] <= 0] = 1e-8
        self.sensor_data["lags_pm_prior"] = torch.log10(
            self.sensor_data["lags_pm_prior"]
        )

        # spatial sampler
        self.sampler = SpatialSampler(self.sensor_data, self.query_data, data_config)

    def split_dataset(self, dataset):
        if self.mode == 'train':
            sensor_data = dataset.filter(pl.col("split") == "train")
            query_data = dataset.filter(pl.col("split") == "train")
        elif self.mode == 'validation':
            sensor_data = dataset.filter(pl.col("split") == "train")
            query_data = dataset.filter(pl.col("split") == "val")
        elif self.mode=='test':
            sensor_data = dataset.filter((pl.col('split')=='train') | (pl.col('split')=='val'))
            query_data = dataset.filter(pl.col('split')=='test')
        elif self.mode == 'inference':
            sensor_data = dataset.filter(pl.col("split") == "train")
            query_data = dataset.filter(pl.col("split") == "test")
        else:
            raise ValueError(f"Invalid input: '{mode}'. Expected one of 'train', 'test', 'validation', or 'inference'.")
        return sensor_data, query_data

    def __len__(self):
        if self.mode == 'train':
            return len(self.times)
        if self.mode == 'validation':
            return len(self.times)
        else:
            return len(self.query_data["lat"])

    def __getitem__(self, idx):
        if self.mode == 'train':
            time = self.times[idx]
            idxs = torch.where(self.sensor_data["time"] == time)[0]
            idxs = idxs[torch.randperm(idxs.shape[0])]
            query_idxs = idxs[0:1] 
            idxs = idxs[1:]
        else:
            time = self.query_data['time'][idx]
            idxs = torch.where(self.sensor_data["time"] == time)[0]
            query_idxs = torch.LongTensor([idx])

        sensor_idxs = self.sampler.sample(idxs, query_idxs)

        sensor_positions = torch.cat([ self.sensor_data[pos][sensor_idxs].unsqueeze(1) for pos in self.position_columns],dim=-1)
        sensor_embs = ps.spatial_encoding(sensor_positions, self.num_freqs)
        sensor_values = torch.cat(
            [self.sensor_data[self.target_col][sensor_idxs].unsqueeze(1)]+
            [self.sensor_data["lags_pm_prior"][sensor_idxs,:self.nlags]]+
            [sensor_embs]+ 
            [torch.cat([self.sensor_data[weather][sensor_idxs].unsqueeze(1) for weather in self.weather_columns],
            dim=-1)],dim=-1
        )
        query_positions = torch.cat([ self.query_data[pos][query_idxs].unsqueeze(1) for pos in self.position_columns],dim=-1)
        query_embs = ps.spatial_encoding(query_positions, self.num_freqs)
        query_values = torch.cat(
            [query_embs]+ 
            [torch.cat([self.query_data[weather][query_idxs].unsqueeze(1) for weather in self.weather_columns],
            dim=-1)],dim=-1
        )

        if self.mode == "inference":
            return sensor_values, query_values

        else:
            field_values = self.query_data[self.target_col][query_idxs]
            return sensor_values, query_values, field_values.unsqueeze(-1)
