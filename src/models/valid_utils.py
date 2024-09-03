import os
import yaml
from argparse import Namespace
import pyarrow.parquet as pq

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sys

sys.path.append("../src/models/")
import network_light as nl
import s_parser as sp
import dataloaders as dl
from dataloaders import SenseiverLoaderInference

from importlib import reload


def get_latest_config():
    yaml_file_path = "/Users/joa/Desktop/pm-modeling/src/models/wandb/run-20240811_010147-t0ofz8t4/files/config.yaml"
    with open(yaml_file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for key, val in config.items():
        try:
            config[key] = config[key]["value"]
        except:
            continue
    args = Namespace(**config)
    data_config, encoder_config, decoder_config = sp.assign_args(args)
    return args, data_config, encoder_config, decoder_config


def load_model(model_id, epoch):
    path_to_checkpoint = f"/Users/joa/Desktop/pm-modeling/src/models/pollution/{model_id}/checkpoints/train-epoch={epoch}.ckpt"
    model = nl.Senseiver.load_from_checkpoint(path_to_checkpoint)
    return model


#Added:
def get_eval_dataloader(data_config, batch_size=32, num_workers=4):
    dataset = SenseiverLoaderInference(data_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

def get_model_results(model, dataloader):
    all_pred = []
    sensor_means = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            sensors, query_coords = batch 
            
            # Forward pass through the model
            out = model(sensors.to('mps'), query_coords.to('mps'))
            
            # Ensure the output shape is compatible
            out = out.view(out.size(0), -1)  # Reshape to (batch_size, num_outputs)
            
            # Populate all_pred and sensor_means
            all_pred += list(out.detach().cpu().numpy().ravel())
            sensor_means += list(sensors[:, 0].mean(dim=1).detach().cpu().numpy().ravel())  # Calculate means for each sample in the batch

    print(f"Total predictions: {len(all_pred)}")
    print(f"Total sensor means: {len(sensor_means)}")
    return all_pred, sensor_means

def process_chunk(model, data_config, start_idx, chunk_size, total_samples):
    end_idx = min(start_idx + chunk_size, total_samples)
    
    dl = get_eval_dataloader(data_config)

    # Slicing the dataset for the chunk
    limited_dataset = torch.utils.data.Subset(dl.dataset, range(start_idx, end_idx))
    limited_dataloader = DataLoader(limited_dataset, batch_size=dl.batch_size, shuffle=False, num_workers=dl.num_workers)

    predictions, sensor_means = get_model_results(model, limited_dataloader)  

    # Load query data to append predictions and if needed, sensor_means
    query_data = pd.read_parquet(data_config['query_data_path'])
    query_data_chunk = query_data.iloc[start_idx:end_idx]

    print(f"Query data chunk shape: {query_data_chunk.shape}")
    print(f"Predictions length: {len(predictions)}")
    print(f"Sensor means length: {len(sensor_means)}")

    query_data_chunk.loc[:, 'predictions'] = predictions
    query_data_chunk.loc[:, 'sensor_means'] = sensor_means

    return query_data_chunk

def run_test_in_chunks(model, data_config, chunk_size=100000):
    output_dir = 'chunks'
    os.makedirs(output_dir, exist_ok=True)

    # Load query data to determine the total number of samples
    query_data = pd.read_parquet(data_config['query_data_path'])
    total_samples = len(query_data)

    all_chunks = []
    for start_idx in range(0, total_samples, chunk_size):
        chunk_filename = os.path.join(output_dir, f'chunk_{start_idx}.csv')
        
        if os.path.exists(chunk_filename):
            print(f"Chunk {chunk_filename} already exists. Skipping.")
        else:
            print(f"Processing chunk starting at index {start_idx}")
            chunk_data = process_chunk(model, data_config, start_idx, chunk_size, total_samples)
            chunk_data.to_csv(chunk_filename, index=False)
        
        all_chunks.append(chunk_filename)
    
    # Combine all chunks into a single CSV file
    combined_data = pd.concat([pd.read_csv(chunk) for chunk in all_chunks])
    combined_data.to_csv('combined_hex_pred.csv', index=False)
    print("All chunks combined into combined_hex_pred")
