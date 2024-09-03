import torch


class SpatialSampler:
    def __init__(self, sensor_data, query_data, config):
        self.sensor_data = sensor_data
        self.query_data = query_data
        self.locality = config['locality']
        self.num_sensors = config['num_sensors']

    def apply_locality(self, idxs, query_idx):
        lat_inds = (self.sensor_data["lat"][idxs] < self.query_data["lat"][query_idx]+self.locality) & (self.sensor_data["lat"][idxs] > self.query_data["lat"][query_idx]-self.locality)
        lon_inds = (self.sensor_data["lon"][idxs] < self.query_data["lon"][query_idx]+self.locality) & (self.sensor_data["lon"][idxs] > self.query_data["lon"][query_idx]-self.locality)
        local_inds = lat_inds*lon_inds
        return local_inds

    def sample(self, idxs, query_idx):
        local_inds = self.apply_locality(idxs, query_idx)
        idxs_local = idxs[local_inds]
        if len(idxs_local) == 0:  # just random
            return idxs[torch.randperm(idxs.shape[0])][:self.num_sensors]

        if len(idxs_local) < self.num_sensors:
            idxs_local = torch.cat(self.num_sensors*[idxs_local])

        idxs_sensors = idxs_local[torch.randperm(idxs_local.shape[0])][:self.num_sensors]
        return idxs_sensors