import copy
import sys
from pathlib import Path

import numpy as np

from .nuscenes_dataset import NuScenesDataset
from ..augmentor.paired_data_augmentor import PairedDataAugmentor


class PairedNuScenesDataset(NuScenesDataset):
    """
    NuScenes dataset that returns paired clean/weather point clouds for siamese training.
    Supports pre-generated weather data (primary) and online simulation (fallback).
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.weather_cfg = dataset_cfg.get('WEATHER', None)
        assert self.weather_cfg is not None, 'WEATHER config is required for PairedNuScenesDataset'

        self.weather_data_path = Path(self.weather_cfg.PREGENERATED_PATH) \
            if self.weather_cfg.get('PREGENERATED_PATH', None) else None
        self.weather_type = self.weather_cfg.get('WEATHER_TYPE', 'rain')
        self.corrupt_sweeps = self.weather_cfg.get('CORRUPT_SWEEPS', False)

        self.online_sim_cfg = self.weather_cfg.get('ONLINE_SIM', None)
        self.online_sim_enabled = self.online_sim_cfg is not None and self.online_sim_cfg.get('ENABLED', False)

        if self.training:
            self.data_augmentor = PairedDataAugmentor(
                self.root_path, self.dataset_cfg.DATA_AUGMENTOR,
                self.class_names, logger=self.logger
            )

    def _simulate_weather_online(self, points_4d):
        """
        Apply online weather simulation to point cloud.
        Args:
            points_4d: (N, 4) array [x, y, z, intensity]
        Returns:
            corrupted: (M, 4) array [x, y, z, intensity]
        """
        wtype = self.weather_type
        if wtype == 'random':
            wtype = np.random.choice(['rain', 'snow', 'fog'])

        weather_process_path = str(Path(__file__).resolve().parents[4] / '..' / 'weather-process')
        if weather_process_path not in sys.path:
            sys.path.insert(0, weather_process_path)

        if wtype == 'rain':
            from rain_simulation import RainSimulation
            rate_range = self.online_sim_cfg.get('RAIN_RATE', [1.0, 50.0])
            rate = np.random.uniform(rate_range[0], rate_range[1])
            sim = RainSimulation(rain_rate=rate)
        elif wtype == 'snow':
            from snow_simulation import SnowSimulation
            rate_range = self.online_sim_cfg.get('SNOW_RATE', [0.5, 10.0])
            rate = np.random.uniform(rate_range[0], rate_range[1])
            sim = SnowSimulation(snowfall_rate=rate)
        elif wtype == 'fog':
            from fog_simulation import FogSimulation
            vis_range = self.online_sim_cfg.get('FOG_VISIBILITY', [50, 1000])
            vis = np.random.uniform(vis_range[0], vis_range[1])
            sim = FogSimulation(visibility=vis)
        else:
            raise ValueError(f'Unknown weather type: {wtype}')

        return sim.simulate(points_4d)

    def _load_weather_points_pregenerated(self, info):
        """Load pre-generated weather point cloud corresponding to the clean frame."""
        lidar_path = Path(info['lidar_path'])
        weather_file = self.weather_data_path / lidar_path.name
        if not weather_file.exists():
            stem = lidar_path.stem
            for suffix in ['.bin', '.npy']:
                candidate = self.weather_data_path / (stem + suffix)
                if candidate.exists():
                    weather_file = candidate
                    break

        if weather_file.exists():
            if weather_file.suffix == '.npy':
                return np.load(str(weather_file)).astype(np.float32)
            else:
                return np.fromfile(str(weather_file), dtype=np.float32).reshape([-1, 5])[:, :4]
        return None

    def _get_weather_points(self, index, clean_points):
        """
        Get weather-corrupted point cloud for a given frame.
        Args:
            index: frame index
            clean_points: (N, 5) clean points already loaded [x, y, z, intensity, timestamp]
        Returns:
            (M, 5) array [x, y, z, intensity, timestamp] matching clean points format.
        """
        info = self.infos[index]

        pregenerated = None
        if self.weather_data_path is not None:
            pregenerated = self._load_weather_points_pregenerated(info)

        if pregenerated is not None:
            if pregenerated.shape[1] == 4:
                timestamps = np.zeros((pregenerated.shape[0], 1), dtype=np.float32)
                pregenerated = np.concatenate([pregenerated, timestamps], axis=1)

            if not self.corrupt_sweeps:
                sweep_pts = clean_points[clean_points[:, 4] != 0.0]
                if len(sweep_pts) > 0:
                    return np.concatenate([pregenerated, sweep_pts], axis=0)
            return pregenerated

        if self.online_sim_enabled:
            points_4d = clean_points[:, :4].copy()
            corrupted_4d = self._simulate_weather_online(points_4d)
            n_corrupted = corrupted_4d.shape[0]
            n_original = clean_points.shape[0]

            if n_corrupted <= n_original:
                timestamps = clean_points[:n_corrupted, 4:5]
            else:
                timestamps = np.zeros((n_corrupted, 1), dtype=np.float32)
                timestamps[:n_original] = clean_points[:n_original, 4:5]
            return np.concatenate([corrupted_4d, timestamps], axis=1)

        return clean_points.copy()

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        weather_points = self._get_weather_points(index, points)

        input_dict = {
            'points': points,
            'weather_points': weather_points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in data_dict:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict
