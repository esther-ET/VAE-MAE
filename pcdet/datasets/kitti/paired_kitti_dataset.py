import copy
import sys
from pathlib import Path

import numpy as np

from .kitti_dataset import KittiDataset
from . import kitti_utils
from ...utils import box_utils, common_utils
from ..augmentor.paired_data_augmentor import PairedDataAugmentor


class PairedKittiDataset(KittiDataset):
    """
    KITTI dataset that returns paired clean/weather point clouds for siamese training.
    Supports pre-generated weather data (primary) and online simulation (fallback).
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.weather_cfg = dataset_cfg.get('WEATHER', None)
        assert self.weather_cfg is not None, 'WEATHER config is required for PairedKittiDataset'

        self.weather_data_path = Path(self.weather_cfg.PREGENERATED_PATH) \
            if self.weather_cfg.get('PREGENERATED_PATH', None) else None
        self.weather_type = self.weather_cfg.get('WEATHER_TYPE', 'rain')

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

    def _load_weather_points_pregenerated(self, sample_idx):
        """Load pre-generated weather point cloud corresponding to the clean frame."""
        weather_file = self.weather_data_path / f'{sample_idx}.bin'
        if not weather_file.exists():
            for suffix in ['.bin', '.npy']:
                candidate = self.weather_data_path / f'{sample_idx}{suffix}'
                if candidate.exists():
                    weather_file = candidate
                    break

        if weather_file.exists():
            if weather_file.suffix == '.npy':
                weather_points = np.load(str(weather_file)).astype(np.float32)
            else:
                weather_points = np.fromfile(str(weather_file), dtype=np.float32)
                weather_points = weather_points.reshape(-1, 4)

            if weather_points.shape[1] > 4:
                weather_points = weather_points[:, :4]
            return weather_points

        return None

    def _get_weather_points(self, sample_idx, clean_points):
        """
        Get weather-corrupted point cloud for a given frame.
        Args:
            sample_idx: KITTI sample id string, e.g. '000123'
            clean_points: (N, 4) clean points [x, y, z, intensity]
        Returns:
            (M, 4) array [x, y, z, intensity]
        """
        pregenerated = None
        if self.weather_data_path is not None:
            pregenerated = self._load_weather_points_pregenerated(sample_idx)

        if pregenerated is not None:
            return pregenerated

        if self.online_sim_enabled:
            return self._simulate_weather_online(clean_points.copy())

        return clean_points.copy()

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if 'gt_boxes2d' in get_item_list:
                input_dict['gt_boxes2d'] = annos['bbox']

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if 'points' in get_item_list:
            points = self.get_lidar(sample_idx)
            weather_points = self._get_weather_points(sample_idx, points)

            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]

                weather_pts_rect = calib.lidar_to_rect(weather_points[:, 0:3])
                weather_fov_flag = self.get_fov_flag(weather_pts_rect, img_shape, calib)
                weather_points = weather_points[weather_fov_flag]

            input_dict['points'] = points
            input_dict['weather_points'] = weather_points

        if 'images' in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if 'depth_maps' in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if 'calib_matricies' in get_item_list:
            input_dict['trans_lidar_to_cam'], input_dict['trans_cam_to_img'] = kitti_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict