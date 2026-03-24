import numpy as np

from ...utils import common_utils
from .data_augmentor import DataAugmentor


class PairedDataAugmentor(DataAugmentor):
    """
    Data augmentor that synchronizes geometric augmentations between clean and
    weather point clouds. Uses numpy random state save/restore to guarantee
    identical random decisions for both branches.

    gt_sampling should be DISABLED via DISABLE_AUG_LIST in the config, because
    GT database points come from clean weather and would break domain consistency.
    """

    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        super().__init__(root_path, augmentor_configs, class_names, logger=logger)

    def forward(self, data_dict):
        """
        Augment both clean points and weather_points with identical geometric
        transforms by replaying the same numpy random state.
        """
        weather_points = data_dict.pop('weather_points', None)

        has_weather = weather_points is not None
        if has_weather:
            gt_boxes_backup = data_dict['gt_boxes'].copy()
            gt_names_backup = data_dict['gt_names'].copy()
            gt_boxes_mask_backup = data_dict.get('gt_boxes_mask', None)
            if gt_boxes_mask_backup is not None:
                gt_boxes_mask_backup = gt_boxes_mask_backup.copy()
            # 记录增强状态
            rng_state = np.random.get_state()

        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        # 对有weather的进行和clear的一样的操作
        if has_weather:
            # 重放rng状态的增强
            np.random.set_state(rng_state)

            weather_dict = {
                'gt_boxes': gt_boxes_backup,
                'gt_names': gt_names_backup,
                'points': weather_points,
            }
            if gt_boxes_mask_backup is not None:
                weather_dict['gt_boxes_mask'] = gt_boxes_mask_backup

            for cur_augmentor in self.data_augmentor_queue:
                weather_dict = cur_augmentor(data_dict=weather_dict)

            weather_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                weather_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
            if 'gt_boxes_mask' in weather_dict:
                weather_mask = weather_dict['gt_boxes_mask']
                weather_dict['gt_boxes'] = weather_dict['gt_boxes'][weather_mask]
                weather_dict['gt_names'] = weather_dict['gt_names'][weather_mask]
                weather_dict.pop('gt_boxes_mask')

            data_dict['weather_points'] = weather_dict['points']

        return data_dict
