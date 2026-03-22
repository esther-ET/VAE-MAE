import os

import torch
import torch.nn as nn

from . import build_detector
from ...utils.distill_loss_utils import FeatureDistillLoss, PredictionDistillLoss


class SiameseWeatherDetector(nn.Module):
    """
    Siamese network for weather-robust 3D object detection.

    Teacher branch: processes clean weather data with frozen weights.
    Student branch: processes weather-corrupted data with trainable weights.
    Supports configurable feature-level and prediction-level knowledge distillation.
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.teacher = build_detector(
            model_cfg=model_cfg.TEACHER, num_class=num_class, dataset=dataset
        )
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = build_detector(
            model_cfg=model_cfg.STUDENT, num_class=num_class, dataset=dataset
        )

        self.distill_cfg = model_cfg.get('DISTILLATION', None)
        self.feat_distill_losses = nn.ModuleDict()
        self.pred_distill_loss = None
        if self.distill_cfg is not None:
            self._build_distill_modules()

    def _build_distill_modules(self):
        feat_cfg = self.distill_cfg.get('FEATURE_DISTILL', None)
        if feat_cfg is not None and feat_cfg.get('ENABLED', False):
            loss_type = feat_cfg.get('LOSS_TYPE', 'mse')
            adapt = feat_cfg.get('ADAPTATION', False)
            for key in feat_cfg.get('FEATURE_KEYS', []):
                t_ch = feat_cfg.get('TEACHER_CHANNELS', None)
                s_ch = feat_cfg.get('STUDENT_CHANNELS', None)
                self.feat_distill_losses[key] = FeatureDistillLoss(
                    loss_type=loss_type,
                    teacher_channels=t_ch if adapt else None,
                    student_channels=s_ch if adapt else None,
                )

        pred_cfg = self.distill_cfg.get('PREDICTION_DISTILL', None)
        if pred_cfg is not None and pred_cfg.get('ENABLED', False):
            self.pred_distill_loss = PredictionDistillLoss(
                heatmap_loss_type=pred_cfg.get('HEATMAP_LOSS_TYPE', 'kl_div'),
                regression_loss_type=pred_cfg.get('REGRESSION_LOSS_TYPE', 'smooth_l1'),
                heatmap_weight=pred_cfg.get('HEATMAP_WEIGHT', 0.5),
                regression_weight=pred_cfg.get('REGRESSION_WEIGHT', 0.25),
                score_thresh=pred_cfg.get('SCORE_THRESH', 0.3),
            )

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def _prepare_clean_batch(self, batch_dict):
        """Extract fields for teacher, using clean points."""
        return {k: v for k, v in batch_dict.items() if not k.startswith('weather_')}

    def _prepare_weather_batch(self, batch_dict):
        """Swap weather_points into 'points' for student consumption."""
        weather = {k: v for k, v in batch_dict.items() if not k.startswith('weather_')}
        if 'weather_points' in batch_dict:
            weather['points'] = batch_dict['weather_points']
        return weather

    def forward(self, batch_dict):
        if not self.training:
            student_batch = self._prepare_weather_batch(batch_dict)
            for module in self.student.module_list:
                student_batch = module(student_batch)
            pred_dicts, recall_dicts = self.student.post_processing(student_batch)
            return pred_dicts, recall_dicts

        # ---- Teacher forward (no gradient) ----
        clean_batch = self._prepare_clean_batch(batch_dict)
        with torch.no_grad():
            self.teacher.eval()
            for module in self.teacher.module_list:
                clean_batch = module(clean_batch)

        # ---- Student forward ----
        weather_batch = self._prepare_weather_batch(batch_dict)
        for module in self.student.module_list:
            weather_batch = module(weather_batch)

        return self._compute_training_loss(clean_batch, weather_batch)

    def _compute_training_loss(self, clean_batch, weather_batch):
        disp_dict = {}

        det_loss, tb_dict = self.student.dense_head.get_loss()
        tb_dict['loss_det'] = det_loss.item()

        total_loss = det_loss
        distill_loss_total = torch.tensor(0.0, device=det_loss.device)

        if self.distill_cfg is not None:
            feat_cfg = self.distill_cfg.get('FEATURE_DISTILL', None)
            if feat_cfg is not None and feat_cfg.get('ENABLED', False):
                feat_weight = feat_cfg.get('WEIGHT', 1.0)
                for key, loss_fn in self.feat_distill_losses.items():
                    t_feat = clean_batch.get(key)
                    s_feat = weather_batch.get(key)
                    if t_feat is not None and s_feat is not None:
                        fl = loss_fn(t_feat, s_feat)
                        distill_loss_total = distill_loss_total + feat_weight * fl
                        tb_dict[f'distill_feat_{key}'] = fl.item()

            pred_cfg = self.distill_cfg.get('PREDICTION_DISTILL', None)
            if pred_cfg is not None and pred_cfg.get('ENABLED', False) \
                    and self.pred_distill_loss is not None:
                t_preds = self.teacher.dense_head.forward_ret_dict.get('pred_dicts', [])
                s_preds = self.student.dense_head.forward_ret_dict.get('pred_dicts', [])
                if t_preds and s_preds:
                    pl, pl_tb = self.pred_distill_loss(t_preds, s_preds)
                    distill_loss_total = distill_loss_total + pl
                    tb_dict.update(pl_tb)

        total_loss = total_loss + distill_loss_total
        tb_dict['loss_distill'] = distill_loss_total.item()
        tb_dict['loss_rpn'] = det_loss.item()

        ret_dict = {'loss': total_loss}
        return ret_dict, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        return self.student.post_processing(batch_dict)

    # ---- Weight loading utilities ----

    def load_teacher_from_ckpt(self, filename, logger, to_cpu=False):
        """Load pre-trained weights into the teacher branch."""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'Teacher checkpoint not found: {filename}')

        logger.info('==> Loading teacher weights from %s' % filename)
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        state_dict, update_state = self.teacher._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_state:
                logger.info('Teacher: not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Teacher loaded (%d/%d weights)' % (len(update_state), len(state_dict)))

        for p in self.teacher.parameters():
            p.requires_grad = False

    def load_params_from_file(self, filename, logger, to_cpu=False):
        """Load checkpoint — student weights only."""
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint.get('model_state', checkpoint)

        student_state = {}
        for key, val in model_state_disk.items():
            if key.startswith('student.'):
                student_state[key[len('student.'):]] = val
            else:
                student_state[key] = val

        state_dict, update_state = self.student._load_state_dict(student_state, strict=False)

        for key in state_dict:
            if key not in update_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        """Load checkpoint with optimizer state for resuming training."""
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        model_state_disk = checkpoint.get('model_state', {})

        full_state = self.state_dict()
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in full_state and full_state[key].shape == val.shape:
                update_model_state[key] = val

        full_state.update(update_model_state)
        self.load_state_dict(full_state)

        logger.info('==> Loaded %d/%d weights' % (len(update_model_state), len(full_state)))

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s' % filename)
                optimizer.load_state_dict(checkpoint['optimizer_state'])

        logger.info('==> Done')
        return it, epoch

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """Delegate to the student's dataset generate_prediction_dicts."""
        from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
        return NuScenesDataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path)
