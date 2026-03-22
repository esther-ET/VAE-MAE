import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAdaptation(nn.Module):
    """1x1 conv to match channel dimensions between teacher and student features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class FeatureDistillLoss(nn.Module):
    """
    Feature-level knowledge distillation loss.
    Aligns intermediate features between teacher and student branches.
    Supports MSE, Cosine similarity, and L1 loss types.
    """

    def __init__(self, loss_type='mse', teacher_channels=None, student_channels=None):
        super().__init__()
        self.loss_type = loss_type

        self.adaptation = None
        if teacher_channels is not None and student_channels is not None \
                and teacher_channels != student_channels:
            self.adaptation = ChannelAdaptation(student_channels, teacher_channels)

    def forward(self, teacher_feat, student_feat):
        """
        Args:
            teacher_feat: (B, C_t, H, W) teacher feature map (detached)
            student_feat: (B, C_s, H, W) student feature map
        Returns:
            loss: scalar tensor
        """
        if self.adaptation is not None:
            student_feat = self.adaptation(student_feat)

        teacher_feat = teacher_feat.detach()

        if self.loss_type == 'mse':
            return F.mse_loss(student_feat, teacher_feat)
        elif self.loss_type == 'l1':
            return F.l1_loss(student_feat, teacher_feat)
        elif self.loss_type == 'cosine':
            b, c, h, w = teacher_feat.shape
            t_flat = teacher_feat.reshape(b, c, -1).permute(0, 2, 1)
            s_flat = student_feat.reshape(b, c, -1).permute(0, 2, 1)
            cos_sim = F.cosine_similarity(t_flat, s_flat, dim=-1)
            return (1 - cos_sim).mean()
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(student_feat, teacher_feat)
        else:
            raise ValueError(f'Unknown feature distill loss type: {self.loss_type}')


class PredictionDistillLoss(nn.Module):
    """
    Prediction-level knowledge distillation loss for CenterHead outputs.
    Aligns heatmap predictions (via KL divergence or MSE) and
    regression outputs (via Smooth L1) between teacher and student.
    """

    def __init__(self, heatmap_loss_type='kl_div', regression_loss_type='smooth_l1',
                 heatmap_weight=0.5, regression_weight=0.25, score_thresh=0.3):
        super().__init__()
        self.heatmap_loss_type = heatmap_loss_type
        self.regression_loss_type = regression_loss_type
        self.heatmap_weight = heatmap_weight
        self.regression_weight = regression_weight
        self.score_thresh = score_thresh

    def _heatmap_loss(self, teacher_hm, student_hm):
        teacher_hm = teacher_hm.detach()
        t_prob = torch.clamp(teacher_hm.sigmoid(), min=1e-6, max=1 - 1e-6)
        s_prob = torch.clamp(student_hm.sigmoid(), min=1e-6, max=1 - 1e-6)

        if self.heatmap_loss_type == 'kl_div':
            t_log = torch.log(t_prob)
            s_log = torch.log(s_prob)
            loss = t_prob * (t_log - s_log) + (1 - t_prob) * (torch.log(1 - t_prob) - torch.log(1 - s_prob))
            return loss.mean()
        elif self.heatmap_loss_type == 'mse':
            return F.mse_loss(s_prob, t_prob)
        else:
            raise ValueError(f'Unknown heatmap distill loss type: {self.heatmap_loss_type}')

    def _regression_loss(self, teacher_preds, student_preds, teacher_hm):
        """Only compute regression distillation at high-confidence teacher locations."""
        teacher_score = teacher_hm.detach().sigmoid()
        mask = (teacher_score.max(dim=1, keepdim=True)[0] > self.score_thresh).float()

        if mask.sum() < 1:
            return torch.tensor(0.0, device=teacher_hm.device)

        loss = 0.0
        count = 0
        for t_pred, s_pred in zip(teacher_preds, student_preds):
            t_val = t_pred.detach()
            if t_val.shape[1] != s_pred.shape[1]:
                continue
            diff = F.smooth_l1_loss(s_pred * mask, t_val * mask, reduction='sum')
            loss = loss + diff / (mask.sum() * t_val.shape[1] + 1e-6)
            count += 1

        return loss / max(count, 1)

    def forward(self, teacher_pred_dicts, student_pred_dicts):
        """
        Args:
            teacher_pred_dicts: list of dicts, each with 'hm', 'center', 'center_z', 'dim', 'rot', 'vel'
            student_pred_dicts: same structure
        Returns:
            total_loss: scalar
            tb_dict: dict of sub-losses for logging
        """
        total_hm_loss = 0.0
        total_reg_loss = 0.0

        for t_dict, s_dict in zip(teacher_pred_dicts, student_pred_dicts):
            t_hm = t_dict['hm']
            s_hm = s_dict['hm']
            total_hm_loss = total_hm_loss + self._heatmap_loss(t_hm, s_hm)

            reg_keys = [k for k in t_dict.keys() if k != 'hm']
            t_regs = [t_dict[k] for k in reg_keys if k in s_dict]
            s_regs = [s_dict[k] for k in reg_keys if k in s_dict]
            total_reg_loss = total_reg_loss + self._regression_loss(t_regs, s_regs, t_hm)

        n_heads = max(len(teacher_pred_dicts), 1)
        total_hm_loss = total_hm_loss / n_heads
        total_reg_loss = total_reg_loss / n_heads

        loss = self.heatmap_weight * total_hm_loss + self.regression_weight * total_reg_loss
        tb_dict = {
            'distill_hm_loss': total_hm_loss.item() if torch.is_tensor(total_hm_loss) else total_hm_loss,
            'distill_reg_loss': total_reg_loss.item() if torch.is_tensor(total_reg_loss) else total_reg_loss,
        }
        return loss, tb_dict
