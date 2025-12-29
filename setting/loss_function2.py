import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice损失：优化类别不平衡，提升边界分割精度"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth  # 防止分母为0

    def forward(self, pred, target):
        # pred: (B, 1, H, W) → 未经过sigmoid；target: (B, 1, H, W) → 0/1标签
        pred = torch.sigmoid(pred)  # 先转为概率
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice  # 损失越小越好


class IoULoss(nn.Module):
    """IoU损失：直接优化交并比，提升整体分割性能"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class SaliencyLoss(nn.Module):
    """显著性检测总损失：BCE（平衡类别） + Dice（优化边界） + IoU（提升交并比）"""
    def __init__(self, ):
        super().__init__()
        # BCE带正样本权重（解决前景少、背景多的类别不平衡）
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()

    def forward(self, pred, target):
        # pred: 模型输出的logits（未经过sigmoid）；target: 标签（0/1）
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        # 总损失 = 加权求和
        total_loss = bce + iou
        # 返回总损失和分项损失（便于日志分析）
        return {
            "total_loss": total_loss,
            "bce_loss": bce,
            "iou_loss": iou
        }
