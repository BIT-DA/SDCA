import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(PixelContrastiveLoss, self).__init__()
        self.cfg = cfg

    def forward(self, Mean, CoVariance, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Mean: shape: (C, A) the mean representation of each class
            CoVariance: shape: (C, A) the diagonals of covariance matrices,
                        i.e., the variance  of each dimension of the features for each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Mean.requires_grad
        assert not CoVariance.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1

        # remove IGNORE_LABEL pixels
        mask = (labels != self.cfg.INPUT.IGNORE_LABEL)
        labels = labels[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        Mean = F.normalize(Mean, p=2, dim=1)
        CoVariance = F.normalize(CoVariance, p=2, dim=1)

        temp1 = feat.mm(Mean.permute(1, 0).contiguous())
        CoVariance = CoVariance / self.cfg.SOLVER.TAU
        temp2 = 0.5 * feat.pow(2).mm(CoVariance.permute(1, 0).contiguous())

        logits = temp1 + temp2
        logits = logits / self.cfg.SOLVER.TAU

        ce_criterion = nn.CrossEntropyLoss()
        ce_loss = ce_criterion(logits, labels)
        pcl_loss = 0.5 * torch.sum(feat.pow(2).mul(CoVariance[labels]), dim=1).mean() / self.cfg.SOLVER.TAU

        loss = ce_loss + pcl_loss
        return loss
