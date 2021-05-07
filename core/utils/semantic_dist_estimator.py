import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn


class semantic_dist_estimator():
    def __init__(self, feature_num, cfg):
        super(semantic_dist_estimator, self).__init__()

        self.cfg = cfg
        self.class_num = cfg.MODEL.NUM_CLASSES
        _, backbone_name = cfg.MODEL.NAME.split('_')
        self.feature_num = 2048 if backbone_name.startswith('resnet') else 1024

        # init mean and covariance
        self.init(feature_num=feature_num, resume=self.cfg.CV_DIR)

    def init(self, feature_num, resume=""):
        if resume:
            if feature_num == self.cfg.MODEL.NUM_CLASSES:
                resume = os.path.join(resume, 'out_dist.pth')
            elif feature_num == self.feature_num:
                resume = os.path.join(resume, 'feat_dist.pth')
            else:
                raise RuntimeError("Feature_num not available: {}".format(feature_num))
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.CoVariance = checkpoint['CoVariance'].cuda(non_blocking=True)
            self.Mean = checkpoint['Mean'].cuda(non_blocking=True)
            self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.CoVariance = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
            self.Mean = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, features, labels):

        # label_mask = (labels == self.cfg.INPUT.IGNORE_LABEL).long()
        # labels = ((1 - label_mask).mul(labels) + label_mask * self.cfg.MODEL.NUM_CLASSES).long()

        mask = (labels != self.cfg.INPUT.IGNORE_LABEL)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]

        N, A = features.size()
        C = self.class_num

        NxCxA_Features = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxA_Features.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        mean_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - mean_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Mean - mean_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(
            weight_CV)).detach() + additional_CV.detach()

        self.Mean = (self.Mean.mul(1 - weight_CV) + mean_CxA.mul(weight_CV)).detach()

        self.Amount = self.Amount + onehot.sum(0)

    def save(self, name):
        torch.save({'CoVariance': self.CoVariance.cpu(),
                    'Mean': self.Mean.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.cfg.OUTPUT_DIR, name))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
