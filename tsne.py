import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import warnings

warnings.filterwarnings('ignore')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(feature_extractor, classifier, image, label):
    size = label.shape[-2:]
    with torch.no_grad():
        feat = feature_extractor(image)
        out = classifier(feat)
    pred = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
    pred = F.softmax(pred, dim=1)
    return pred, feat, out


cityspallete = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]) / 255.


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    map_color = {i: cityspallete[i] for i in range(19)}
    map_size = {i: 10 for i in range(19)}
    color = list(map(lambda x: map_color[x], label))
    size = list(map(lambda x: map_size[x], label))
    fig = plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=color, s=size)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def draw(tSNE_features, tSNE_labels, name, cfg):
    print("Generating T-SNE...")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tSNE_result = tsne.fit_transform(tSNE_features.data.cpu().numpy())
    tSNE_label = tSNE_labels.data.cpu().numpy()
    fig = plot_embedding(tSNE_result, tSNE_label,
                         name.split('.')[0]
                         )
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'inference', name), bbox_inches='tight')
    print('finished tsne, saved in', os.path.join(cfg.OUTPUT_DIR, 'inference', name))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("TSNE", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger = logging.getLogger("TSNE.tester")
    logger.info("Start")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()

    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TSNE.BATCH_SIZE, shuffle=False, num_workers=4,

                                              pin_memory=True, sampler=None)
    for batch in tqdm(test_loader):
        x, y, name = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()

        pred, feat, outs = inference(feature_extractor, classifier, x, y)
        filename = name[0] if len(name[0].split("/")) < 2 else name[0].split("/")[1]

        # draw t-sne
        B, A, Ht, Wt = outs.size()
        tSNE_features = outs.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)
        tSNE_labels = F.interpolate(y.unsqueeze(0).float(), size=(Ht, Wt), mode='nearest').squeeze(0).long()
        tSNE_labels = tSNE_labels.contiguous().view(B * Ht * Wt, )

        mask = (tSNE_labels != cfg.INPUT.IGNORE_LABEL)  # remove IGNORE_LABEL pixels
        tSNE_labels = tSNE_labels[mask]
        tSNE_features = tSNE_features[mask]

        draw(tSNE_features=tSNE_features, tSNE_labels=tSNE_labels, name=filename, cfg=cfg)


if __name__ == "__main__":
    main()
