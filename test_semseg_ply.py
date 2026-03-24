"""
Testing / inference script for PointNet++ semantic segmentation on .ply point clouds.

Usage:
    python test_semseg_ply.py --data_root data/ --checkpoint log/best_model.pth
    python test_semseg_ply.py --data_root data/ --checkpoint log/best_model.pth --split test
"""

import argparse
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils.ply_dataset import PLYDataset, NUM_CLASSES
from models.pointnet2_sem_seg import PointNet2SemSeg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='PointNet++ Semantic Segmentation Evaluation')
    parser.add_argument('--data_root',   type=str, default='data/',
                        help='Root directory containing dataset splits')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size',  type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num_points',  type=int, default=4096,
                        help='Number of points per sample')
    parser.add_argument('--log_dir',     type=str, default='log/',
                        help='Directory to save evaluation logs')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(pred, target, num_classes):
    """
    Compute per-class precision, recall, IoU and overall accuracy.

    Args:
        pred:        (M,) int array of predicted labels
        target:      (M,) int array of true labels
        num_classes: number of classes
    Returns:
        dict with keys: accuracy, miou, iou, precision, recall
    """
    iou       = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall    = np.zeros(num_classes)

    for cls in range(num_classes):
        tp = int(np.sum((pred == cls) & (target == cls)))
        fp = int(np.sum((pred == cls) & (target != cls)))
        fn = int(np.sum((pred != cls) & (target == cls)))
        union = tp + fp + fn

        iou[cls]       = tp / (union + 1e-10)
        precision[cls] = tp / (tp + fp + 1e-10)
        recall[cls]    = tp / (tp + fn + 1e-10)

    accuracy = float(np.mean(pred == target))
    miou     = float(np.mean(iou))

    return {
        'accuracy':  accuracy,
        'miou':      miou,
        'iou':       iou,
        'precision': precision,
        'recall':    recall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Logging ----------------------------------------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'eval_{args.split}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger()

    # ---- Device -----------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # ---- Dataset ----------------------------------------------------------
    dataset = PLYDataset(
        args.data_root, split=args.split,
        num_points=args.num_points, augment=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    # ---- Load model -------------------------------------------------------
    model = PointNet2SemSeg(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'N/A')
    ckpt_miou = checkpoint.get('miou', 'N/A')
    log.info(f"Loaded checkpoint from epoch {epoch} (training mIoU: {ckpt_miou})")
    model.eval()

    # ---- Inference --------------------------------------------------------
    all_preds        = []
    all_labels       = []
    all_confidences  = []

    with torch.no_grad():
        for batch_idx, (xyz, labels) in enumerate(loader):
            xyz    = xyz.to(device)
            labels = labels.to(device).long()

            logits = model(xyz)                         # (B, N, C)
            probs  = F.softmax(logits, dim=-1)          # (B, N, C)
            preds  = logits.argmax(dim=-1)              # (B, N)
            conf   = probs.max(dim=-1).values           # (B, N)

            all_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
            all_confidences.append(conf.cpu().numpy().ravel())

            if (batch_idx + 1) % 10 == 0:
                log.info(f"  Processed {(batch_idx + 1) * args.batch_size} samples...")

    all_preds       = np.concatenate(all_preds)
    all_labels      = np.concatenate(all_labels)
    all_confidences = np.concatenate(all_confidences)

    # ---- Metrics ----------------------------------------------------------
    metrics = compute_metrics(all_preds, all_labels, NUM_CLASSES)

    log.info("=" * 60)
    log.info(f"Evaluation results on '{args.split}' split")
    log.info(f"  Overall Accuracy : {metrics['accuracy']:.4f}")
    log.info(f"  Mean IoU (mIoU)  : {metrics['miou']:.4f}")
    log.info(f"  Mean Confidence  : {np.mean(all_confidences):.4f}")
    log.info("")
    log.info(f"{'Class':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
    log.info("-" * 40)
    for cls in range(NUM_CLASSES):
        log.info(
            f"  cls {cls:>2}  "
            f"{metrics['iou'][cls]:>8.4f}  "
            f"{metrics['precision'][cls]:>10.4f}  "
            f"{metrics['recall'][cls]:>8.4f}"
        )
    log.info("=" * 60)


if __name__ == '__main__':
    main()
