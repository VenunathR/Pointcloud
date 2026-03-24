"""
Training script for PointNet++ semantic segmentation on .ply point clouds.

Usage:
    python train_semseg_ply.py --data_root data/ --epoch 32
    python train_semseg_ply.py --data_root data/ --epoch 64 --batch_size 8 --lr 0.001
"""

import argparse
import os
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils.ply_dataset import PLYDataset, NUM_CLASSES
from models.pointnet2_sem_seg import PointNet2SemSeg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='PointNet++ Semantic Segmentation Training')
    parser.add_argument('--data_root',  type=str,   default='data/',
                        help='Root directory containing train/val splits')
    parser.add_argument('--epoch',      type=int,   default=32,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,   default=16,
                        help='Batch size')
    parser.add_argument('--lr',         type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--num_points', type=int,   default=4096,
                        help='Number of points per sample')
    parser.add_argument('--log_dir',    type=str,   default='log/',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='LR decay factor (applied every 10 epochs)')
    parser.add_argument('--step_size',  type=int,   default=10,
                        help='Epoch interval for LR decay')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_iou_per_class(pred, target, num_classes):
    """Compute per-class IoU from flattened prediction / target arrays."""
    iou = np.zeros(num_classes)
    for cls in range(num_classes):
        intersection = np.sum((pred == cls) & (target == cls))
        union = np.sum((pred == cls) | (target == cls))
        iou[cls] = intersection / (union + 1e-10)
    return iou


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0

    for xyz, labels in loader:
        xyz    = xyz.to(device)                      # (B, 3, N)
        labels = labels.to(device).long()            # (B, N)

        optimizer.zero_grad()
        logits = model(xyz)                          # (B, N, C)
        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_points  += labels.numel()

    avg_loss = total_loss / total_points
    accuracy = total_correct / total_points
    return avg_loss, accuracy


@torch.no_grad()
def val_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    all_preds  = []
    all_labels = []

    for xyz, labels in loader:
        xyz    = xyz.to(device)
        labels = labels.to(device).long()

        logits = model(xyz)
        loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))

        total_loss += loss.item() * labels.numel()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_points  += labels.numel()
        all_preds.append(preds.cpu().numpy().ravel())
        all_labels.append(labels.cpu().numpy().ravel())

    avg_loss = total_loss / total_points
    accuracy = total_correct / total_points
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    iou = compute_iou_per_class(all_preds, all_labels, num_classes)
    miou = float(np.mean(iou))
    return avg_loss, accuracy, miou, iou


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Logging ----------------------------------------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'train_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger()
    log.info(f"Arguments: {args}")

    # ---- Device -----------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # ---- Datasets ---------------------------------------------------------
    train_dataset = PLYDataset(args.data_root, split='train',
                               num_points=args.num_points, augment=True)
    val_dataset   = PLYDataset(args.data_root, split='val',
                               num_points=args.num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ---- Model / optimizer / loss -----------------------------------------
    model = PointNet2SemSeg(num_classes=NUM_CLASSES).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.decay_rate
    )

    # ---- Training loop ----------------------------------------------------
    best_miou = 0.0
    best_ckpt = os.path.join(args.log_dir, 'best_model.pth')

    for epoch in range(1, args.epoch + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, miou, iou_per_class = val_epoch(
            model, val_loader, criterion, device, NUM_CLASSES
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:03d}/{args.epoch} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} mIoU={miou:.4f}"
        )

        iou_str = '  '.join(f'cls{c}={iou_per_class[c]:.3f}' for c in range(NUM_CLASSES))
        log.info(f"  Per-class IoU: {iou_str}")

        if miou > best_miou:
            best_miou = miou
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'miou': best_miou}, best_ckpt)
            log.info(f"  --> New best mIoU={best_miou:.4f}, checkpoint saved to {best_ckpt}")

    log.info(f"Training complete. Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
