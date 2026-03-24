# Point Cloud Semantic Segmentation with PointNet++

Semantic segmentation of 3-D point clouds stored as `.ply` files using a
**PointNet++** encoder-decoder network.

- **Input**: x, y, z coordinates (`scalar_Classification` label field)
- **Classes**: 9 semantic classes (0 – 8)
- **Points per sample**: 4 096 (sampled / padded automatically)

---

## Repository structure

```
.
├── data_utils/
│   └── ply_dataset.py      # Custom PLY dataset loader
├── models/
│   └── pointnet2_sem_seg.py # PointNet++ encoder-decoder model
├── provider.py              # Augmentation utilities
├── train_semseg_ply.py      # Training script
├── test_semseg_ply.py       # Evaluation / inference script
├── requirements.txt
└── README.md
```

---

## Dataset layout

Place your `.ply` files in the following structure:

```
data/
├── train/   *.ply
├── val/     *.ply
└── test/    *.ply
```

Each `.ply` file must contain vertex properties `x`, `y`, `z` and
`scalar_Classification` (integer, 0 – 8).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train_semseg_ply.py \
    --data_root data/ \
    --epoch 32 \
    --batch_size 16 \
    --lr 0.001 \
    --log_dir log/
```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `data/` | Root directory with train/val/test sub-folders |
| `--epoch` | `32` | Number of training epochs |
| `--batch_size` | `16` | Mini-batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--num_points` | `4096` | Points sampled per file |
| `--log_dir` | `log/` | Log / checkpoint directory |
| `--decay_rate` | `0.7` | LR decay factor |
| `--step_size` | `10` | Epoch interval for LR decay |

The best checkpoint (by validation mIoU) is saved to `log/best_model.pth`.

---

## Evaluation

```bash
python test_semseg_ply.py \
    --data_root data/ \
    --checkpoint log/best_model.pth \
    --split test
```

Outputs per-class IoU, precision, recall, overall accuracy, and mean
confidence score.

---

## Model architecture

```
PointNet2SemSeg
  Encoder
    SA1  1024 pts  r=0.1  MLP [32, 32,  64]
    SA2   256 pts  r=0.2  MLP [64, 64, 128]
    SA3    64 pts  r=0.4  MLP [128, 128, 256]
    SA4    16 pts  r=0.8  MLP [256, 256, 512]
  Decoder
    FP4  → 256 channels
    FP3  → 128 channels
    FP2  → 128 channels
    FP1  → 128 channels
  Head
    Conv1d(128→128) + BN + ReLU + Dropout(0.5)
    Conv1d(128→9)
```
