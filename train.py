import os
import sys
import json
import time
import random
import shutil
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.1)
except ImportError:
    sns = None

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, cohen_kappa_score,
    brier_score_loss
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

warnings.filterwarnings('ignore')

def setup_logging(output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    logger = logging.getLogger("btxrd_bone_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@dataclass
class Config:


    BASE_DIR: str = os.environ.get("BTXRD_DATA_DIR", "data/btxrd")
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "outputs")


    USE_LOCAL_CACHE: bool = False
    LOCAL_CACHE_DIR: str = os.environ.get("LOCAL_CACHE_DIR", "cache/btxrd")


    BACKBONE: str = "tf_efficientnetv2_s"
    IMG_SIZE: int = 384
    NUM_CLASSES: int = 3
    CLASS_NAMES: tuple = ("Normal", "Benign", "Malignant")
    DROP_RATE: float = 0.30
    META_DIM: int = 11
    META_HIDDEN: int = 32


    ROI_MARGIN: float = 0.25
    ROI_MARGIN_JITTER: float = 0.10


    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    TOTAL_EPOCHS: int = 70
    HEAD_EPOCHS: int = 5
    HEAD_LR: float = 1e-3
    BACKBONE_LR: float = 2e-5
    WEIGHT_DECAY: float = 0.01
    LABEL_SMOOTHING: float = 0.05
    GRAD_CLIP: float = 1.0


    FOCAL_GAMMA: float = 2.0


    MIXUP_ALPHA: float = 0.3
    MIXUP_PROB: float = 0.3
    CUTMIX_PROB: float = 0.2


    SWA_START_FRAC: float = 0.70
    SWA_LR: float = 1e-5


    PATIENCE: int = 15


    OVERFIT_GAP_THRESHOLD: float = 0.08
    OVERFIT_DROPOUT_BOOST: float = 0.05


    TTA_COUNT: int = 4


    N_FOLDS: int = 5
    SEED: int = 42
    SEEDS: tuple = (42, 123, 7)


    USE_ROI: bool = True
    USE_METADATA: bool = True


    BACKBONES: tuple = ("tf_efficientnetv2_s", "resnet50")


    FREEZE_BLOCKS: int = 4


    FIGURE_DPI: int = 300
    N_GRADCAM_SAMPLES: int = 8
    N_PREDICTION_SAMPLES: int = 5


    IMAGE_DIR: str = ""
    ANN_DIR: str = ""
    XLSX_PATH: str = ""
    FIGURES_DIR: str = ""
    TABLES_DIR: str = ""
    CHECKPOINT_DIR: str = ""
    GRADCAM_DIR: str = ""

    def __post_init__(self):
        self.IMAGE_DIR = os.path.join(self.BASE_DIR, "images")
        self.ANN_DIR = os.path.join(self.BASE_DIR, "Annotations")
        self.XLSX_PATH = os.path.join(self.BASE_DIR, "dataset.xlsx")
        self.FIGURES_DIR = os.path.join(self.OUTPUT_DIR, "figures")
        self.TABLES_DIR = os.path.join(self.OUTPUT_DIR, "tables")
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.GRADCAM_DIR = os.path.join(self.OUTPUT_DIR, "gradcam")
        for d in [self.OUTPUT_DIR, self.FIGURES_DIR, self.TABLES_DIR,
                  self.CHECKPOINT_DIR, self.GRADCAM_DIR]:
            os.makedirs(d, exist_ok=True)


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop_roi(img: np.ndarray, roi: dict, margin: float,
             jitter: float = 0.0, is_train: bool = False) -> np.ndarray:
\
\
\
\
\
\
\
\
\
\

    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
    orig_w, orig_h = roi['img_w'], roi['img_h']


    sx = w / orig_w if orig_w > 0 else 1.0
    sy = h / orig_h if orig_h > 0 else 1.0
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy


    effective_margin = margin
    if is_train and jitter > 0:
        effective_margin += random.uniform(-jitter * 0.5, jitter)

    bw, bh = x2 - x1, y2 - y1
    mx = bw * effective_margin
    my = bh * effective_margin

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w, int(x2 + mx))
    cy2 = min(h, int(y2 + my))


    if cx2 - cx1 < 32 or cy2 - cy1 < 32:
        return img

    return img[cy1:cy2, cx1:cx2]


@torch.no_grad()
def custom_swa_update_bn(loader, model, device):

    momenta = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.running_mean is not None:
                module.running_mean.zero_()
                module.running_var.fill_(1)
                momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for m in momenta:
        m.momentum = None

    for images, metadata, _ in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        model(images, metadata)

    for m, mom in momenta.items():
        m.momentum = mom
    model.train(was_training)


def prepare_local_cache(config, logger):
\
\
\
\
\
\

    if not config.USE_LOCAL_CACHE:
        logger.info("Local cache disabled or not on notebook — using remote storage paths")
        return config.BASE_DIR

    local_base = config.LOCAL_CACHE_DIR
    local_img_dir = os.path.join(local_base, "images")
    local_ann_dir = os.path.join(local_base, "Annotations")
    local_xlsx = os.path.join(local_base, "dataset.xlsx")


    if (os.path.exists(local_img_dir)
            and len(os.listdir(local_img_dir)) >= 3700
            and os.path.exists(local_ann_dir)
            and os.path.exists(local_xlsx)):
        logger.info(f"Local cache found: {local_base}")
        return local_base

    logger.info("="*60)
    logger.info("Caching dataset locally")
    logger.info("Bu i\u015flem bir kez yap\u0131l\u0131r ve e\u011fitimi ~10\u00d7 h\u0131zland\u0131r\u0131r.")
    logger.info("="*60)
    os.makedirs(local_base, exist_ok=True)


    src_img = config.IMAGE_DIR
    if os.path.exists(src_img):
        os.makedirs(local_img_dir, exist_ok=True)
        files = os.listdir(src_img)
        for fname in tqdm(files, desc="G\u00f6r\u00fcnt\u00fcler", unit="img", ncols=80):
            dst = os.path.join(local_img_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src_img, fname), dst)
        logger.info(f"  \u2713 {len(os.listdir(local_img_dir))} g\u00f6r\u00fcnt\u00fc kopyaland\u0131")
    else:
        logger.warning(f"  images klas\u00f6r\u00fc bulunamad\u0131: {src_img}")
        return config.BASE_DIR


    src_ann = config.ANN_DIR
    if os.path.exists(src_ann):
        os.makedirs(local_ann_dir, exist_ok=True)
        for fname in os.listdir(src_ann):
            dst = os.path.join(local_ann_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src_ann, fname), dst)
        logger.info(f"  \u2713 {len(os.listdir(local_ann_dir))} annotation kopyaland\u0131")


    src_xlsx = config.XLSX_PATH
    if os.path.exists(src_xlsx) and not os.path.exists(local_xlsx):
        shutil.copy2(src_xlsx, local_xlsx)
        logger.info("  \u2713 dataset.xlsx kopyaland\u0131")

    logger.info("="*60)
    logger.info(f"\u2713 Veri local SSD'ye kopyaland\u0131: {local_base}")
    logger.info("="*60)
    return local_base


def mixup_data(x, y, alpha=0.3):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
    return x, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
class DataModule:


    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.df: Optional[pd.DataFrame] = None
        self.roi_map: Dict[str, dict] = {}

    def load(self) -> pd.DataFrame:

        log = self.logger
        cfg = self.config


        log.info(f"Loading {cfg.XLSX_PATH}")
        df = pd.read_excel(cfg.XLSX_PATH)
        df['img_id'] = df['image_id'].str.replace(r'\.(jpeg|jpg)$', '', regex=True)


        df['label'] = 'Normal'
        df.loc[df['benign'] == 1, 'label'] = 'Benign'
        df.loc[df['malignant'] == 1, 'label'] = 'Malignant'
        df['label_idx'] = df['label'].map({'Normal': 0, 'Benign': 1, 'Malignant': 2})


        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(cfg.IMAGE_DIR, x))


        sample = df['image_path'].iloc[0]
        if not os.path.exists(sample):
            log.warning(f"Image not found: {sample}  — check BASE_DIR!")


        df['gender_enc'] = (df['gender'] == 'M').astype(np.float32)
        df['center_1'] = (df['center'] == 1).astype(np.float32)
        df['center_2'] = (df['center'] == 2).astype(np.float32)
        df['center_3'] = (df['center'] == 3).astype(np.float32)

        meta_cols = [
            'age', 'gender_enc',
            'center_1', 'center_2', 'center_3',
            'upper limb', 'lower limb', 'pelvis',
            'frontal', 'lateral', 'oblique'
        ]
        self._meta_array = df[meta_cols].values.astype(np.float32)


        self._parse_annotations()
        df['has_roi'] = df['img_id'].isin(self.roi_map)


        lc = df['label'].value_counts()
        log.info(f"Dataset: {len(df)} images")
        log.info(f"  Normal:    {lc.get('Normal', 0)}")
        log.info(f"  Benign:    {lc.get('Benign', 0)}")
        log.info(f"  Malignant: {lc.get('Malignant', 0)}")
        log.info(f"  ROI annotations: {df['has_roi'].sum()}")

        self.df = df
        return df

    def _parse_annotations(self):

        ann_dir = self.config.ANN_DIR
        count = 0

        for fname in os.listdir(ann_dir):
            if not fname.endswith('.json'):
                continue
            img_id = fname[:-5]
            fpath = os.path.join(ann_dir, fname)
            try:
                with open(fpath, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            img_w = data.get('imageWidth', 1)
            img_h = data.get('imageHeight', 1)


            min_x = min_y = float('inf')
            max_x = max_y = 0
            found = False

            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'rectangle':
                    pts = shape.get('points', [])
                    if len(pts) >= 2:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        min_x = min(min_x, min(xs))
                        min_y = min(min_y, min(ys))
                        max_x = max(max_x, max(xs))
                        max_y = max(max_y, max(ys))
                        found = True

            if found:
                self.roi_map[img_id] = {
                    'x1': min_x, 'y1': min_y,
                    'x2': max_x, 'y2': max_y,
                    'img_w': img_w, 'img_h': img_h,
                }
                count += 1

        self.logger.info(f"  Parsed {count} ROI annotations")

    def get_metadata(self, indices, train_indices=None):
\
\
\
\
\
\

        meta = self._meta_array[indices].copy()
        if train_indices is not None:
            train_ages = self._meta_array[train_indices, 0]
            age_mean, age_std = train_ages.mean(), train_ages.std()
        else:
            age_mean, age_std = self._meta_array[:, 0].mean(), self._meta_array[:, 0].std()
        meta[:, 0] = (meta[:, 0] - age_mean) / (age_std + 1e-8)
        return meta


class BoneDataset(Dataset):


    def __init__(self, df: pd.DataFrame, metadata: np.ndarray,
                 roi_map: dict, config: Config,
                 transform=None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.metadata = metadata.astype(np.float32)
        self.roi_map = roi_map
        self.config = config
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]


        try:
            img = Image.open(row['image_path']).convert('RGB')
            img_np = np.array(img)
        except Exception:

            img_np = np.zeros((self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=np.uint8)


        img_id = row['img_id']
        if self.config.USE_ROI:
            if img_id in self.roi_map:

                img_np = crop_roi(
                    img_np, self.roi_map[img_id],
                    margin=self.config.ROI_MARGIN,
                    jitter=self.config.ROI_MARGIN_JITTER,
                    is_train=self.is_train
                )
            else:

                h, w = img_np.shape[:2]
                crop_frac = 0.70
                ch, cw = int(h * crop_frac), int(w * crop_frac)
                y1 = (h - ch) // 2
                x1 = (w - cw) // 2
                img_np = img_np[y1:y1+ch, x1:x1+cw]


        if self.transform:
            transformed = self.transform(image=img_np)
            img_tensor = transformed['image']
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        meta_tensor = torch.from_numpy(self.metadata[idx])
        label = torch.tensor(row['label_idx'], dtype=torch.long)

        return img_tensor, meta_tensor, label


def get_train_transforms(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=0, p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            A.CLAHE(clip_limit=2.0),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0, p=0.2
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int):

    norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = A.Resize(img_size, img_size)
    return [
        A.Compose([resize, norm, ToTensorV2()]),
        A.Compose([resize, A.HorizontalFlip(p=1.0), norm, ToTensorV2()]),
        A.Compose([
            resize,
            A.Rotate(limit=(5, 5), border_mode=0, p=1.0),
            norm, ToTensorV2()
        ]),
        A.Compose([
            resize,
            A.Rotate(limit=(-5, -5), border_mode=0, p=1.0),
            norm, ToTensorV2()
        ]),
    ]


class FocalLoss(nn.Module):


    def __init__(self, alpha=None, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()


        if self.label_smoothing > 0:
            with torch.no_grad():
                dist = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            dist = F.one_hot(targets, num_classes).float()


        focal_weight = (1.0 - probs) ** self.gamma
        loss = -(focal_weight * log_probs * dist).sum(dim=1)


        if self.alpha is not None:
            alpha_w = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_w

        return loss.mean()


class BoneClassifier(nn.Module):
\
\
\


    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.use_metadata = config.USE_METADATA


        backbone_name = config.BACKBONE
        try:
            self.backbone = timm.create_model(
                backbone_name + ".in21k_ft_in1k",
                pretrained=True, drop_rate=config.DROP_RATE, num_classes=0
            )
        except Exception:
            try:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=True, drop_rate=config.DROP_RATE, num_classes=0
                )
            except Exception:

                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=True, num_classes=0
                )
        self.backbone_dim = self.backbone.num_features


        if self.use_metadata:
            self.meta_net = nn.Sequential(
                nn.Linear(config.META_DIM, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, config.META_HIDDEN),
                nn.BatchNorm1d(config.META_HIDDEN),
                nn.GELU(),
            )
            combined = self.backbone_dim + config.META_HIDDEN
        else:
            self.meta_net = None
            combined = self.backbone_dim


        self.head = nn.Sequential(
            nn.Linear(combined, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config.DROP_RATE),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, images, metadata):
        img_feat = self.backbone(images)
        if self.use_metadata and self.meta_net is not None:
            meta_feat = self.meta_net(metadata)
            combined = torch.cat([img_feat, meta_feat], 1)
        else:
            combined = img_feat
        return self.head(combined)

    def freeze_backbone(self, num_blocks: int):

        try:
            for p in self.backbone.conv_stem.parameters():
                p.requires_grad = False
            for p in self.backbone.bn1.parameters():
                p.requires_grad = False
            for i, block in enumerate(self.backbone.blocks):
                if i < num_blocks:
                    for p in block.parameters():
                        p.requires_grad = False
        except AttributeError:

            frozen = 0
            for name, child in self.backbone.named_children():
                if frozen < num_blocks:
                    for p in child.parameters():
                        p.requires_grad = False
                    frozen += 1

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        return sum(p.numel() for p in self.parameters())


class Trainer:


    def __init__(self, model: nn.Module, config: Config,
                 device: torch.device, class_weights: list, logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger

        alpha = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = FocalLoss(
            alpha=alpha, gamma=config.FOCAL_GAMMA,
            label_smoothing=config.LABEL_SMOOTHING
        )

        self.best_state = None
        self.best_bal_acc = 0.0
        self.history = []


    def fit(self, train_loader, val_loader, fold: int):

        cfg = self.config
        log = self.logger


        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold+1} | Phase 1: Head warm-up ({cfg.HEAD_EPOCHS} epochs)")
        log.info(f"{'='*60}")

        trainable, total = self.model.freeze_backbone(cfg.FREEZE_BLOCKS)
        log.info(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.HEAD_LR, weight_decay=cfg.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.HEAD_EPOCHS
        )

        for epoch in range(cfg.HEAD_EPOCHS):
            t_loss, t_acc = self._train_epoch(train_loader, optimizer)
            v_loss, v_acc, v_bal, _ = self._validate(val_loader)
            scheduler.step()
            log.info(
                f"  E{epoch+1}/{cfg.HEAD_EPOCHS} | "
                f"trn loss={t_loss:.4f} acc={t_acc:.4f} | "
                f"val loss={v_loss:.4f} acc={v_acc:.4f} bal={v_bal:.4f}"
            )


        ft_epochs = cfg.TOTAL_EPOCHS - cfg.HEAD_EPOCHS
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold+1} | Phase 2: Full fine-tune ({ft_epochs} epochs)")
        log.info(f"{'='*60}")

        n_params = self.model.unfreeze_all()
        log.info(f"  All unfrozen: {n_params/1e6:.1f}M params")


        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.head.parameters())
        if self.model.meta_net is not None:
            head_params = list(self.model.meta_net.parameters()) + head_params
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': cfg.BACKBONE_LR},
            {'params': head_params, 'lr': cfg.HEAD_LR * 0.5},
        ], weight_decay=cfg.WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ft_epochs
        )


        swa_start = int(ft_epochs * cfg.SWA_START_FRAC)
        swa_model = AveragedModel(self.model)
        swa_sched = SWALR(optimizer, swa_lr=cfg.SWA_LR)
        swa_active = False

        self.best_bal_acc = 0.0
        self.best_state = None
        patience_ctr = 0

        for epoch in range(ft_epochs):
            t_loss, t_acc = self._train_epoch(train_loader, optimizer)
            v_loss, v_acc, v_bal, _ = self._validate(val_loader)


            if epoch >= swa_start:
                if not swa_active:
                    log.info(f"  >>> SWA activated at epoch {cfg.HEAD_EPOCHS + epoch + 1}")
                    swa_active = True
                swa_model.update_parameters(self.model)
                swa_sched.step()
            else:
                scheduler.step()


            gap = t_acc - v_acc
            self.history.append({
                'epoch': cfg.HEAD_EPOCHS + epoch + 1,
                'train_loss': t_loss, 'train_acc': t_acc,
                'val_loss': v_loss, 'val_acc': v_acc,
                'val_bal_acc': v_bal, 'gap': gap
            })


            if gap > cfg.OVERFIT_GAP_THRESHOLD:
                for m in self.model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = min(0.5, m.p + cfg.OVERFIT_DROPOUT_BOOST)
                if epoch > 0 and self.history[-2]['gap'] <= cfg.OVERFIT_GAP_THRESHOLD:
                    log.info(f"  ⚠ Overfit detected (gap={gap:.3f}), dropout boosted")


            marker = ""
            if v_bal > self.best_bal_acc:
                self.best_bal_acc = v_bal
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}
                patience_ctr = 0
                marker = " ★"
            else:
                patience_ctr += 1

            log.info(
                f"  E{cfg.HEAD_EPOCHS+epoch+1}/{cfg.TOTAL_EPOCHS} | "
                f"trn {t_loss:.4f}/{t_acc:.4f} | "
                f"val {v_loss:.4f}/{v_acc:.4f}/{v_bal:.4f} | "
                f"gap={gap:+.3f} pat={patience_ctr}{marker}"
            )


            if (epoch + 1) % 10 == 0 or marker:
                ckpt = {
                    'fold': fold, 'epoch': cfg.HEAD_EPOCHS + epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_bal_acc': self.best_bal_acc,
                    'history': self.history,
                }
                ckpt_path = os.path.join(
                    cfg.CHECKPOINT_DIR, f"fold{fold+1}_epoch{cfg.HEAD_EPOCHS+epoch+1}.pt")
                torch.save(ckpt, ckpt_path)

            if patience_ctr >= cfg.PATIENCE:
                log.info(f"  Early stopping at epoch {cfg.HEAD_EPOCHS+epoch+1}")
                break


        if swa_active:
            custom_swa_update_bn(train_loader, swa_model, self.device)
            log.info("  SWA BN statistics updated")


        if self.best_state:
            self.model.load_state_dict(self.best_state)
            log.info(f"  Best model restored (val_bal_acc={self.best_bal_acc:.4f})")

        return self.model, swa_model if swa_active else None


    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0
        correct = total = 0
        cfg = self.config

        pbar = tqdm(loader, desc="  train", leave=False, ncols=100,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for images, meta, labels in pbar:
            images = images.to(self.device)
            meta = meta.to(self.device)
            labels = labels.to(self.device)


            use_mix = False
            r = random.random()
            if r < cfg.MIXUP_PROB:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, cfg.MIXUP_ALPHA)
                use_mix = True
            elif r < cfg.MIXUP_PROB + cfg.CUTMIX_PROB:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)
                use_mix = True

            optimizer.zero_grad()
            logits = self.model(images, meta)

            if use_mix:
                loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
            else:
                loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.GRAD_CLIP
            )
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

        return total_loss / total, correct / total


    @torch.no_grad()
    def _validate(self, loader, model=None):

        mdl = model or self.model
        mdl.eval()

        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        total = 0

        for images, meta, labels in loader:
            images = images.to(self.device)
            meta = meta.to(self.device)
            labels = labels.to(self.device)

            logits = mdl(images, meta)
            loss = self.criterion(logits, labels)

            probs = F.softmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total += labels.size(0)

        preds = np.array(all_preds)
        labels_np = np.array(all_labels)
        acc = accuracy_score(labels_np, preds)
        bal = balanced_accuracy_score(labels_np, preds)

        return total_loss / total, acc, bal, np.array(all_probs)


    @torch.no_grad()
    def predict_tta(self, test_df, roi_map, metadata):

        self.model.eval()
        cfg = self.config
        tta_tfms = get_tta_transforms(cfg.IMG_SIZE)
        n = len(test_df)

        all_probs = np.zeros((n, cfg.NUM_CLASSES), dtype=np.float64)

        for idx in range(n):
            row = test_df.iloc[idx]


            try:
                img = Image.open(row['image_path']).convert('RGB')
                img_np = np.array(img)
            except Exception:
                img_np = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)


            if cfg.USE_ROI:
                if row['img_id'] in roi_map:
                    img_np = crop_roi(img_np, roi_map[row['img_id']],
                                      margin=cfg.ROI_MARGIN, is_train=False)
                else:

                    h, w = img_np.shape[:2]
                    crop_frac = 0.70
                    ch, cw = int(h * crop_frac), int(w * crop_frac)
                    y1 = (h - ch) // 2
                    x1 = (w - cw) // 2
                    img_np = img_np[y1:y1+ch, x1:x1+cw]


            tensors = []
            for tfm in tta_tfms:
                t = tfm(image=img_np)['image']
                tensors.append(t)
            batch = torch.stack(tensors).to(self.device)

            meta_t = torch.from_numpy(metadata[idx:idx+1]).to(self.device)
            meta_batch = meta_t.expand(len(tta_tfms), -1)

            logits = self.model(batch, meta_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs[idx] = probs.mean(axis=0)

        return all_probs


class CrossValidator:


    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.fold_results = []

    def run(self, df: pd.DataFrame, data_module: DataModule):

        cfg = self.config
        log = self.logger
        self._df_ref = df
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Device: {device}")
        if device.type == 'cuda':
            log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


        counts = df['label_idx'].value_counts().sort_index().values
        n_total = len(df)
        class_weights = [n_total / (cfg.NUM_CLASSES * c) for c in counts]
        log.info(f"Class weights: {[f'{w:.3f}' for w in class_weights]}")


        skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                              random_state=cfg.SEED)

        all_test_probs, all_test_labels = [], []
        all_test_dfs, all_histories = [], []
        total_start = time.time()
        ckpt_mgr = CheckpointManager(cfg)


        completed_folds = set()
        resume_state = ckpt_mgr.load_cv_state()
        if resume_state and resume_state.get('completed_folds'):
            for prev_fold in sorted(resume_state['completed_folds']):
                saved = ckpt_mgr.load_fold_results(prev_fold - 1)
                if saved is not None:
                    self.fold_results.append(saved['fold_result'])
                    all_test_probs.append(saved['fold_result']['probs'])
                    all_test_labels.append(saved['fold_result']['labels'])
                    all_histories.append(saved['history'])
                    completed_folds.add(prev_fold)
                    log.info(f"Fold {prev_fold} loaded from checkpoint")
                else:
                    log.warning(f"Fold {prev_fold} checkpoint missing, retraining from here")
                    break
            if completed_folds:
                log.info(f"Resume: {len(completed_folds)} folds loaded, "
                         f"continuing from fold {max(completed_folds) + 1}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label_idx'])):

            if fold + 1 in completed_folds:
                continue

            fold_start = time.time()
            log.info(f"\n{'#'*70}")
            log.info(f"# FOLD {fold+1}/{cfg.N_FOLDS}")
            log.info(f"{'#'*70}")


            train_full_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            val_splitter = StratifiedKFold(n_splits=7, shuffle=True,
                                           random_state=cfg.SEED + fold)
            tr_sub, val_sub = next(val_splitter.split(
                train_full_df, train_full_df['label_idx']))

            train_df = train_full_df.iloc[tr_sub]
            val_df = train_full_df.iloc[val_sub]

            log.info(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
            for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
                dist = dict(Counter(split_df['label'].values))
                log.info(f"    {split_name}: {dist}")


            train_indices = train_df.index.values
            train_meta = data_module.get_metadata(train_indices, train_indices=train_indices)
            val_meta = data_module.get_metadata(val_df.index.values, train_indices=train_indices)
            test_meta = data_module.get_metadata(test_df.index.values, train_indices=train_indices)


            train_ds = BoneDataset(train_df, train_meta, data_module.roi_map,
                                   cfg, get_train_transforms(cfg.IMG_SIZE), is_train=True)
            val_ds = BoneDataset(val_df, val_meta, data_module.roi_map,
                                 cfg, get_val_transforms(cfg.IMG_SIZE), is_train=False)
            test_ds = BoneDataset(test_df, test_meta, data_module.roi_map,
                                  cfg, get_val_transforms(cfg.IMG_SIZE), is_train=False)


            train_loader = DataLoader(
                train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                num_workers=cfg.NUM_WORKERS, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                num_workers=cfg.NUM_WORKERS, pin_memory=True
            )


            model = BoneClassifier(cfg)
            trainer = Trainer(model, cfg, device, class_weights, log)


            best_model, swa_model = trainer.fit(train_loader, val_loader, fold)


            log.info(f"\n  --- Strategy Selection on Val (Fold {fold+1}) ---")


            _, val_std_acc, val_std_bal, _ = trainer._validate(val_loader)
            log.info(f"  Val Standard: bal={val_std_bal:.4f}")

            val_tta_probs = trainer.predict_tta(val_df, data_module.roi_map, val_meta)
            val_tta_preds = val_tta_probs.argmax(1)
            val_tta_bal = balanced_accuracy_score(val_df['label_idx'].values, val_tta_preds)
            log.info(f"  Val TTA:      bal={val_tta_bal:.4f}")

            val_strategy_map = {
                'standard': val_std_bal,
                'tta': val_tta_bal,
            }
            if swa_model is not None:
                _, val_swa_acc, val_swa_bal, _ = trainer._validate(
                    val_loader, model=swa_model)
                val_strategy_map['swa'] = val_swa_bal
                log.info(f"  Val SWA:      bal={val_swa_bal:.4f}")

            best_strategy = max(val_strategy_map, key=lambda k: val_strategy_map[k])
            log.info(f"  >>> Best strategy (chosen on val): {best_strategy}")


            log.info(f"\n  --- Test Evaluation (Fold {fold+1}) ---")


            _, std_acc, std_bal, std_probs = trainer._validate(test_loader)
            test_labels = test_df['label_idx'].values
            log.info(f"  Standard: acc={std_acc:.4f} bal={std_bal:.4f}")


            log.info("  Running TTA on test...")
            tta_probs = trainer.predict_tta(test_df, data_module.roi_map, test_meta)
            tta_preds = tta_probs.argmax(1)
            tta_acc = accuracy_score(test_labels, tta_preds)
            tta_bal = balanced_accuracy_score(test_labels, tta_preds)
            log.info(f"  TTA:      acc={tta_acc:.4f} bal={tta_bal:.4f}")


            swa_bal = 0.0
            if swa_model is not None:
                _, swa_acc, swa_bal, swa_probs = trainer._validate(
                    test_loader, model=swa_model)
                log.info(f"  SWA:      acc={swa_acc:.4f} bal={swa_bal:.4f}")


            test_results_map = {
                'standard': (std_bal, std_probs),
                'tta': (tta_bal, tta_probs),
            }
            if swa_model is not None:
                test_results_map['swa'] = (swa_bal, swa_probs)

            final_bal, final_probs = test_results_map[best_strategy]
            final_preds = final_probs.argmax(1)
            final_acc = accuracy_score(test_labels, final_preds)
            log.info(f"  Applied strategy: {best_strategy} (test bal_acc={final_bal:.4f})")


            report = classification_report(
                test_labels, final_preds,
                target_names=list(cfg.CLASS_NAMES),
                output_dict=True, zero_division=0
            )
            try:
                auc = roc_auc_score(test_labels, final_probs,
                                    multi_class='ovr', average='macro')
            except Exception:
                auc = 0.0

            cm = confusion_matrix(test_labels, final_preds)

            fold_result = {
                'fold': fold + 1,
                'strategy': best_strategy,
                'acc': final_acc,
                'bal_acc': final_bal,
                'f1_macro': f1_score(test_labels, final_preds, average='macro'),
                'auc_macro': auc,
                'precision_macro': precision_score(
                    test_labels, final_preds, average='macro', zero_division=0),
                'recall_macro': recall_score(
                    test_labels, final_preds, average='macro', zero_division=0),
                'cm': cm,
                'per_class': report,
                'probs': final_probs,
                'labels': test_labels,
                'time_min': (time.time() - fold_start) / 60,
            }
            self.fold_results.append(fold_result)
            all_test_probs.append(final_probs)
            all_test_labels.append(test_labels)

            self._print_fold(fold_result)
            all_histories.append(trainer.history)
            all_test_dfs.append(test_df.reset_index(drop=True))


            try:
                pred_viz = PredictionVisualizer(cfg, log)
                pred_viz.generate(test_df.reset_index(drop=True),
                                  data_module.roi_map, final_probs,
                                  test_labels, fold)
            except Exception as e:
                log.warning(f"  PredictionVisualizer error: {e}")


            try:
                if best_strategy == 'swa' and swa_model is not None:
                    gcam_model = swa_model.module
                else:
                    gcam_model = best_model
                gcam = GradCAMGenerator(gcam_model, cfg, device, log)
                gcam.generate(test_df.reset_index(drop=True),
                              data_module.roi_map, test_meta,
                              test_labels, final_probs, fold)
            except Exception as e:
                log.warning(f"  GradCAM error: {e}")


            save_path = os.path.join(cfg.OUTPUT_DIR, f"fold{fold+1}_best.pt")
            if best_strategy == 'swa' and swa_model is not None:
                torch.save(swa_model.module.state_dict(), save_path)
            else:
                torch.save(best_model.state_dict(), save_path)
            log.info(f"  Model saved: {save_path} (strategy: {best_strategy})")


            ckpt_mgr.save_fold_results(fold, fold_result, trainer.history)


            ckpt_mgr.save_cv_state(
                [r['fold'] for r in self.fold_results],
                self.fold_results
            )


            del model, trainer, best_model, swa_model
            torch.cuda.empty_cache()


        total_min = (time.time() - total_start) / 60
        self._print_summary(total_min)


        all_probs = np.concatenate(all_test_probs)
        all_labels = np.concatenate(all_test_labels)
        boot_results = self._bootstrap_ci(all_probs, all_labels)


        log.info(f"\n{'='*50}")
        log.info("EXTENDED STATISTICS")
        log.info(f"{'='*50}")
        calibration_results = None
        try:
            stats = StatisticalAnalyzer(cfg, log)
            calibration_results = stats.analyze(all_probs, all_labels)
        except Exception as e:
            log.warning(f"  Stats error: {e}")


        try:
            viz = Visualizer(cfg, log)
            viz.plot_all(self.fold_results, all_probs, all_labels,
                         self._df_ref, all_histories)
        except Exception as e:
            log.warning(f"  Visualizer error: {e}")


        try:
            tgen = TableGenerator(cfg, log)
            tgen.generate_all(self.fold_results, self._df_ref, boot_results,
                              calibration_results)
        except Exception as e:
            log.warning(f"  TableGenerator error: {e}")


        self._save_results()


        ckpt_mgr.clear()

        return self.fold_results


    def _print_fold(self, r):
        log = self.logger
        log.info(f"\n  {'='*50}")
        log.info(f"  FOLD {r['fold']} RESULTS ({r['strategy']}) [{r['time_min']:.1f} min]")
        log.info(f"  {'='*50}")
        log.info(f"  Accuracy:       {r['acc']:.4f}")
        log.info(f"  Balanced Acc:   {r['bal_acc']:.4f}")
        log.info(f"  F1 (macro):     {r['f1_macro']:.4f}")
        log.info(f"  AUC (macro):    {r['auc_macro']:.4f}")
        log.info(f"  Precision (m):  {r['precision_macro']:.4f}")
        log.info(f"  Recall (m):     {r['recall_macro']:.4f}")

        log.info(f"\n  Per-class:")
        for cls in self.config.CLASS_NAMES:
            c = r['per_class'].get(cls, {})
            log.info(f"    {cls:>10}: P={c.get('precision',0):.3f}  "
                     f"R={c.get('recall',0):.3f}  F1={c.get('f1-score',0):.3f}")

        log.info(f"\n  Confusion Matrix:")
        cm = r['cm']
        hdr = "            " + "  ".join(f"{n:>9}" for n in self.config.CLASS_NAMES)
        log.info(f"  {hdr}")
        for i, name in enumerate(self.config.CLASS_NAMES):
            row_str = "  ".join(f"{cm[i][j]:>9d}" for j in range(len(self.config.CLASS_NAMES)))
            log.info(f"  {name:>10}  {row_str}")


    def _print_summary(self, total_min):
        log = self.logger
        log.info(f"\n{'='*70}")
        log.info(f"CROSS-VALIDATION SUMMARY ({self.config.N_FOLDS} folds, {total_min:.1f} min)")
        log.info(f"{'='*70}")

        metrics = [
            ('acc', 'Accuracy'), ('bal_acc', 'Balanced Acc'),
            ('f1_macro', 'F1 (macro)'), ('auc_macro', 'AUC (macro)'),
            ('precision_macro', 'Precision'), ('recall_macro', 'Recall'),
        ]
        for key, name in metrics:
            vals = [r[key] for r in self.fold_results]
            log.info(f"  {name:>15}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                     f"[{', '.join(f'{v:.4f}' for v in vals)}]")


        log.info(f"\n  Per-class (mean ± std):")
        for cls in self.config.CLASS_NAMES:
            ps = [r['per_class'][cls]['precision'] for r in self.fold_results]
            rs = [r['per_class'][cls]['recall'] for r in self.fold_results]
            f1s = [r['per_class'][cls]['f1-score'] for r in self.fold_results]
            log.info(f"    {cls:>10}: P={np.mean(ps):.3f}±{np.std(ps):.3f}  "
                     f"R={np.mean(rs):.3f}±{np.std(rs):.3f}  "
                     f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")


    def _bootstrap_ci(self, probs, labels, n_boot=2000, ci=0.95):
        log = self.logger
        log.info(f"\n{'='*50}")
        log.info(f"BOOTSTRAP {ci*100:.0f}% CI (n={n_boot})")
        log.info(f"{'='*50}")

        n = len(labels)
        rng = np.random.RandomState(self.config.SEED)
        preds = probs.argmax(1)

        boot = {'Accuracy': [], 'Balanced Acc': [], 'F1 (macro)': [], 'AUC (macro)': []}

        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            bl, bp, bpr = labels[idx], preds[idx], probs[idx]
            boot['Accuracy'].append(accuracy_score(bl, bp))
            boot['Balanced Acc'].append(balanced_accuracy_score(bl, bp))
            boot['F1 (macro)'].append(f1_score(bl, bp, average='macro', zero_division=0))
            try:
                boot['AUC (macro)'].append(
                    roc_auc_score(bl, bpr, multi_class='ovr', average='macro'))
            except Exception:
                pass

        alpha = (1 - ci) / 2
        for name, vals in boot.items():
            if vals:
                lo = np.percentile(vals, alpha * 100)
                hi = np.percentile(vals, (1 - alpha) * 100)
                log.info(f"  {name:>15}: {np.mean(vals):.4f}  "
                         f"[{lo:.4f} – {hi:.4f}]")

        return boot


    def _save_results(self):

        out = []
        for r in self.fold_results:
            entry = {k: v for k, v in r.items()
                     if k not in ('probs', 'labels', 'cm', 'per_class')}
            entry['cm'] = r['cm'].tolist()
            entry['per_class'] = r['per_class']
            out.append(entry)

        path = os.path.join(self.config.OUTPUT_DIR, "cv_results.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        self.logger.info(f"Results saved: {path}")


class CheckpointManager:


    def __init__(self, config: Config):
        self.config = config
        self.state_path = os.path.join(config.CHECKPOINT_DIR, "cv_state.json")

    def save_cv_state(self, completed_folds: list, fold_results: list):
        state = {
            'completed_folds': completed_folds,
            'n_results': len(fold_results),
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f)

    def load_cv_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return None

    def save_fold_results(self, fold, fold_result, history):

        data = {
            'fold_result': fold_result,
            'history': history,
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, f"fold{fold+1}_results.pt")
        torch.save(data, path)
        return path

    def load_fold_results(self, fold):

        path = os.path.join(self.config.CHECKPOINT_DIR, f"fold{fold+1}_results.pt")
        if os.path.exists(path):
            return torch.load(path, map_location='cpu', weights_only=False)
        return None

    def clear(self):
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def save_fold_checkpoint(self, fold, model, optimizer, epoch, best_bal_acc, history):
        ckpt = {
            'fold': fold, 'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_bal_acc': best_bal_acc,
            'history': history,
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, f"fold{fold+1}_latest.pt")
        torch.save(ckpt, path)
        return path

    def load_fold_checkpoint(self, fold):
        path = os.path.join(self.config.CHECKPOINT_DIR, f"fold{fold+1}_latest.pt")
        if os.path.exists(path):
            return torch.load(path, map_location='cpu', weights_only=False)
        return None


class Visualizer:


    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

    def plot_all(self, fold_results, all_probs, all_labels, df, histories):
        self._plot_confusion_matrix(fold_results)
        self._plot_roc_curves(all_probs, all_labels)
        self._plot_training_curves(histories)
        self._plot_class_distribution(df)
        self._plot_metrics_summary(fold_results)
        self._plot_fold_comparison(fold_results)
        self.logger.info(f"All figures saved to {self.config.FIGURES_DIR}")

    def _plot_confusion_matrix(self, fold_results):

        cfg = self.config
        cm_total = sum(r['cm'] for r in fold_results)
        cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)
        names = list(cfg.CLASS_NAMES)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, data, fmt, title in [
            (axes[0], cm_total, 'd', 'Counts'),
            (axes[1], cm_norm, '.2f', 'Normalized'),
        ]:
            if sns:
                sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                            xticklabels=names, yticklabels=names, ax=ax)
            else:
                im = ax.imshow(data, cmap='Blues')
                for i in range(len(names)):
                    for j in range(len(names)):
                        ax.text(j, i, format(data[i, j], fmt),
                                ha='center', va='center', fontsize=11)
                ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
                ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "confusion_matrix.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_roc_curves(self, all_probs, all_labels):

        from sklearn.preprocessing import label_binarize
        cfg = self.config
        names = list(cfg.CLASS_NAMES)
        n_cls = cfg.NUM_CLASSES
        y_bin = label_binarize(all_labels, classes=list(range(n_cls)))

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ['#2196F3', '#FF9800', '#F44336']
        for i in range(n_cls):
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{names[i]} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves'); ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "roc_curves.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_training_curves(self, histories):

        cfg = self.config
        n_folds = len(histories)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for fold_idx, hist in enumerate(histories):
            epochs = [h['epoch'] for h in hist]
            axes[0].plot(epochs, [h['train_loss'] for h in hist],
                         alpha=0.7, label=f'Fold {fold_idx+1} Train')
            axes[0].plot(epochs, [h['val_loss'] for h in hist],
                         '--', alpha=0.7, label=f'Fold {fold_idx+1} Val')
            axes[1].plot(epochs, [h['train_acc'] for h in hist],
                         alpha=0.7, label=f'Fold {fold_idx+1} Train')
            axes[1].plot(epochs, [h['val_acc'] for h in hist],
                         '--', alpha=0.7, label=f'Fold {fold_idx+1} Val')

        axes[0].set_ylabel('Loss'); axes[0].set_title('Training / Validation Loss')
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].set_ylabel('Accuracy'); axes[1].set_xlabel('Epoch')
        axes[1].set_title('Training / Validation Accuracy')
        axes[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "training_curves.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_class_distribution(self, df):

        cfg = self.config
        counts = df['label'].value_counts().reindex(list(cfg.CLASS_NAMES))
        colors = ['#4CAF50', '#FF9800', '#F44336']

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black')
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 20,
                    str(v), ha='center', fontweight='bold')
        ax.set_ylabel('Count'); ax.set_title('Class Distribution')
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "class_distribution.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_metrics_summary(self, fold_results):

        cfg = self.config
        names = list(cfg.CLASS_NAMES)
        metrics_data = {}
        for cls in names:
            ps = [r['per_class'][cls]['precision'] for r in fold_results]
            rs = [r['per_class'][cls]['recall'] for r in fold_results]
            f1s = [r['per_class'][cls]['f1-score'] for r in fold_results]
            metrics_data[cls] = {
                'Precision': (np.mean(ps), np.std(ps)),
                'Recall': (np.mean(rs), np.std(rs)),
                'F1-Score': (np.mean(f1s), np.std(f1s)),
            }

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(names))
        w = 0.25
        colors_m = ['#2196F3', '#4CAF50', '#FF9800']
        for i, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
            means = [metrics_data[c][metric][0] for c in names]
            stds = [metrics_data[c][metric][1] for c in names]
            ax.bar(x + i * w, means, w, yerr=stds, label=metric,
                   color=colors_m[i], capsize=3, edgecolor='black')

        ax.set_xticks(x + w); ax.set_xticklabels(names)
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
        ax.set_title('Per-Class Metrics (mean ± std)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "metrics_summary.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_fold_comparison(self, fold_results):

        cfg = self.config
        folds = [r['fold'] for r in fold_results]
        accs = [r['acc'] for r in fold_results]
        aucs = [r['auc_macro'] for r in fold_results]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(folds))
        ax.bar(x - 0.15, accs, 0.3, label='Accuracy', color='#2196F3', edgecolor='black')
        ax.bar(x + 0.15, aucs, 0.3, label='AUC', color='#FF9800', edgecolor='black')
        ax.axhline(np.mean(accs), ls='--', color='#2196F3', alpha=0.5)
        ax.axhline(np.mean(aucs), ls='--', color='#FF9800', alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.set_ylabel('Score'); ax.set_ylim(0.5, 1.0)
        ax.set_title('Fold Comparison'); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "fold_comparison.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


class GradCAMGenerator:


    def __init__(self, model, config: Config, device, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.target_layer = self._get_target_layer()

    def _get_target_layer(self):

        try:
            return self.model.backbone.blocks[-1]
        except AttributeError:
            try:
                return self.model.backbone.layer4
            except AttributeError:

                children = list(self.model.backbone.children())
                return children[-2] if len(children) > 1 else children[-1]

    def generate(self, df, roi_map, metadata, labels, probs, fold):

        cfg = self.config
        preds = probs.argmax(1)
        names = list(cfg.CLASS_NAMES)
        n_per_class = cfg.N_GRADCAM_SAMPLES // cfg.NUM_CLASSES

        for cls_idx, cls_name in enumerate(names):
            mask = (labels == cls_idx) & (preds == cls_idx)
            indices = np.where(mask)[0][:n_per_class]
            for i, idx in enumerate(indices):
                row = df.iloc[idx]
                try:
                    img = Image.open(row['image_path']).convert('RGB')
                    img_np = np.array(img)
                except Exception:
                    continue

                if cfg.USE_ROI and row['img_id'] in roi_map:
                    img_np = crop_roi(img_np, roi_map[row['img_id']],
                                      margin=cfg.ROI_MARGIN, is_train=False)

                heatmap = self._compute_gradcam(img_np, metadata[idx], cls_idx)
                self._save_figure(img_np, heatmap, cls_name, probs[idx],
                                  fold, f"{cls_name}_{i}")

    def _compute_gradcam(self, img_np, meta, target_class):

        cfg = self.config
        tfm = get_val_transforms(cfg.IMG_SIZE)
        tensor = tfm(image=img_np)['image'].unsqueeze(0).to(self.device)
        meta_t = torch.from_numpy(meta).unsqueeze(0).float().to(self.device)

        activations = []
        gradients = []

        def fwd_hook(m, inp, out):
            activations.append(out)
        def bwd_hook(m, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = self.target_layer.register_forward_hook(fwd_hook)
        h2 = self.target_layer.register_full_backward_hook(bwd_hook)

        self.model.eval()
        self.model.zero_grad()
        out = self.model(tensor, meta_t)
        out[0, target_class].backward()

        h1.remove(); h2.remove()

        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        return cam

    def _save_figure(self, img_np, heatmap, cls_name, probs, fold, tag):

        cfg = self.config
        img_resized = cv2.resize(img_np, (cfg.IMG_SIZE, cfg.IMG_SIZE))

        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_resized); axes[0].set_title('Original')
        axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title('Grad-CAM')
        axes[2].imshow(overlay); axes[2].set_title('Overlay')

        prob_str = " | ".join(f"{n}:{p:.1%}" for n, p in
                              zip(cfg.CLASS_NAMES, probs))
        fig.suptitle(f'{cls_name} — {prob_str}', fontsize=10)
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.GRADCAM_DIR, f"fold{fold+1}_{tag}.png"),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


class PredictionVisualizer:


    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

    def generate(self, df, roi_map, probs, labels, fold):

        cfg = self.config
        preds = probs.argmax(1)
        names = list(cfg.CLASS_NAMES)
        n_samples = cfg.N_PREDICTION_SAMPLES

        for cls_idx, cls_name in enumerate(names):
            correct_mask = (labels == cls_idx) & (preds == cls_idx)
            incorrect_mask = (labels == cls_idx) & (preds != cls_idx)

            self._create_grid(df, roi_map, probs, preds, labels,
                              np.where(correct_mask)[0][:n_samples],
                              fold, cls_name, "correct")
            self._create_grid(df, roi_map, probs, preds, labels,
                              np.where(incorrect_mask)[0][:n_samples],
                              fold, cls_name, "incorrect")

    def _create_grid(self, df, roi_map, probs, preds, labels,
                     indices, fold, cls_name, status):

        cfg = self.config
        if len(indices) == 0:
            return

        n = min(len(indices), cfg.N_PREDICTION_SAMPLES)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 9),
                                 gridspec_kw={'height_ratios': [4, 1]})
        if n == 1:
            axes = axes.reshape(2, 1)

        names = list(cfg.CLASS_NAMES)
        colors = {'correct': '#4CAF50', 'incorrect': '#F44336'}
        border_color = colors[status]

        for col, idx in enumerate(indices[:n]):
            row = df.iloc[idx]
            try:
                img = Image.open(row['image_path']).convert('RGB')
                img_np = np.array(img)
            except Exception:
                img_np = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)

            if cfg.USE_ROI and row['img_id'] in roi_map:
                img_np = crop_roi(img_np, roi_map[row['img_id']],
                                  margin=cfg.ROI_MARGIN, is_train=False)

            img_show = cv2.resize(img_np, (cfg.IMG_SIZE, cfg.IMG_SIZE))


            ax_img = axes[0, col]
            ax_img.imshow(img_show)
            for spine in ax_img.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            true_name = names[labels[idx]]
            pred_name = names[preds[idx]]
            ax_img.set_title(f'True: {true_name}\nPred: {pred_name}', fontsize=9)
            ax_img.set_xticks([]); ax_img.set_yticks([])


            ax_bar = axes[1, col]
            bar_colors = ['#4CAF50' if i == labels[idx] else '#9E9E9E'
                          for i in range(cfg.NUM_CLASSES)]
            bar_colors[preds[idx]] = '#F44336' if preds[idx] != labels[idx] else '#4CAF50'
            ax_bar.barh(names, probs[idx], color=bar_colors, edgecolor='black')
            for i, p in enumerate(probs[idx]):
                ax_bar.text(p + 0.01, i, f'{p:.1%}', va='center', fontsize=8)
            ax_bar.set_xlim(0, 1.15)
            ax_bar.set_xlabel('Probability')

        fig.suptitle(f'Fold {fold+1} — {cls_name} ({status.title()})',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        fname = f"fold{fold+1}_{cls_name}_{status}.png"
        fig.savefig(os.path.join(cfg.FIGURES_DIR, fname),
                    dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)


class TableGenerator:


    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

    def generate_all(self, fold_results, df, bootstrap_results=None,
                      calibration_results=None):
        self._dataset_summary(df)
        self._cv_results(fold_results)
        self._per_class_metrics(fold_results)
        self._confusion_matrix_table(fold_results)
        if bootstrap_results:
            self._bootstrap_table(bootstrap_results)
        if calibration_results:
            self._calibration_table(calibration_results)
        self._clopper_pearson_table(fold_results)
        self.logger.info(f"All tables saved to {self.config.TABLES_DIR}")

    def _save(self, df_out, name):
        csv_path = os.path.join(self.config.TABLES_DIR, f"{name}.csv")
        xlsx_path = os.path.join(self.config.TABLES_DIR, f"{name}.xlsx")
        df_out.to_csv(csv_path, index=False)
        df_out.to_excel(xlsx_path, index=False)

    def _dataset_summary(self, df):

        cfg = self.config
        rows = []
        for cls in list(cfg.CLASS_NAMES):
            sub = df[df['label'] == cls]
            rows.append({
                'Class': cls,
                'N': len(sub),
                'Percentage': f"{len(sub)/len(df)*100:.1f}%",
                'Age (mean±std)': f"{sub['age'].mean():.1f}±{sub['age'].std():.1f}",
                'Male %': f"{(sub['gender']=='M').mean()*100:.1f}%",
            })
        rows.append({
            'Class': 'Total',
            'N': len(df),
            'Percentage': '100.0%',
            'Age (mean±std)': f"{df['age'].mean():.1f}±{df['age'].std():.1f}",
            'Male %': f"{(df['gender']=='M').mean()*100:.1f}%",
        })
        self._save(pd.DataFrame(rows), "table1_dataset_summary")

    def _cv_results(self, fold_results):

        rows = []
        for r in fold_results:
            rows.append({
                'Fold': r['fold'],
                'Strategy': r['strategy'],
                'Accuracy': f"{r['acc']:.4f}",
                'Balanced Acc': f"{r['bal_acc']:.4f}",
                'F1 (macro)': f"{r['f1_macro']:.4f}",
                'AUC (macro)': f"{r['auc_macro']:.4f}",
                'Precision': f"{r['precision_macro']:.4f}",
                'Recall': f"{r['recall_macro']:.4f}",
                'Time (min)': f"{r['time_min']:.1f}",
            })

        metrics = ['acc', 'bal_acc', 'f1_macro', 'auc_macro',
                   'precision_macro', 'recall_macro']
        mean_row = {'Fold': 'Mean±Std', 'Strategy': '-'}
        for m in metrics:
            vals = [r[m] for r in fold_results]
            col = m.replace('_', ' ').title()
            if m == 'auc_macro':
                col = 'AUC (macro)'
            elif m == 'f1_macro':
                col = 'F1 (macro)'
            elif m == 'bal_acc':
                col = 'Balanced Acc'
            mean_row[col] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
        mean_row['Time (min)'] = f"{sum(r['time_min'] for r in fold_results):.1f}"
        rows.append(mean_row)
        self._save(pd.DataFrame(rows), "table2_cv_results")

    def _per_class_metrics(self, fold_results):

        cfg = self.config
        rows = []
        for cls in list(cfg.CLASS_NAMES):
            ps = [r['per_class'][cls]['precision'] for r in fold_results]
            rs = [r['per_class'][cls]['recall'] for r in fold_results]
            f1s = [r['per_class'][cls]['f1-score'] for r in fold_results]
            rows.append({
                'Class': cls,
                'Precision': f"{np.mean(ps):.4f}±{np.std(ps):.4f}",
                'Recall': f"{np.mean(rs):.4f}±{np.std(rs):.4f}",
                'F1-Score': f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
            })
        self._save(pd.DataFrame(rows), "table3_per_class")

    def _confusion_matrix_table(self, fold_results):

        cfg = self.config
        cm = sum(r['cm'] for r in fold_results)
        df_cm = pd.DataFrame(cm, index=list(cfg.CLASS_NAMES),
                             columns=list(cfg.CLASS_NAMES))
        csv_path = os.path.join(cfg.TABLES_DIR, "table4_confusion_matrix.csv")
        xlsx_path = os.path.join(cfg.TABLES_DIR, "table4_confusion_matrix.xlsx")
        df_cm.to_csv(csv_path)
        df_cm.to_excel(xlsx_path)

    def _bootstrap_table(self, boot_results):

        rows = []
        for name, vals in boot_results.items():
            if vals:
                lo = np.percentile(vals, 2.5)
                hi = np.percentile(vals, 97.5)
                rows.append({
                    'Metric': name,
                    'Mean': f"{np.mean(vals):.4f}",
                    '95% CI Lower': f"{lo:.4f}",
                    '95% CI Upper': f"{hi:.4f}",
                })
        self._save(pd.DataFrame(rows), "table5_bootstrap_ci")

    def _calibration_table(self, calibration_results):

        rows = []
        rows.append({'Metric': "Cohen's Kappa",
                     'Value': f"{calibration_results.get('kappa', 0):.4f}"})
        rows.append({'Metric': 'Brier Score (macro)',
                     'Value': f"{calibration_results.get('brier_macro', 0):.4f}"})
        rows.append({'Metric': 'ECE (15 bins)',
                     'Value': f"{calibration_results.get('ece', 0):.4f}"})

        for cls_name in self.config.CLASS_NAMES:
            key = f'brier_{cls_name}'
            if key in calibration_results:
                rows.append({'Metric': f'Brier Score ({cls_name})',
                             'Value': f"{calibration_results[key]:.4f}"})
        self._save(pd.DataFrame(rows), "table_calibration")

    def _clopper_pearson_table(self, fold_results):
\
\
\
\

        from scipy.stats import beta as beta_dist
        cfg = self.config
        cm = sum(r['cm'] for r in fold_results)
        names = list(cfg.CLASS_NAMES)
        alpha = 0.05
        rows = []
        for i, true_cls in enumerate(names):
            n_total = cm[i].sum()
            for j, pred_cls in enumerate(names):
                if i == j:
                    continue
                k = cm[i][j]

                if k == 0:
                    lo = 0.0
                    hi = 1.0 - (alpha / 2) ** (1.0 / n_total) if n_total > 0 else 1.0
                elif k == n_total:
                    lo = (alpha / 2) ** (1.0 / n_total) if n_total > 0 else 0.0
                    hi = 1.0
                else:
                    lo = beta_dist.ppf(alpha / 2, k, n_total - k + 1)
                    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n_total - k)
                rows.append({
                    'True Class': true_cls,
                    'Predicted Class': pred_cls,
                    'Count': k,
                    'Total': n_total,
                    'Rate': f"{k/n_total:.4f}" if n_total > 0 else "N/A",
                    '95% CI Lower': f"{lo:.4f}",
                    '95% CI Upper': f"{hi:.4f}",
                })
        self._save(pd.DataFrame(rows), "table_clopper_pearson_ci")
        self.logger.info(f"  Clopper-Pearson CI table saved (off-diagonal cells)")

        for r in rows:
            if r['Count'] == 0:
                self.logger.info(
                    f"    {r['True Class']}→{r['Predicted Class']}: "
                    f"0/{r['Total']} rate=0.0000 "
                    f"[{r['95% CI Lower']} – {r['95% CI Upper']}]"
                )


class StatisticalAnalyzer:


    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger

    def analyze(self, all_probs, all_labels):

        preds = all_probs.argmax(1)
        log = self.logger
        results = {}


        kappa = cohen_kappa_score(all_labels, preds)
        results['kappa'] = kappa
        log.info(f"  Cohen's Kappa: {kappa:.4f}")


        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(all_labels, classes=list(range(self.config.NUM_CLASSES)))
        brier_scores = []
        for i in range(self.config.NUM_CLASSES):
            bs = brier_score_loss(y_bin[:, i], all_probs[:, i])
            brier_scores.append(bs)
            results[f'brier_{self.config.CLASS_NAMES[i]}'] = bs
            log.info(f"  Brier Score ({self.config.CLASS_NAMES[i]}): {bs:.4f}")
        results['brier_macro'] = np.mean(brier_scores)
        log.info(f"  Brier Score (macro): {results['brier_macro']:.4f}")


        ece = self._compute_ece(all_probs, all_labels)
        results['ece'] = ece
        log.info(f"  ECE: {ece:.4f}")

        return results

    def _compute_ece(self, probs, labels, n_bins=15):

        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
        return ece


def generate_ablation_table(all_experiment_results, output_dir, logger):
\
\
\
\

    rows = []
    for exp in all_experiment_results:
        if exp['seed'] != 42:
            continue
        if exp['backbone'] != "tf_efficientnetv2_s":
            continue
        tag = exp['tag']
        results = exp['fold_results']
        accs = [r['acc'] for r in results]
        bals = [r['bal_acc'] for r in results]
        f1s = [r['f1_macro'] for r in results]
        aucs = [r['auc_macro'] for r in results]
        rows.append({
            'Condition': tag,
            'ROI': '✓' if exp['use_roi'] else '✗',
            'Metadata': '✓' if exp['use_metadata'] else '✗',
            'Accuracy': f"{np.mean(accs):.4f}±{np.std(accs):.4f}",
            'Balanced Acc': f"{np.mean(bals):.4f}±{np.std(bals):.4f}",
            'F1 (macro)': f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
            'AUC (macro)': f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}",
        })

    if rows:
        df = pd.DataFrame(rows)
        tables_dir = os.path.join(output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        df.to_csv(os.path.join(tables_dir, "table_ablation_study.csv"), index=False)
        df.to_excel(os.path.join(tables_dir, "table_ablation_study.xlsx"), index=False)
        logger.info(f"\n{'='*60}")
        logger.info("ABLATION STUDY RESULTS (M1)")
        logger.info(f"{'='*60}")
        for _, r in df.iterrows():
            logger.info(f"  {r['Condition']:30s} | Acc={r['Accuracy']} | "
                         f"Bal={r['Balanced Acc']} | F1={r['F1 (macro)']} | "
                         f"AUC={r['AUC (macro)']}")


def generate_multiseed_table(all_experiment_results, output_dir, logger):
\
\
\

    rows = []
    seed_metrics = {'acc': [], 'bal_acc': [], 'f1_macro': [], 'auc_macro': []}

    for exp in all_experiment_results:
        if exp['backbone'] != "tf_efficientnetv2_s":
            continue
        if not (exp['use_roi'] and exp['use_metadata']):
            continue
        results = exp['fold_results']
        accs = [r['acc'] for r in results]
        bals = [r['bal_acc'] for r in results]
        f1s = [r['f1_macro'] for r in results]
        aucs = [r['auc_macro'] for r in results]
        rows.append({
            'Seed': exp['seed'],
            'Accuracy': f"{np.mean(accs):.4f}±{np.std(accs):.4f}",
            'Balanced Acc': f"{np.mean(bals):.4f}±{np.std(bals):.4f}",
            'F1 (macro)': f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
            'AUC (macro)': f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}",
        })
        seed_metrics['acc'].append(np.mean(accs))
        seed_metrics['bal_acc'].append(np.mean(bals))
        seed_metrics['f1_macro'].append(np.mean(f1s))
        seed_metrics['auc_macro'].append(np.mean(aucs))

    if rows:

        rows.append({
            'Seed': 'Cross-seed',
            'Accuracy': f"{np.mean(seed_metrics['acc']):.4f}±{np.std(seed_metrics['acc']):.4f}",
            'Balanced Acc': f"{np.mean(seed_metrics['bal_acc']):.4f}±{np.std(seed_metrics['bal_acc']):.4f}",
            'F1 (macro)': f"{np.mean(seed_metrics['f1_macro']):.4f}±{np.std(seed_metrics['f1_macro']):.4f}",
            'AUC (macro)': f"{np.mean(seed_metrics['auc_macro']):.4f}±{np.std(seed_metrics['auc_macro']):.4f}",
        })
        df = pd.DataFrame(rows)
        tables_dir = os.path.join(output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        df.to_csv(os.path.join(tables_dir, "table_multiseed.csv"), index=False)
        df.to_excel(os.path.join(tables_dir, "table_multiseed.xlsx"), index=False)
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-SEED RESULTS (M2)")
        logger.info(f"{'='*60}")
        for _, r in df.iterrows():
            logger.info(f"  Seed {r['Seed']:>10} | Acc={r['Accuracy']} | "
                         f"Bal={r['Balanced Acc']}")


def generate_backbone_table(all_experiment_results, output_dir, logger):
\
\
\

    rows = []
    for exp in all_experiment_results:
        if exp['seed'] != 42:
            continue
        if not (exp['use_roi'] and exp['use_metadata']):
            continue
        results = exp['fold_results']
        accs = [r['acc'] for r in results]
        bals = [r['bal_acc'] for r in results]
        f1s = [r['f1_macro'] for r in results]
        aucs = [r['auc_macro'] for r in results]
        rows.append({
            'Backbone': exp['backbone'],
            'Accuracy': f"{np.mean(accs):.4f}±{np.std(accs):.4f}",
            'Balanced Acc': f"{np.mean(bals):.4f}±{np.std(bals):.4f}",
            'F1 (macro)': f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
            'AUC (macro)': f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}",
        })

    if rows:
        df = pd.DataFrame(rows)
        tables_dir = os.path.join(output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        df.to_csv(os.path.join(tables_dir, "table_backbone_comparison.csv"), index=False)
        df.to_excel(os.path.join(tables_dir, "table_backbone_comparison.xlsx"), index=False)
        logger.info(f"\n{'='*60}")
        logger.info("BACKBONE COMPARISON (M3)")
        logger.info(f"{'='*60}")
        for _, r in df.iterrows():
            logger.info(f"  {r['Backbone']:30s} | Acc={r['Accuracy']} | "
                         f"Bal={r['Balanced Acc']} | AUC={r['AUC (macro)']}")


def run_single_experiment(config, logger, dm, df, tag):
\


    results_path = os.path.join(config.OUTPUT_DIR, "cv_results.json")
    if os.path.exists(results_path):
        logger.info(f"\n{'⏭'*35}")
        logger.info(f"SKIPPING (already completed): {tag}")
        logger.info(f"  Cached results: {results_path}")
        logger.info(f"{'⏭'*35}")
        with open(results_path, 'r') as f:
            cached = json.load(f)

        if isinstance(cached, list):
            return cached
        return cached.get('fold_results', cached.get('results', [cached]))

    logger.info(f"\n{'★'*70}")
    logger.info(f"EXPERIMENT: {tag}")
    logger.info(f"  Backbone={config.BACKBONE}  USE_ROI={config.USE_ROI}  "
                f"USE_METADATA={config.USE_METADATA}  SEED={config.SEED}")
    logger.info(f"{'★'*70}")

    set_seed(config.SEED)
    cv = CrossValidator(config, logger)
    results = cv.run(df, dm)
    return results


def main():
\
\
\
\

    config = Config()
    set_seed(config.SEED)
    base_output = config.OUTPUT_DIR
    logger = setup_logging(base_output)

    logger.info("=" * 70)
    logger.info("BTXRD BONE LESION CLASSIFIER")
    logger.info("Multi-seed + Ablation + Multi-backbone")
    logger.info("=" * 70)


    local_base = prepare_local_cache(config, logger)
    if local_base != config.BASE_DIR:
        logger.info(f"Paths updated: BASE_DIR = {local_base}")
        config.BASE_DIR = local_base
        config.__post_init__()


    dm = DataModule(config, logger)
    df = dm.load()


    experiments = []


    for seed in config.SEEDS:
        experiments.append({
            'backbone': "tf_efficientnetv2_s",
            'use_roi': True,
            'use_metadata': True,
            'seed': seed,
            'tag': f"EfficientNetV2-S_ROI+Meta_seed{seed}",
        })


    ablation_conditions = [
        (True,  False, "EfficientNetV2-S_ROI_noMeta_seed42"),
        (False, True,  "EfficientNetV2-S_Whole+Meta_seed42"),
        (False, False, "EfficientNetV2-S_Whole_noMeta_seed42"),
    ]
    for use_roi, use_meta, tag in ablation_conditions:
        experiments.append({
            'backbone': "tf_efficientnetv2_s",
            'use_roi': use_roi,
            'use_metadata': use_meta,
            'seed': 42,
            'tag': tag,
        })


    for bb in config.BACKBONES:
        if bb == "tf_efficientnetv2_s":
            continue
        experiments.append({
            'backbone': bb,
            'use_roi': True,
            'use_metadata': True,
            'seed': 42,
            'tag': f"{bb}_ROI+Meta_seed42",
        })

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT PLAN: {len(experiments)} experiments")
    logger.info(f"{'='*60}")
    for i, exp in enumerate(experiments):
        logger.info(f"  [{i+1}] {exp['tag']}")


    all_experiment_results = []
    total_start = time.time()

    for i, exp in enumerate(experiments):
        exp_start = time.time()
        logger.info(f"\n{'#'*70}")
        logger.info(f"# EXPERIMENT {i+1}/{len(experiments)}: {exp['tag']}")
        logger.info(f"{'#'*70}")


        exp_config = Config()
        exp_config.BASE_DIR = config.BASE_DIR
        exp_config.BACKBONE = exp['backbone']
        exp_config.USE_ROI = exp['use_roi']
        exp_config.USE_METADATA = exp['use_metadata']
        exp_config.SEED = exp['seed']


        exp_config.OUTPUT_DIR = os.path.join(base_output, exp['tag'])
        exp_config.__post_init__()

        try:
            fold_results = run_single_experiment(exp_config, logger, dm, df, exp['tag'])
            exp_time = (time.time() - exp_start) / 60
            logger.info(f"  Experiment completed in {exp_time:.1f} min")

            all_experiment_results.append({
                **exp,
                'fold_results': fold_results,
                'time_min': exp_time,
            })
        except Exception as e:
            logger.error(f"  EXPERIMENT FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())


    total_min = (time.time() - total_start) / 60
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL EXPERIMENTS COMPLETE ({total_min:.1f} min total)")
    logger.info(f"{'='*70}")

    try:
        generate_ablation_table(all_experiment_results, base_output, logger)
    except Exception as e:
        logger.warning(f"Ablation table error: {e}")

    try:
        generate_multiseed_table(all_experiment_results, base_output, logger)
    except Exception as e:
        logger.warning(f"Multi-seed table error: {e}")

    try:
        generate_backbone_table(all_experiment_results, base_output, logger)
    except Exception as e:
        logger.warning(f"Backbone table error: {e}")


    summary = []
    for exp in all_experiment_results:
        entry = {k: v for k, v in exp.items() if k != 'fold_results'}
        if 'fold_results' in exp:
            entry['mean_acc'] = np.mean([r['acc'] for r in exp['fold_results']])
            entry['mean_bal_acc'] = np.mean([r['bal_acc'] for r in exp['fold_results']])
            entry['mean_f1'] = np.mean([r['f1_macro'] for r in exp['fold_results']])
            entry['mean_auc'] = np.mean([r['auc_macro'] for r in exp['fold_results']])
        summary.append(entry)

    summary_path = os.path.join(base_output, "full_experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Full experiment summary saved: {summary_path}")

    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
