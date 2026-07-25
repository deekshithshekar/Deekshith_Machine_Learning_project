"""
Dataset classes for the FLS Marine Debris project.

Three concrete datasets, all reading PNG files from disk:
  - FLSClassificationDataset : class-named subfolders (turntable / watertank-cropped)
  - FLSDetectionDataset      : Images/ + BoxAnnotations/ XML (x,y,w,h bboxes)
  - FLSSegmentationDataset   : Images/ + Masks/ PNG pixel labels (0-11)
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# ── shared preprocessing ───────────────────────────────────────────────────────

def _clahe_preprocess(img_gray: np.ndarray,
                      clip_limit: float = 2.0,
                      tile_grid: int = 8,
                      sigma: float = 1.0) -> np.ndarray:
    """CLAHE + Gaussian blur on a uint8 grayscale image. Returns float32 [0,1]."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_grid, tile_grid))
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray / img_gray.max() * 255).clip(0, 255).astype(np.uint8) \
            if img_gray.max() > 0 else img_gray.astype(np.uint8)
    img = clahe.apply(img_gray)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=sigma)
    return img.astype(np.float32) / 255.0


def _to_3ch_tensor(img_float: np.ndarray) -> torch.Tensor:
    """float32 HxW → torch float32 [3, H, W]."""
    img_3ch = np.stack([img_float, img_float, img_float], axis=0)
    return torch.from_numpy(img_3ch)


def _load_sonar_png(path: str) -> torch.Tensor:
    """Load a grayscale sonar PNG, apply CLAHE, return [3, H, W] tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return _to_3ch_tensor(_clahe_preprocess(img))


# ── A: Classification dataset ──────────────────────────────────────────────────

class FLSClassificationDataset(Dataset):
    """
    Reads sonar images from class-named subfolders.

    Works with:
      - turntable-cropped/   (18 classes)
      - watertank-cropped/   (10 classes)

    Args:
        root           : path to the dataset root folder (e.g. "turntable-cropped/")
        split          : "train" | "val" | "test"
        val_ratio      : fraction for validation (default 0.15)
        test_ratio     : fraction for test      (default 0.15)
        seed           : random seed for reproducible splits
        transform      : optional callable applied to the [3,H,W] float tensor
        img_size       : resize target (square) after CLAHE, default 224
        split_strategy : "random" (default, image-level stratified split) or
                         "blocked" (approximate leakage-safe split, see
                         _blocked_split below)
        block_size     : contiguous-frame block size used by "blocked"
    """

    def __init__(self, root: str, split: str = "train",
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42, transform: Optional[Callable] = None,
                 img_size: int = 224, split_strategy: str = "random",
                 block_size: int = 15):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        self.classes = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and next(d.glob("*.png"), None) is not None
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split_strategy == "blocked":
            self.paths, self.labels = self._blocked_split(
                val_ratio, test_ratio, seed, block_size, split)
            return

        all_paths, all_labels = [], []
        for cls in self.classes:
            cls_dir = self.root / cls
            for f in sorted(cls_dir.glob("*.png")):
                all_paths.append(str(f))
                all_labels.append(self.class_to_idx[cls])

        # Stratified split
        idx = list(range(len(all_paths)))
        test_size = test_ratio
        val_size  = val_ratio / (1 - test_ratio)

        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, stratify=all_labels, random_state=seed)
        train_val_labels = [all_labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size,
            stratify=train_val_labels, random_state=seed)

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        chosen = split_map[split]
        self.paths  = [all_paths[i]  for i in chosen]
        self.labels = [all_labels[i] for i in chosen]

    def _blocked_split(self, val_ratio: float, test_ratio: float, seed: int,
                        block_size: int, split: str) -> Tuple[List[str], List[int]]:
        """
        Approximate object/session-grouped split. The released dataset encodes
        no object or session ID in its filenames, but frames within a class are
        sequentially numbered (turntable rotation frames, watertank video
        frames), so adjacent frames are near-duplicate views of the same
        physical object. Chunking into contiguous blocks and splitting at the
        block level (instead of shuffling individual frames) keeps
        near-duplicates on the same side of the split, which is the actual
        leakage vector in the random image-level split above.
        """
        rng = np.random.RandomState(seed)
        paths, labels = [], []
        for cls in self.classes:
            cls_dir = self.root / cls
            files = list(cls_dir.glob("*.png"))
            files.sort(key=lambda f: int(re.search(r"(\d+)$", f.stem).group(1)))

            blocks = [files[i:i + block_size] for i in range(0, len(files), block_size)]
            order = rng.permutation(len(blocks))

            n_test = max(1, round(len(blocks) * test_ratio))
            n_val = max(1, round(len(blocks) * val_ratio))
            block_map = {
                "test": order[:n_test],
                "val": order[n_test:n_test + n_val],
                "train": order[n_test + n_val:],
            }
            for b in block_map[split]:
                for f in blocks[b]:
                    paths.append(str(f))
                    labels.append(self.class_to_idx[cls])
        return paths, labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = _load_sonar_png(self.paths[idx])
        # Resize
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.resize(img_np, (self.img_size, self.img_size))
        img = torch.from_numpy(img_np.transpose(2, 0, 1))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def class_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(self.labels))

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        counts = np.bincount(self.labels, minlength=len(self.classes))
        counts = np.where(counts == 0, 1, counts)
        weights = (1.0 / counts)[self.labels]
        return WeightedRandomSampler(weights=weights,
                                     num_samples=len(weights),
                                     replacement=True)


# ── B: Detection dataset (XML bounding boxes) ─────────────────────────────────

# Classes defined by the dataset README (pixel value = class index)
WATERTANK_SEG_CLASSES = [
    "background", "bottle", "can", "chain", "drink-carton",
    "hook", "propeller", "shampoo-bottle", "standing-bottle", "tire", "valve", "wall",
]
WATERTANK_SEG_CLASS_TO_IDX = {c: i for i, c in enumerate(WATERTANK_SEG_CLASSES)}

# XML uses these names — map to our canonical class indices
_XML_NAME_MAP = {
    "Background": 0, "Bottle": 1, "Can": 2, "Chain": 3, "Drink-carton": 4,
    "Hook": 5, "Propeller": 6, "Shampoo-bottle": 7, "Standing-bottle": 8,
    "Tire": 9, "Valve": 10, "Wall": 11,
}


def _parse_xml(xml_path: str) -> Tuple[int, int, List[dict]]:
    """
    Parse a BoxAnnotations XML file.
    Returns (img_w, img_h, list_of_boxes).
    Each box: {"label": int, "x": int, "y": int, "w": int, "h": int}
    Bounding box format in XML is x, y, w, h (COCO-style).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        label = _XML_NAME_MAP.get(name, -1)
        if label < 0:
            # Try case-insensitive match
            label = next((v for k, v in _XML_NAME_MAP.items()
                          if k.lower() == name.lower()), -1)
        if label < 0:
            continue
        bb = obj.find("bndbox")
        x = int(float(bb.find("x").text))
        y = int(float(bb.find("y").text))
        w = int(float(bb.find("w").text))
        h = int(float(bb.find("h").text))
        boxes.append({"label": label, "x": x, "y": y, "w": w, "h": h})

    return W, H, boxes


class FLSDetectionDataset(Dataset):
    """
    Watertank-Segmentation dataset for object detection.
    Reads Images/ + BoxAnnotations/ XML files.

    Returns:
        image  : float32 tensor [3, img_size, img_size]  — CLAHE preprocessed
        target : dict with keys:
                   "boxes"  : FloatTensor [N, 4]  — (x1, y1, x2, y2) normalized to [0,1]
                   "labels" : LongTensor [N]       — class indices
                   "image_id": int
    """

    def __init__(self, seg_root: str, split: str = "train",
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42, transform: Optional[Callable] = None,
                 img_size: int = 640):
        self.seg_root  = Path(seg_root)
        self.img_dir   = self.seg_root / "Images"
        self.ann_dir   = self.seg_root / "BoxAnnotations"
        self.img_size  = img_size
        self.transform = transform
        self.classes   = WATERTANK_SEG_CLASSES

        all_stems = sorted(
            p.stem for p in self.img_dir.glob("*.png")
            if (self.ann_dir / (p.stem + ".xml")).exists()
        )

        train_val, test = train_test_split(all_stems, test_size=test_ratio,
                                           random_state=seed)
        val_frac = val_ratio / (1 - test_ratio)
        train, val = train_test_split(train_val, test_size=val_frac,
                                      random_state=seed)

        self.stems = {"train": train, "val": val, "test": test}[split]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        stem     = self.stems[idx]
        img_path = self.img_dir / (stem + ".png")
        xml_path = self.ann_dir / (stem + ".xml")

        img = _load_sonar_png(img_path)
        orig_H, orig_W = img.shape[1], img.shape[2]

        # Resize
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.resize(img_np, (self.img_size, self.img_size))
        img    = torch.from_numpy(img_np.transpose(2, 0, 1))

        W, H, raw_boxes = _parse_xml(str(xml_path))
        scale_x = self.img_size / W
        scale_y = self.img_size / H

        boxes, labels = [], []
        for b in raw_boxes:
            x1 = b["x"] * scale_x / self.img_size
            y1 = b["y"] * scale_y / self.img_size
            x2 = (b["x"] + b["w"]) * scale_x / self.img_size
            y2 = (b["y"] + b["h"]) * scale_y / self.img_size
            x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2)]
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(b["label"])

        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels":   torch.tensor(labels, dtype=torch.int64),
            "image_id": idx,
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


# ── C: Segmentation dataset (PNG pixel masks) ─────────────────────────────────

class FLSSegmentationDataset(Dataset):
    """
    Watertank-Segmentation dataset for semantic segmentation.
    Reads Images/ + Masks/ (PNG files with pixel values 0-11).

    Returns:
        image : float32 tensor [3, img_size, img_size]
        mask  : long tensor    [img_size, img_size]   — class indices 0-11
    """

    NUM_CLASSES = 12  # 0=background … 11=wall
    CLASSES     = WATERTANK_SEG_CLASSES

    def __init__(self, seg_root: str, split: str = "train",
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42, transform: Optional[Callable] = None,
                 img_size: int = 640):
        self.seg_root  = Path(seg_root)
        self.img_dir   = self.seg_root / "Images"
        self.mask_dir  = self.seg_root / "Masks"
        self.img_size  = img_size
        self.transform = transform

        all_stems = sorted(
            p.stem for p in self.img_dir.glob("*.png")
            if (self.mask_dir / (p.stem + ".png")).exists()
        )

        train_val, test = train_test_split(all_stems, test_size=test_ratio,
                                           random_state=seed)
        val_frac = val_ratio / (1 - test_ratio)
        train, val = train_test_split(train_val, test_size=val_frac,
                                      random_state=seed)

        self.stems = {"train": train, "val": val, "test": test}[split]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem      = self.stems[idx]
        img_path  = self.img_dir  / (stem + ".png")
        mask_path = self.mask_dir / (stem + ".png")

        img  = _load_sonar_png(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # Resize image
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.resize(img_np, (self.img_size, self.img_size))
        img    = torch.from_numpy(img_np.transpose(2, 0, 1))

        # Resize mask with nearest-neighbour to keep label values exact
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        if self.transform:
            img, mask_tensor = self.transform(img, mask_tensor)

        return img, mask_tensor

    def pixel_class_counts(self) -> np.ndarray:
        """Count total pixels per class across ALL samples (slow — run once)."""
        counts = np.zeros(self.NUM_CLASSES, dtype=np.int64)
        for stem in self.stems:
            mask = cv2.imread(str(self.mask_dir / (stem + ".png")),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            for cls in range(self.NUM_CLASSES):
                counts[cls] += (mask == cls).sum()
        return counts

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Weight each sample by inverse frequency of its dominant foreground class."""
        pixel_counts = self.pixel_class_counts()
        pixel_counts = np.where(pixel_counts == 0, 1, pixel_counts)
        class_weights = 1.0 / pixel_counts

        sample_weights = []
        for stem in self.stems:
            mask = cv2.imread(str(self.mask_dir / (stem + ".png")),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None:
                sample_weights.append(1.0)
                continue
            # Use dominant non-background class for sample weight
            fg = mask[(mask > 0) & (mask < 11)]  # exclude background=0 and wall=11
            if len(fg) == 0:
                sample_weights.append(class_weights[0])
            else:
                unique, cnts = np.unique(fg, return_counts=True)
                dominant = unique[cnts.argmax()]
                sample_weights.append(class_weights[dominant])

        sample_weights = np.array(sample_weights, dtype=np.float64)
        return WeightedRandomSampler(weights=sample_weights,
                                     num_samples=len(sample_weights),
                                     replacement=True)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA = "marine-debris-fls-datasets/md_fls_dataset/data"

    print("── Classification: Watertank-Cropped ──")
    ds = FLSClassificationDataset(f"{DATA}/watertank-cropped", split="train")
    img, label = ds[0]
    print(f"  Classes ({len(ds.classes)}): {ds.classes}")
    print(f"  Train size: {len(ds)}  |  img shape: {img.shape}  |  label: {label}")
    counts = ds.class_counts()
    print(f"  Class counts: { {ds.classes[k]: v for k, v in counts.items()} }")

    print("\n── Detection: Watertank-Segmentation ──")
    det = FLSDetectionDataset(f"{DATA}/watertank-segmentation", split="train")
    img, target = det[0]
    print(f"  Train size: {len(det)}")
    print(f"  img shape: {img.shape}  |  boxes: {target['boxes'].shape}  |  labels: {target['labels']}")

    print("\n── Segmentation: Watertank-Segmentation ──")
    seg = FLSSegmentationDataset(f"{DATA}/watertank-segmentation", split="train")
    img, mask = seg[0]
    print(f"  Train size: {len(seg)}")
    print(f"  img shape: {img.shape}  |  mask shape: {mask.shape}  |  unique labels: {mask.unique().tolist()}")

    print("\nAll datasets loaded successfully.")
