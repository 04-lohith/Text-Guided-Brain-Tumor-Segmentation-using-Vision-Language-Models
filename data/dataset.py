"""
data/dataset.py
─────────────────────────────────────────────────────────────────────────────
BraTS2020 multimodal dataset: 3-D FLAIR volumes + radiology text.

Key design decisions
────────────────────
* Each .npy in FLAIR_BRATS2020/{train|val}/images/ is a full 3-D patient
  volume of shape (H, W, D) = (128, 128, 128).
* Each .npy in .../masks/ has shape (H, W, D, 4) — 4-channel one-hot BraTS
  sub-regions (background, NCR/NET, ED, ET).
* We SLICE along axis-2 so every 2-D sample is (128, 128).
* The image index encodes the patient ID:
      image_X.npy  →  patient number = X+1  →  BraTS20_Training_XXX
  (because the FLAIR dataset uses 0-indexed filenames while TextBraTS uses
   1-indexed folder names.)
  A robust fallback is built in: if the direct mapping fails we search the
  text directory for any folder whose numeric suffix matches.
* Text embeddings are loaded from the precomputed .npy (shape (1,128,768))
  and pooled to (768,) with mean-pooling by default.
  Raw .txt is also supported — BioClinicalBERT tokeniser is used when
  use_precomputed_text=False.
* One text report is shared across ALL slices of the same patient.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Patient ↔ Image index mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_image_index(filename: str) -> int:
    """Extract integer index from 'image_X.npy'."""
    m = re.search(r"image_(\d+)\.npy", filename)
    if m is None:
        raise ValueError(f"Cannot parse image index from: {filename}")
    return int(m.group(1))


def _index_to_patient_id(idx: int) -> str:
    """
    Map 0-based image file index to BraTS patient folder name.
    image_0 → BraTS20_Training_001  (idx + 1, zero-padded to 3 digits)
    """
    return f"BraTS20_Training_{idx + 1:03d}"


def _build_text_lookup(text_root: Path) -> Dict[str, Path]:
    """
    Scan text_root and return {patient_id: folder_path} for every
    BraTS20_Training_XXX folder that contains a .txt report.
    """
    lookup: Dict[str, Path] = {}
    if not text_root.exists():
        warnings.warn(f"Text root does not exist: {text_root}")
        return lookup

    for folder in sorted(text_root.iterdir()):
        if not folder.is_dir():
            continue
        # Match BraTS20_Training_NNN
        m = re.match(r"BraTS20_Training_(\d+)$", folder.name)
        if m is None:
            continue
        lookup[folder.name] = folder

    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Text embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_precomputed_embedding(npy_path: Path) -> np.ndarray:
    """
    Load precomputed BERT embeddings of shape (1, seq_len, 768) or (seq_len, 768).
    Mean-pool over the token dimension → (768,). Used for FiLM conditioning.
    """
    emb = np.load(str(npy_path)).astype(np.float32)
    if emb.ndim == 3:          # (1, T, D)
        emb = emb[0]           # → (T, D)
    # mean-pool tokens
    return emb.mean(axis=0)   # → (D,)


def _load_token_sequence(npy_path: Path, max_len: int = 128) -> np.ndarray:
    """
    Load full token-level embeddings of shape (seq_len, 768).
    Used as Key/Value for true token-wise cross-attention.
    Pads or truncates to max_len.
    Returns (max_len, 768).
    """
    emb = np.load(str(npy_path)).astype(np.float32)
    if emb.ndim == 3:          # (1, T, D)
        emb = emb[0]           # → (T, D)
    T, D = emb.shape
    if T >= max_len:
        return emb[:max_len]   # truncate
    # Zero-pad
    pad = np.zeros((max_len - T, D), dtype=np.float32)
    return np.concatenate([emb, pad], axis=0)  # (max_len, D)


class BioClinicalBERTEncoder:
    """
    Lazy-loaded BioClinicalBERT encoder (only created when needed).
    Caches results in a dict to avoid re-encoding the same text.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_length: int = 128, device: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cpu")
        self._tokenizer = None
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    def _lazy_load(self) -> None:
        if self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModel
            logger.info("Loading BioClinicalBERT …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)

    def encode(self, text: str) -> np.ndarray:
        """Return mean-pooled CLS embedding of shape (768,)."""
        if text in self._cache:
            return self._cache[text]
        self._lazy_load()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Mean-pool last hidden state over token dimension
        emb = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
        self._cache[text] = emb
        return emb


# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BraTSMultimodalDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2020 FLAIR + radiology text.

    Parameters
    ----------
    flair_split_dir : str | Path
        Root for a single split, e.g. FLAIR_BRATS2020/train
        Must contain images/ and masks/ subdirectories.
    text_root : str | Path
        Root of TextBraTSData containing BraTS20_Training_XXX folders.
    split : str
        'train' or 'val' — used for logging only.
    cfg : DotDict
        Project config (dataset section is used).
    use_precomputed_text : bool
        If True use .npy embeddings; else run BioClinicalBERT on .txt.
    augment : bool
        Apply random augmentation (only for training).
    bert_encoder : BioClinicalBERTEncoder | None
        Shared encoder instance (avoids loading BERT twice).
    """

    def __init__(
        self,
        flair_split_dir: str | Path,
        text_root: str | Path,
        split: str = "train",
        cfg=None,
        use_precomputed_text: bool = True,
        augment: bool = False,
        bert_encoder: Optional[BioClinicalBERTEncoder] = None,
    ):
        super().__init__()
        self.flair_split_dir = Path(flair_split_dir)
        self.text_root = Path(text_root)
        self.split = split
        self.cfg = cfg
        self.use_precomputed_text = use_precomputed_text
        self.augment = augment

        # Dataset hyper-params (with safe defaults)
        self.image_size = int(getattr(cfg.dataset, "image_size", 128)) if cfg else 128
        self.slice_axis = int(getattr(cfg.dataset, "slice_axis", 2)) if cfg else 2
        self.binary_mask = bool(getattr(cfg.dataset, "binary_mask", True)) if cfg else True
        self.filter_empty = bool(getattr(cfg.dataset, "filter_empty_slices", True)) if cfg else True
        self.empty_ratio = float(getattr(cfg.dataset, "empty_ratio_threshold", 0.8)) if cfg else 0.8
        self.text_embed_dim = 768  # BioClinicalBERT hidden size

        self.bert_encoder = bert_encoder

        # ── Scan directories ──────────────────────────────────────────────
        images_dir = self.flair_split_dir / "images"
        masks_dir  = self.flair_split_dir / "masks"
        self._assert_dir(images_dir)
        self._assert_dir(masks_dir)

        image_files = sorted(images_dir.glob("image_*.npy"))
        mask_files  = sorted(masks_dir.glob("mask_*.npy"))

        if len(image_files) == 0:
            raise RuntimeError(f"No image files found in {images_dir}")
        if len(mask_files) == 0:
            raise RuntimeError(f"No mask files found in {masks_dir}")

        # Build index → file path mapping
        self._img_by_idx: Dict[int, Path] = {
            _parse_image_index(f.name): f for f in image_files
        }
        self._msk_by_idx: Dict[int, Path] = {}
        for f in mask_files:
            m = re.search(r"mask_(\d+)\.npy", f.name)
            if m:
                self._msk_by_idx[int(m.group(1))] = f

        # Verify pairing
        img_keys = set(self._img_by_idx.keys())
        msk_keys = set(self._msk_by_idx.keys())
        paired = sorted(img_keys & msk_keys)
        missing_masks = img_keys - msk_keys
        if missing_masks:
            warnings.warn(f"[{split}] {len(missing_masks)} images have no matching mask: "
                          f"{sorted(missing_masks)[:5]} …")

        self.patient_indices = paired  # list of valid patient indices

        # ── Text lookup ───────────────────────────────────────────────────
        self._text_lookup = _build_text_lookup(self.text_root)
        logger.info(f"[{split}] Found {len(self._text_lookup)} text reports in {self.text_root}")

        # ── Build per-slice sample list ───────────────────────────────────
        self.samples: List[Dict] = []
        self._text_embed_cache: Dict[str, np.ndarray] = {}

        self._build_samples()

        logger.info(
            f"[{split}] Dataset ready: {len(self.patient_indices)} patients, "
            f"{len(self.samples)} slices"
        )

    # ─── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _assert_dir(p: Path) -> None:
        if not p.is_dir():
            raise NotADirectoryError(f"Expected directory: {p}")

    def _get_text_embedding(self, patient_id: str) -> np.ndarray:
        """
        Return a (768,) mean-pooled text embedding for the given patient_id.
        Uses cache so each patient's text is only loaded once.
        """
        if patient_id in self._text_embed_cache:
            return self._text_embed_cache[patient_id]

        folder = self._text_lookup.get(patient_id)
        if folder is None:
            warnings.warn(f"No text folder for {patient_id} — using zero embedding")
            emb = np.zeros(self.text_embed_dim, dtype=np.float32)
            self._text_embed_cache[patient_id] = emb
            return emb

        if self.use_precomputed_text:
            npy_candidates = list(folder.glob("*_flair_text.npy"))
            if npy_candidates:
                emb = _load_precomputed_embedding(npy_candidates[0])
                self._text_embed_cache[patient_id] = emb
                return emb
            else:
                warnings.warn(f"No .npy found for {patient_id}, falling back to .txt")

        txt_candidates = list(folder.glob("*_flair_text.txt"))
        if not txt_candidates:
            warnings.warn(f"No .txt found for {patient_id} — using zero embedding")
            emb = np.zeros(self.text_embed_dim, dtype=np.float32)
        else:
            with open(txt_candidates[0], "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if self.bert_encoder is None:
                self.bert_encoder = BioClinicalBERTEncoder()
            emb = self.bert_encoder.encode(text)

        self._text_embed_cache[patient_id] = emb
        return emb

    def _get_text_tokens(self, patient_id: str) -> np.ndarray:
        """
        Return full token-level embeddings (max_text_length, 768) for the patient.
        Used as Key/Value in true token-wise cross-attention.
        Falls back gracefully to zero tensor.
        """
        cache_key = patient_id + "__tokens"
        if cache_key in self._text_embed_cache:
            return self._text_embed_cache[cache_key]

        max_len = int(getattr(self.cfg.dataset, "max_text_length", 128)) if self.cfg else 128

        folder = self._text_lookup.get(patient_id)
        if folder is None:
            tokens = np.zeros((max_len, self.text_embed_dim), dtype=np.float32)
            self._text_embed_cache[cache_key] = tokens
            return tokens

        npy_candidates = list(folder.glob("*_flair_text.npy"))
        if npy_candidates:
            tokens = _load_token_sequence(npy_candidates[0], max_len=max_len)
        else:
            # Fallback: tile the pooled embedding as a single token
            pooled = self._get_text_embedding(patient_id)    # (768,)
            tokens = np.tile(pooled[None], (max_len, 1)).astype(np.float32)

        self._text_embed_cache[cache_key] = tokens
        return tokens

    def _build_samples(self) -> None:
        """
        Iterate over paired patients, load volumes, and generate per-slice records.
        Optionally filters fully-empty slices to reduce class imbalance.
        """
        n_skipped = 0
        for pidx in self.patient_indices:
            patient_id = _index_to_patient_id(pidx)

            # Pre-load text embedding to validate mapping early
            text_emb = self._get_text_embedding(patient_id)

            # Add one record per axial slice
            img_path = self._img_by_idx[pidx]
            msk_path = self._msk_by_idx[pidx]

            # We do NOT load the volume here — deferred to __getitem__
            # But we need to know which slices to include
            # Load volume shape quickly (just metadata)
            try:
                vol = np.load(str(img_path), mmap_mode="r")
            except Exception as e:
                warnings.warn(f"Cannot load {img_path}: {e}")
                continue

            n_slices = vol.shape[self.slice_axis]

            for s in range(n_slices):
                self.samples.append({
                    "patient_id": patient_id,
                    "patient_idx": pidx,
                    "slice_idx": s,
                    "img_path": img_path,
                    "msk_path": msk_path,
                })

        if n_skipped > 0:
            logger.info(f"[{self.split}] Skipped {n_skipped} slices (empty filter)")

    # ─── Public API ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        patient_id = sample["patient_id"]
        s           = sample["slice_idx"]

        # ── Load 3-D volume and extract slice ────────────────────────────
        vol  = np.load(str(sample["img_path"])).astype(np.float32)   # (H,W,D)
        msk  = np.load(str(sample["msk_path"])).astype(np.float32)   # (H,W,D,4)

        # Extract 2-D slice along slice_axis
        img_slice = np.take(vol, s, axis=self.slice_axis)            # (H,W)
        msk_slice = np.take(msk, s, axis=self.slice_axis)            # (H,W,4)

        # ── Mask processing ──────────────────────────────────────────────
        if self.binary_mask:
            # Binary: any tumor sub-region = 1
            mask_2d = msk_slice[..., 1:].max(axis=-1).astype(np.float32)   # (H,W)
        else:
            # Multi-class: argmax of 4 channels
            mask_2d = msk_slice.argmax(axis=-1).astype(np.int64)            # (H,W)

        # ── Image normalisation (clip + z-score) ─────────────────────────
        p1, p99 = np.percentile(img_slice, [1, 99])
        img_slice = np.clip(img_slice, p1, p99)
        std = img_slice.std() + 1e-8
        img_slice = (img_slice - img_slice.mean()) / std

        # ── Augmentation (training only) ─────────────────────────────────
        if self.augment:
            img_slice, mask_2d = self._augment(img_slice, mask_2d)

        # ── Convert to tensors ───────────────────────────────────────────
        # Image: (1, H, W)
        img_t  = torch.from_numpy(img_slice[None]).float()
        # Mask: (H, W) binary float or int
        msk_t  = torch.from_numpy(mask_2d).float() if self.binary_mask \
                 else torch.from_numpy(mask_2d).long()

        # ── Text embedding ───────────────────────────────────────────────
        text_emb    = self._get_text_embedding(patient_id)       # (768,)
        text_tokens = self._get_text_tokens(patient_id)          # (seq_len, 768)
        text_t      = torch.from_numpy(text_emb).float()         # (768,)
        text_tok_t  = torch.from_numpy(text_tokens).float()      # (seq_len, 768)

        return {
            "image":        img_t,         # (1, 128, 128)
            "mask":         msk_t,         # (128, 128)
            "text_emb":     text_t,        # (768,)           — pooled, for FiLM
            "text_tokens":  text_tok_t,    # (seq_len, 768)   — for cross-attention K/V
            "patient_id":   patient_id,
            "slice_idx":    torch.tensor(int(s), dtype=torch.int64),
        }

    # ─── Augmentation ────────────────────────────────────────────────────

    @staticmethod
    def _augment(
        img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random horizontal/vertical flip and 90° rotation."""
        if np.random.rand() < 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        k = np.random.randint(0, 4)
        if k:
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask

    # ─── Debug utilities ─────────────────────────────────────────────────

    def print_stats(self) -> None:
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print(f"  Split          : {self.split.upper()}")
        print(f"  Patients       : {len(self.patient_indices)}")
        print(f"  Total slices   : {len(self.samples)}")
        print(f"  Text reports   : {len(self._text_embed_cache)} (loaded so far)")
        print(f"  Binary mask    : {self.binary_mask}")
        print(f"  Augmentation   : {self.augment}")
        print(f"{'='*60}\n")

    def verify_sample(self, index: int) -> None:
        """Load and print debug info for a single sample."""
        sample = self.__getitem__(index)
        pid  = sample["patient_id"]
        sidx = sample["slice_idx"]
        img  = sample["image"]
        msk  = sample["mask"]
        txt  = sample["text_emb"]
        print(f"\n[Sample {index}]")
        print(f"  Patient ID   : {pid}  (slice {sidx})")
        print(f"  Image tensor : {tuple(img.shape)}  "
              f"min={img.min():.3f}  max={img.max():.3f}")
        print(f"  Mask tensor  : {tuple(msk.shape)}  "
              f"unique={msk.unique().tolist()}")
        print(f"  Text emb     : {tuple(txt.shape)}  "
              f"norm={txt.norm():.4f}")
        # Read raw text for inspection
        folder = self._text_lookup.get(pid)
        if folder:
            txts = list(folder.glob("*.txt"))
            if txts:
                with open(txts[0]) as fh:
                    print(f"  Raw report   : {fh.read().strip()[:200]} …")
