"""
data/dataloader.py
Factory functions to build train / val DataLoaders.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from .dataset import BraTSMultimodalDataset, BioClinicalBERTEncoder


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader) from config.

    A single BioClinicalBERTEncoder instance is shared between splits
    so the model is only loaded once (when use_precomputed_text=False).
    """
    flair_root = Path(cfg.paths.flair_root)
    text_root  = Path(cfg.paths.text_root)
    use_pre    = cfg.dataset.use_precomputed_text

    # Shared encoder (lazy-loaded)
    shared_bert = BioClinicalBERTEncoder(
        model_name=cfg.model.bert_model,
        max_length=cfg.dataset.max_text_length,
    ) if not use_pre else None

    train_ds = BraTSMultimodalDataset(
        flair_split_dir=flair_root / "train",
        text_root=text_root,
        split="train",
        cfg=cfg,
        use_precomputed_text=use_pre,
        augment=True,
        bert_encoder=shared_bert,
    )

    val_ds = BraTSMultimodalDataset(
        flair_split_dir=flair_root / "val",
        text_root=text_root,
        split="val",
        cfg=cfg,
        use_precomputed_text=use_pre,
        augment=False,
        bert_encoder=shared_bert,
    )

    train_ds.print_stats()
    val_ds.print_stats()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataloader.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    return train_loader, val_loader
