"""Обёртки PyTorch Dataset над генератором изображений."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .config import NUM_PLANES
from .dataset import ImageSource
from .io import list_images, load_gray


class RandomSets(IterableDataset):
    """Бесконечный поток случайных наборов по 4 изображения (генерируются на лету)."""

    def __init__(self, extra_dir=None, extra_prob: float = 0.5, seed: int = 0, k: int = NUM_PLANES):
        self.src = ImageSource(extra_dir, extra_prob)
        self.seed = seed
        self.k = k

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, int(torch.initial_seed() % 2 ** 31)])
        while True:
            yield torch.from_numpy(self.src.sample_set(rng, self.k).astype(np.float32) / 255.0)


class FolderSets(Dataset):
    """Фиксированные наборы: папка, в которой каждая подпапка содержит 4 изображения."""

    def __init__(self, root):
        self.sets = sorted(p for p in Path(root).iterdir() if p.is_dir())

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, i):
        imgs = np.stack([load_gray(p) for p in list_images(self.sets[i])[:NUM_PLANES]])
        return torch.from_numpy(imgs.astype(np.float32) / 255.0)
