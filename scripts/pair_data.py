import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairConfig:
    pairs_dir: str
    target_len: int = 1868
    test_size: float = 0.2
    split_seed: int = 42


def get_pair_file_lists(pairs_dir: str) -> Tuple[List[str], List[str]]:
    clean_files = sorted(
        os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("_clean.npy")
    )
    noisy_files = sorted(
        os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("_noisy.npy")
    )
    if not clean_files or not noisy_files:
        raise FileNotFoundError(f"No *_clean.npy / *_noisy.npy files found in {pairs_dir}")
    if len(clean_files) != len(noisy_files):
        raise ValueError("Mismatched clean/noisy file counts in pairs directory")
    return clean_files, noisy_files


def split_pair_files(
    clean_files: List[str],
    noisy_files: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    return train_test_split(clean_files, noisy_files, test_size=test_size, random_state=seed)


def normalize_joint(clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    joint_min = min(float(clean.min()), float(noisy.min()))
    joint_max = max(float(clean.max()), float(noisy.max()))
    scale = joint_max - joint_min + 1e-8
    clean_norm = (clean - joint_min) / scale
    noisy_norm = (noisy - joint_min) / scale
    return clean_norm.astype(np.float32), noisy_norm.astype(np.float32)


def normalize_zscore(clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clean_mean = float(clean.mean())
    clean_std = float(clean.std())
    noisy_mean = float(noisy.mean())
    noisy_std = float(noisy.std())
    clean_norm = (clean - clean_mean) / (clean_std + 1e-8)
    noisy_norm = (noisy - noisy_mean) / (noisy_std + 1e-8)
    return clean_norm.astype(np.float32), noisy_norm.astype(np.float32)


def normalize_input_zscore(clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    noisy_mean = float(noisy.mean())
    noisy_std = float(noisy.std())
    scale = noisy_std + 1e-8
    clean_norm = (clean - noisy_mean) / scale
    noisy_norm = (noisy - noisy_mean) / scale
    return clean_norm.astype(np.float32), noisy_norm.astype(np.float32)


def normalize_joint_zscore(clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([clean, noisy], axis=0)
    joint_mean = float(stacked.mean())
    joint_std = float(stacked.std())
    scale = joint_std + 1e-8
    clean_norm = (clean - joint_mean) / scale
    noisy_norm = (noisy - joint_mean) / scale
    return clean_norm.astype(np.float32), noisy_norm.astype(np.float32)


def normalize_pair(
    clean: np.ndarray,
    noisy: np.ndarray,
    normalization: str = "joint_minmax",
) -> Tuple[np.ndarray, np.ndarray]:
    if normalization == "joint_minmax":
        return normalize_joint(clean, noisy)
    if normalization == "zscore":
        return normalize_zscore(clean, noisy)
    if normalization == "input_zscore":
        return normalize_input_zscore(clean, noisy)
    if normalization == "joint_zscore":
        return normalize_joint_zscore(clean, noisy)
    if normalization == "none":
        return clean.astype(np.float32), noisy.astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {normalization}")


def denormalize_output(
    output: np.ndarray,
    clean_raw: np.ndarray,
    noisy_raw: np.ndarray,
    normalization: str = "joint_minmax",
) -> np.ndarray:
    output = output.astype(np.float32, copy=False)
    clean_raw = clean_raw.astype(np.float32, copy=False)
    noisy_raw = noisy_raw.astype(np.float32, copy=False)

    if normalization == "joint_minmax":
        joint_min = min(float(clean_raw.min()), float(noisy_raw.min()))
        joint_max = max(float(clean_raw.max()), float(noisy_raw.max()))
        scale = joint_max - joint_min + 1e-8
        return output * scale + joint_min
    if normalization == "zscore":
        clean_mean = float(clean_raw.mean())
        clean_std = float(clean_raw.std()) + 1e-8
        return output * clean_std + clean_mean
    if normalization == "input_zscore":
        noisy_mean = float(noisy_raw.mean())
        noisy_std = float(noisy_raw.std()) + 1e-8
        return output * noisy_std + noisy_mean
    if normalization == "joint_zscore":
        stacked = np.concatenate([clean_raw, noisy_raw], axis=0)
        joint_mean = float(stacked.mean())
        joint_std = float(stacked.std()) + 1e-8
        return output * joint_std + joint_mean
    if normalization == "none":
        return output
    raise ValueError(f"Unsupported normalization mode: {normalization}")


class FTIRPairsDataset(Dataset):
    def __init__(
        self,
        clean_files: List[str],
        noisy_files: List[str],
        target_len: int = 1868,
        augment_fn=None,
        normalization: str = "joint_minmax",
        cache_in_memory: bool = True,
    ):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.target_len = target_len
        self.augment_fn = augment_fn
        self.normalization = normalization
        self.cache_in_memory = cache_in_memory
        self._cache = {}
        if cache_in_memory:
            for idx in range(len(self.clean_files)):
                self._cache[idx] = self._load_pair(idx)

    def __len__(self) -> int:
        return len(self.clean_files)

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        clean = np.load(self.clean_files[idx]).astype(np.float32)
        noisy = np.load(self.noisy_files[idx]).astype(np.float32)
        if clean.shape[0] != self.target_len:
            clean = resample(clean, self.target_len).astype(np.float32)
        if noisy.shape[0] != self.target_len:
            noisy = resample(noisy, self.target_len).astype(np.float32)
        return clean, noisy

    def __getitem__(self, idx: int):
        if self.cache_in_memory:
            clean_raw, noisy_raw = self._cache[idx]
        else:
            clean_raw, noisy_raw = self._load_pair(idx)

        clean_aug = clean_raw.copy()
        noisy_aug = noisy_raw.copy()
        if self.augment_fn is not None:
            clean_aug, noisy_aug = self.augment_fn(clean_aug, noisy_aug)

        clean_aug, noisy_aug = normalize_pair(clean_aug, noisy_aug, normalization=self.normalization)

        clean_t = torch.from_numpy(clean_aug).unsqueeze(0)
        noisy_t = torch.from_numpy(noisy_aug).unsqueeze(0)
        return noisy_t, clean_t

    def get_raw_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_in_memory:
            clean_raw, noisy_raw = self._cache[idx]
        else:
            clean_raw, noisy_raw = self._load_pair(idx)
        return clean_raw.copy(), noisy_raw.copy()
