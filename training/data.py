from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pre_processing.fitbit_preprocessing import WindowedSleepDataset


@dataclass(frozen=True)
class NightSequenceDataset:
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    patient_ids: np.ndarray
    night_ids: np.ndarray
    timestamps: np.ndarray
    stage_names: list[str]


def _infer_night_boundaries(
    patient_ids: np.ndarray,
    timestamps: np.ndarray,
    epoch_sec: int,
    gap_factor: float = 1.5,
) -> np.ndarray:
    boundaries = [0]
    for idx in range(1, len(timestamps)):
        same_patient = patient_ids[idx] == patient_ids[idx - 1]
        delta_sec = (timestamps[idx] - timestamps[idx - 1]) / np.timedelta64(1, "s")
        if (not same_patient) or (delta_sec > gap_factor * epoch_sec):
            boundaries.append(idx)
    boundaries.append(len(timestamps))
    return np.asarray(boundaries, dtype=np.int64)


def build_paper_night_sequences(
    dataset: WindowedSleepDataset,
    max_epochs: int = 1200,
    pad_value: float = 0.0,
    pad_label: int = -100,
    gap_factor: float = 1.5,
) -> NightSequenceDataset:
    if dataset.patient_ids is None:
        raise ValueError("WindowedSleepDataset.patient_ids is required to build nightly sequences")
    if dataset.window_starts is None or dataset.window_ends is None:
        raise ValueError("WindowedSleepDataset window bounds are required")

    patient_ids = np.asarray(dataset.patient_ids, dtype=object)
    timestamps = dataset.timestamps.to_numpy()
    boundaries = _infer_night_boundaries(patient_ids, timestamps, epoch_sec=dataset.epoch_sec, gap_factor=gap_factor)

    nights_X: list[np.ndarray] = []
    nights_y: list[np.ndarray] = []
    nights_mask: list[np.ndarray] = []
    nights_patient_ids: list[str] = []
    nights_ids: list[str] = []
    nights_timestamps: list[np.ndarray] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        X_night = dataset.X[start:end]
        y_night = dataset.y[start:end]
        ts_night = timestamps[start:end]
        patient_id = str(patient_ids[start])

        if len(X_night) == 0:
            continue

        if len(X_night) > max_epochs:
            X_night = X_night[:max_epochs]
            y_night = y_night[:max_epochs]
            ts_night = ts_night[:max_epochs]

        padded_X = np.full((max_epochs, X_night.shape[1]), pad_value, dtype=np.float32)
        padded_y = np.full((max_epochs,), pad_label, dtype=np.int64)
        padded_mask = np.zeros((max_epochs,), dtype=np.float32)
        padded_timestamps = np.full((max_epochs,), np.datetime64("NaT"), dtype="datetime64[ns]")

        valid_length = len(X_night)
        padded_X[:valid_length] = X_night
        padded_y[:valid_length] = y_night
        padded_mask[:valid_length] = 1.0
        padded_timestamps[:valid_length] = ts_night

        first_timestamp = str(ts_night[0]).replace(":", "-")
        night_id = f"{patient_id}__{first_timestamp}"

        nights_X.append(padded_X)
        nights_y.append(padded_y)
        nights_mask.append(padded_mask)
        nights_patient_ids.append(patient_id)
        nights_ids.append(night_id)
        nights_timestamps.append(padded_timestamps)

    if not nights_X:
        raise ValueError("No nightly sequences could be constructed from the provided dataset")

    return NightSequenceDataset(
        X=np.stack(nights_X).astype(np.float32),
        y=np.stack(nights_y).astype(np.int64),
        mask=np.stack(nights_mask).astype(np.float32),
        patient_ids=np.asarray(nights_patient_ids, dtype=object),
        night_ids=np.asarray(nights_ids, dtype=object),
        timestamps=np.stack(nights_timestamps),
        stage_names=dataset.stage_names,
    )


def split_by_patient_ids(
    night_dataset: NightSequenceDataset,
    train_patient_ids: Iterable[str],
    val_patient_ids: Iterable[str] | None = None,
    test_patient_ids: Iterable[str] | None = None,
) -> dict[str, NightSequenceDataset]:
    def subset(patient_ids_subset: set[str]) -> NightSequenceDataset:
        mask = np.isin(night_dataset.patient_ids, list(patient_ids_subset))
        return NightSequenceDataset(
            X=night_dataset.X[mask],
            y=night_dataset.y[mask],
            mask=night_dataset.mask[mask],
            patient_ids=night_dataset.patient_ids[mask],
            night_ids=night_dataset.night_ids[mask],
            timestamps=night_dataset.timestamps[mask],
            stage_names=night_dataset.stage_names,
        )

    result = {"train": subset(set(train_patient_ids))}
    if val_patient_ids is not None:
        result["val"] = subset(set(val_patient_ids))
    if test_patient_ids is not None:
        result["test"] = subset(set(test_patient_ids))
    return result


class SequenceTorchDataset(Dataset):
    def __init__(self, dataset: NightSequenceDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.dataset.X[idx]),
            "y": torch.from_numpy(self.dataset.y[idx]),
            "mask": torch.from_numpy(self.dataset.mask[idx]),
        }
