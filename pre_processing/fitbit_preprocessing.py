from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FITBIT_STAGE_MAP = {
    "wake": "wake",
    "awake": "wake",
    "restless": "wake",
    "asleep": "light",
    "light": "light",
    "deep": "deep",
    "rem": "rem",
}


@dataclass(frozen=True)
class WindowedSleepDataset:
    X: np.ndarray
    y: np.ndarray
    timestamps: pd.DatetimeIndex
    stage_names: list[str]
    samples_per_window: int
    sampling_hz: float
    epoch_sec: int
    window_starts: pd.DatetimeIndex | None = None
    window_ends: pd.DatetimeIndex | None = None
    patient_ids: np.ndarray | None = None


def load_fitbit_sleep_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer").copy()
    required = {"sleepId", "dateTime", "level", "seconds"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Sleep file is missing columns: {sorted(missing)}")

    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df["seconds"] = df["seconds"].astype(int)
    df["level"] = df["level"].astype(str).str.lower().map(FITBIT_STAGE_MAP)
    df = df.dropna(subset=["level"]).sort_values(["sleepId", "dateTime"]).reset_index(drop=True)
    df["endTime"] = df["dateTime"] + pd.to_timedelta(df["seconds"], unit="s")
    return df


def load_fitbit_heart_rate_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer").copy()
    column_map = {
        "Timestamp": "timestamp",
        "timestamp": "timestamp",
        "HeartRatePPG": "heart_rate",
        "heart_rate": "heart_rate",
    }
    df = df.rename(columns=column_map)
    required = {"timestamp", "heart_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Heart-rate file is missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "heart_rate"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _patient_id_from_path(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    if name.endswith(".csv.gz"):
        return name[:-7]
    if name.endswith(".csv"):
        return name[:-4]
    return path.stem


def collect_patient_file_pairs(
    sleep_dir: str | Path,
    heart_dir: str | Path,
) -> list[tuple[str, Path, Path]]:
    sleep_dir = Path(sleep_dir)
    heart_dir = Path(heart_dir)

    sleep_files = {
        _patient_id_from_path(path): path
        for path in sorted(sleep_dir.glob("*.csv*"))
        if path.is_file()
    }
    heart_files = {
        _patient_id_from_path(path): path
        for path in sorted(heart_dir.glob("*.csv*"))
        if path.is_file()
    }

    common_ids = sorted(set(sleep_files).intersection(heart_files))
    return [(patient_id, sleep_files[patient_id], heart_files[patient_id]) for patient_id in common_ids]


def collect_patient_file_pairs_from_lists(
    sleep_paths: Iterable[str | Path],
    heart_paths: Iterable[str | Path],
) -> list[tuple[str, Path, Path]]:
    sleep_files = {_patient_id_from_path(path): Path(path) for path in sleep_paths}
    heart_files = {_patient_id_from_path(path): Path(path) for path in heart_paths}

    common_ids = sorted(set(sleep_files).intersection(heart_files))
    return [(patient_id, sleep_files[patient_id], heart_files[patient_id]) for patient_id in common_ids]


def sleep_intervals_to_epoch_labels(
    sleep_df: pd.DataFrame,
    epoch_sec: int = 30,
    label_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Convert interval-style Fitbit sleep stages into fixed-length epoch labels.

    The returned label timestamp marks the start of each epoch.
    """
    if label_strategy not in {"mode", "start"}:
        raise ValueError("label_strategy must be 'mode' or 'start'")

    rows: list[dict[str, object]] = []

    for sleep_id, group in sleep_df.groupby("sleepId", sort=False):
        group = group.sort_values("dateTime")
        start = group["dateTime"].min().floor(f"{epoch_sec}s")
        end = group["endTime"].max().ceil(f"{epoch_sec}s")

        per_second = pd.Series(index=pd.date_range(start=start, end=end, freq="1s", inclusive="left"), dtype="object")

        for row in group.itertuples(index=False):
            interval_index = pd.date_range(
                start=row.dateTime,
                end=row.endTime,
                freq="1s",
                inclusive="left",
            )
            per_second.loc[interval_index] = row.level

        if per_second.isna().all():
            continue

        epoch_groups = per_second.groupby(pd.Grouper(freq=f"{epoch_sec}s"))
        for epoch_start, epoch_values in epoch_groups:
            epoch_values = epoch_values.dropna()
            if epoch_values.empty:
                continue

            if label_strategy == "mode":
                label = epoch_values.mode().iloc[0]
            else:
                label = epoch_values.iloc[0]

            rows.append(
                {
                    "sleepId": sleep_id,
                    "timestamp": epoch_start,
                    "level": label,
                }
            )

    return pd.DataFrame(rows)


def resample_heart_rate(
    heart_df: pd.DataFrame,
    sampling_hz: float = 1.0,
    interpolation: str = "linear",
    normalize_per_night: bool = False,
    normalize_per_sleep_id: bool = False,
    segments: Iterable[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    freq = pd.to_timedelta(1 / sampling_hz, unit="s")

    indexed = heart_df.set_index("timestamp")[["heart_rate"]].sort_index()
    indexed = indexed.groupby(level=0).mean()

    resampled = indexed.resample(freq).mean()
    resampled["heart_rate"] = resampled["heart_rate"].interpolate(method=interpolation, limit_direction="both")
    resampled = resampled.reset_index()

    if (normalize_per_night or normalize_per_sleep_id) and segments is not None:
        normalized_parts: list[pd.DataFrame] = []
        for start, end in segments:
            part = resampled[(resampled["timestamp"] >= start) & (resampled["timestamp"] < end)].copy()
            if part.empty:
                continue
            std = part["heart_rate"].std()
            if pd.isna(std) or std == 0:
                part["heart_rate"] = 0.0
            else:
                part["heart_rate"] = (part["heart_rate"] - part["heart_rate"].mean()) / std
            normalized_parts.append(part)
        return pd.concat(normalized_parts, ignore_index=True) if normalized_parts else resampled.iloc[0:0].copy()

    return resampled


def build_windowed_training_data(
    heart_df: pd.DataFrame,
    epoch_labels: pd.DataFrame,
    window_sec: int = 60,
    epoch_sec: int = 30,
    sampling_hz: float = 1.0,
    centered: bool = True,
    drop_incomplete_windows: bool = True,
    normalize_per_window: bool = False,
    patient_id: str | None = None,
) -> WindowedSleepDataset:
    """
    Build window-label pairs.

    If centered=True, the window is centered on the label epoch start.
    For the paper, use `sampling_hz=2.0`, `epoch_sec=30`, `window_sec=128`,
    and `centered=True`.
    """
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")

    samples_per_window = int(round(window_sec * sampling_hz))
    if samples_per_window <= 0:
        raise ValueError("window_sec * sampling_hz must be at least 1")

    epoch_labels = epoch_labels.copy().sort_values("timestamp").reset_index(drop=True)
    heart_series = heart_df.copy().sort_values("timestamp").set_index("timestamp")["heart_rate"]
    stage_names = ["wake", "light", "deep", "rem"]
    stage_to_int = {name: i for i, name in enumerate(stage_names)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    ts_list: list[pd.Timestamp] = []
    window_start_list: list[pd.Timestamp] = []
    window_end_list: list[pd.Timestamp] = []
    patient_id_list: list[str] = []

    for row in epoch_labels.itertuples(index=False):
        if row.level not in stage_to_int:
            continue

        if centered:
            half_window = pd.to_timedelta(window_sec / 2, unit="s")
            window_start = row.timestamp - half_window
            window_end = window_start + pd.to_timedelta(window_sec, unit="s")
        else:
            window_end = row.timestamp + pd.to_timedelta(epoch_sec, unit="s")
            window_start = window_end - pd.to_timedelta(window_sec, unit="s")

        window = heart_series.loc[(heart_series.index >= window_start) & (heart_series.index < window_end)]
        values = window.to_numpy(dtype=np.float32)

        if len(values) != samples_per_window:
            if drop_incomplete_windows:
                continue
            padded = np.full(samples_per_window, np.nan, dtype=np.float32)
            padded[: min(len(values), samples_per_window)] = values[:samples_per_window]
            values = padded

        if normalize_per_window:
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if np.isnan(std) or std == 0:
                values = np.nan_to_num(values - mean, nan=0.0).astype(np.float32)
            else:
                values = ((values - mean) / std).astype(np.float32)

        X_list.append(values)
        y_list.append(stage_to_int[row.level])
        ts_list.append(row.timestamp)
        window_start_list.append(window_start)
        window_end_list.append(window_end)
        if patient_id is not None:
            patient_id_list.append(patient_id)

    X = np.stack(X_list) if X_list else np.empty((0, samples_per_window), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    return WindowedSleepDataset(
        X=X,
        y=y,
        timestamps=pd.DatetimeIndex(ts_list),
        window_starts=pd.DatetimeIndex(window_start_list),
        window_ends=pd.DatetimeIndex(window_end_list),
        stage_names=stage_names,
        samples_per_window=samples_per_window,
        sampling_hz=sampling_hz,
        epoch_sec=epoch_sec,
        patient_ids=np.asarray(patient_id_list, dtype=object) if patient_id is not None else None,
    )


def prepare_fitbit_training_data(
    sleep_path: str | Path,
    heart_path: str | Path,
    window_sec: int = 60,
    epoch_sec: int = 30,
    sampling_hz: float = 1.0,
    centered: bool = True,
    normalize_per_night: bool = False,
    normalize_per_sleep_id: bool = False,
    normalize_per_window: bool = False,
) -> WindowedSleepDataset:
    patient_id = _patient_id_from_path(sleep_path)
    sleep_df = load_fitbit_sleep_csv(sleep_path)
    heart_df = load_fitbit_heart_rate_csv(heart_path)
    epoch_labels = sleep_intervals_to_epoch_labels(sleep_df, epoch_sec=epoch_sec)

    segments = list(
        zip(
            sleep_df.groupby("sleepId")["dateTime"].min(),
            sleep_df.groupby("sleepId")["endTime"].max(),
        )
    )
    heart_resampled = resample_heart_rate(
        heart_df,
        sampling_hz=sampling_hz,
        normalize_per_night=normalize_per_night,
        normalize_per_sleep_id=normalize_per_sleep_id,
        segments=segments,
    )

    return build_windowed_training_data(
        heart_df=heart_resampled,
        epoch_labels=epoch_labels,
        window_sec=window_sec,
        epoch_sec=epoch_sec,
        sampling_hz=sampling_hz,
        centered=centered,
        normalize_per_window=normalize_per_window,
        patient_id=patient_id,
    )


def prepare_fitbit_training_data_from_directories(
    sleep_dir: str | Path,
    heart_dir: str | Path,
    window_sec: int = 60,
    epoch_sec: int = 30,
    sampling_hz: float = 1.0,
    centered: bool = True,
    normalize_per_night: bool = False,
    normalize_per_sleep_id: bool = False,
    normalize_per_window: bool = False,
) -> WindowedSleepDataset:
    pairs = collect_patient_file_pairs(sleep_dir=sleep_dir, heart_dir=heart_dir)
    return prepare_fitbit_training_data_from_pairs(
        pairs=pairs,
        window_sec=window_sec,
        epoch_sec=epoch_sec,
        sampling_hz=sampling_hz,
        centered=centered,
        normalize_per_night=normalize_per_night,
        normalize_per_sleep_id=normalize_per_sleep_id,
        normalize_per_window=normalize_per_window,
    )


def prepare_fitbit_training_data_from_file_lists(
    sleep_paths: Iterable[str | Path],
    heart_paths: Iterable[str | Path],
    window_sec: int = 60,
    epoch_sec: int = 30,
    sampling_hz: float = 1.0,
    centered: bool = True,
    normalize_per_night: bool = False,
    normalize_per_sleep_id: bool = False,
    normalize_per_window: bool = False,
) -> WindowedSleepDataset:
    pairs = collect_patient_file_pairs_from_lists(sleep_paths=sleep_paths, heart_paths=heart_paths)
    return prepare_fitbit_training_data_from_pairs(
        pairs=pairs,
        window_sec=window_sec,
        epoch_sec=epoch_sec,
        sampling_hz=sampling_hz,
        centered=centered,
        normalize_per_night=normalize_per_night,
        normalize_per_sleep_id=normalize_per_sleep_id,
        normalize_per_window=normalize_per_window,
    )


def prepare_fitbit_training_data_from_pairs(
    pairs: Iterable[tuple[str, str | Path, str | Path]],
    window_sec: int = 60,
    epoch_sec: int = 30,
    sampling_hz: float = 1.0,
    centered: bool = True,
    normalize_per_night: bool = False,
    normalize_per_sleep_id: bool = False,
    normalize_per_window: bool = False,
) -> WindowedSleepDataset:
    pairs = [(patient_id, Path(sleep_path), Path(heart_path)) for patient_id, sleep_path, heart_path in pairs]
    if not pairs:
        raise ValueError("No matching patient ids were found")

    datasets: list[WindowedSleepDataset] = []
    for patient_id, sleep_path, heart_path in pairs:
        dataset = prepare_fitbit_training_data(
            sleep_path=sleep_path,
            heart_path=heart_path,
            window_sec=window_sec,
            epoch_sec=epoch_sec,
            sampling_hz=sampling_hz,
            centered=centered,
            normalize_per_night=normalize_per_night,
            normalize_per_sleep_id=normalize_per_sleep_id,
            normalize_per_window=normalize_per_window,
        )
        if dataset.X.shape[0] == 0:
            continue
        if dataset.patient_ids is None:
            dataset = WindowedSleepDataset(
                X=dataset.X,
                y=dataset.y,
                timestamps=dataset.timestamps,
                window_starts=dataset.window_starts,
                window_ends=dataset.window_ends,
                stage_names=dataset.stage_names,
                samples_per_window=dataset.samples_per_window,
                sampling_hz=dataset.sampling_hz,
                epoch_sec=dataset.epoch_sec,
                patient_ids=np.full(dataset.y.shape[0], patient_id, dtype=object),
            )
        datasets.append(dataset)

    if not datasets:
        samples_per_window = int(round(window_sec * sampling_hz))
        return WindowedSleepDataset(
            X=np.empty((0, samples_per_window), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            timestamps=pd.DatetimeIndex([]),
            window_starts=pd.DatetimeIndex([]),
            window_ends=pd.DatetimeIndex([]),
            stage_names=["wake", "light", "deep", "rem"],
            samples_per_window=samples_per_window,
            sampling_hz=sampling_hz,
            epoch_sec=epoch_sec,
            patient_ids=np.empty((0,), dtype=object),
        )

    return WindowedSleepDataset(
        X=np.concatenate([dataset.X for dataset in datasets], axis=0),
        y=np.concatenate([dataset.y for dataset in datasets], axis=0),
        timestamps=pd.DatetimeIndex(
            np.concatenate([dataset.timestamps.to_numpy() for dataset in datasets], axis=0)
        ),
        window_starts=pd.DatetimeIndex(
            np.concatenate([dataset.window_starts.to_numpy() for dataset in datasets], axis=0)
        ),
        window_ends=pd.DatetimeIndex(
            np.concatenate([dataset.window_ends.to_numpy() for dataset in datasets], axis=0)
        ),
        stage_names=datasets[0].stage_names,
        samples_per_window=datasets[0].samples_per_window,
        sampling_hz=datasets[0].sampling_hz,
        epoch_sec=datasets[0].epoch_sec,
        patient_ids=np.concatenate(
            [
                dataset.patient_ids
                if dataset.patient_ids is not None
                else np.full(dataset.y.shape[0], "unknown", dtype=object)
                for dataset in datasets
            ],
            axis=0,
        ),
    )
