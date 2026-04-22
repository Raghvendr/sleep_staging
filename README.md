# sleep_staging

Code for sleep staging work.

## Overview

The preprocessing utilities in
[pre_processing/fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py)
convert Fitbit sleep-stage intervals and heart-rate measurements into training
samples for a sleep staging model.

For a step-by-step walkthrough of the full transformation with worked examples
and code-line references, see
[README_preprocess.md](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/README_preprocess.md:1).

For a step-by-step guide to the paper-inspired training pipeline, model,
nightly sequence construction, loss, and evaluation metrics, see
[README_training.md](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/README_training.md:1).

At a high level, the pipeline does this:

1. Reads Fitbit sleep-stage files and heart-rate files.
2. Standardizes the column names and stage names.
3. Converts sleep intervals into fixed-length epoch labels.
4. Resamples heart rate to a uniform sampling rate.
5. Extracts a heart-rate window for each epoch label.
6. Returns one combined dataset containing `X`, `y`, timestamps, window bounds,
   and patient ids.

## Fitbit Data Assumptions

The code assumes the Fitbit files have these columns.

Sleep file:
- `sleepId`
- `dateTime`
- `level`
- `seconds`

Heart-rate file:
- `Timestamp` or `timestamp`
- `HeartRatePPG` or `heart_rate`

Stage names are mapped as follows:

- `wake` -> `wake`
- `awake` -> `wake`
- `restless` -> `wake`
- `asleep` -> `light`
- `light` -> `light`
- `deep` -> `deep`
- `rem` -> `rem`

This means the final label space used by the preprocessing code is:

- `wake`
- `light`
- `deep`
- `rem`

## Main Concepts

### Epoch Labels

Fitbit sleep data is interval based. A row says that a stage began at a given
time and lasted for `seconds`.

Example:

```text
2018-04-09 23:20:00, light, 90
```

This does not directly give one label per training sample. The preprocessing
therefore converts intervals into fixed-size epochs, usually 30-second epochs.

If `epoch_sec=30`, a single night becomes labels like:

- `23:20:00`
- `23:20:30`
- `23:21:00`
- `23:21:30`

Each of those timestamps corresponds to one label.

### Windowed Heart-Rate Inputs

Each label is paired with a heart-rate window extracted from the resampled
heart-rate series.

If:

- `window_sec=60`
- `sampling_hz=1.0`

then each training sample contains:

- `60 * 1.0 = 60` heart-rate values

If:

- `window_sec=128`
- `sampling_hz=2.0`

then each training sample contains:

- `128 * 2.0 = 256` heart-rate values

### Why Timestamps Advance Every 30 Seconds Even When the Window Is 60 Seconds

This is expected when:

- `epoch_sec=30`
- `window_sec=60`

The dataset produces one label every 30 seconds, but each label can use a
60-second heart-rate window. That means consecutive samples overlap.

Example with `centered=True`:

- label timestamp `23:20:30` may use roughly `23:20:00` to `23:20:59`
- label timestamp `23:21:00` may use roughly `23:20:30` to `23:21:29`

So:

- `timestamps` step by `epoch_sec`
- `X[i]` length is determined by `window_sec * sampling_hz`

## Returned Dataset Object

Most top-level functions return a `WindowedSleepDataset`.

Fields:

- `X`: NumPy array of shape `(num_samples, samples_per_window)`
  Each row is one heart-rate window.
- `y`: NumPy array of shape `(num_samples,)`
  Integer labels aligned with `stage_names`.
- `timestamps`: `DatetimeIndex`
  The label timestamp for each training sample.
- `stage_names`: list of class names
  Currently `["wake", "light", "deep", "rem"]`.
- `samples_per_window`: integer
  Number of heart-rate values in each row of `X`.
- `sampling_hz`: float
  Sampling frequency used after resampling heart rate.
- `epoch_sec`: integer
  Spacing between label timestamps.
- `window_starts`: `DatetimeIndex`
  Exact start of the heart-rate window for each sample.
- `window_ends`: `DatetimeIndex`
  Exact end of the heart-rate window for each sample.
- `patient_ids`: NumPy array or `None`
  Patient id aligned one-to-one with samples in `X`.

Useful debug example:

```python
import pandas as pd

debug_df = pd.DataFrame({
    "patient_id": dataset.patient_ids,
    "label_timestamp": dataset.timestamps,
    "window_start": dataset.window_starts,
    "window_end": dataset.window_ends,
    "label": [dataset.stage_names[i] for i in dataset.y],
})

display(debug_df.head(10))
```

## Public Functions

### `load_fitbit_sleep_csv`

Signature:

```python
load_fitbit_sleep_csv(path)
```

Purpose:

- Reads one Fitbit sleep-stage file.
- Parses timestamps.
- Converts stage strings into the unified label set.
- Adds an `endTime` column using `dateTime + seconds`.

Parameters:

- `path`
  Path to one sleep file. Supports plain `.csv` and compressed `.csv.gz`
  because `pandas.read_csv(..., compression="infer")` is used.

Returns:

- A pandas DataFrame with at least:
  - `sleepId`
  - `dateTime`
  - `level`
  - `seconds`
  - `endTime`

Notes:

- Unknown or unsupported sleep stages are dropped because they map to `NaN`.

### `load_fitbit_heart_rate_csv`

Signature:

```python
load_fitbit_heart_rate_csv(path)
```

Purpose:

- Reads one Fitbit heart-rate file.
- Standardizes column names to `timestamp` and `heart_rate`.
- Parses timestamps and heart-rate values.

Parameters:

- `path`
  Path to one heart-rate file.

Returns:

- A pandas DataFrame with columns:
  - `timestamp`
  - `heart_rate`

Notes:

- Rows with invalid timestamps or heart-rate values are dropped.

### `collect_patient_file_pairs`

Signature:

```python
collect_patient_file_pairs(sleep_dir, heart_dir)
```

Purpose:

- Scans two folders.
- Matches sleep files and heart-rate files by patient id extracted from the
  filename.

Parameters:

- `sleep_dir`
  Directory containing sleep files.
- `heart_dir`
  Directory containing heart-rate files.

Returns:

- A list of tuples:
  - `(patient_id, sleep_path, heart_path)`

Matching behavior:

- A file named `abc.csv.gz` and another file named `abc.csv` are treated as
  having the same patient id: `abc`.
- Only ids present in both folders are returned.

### `collect_patient_file_pairs_from_lists`

Signature:

```python
collect_patient_file_pairs_from_lists(sleep_paths, heart_paths)
```

Purpose:

- Matches explicit lists of sleep files and heart-rate files by patient id.

Parameters:

- `sleep_paths`
  Iterable of explicit sleep file paths.
- `heart_paths`
  Iterable of explicit heart-rate file paths.

Returns:

- A list of tuples:
  - `(patient_id, sleep_path, heart_path)`

Why use this:

- Use this when you do not want to preprocess every patient in a directory.
- The order of the two input lists does not matter.

### `sleep_intervals_to_epoch_labels`

Signature:

```python
sleep_intervals_to_epoch_labels(
    sleep_df,
    epoch_sec=30,
    label_strategy="mode",
)
```

Purpose:

- Converts interval-based sleep staging into fixed-length epoch labels.

Parameters:

- `sleep_df`
  DataFrame produced by `load_fitbit_sleep_csv`.
- `epoch_sec`
  Length of each label epoch in seconds.
  Common value:
  - `30` for standard sleep staging epochs
- `label_strategy`
  How to assign a single label to an epoch.
  Options:
  - `"mode"`: choose the most frequent stage within the epoch
  - `"start"`: choose the stage at the beginning of the epoch

Returns:

- A DataFrame with columns:
  - `sleepId`
  - `timestamp`
  - `level`

Behavior details:

- The function expands each sleep interval to per-second labels internally.
- It then groups those seconds into epochs of length `epoch_sec`.
- Epoch timestamps mark the start of the epoch.

### `resample_heart_rate`

Signature:

```python
resample_heart_rate(
    heart_df,
    sampling_hz=1.0,
    interpolation="linear",
    normalize_per_night=False,
    segments=None,
)
```

Purpose:

- Converts irregularly sampled heart-rate measurements into a uniformly sampled
  time series.

Parameters:

- `heart_df`
  DataFrame produced by `load_fitbit_heart_rate_csv`.
- `sampling_hz`
  Target sampling frequency after resampling.
  Examples:
  - `1.0` means one sample per second
  - `2.0` means two samples per second
- `interpolation`
  Pandas interpolation method used after resampling.
  Default:
  - `"linear"`
- `normalize_per_night`
  Whether to z-score normalize heart rate separately inside each sleep segment.
  Options:
  - `False`: keep heart rate on its original scale
  - `True`: normalize each night independently
- `segments`
  Iterable of `(start, end)` timestamps defining sleep segments.
  This is only used when `normalize_per_night=True`.

Returns:

- A DataFrame with:
  - `timestamp`
  - `heart_rate`

Behavior details:

- Duplicate timestamps are averaged before resampling.
- Resampling happens on a regular time grid.
- Missing points after resampling are interpolated.
- If per-night normalization is enabled, each segment is standardized using its
  own mean and standard deviation.

### `build_windowed_training_data`

Signature:

```python
build_windowed_training_data(
    heart_df,
    epoch_labels,
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
    drop_incomplete_windows=True,
    normalize_per_window=False,
    patient_id=None,
)
```

Purpose:

- Takes resampled heart rate and epoch labels.
- Builds the final `(X, y)` training examples.

Parameters:

- `heart_df`
  Resampled heart-rate DataFrame, usually from `resample_heart_rate`.
- `epoch_labels`
  Epoch label DataFrame, usually from `sleep_intervals_to_epoch_labels`.
- `window_sec`
  Duration of the heart-rate window for each sample, in seconds.
  Examples:
  - `60` for a 60-second input window
  - `128` for a paper-like 128-second input window
- `epoch_sec`
  Step between successive labels.
  Examples:
  - `30` produces one label every 30 seconds
  - `60` produces non-overlapping 60-second labels if `window_sec=60`
- `sampling_hz`
  Heart-rate sampling rate used to interpret `window_sec`.
  Number of values per sample is:
  - `round(window_sec * sampling_hz)`
- `centered`
  Controls where the window is placed relative to the label timestamp.
  Options:
  - `True`: center the window on the label timestamp
  - `False`: use a trailing window that ends at `timestamp + epoch_sec`
- `drop_incomplete_windows`
  What to do when a full heart-rate window is not available.
  Options:
  - `True`: drop the sample
  - `False`: keep the sample and pad missing values with `NaN`
- `normalize_per_window`
  Whether to z-score normalize each individual window independently.
  Options:
  - `False`: keep values as they are after optional per-night normalization
  - `True`: normalize each row of `X` independently
- `patient_id`
  Optional patient id attached to each generated sample.

Returns:

- A `WindowedSleepDataset`

Behavior details:

- `timestamps` store label times, not every per-sample heart-rate timestamp.
- `window_starts` and `window_ends` store the exact window boundaries.
- If `epoch_sec < window_sec`, windows overlap.

Centered example with `window_sec=60`:

- label timestamp: `23:20:30`
- window: `23:20:00` to `23:21:00`

Trailing example with `window_sec=60` and `epoch_sec=30`:

- label timestamp: `23:20:30`
- window end: `23:21:00`
- window start: `23:20:00`

### `prepare_fitbit_training_data`

Signature:

```python
prepare_fitbit_training_data(
    sleep_path,
    heart_path,
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
    normalize_per_night=False,
    normalize_per_window=False,
)
```

Purpose:

- Single-patient convenience wrapper.
- Loads one sleep file and one heart-rate file.
- Creates epoch labels.
- Resamples heart rate.
- Builds the final dataset.

Parameters:

- `sleep_path`
  One patient sleep-stage file.
- `heart_path`
  The matching heart-rate file for the same patient.
- `window_sec`
  Passed to `build_windowed_training_data`.
- `epoch_sec`
  Passed to `sleep_intervals_to_epoch_labels` and used in window alignment.
- `sampling_hz`
  Passed to `resample_heart_rate` and `build_windowed_training_data`.
- `centered`
  Passed to `build_windowed_training_data`.
- `normalize_per_night`
  Passed to `resample_heart_rate`.
- `normalize_per_window`
  Passed to `build_windowed_training_data`.

Returns:

- A `WindowedSleepDataset` for one patient.

### `prepare_fitbit_training_data_from_directories`

Signature:

```python
prepare_fitbit_training_data_from_directories(
    sleep_dir,
    heart_dir,
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
    normalize_per_night=False,
    normalize_per_window=False,
)
```

Purpose:

- Multi-patient convenience wrapper for directory-based processing.

Parameters:

- `sleep_dir`
  Folder containing sleep files.
- `heart_dir`
  Folder containing heart-rate files.
- `window_sec`
  Duration of the input window in seconds.
- `epoch_sec`
  Label step in seconds.
- `sampling_hz`
  Heart-rate resampling frequency.
- `centered`
  Whether windows are centered or trailing.
- `normalize_per_night`
  Whether to normalize heart rate separately for each night.
- `normalize_per_window`
  Whether to normalize each extracted window independently.

Returns:

- A combined `WindowedSleepDataset` across all matched patient ids.

Notes:

- Matching is filename based.
- Patients without both file types are skipped automatically.

### `prepare_fitbit_training_data_from_file_lists`

Signature:

```python
prepare_fitbit_training_data_from_file_lists(
    sleep_paths,
    heart_paths,
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
    normalize_per_night=False,
    normalize_per_window=False,
)
```

Purpose:

- Multi-patient convenience wrapper for explicit file lists.

Parameters:

- `sleep_paths`
  Iterable of explicit sleep file paths.
- `heart_paths`
  Iterable of explicit heart-rate file paths.
- `window_sec`
  Duration of the input window in seconds.
- `epoch_sec`
  Label step in seconds.
- `sampling_hz`
  Heart-rate resampling frequency.
- `centered`
  Whether windows are centered or trailing.
- `normalize_per_night`
  Whether to normalize heart rate separately for each night.
- `normalize_per_window`
  Whether to normalize each extracted window independently.

Returns:

- A combined `WindowedSleepDataset` across all matched ids from the two lists.

Why use this:

- Use it when you want to manually control which patients are included.

### `prepare_fitbit_training_data_from_pairs`

Signature:

```python
prepare_fitbit_training_data_from_pairs(
    pairs,
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
    normalize_per_night=False,
    normalize_per_window=False,
)
```

Purpose:

- Lowest-level multi-patient wrapper.
- Accepts pre-matched `(patient_id, sleep_path, heart_path)` tuples.

Parameters:

- `pairs`
  Iterable of:
  - `(patient_id, sleep_path, heart_path)`
- `window_sec`
  Duration of the input window in seconds.
- `epoch_sec`
  Label step in seconds.
- `sampling_hz`
  Heart-rate resampling frequency.
- `centered`
  Whether windows are centered or trailing.
- `normalize_per_night`
  Whether to normalize heart rate separately for each night.
- `normalize_per_window`
  Whether to normalize each extracted window independently.

Returns:

- A combined `WindowedSleepDataset`.

Why use this:

- Use it when you already have your own matching logic and do not want the code
  to infer patient ids from filenames.

## Common Usage Patterns

### Single Patient

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data

dataset = prepare_fitbit_training_data(
    sleep_path="/Users/raghvendraomer/Downloads/fitbit/sleep-data/abc.csv/abc.csv",
    heart_path="/Users/raghvendraomer/Downloads/fitbit/heart-rate/abc.csv.gz",
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
)
```

### All Patients From Folders

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data_from_directories

dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
)
```

### Explicit Patient Files Only

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data_from_file_lists

sleep_paths = [
    "/Users/raghvendraomer/Downloads/fitbit/sleep-data/0e615090-ca26-4635-9590-f6e6cde1094f.csv/0e615090-ca26-4635-9590-f6e6cde1094f.csv",
    "/Users/raghvendraomer/Downloads/fitbit/sleep-data/7607c6de-7244-4ba1-ab7c-5c4c4d7151ad.csv/7607c6de-7244-4ba1-ab7c-5c4c4d7151ad.csv",
]

heart_paths = [
    "/Users/raghvendraomer/Downloads/fitbit/heart-rate/0e615090-ca26-4635-9590-f6e6cde1094f.csv.gz",
    "/Users/raghvendraomer/Downloads/fitbit/heart-rate/7607c6de-7244-4ba1-ab7c-5c4c4d7151ad.csv.gz",
]

dataset = prepare_fitbit_training_data_from_file_lists(
    sleep_paths=sleep_paths,
    heart_paths=heart_paths,
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
)
```

## Recommended Parameter Sets

### Simple 60-Second Local Setup

Use this when you want one-minute heart-rate windows sampled at 1 Hz:

```python
dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
    normalize_per_night=False,
    normalize_per_window=False,
)
```

Interpretation:

- labels every 30 seconds
- 60 heart-rate values per sample
- overlapping windows

### Non-Overlapping 60-Second Setup

Use this when you want each label and each window to correspond to its own
separate 60-second block:

```python
dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    window_sec=60,
    epoch_sec=60,
    sampling_hz=1.0,
    centered=True,
)
```

Interpretation:

- labels every 60 seconds
- 60 heart-rate values per sample
- no overlap if the windows line up continuously

### Paper-Like Setup

To approximate the paper "Deep learning for automated sleep staging using
instantaneous heart rate":

```python
dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    window_sec=128,
    epoch_sec=30,
    sampling_hz=2.0,
    centered=True,
    normalize_per_night=True,
)
```

Interpretation:

- labels every 30 seconds
- 256 heart-rate values per sample
- strong overlap between neighboring windows
- per-night normalization applied

## Practical Notes

- `timestamps` are label timestamps, not every timestamp inside `X`.
- `window_starts` and `window_ends` tell you the exact input segment used for
  each label.
- If you see timestamps spaced every 30 seconds while `X.shape[1] == 60`, that
  means you are generating one label every 30 seconds while feeding a 60-second
  window into the model.
- If you want strict one-minute blocks, use `epoch_sec=60`.
- If your sleep files and heart-rate files use the same patient id in the
  filename, directory and file-list matching will work automatically.
- If your filenames do not match cleanly, use
  `prepare_fitbit_training_data_from_pairs`.

## Minimal Debug Checks

```python
print("X shape:", dataset.X.shape)
print("y shape:", dataset.y.shape)
print("patient_ids shape:", None if dataset.patient_ids is None else dataset.patient_ids.shape)
print("classes:", dataset.stage_names)
print("first label timestamp:", dataset.timestamps[0])
print("first window start:", dataset.window_starts[0])
print("first window end:", dataset.window_ends[0])
print("first label:", dataset.stage_names[dataset.y[0]])
```
