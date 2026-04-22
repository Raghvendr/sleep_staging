# Preprocessing Walkthrough

This document explains the full data preparation flow implemented in
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:1),
step by step, with examples.

The goal of the preprocessing pipeline is to convert:

- interval-based Fitbit sleep-stage data
- irregular Fitbit heart-rate data

into:

- `X`: fixed-length heart-rate windows
- `y`: aligned sleep-stage labels
- timestamps and metadata for debugging

## End-to-End Flow

For one patient, the high-level function is
[prepare_fitbit_training_data](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:295).

It does these steps:

1. Load sleep CSV.
2. Load heart-rate CSV.
3. Convert sleep intervals into fixed epoch labels.
4. Resample heart rate to a regular timeline.
5. Slice one heart-rate window for each label timestamp.
6. Return a `WindowedSleepDataset`.

The core lines are:

```python
sleep_df = load_fitbit_sleep_csv(sleep_path)
heart_df = load_fitbit_heart_rate_csv(heart_path)
epoch_labels = sleep_intervals_to_epoch_labels(sleep_df, epoch_sec=epoch_sec)
heart_resampled = resample_heart_rate(...)
return build_windowed_training_data(...)
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:305)

## Step 1: Load And Standardize Sleep Data

Function:
[load_fitbit_sleep_csv](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:36)

What it does:

- reads one sleep CSV
- checks required columns
- converts `dateTime` to pandas datetime
- converts `seconds` to integer
- maps Fitbit stage names into the internal label set
- adds `endTime = dateTime + seconds`

Relevant code:

```python
df["dateTime"] = pd.to_datetime(df["dateTime"])
df["seconds"] = df["seconds"].astype(int)
df["level"] = df["level"].astype(str).str.lower().map(FITBIT_STAGE_MAP)
df["endTime"] = df["dateTime"] + pd.to_timedelta(df["seconds"], unit="s")
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:43)

### Example

Input sleep CSV rows:

```text
sleepId,dateTime,level,seconds
1,2018-04-09 23:20:00,light,90
1,2018-04-09 23:21:30,deep,60
1,2018-04-09 23:22:30,rem,30
```

After loading:

```text
sleepId  dateTime              level  seconds  endTime
1        2018-04-09 23:20:00   light  90       2018-04-09 23:21:30
1        2018-04-09 23:21:30   deep   60       2018-04-09 23:22:30
1        2018-04-09 23:22:30   rem    30       2018-04-09 23:23:00
```

This means:

- `light` from `23:20:00` to `23:21:29`
- `deep` from `23:21:30` to `23:22:29`
- `rem` from `23:22:30` to `23:22:59`

## Step 2: Load And Standardize Heart-Rate Data

Function:
[load_fitbit_heart_rate_csv](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:51)

What it does:

- reads one heart-rate CSV
- renames columns to internal names
- parses timestamps
- converts heart rate to numeric
- drops invalid rows

Relevant code:

```python
df = df.rename(columns=column_map)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:59)

### Example

Input heart-rate CSV rows:

```text
Timestamp,HeartRatePPG
2018-04-09 23:20:02,65
2018-04-09 23:20:07,64
2018-04-09 23:20:15,66
2018-04-09 23:20:22,65
```

After loading:

```text
timestamp              heart_rate
2018-04-09 23:20:02    65.0
2018-04-09 23:20:07    64.0
2018-04-09 23:20:15    66.0
2018-04-09 23:20:22    65.0
```

At this stage, heart rate is still irregularly sampled.

## Step 3: Convert Sleep Intervals Into Fixed Epoch Labels

Function:
[sleep_intervals_to_epoch_labels](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:114)

What it does:

- expands each sleep interval to per-second labels
- groups the per-second labels into fixed epochs
- assigns one label per epoch

Relevant code:

```python
per_second = pd.Series(index=pd.date_range(..., freq="1s", inclusive="left"), dtype="object")
...
per_second.loc[interval_index] = row.level
...
epoch_groups = per_second.groupby(pd.Grouper(freq=f"{epoch_sec}s"))
```

Code references:
- [fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:134)
- [fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:148)

### Example With `epoch_sec=30`

Suppose the sleep intervals are:

```text
23:20:00 -> 23:21:29  light
23:21:30 -> 23:22:29  deep
23:22:30 -> 23:22:59  rem
```

The per-second label stream becomes:

```text
23:20:00 light
23:20:01 light
...
23:21:29 light
23:21:30 deep
...
23:22:29 deep
23:22:30 rem
...
23:22:59 rem
```

Then the function groups into 30-second epochs:

```text
23:20:00 -> 23:20:29  light
23:20:30 -> 23:20:59  light
23:21:00 -> 23:21:29  light
23:21:30 -> 23:21:59  deep
23:22:00 -> 23:22:29  deep
23:22:30 -> 23:22:59  rem
```

Final epoch label table:

```text
timestamp              level
2018-04-09 23:20:00    light
2018-04-09 23:20:30    light
2018-04-09 23:21:00    light
2018-04-09 23:21:30    deep
2018-04-09 23:22:00    deep
2018-04-09 23:22:30    rem
```

### Label Selection Rule

The function supports:

- `label_strategy="mode"`
- `label_strategy="start"`

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:154)

`"mode"` means:

- choose the stage that appears most often inside the epoch

`"start"` means:

- choose the stage present at the first second of the epoch

## Step 4: Resample Heart Rate To A Regular Timeline

Function:
[resample_heart_rate](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:170)

What it does:

- sorts heart rate by timestamp
- averages duplicate timestamps
- resamples onto a regular interval
- interpolates missing values
- optionally normalizes each sleep segment independently

Relevant code:

```python
indexed = heart_df.set_index("timestamp")[["heart_rate"]].sort_index()
indexed = indexed.groupby(level=0).mean()

resampled = indexed.resample(freq).mean()
resampled["heart_rate"] = resampled["heart_rate"].interpolate(method=interpolation, limit_direction="both")
```

Code references:
- [fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:179)
- [fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:182)

### Example With `sampling_hz=1.0`

Suppose loaded heart rate is:

```text
23:20:02  65
23:20:07  64
23:20:15  66
23:20:22  65
```

After resampling to 1 second and interpolating, it becomes conceptually:

```text
23:20:02  65.0
23:20:03  64.8
23:20:04  64.6
23:20:05  64.4
23:20:06  64.2
23:20:07  64.0
...
23:20:15  66.0
...
23:20:22  65.0
```

Now heart rate exists on a regular timeline, which makes window slicing easy.

### Per-Night Normalization

If `normalize_per_night=True`, the function normalizes heart rate separately
within each sleep segment:

```python
part["heart_rate"] = (part["heart_rate"] - part["heart_rate"].mean()) / std
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:196)

This is the closest match to the paper’s nightly normalization.

## Step 5: Align Sleep Labels With Heart-Rate Windows

Function:
[build_windowed_training_data](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:203)

This is the most important alignment step in the pipeline.

What it does:

- loops over each epoch label
- computes `window_start` and `window_end`
- slices the resampled heart-rate series between those times
- stores the heart-rate values in `X`
- stores the stage label in `y`

Relevant code:

```python
for row in epoch_labels.itertuples(index=False):
    ...
    window = heart_series.loc[(heart_series.index >= window_start) & (heart_series.index < window_end)]
    values = window.to_numpy(dtype=np.float32)
```

Code references:
- [fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:240)
- [fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:252)

This is not a direct `merge` on matching timestamps. It is:

- label timestamp from sleep data
- time slice on heart-rate timeline
- attach the label to that sliced window

### Example With `window_sec=60`, `epoch_sec=30`, `sampling_hz=1.0`, `centered=True`

Suppose one epoch label is:

```text
timestamp = 2018-04-09 23:20:30
level = light
```

The code computes:

```python
half_window = 30 seconds
window_start = 23:20:00
window_end = 23:21:00
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:244)

Then it slices heart rate from:

```text
23:20:00 through 23:20:59
```

That gives:

- `X[i]`: 60 heart-rate values
- `y[i]`: label index for `light`
- `timestamps[i]`: `23:20:30`
- `window_starts[i]`: `23:20:00`
- `window_ends[i]`: `23:21:00`

### Next Example Row

Suppose the next label timestamp is:

```text
23:21:00
```

With the same settings:

- `window_start = 23:20:30`
- `window_end = 23:21:30`

So the next input window overlaps the previous one by 30 seconds.

This is why:

- timestamps can be 30 seconds apart
- but each `X[i]` still contains 60 samples

### Example With `centered=False`

If:

- `timestamp = 23:20:30`
- `window_sec = 128`
- `epoch_sec = 30`

then the code uses:

```python
window_end = timestamp + epoch_sec
window_start = window_end - window_sec
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:248)

That produces:

- `window_end = 23:21:00`
- `window_start = 23:18:52`

So:

- `centered=True` gives a window centered around the label
- `centered=False` gives a more trailing window

## Step 6: Build Final Arrays

Still inside
[build_windowed_training_data](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:203),
the function accumulates lists and then converts them to arrays.

Relevant code:

```python
X = np.stack(X_list) if X_list else np.empty((0, samples_per_window), dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:278)

The returned object is:

```python
WindowedSleepDataset(
    X=X,
    y=y,
    timestamps=...,
    window_starts=...,
    window_ends=...,
    stage_names=...,
    ...
)
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:281)

## Full Worked Example

Assume:

- `epoch_sec = 30`
- `window_sec = 60`
- `sampling_hz = 1.0`
- `centered = True`

### Raw Sleep Input

```text
sleepId,dateTime,level,seconds
1,2018-04-09 23:20:00,light,90
1,2018-04-09 23:21:30,deep,60
```

### Raw Heart-Rate Input

```text
Timestamp,HeartRatePPG
2018-04-09 23:20:02,65
2018-04-09 23:20:07,64
2018-04-09 23:20:15,66
2018-04-09 23:20:22,65
2018-04-09 23:20:31,67
2018-04-09 23:20:45,68
2018-04-09 23:20:57,66
2018-04-09 23:21:08,64
```

### Epoch Labels After Step 3

```text
23:20:00  light
23:20:30  light
23:21:00  light
23:21:30  deep
```

### Resampled Heart Rate After Step 4

Conceptually:

```text
23:20:00  ...
23:20:01  ...
23:20:02  65.0
23:20:03  ...
...
23:21:29  ...
```

### Final Sample Construction

Sample 1:

- label timestamp: `23:20:30`
- window: `23:20:00` to `23:21:00`
- label: `light`
- `X[0]`: 60 heart-rate values

Sample 2:

- label timestamp: `23:21:00`
- window: `23:20:30` to `23:21:30`
- label: `light`
- `X[1]`: next 60 heart-rate values

So the training data looks conceptually like:

```text
X[0] -> HR from 23:20:00 to 23:20:59, y[0] = light
X[1] -> HR from 23:20:30 to 23:21:29, y[1] = light
X[2] -> HR from 23:21:00 to 23:21:59, y[2] = light
X[3] -> HR from 23:21:30 to 23:22:29, y[3] = deep
```

## Multi-Patient Processing

There are three ways the code handles multiple patients.

### 1. Scan Directories

Function:
[collect_patient_file_pairs](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:81)

This extracts patient ids from filenames and keeps only ids present in both:

- sleep directory
- heart-rate directory

Then
[prepare_fitbit_training_data_from_directories](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:335)
passes those pairs into:
[prepare_fitbit_training_data_from_pairs](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:379)

### 2. Use Explicit File Lists

Function:
[collect_patient_file_pairs_from_lists](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:103)

This does the same matching, but from provided lists instead of scanning a
folder.

### 3. Use Pre-Matched Pairs

Function:
[prepare_fitbit_training_data_from_pairs](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:379)

This is the most explicit option. You provide:

```python
[
    ("patient_1", sleep_path_1, heart_path_1),
    ("patient_2", sleep_path_2, heart_path_2),
]
```

Each patient is processed independently using
[prepare_fitbit_training_data](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:295),
and then all outputs are concatenated.

Concatenation happens here:

```python
X=np.concatenate([dataset.X for dataset in datasets], axis=0)
y=np.concatenate([dataset.y for dataset in datasets], axis=0)
```

Code reference:
[fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py:432)

## How To Inspect The Final Alignment In A Notebook

Use this after creating `dataset`:

```python
import pandas as pd

debug_df = pd.DataFrame({
    "patient_id": dataset.patient_ids,
    "label_timestamp": dataset.timestamps,
    "window_start": dataset.window_starts,
    "window_end": dataset.window_ends,
    "label": [dataset.stage_names[i] for i in dataset.y],
})

display(debug_df.head(20))
```

To inspect one specific example:

```python
i = 0
print("patient       :", dataset.patient_ids[i])
print("label time    :", dataset.timestamps[i])
print("window start  :", dataset.window_starts[i])
print("window end    :", dataset.window_ends[i])
print("label         :", dataset.stage_names[dataset.y[i]])
print("samples in X  :", len(dataset.X[i]))
print("first 10 vals :", dataset.X[i][:10])
```

## Summary

The preprocessing pipeline does not merge sleep and heart rate by exact
timestamp equality.

Instead it does:

1. convert sleep intervals into label timestamps
2. resample heart rate onto a regular time axis
3. use each label timestamp to cut a heart-rate window
4. attach the sleep label to that window

That is how the final training pairs are created.
