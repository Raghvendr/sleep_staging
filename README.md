# sleep_staging
codes for sleep staging work

## Fitbit preprocessing

Reusable Fitbit preprocessing helpers live in
[pre_processing/fitbit_preprocessing.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/pre_processing/fitbit_preprocessing.py).

Example:

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data

dataset = prepare_fitbit_training_data(
    sleep_path=".../sleep-data/<participant>.csv.gz",
    heart_path=".../heart-rate/<participant>.csv.gz",
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
)

print(dataset.X.shape, dataset.y.shape)
```

All patients from Google Drive folders:

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data_from_directories

dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/content/drive/MyDrive/fitbit/sleep-data",
    heart_dir="/content/drive/MyDrive/fitbit/heart-rate",
    window_sec=60,
    epoch_sec=30,
    sampling_hz=1.0,
    centered=True,
)

X, y = dataset.X, dataset.y
patient_ids = dataset.patient_ids

print(X.shape, y.shape, patient_ids.shape)
```

To approximate the paper "Deep learning for automated sleep staging using instantaneous heart rate",
use:

```python
dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/content/drive/MyDrive/fitbit/sleep-data",
    heart_dir="/content/drive/MyDrive/fitbit/heart-rate",
    window_sec=128,
    epoch_sec=30,
    sampling_hz=2.0,
    centered=True,
    normalize_per_night=True,
)
```
