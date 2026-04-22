# Training Walkthrough

This document explains the training pipeline implemented in:

- [modeling/paper_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:1)
- [training/data.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/data.py:1)
- [training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:1)
- [training/metrics.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/metrics.py:1)
- [train_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/train_sleep_model.py:1)

The goal is to train a paper-inspired sleep staging model on the preprocessed
Fitbit data.

This guide explains:

1. what the model expects as input
2. how the preprocessed windows are converted into nightly sequences
3. how the paper-style model is structured
4. how training and validation are run
5. how evaluation metrics are computed
6. how to run the full training flow with examples

## Big Picture

The paper does not train on isolated windows independently. It trains on
sequences of heart-rate patches covering a whole night.

So the training flow here is:

1. preprocess Fitbit data into overlapping windows
2. group those windows back into nightly sequences
3. pad each night to a fixed length
4. feed the full nightly sequence into the model
5. compute loss only on valid epochs, ignoring padded positions

This is why there are two data stages:

- `WindowedSleepDataset`
  from preprocessing
- `NightSequenceDataset`
  for training

## Step 1: Start From Preprocessed Windowed Data

You first create the paper-like preprocessing output:

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data_from_directories

windowed_dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    window_sec=128,
    epoch_sec=30,
    sampling_hz=2.0,
    centered=True,
    normalize_per_night=True,
)
```

Why these values:

- `window_sec=128`
  because the paper uses 256-sample patches and 2 Hz sampling
- `epoch_sec=30`
  because the paper predicts one label every 30 seconds
- `sampling_hz=2.0`
  because the paper resamples IHR to 2 Hz
- `centered=True`
  because the paper centers each patch on the epoch
- `normalize_per_night=True`
  because the paper normalizes each night independently

### What This Gives You

At this stage:

- `windowed_dataset.X` shape is roughly:
  - `(num_epochs_total, 256)`
- each row is one 128-second patch
- `windowed_dataset.y` contains one label per row
- `windowed_dataset.patient_ids` tells you which patient each row belongs to
- `windowed_dataset.timestamps` tells you which 30-second epoch that row
  corresponds to

This is still not the final training input to the paper-style model.

## Step 2: Convert Windowed Data Into Nightly Sequences

Function:
[build_paper_night_sequences](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/data.py:34)

### Why This Step Exists

The paper model expects a whole-night sequence, not a shuffled collection of
independent epochs.

So we need to regroup:

- all epochs from the same patient
- belonging to the same night

into one array of shape:

- `epochs_in_night x patch_samples`

Then every night is padded or clipped to a fixed sequence length.

### How Night Boundaries Are Inferred

Night boundaries are detected by
[ _infer_night_boundaries ](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/data.py:21).

It starts a new night if:

- the patient id changes, or
- the time gap between consecutive labels is larger than `gap_factor * epoch_sec`

Relevant code:

```python
same_patient = patient_ids[idx] == patient_ids[idx - 1]
delta_sec = (timestamps[idx] - timestamps[idx - 1]) / np.timedelta64(1, "s")
if (not same_patient) or (delta_sec > gap_factor * epoch_sec):
    boundaries.append(idx)
```

Code reference:
[training/data.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/data.py:27)

### Padding To Paper Length

The paper uses:

- `1200` epochs per night

because:

- `10 hours`
- one epoch every `30 seconds`
- `10 * 60 * 60 / 30 = 1200`

This is handled here:

```python
padded_X = np.full((max_epochs, X_night.shape[1]), pad_value, dtype=np.float32)
padded_y = np.full((max_epochs,), pad_label, dtype=np.int64)
padded_mask = np.zeros((max_epochs,), dtype=np.float32)
```

Code reference:
[training/data.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/data.py:67)

### Example

Suppose one night has:

- `840` valid epochs
- each epoch has `256` HR values

Then:

- original shape: `(840, 256)`
- padded shape: `(1200, 256)`

The first `840` rows are real.
The remaining `360` rows are padding.

The matching label vector becomes:

- real labels for the first `840`
- `-100` for the remaining padded positions

The mask becomes:

- `1` for real epochs
- `0` for padded epochs

### Output

This function returns a `NightSequenceDataset` with:

- `X`: shape `(num_nights, 1200, 256)`
- `y`: shape `(num_nights, 1200)`
- `mask`: shape `(num_nights, 1200)`
- `patient_ids`: one patient id per night
- `night_ids`: one id per night
- `timestamps`: one timestamp sequence per night

Example usage:

```python
from training.data import build_paper_night_sequences

nightly_dataset = build_paper_night_sequences(
    dataset=windowed_dataset,
    max_epochs=1200,
    pad_value=0.0,
    pad_label=-100,
)
```

## Step 3: Split Train, Validation, And Test By Patient

Function:
[split_by_patient_ids](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/data.py:101)

### Why Split By Patient

If the same patient appears in both train and test, the evaluation becomes too
optimistic because the model may learn subject-specific patterns.

So the split should happen by patient, not by epoch.

### Example

```python
from training.data import split_by_patient_ids

splits = split_by_patient_ids(
    night_dataset=nightly_dataset,
    train_patient_ids=["p1", "p2", "p3"],
    val_patient_ids=["p4"],
    test_patient_ids=["p5"],
)
```

This returns:

- `splits["train"]`
- `splits["val"]`
- `splits["test"]`

Each is still a `NightSequenceDataset`.

## Step 4: Understand The Paper-Inspired Model

Model file:
[paper_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:1)

Main class:
[PaperSleepStageModel](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:90)

### Model Input

Expected input shape:

- `batch x epochs x patch_samples`

For paper-like settings:

- `batch x 1200 x 256`

Meaning:

- `batch`
  number of nights in the batch
- `1200`
  epochs per night
- `256`
  HR values per epoch patch

### Model Output

Output shape:

- `batch x epochs x num_classes`

For 4 classes:

- `batch x 1200 x 4`

Each epoch gets logits for:

- wake
- light
- deep
- rem

### Architecture Intuition

The model has two main stages.

#### Stage A: Local Feature Extraction Per Patch

This is done by the local convolution blocks.

Code reference:
[paper_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:110)

Intuition:

- Each 256-sample patch contains local HR shape information.
- The local CNN extracts useful local patterns such as short-term HR changes.

Each block:

- applies two Conv1D layers
- uses LeakyReLU
- uses pooling
- keeps a residual connection

Block definition:
[LocalConvBlock](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:27)

#### Stage B: Temporal Context Across The Whole Night

This is done by the dilated residual blocks.

Code reference:
[paper_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:133)

Intuition:

- Sleep staging depends not just on one epoch, but on neighboring epochs.
- Dilated convolutions let the model see short-range and long-range temporal
  context without using an RNN.

Dilations used:

- `2, 4, 8, 16, 32`

This means the model can learn relationships across many minutes of the night.

#### Final Classifier

After temporal modeling, a `1x1` Conv1D maps the embedding at each epoch to 4
class logits.

Code reference:
[paper_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:145)

### Config Object

The model is configured through
[PaperSleepStageModelConfig](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/modeling/paper_sleep_model.py:9).

Important fields:

- `num_classes=4`
- `patch_samples=256`
- `embedding_dim=128`
- `local_blocks=3`
- `temporal_blocks=2`
- `dilations=(2, 4, 8, 16, 32)`
- `dropout=0.2`

Example:

```python
from modeling import PaperSleepStageModel, PaperSleepStageModelConfig

model_config = PaperSleepStageModelConfig()
model = PaperSleepStageModel(config=model_config)
```

## Step 5: Training Configuration

Training config class:
[PaperTrainingConfig](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:14)

Fields:

- `batch_size=2`
- `learning_rate=1e-4`
- `weight_decay=0.25`
- `epochs=20`
- `device="cpu"`
- `l1_lambda=None`
- `num_workers=0`

### Intuition For Key Parameters

- `batch_size=2`
  The paper used very small batches because the nightly sequence input is large.
- `learning_rate=1e-4`
  Standard conservative learning rate for Adam.
- `weight_decay=0.25`
  In this implementation this acts as the default L1 regularization strength
  unless `l1_lambda` is explicitly provided.
- `epochs`
  Controls how many passes through the training set you run.
- `device`
  Use `"cuda"` if GPU is available.

Example:

```python
from training import PaperTrainingConfig

train_config = PaperTrainingConfig(
    batch_size=2,
    learning_rate=1e-4,
    weight_decay=0.25,
    epochs=20,
    device="cuda",
)
```

## Step 6: Loss Function And Padding Mask

Function:
[masked_cross_entropy_loss](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:24)

### Why Masked Loss Is Needed

All nights are padded to 1200 epochs, but many nights are shorter than that.

We do not want the model to learn from padded labels.

So:

- padded positions use label `-100`
- mask is `0` at padded positions
- only real epochs contribute to loss

Relevant code:

```python
losses = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=-100)
losses = losses * mask_flat
denom = mask_flat.sum().clamp(min=1.0)
return losses.sum() / denom
```

Code reference:
[training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:33)

### Intuition

Without masking:

- padding would create fake training targets
- gradients would be corrupted

With masking:

- the model only learns from real sleep epochs

## Step 7: One Training Epoch

Function:
[train_one_epoch](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:54)

What it does:

1. loops over batches of nightly sequences
2. moves data to CPU or GPU
3. computes logits
4. computes masked loss
5. optionally adds L1 regularization
6. backpropagates
7. updates weights

Relevant code:

```python
logits = model(x)
loss = masked_cross_entropy_loss(logits=logits, targets=y, mask=mask)
loss.backward()
optimizer.step()
```

Code references:
- [training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:71)
- [training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:74)

### Intuition

Each batch contains a few complete nights.
The model predicts all epochs in those nights together.

This preserves temporal context inside the night.

## Step 8: Full Training Loop

Function:
[fit_model](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:145)

What it does:

1. creates Adam optimizer
2. creates training DataLoader
3. runs `train_one_epoch` repeatedly
4. evaluates on validation set after each epoch
5. keeps the best model based on validation kappa

Relevant code:

```python
optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
...
train_loss = train_one_epoch(...)
...
val_metrics = evaluate_model(...)
...
if float(val_metrics["kappa"]) > best_val_kappa:
    best_state_dict = ...
```

Code references:
- [training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:155)
- [training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:167)
- [training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:182)

### Why Kappa Is Used For Model Selection

Sleep staging is class-imbalanced.
Accuracy alone can be misleading.

Cohen’s kappa adjusts for agreement expected by chance and is commonly reported
in sleep staging papers.

## Step 9: Evaluation Metrics

Metric functions live in
[training/metrics.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/metrics.py:1).

Main entry point:
[compute_classification_metrics](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/metrics.py:45)

### Metrics Reported

The evaluation code returns:

- `accuracy`
- `kappa`
- `macro_f1`
- `confusion_matrix`
- `per_class`
- nightly mean and std for accuracy
- nightly mean and std for kappa

### Why These Metrics Matter

- `accuracy`
  overall correctness
- `kappa`
  agreement adjusted for chance
- `macro_f1`
  balances performance across all classes
- `confusion_matrix`
  shows which classes get confused
- `per_class precision/recall/f1`
  tells you if wake, rem, deep, or light is weak

### Evaluation Function

Use:

```python
from training import evaluate_model

metrics = evaluate_model(
    model=model,
    dataset=splits["test"],
    device="cuda",
    batch_size=2,
)
```

Code reference:
[training/trainer.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/training/trainer.py:84)

## Step 10: Full Example Training Run

The file
[train_sleep_model.py](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/train_sleep_model.py:1)
contains a ready example function:
[example_training_run](/Users/raghvendraomer/Projects/Sleep_Staging/sleep_staging/train_sleep_model.py:13)

### Example

```python
from train_sleep_model import example_training_run

result = example_training_run(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    train_patient_ids=["id_1", "id_2", "id_3"],
    val_patient_ids=["id_4"],
    test_patient_ids=["id_5"],
)
```

Returned keys:

- `result["model_config"]`
- `result["training_config"]`
- `result["history"]`
- `result["val_metrics"]`
- `result["test_metrics"]`
- `result["model"]`

## Step 11: Recommended Notebook Workflow

A practical notebook flow is:

1. preprocess all patients with paper-like settings
2. convert to nightly sequences
3. inspect shapes
4. define train, validation, and test patient splits
5. initialize model
6. fit model
7. evaluate metrics
8. inspect confusion matrix and per-class results

Example:

```python
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data_from_directories
from training.data import build_paper_night_sequences, split_by_patient_ids
from modeling import PaperSleepStageModel, PaperSleepStageModelConfig
from training import PaperTrainingConfig, fit_model, evaluate_model

windowed_dataset = prepare_fitbit_training_data_from_directories(
    sleep_dir="/Users/raghvendraomer/Downloads/fitbit/sleep-data",
    heart_dir="/Users/raghvendraomer/Downloads/fitbit/heart-rate",
    window_sec=128,
    epoch_sec=30,
    sampling_hz=2.0,
    centered=True,
    normalize_per_night=True,
)

nightly_dataset = build_paper_night_sequences(
    dataset=windowed_dataset,
    max_epochs=1200,
    pad_value=0.0,
    pad_label=-100,
)

splits = split_by_patient_ids(
    night_dataset=nightly_dataset,
    train_patient_ids=["id_1", "id_2", "id_3"],
    val_patient_ids=["id_4"],
    test_patient_ids=["id_5"],
)

model = PaperSleepStageModel(PaperSleepStageModelConfig())
config = PaperTrainingConfig(
    batch_size=2,
    learning_rate=1e-4,
    weight_decay=0.25,
    epochs=20,
    device="cuda",
)

fit_output = fit_model(
    model=model,
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    config=config,
)

test_metrics = evaluate_model(
    model=fit_output["model"],
    dataset=splits["test"],
    device=config.device,
    batch_size=config.batch_size,
)

print(test_metrics["accuracy"])
print(test_metrics["kappa"])
print(test_metrics["macro_f1"])
print(test_metrics["confusion_matrix"])
```

## Step 12: Shape Checklist

Use this checklist to debug shapes.

After preprocessing:

- `windowed_dataset.X.shape == (num_epochs_total, 256)`
- `windowed_dataset.y.shape == (num_epochs_total,)`

After nightly conversion:

- `nightly_dataset.X.shape == (num_nights, 1200, 256)`
- `nightly_dataset.y.shape == (num_nights, 1200)`
- `nightly_dataset.mask.shape == (num_nights, 1200)`

Model input:

- `batch x 1200 x 256`

Model output:

- `batch x 1200 x 4`

## Step 13: Practical Intuition

Why not train on one epoch at a time?

- Sleep stage depends on context.
- A single epoch can look ambiguous.
- Whole-night temporal structure improves prediction.

Why pad to 1200 epochs?

- Makes every night the same tensor shape.
- Lets you batch nights efficiently.
- Matches the paper’s fixed 10-hour sequence design.

Why use masked loss?

- Real nights have different lengths.
- Padding should not affect optimization.

Why use kappa?

- Sleep classes are imbalanced.
- Kappa is standard in sleep staging literature.

## Step 14: Known Gaps Relative To The Paper

This implementation is paper-inspired, but your data setup is still not an
exact replication.

Differences:

- the paper used ECG-derived instantaneous heart rate
- your data uses Fitbit heart rate
- the paper used PSG expert labels
- your labels come from Fitbit staging

So this code is best viewed as:

- architecture and training process close to the paper
- data source and labels approximate the paper

## Step 15: Minimal Sanity Checks

Before training:

```python
print(windowed_dataset.X.shape)
print(windowed_dataset.y.shape)
print(nightly_dataset.X.shape)
print(nightly_dataset.y.shape)
print(nightly_dataset.mask.shape)
print(nightly_dataset.stage_names)
```

After training:

```python
print(fit_output["history"][-1])
print(test_metrics["accuracy"])
print(test_metrics["kappa"])
print(test_metrics["macro_f1"])
print(test_metrics["per_class"])
print(test_metrics["confusion_matrix"])
```
