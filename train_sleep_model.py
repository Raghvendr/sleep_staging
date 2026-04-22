from __future__ import annotations

from dataclasses import asdict

from modeling import PaperSleepStageModel, PaperSleepStageModelConfig
from pre_processing.fitbit_preprocessing import prepare_fitbit_training_data_from_directories
from training import (
    PaperTrainingConfig,
    build_paper_night_sequences,
    evaluate_model,
    fit_model,
    split_by_patient_ids,
)


def example_training_run(
    sleep_dir: str,
    heart_dir: str,
    train_patient_ids: list[str],
    val_patient_ids: list[str],
    test_patient_ids: list[str] | None = None,
) -> dict[str, object]:
    windowed_dataset = prepare_fitbit_training_data_from_directories(
        sleep_dir=sleep_dir,
        heart_dir=heart_dir,
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

    split_datasets = split_by_patient_ids(
        night_dataset=nightly_dataset,
        train_patient_ids=train_patient_ids,
        val_patient_ids=val_patient_ids,
        test_patient_ids=test_patient_ids,
    )

    model_config = PaperSleepStageModelConfig()
    training_config = PaperTrainingConfig()
    model = PaperSleepStageModel(config=model_config)

    training_output = fit_model(
        model=model,
        train_dataset=split_datasets["train"],
        val_dataset=split_datasets.get("val"),
        config=training_config,
    )

    result = {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "history": training_output["history"],
    }

    if "val" in split_datasets:
        result["val_metrics"] = evaluate_model(
            model=training_output["model"],
            dataset=split_datasets["val"],
            device=training_config.device,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
        )

    if "test" in split_datasets:
        result["test_metrics"] = evaluate_model(
            model=training_output["model"],
            dataset=split_datasets["test"],
            device=training_config.device,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
        )

    result["model"] = training_output["model"]
    return result
