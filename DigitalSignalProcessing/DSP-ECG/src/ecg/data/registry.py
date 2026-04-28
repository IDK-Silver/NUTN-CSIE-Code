"""Registry of supported dataset pipelines."""

from __future__ import annotations

from ecg.data import ds004504_rbp_paper, raw_ds004504
from ecg.data.pipelines import ProcessedDatasetPipeline, RawDatasetPipeline


SUPPORTED_RAW_DATASETS: dict[str, RawDatasetPipeline] = {
    raw_ds004504.OPENNEURO_DATASET_ID: RawDatasetPipeline(
        dataset_id=raw_ds004504.OPENNEURO_DATASET_ID,
        build_download_command=raw_ds004504.build_ds004504_download_command,
    ),
}

SUPPORTED_PROCESSED_DATASETS: dict[str, ProcessedDatasetPipeline] = {
    ds004504_rbp_paper.PROCESSED_DATASET_ID: ProcessedDatasetPipeline(
        dataset_id=ds004504_rbp_paper.PROCESSED_DATASET_ID,
        process_raw_dataset=ds004504_rbp_paper.process_raw_dataset,
    ),
}


class UnsupportedDatasetError(ValueError):
    pass


def supported_raw_dataset_ids() -> tuple[str, ...]:
    return tuple(sorted(SUPPORTED_RAW_DATASETS))


def supported_processed_dataset_ids() -> tuple[str, ...]:
    return tuple(sorted(SUPPORTED_PROCESSED_DATASETS))


def supported_raw_dataset_text() -> str:
    return ", ".join(supported_raw_dataset_ids())


def supported_processed_dataset_text() -> str:
    return ", ".join(supported_processed_dataset_ids())


def get_raw_dataset_pipeline(dataset_id: str) -> RawDatasetPipeline:
    try:
        return SUPPORTED_RAW_DATASETS[dataset_id]
    except KeyError as exc:
        supported = supported_raw_dataset_text()
        raise UnsupportedDatasetError(f"Unsupported dataset: {dataset_id!r}. Supported datasets: {supported}.") from exc


def get_processed_dataset_pipeline(dataset_id: str) -> ProcessedDatasetPipeline:
    try:
        return SUPPORTED_PROCESSED_DATASETS[dataset_id]
    except KeyError as exc:
        supported = supported_processed_dataset_text()
        raise UnsupportedDatasetError(f"Unsupported dataset: {dataset_id!r}. Supported datasets: {supported}.") from exc
