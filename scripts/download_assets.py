#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("/mnt/pami23/dzhu/models")
DEFAULT_DATASET_ROOT = Path("/mnt/pami23/dzhu/datasets")

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-Embedding-0.6B",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    config: str | None = None


DEFAULT_DATASETS = [
    DatasetSpec("open-r1/OpenR1-Math-220k", "default"),
    DatasetSpec("BytedTsinghua-SIA/DAPO-Math-17k"),
    DatasetSpec("agentica-org/DeepScaler-Preview-Dataset"),
    DatasetSpec("stepfun-ai/Step-3.5-Flash-SFT"),
    DatasetSpec("HuggingFaceH4/MATH-500"),
    DatasetSpec("openai/gsm8k", "main"),
]


def _target_path(root: Path, asset_id: str) -> Path:
    return root.joinpath(*asset_id.split("/"))


def _parse_dataset_spec(value: str) -> DatasetSpec:
    if "::" in value:
        dataset_id, config = value.split("::", 1)
        return DatasetSpec(dataset_id, config or None)
    return DatasetSpec(value)


def download_model(model_id: str, model_root: Path, dry_run: bool) -> None:
    target = _target_path(model_root, model_id)
    print(f"model: {model_id} -> {target}")
    if dry_run:
        return
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        resume_download=True,
    )


def download_dataset(spec: DatasetSpec, dataset_root: Path, dry_run: bool) -> None:
    target = _target_path(dataset_root, spec.dataset_id)
    config_text = f" config={spec.config}" if spec.config else ""
    print(f"dataset: {spec.dataset_id}{config_text} -> {target}")
    if dry_run:
        return
    if target.exists():
        print(f"  skip existing dataset at {target}")
        return
    from datasets import load_dataset

    target.parent.mkdir(parents=True, exist_ok=True)
    if spec.config:
        dataset = load_dataset(spec.dataset_id, spec.config)
    else:
        dataset = load_dataset(spec.dataset_id)
    dataset.save_to_disk(str(target))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LRT models and datasets to shared offline asset roots."
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model", action="append", default=[], help="Extra HF model ID to download.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Extra HF dataset ID to download. Use DATASET_ID::CONFIG for configured datasets.",
    )
    parser.add_argument("--skip-defaults", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.model_root.expanduser()
    dataset_root = args.dataset_root.expanduser()

    models = [] if args.skip_defaults else list(DEFAULT_MODELS)
    models.extend(args.model)

    datasets = [] if args.skip_defaults else list(DEFAULT_DATASETS)
    datasets.extend(_parse_dataset_spec(value) for value in args.dataset)

    print(f"model root: {model_root}")
    print(f"dataset root: {dataset_root}")
    if args.dry_run:
        print("dry run: no files will be downloaded")

    for model_id in models:
        download_model(model_id, model_root, args.dry_run)

    for dataset_spec in datasets:
        download_dataset(dataset_spec, dataset_root, args.dry_run)


if __name__ == "__main__":
    main()
