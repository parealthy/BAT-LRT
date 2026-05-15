#!/usr/bin/env python
import argparse
import os
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("/mnt/pami23/dzhu/models")
DEFAULT_DATASET_ROOT = Path("/mnt/pami23/dzhu/datasets")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_REVISION = "main"

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


def _resolve_url(endpoint: str, repo_id: str, revision: str, filename: str) -> str:
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_filename = urllib.parse.quote(filename, safe="/")
    return f"{endpoint.rstrip('/')}/{repo_id}/resolve/{quoted_revision}/{quoted_filename}"


def _download_url(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"  skip existing file {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.incomplete")
    request = urllib.request.Request(url, headers={"User-Agent": "LRT-offline-assets/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tmp_target.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp_target.replace(target)


def _download_model_from_resolve_urls(
    model_id: str,
    model_root: Path,
    endpoint: str,
    revision: str,
    max_workers: int,
) -> None:
    from huggingface_hub import HfApi

    target = _target_path(model_root, model_id)
    api = HfApi(endpoint=endpoint)
    files = api.list_repo_files(repo_id=model_id, repo_type="model", revision=revision)
    print(f"  fallback: downloading {len(files)} files from {endpoint}/.../resolve/{revision}/...")

    def download_one(filename: str) -> str:
        url = _resolve_url(endpoint, model_id, revision, filename)
        _download_url(url, target / filename)
        return filename

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, filename) for filename in files]
        for future in as_completed(futures):
            filename = future.result()
            print(f"  ok {filename}")


def download_model(
    model_id: str,
    model_root: Path,
    dry_run: bool,
    endpoint: str,
    revision: str,
    max_workers: int,
) -> None:
    target = _target_path(model_root, model_id)
    print(f"model: {model_id} -> {target}")
    if dry_run:
        return
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            revision=revision,
            max_workers=max_workers,
        )
    except Exception as exc:
        if endpoint.rstrip("/") == "https://huggingface.co":
            raise
        print(f"  snapshot_download failed through {endpoint}: {exc}")
        _download_model_from_resolve_urls(
            model_id=model_id,
            model_root=model_root,
            endpoint=endpoint,
            revision=revision,
            max_workers=max_workers,
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
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
        help="Hugging Face endpoint to use for all downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    model_root = args.model_root.expanduser()
    dataset_root = args.dataset_root.expanduser()

    models = [] if args.skip_defaults else list(DEFAULT_MODELS)
    models.extend(args.model)

    datasets = [] if args.skip_defaults else list(DEFAULT_DATASETS)
    datasets.extend(_parse_dataset_spec(value) for value in args.dataset)

    print(f"model root: {model_root}")
    print(f"dataset root: {dataset_root}")
    print(f"hf endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    if args.dry_run:
        print("dry run: no files will be downloaded")

    for model_id in models:
        download_model(
            model_id=model_id,
            model_root=model_root,
            dry_run=args.dry_run,
            endpoint=args.hf_endpoint,
            revision=args.revision,
            max_workers=args.max_workers,
        )

    for dataset_spec in datasets:
        download_dataset(dataset_spec, dataset_root, args.dry_run)


if __name__ == "__main__":
    main()
