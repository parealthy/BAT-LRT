import os
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("/mnt/pami23/dzhu/models")
DEFAULT_DATA_ROOT = Path("/mnt/pami23/dzhu/datasets")

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def is_offline_mode() -> bool:
    """Return whether LRT/HF loaders should avoid network access."""
    lrt_offline = _env_flag("LRT_OFFLINE")
    if lrt_offline is not None:
        return lrt_offline

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        value = _env_flag(name)
        if value:
            return True
    return False


def get_model_root() -> Path:
    return Path(os.environ.get("LRT_MODEL_ROOT", str(DEFAULT_MODEL_ROOT))).expanduser()


def get_dataset_root() -> Path:
    return Path(os.environ.get("LRT_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()


def _looks_like_explicit_path(value: str) -> bool:
    expanded = Path(value).expanduser()
    if expanded.exists():
        return True
    return value.startswith(("/", "./", "../", "~"))


def _local_asset_path(root: Path, asset_id: str) -> Path:
    return root.joinpath(*asset_id.split("/"))


def resolve_model_path(model_name_or_path: str | os.PathLike) -> str:
    """Resolve a HF model ID to the local shared model mirror when available."""
    value = str(model_name_or_path)
    if _looks_like_explicit_path(value):
        path = Path(value).expanduser()
        if is_offline_mode() and not path.exists():
            raise FileNotFoundError(f"Offline mode is enabled, but local model path does not exist: {path}")
        return str(path)

    local_path = _local_asset_path(get_model_root(), value)
    if local_path.exists():
        return str(local_path)

    if is_offline_mode():
        raise FileNotFoundError(
            "Offline mode is enabled, but model was not found locally: "
            f"{local_path}. Download it on pami144 first or set LRT_MODEL_ROOT."
        )
    return value


def resolve_dataset_path(dataset_name_or_path: str | os.PathLike) -> Path | None:
    """Return a local dataset path for an explicit path or mirrored HF dataset ID."""
    value = str(dataset_name_or_path)
    if _looks_like_explicit_path(value):
        path = Path(value).expanduser()
        if path.exists():
            return path
        if is_offline_mode():
            raise FileNotFoundError(f"Offline mode is enabled, but local dataset path does not exist: {path}")
        return None

    local_path = _local_asset_path(get_dataset_root(), value)
    if local_path.exists():
        return local_path

    if is_offline_mode():
        raise FileNotFoundError(
            "Offline mode is enabled, but dataset was not found locally: "
            f"{local_path}. Download it on pami144 first or set LRT_DATA_ROOT."
        )
    return None
