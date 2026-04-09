import csv
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CONFIG_PATH = REPO_ROOT / ".sexism_cli_config.json"
DEFAULT_DATASETS = {
    "train": "dataset/train.csv",
    "test": "dataset/test.csv",
    "dev": "dataset/dev.csv",
}
DEFAULT_OPTIONS = {
    "delete_duplicated_datasets_on_exit": True,
    "debug": False,
}
REQUIRED_COLUMNS = {"text", "label_sexist"}


def load_dataset_config() -> dict[str, str]:
    data = _load_config_payload()
    datasets = data.get("datasets", {})
    merged = dict(DEFAULT_DATASETS)
    for split in DEFAULT_DATASETS:
        value = datasets.get(split)
        if isinstance(value, str) and value.strip():
            merged[split] = value

    return merged


def load_app_options() -> dict[str, bool]:
    data = _load_config_payload()
    saved_options = data.get("options", {})
    merged = dict(DEFAULT_OPTIONS)
    for key in DEFAULT_OPTIONS:
        value = saved_options.get(key)
        if isinstance(value, bool):
            merged[key] = value

    return merged


def load_selected_methods() -> list[str]:
    data = _load_config_payload()
    selected_methods = data.get("selected_methods", [])
    if not isinstance(selected_methods, list):
        return []

    return [item for item in selected_methods if isinstance(item, str) and item.strip()]


def save_dataset_config(config: dict[str, str]) -> None:
    payload = {
        "datasets": dict(DEFAULT_DATASETS) | config,
        "options": load_app_options(),
        "selected_methods": load_selected_methods(),
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_app_options(options: dict[str, bool]) -> None:
    payload = {
        "datasets": load_dataset_config(),
        "options": dict(DEFAULT_OPTIONS) | options,
        "selected_methods": load_selected_methods(),
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_selected_methods(selected_methods: list[str]) -> None:
    payload = {
        "datasets": load_dataset_config(),
        "options": load_app_options(),
        "selected_methods": selected_methods,
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_dataset_path(split: str, config: dict[str, str] | None = None) -> Path:
    datasets = config or load_dataset_config()
    if split not in datasets:
        raise ValueError(f"Unknown split: {split}")

    candidate = Path(datasets[split]).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate

    return candidate.resolve()


def get_active_dataset_paths(config: dict[str, str] | None = None) -> dict[str, Path]:
    datasets = config or load_dataset_config()
    return {split: get_dataset_path(split, datasets) for split in DEFAULT_DATASETS}


def list_local_dataset_candidates() -> list[Path]:
    if not DATASET_DIR.exists():
        return []

    return sorted(DATASET_DIR.glob("*.csv"))


def validate_dataset_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved

    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])

    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return resolved


def assign_dataset(split: str, path: str | Path) -> Path:
    if split not in DEFAULT_DATASETS:
        raise ValueError(f"Unknown split: {split}")

    validated = validate_dataset_file(path)
    config = load_dataset_config()
    config[split] = _to_repo_relative_or_absolute(validated)
    save_dataset_config(config)
    return validated


def dataset_summary(path: str | Path) -> dict[str, int | str]:
    validated = validate_dataset_file(path)
    total = 0
    sexist = 0
    not_sexist = 0

    with validated.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            label = (row.get("label_sexist") or "").strip().lower()
            if label == "sexist":
                sexist += 1
            elif label == "not sexist":
                not_sexist += 1

    return {
        "path": str(validated),
        "rows": total,
        "sexist": sexist,
        "not_sexist": not_sexist,
    }


def duplicate_active_dataset(split: str, new_file_name: str) -> Path:
    return duplicate_dataset_file(get_dataset_path(split), new_file_name)


def duplicate_dataset_file(source_path: str | Path, new_file_name: str) -> Path:
    source = validate_dataset_file(source_path)

    file_name = new_file_name.strip()
    if not file_name:
        raise ValueError("File name must not be empty.")

    if not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"

    target = DATASET_DIR / file_name
    if target.exists():
        raise FileExistsError(f"Target dataset already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.resolve()


def append_dataset_row(split: str, text: str, label: str) -> Path:
    dataset_path = get_dataset_path(split)
    return append_dataset_row_to_path(dataset_path, text, label, split=split)


def append_dataset_row_to_path(
    path: str | Path,
    text: str,
    label: str,
    split: str | None = None,
) -> Path:
    clean_label = label.strip().lower()
    if clean_label not in {"sexist", "not sexist"}:
        raise ValueError("Label must be 'sexist' or 'not sexist'.")

    dataset_path = validate_dataset_file(path)

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

    row = {field: "" for field in fieldnames}
    if "rewire_id" in row:
        row["rewire_id"] = f"custom-{uuid4().hex[:10]}"
    if "text" in row:
        row["text"] = text
    if "label_sexist" in row:
        row["label_sexist"] = clean_label
    if "label_category" in row:
        row["label_category"] = "custom" if clean_label == "sexist" else "none"
    if "label_vector" in row:
        row["label_vector"] = "custom" if clean_label == "sexist" else "none"
    if "split" in row and split is not None:
        row["split"] = split

    with dataset_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)

    return dataset_path


def remove_dataset_file(path: str | Path) -> None:
    resolved = _resolve_path(path)
    if resolved.exists():
        os.remove(resolved)


def _load_config_payload() -> dict:
    if not CONFIG_PATH.exists():
        return {}

    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _to_repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
