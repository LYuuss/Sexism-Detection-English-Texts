import csv
import json
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
REQUIRED_COLUMNS = {"text", "label_sexist"}


def load_dataset_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_DATASETS)

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_DATASETS)

    datasets = data.get("datasets", {})
    merged = dict(DEFAULT_DATASETS)
    for split in DEFAULT_DATASETS:
        value = datasets.get(split)
        if isinstance(value, str) and value.strip():
            merged[split] = value

    return merged


def save_dataset_config(config: dict[str, str]) -> None:
    payload = {"datasets": config}
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


def assign_dataset(split: str, path: str) -> Path:
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
    source = get_dataset_path(split)

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
    return target


def append_dataset_row(split: str, text: str, label: str) -> Path:
    if split not in DEFAULT_DATASETS:
        raise ValueError(f"Unknown split: {split}")

    clean_label = label.strip().lower()
    if clean_label not in {"sexist", "not sexist"}:
        raise ValueError("Label must be 'sexist' or 'not sexist'.")

    dataset_path = get_dataset_path(split)
    validate_dataset_file(dataset_path)

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
    if "split" in row:
        row["split"] = split

    with dataset_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)

    return dataset_path


def _to_repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
