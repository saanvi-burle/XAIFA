import csv
import json
from pathlib import Path


def parse_labels(labels_path: str | None) -> list[str] | None:
    if not labels_path:
        return None

    path = Path(labels_path)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [str(data[key]) for key in sorted(data, key=lambda item: int(item))]
        if isinstance(data, list):
            return [str(item) for item in data]
        return None

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        if {"index", "label"}.issubset(rows[0]):
            rows.sort(key=lambda row: int(row["index"]))
            return [row["label"] for row in rows]
        if "label" in rows[0]:
            return [row["label"] for row in rows]
        return None

    labels: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if ":" in text:
            index, label = text.split(":", 1)
            labels.append((int(index.strip()), label.strip()))
        else:
            labels.append((len(labels), text))

    labels.sort(key=lambda item: item[0])
    return [label for _, label in labels]
