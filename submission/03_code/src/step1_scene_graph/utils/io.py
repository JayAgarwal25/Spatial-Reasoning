import json
import os


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(data, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
