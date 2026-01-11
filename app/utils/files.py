from pathlib import Path
from typing import List


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".wav", ".mov"}


def collect_input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))

    return sorted(files)
