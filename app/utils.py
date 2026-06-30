from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_supported_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )