"""Download and cache pretrained CloneTrast checkpoints from Figshare."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from tqdm import tqdm

FIGSHARE_CONTRASTIVE_MODEL_URL = (
    "https://figshare.com/ndownloader/articles/32228991/versions/1"
    "?folder_path=contrastive_model"
)
FIGSHARE_CLONE_SIZE_MODEL_URL = (
    "https://figshare.com/ndownloader/articles/32228991/versions/1"
    "?folder_path=clone_size_model"
)


def default_pretrained_cache_dir() -> Path:
    """Default directory for cached pretrained weights (~/.cache/clonetrast/pretrained_models)."""
    env = os.environ.get("CLONETRAST_PRETRAINED_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "clonetrast" / "pretrained_models").resolve()


def normalize_figshare_download_url(url: str) -> str:
    """Rewrite figshare.com/ndownloader URLs to the ndownloader.figshare.com host."""
    parsed = urlparse(url)
    if parsed.netloc == "figshare.com" and parsed.path.startswith("/ndownloader/"):
        return urlunparse(
            parsed._replace(
                netloc="ndownloader.figshare.com",
                path=parsed.path.removeprefix("/ndownloader"),
            )
        )
    return url


def _download_figshare(url: str, output_path: Path) -> Path:
    url = normalize_figshare_download_url(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "figshare-downloader-clonetrast"}

    with requests.get(url, stream=True, timeout=(10, 120), headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        tmp = output_path.with_suffix(output_path.suffix + ".part")
        try:
            with open(tmp, "wb") as f, tqdm(
                total=total_size or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(output_path)
        except BaseException:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
    return output_path


def _find_bundle_root(extracted: Path, required_filenames: frozenset[str]) -> Path:
    """Return the directory that directly contains all required files."""
    for path in extracted.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in required_filenames:
            continue
        parent = path.parent
        if all((parent / name).is_file() for name in required_filenames):
            return parent
    raise FileNotFoundError(
        f"Archive did not contain a folder with all of: {', '.join(sorted(required_filenames))}"
    )


def _install_from_zip_archive(archive_path: Path, dest_dir: Path, required_filenames: frozenset[str]) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Downloaded file is not a zip archive: {archive_path}")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(td_path)
        root = _find_bundle_root(td_path, required_filenames)
        shutil.copytree(root, dest_dir)


def _ensure_bundle(
    cache_root: Path,
    subdir_name: str,
    url: str,
    required_filenames: frozenset[str],
    archive_filename: str,
) -> Path:
    dest_dir = cache_root / subdir_name
    if all((dest_dir / name).is_file() for name in required_filenames):
        return dest_dir

    dl_dir = cache_root / "_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dl_dir / archive_filename
    if not archive_path.is_file():
        print(f"[CloneTrast] Downloading pretrained bundle to {archive_path} ...")
        _download_figshare(url, archive_path)
    else:
        print(f"[CloneTrast] Using cached archive {archive_path}")

    print(f"[CloneTrast] Extracting pretrained bundle → {dest_dir} ...")
    _install_from_zip_archive(archive_path, dest_dir, required_filenames)
    return dest_dir


def ensure_contrastive_checkpoint(pretrained_cache_dir: str | Path | None = None) -> Path:
    """Ensure contrastive model.pt, config.json, and gene_order.json exist; download if missing."""
    root = Path(pretrained_cache_dir).expanduser().resolve() if pretrained_cache_dir else default_pretrained_cache_dir()
    return _ensure_bundle(
        root,
        "contrastive_model",
        FIGSHARE_CONTRASTIVE_MODEL_URL,
        frozenset({"model.pt", "config.json", "gene_order.json"}),
        "contrastive_model.zip",
    )


def ensure_clone_size_checkpoint(pretrained_cache_dir: str | Path | None = None) -> Path:
    """Ensure size_model.pt, config.json, and gene_order.json exist; download if missing."""
    root = Path(pretrained_cache_dir).expanduser().resolve() if pretrained_cache_dir else default_pretrained_cache_dir()
    return _ensure_bundle(
        root,
        "clone_size_model",
        FIGSHARE_CLONE_SIZE_MODEL_URL,
        frozenset({"size_model.pt", "config.json", "gene_order.json"}),
        "clone_size_model.zip",
    )
