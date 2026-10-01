"""Tests for Figshare pretrained download helpers (no network)."""

import json
import zipfile
from pathlib import Path

import pytest

from clonetrast.tl import pretrained


def _write_bundle_zip(zip_path: Path, files: tuple[str, ...], folder: str) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in files:
            zf.writestr(f"{folder}/{name}", b"a")
    return zip_path


class _FakeResponse:
    """Minimal context-manager stand-in for requests.Response."""

    def __init__(self, content: bytes, headers: dict | None = None, error: Exception | None = None):
        self._content = content
        self.headers = headers if headers is not None else {"content-length": str(len(content))}
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        yield b""
        if self._error is not None:
            yield b"partial"
            raise self._error
        yield self._content


def test_normalize_figshare_download_url():
    url = (
        "https://figshare.com/ndownloader/articles/32228991"
        "?folder_path=contrastive_model&private_link=01fe17c21afddf090c79"
    )
    out = pretrained.normalize_figshare_download_url(url)
    assert out.startswith("https://ndownloader.figshare.com/articles/32228991")
    assert "folder_path=contrastive_model" in out


def test_normalize_figshare_download_url_unchanged():
    already = "https://ndownloader.figshare.com/articles/32228991?folder_path=x"
    assert pretrained.normalize_figshare_download_url(already) == already
    other = "https://figshare.com/articles/32228991"
    assert pretrained.normalize_figshare_download_url(other) == other


def test_default_pretrained_cache_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLONETRAST_PRETRAINED_CACHE", str(tmp_path / "custom_cache"))
    assert pretrained.default_pretrained_cache_dir() == (tmp_path / "custom_cache").resolve()


def test_default_pretrained_cache_dir_home(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLONETRAST_PRETRAINED_CACHE", raising=False)
    expected = (Path.home() / ".cache" / "clonetrast" / "pretrained_models").resolve()
    assert pretrained.default_pretrained_cache_dir() == expected


def test_find_bundle_root_nested(tmp_path: Path):
    root = tmp_path / "nested" / "contrastive_model"
    root.mkdir(parents=True)
    for name in ("model.pt", "config.json", "gene_order.json"):
        (root / name).write_bytes(b"x")
    found = pretrained._find_bundle_root(tmp_path, frozenset({"model.pt", "config.json", "gene_order.json"}))
    assert found == root


def test_find_bundle_root_skips_incomplete_and_missing(tmp_path: Path):
    incomplete = tmp_path / "partial"
    incomplete.mkdir()
    (incomplete / "model.pt").write_bytes(b"x")
    (incomplete / "notes.txt").write_bytes(b"ignore")
    required = frozenset({"model.pt", "config.json", "gene_order.json"})
    with pytest.raises(FileNotFoundError, match="Archive did not contain"):
        pretrained._find_bundle_root(tmp_path, required)

    complete = tmp_path / "full" / "bundle"
    complete.mkdir(parents=True)
    for name in required:
        (complete / name).write_bytes(b"x")
    assert pretrained._find_bundle_root(tmp_path, required) == complete


def test_install_from_zip_contrastive(tmp_path: Path):
    files = ("model.pt", "config.json", "gene_order.json")
    zip_path = _write_bundle_zip(tmp_path / "bundle.zip", files, "contrastive_model")
    dest = tmp_path / "out" / "contrastive_model"
    pretrained._install_from_zip_archive(zip_path, dest, frozenset(files))
    for name in files:
        assert (dest / name).is_file()


def test_install_from_zip_replaces_existing_dest(tmp_path: Path):
    files = ("model.pt", "config.json", "gene_order.json")
    zip_path = _write_bundle_zip(tmp_path / "bundle.zip", files, "contrastive_model")
    dest = tmp_path / "out" / "contrastive_model"
    dest.mkdir(parents=True)
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    pretrained._install_from_zip_archive(zip_path, dest, frozenset(files))
    assert not (dest / "stale.txt").exists()
    assert (dest / "model.pt").is_file()


def test_install_from_zip_rejects_non_zip(tmp_path: Path):
    not_zip = tmp_path / "bundle.bin"
    not_zip.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a zip archive"):
        pretrained._install_from_zip_archive(
            not_zip,
            tmp_path / "dest",
            frozenset({"model.pt", "config.json", "gene_order.json"}),
        )


def test_ensure_bundle_skips_download_when_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "contrastive_model"
    dest.mkdir(parents=True)
    for name in ("model.pt", "config.json", "gene_order.json"):
        (dest / name).write_bytes(b"x")

    called: list[str] = []

    def fail_download(*_a, **_k):
        called.append("download")
        raise AssertionError("should not download")

    monkeypatch.setattr(pretrained, "_download_figshare", fail_download)
    out = pretrained._ensure_bundle(
        tmp_path,
        "contrastive_model",
        "http://example.invalid",
        frozenset({"model.pt", "config.json", "gene_order.json"}),
        "contrastive_model.zip",
    )
    assert out == dest
    assert called == []


def test_ensure_bundle_uses_cached_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = ("model.pt", "config.json", "gene_order.json")
    archive = tmp_path / "_downloads" / "contrastive_model.zip"
    _write_bundle_zip(archive, files, "contrastive_model")

    def fail_download(*_a, **_k):
        raise AssertionError("should not download when zip is cached")

    monkeypatch.setattr(pretrained, "_download_figshare", fail_download)
    out = pretrained._ensure_bundle(
        tmp_path,
        "contrastive_model",
        "http://example.invalid",
        frozenset(files),
        "contrastive_model.zip",
    )
    assert out == tmp_path / "contrastive_model"
    for name in files:
        assert (out / name).is_file()


def test_ensure_bundle_downloads_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = ("model.pt", "config.json", "gene_order.json")
    src_zip = _write_bundle_zip(tmp_path / "src.zip", files, "contrastive_model")

    def fake_download(_url: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(src_zip.read_bytes())
        return output_path

    monkeypatch.setattr(pretrained, "_download_figshare", fake_download)
    out = pretrained._ensure_bundle(
        tmp_path / "cache",
        "contrastive_model",
        "http://example.invalid",
        frozenset(files),
        "contrastive_model.zip",
    )
    assert (tmp_path / "cache" / "_downloads" / "contrastive_model.zip").is_file()
    for name in files:
        assert (out / name).is_file()


def test_download_figshare_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"zip-bytes"

    def fake_get(url, stream=False, timeout=None, headers=None):
        assert "ndownloader.figshare.com" in url
        assert headers["User-Agent"] == "figshare-downloader-clonetrast"
        return _FakeResponse(payload, headers={})

    monkeypatch.setattr(pretrained.requests, "get", fake_get)
    out = tmp_path / "dl" / "bundle.zip"
    result = pretrained._download_figshare(
        "https://figshare.com/ndownloader/articles/1",
        out,
    )
    assert result == out
    assert out.read_bytes() == payload


def test_download_figshare_cleans_partial_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_get(*_a, **_k):
        return _FakeResponse(b"", error=ConnectionError("network dropped"))

    monkeypatch.setattr(pretrained.requests, "get", fake_get)
    out = tmp_path / "bundle.zip"
    with pytest.raises(ConnectionError, match="network dropped"):
        pretrained._download_figshare("http://example.invalid", out)
    assert not out.exists()
    assert not Path(str(out) + ".part").exists()
    assert not out.with_suffix(".zip.part").exists()


def test_ensure_contrastive_checkpoint_existing_cache(tmp_path: Path):
    dest = tmp_path / "contrastive_model"
    dest.mkdir()
    for name in ("model.pt", "config.json", "gene_order.json"):
        (dest / name).write_bytes(b"x")
    out = pretrained.ensure_contrastive_checkpoint(tmp_path)
    assert out == dest.resolve()


def test_ensure_clone_size_checkpoint_from_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = ("size_model.pt", "config.json", "gene_order.json")
    src_zip = _write_bundle_zip(tmp_path / "src.zip", files, "clone_size_model")

    def fake_download(_url: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(src_zip.read_bytes())
        return output_path

    monkeypatch.setattr(pretrained, "_download_figshare", fake_download)
    out = pretrained.ensure_clone_size_checkpoint(tmp_path / "cache")
    assert out.name == "clone_size_model"
    for name in files:
        assert (out / name).is_file()


def test_ensure_contrastive_checkpoint_uses_default_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "contrastive_model"
    dest.mkdir()
    for name in ("model.pt", "config.json", "gene_order.json"):
        (dest / name).write_bytes(b"x")
    monkeypatch.setattr(pretrained, "default_pretrained_cache_dir", lambda: tmp_path)
    out = pretrained.ensure_contrastive_checkpoint()
    assert out == dest


def test_embed_requires_checkpoint_without_pretrained(tiny_ckpt_dir):
    import numpy as np
    from anndata import AnnData

    from clonetrast.tl.embedding import embed

    genes = json.loads((tiny_ckpt_dir / "gene_order.json").read_text(encoding="utf-8"))
    adata = AnnData(np.zeros((2, len(genes)), dtype=np.int32))
    adata.var_names = genes
    with pytest.raises(ValueError, match="checkpoint_dir is required"):
        embed(adata, None, device="cpu", use_pretrained=False)


def test_predict_clone_size_requires_checkpoint_without_pretrained(tiny_ckpt_dir):
    import numpy as np
    from anndata import AnnData

    from clonetrast.tl.embedding import predict_clone_size

    genes = json.loads((tiny_ckpt_dir / "gene_order.json").read_text(encoding="utf-8"))
    adata = AnnData(np.zeros((2, len(genes)), dtype=np.int32))
    adata.var_names = genes
    with pytest.raises(ValueError, match="checkpoint_dir is required"):
        predict_clone_size(adata, None, device="cpu", use_pretrained=False)
