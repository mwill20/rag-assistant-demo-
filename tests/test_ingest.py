import shutil
import subprocess
import sys


def test_ingest_creates_storage(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    storage_dir = tmp_path / "storage"
    data_dir.mkdir()

    (data_dir / "dummy.md").write_text(
        "Ready Tensor teaches RAG. Phase 5 polishes tests.", encoding="utf-8"
    )
    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("CHROMA_DIR", str(storage_dir))

    subprocess.check_call([sys.executable, "-m", "rag_assistant.ingest"])

    assert storage_dir.exists(), "Chroma storage dir not created"
    assert any(storage_dir.iterdir()), "Expected Chroma persistence files"
