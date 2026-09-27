import json
import pytest
from pathlib import Path
from prereg.freeze import freeze, verify_freeze, compute_hashes, FreezeViolation, FROZEN_PATHS


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    for rel in FROZEN_PATHS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {rel}\n")
    return tmp_path


def test_frozen_paths_cover_score_baselines_and_plan():
    assert "src/amenability/scoring/score.py" in FROZEN_PATHS
    assert "src/amenability/scoring/baselines.py" in FROZEN_PATHS
    assert "prereg/stage0.md" in FROZEN_PATHS
    assert "src/amenability/registry/models.yaml" in FROZEN_PATHS


def test_freeze_writes_manifest_with_all_paths(fake_root):
    manifest = freeze(fake_root)
    assert set(manifest["hashes"]) == set(FROZEN_PATHS)
    assert (fake_root / "prereg" / "FROZEN.json").exists()
    assert "frozen_at" in manifest


def test_verify_passes_immediately_after_freeze(fake_root):
    freeze(fake_root)
    verify_freeze(fake_root)  # must not raise


def test_verify_detects_a_modified_file(fake_root):
    freeze(fake_root)
    (fake_root / "src/amenability/scoring/score.py").write_text("tuned!\n")
    with pytest.raises(FreezeViolation, match="score.py"):
        verify_freeze(fake_root)


def test_verify_detects_a_deleted_file(fake_root):
    freeze(fake_root)
    (fake_root / "src/amenability/scoring/baselines.py").unlink()
    with pytest.raises(FreezeViolation, match="baselines.py"):
        verify_freeze(fake_root)


def test_verify_without_a_manifest_raises(fake_root):
    with pytest.raises(FreezeViolation, match="not frozen"):
        verify_freeze(fake_root)


def test_refreezing_over_an_existing_manifest_raises(fake_root):
    freeze(fake_root)
    with pytest.raises(FreezeViolation, match="already frozen"):
        freeze(fake_root)


def test_hashes_are_content_addressed_not_path_addressed(fake_root):
    h1 = compute_hashes(fake_root)
    (fake_root / "prereg" / "stage0.md").write_text("content of src/amenability/scoring/score.py\n")
    h2 = compute_hashes(fake_root)
    assert h2["prereg/stage0.md"] == h1["src/amenability/scoring/score.py"]
