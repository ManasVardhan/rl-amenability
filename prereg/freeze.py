from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FROZEN_PATHS = [
    "src/amenability/scoring/score.py",
    "src/amenability/scoring/baselines.py",
    "src/amenability/scoring/gates.py",
    "src/amenability/registry/models.yaml",
    "prereg/stage0.md",
]

MANIFEST = "prereg/FROZEN.json"


class FreezeViolation(Exception):
    """Raised when frozen analysis code has changed, or was never frozen."""


def _git_head(root: Path) -> str | None:
    """Current HEAD SHA, or None if git is unavailable or this is not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def compute_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in FROZEN_PATHS:
        p = root / rel
        if not p.exists():
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def freeze(root: Path) -> dict:
    manifest_path = root / MANIFEST
    if manifest_path.exists():
        raise FreezeViolation(f"already frozen: {MANIFEST} exists")
    hashes = compute_hashes(root)
    missing = set(FROZEN_PATHS) - set(hashes)
    if missing:
        raise FreezeViolation(f"cannot freeze, missing files: {sorted(missing)}")
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(root),
        "hashes": hashes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def verify_freeze(root: Path) -> None:
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        raise FreezeViolation(
            f"analysis is not frozen: {MANIFEST} missing. Run `python prereg/freeze.py` "
            "before any ground-truth run."
        )
    expected = json.loads(manifest_path.read_text())["hashes"]
    # NOTE: git_commit is provenance only, recorded once at freeze time. It is
    # deliberately NOT checked here against the current HEAD: legitimate later
    # commits (docs, unrelated code, this very fix) move HEAD forward, and
    # requiring the working tree to still sit on the frozen commit would break
    # every subsequent run. Only the file hashes are enforced.
    actual = compute_hashes(root)
    for rel, digest in expected.items():
        if rel not in actual:
            raise FreezeViolation(f"frozen file deleted: {rel}")
        if actual[rel] != digest:
            raise FreezeViolation(
                f"frozen file modified after freeze: {rel}. Stage 0 is invalidated."
            )


if __name__ == "__main__":
    print(json.dumps(freeze(Path(__file__).resolve().parents[1]), indent=2))
