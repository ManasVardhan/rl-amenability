"""Download every roster model once and confirm transformers can load its config.

Run on a CPU node before the GPU array so thirty array tasks do not each download
the same weights, and so a gated or missing repository (#20) fails here, in one
place, rather than as ten separate array-task failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from amenability.registry.loader import load_registry


def _default_download(repo_id: str, **kw) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken"],
        **kw,
    )


def _default_check_config(path: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(path)
    AutoTokenizer.from_pretrained(path)


def prefetch(registry: dict, download=_default_download, check_config=_default_check_config) -> list[dict]:
    rows: list[dict] = []
    for key in sorted(registry):
        hf_id = registry[key].hf_id
        row = {"key": key, "hf_id": hf_id, "ok": False, "path": None, "error": None}
        try:
            row["path"] = download(hf_id)
            check_config(row["path"])
            row["ok"] = True
        except Exception as e:  # noqa: BLE001 - every failure must be reported, none may abort the sweep
            row["error"] = f"{type(e).__name__}: {e}"
        status = "ok" if row["ok"] else f"FAILED: {row['error']}"
        print(f"{key:20s} {hf_id:40s} {status}")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None, download=_default_download, check_config=_default_check_config) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/transfer/prefetch.json"))
    args = parser.parse_args(argv)
    rows = prefetch(load_registry(), download=download, check_config=check_config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    failed = [r for r in rows if not r["ok"]]
    if failed:
        print(
            f"\n{len(failed)} model(s) failed. A 401/403 means the licence has not been "
            "accepted on huggingface.co for the account behind this token, or the token "
            "is not logged in (uv run hf auth login).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\nall {len(rows)} models present")


if __name__ == "__main__":
    main()
