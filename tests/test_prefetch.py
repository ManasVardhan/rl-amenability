import json

import pytest

from scripts.prefetch_models import main, prefetch


class _Spec:
    def __init__(self, key, hf_id):
        self.key, self.hf_id = key, hf_id


def test_prefetch_reports_each_model_and_keeps_going_after_a_failure():
    registry = {"ok": _Spec("ok", "org/ok"), "gated": _Spec("gated", "org/gated")}

    def download(repo_id, **kw):
        if repo_id == "org/gated":
            raise RuntimeError("401 Client Error: gated repo")
        return f"/cache/{repo_id}"

    rows = prefetch(registry, download=download, check_config=lambda path: None)
    by_key = {r["key"]: r for r in rows}
    assert by_key["ok"] == {"key": "ok", "hf_id": "org/ok", "ok": True, "path": "/cache/org/ok", "error": None}
    assert by_key["gated"]["ok"] is False and "401" in by_key["gated"]["error"]


def test_prefetch_marks_unloadable_config_as_failure():
    registry = {"m": _Spec("m", "org/m")}

    def check(path):
        raise ValueError("unknown architecture")

    rows = prefetch(registry, download=lambda repo_id, **kw: "/cache/m", check_config=check)
    assert rows[0]["ok"] is False and "unknown architecture" in rows[0]["error"]


def test_cli_exits_nonzero_when_any_model_failed(tmp_path, monkeypatch):
    import scripts.prefetch_models as pm
    monkeypatch.setattr(pm, "load_registry", lambda: {"m": _Spec("m", "org/m")})
    out = tmp_path / "prefetch.json"
    with pytest.raises(SystemExit) as e:
        main(["--out", str(out)], download=lambda repo_id, **kw: (_ for _ in ()).throw(RuntimeError("403")),
             check_config=lambda p: None)
    assert e.value.code == 1
    assert json.loads(out.read_text())[0]["ok"] is False
