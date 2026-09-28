import json

import pytest

from scripts.prefetch_models import _check_weights, _default_check_config, main, prefetch


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


def test_check_weights_requires_a_safetensors_file(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match=r"no \*\.safetensors"):
        _check_weights(str(tmp_path))


def test_check_weights_requires_every_indexed_shard(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"w")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="model-00002-of-00002.safetensors"):
        _check_weights(str(tmp_path))


def test_check_weights_accepts_a_complete_snapshot(tmp_path):
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (tmp_path / name).write_bytes(b"w")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    _check_weights(str(tmp_path))
    single = tmp_path / "single"
    single.mkdir()
    (single / "model.safetensors").write_bytes(b"w")
    _check_weights(str(single))


def _stub_transformers(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda path: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: None)


def test_prefetch_row_fails_when_the_snapshot_has_no_weights(tmp_path, monkeypatch):
    _stub_transformers(monkeypatch)
    rows = prefetch({"m": _Spec("m", "org/m")}, download=lambda repo_id, **kw: str(tmp_path),
                    check_config=_default_check_config)
    assert rows[0]["ok"] is False and "no *.safetensors" in rows[0]["error"]


def test_prefetch_row_fails_when_an_indexed_shard_is_missing(tmp_path, monkeypatch):
    _stub_transformers(monkeypatch)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"w")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    rows = prefetch({"m": _Spec("m", "org/m")}, download=lambda repo_id, **kw: str(tmp_path),
                    check_config=_default_check_config)
    assert rows[0]["ok"] is False and "model-00002-of-00002.safetensors" in rows[0]["error"]


def test_prefetch_row_ok_for_a_complete_snapshot(tmp_path, monkeypatch):
    _stub_transformers(monkeypatch)
    (tmp_path / "model.safetensors").write_bytes(b"w")
    rows = prefetch({"m": _Spec("m", "org/m")}, download=lambda repo_id, **kw: str(tmp_path),
                    check_config=_default_check_config)
    assert rows[0]["ok"] is True
