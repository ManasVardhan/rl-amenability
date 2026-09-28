# tests/test_run_probe_cli.py
import pytest
from pathlib import Path

from scripts.run_probe import main, parse_args, resolve_model


def test_registry_key_resolves_to_hf_id():
    key, path = resolve_model("qwen2.5-0.5b", {"qwen2.5-0.5b": type("S", (), {"hf_id": "Qwen/Qwen2.5-0.5B"})()})
    assert (key, path) == ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B")


def test_local_path_resolves_to_itself(tmp_path):
    d = tmp_path / "my-ckpt"
    d.mkdir()
    key, path = resolve_model(str(d), {})
    assert key == "my-ckpt" and path == str(d)


def test_unknown_model_fails_before_any_gpu_work(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--model", "nope", "--suite", "countdown"], run_probe_fn=lambda **kw: pytest.fail("ran"))
    assert e.value.code == 2
    assert "qwen2.5-0.5b" in capsys.readouterr().err


def test_unknown_suite_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        parse_args(["--model", "qwen2.5-0.5b", "--suite", "gsm8k"])


def test_arguments_reach_run_probe(tmp_path):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        from amenability.eval.passk import PassKResult
        return PassKResult(ks={1: 0.1, 8: 0.1, 32: 0.2, 64: 0.2}, n_samples=64, n_items=1, per_item_correct={})

    main(["--model", "qwen2.5-0.5b", "--suite", "graphpath", "--run-seed", "1",
          "--work-dir", str(tmp_path), "--pre-only", "--steps", "2", "--lr", "3e-6"], run_probe_fn=fake)
    assert seen["model_key"] == "qwen2.5-0.5b" and seen["model_path"] == "Qwen/Qwen2.5-0.5B"
    assert seen["suite_key"] == "graphpath" and seen["run_seed"] == 1 and seen["item_seed"] == 0
    assert seen["work_dir"] == Path(tmp_path) and seen["pre_only"] is True
    assert seen["config"].probe_steps == 2 and seen["config"].learning_rate == pytest.approx(3e-6)
    assert seen["config"].num_generations == 8 and seen["config"].n_probe_items == 300
