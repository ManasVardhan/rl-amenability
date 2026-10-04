"""scripts/calibrate_difficulty.py with a fake vLLM engine."""
import json

import pytest

from amenability.eval.calibration import CATEGORIES
from amenability.suites.difficulty import CANDIDATES
from scripts.calibrate_difficulty import ROSTER, calibrate_model, main, parse_args


class FakeEngine:
    """Answers every prompt with 8 completions of prose inside the answer tag."""

    def __init__(self, log):
        self.log = log
        self.released = False

    def generate(self, prompts, sampling):
        assert not self.released
        self.log.append(("generate", len(prompts), sampling["n"], sampling["max_tokens"]))
        return [["<answer>nope</answer>"] * sampling["n"] for _ in prompts]

    def release(self):
        self.released = True
        self.log.append(("release",))


def _factory(log):
    engines = []

    def make(model_path, seed):
        log.append(("load", model_path, seed))
        e = FakeEngine(log)
        engines.append(e)
        return e

    return make, engines


def test_one_engine_per_model_reused_across_candidates_then_released(tmp_path):
    log = []
    make, engines = _factory(log)
    names = ["cd-current", "cd-easy", "gp-current"]
    out = calibrate_model("m1", "hf/m1", names, out_dir=tmp_path, make_engine=make,
                          n_per_bucket=2)
    assert [e[0] for e in log] == ["load", "generate", "generate", "generate", "release"]
    assert all(g[1] == 6 and g[2] == 8 and g[3] == 768 for g in log if g[0] == "generate")
    assert engines[0].released
    data = json.loads(out.read_text())
    assert out == tmp_path / "m1.json"
    assert list(data["candidates"]) == names
    c = data["candidates"]["cd-current"]
    assert c["n_items"] == 6 and c["suite_key"] == "countdown"
    assert c["eligible"] is True
    assert c["pool"]["taxonomy"]["unparseable_answer"] == 1.0
    assert c["pool"]["group_signal_rate"] == 0.0
    assert set(c["by_bucket"]) == {"3", "4", "5"}
    assert set(c["examples"]) == set(CATEGORIES)
    assert data["sampling"]["top_p"] == 1.0 and data["sampling"]["max_tokens"] == 768
    assert data["candidates"]["gp-current"]["guesser"]["rates"] is not None


def test_the_engine_is_released_even_when_a_candidate_fails(tmp_path):
    log = []

    class Boom(FakeEngine):
        def generate(self, prompts, sampling):
            raise RuntimeError("cuda")

    with pytest.raises(RuntimeError):
        calibrate_model("m1", "hf/m1", ["cd-current"], out_dir=tmp_path,
                        make_engine=lambda p, s: Boom(log), n_per_bucket=1)
    assert log[-1] == ("release",)


def test_finished_candidates_are_reused_and_the_engine_skipped(tmp_path):
    log = []
    make, _ = _factory(log)
    calibrate_model("m1", "hf/m1", ["cd-current"], out_dir=tmp_path, make_engine=make, n_per_bucket=1)
    log.clear()
    calibrate_model("m1", "hf/m1", ["cd-current"], out_dir=tmp_path, make_engine=make, n_per_bucket=1)
    assert log == []                       # nothing to do, no engine loaded
    calibrate_model("m1", "hf/m1", ["cd-current", "cd-easy"], out_dir=tmp_path,
                    make_engine=make, n_per_bucket=1)
    assert [e[0] for e in log] == ["load", "generate", "release"]


def test_a_cached_result_from_other_settings_is_not_reused(tmp_path):
    log = []
    make, _ = _factory(log)
    calibrate_model("m1", "hf/m1", ["cd-current"], out_dir=tmp_path, make_engine=make, n_per_bucket=1)
    log.clear()
    calibrate_model("m1", "hf/m1", ["cd-current"], out_dir=tmp_path, make_engine=make, n_per_bucket=2)
    assert [e[0] for e in log] == ["load", "generate", "release"]


def test_roster_is_the_nine_pilot_models():
    assert ROSTER == (
        "falcon3-1b-base", "gemma-3-1b-pt", "olmo-2-1b", "qwen2.5-0.5b", "qwen2.5-1.5b",
        "qwen2.5-math-1.5b", "qwen3-0.6b-base", "smollm2-1.7b", "stablelm-2-1.6b",
    )


def test_every_roster_model_is_in_the_registry():
    from amenability.registry.loader import load_registry

    assert set(ROSTER) <= set(load_registry())


def test_cli_defaults_to_every_candidate_and_the_full_pool():
    args = parse_args(["--model", "qwen2.5-0.5b"])
    assert args.candidates == list(CANDIDATES)
    assert args.n_per_bucket == 100


def test_cli_rejects_unknown_models_and_candidates(capsys):
    with pytest.raises(SystemExit):
        main(["--model", "nope"], make_engine=lambda p, s: pytest.fail("loaded"))
    with pytest.raises(SystemExit):
        parse_args(["--model", "qwen2.5-0.5b", "--candidates", "cd-nope"])


def test_cli_model_index_selects_from_the_roster(tmp_path):
    log = []
    make, _ = _factory(log)
    main(["--model-index", "3", "--candidates", "cd-current", "--n-per-bucket", "1",
          "--out-dir", str(tmp_path)], make_engine=make)
    assert (tmp_path / "qwen2.5-0.5b.json").exists()
    assert log[0] == ("load", "Qwen/Qwen2.5-0.5B", 0)
