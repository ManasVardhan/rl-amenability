"""scripts/analyze_calibration.py on synthetic calibration files."""
import json

from amenability.suites.difficulty import candidates_for
from scripts.analyze_calibration import analyze, load_results, main, render_suite


def _pool(gsr, p1, tag=0.0, unp=0.0, shaped=None):
    correct = p1
    wrong = max(0.0, 1.0 - tag - unp - correct)
    extra = {} if shaped is None else {"shaped_group_signal_rate": shaped}
    return {**extra, "group_signal_rate": gsr, "pass@1": p1, "pass@8": gsr,
            "format_failure_share": tag + unp,
            "taxonomy": {"no_answer_tag": tag, "unparseable_answer": unp,
                         "parseable_wrong": wrong, "correct": correct}}


def _write(tmp_path, model, cands):
    data = {"model_key": model, "candidates": {
        name: {"suite_key": suite, "pool": pool} for name, (suite, pool) in cands.items()}}
    (tmp_path / f"{model}.json").write_text(json.dumps(data))


def _fixture(tmp_path):
    cd = [c.name for c in candidates_for("countdown")]
    gp = [c.name for c in candidates_for("graphpath")]
    # strong: fine everywhere but saturates cd-easier; weak: needs cd-easy.
    _write(tmp_path, "strong", {
        **{n: ("countdown", _pool(0.9, p)) for n, p in zip(cd, (0.5, 0.7, 0.78, 0.9))},
        **{n: ("graphpath", _pool(0.9, 0.6)) for n in gp},
    })
    _write(tmp_path, "weak", {
        **{n: ("countdown", _pool(g, 0.05)) for n, g in zip(cd, (0.02, 0.2, 0.3, 0.5))},
        **{n: ("graphpath", _pool(0.05, 0.01, tag=0.3, unp=0.4)) for n in gp},
    })
    _write(tmp_path, "stablelm-2-1.6b", {
        **{n: ("countdown", _pool(0.0, 0.0, tag=1.0)) for n in cd},
        **{n: ("graphpath", _pool(0.0, 0.0, tag=1.0)) for n in gp},
    })
    (tmp_path / "selection.json").write_text("{}")   # an earlier output, not a model
    return cd, gp


def test_selection_per_suite(tmp_path):
    _fixture(tmp_path)
    results = load_results(tmp_path)
    assert sorted(results) == ["stablelm-2-1.6b", "strong", "weak"]
    cd = analyze(results, "countdown", reward="strict")
    assert cd["selected"] == "cd-easy" and cd["launch"] is True
    assert cd["floored_exempt"] == ["stablelm-2-1.6b"]
    gp = analyze(results, "graphpath", reward="strict")
    assert gp["selected"] is None and gp["launch"] is False
    assert gp["diagnosis_candidate"] == "gp-tree-6s"
    assert gp["diagnosis"] == [{"model": "weak", "signal_rate": 0.05, "correct_signal_rate": 0.05,
                                "format_failure_share": 0.7, "failure_mode": "format"}]
    rows = {r["candidate"]: r for r in gp["candidates"]}
    assert rows["gp-contingency"]["eligible"] is False


def test_rendered_tables_name_every_model_and_candidate(tmp_path):
    cd, _ = _fixture(tmp_path)
    results = load_results(tmp_path)
    text = render_suite(results, "countdown", analyze(results, "countdown", reward="strict"))
    for name in cd + ["strong", "weak", "stablelm-2-1.6b"]:
        assert name in text
    assert "Group-signal rate" in text and "pass@1" in text and "Failure taxonomy" in text
    assert "Selected: cd-easy" in text


def test_main_writes_the_selection_and_report(tmp_path, capsys):
    _fixture(tmp_path)
    main(["--dir", str(tmp_path), "--models", "strong", "weak", "stablelm-2-1.6b",
          "--reward", "strict"])
    sel = json.loads((tmp_path / "selection_t22.json").read_text())
    assert sel["countdown"]["selected"] == "cd-easy"
    assert sel["graphpath"]["launch"] is False
    assert sel["batch_may_launch"] is False
    report = (tmp_path / "report_t22.md").read_text()
    assert "does NOT launch" in report
    assert "format" in capsys.readouterr().out


def test_a_roster_model_without_any_result_blocks_every_candidate(tmp_path):
    _fixture(tmp_path)
    results = load_results(tmp_path)
    cd = analyze(results, "countdown", expected_models=("strong", "weak", "absent"),
                 reward="strict")
    assert cd["selected"] is None
    assert all(r["missing_models"] == ["absent"] for r in cd["candidates"])
    assert "absent" in render_suite(results, "countdown", cd)


def test_main_requires_the_full_roster_by_default(tmp_path):
    _fixture(tmp_path)
    sel = main(["--dir", str(tmp_path), "--reward", "strict"])
    assert sel["countdown"]["selected"] is None
    assert "qwen2.5-0.5b" in sel["countdown"]["candidates"][0]["missing_models"]



# ---- transfer-rulings T23: the shaped-reward rule (the default) ----

def _shaped_fixture(tmp_path):
    cd = [c.name for c in candidates_for("countdown")]
    gp = [c.name for c in candidates_for("graphpath")]
    # weak has almost no strict signal anywhere but shaped signal from cd-current;
    # on graph its shaped signal is enough only from gp-tree-6.
    _write(tmp_path, "strong", {
        **{n: ("countdown", _pool(0.9, p, shaped=0.9)) for n, p in zip(cd, (0.5, 0.7, 0.78, 0.9))},
        **{n: ("graphpath", _pool(0.9, 0.6, shaped=0.9)) for n in gp},
    })
    _write(tmp_path, "weak", {
        **{n: ("countdown", _pool(0.01, 0.0, tag=0.6, unp=0.3, shaped=0.4)) for n in cd},
        **{n: ("graphpath", _pool(0.0, 0.0, tag=0.7, shaped=s))
           for n, s in zip(gp, (0.05, 0.05, 0.1, 0.2, 0.3))},
    })
    _write(tmp_path, "stablelm-2-1.6b", {
        **{n: ("countdown", _pool(0.0, 0.0, tag=1.0, shaped=0.0)) for n in cd},
        **{n: ("graphpath", _pool(0.0, 0.0, tag=1.0, shaped=0.0)) for n in gp},
    })


def test_shaped_selection_is_the_default(tmp_path):
    _shaped_fixture(tmp_path)
    sel = main(["--dir", str(tmp_path), "--models", "strong", "weak", "stablelm-2-1.6b"])
    assert sel["reward"] == "shaped"
    assert sel["countdown"]["selected"] == "cd-current"
    assert sel["countdown"]["signal_key"] == "shaped_group_signal_rate"
    assert sel["graphpath"]["selected"] == "gp-tree-6"
    assert sel["batch_may_launch"] is True
    assert json.loads((tmp_path / "selection.json").read_text())["reward"] == "shaped"
    report = (tmp_path / "report.md").read_text()
    assert "T23" in report
    assert "Shaped group-signal rate" in report
    assert "Correct-signal rate" in report and "reported, not gated" in report


def test_shaped_mode_names_results_that_need_a_calibration_rerun(tmp_path):
    _fixture(tmp_path)            # pre-T23 files: no shaped signal anywhere
    results = load_results(tmp_path)
    cd = analyze(results, "countdown", reward="shaped")
    assert cd["selected"] is None
    assert cd["needs_rerun"] == ["stablelm-2-1.6b", "strong", "weak"]
    assert cd["reason"].startswith("strong, weak have calibration results without")
    assert cd["remedies"] == []
    text = render_suite(results, "countdown", cd)
    assert "rerun the calibration" in text
