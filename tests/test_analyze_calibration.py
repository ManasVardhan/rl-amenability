"""scripts/analyze_calibration.py on synthetic calibration files."""
import json

from amenability.suites.difficulty import candidates_for
from scripts.analyze_calibration import analyze, load_results, main, render_suite


def _pool(gsr, p1, tag=0.0, unp=0.0):
    correct = p1
    wrong = max(0.0, 1.0 - tag - unp - correct)
    return {"group_signal_rate": gsr, "pass@1": p1, "pass@8": gsr,
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
    cd = analyze(results, "countdown")
    assert cd["selected"] == "cd-easy" and cd["launch"] is True
    assert cd["floored_exempt"] == ["stablelm-2-1.6b"]
    gp = analyze(results, "graphpath")
    assert gp["selected"] is None and gp["launch"] is False
    assert gp["diagnosis_candidate"] == "gp-tree-6s"
    assert gp["diagnosis"] == [{"model": "weak", "group_signal_rate": 0.05,
                                "format_failure_share": 0.7, "failure_mode": "format"}]
    rows = {r["candidate"]: r for r in gp["candidates"]}
    assert rows["gp-contingency"]["eligible"] is False


def test_rendered_tables_name_every_model_and_candidate(tmp_path):
    cd, _ = _fixture(tmp_path)
    results = load_results(tmp_path)
    text = render_suite(results, "countdown", analyze(results, "countdown"))
    for name in cd + ["strong", "weak", "stablelm-2-1.6b"]:
        assert name in text
    assert "Group-signal rate" in text and "pass@1" in text and "Failure taxonomy" in text
    assert "Selected: cd-easy" in text


def test_main_writes_the_selection_and_report(tmp_path, capsys):
    _fixture(tmp_path)
    main(["--dir", str(tmp_path), "--models", "strong", "weak", "stablelm-2-1.6b"])
    sel = json.loads((tmp_path / "selection.json").read_text())
    assert sel["countdown"]["selected"] == "cd-easy"
    assert sel["graphpath"]["launch"] is False
    assert sel["batch_may_launch"] is False
    report = (tmp_path / "report.md").read_text()
    assert "does NOT launch" in report
    assert "format" in capsys.readouterr().out


def test_a_roster_model_without_any_result_blocks_every_candidate(tmp_path):
    _fixture(tmp_path)
    results = load_results(tmp_path)
    cd = analyze(results, "countdown", expected_models=("strong", "weak", "absent"))
    assert cd["selected"] is None
    assert all(r["missing_models"] == ["absent"] for r in cd["candidates"])
    assert "absent" in render_suite(results, "countdown", cd)


def test_main_requires_the_full_roster_by_default(tmp_path):
    _fixture(tmp_path)
    sel = main(["--dir", str(tmp_path)])
    assert sel["countdown"]["selected"] is None
    assert "qwen2.5-0.5b" in sel["countdown"]["candidates"][0]["missing_models"]
