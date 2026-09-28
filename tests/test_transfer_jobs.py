from pathlib import Path

from scripts.make_transfer_jobs import ARMS, pilot_jobs, transfer_jobs, write_jobs

KEYS = ["a", "b", "c"]


def test_transfer_jobs_cover_three_arms_for_every_model():
    jobs = transfer_jobs(KEYS)
    assert len(jobs) == 9
    assert ARMS == [("countdown", 0), ("graphpath", 0), ("countdown", 1)]
    assert jobs[:3] == [("a", "countdown", 0), ("b", "countdown", 0), ("c", "countdown", 0)]
    assert set(jobs) == {(m, s, seed) for m in KEYS for s, seed in ARMS}


def test_pilot_jobs_are_seed_zero_for_both_suites():
    jobs = pilot_jobs(KEYS)
    assert len(jobs) == 6
    assert all(seed == 0 for _, _, seed in jobs)
    assert {s for _, s, _ in jobs} == {"countdown", "graphpath"}


def test_write_jobs_one_line_each(tmp_path):
    p = tmp_path / "jobs.txt"
    write_jobs([("a", "countdown", 0), ("b", "graphpath", 1)], p)
    assert p.read_text() == "a countdown 0\nb graphpath 1\n"


def test_cli_writes_thirty_jobs_for_the_real_roster(tmp_path):
    from scripts.make_transfer_jobs import main
    out = tmp_path / "jobs.txt"
    main(["--out", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 30
    assert len({line.split()[0] for line in lines}) == 10


def test_cli_only_model_filters(tmp_path):
    from scripts.make_transfer_jobs import main
    out = tmp_path / "jobs.txt"
    main(["--out", str(out), "--only-model", "qwen2.5-0.5b", "smollm2-1.7b"])
    lines = out.read_text().splitlines()
    assert len(lines) == 6
