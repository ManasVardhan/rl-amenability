import pytest
from amenability.suites.catalog import PROBE_SUITES, ProbeSuite, get_probe_suite


def test_catalog_has_both_probe_suites():
    assert sorted(PROBE_SUITES) == ["countdown", "graphpath"]
    assert all(isinstance(s, ProbeSuite) for s in PROBE_SUITES.values())


def test_suite_names_match_the_items_they_generate():
    for key, suite in PROBE_SUITES.items():
        items = suite.generate(6, 0)
        assert suite.key == key
        assert all(it.suite == suite.name for it in items)
        assert all(it.task_id.startswith(f"probe/{key}/") for it in items)


def test_unknown_key_names_the_valid_ones():
    with pytest.raises(KeyError, match="countdown, graphpath"):
        get_probe_suite("gsm8k")
