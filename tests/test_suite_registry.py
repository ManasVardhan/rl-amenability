import pytest
from amenability.suites.base import TaskItem, SuiteRegistry, SuiteOverlapError


def item(tid: str, suite: str = "s") -> TaskItem:
    return TaskItem(task_id=tid, suite=suite, prompt="p", answer="a", difficulty=1)


def test_register_and_get():
    reg = SuiteRegistry()
    reg.register("probe", [item("probe/1"), item("probe/2")])
    assert len(reg.get("probe")) == 2
    assert reg.names() == ["probe"]


def test_overlapping_task_ids_across_suites_raise():
    reg = SuiteRegistry()
    reg.register("probe", [item("shared/1")])
    with pytest.raises(SuiteOverlapError, match="shared/1"):
        reg.register("target", [item("shared/1")])


def test_duplicate_task_ids_within_a_suite_raise():
    reg = SuiteRegistry()
    with pytest.raises(SuiteOverlapError, match="dup/1"):
        reg.register("probe", [item("dup/1"), item("dup/1")])


def test_registering_same_suite_name_twice_raises():
    reg = SuiteRegistry()
    reg.register("probe", [item("probe/1")])
    with pytest.raises(SuiteOverlapError, match="probe"):
        reg.register("probe", [item("probe/2")])


def test_get_unknown_suite_raises():
    reg = SuiteRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_failed_registration_leaves_registry_unchanged():
    reg = SuiteRegistry()
    reg.register("probe", [item("probe/1"), item("probe/2")])

    # Attempt to register with a non-colliding item first, collision second
    with pytest.raises(SuiteOverlapError, match="probe/1"):
        reg.register("target", [item("target/1"), item("probe/1")])

    # Verify registry unchanged
    assert "target" not in reg.names()
    assert reg.names() == ["probe"]
    assert len(reg.get("probe")) == 2
    with pytest.raises(KeyError):
        reg.get("target")

    # Verify target/1 was not leaked into internal seen set
    # by successfully registering another suite with it
    reg.register("other", [item("target/1")])
    assert "other" in reg.names()
