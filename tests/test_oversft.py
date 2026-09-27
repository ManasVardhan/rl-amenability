import pytest
from scripts.make_oversft_variants import (
    OVERSFT_LEVELS, variant_key, variant_plan, BASE_SFT_STEPS,
)


def test_variant_key_encodes_base_and_level():
    assert variant_key("qwen2.5-0.5b", 4) == "qwen2.5-0.5b-oversft4x"


def test_plan_includes_the_fresh_base_as_level_zero():
    plan = variant_plan(["qwen2.5-0.5b"])
    levels = [p["level"] for p in plan]
    assert levels == [0, 1, 4]
    assert plan[0]["variant_key"] == "qwen2.5-0.5b"
    assert plan[0]["max_steps"] == 0


def test_plan_covers_both_control_bases():
    plan = variant_plan(["qwen2.5-0.5b", "llama-3.2-1b"])
    assert len({p["base_key"] for p in plan}) == 2
    assert len(plan) == 6  # 2 bases x (base + 2 over-SFT levels)


def test_known_order_is_descending_in_amenability():
    plan = variant_plan(["qwen2.5-0.5b"])
    orders = [p["known_order"] for p in plan]
    # Highest known_order = most amenable = the fresh base.
    assert orders == [2, 1, 0]


def test_step_budgets_scale_with_level():
    plan = variant_plan(["qwen2.5-0.5b"])
    assert [p["max_steps"] for p in plan] == [0, BASE_SFT_STEPS, 4 * BASE_SFT_STEPS]


def test_unknown_base_key_raises():
    with pytest.raises(KeyError):
        variant_plan(["not-a-model"])


def test_non_control_base_raises():
    with pytest.raises(ValueError, match="control_base"):
        variant_plan(["qwen2.5-1.5b"])


def test_levels_are_the_pre_registered_pair():
    assert OVERSFT_LEVELS == (1, 4)
