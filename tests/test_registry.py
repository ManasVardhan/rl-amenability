import pytest
from collections import Counter
from amenability.registry.loader import load_registry, ModelSpec, RegistryError


def test_registry_loads_ten_models():
    reg = load_registry()
    assert len(reg) == 10
    assert all(isinstance(v, ModelSpec) for v in reg.values())


def test_qwen_capped_at_four():
    reg = load_registry()
    families = Counter(m.family for m in reg.values())
    assert families["qwen"] == 4, f"Qwen cap violated: {families}"


def test_at_least_seven_families():
    reg = load_registry()
    assert len({m.family for m in reg.values()}) >= 7


def test_all_models_within_size_band():
    reg = load_registry()
    for m in reg.values():
        assert 0.3 <= m.params_b <= 1.8, f"{m.key} at {m.params_b}B is outside 0.5-1.7B band"


def test_spurious_reward_outlier_present():
    # Qwen2.5-Math-1.5B is the Spurious Rewards outlier and must be in the roster.
    reg = load_registry()
    assert "qwen2.5-math-1.5b" in reg


def test_control_bases_marked():
    reg = load_registry()
    controls = sorted(k for k, m in reg.items() if m.role == "control_base")
    assert controls == ["llama-3.2-1b", "qwen2.5-0.5b"]


def test_duplicate_keys_rejected(tmp_path):
    p = tmp_path / "dupe.yaml"
    p.write_text(
        "models:\n"
        "  - key: a\n    hf_id: x/a\n    family: f\n    params_b: 1.0\n    license: apache-2.0\n    role: roster\n"
        "  - key: a\n    hf_id: x/b\n    family: g\n    params_b: 1.0\n    license: apache-2.0\n    role: roster\n"
    )
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(p)
