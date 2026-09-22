import math
import torch
import pytest
from amenability.probe.entropy import mean_next_token_entropy


def test_uniform_distribution_gives_log_vocab():
    logits = torch.zeros(1, 4, 8)  # batch 1, 4 positions, vocab 8 -> uniform
    mask = torch.ones(1, 4, dtype=torch.long)
    assert mean_next_token_entropy(logits, mask) == pytest.approx(math.log(8), abs=1e-5)


def test_peaked_distribution_gives_near_zero():
    logits = torch.full((1, 3, 5), -50.0)
    logits[:, :, 0] = 50.0
    mask = torch.ones(1, 3, dtype=torch.long)
    assert mean_next_token_entropy(logits, mask) == pytest.approx(0.0, abs=1e-4)


def test_masked_positions_are_excluded():
    logits = torch.zeros(1, 4, 8)
    logits[:, 2:, :] = torch.full((1, 2, 8), -50.0)
    logits[:, 2:, 0] = 50.0
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    # Only the two uniform positions count.
    assert mean_next_token_entropy(logits, mask) == pytest.approx(math.log(8), abs=1e-5)


def test_all_masked_raises():
    logits = torch.zeros(1, 2, 4)
    mask = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="no unmasked"):
        mean_next_token_entropy(logits, mask)


def test_batch_is_averaged_across_sequences():
    uniform = torch.zeros(1, 2, 4)
    peaked = torch.full((1, 2, 4), -50.0)
    peaked[:, :, 0] = 50.0
    logits = torch.cat([uniform, peaked], dim=0)
    mask = torch.ones(2, 2, dtype=torch.long)
    assert mean_next_token_entropy(logits, mask) == pytest.approx(math.log(4) / 2, abs=1e-4)
