import math
import torch
import pytest
from amenability.probe.entropy import mean_next_token_entropy, EntropyProbe


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


# Mock-based tests for EntropyProbe class

class FakeTokenizer:
    """Minimal fake tokenizer for testing EntropyProbe."""

    def __call__(self, prompts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        batch_size = len(prompts)
        seq_length = 3
        return {
            "input_ids": torch.zeros((batch_size, seq_length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_length), dtype=torch.long),
        }


class FakeModel:
    """Minimal fake model for testing EntropyProbe."""

    def __init__(self, vocab_size=4, training=True):
        self.vocab_size = vocab_size
        self.training = training
        self.eval_called = False
        self.train_called = False

    def eval(self):
        self.eval_called = True
        self.training = False
        return self

    def train(self):
        self.train_called = True
        self.training = True
        return self

    def __call__(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        logits = torch.zeros((batch_size, seq_length, self.vocab_size))

        class Output:
            pass

        output = Output()
        output.logits = logits
        return output


def test_measure_returns_entropy_of_the_fixed_batch():
    """EntropyProbe.measure() returns entropy of the fixed batch."""
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel(vocab_size=4)

    probe = EntropyProbe(fake_tokenizer, ["test"], device="cpu")
    entropy = probe.measure(fake_model)

    assert entropy == pytest.approx(math.log(4), abs=1e-5)


def test_measure_restores_training_mode():
    """EntropyProbe.measure() restores training mode when model was training."""
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel(vocab_size=4, training=True)

    probe = EntropyProbe(fake_tokenizer, ["test"], device="cpu")
    probe.measure(fake_model)

    assert fake_model.training is True
    assert fake_model.eval_called is True


def test_measure_leaves_eval_model_in_eval():
    """EntropyProbe.measure() leaves an eval model in eval mode."""
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel(vocab_size=4, training=False)

    probe = EntropyProbe(fake_tokenizer, ["test"], device="cpu")
    probe.measure(fake_model)

    assert fake_model.training is False


def test_measure_restores_training_mode_even_when_forward_raises():
    """EntropyProbe.measure() restores training mode even if forward pass raises."""
    fake_tokenizer = FakeTokenizer()

    class FailingModel(FakeModel):
        def __call__(self, input_ids, attention_mask):
            raise RuntimeError("Forward pass failed")

    fake_model = FailingModel(vocab_size=4, training=True)

    probe = EntropyProbe(fake_tokenizer, ["test"], device="cpu")

    with pytest.raises(RuntimeError, match="Forward pass failed"):
        probe.measure(fake_model)

    assert fake_model.training is True
