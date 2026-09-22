from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_next_token_entropy(logits: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """Mean token-level entropy of the next-token distribution over unmasked positions."""
    if attention_mask.sum() == 0:
        raise ValueError("no unmasked positions to measure entropy over")
    logits = logits.float()
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)  # (batch, seq)
    mask = attention_mask.to(ent.dtype)
    return float((ent * mask).sum() / mask.sum())


class EntropyProbe:
    """Measures policy entropy on a fixed prompt batch, held constant across all steps."""

    def __init__(self, tokenizer, prompts: list[str], max_length: int = 256,
                 device: str = "cuda") -> None:
        self.device = device
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    @torch.no_grad()
    def measure(self, model) -> float:
        was_training = model.training
        model.eval()
        try:
            out = model(
                input_ids=self.input_ids.to(self.device),
                attention_mask=self.attention_mask.to(self.device),
            )
            return mean_next_token_entropy(out.logits, self.attention_mask.to(self.device))
        finally:
            if was_training:
                model.train()
