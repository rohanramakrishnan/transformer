"""Language models."""

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next token from the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_scores = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.token_scores.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_scores(token_ids)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Append tokens sampled from the model's next-token scores."""
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        for _ in range(max_new_tokens):
            logits = self(token_ids)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids


def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return cross-entropy loss for next-token predictions."""
    batch_size, block_size, vocab_size = logits.shape
    flat_logits = logits.reshape(batch_size * block_size, vocab_size)
    flat_targets = targets.reshape(batch_size * block_size)
    return F.cross_entropy(flat_logits, flat_targets)


def train_step(
    model: BigramLanguageModel,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Update the model once and return the loss before the update."""
    optimizer.zero_grad()
    logits = model(token_ids)
    loss = calculate_loss(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()
