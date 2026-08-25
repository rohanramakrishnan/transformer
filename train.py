"""Train a language model."""

import torch

from dataset import get_batch, load_token_ids
from model import BigramLanguageModel, train_step
from tokenizer import CharTokenizer

BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_STEPS = 1_000
LEARNING_RATE = 1e-2


def main() -> None:
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = CharTokenizer.load("data/vocab.json")
    train_data = load_token_ids("data/train.bin")
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"device: {device}")
    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(
            train_data,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
            generator=generator,
        )
        loss = train_step(model, x, y, optimizer)

        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    start_token = tokenizer.encode("\n")[0]
    prompt = torch.tensor([[start_token]], dtype=torch.long, device=device)
    generated_ids = model.generate(prompt, max_new_tokens=500)

    print("\ngenerated text:\n")
    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
