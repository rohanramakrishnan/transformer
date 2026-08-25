import torch

from model import BigramLanguageModel, calculate_loss, train_step


def test_bigram_returns_one_score_per_vocabulary_token() -> None:
    model = BigramLanguageModel(vocab_size=65)
    token_ids = torch.tensor(
        [
            [4, 8, 15, 16],
            [23, 42, 4, 8],
        ]
    )

    scores = model(token_ids)

    assert scores.shape == (2, 4, 65)


def test_same_token_uses_same_score_row() -> None:
    model = BigramLanguageModel(vocab_size=65)
    token_ids = torch.tensor([[7, 3, 7]])

    scores = model(token_ids)

    assert torch.equal(scores[0, 0], scores[0, 2])


def test_equal_scores_have_random_guess_loss() -> None:
    logits = torch.zeros((2, 3, 5))
    targets = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 0],
        ]
    )

    loss = calculate_loss(logits, targets)

    assert torch.isclose(loss, torch.tensor(5.0).log())


def test_loss_is_small_when_correct_tokens_have_highest_scores() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        ]
    )
    targets = torch.tensor([[0, 1]])

    loss = calculate_loss(logits, targets)

    assert loss.item() < 0.001


def test_train_step_lowers_loss_on_the_same_batch() -> None:
    torch.manual_seed(42)
    model = BigramLanguageModel(vocab_size=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    token_ids = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])

    loss_before = train_step(model, token_ids, targets, optimizer)
    with torch.no_grad():
        loss_after = calculate_loss(model(token_ids), targets)

    assert loss_after < loss_before


def test_generate_appends_tokens_using_the_previous_token() -> None:
    model = BigramLanguageModel(vocab_size=3)
    with torch.no_grad():
        model.token_scores.weight.fill_(-100.0)
        model.token_scores.weight[0, 1] = 100.0
        model.token_scores.weight[1, 2] = 100.0

    generated = model.generate(torch.tensor([[0]]), max_new_tokens=2)

    assert generated.tolist() == [[0, 1, 2]]


def test_generate_preserves_the_prompt() -> None:
    model = BigramLanguageModel(vocab_size=5)
    prompt = torch.tensor([[1, 3, 2]])

    generated = model.generate(
        prompt,
        max_new_tokens=4,
        generator=torch.Generator().manual_seed(42),
    )

    assert generated.shape == (1, 7)
    assert torch.equal(generated[:, :3], prompt)


def test_generate_accepts_zero_new_tokens() -> None:
    model = BigramLanguageModel(vocab_size=5)
    prompt = torch.tensor([[1, 2]])

    assert torch.equal(model.generate(prompt, max_new_tokens=0), prompt)
