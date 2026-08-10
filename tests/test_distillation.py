import json

import torch

from reasonlab.distillation import (
    DistillationExample,
    encode_distillation_example,
    hard_distillation_loss,
    load_distillation_examples,
    render_distillation_prompt,
)


class TinyTokenizer:
    def encode(self, text):
        return [ord(character) % 11 + 1 for character in text]


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        return self.head(self.embedding(input_ids))


def test_distillation_format_and_encoding_preserve_prompt_boundary():
    example = DistillationExample(3, "2 + 2?", "4", "Work briefly.", r"\boxed{4}")
    encoded = encode_distillation_example(TinyTokenizer(), example)
    assert encoded.prompt_length > 0
    assert encoded.prompt_length < encoded.token_count
    assert render_distillation_prompt(example.problem).endswith("Solution:\n")
    assert example.target_text.endswith(r"\boxed{4}")


def test_hard_distillation_loss_is_finite_and_differentiable():
    tokenizer = TinyTokenizer()
    example = DistillationExample(0, "1 + 1?", "2", "One plus one is two.", r"\boxed{2}")
    encoded = encode_distillation_example(tokenizer, example)
    input_ids = torch.tensor(encoded.input_ids, dtype=torch.long)
    model = TinyLanguageModel()
    loss = hard_distillation_loss(model, input_ids, encoded.prompt_length)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None


def test_distillation_loader_validates_schema(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {
                    "problem": "1 + 1?",
                    "gtruth_answer": "2",
                    "message_thinking": "Compute it.",
                    "message_content": r"\boxed{2}",
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = load_distillation_examples(path)
    assert len(examples) == 1
    assert examples[0].answer == "2"
