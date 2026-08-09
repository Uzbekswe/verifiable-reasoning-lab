from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from reasonlab.models.qwen3 import Qwen3Tokenizer


def test_tokenizer_encode_decode_round_trip(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(
        [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
    )
    path = tmp_path / "tokenizer-base.json"
    tokenizer.save(str(path))
    wrapper = Qwen3Tokenizer(path)
    text = "hello world"
    assert wrapper.decode(wrapper.encode(text)) == text
    assert wrapper.eos_token_id is not None
