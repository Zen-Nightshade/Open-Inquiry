from .base_tokenizer import BaseTokenizer
from .llama_tokenizer import LlamaTokenizer
from .gemma_tokenizer import GemmaTokenizer
from .qwen_tokenizer import QwenTokenizer

class TokenizerFactory:
    _REGISTRY = {
        "llama": LlamaTokenizer,
        "gemma": GemmaTokenizer,
        "qwen": QwenTokenizer,
    }

    @classmethod
    def create(cls, tokenizer_name: str) -> BaseTokenizer:

        if tokenizer_name not in cls._REGISTRY:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

        tokenizer_class = cls._REGISTRY[tokenizer_name]

        return tokenizer_class()