import tiktoken

from .base_tokenizer import BaseTokenizer

class QwenTokenizer(BaseTokenizer):
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str):
        return self.encoding.encode(text)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
