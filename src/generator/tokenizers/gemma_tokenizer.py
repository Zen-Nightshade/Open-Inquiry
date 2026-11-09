import sentencepiece as spm

from .base_tokenizer import BaseTokenizer

class GemmaTokenizer(BaseTokenizer):
    def __init__(self, model_path: str | None = None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def count_tokens(self, text: str) -> int:
        return len(self.sp.encode(text, out_type=int))