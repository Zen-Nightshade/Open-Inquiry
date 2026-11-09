from typing import List, Optional, Dict, Any


class DecodeConfig:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        max_tokens: int = 512,
        min_tokens: int = 0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
        seed: Optional[int] = None,
        enforce_json: bool = False,
        citation_required: bool = False,
        allowed_tokens: Optional[List[int]] = None,
        banned_tokens: Optional[List[int]] = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop_sequences = stop_sequences
        self.seed = seed
        self.enforce_json = enforce_json
        self.citation_required = citation_required
        self.allowed_tokens = allowed_tokens
        self.banned_tokens = banned_tokens

        self.validate()

    def _require_type(self, name: str, value: Any, expected_type: Any):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{name} must be of type {expected_type}, "
                f"got value {value!r} (type={type(value).__name__})"
            )

    def validate(self) -> None:
        # ---- Type checks ----
        self._require_type("temperature", self.temperature, (int, float))
        self._require_type("top_p", self.top_p, (int, float))
        self._require_type("top_k", self.top_k, int)
        self._require_type("do_sample", self.do_sample, bool)
        self._require_type("max_tokens", self.max_tokens, int)
        self._require_type("min_tokens", self.min_tokens, int)
        self._require_type("length_penalty", self.length_penalty, (int, float))
        self._require_type("repetition_penalty", self.repetition_penalty, (int, float))
        self._require_type("presence_penalty", self.presence_penalty, (int, float))
        self._require_type("frequency_penalty", self.frequency_penalty, (int, float))
        self._require_type("enforce_json", self.enforce_json, bool)
        self._require_type("citation_required", self.citation_required, bool)

        if self.stop_sequences is not None:
            self._require_type("stop_sequences", self.stop_sequences, list)
            if not all(isinstance(s, str) for s in self.stop_sequences):
                raise TypeError(
                    f"stop_sequences must be List[str], got {self.stop_sequences!r}"
                )

        if self.allowed_tokens is not None:
            self._require_type("allowed_tokens", self.allowed_tokens, list)
            if not all(isinstance(t, int) for t in self.allowed_tokens):
                raise TypeError(
                    f"allowed_tokens must be List[int], got {self.allowed_tokens!r}"
                )

        if self.banned_tokens is not None:
            self._require_type("banned_tokens", self.banned_tokens, list)
            if not all(isinstance(t, int) for t in self.banned_tokens):
                raise TypeError(
                    f"banned_tokens must be List[int], got {self.banned_tokens!r}"
                )

        # ---- Value checks ----
        
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}"
            )

        if not (0 <= self.top_p <= 1):
            raise ValueError(
                f"top_p must be between 0 and 1, got {self.top_p}"
            )

        if self.top_k < 0:
            raise ValueError(
                f"top_k must be >= 0, got {self.top_k}"
            )

        if self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be > 0, got {self.max_tokens}"
            )

        if self.min_tokens < 0:
            raise ValueError(
                f"min_tokens must be >= 0, got {self.min_tokens}"
            )

        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) cannot exceed "
                f"max_tokens ({self.max_tokens})"
            )

        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}"
            )

        if self.presence_penalty < 0:
            raise ValueError(
                f"presence_penalty must be >= 0, got {self.presence_penalty}"
            )

        if self.frequency_penalty < 0:
            raise ValueError(
                f"frequency_penalty must be >= 0, got {self.frequency_penalty}"
            )

        if not self.do_sample:
            if self.temperature != 1.0:
                raise ValueError(
                    f"temperature must be 1.0 when do_sample=False, "
                    f"got {self.temperature}"
                )
            if self.top_p != 1.0:
                raise ValueError(
                    f"top_p must be 1.0 when do_sample=False, got {self.top_p}"
                )
            if self.top_k != 0:
                raise ValueError(
                    f"top_k must be 0 when do_sample=False, got {self.top_k}"
                )

    def to_model_kwargs(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_tokens,
            "min_new_tokens": self.min_tokens,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
            "enforce_json": self.enforce_json,
            "citation_required": self.citation_required,
            "allowed_tokens": self.allowed_tokens,
            "banned_tokens": self.banned_tokens,
        }