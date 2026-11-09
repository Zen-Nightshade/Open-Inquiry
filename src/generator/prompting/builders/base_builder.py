from ..prompt import Prompt
from ...decoding import DecodeConfig
from abc import ABC, abstractmethod

class BasePromptBuilder(ABC):

    @abstractmethod
    def build(self, **kwargs) -> Prompt:
        pass

    @abstractmethod
    def validation(self, **kwargs) -> None:
        pass

    @abstractmethod
    def default_decode_config(self) -> DecodeConfig:
        pass