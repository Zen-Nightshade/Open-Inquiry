from abc import ABC, abstractmethod
from typing import List

class BaseTokenizer(ABC):
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass