from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def __init__(self, output_schema):
        pass

    @abstractmethod
    def parse(self, text):
        pass