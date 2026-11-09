from typing import List, Optional
from copy import deepcopy
from .message import Message, Role


class Prompt:
    def __init__(self, messages: Optional[List[Message]] = None):
        self._messages: List[Message] = list(messages) if messages else []

    def add_message(self, message: Message) -> None:
        if not isinstance(message, Message):
            raise TypeError("add_message expects a Message instance")
        self._messages.append(message)

    def extend(self, messages: List[Message]) -> None:
        for message in messages:
            self.add_message(message)

    def copy(self) -> "Prompt":
        return Prompt(messages=deepcopy(self._messages))

    def get_messages(self) -> List[Message]:
        return list(self._messages)