from typing import List, Tuple, Optional

from .base_builder import BasePromptBuilder
from ..prompt import Prompt
from ..message import Message, Role
from ...decoding import DecodeConfig


class ChatPromptBuilder(BasePromptBuilder):

    def build(
        self,
        *,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Tuple[Message, Message]]] = None,
        examples: Optional[List[List[Message]]] = None,
        extra_messages: Optional[List[Message]] = None,
    ) -> Prompt:

        # validation call
        self.validation(user_input, system_prompt, history, examples, extra_messages)

        prompt = Prompt()

        # System prompt
        if system_prompt and system_prompt.strip():
            prompt.add_message(Message(Role.SYSTEM, system_prompt.strip()))

        # Few-shot examples
        if examples:
            for example in examples:
                prompt.extend(example)

        # Conversation history
        if history:
            for user_msg, assistant_msg in history:
                prompt.add_message(user_msg)
                prompt.add_message(assistant_msg)

        # Additional messages
        if extra_messages:
            prompt.extend(extra_messages)

        # User query
        prompt.add_message(Message(Role.USER, user_input.strip()))

        return prompt

    # -----------------------------
    # Single Validation Method
    # -----------------------------
    def validation(
        self,
        user_input: str,
        system_prompt: Optional[str],
        history: Optional[List[Tuple[Message, Message]]],
        examples: Optional[List[List[Message]]],
        extra_messages: Optional[List[Message]],
    ) -> None:

        # user_input
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        # system_prompt
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")

        # examples
        if examples is not None:
            if not isinstance(examples, list):
                raise TypeError("examples must be a list of message lists")

            for example in examples:
                if not isinstance(example, list):
                    raise TypeError("each example must be a list")

                for msg in example:
                    if not isinstance(msg, Message):
                        raise TypeError("examples must contain Message objects")

        # history
        if history is not None:
            if not isinstance(history, list):
                raise TypeError("history must be a list of tuples")

            for pair in history:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise TypeError("history items must be tuples (user, assistant)")

                user_msg, assistant_msg = pair

                if not isinstance(user_msg, Message) or not isinstance(assistant_msg, Message):
                    raise TypeError("history must contain Message objects")

                if user_msg.role != Role.USER:
                    raise ValueError("first history message must have Role.USER")

                if assistant_msg.role != Role.ASSISTANT:
                    raise ValueError("second history message must have Role.ASSISTANT")

        # extra_messages
        if extra_messages is not None:
            if not isinstance(extra_messages, list):
                raise TypeError("extra_messages must be a list")

            for msg in extra_messages:
                if not isinstance(msg, Message):
                    raise TypeError("extra_messages must contain Message objects")

    def default_decode_config(self) -> DecodeConfig:
        return DecodeConfig(
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=1024,
            stop_sequences=None,
        )