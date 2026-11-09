from enum import Enum

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message():
    def __init__(self, role: Role, content: str, metadata: dict = None):
        self.role = role
        self.content = content
        self.metadata = metadata

    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata
        }