from .prompt import Prompt

class PromptRenderer:

    def render(self, prompt: Prompt) -> str:
        parts = []

        for message in prompt.get_messages():
            parts.append(f"[{message.role.name}]")
            parts.append(message.content)
            parts.append("")

        return "\n".join(parts).strip()