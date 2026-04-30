from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str


class ContextWindow:
    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add(self, role: Literal["user", "assistant"], content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def get_user_turns(self) -> list[str]:
        return [t.content for t in self._turns if t.role == "user"]

    def to_messages(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        self._turns = []
