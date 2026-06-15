from dataclasses import dataclass, field
from internal.domain.model.entity import Entity

@dataclass
class Tag:
    Entity: Entity = field(default_factory=Entity)
    Weight: float = 1.0

    def increase_weight(self, increment: float = 0.1) -> None:
        self.Weight = self.Weight + max(0.1, increment)

    def decrease_weight(self, decrement: float = 0.1) -> None:
        self.Weight = self.Weight - max(0.1, decrement)
