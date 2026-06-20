from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Metadata:
    Person: str = ""
    Object: str = ""
    Location: str = ""
    Event: str = ""
    Organization: str = ""
    Preference: str = ""
    HappendTime: datetime | None = None
    MentionedTime: datetime | None = None
    History: List[str] | None = None
    source_session_id: str = ""
    source_turn_index: int | None = None
    source_turn_role: str = ""

