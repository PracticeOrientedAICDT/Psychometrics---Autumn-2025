from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------
# BALLOON DATA MODEL
# ------------------------------------------------------------

@dataclass
class Balloon:
    id: int
    level: int
    difficulty: int
    operation: str
    left: str
    right: str
    answer: str
    start_time: float
    ttl: float
    channel: int  # 0–3


# ------------------------------------------------------------
# RESPONSE MODEL (optional extension)
# ------------------------------------------------------------

@dataclass
class ResponseEvent:
    timestamp: float
    user_input: str
    correct: bool
    balloon_id: Optional[int] = None
    reaction_ms: Optional[float] = None


# ------------------------------------------------------------
# LEVEL STATE MODEL (optional extension)
# ------------------------------------------------------------

@dataclass
class LevelState:
    level: int
    hits_this_level: int
    total_correct: int
    total_missed: int
