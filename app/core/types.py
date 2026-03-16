from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List
import time


class AppMode(str, Enum):
    WORD = "word"
    LETTER = "letter"


@dataclass
class Prediction:
    mode: AppMode
    top1_label: str
    top1_confidence: float
    topk: List[tuple[str, float]]
    timestamp: float = field(default_factory=time.time)


@dataclass
class FramePacket:
    frame_id: int
    bgr: Any  # np.ndarray
    timestamp: float = field(default_factory=time.time)


@dataclass
class TranscriptToken:
    token: str
    confidence: float
    mode: AppMode
    timestamp: float = field(default_factory=time.time)

