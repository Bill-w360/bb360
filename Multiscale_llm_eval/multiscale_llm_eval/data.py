from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelMeta:
    parameters: str
    family: str
    release_date: str = "unknown"
