from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Task:
    id: str
    prompt: str
    meta: Dict[str, Any]
    features: Optional[Dict[str, float]] = None


@dataclass
class ModelInfo:
    name: str
    labels: List[str]
    capability: List[float]
    cost: Dict[str, float]
    queue: Dict[str, float]


@dataclass
class RouteConfig:
    cap_dims: List[Dict[str, Any]] # 每项: {name,type,weight}
    alpha: float
    beta: float
    gamma: float
    delta: float
    cost_weights: Dict[str, float]
    backpressure: Dict[str, Any]
    label_matcher: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Assignment:
    task_id: str
    task_label: str
    chosen_model: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)