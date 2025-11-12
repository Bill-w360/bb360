from typing import Dict
from .schemas import Task


DOMAIN_KEYWORDS = {
    "math": ["sum","add","average","number","solve","ratio","probability","algebra","算","和","比例","概率"],
    "commonsense": ["plausible","likely","most","sensible","常识","更可能","合理"],
    "code": ["python","function","class","compile","bug","实现","代码","程序"],
    "reading": ["passage","paragraph","according to","阅读","段落","根据"],
    "science": ["physics","chemistry","biology","force","cell","实验","物理","化学","生物"],
}


def heuristic_domain(prompt: str) -> str:
    p = prompt.lower()
    best, bestc = "commonsense", 0
    for d, kws in DOMAIN_KEYWORDS.items():
        c = sum(1 for w in kws if w in p)
        if c > bestc:
            best, bestc = d, c
    return best


def difficulty_to_scalar(diff: str) -> float:
    table = {"easy":0.2, "medium":0.5, "hard":0.8}
    return table.get(str(diff).lower(), 0.5)


def make_task_vector(task: Task, cap_dims_spec: Dict[str, Dict]) -> Dict[str, float]:
    domain = task.meta.get("domain") or heuristic_domain(task.prompt)
    difficulty = difficulty_to_scalar(task.meta.get("difficulty"))
    vec = []
    for name, spec in cap_dims_spec.items():
        if name == "math_reasoning":
            v = 0.7 if domain=="math" else 0.5
        elif name == "commonsense":
            v = 0.7 if domain=="commonsense" else 0.5
        elif name == "code":
            v = 0.7 if domain=="code" else 0.5
        elif name == "reading":
            v = 0.7 if domain=="reading" else 0.5
        elif name == "science":
            v = 0.7 if domain=="science" else 0.5
        else:
            v = 0.5
        v = min(1.0, max(0.0, 0.5*v + 0.5*difficulty))
        vec.append(v)
    return {"domain": domain, "difficulty": difficulty, "vec": vec}


def classify_task_label(task: Task) -> str:
    domain = task.meta.get("domain") or heuristic_domain(task.prompt)
    diff = task.meta.get("difficulty","medium").lower()
    return f"{domain}-{diff}"