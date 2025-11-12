import json, math, os
from typing import Dict, List


# ------- 工具 -------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ------- 双线性（数值派） -------


class BilinearMatcher:
    """
    Match(t,m) = sigma( t^T W m + b )
    t,m 与 cap_dims 对齐且在 [0,1]
    W: DxD, b: scalar
    """
    def __init__(self, dim: int, weights_path: str):
        self.dim = dim
        self.weights_path = weights_path
        self.W = [[0.5 if i==j else 0.0 for j in range(dim)] for i in range(dim)]
        self.b = 0.0
        if os.path.exists(weights_path):
            self.load(weights_path)


    def save(self, path: str = None):
        path = path or self.weights_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"W": self.W, "b": self.b}, f)


    def load(self, path: str = None):
        path = path or self.weights_path
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.W = obj["W"]; self.b = obj["b"]


    def score_raw(self, t: List[float], m: List[float]) -> float:
        s = 0.0
        for i in range(self.dim):
            if t[i]==0:
                continue
            row = self.W[i]
            s += t[i] * sum(row[j]*m[j] for j in range(self.dim))
        s += self.b
        return s


    def match(self, t: List[float], m: List[float]) -> float:
        return _clip01(_sigmoid(self.score_raw(t, m)))


    def train_bce(self, data: List[Dict], lr: float = 0.05, epochs: int = 5, l2: float = 1e-4):
        for _ in range(epochs):
            for ex in data:
                t = ex["task_vec"]; m = ex["model_cap"]; y = float(ex["y"])
                z = self.score_raw(t, m)
                p = _sigmoid(z)
                g = (p - y)
                for i in range(self.dim):
                    ti = t[i]
                    if ti==0: continue
                    row = self.W[i]
                    for j in range(self.dim):
                        row[j] -= lr * (g * ti * m[j] + l2*row[j])
                self.b -= lr * g


# ------- 语义派（可选占位） -------


class SemanticMatcher:
    def __init__(self):
        pass


    def encode_text(self, text: str) -> List[float]:
    # 占位实现：接入你的文本嵌入后替换
        return [0.5, 0.5, 0.5, 0.5]


    def match(self, task_text: str, model_text: str) -> float:
        return 0.5