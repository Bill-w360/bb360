from typing import Dict
import math

def shadow_price(queue: Dict[str, float], cfg: Dict) -> float:
    """
    π = f(queue) 的可插拔实现。
    支持:
      - linear: π = k * (wait_ms if use_wait_ms else qlen)
      - log:    π = log(1 + k * x)
      - sqrt:   π = sqrt(k * x)
    你也可以定制：利用 util 或 加权合成。
    """
    fn  = cfg.get("fn","linear")
    k   = float(cfg.get("k", 0.001))
    use_wait = bool(cfg.get("use_wait_ms", True))
    x = float(queue.get("wait_ms", 0.0) if use_wait else queue.get("qlen", 0.0))
    if x < 0: x = 0.0
    if fn == "linear":
        return k * x
    if fn == "log":
        return math.log1p(k * x)
    if fn == "sqrt":
        import math
        return (k * x) ** 0.5
    else:
        # 自定义可在此扩展
        return k * x
