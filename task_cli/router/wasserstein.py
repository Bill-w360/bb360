from typing import List, Literal


def w1_1d_ordered(p: List[float], q: List[float]) -> float:
    """
    一维Wasserstein-1（Earth Mover's Distance）在有序bins上的闭式：
    W1 = sum |CDF_p[i] - CDF_q[i]| over i
    要求 p,q≥0，sum为1。若非概率向量，外部需先normalize。
    """
    if len(p) != len(q):
        raise ValueError("p and q dim mismatch")
    sp, sq = sum(p), sum(q)
    p = [0.0]*len(p) if sp <= 0 else [x/sp for x in p]
    q = [0.0]*len(q) if sq <= 0 else [x/sq for x in q]
    cdf_p = cdf_q = 0.0
    w = 0.0
    for i in range(len(p)):
        cdf_p += p[i]
        cdf_q += q[i]
        w += abs(cdf_p - cdf_q)
    return w


def l1_unordered(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x,y dim mismatch")
    return sum(abs(a-b) for a,b in zip(x,y))


def w1_multidim(model_vec: List[float],
                task_vec: List[float],
                dim_types: List[Literal["ordered","unordered"]],
                dim_weights: List[float]) -> float:
    assert len(model_vec)==len(task_vec)==len(dim_types)==len(dim_weights)
    total = 0.0
    for (m,t,typ,w) in zip(model_vec, task_vec, dim_types, dim_weights):
        if w<=0:
            continue
        if typ=="unordered":
            total += w * abs(m - t)
        else:
            # ordered维度：用二元有序分布 [x,1-x]
            m = max(0.0, min(1.0, m))
            t = max(0.0, min(1.0, t))
            p = [m, 1.0-m]
            q = [t, 1.0-t]
            total += w * w1_1d_ordered(p, q)
    return total