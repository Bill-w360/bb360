# router/worker.py
# -*- coding: utf-8 -*-
import argparse, time, requests, random
from typing import List, Dict
from .route_score import route_score_single

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="模型名（与 models.yaml 一致）")
    ap.add_argument("--cap", required=True, help="模型能力向量, 逗号分隔，如 0.3,0.6,0.8,0.4,0.5")
    ap.add_argument("--domain-index", required=True, help="类别名->index，如 math:0,reading:1,commonsense:2,science:3,coding:4")
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--shadow-mode", default="log", choices=["log","linear","sqrt"])
    ap.add_argument("--shadow-alpha", type=float, default=1.0)
    ap.add_argument("--idle-interval", type=float, default=0.3, help="空闲轮询间隔秒")
    return ap.parse_args()

def parse_cap(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def parse_domain_index(s: str) -> Dict[str,int]:
    out = {}
    for kv in s.split(","):
        k,v = kv.split(":")
        out[k.strip()] = int(v)
    return out

def main():
    args = parse_args()
    base = args.broker.rstrip("/")
    model = args.model
    cap = parse_cap(args.cap)
    domain_index = parse_domain_index(args.domain_index)

    q_len = 0.0             # 你也可以接真实队列长度；这里用一个本地变量演示
    busy_until = 0.0

    while True:
        now = time.time()
        idle = (now >= busy_until)
        # 1) 心跳（声明是否空闲 + 当前队列长度）
        try:
            requests.post(f"{base}/heartbeat", json={"model": model, "idle": idle, "queue_len": q_len}, timeout=2)
        except Exception:
            time.sleep(1.0); continue

        if not idle:
            time.sleep(0.05); continue

        # 2) 仅空闲时拉取候选任务
        try:
            r = requests.get(f"{base}/task/candidate", params={"model": model}, timeout=2)
            if r.status_code == 404:
                time.sleep(args.idle_interval); continue
            r.raise_for_status()
            cand = r.json()
        except Exception:
            time.sleep(args.idle_interval); continue

        task_id = cand["task_id"]
        category = cand["category"]
        difficulty = cand["difficulty"]
        token = cand["token"]

        # 3) 本地计算路由分
        if category not in domain_index:
            # 不认识的类别，给大分
            score = 1e9
        else:
            cat_idx = domain_index[category]
            score = route_score_single(
                model_cap=cap, cat_index=cat_idx, difficulty=difficulty,
                queue_len=q_len, k=args.k, shadow_mode=args.shadow_mode, shadow_alpha=args.shadow_alpha
            )

        # 4) 提交 claim（竞争窗口里比拼最小分）
        try:
            cr = requests.post(f"{base}/task/claim", json={
                "model": model, "task_id": task_id, "token": token, "score": score
            }, timeout=3).json()
        except Exception:
            time.sleep(args.idle_interval); continue

        if cr.get("assigned"):
            # 抢到任务：模拟处理（这里 sleep 表示执行时延），并增加队列占用
            print(f"[{model}] ASSIGNED task={task_id} category={category} diff={difficulty} score={score:.4f}")
            # 假装处理 1.5~3.0 秒
            proc = 1.5 + random.random() * 1.5
            q_len += 1
            busy_until = time.time() + proc
            # “处理完成”后再把队列长度减回去（真实系统应由推理完成回调）
            def finish():
                nonlocal q_len
                time.sleep(proc)
                q_len = max(0.0, q_len - 1.0)
            import threading; threading.Thread(target=finish, daemon=True).start()
        else:
            # 没抢到，稍候再拉
            time.sleep(args.idle_interval)

if __name__ == "__main__":
    main()
