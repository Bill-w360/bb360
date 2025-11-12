# -*- coding: utf-8 -*-
import json, argparse, math

def norm(v): 
    return math.sqrt(sum(float(x)*float(x) for x in v))

def dif_by_norm(v):
    r = norm(v)
    # 归一到 [0,1]，再分档
    if r <= 0: x = 0.5
    else: x = min(1.0, r / (len(v)**0.5))  # 简单缩放
    return "easy" if x < 0.35 else ("medium" if x < 0.7 else "hard")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)   # 你的成对样本 jsonl
    ap.add_argument("--out", required=True)              # 生成的模拟任务 jsonl
    args = ap.parse_args()

    out = open(args.out, "w", encoding="utf-8")
    n=0
    for i, line in enumerate(open(args.inp,"r",encoding="utf-8"),1):
        s=line.strip()
        if not s: continue
        ex=json.loads(s)
        # 需要 task_label / task_vec 至少存在
        if "task_label" not in ex or "task_vec" not in ex:
            continue
        t = {
            "id": f"synth-{i}",
            "dataset": "synthetic",
            "category": ex["task_label"],        # 用你已有的标签当作路由维度名
            "difficulty": dif_by_norm(ex["task_vec"]),
            "arrival_time": float(i),            # 简单按顺序到达
            "prompt": "", "meta": {}
        }
        out.write(json.dumps(t, ensure_ascii=False)+"\n"); n+=1
    out.close()
    print(f"[ok] wrote {args.out}, tasks={n}")

if __name__ == "__main__":
    main()
