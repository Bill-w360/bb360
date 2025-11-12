# router/broker.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import time, uuid, asyncio

app = FastAPI(title="Decentralized Router Broker")

# 内存状态（生产可换成 Redis/DB）
TASKS: Dict[str, dict] = {}      # task_id -> {category,difficulty,status,assigned_to}
QUEUE: List[str] = []            # 待分配 task_id 列表
MODELS: Dict[str, dict] = {}     # model_name -> {idle:bool, queue_len:float, updated:ts}
CLAIMS: Dict[str, dict] = {}     # task_id -> { window_end:ts, claims:{model:{score,token,ts}} }

COMPETE_WINDOW_SEC = 0.2         # 竞争窗口
CANDIDATE_LOCK_SEC = 0.5         # 单模型持有候选的软锁（避免重复发给同一模型）

class HB(BaseModel):
    model: str
    idle: bool
    queue_len: float

class NewTask(BaseModel):
    category: str
    difficulty: str

class CandidateResp(BaseModel):
    task_id: str
    category: str
    difficulty: str
    token: str          # 防止伪造的随机令牌
    window_ms: int

class ClaimReq(BaseModel):
    model: str
    task_id: str
    token: str
    score: float

@app.post("/heartbeat")
def heartbeat(hb: HB):
    MODELS[hb.model] = {"idle": bool(hb.idle), "queue_len": float(hb.queue_len), "updated": time.time()}
    return {"ok": True}

@app.post("/task/offer")
def task_offer(t: NewTask):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "category": t.category, "difficulty": t.difficulty,
        "status": "queued", "assigned_to": None,
        "last_candidate_at": 0.0
    }
    QUEUE.append(task_id)
    return {"task_id": task_id}

@app.get("/task/candidate", response_model=CandidateResp)
def task_candidate(model: str):
    # 只有空闲模型能拿候选
    st = MODELS.get(model)
    if not st or not st.get("idle", False):
        raise HTTPException(403, "model not idle or not registered")

    # 找一个尚未分配 & 没在竞争窗口的任务
    now = time.time()
    for tid in list(QUEUE):
        t = TASKS.get(tid)
        if not t or t["status"] != "queued":
            continue
        # 简单的软锁，避免一个模型在短时间内重复拿到同一任务
        if now - t["last_candidate_at"] < CANDIDATE_LOCK_SEC:
            continue

        # 开启/续期竞争窗口
        claims = CLAIMS.get(tid)
        if not claims or now > claims.get("window_end", 0):
            CLAIMS[tid] = {"window_end": now + COMPETE_WINDOW_SEC, "claims": {}}

        token = str(uuid.uuid4())
        t["last_candidate_at"] = now
        return CandidateResp(
            task_id=tid,
            category=t["category"],
            difficulty=t["difficulty"],
            token=token,
            window_ms=int(COMPETE_WINDOW_SEC * 1000),
        )
    raise HTTPException(404, "no queued task")

@app.post("/task/claim")
async def task_claim(req: ClaimReq):
    t = TASKS.get(req.task_id)
    if not t or t["status"] != "queued":
        raise HTTPException(409, "task not available")
    if req.model not in MODELS or not MODELS[req.model].get("idle", False):
        raise HTTPException(403, "model not idle or not registered")

    slot = CLAIMS.get(req.task_id)
    if not slot:
        # 没窗口也允许：单独抢到即刻成功（退化情形）
        t["status"] = "assigned"; t["assigned_to"] = req.model
        QUEUE.remove(req.task_id)
        return {"assigned": True, "winner": req.model}

    now = time.time()
    window_end = slot["window_end"]
    slot["claims"][req.model] = {"score": float(req.score), "token": req.token, "ts": now}

    # 等窗口结束（如果还没到）
    if now < window_end:
        await asyncio.sleep(window_end - now)

    # 窗口结束时若任务还未被分配，选最小分
    if t["status"] != "queued":
        assigned = (t["assigned_to"] == req.model)
        return {"assigned": assigned, "winner": t["assigned_to"]}

    # 选 winner
    claims = slot["claims"]
    if not claims:
        return {"assigned": False, "winner": None}

    winner = min(claims.items(), key=lambda kv: kv[1]["score"])[0]
    t["status"] = "assigned"; t["assigned_to"] = winner
    if req.task_id in QUEUE:
        QUEUE.remove(req.task_id)
    return {"assigned": req.model == winner, "winner": winner}
