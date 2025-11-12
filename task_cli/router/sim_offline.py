# router/sim_offline.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Union, Callable, Any, Optional
import math, heapq, random, statistics
from dataclasses import dataclass, field
from .route_score import pick_model  # 用到“相对背压”的打分
from .utils import read_jsonl, load_yaml

Number = Union[int, float]

@dataclass
class TaskEvt:
    tid: str
    category: str
    difficulty: Union[str, Number]
    t_arrival: float

@dataclass
class ModelState:
    name: str
    cap: List[float]
    busy_until: float = 0.0
    q_inflight: int = 0                 # 已分配但未完成的任务数
    served: int = 0                     # 完成任务数
    busy_time: float = 0.0              # 累计占用时间
    last_update: float = 0.0            # 为计算busy_time的积分

@dataclass
class AssignRecord:
    tid: str
    category: str
    difficulty: Union[str, Number]
    chosen: str
    score: float
    t_arrival: float
    t_start: float
    t_finish: float
    wait: float
    runtime: float

@dataclass
class ServiceConf:
    """
    配置服务时间分布。
    支持三种形式（按优先级匹配）：
      1) per_pair[(model, category)] = {"dist":"const|expo|normal", "param":...}
      2) per_model[model] = {...}
      3) default = {...}
    """
    per_pair: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    per_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default: Dict[str, float] = field(default_factory=lambda: {"dist":"const", "param":1.0})

    def draw(self, model: str, category: str) -> float:
        conf = self.per_pair.get((model, category)) or self.per_model.get(model) or self.default
        dist = conf.get("dist", "const")
        p = float(conf.get("param", 1.0))
        if dist == "const":
            return p
        if dist == "expo":
            # 指数分布，param=均值
            u = random.random()
            return max(1e-6, -p * math.log(1.0 - u))
        if dist == "normal":
            # 正态，param=均值；std 可选参数 std=...
            std = float(conf.get("std", p * 0.25))
            x = random.gauss(p, std)
            return max(1e-6, x)
        raise ValueError(f"unknown dist: {dist}")

class OfflineSim:
    def __init__(
        self,
        domain_index: Dict[str, int],
        models_cap: Dict[str, List[float]],
        tasks: List[TaskEvt],
        service_conf: ServiceConf | None = None,
        k: float = 0.2,
        shadow_mode: str = "log",
        shadow_alpha: float = 1.0,
        compete_window: float = 0.2,
        rng_seed: Optional[int] = 1234,
    ):
        self.domain_index = domain_index
        self.models: Dict[str, ModelState] = {
            m: ModelState(name=m, cap=cap) for m, cap in models_cap.items()
        }
        self.tasks = sorted(tasks, key=lambda x: x.t_arrival)
        self.service = service_conf or ServiceConf()
        self.k, self.shadow_mode, self.shadow_alpha = k, shadow_mode, shadow_alpha
        self.win = float(compete_window)
        self.t: float = 0.0
        self.events: List[Tuple[float, str, Any]] = []  # (time, type, payload)
        self.waiting: List[TaskEvt] = []               # 先进先出
        self.records: List[AssignRecord] = []
        if rng_seed is not None:
            random.seed(rng_seed)

    # ---------- 事件驱动 ----------
    def push_evt(self, when: float, etype: str, payload: Any):
        heapq.heappush(self.events, (when, etype, payload))

    def run(self) -> Dict[str, Any]:
        # 初始化：投递所有到达事件
        for tk in self.tasks:
            self.push_evt(tk.t_arrival, "arrival", tk)

        while self.events:
            t, etype, payload = heapq.heappop(self.events)
            self._advance_time(t)
            if etype == "arrival":
                self._on_arrival(payload)
            elif etype == "finish":
                self._on_finish(payload)
            else:
                raise RuntimeError(f"unknown event {etype}")

            # 每次事件后尽量把能分配的任务都分掉（模拟短竞争窗口：把当前空闲模型视为同一窗口内的竞争者）
            self._greedy_assign()

        # 结束时把 busy_time 积分到最终时刻
        t_end = max([self.t] + [rec.t_finish for rec in self.records] + [0.0])
        for m in self.models.values():
            self._integrate_busy(m, t_end)

        return self._summarize(t_end)

    def _advance_time(self, new_t: float):
        self.t = float(new_t)

    def _integrate_busy(self, m: ModelState, now: float):
        # 把上个时间点到 now 的“忙状态”积分进 busy_time
        last = m.last_update
        if now > last:
            busy = 1.0 if m.busy_until > last else 0.0
            # 如果期间有完成，我们在 _on_finish 时已推进过，这里只做尾段积累
            if m.busy_until > last:
                dt = min(now, m.busy_until) - last
                if dt > 0:
                    m.busy_time += dt
            m.last_update = now

    # ---------- 事件处理 ----------
    def _on_arrival(self, tk: TaskEvt):
        self.waiting.append(tk)

    def _on_finish(self, payload: Dict[str, Any]):
        name = payload["model"]
        m = self.models[name]
        # 先把 busy_time 积分到当前时刻
        self._integrate_busy(m, self.t)
        # 完成一个
        m.q_inflight = max(0, m.q_inflight - 1)
        m.busy_until = max(self.t, m.busy_until)  # 归零于现在
        m.served += 1

    # ---------- 分配逻辑（窗口竞争的离线等价：当下空闲模型共同参与） ----------
    def _greedy_assign(self):
        while self.waiting:
            idle = [m for m in self.models.values() if m.busy_until <= self.t]
            if not idle:
                break

            # 取队头任务（先进先出）
            tk = self.waiting[0]

            # 构造 queues: 用各模型的“在制数量”(inflight) 近似排队长度
            queues = {m.name: m.q_inflight for m in self.models.values()}
            models_cap = {m.name: m.cap for m in self.models.values()}

            try:
                chosen, score, _detail = pick_model(
                    task_category=tk.category,
                    task_difficulty=tk.difficulty,
                    domain_index=self.domain_index,
                    models_cap=models_cap,
                    queues=queues,
                    k=self.k,
                    shadow_mode=self.shadow_mode,
                    shadow_alpha=self.shadow_alpha,
                )
            except Exception:
                # 如果该任务类别没人能处理，直接丢弃/或入冷队列
                self.waiting.pop(0)
                continue

            m = self.models[chosen]
            if m.busy_until > self.t:
                # 刚刚被别人拿走，下一轮再试
                break

            # 分配：记录、更新q、安排完成事件
            self.waiting.pop(0)
            t_start = self.t
            runtime = self.service.draw(chosen, tk.category)
            t_finish = t_start + runtime

            # busy_time 积分：从 t_start 到 t_finish 算一次
            self._integrate_busy(m, t_start)
            m.busy_until = t_finish
            m.q_inflight += 1
            self.push_evt(t_finish, "finish", {"model": chosen})

            self.records.append(AssignRecord(
                tid=tk.tid, category=tk.category, difficulty=tk.difficulty,
                chosen=chosen, score=score, t_arrival=tk.t_arrival,
                t_start=t_start, t_finish=t_finish,
                wait=t_start - tk.t_arrival, runtime=runtime
            ))

    # ---------- 统计 ----------
    def _summarize(self, t_end: float) -> Dict[str, Any]:
        waits = [r.wait for r in self.records]
        sojs  = [r.t_finish - r.t_arrival for r in self.records]  # 系统逗留时间
        per_model_served = {m.name: m.served for m in self.models.values()}
        per_model_busy   = {m.name: m.busy_time for m in self.models.values()}

        # Jain’s fairness (按完成量)
        xs = list(per_model_served.values())
        jain = (sum(xs)**2) / (len(xs) * sum(x*x for x in xs)) if xs and sum(x*x for x in xs)>0 else 1.0

        # 队列/负载的方差（结束时刻的在制数）
        q_vec = [m.q_inflight for m in self.models.values()]
        q_var = statistics.pvariance(q_vec) if len(q_vec) >= 2 else 0.0

        util = {m.name: (per_model_busy[m.name] / max(t_end, 1e-6)) for m in self.models.values()}

        def p(vs, q):
            if not vs:
                return 0.0
            i = max(0, min(len(vs)-1, int(math.ceil(q/100.0*len(vs))-1)))
            return sorted(vs)[i]

        summary = {
            "tasks_total": len(self.records),
            "fairness_jain": jain,
            "queue_variance_end": q_var,
            "wait_p50": p(waits, 50), "wait_p90": p(waits, 90), "wait_p99": p(waits, 99),
            "soj_p50":  p(sojs,  50), "soj_p90":  p(sojs,  90), "soj_p99":  p(sojs,  99),
            "per_model_served": per_model_served,
            "per_model_busy": per_model_busy,
            "per_model_util": util,
            "t_end": t_end,
        }
        return {"summary": summary, "records": [r.__dict__ for r in self.records]}
