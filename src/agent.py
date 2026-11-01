"""Part D (tasks 5-8): tools, ReAct loop, safe stopping and checkpoint/resume for the refund agent.
All model calls go through the offline mock_llm.py."""
import json, os, sqlite3, sys, time, uuid
from datetime import datetime, timezone
sys.path.insert(0, os.getcwd())
import mock_llm
from rag import PolicyIndex

DB_PATH = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
STATE_DIR = "state"
INDEX = PolicyIndex(size=40, overlap=10)          # the same retrieval built in tasks 1-3

# ------------------------------------------------------------------ tools
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": INDEX.retrieve(query, k=3)}

def calculate_refund(order_amount, delivery_minutes):
    """Refund rules: delivered in more than 45 minutes -> 100% of the amount, otherwise 0."""
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0}

def _ledger_write(tool, order_id, refund_inr):
    """Append one JSON line. Idempotent: an identical (tool, order_id) entry is never written twice,
    so re-running a demo or resuming a crashed run cannot pay the same refund twice."""
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            e = json.loads(line)
            if e["tool"] == tool and e["order_id"] == order_id:
                return {"refund_inr": e["refund_inr"]}
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger_write("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger_write("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}
TOOLS_SPEC = [
    _spec("get_order", "Look up one order in the Instamart database by its order id.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the company policy documents and return the three best matching passages.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and the delivery time in minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM_PROMPT = ("You are an AGENT that decides customer refund requests. Use the tools get_order, search_policy, "
                 "calculate_refund, issue_refund and escalate_to_human. When finished, reply with a final answer that starts "
                 "with REFUND:, ESCALATE:, DENIED: or NOT_FOUND:.")

class SimulatedCrash(Exception):
    pass

# ------------------------------------------------------------------ checkpoints
def _ckpt_path(run_id):
    return os.path.join(STATE_DIR, f"{run_id}.json")

def save_checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_ckpt_path(st["run_id"]), "w") as f:
        json.dump(st, f, indent=1, default=str)

def _refund_from(steps):
    for s in reversed(steps):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"].get("refund_inr", 0)
    return 0

def _finish(st, status, final_answer):
    st.update(status=status, final_answer=final_answer, finished=True, refund_inr=_refund_from(st["steps"]))
    save_checkpoint(st)
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final_answer,
            "refund_inr": st["refund_inr"], "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "tools_executed": st["tools_executed"]}

def _fallback(st, status, reason):
    tools = ", ".join(st["tools_executed"]) or "none"
    summary = (f"HANDOFF TO HUMAN: automated run stopped ({reason}) after {st['llm_calls']} model calls, "
               f"{len(st['steps'])} tool steps ({tools}) and {st['tokens_used']} tokens. Request: {st['user_message']}")
    return _finish(st, status, summary)

# ------------------------------------------------------------------ the ReAct loop
def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after):
    t0 = time.monotonic()
    while True:
        if st["llm_calls"] >= max_iterations:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        if time.monotonic() - t0 >= time_limit_seconds:
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:          # stop BEFORE executing the tool that was just requested
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded ({st['tokens_used']} used)")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            status = answer.split(":", 1)[0].strip().upper() if ":" in answer else "UNKNOWN"
            return _finish(st, status, answer)
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        st["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for tc in calls:
            try:
                result = TOOL_MAP[tc["name"]](**tc["arguments"])
            except Exception as e:                      # unknown tool / bad arguments become an observation
                result = {"error": f"{type(e).__name__}: {e}"}
            st["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": json.dumps(result)})
            st["steps"].append({"tool": tc["name"], "arguments": tc["arguments"], "observation": result})
            st["tools_executed"].append(tc["name"])
            st["step"] = len(st["steps"])
            save_checkpoint(st)                          # checkpoint after EVERY tool observation
            if crash_after and st["step"] == crash_after:
                raise SimulatedCrash(f"simulated crash after tool execution #{crash_after}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after=None):
    st = {"run_id": run_id or f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
          "user_message": user_message, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                      {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [], "tools_executed": [], "finished": False,
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after)

def resume_agent(run_id, crash_after=None):
    """Load state/<run_id>.json and continue exactly where the run stopped (no tool is executed twice).
    The wall-clock timer restarts on resume; model-call and token counters continue from the checkpoint."""
    st = json.load(open(_ckpt_path(run_id)))
    if st.get("finished"):
        raise ValueError(f"run {run_id} already finished with status {st['status']}")
    lim = st["limits"]
    return _loop(st, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"], crash_after)
