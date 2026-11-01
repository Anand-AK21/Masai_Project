"""Part D runner: RAG evaluation, the seven agent scenarios, stop-safely demos, crash-and-resume demo.
Run: python src/part_d_run.py  -> outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/, outputs/part_d_report.txt"""
import glob, json, os, shutil, sys
from collections import Counter
import pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "src")
import agent
from rag import PolicyIndex, answer, chunk_words, load_docs

REPORT = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); REPORT.append(s)

# ------------------------------------------------------------------ RAG tasks 1-4
say("=== CHUNKING (40 words, overlap 10) ===")
idx = PolicyIndex(size=40, overlap=10)
say("chunks:", len(idx.chunks), dict(Counter(c["source"] for c in idx.chunks)))
say("top-3 sources for 'refund policy late delivery':", [h["source"] for h in idx.retrieve("refund policy late delivery")])

def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = 0; fails = []
    for r in ev.itertuples():
        ans, srcs = answer(r.question, index)
        if r.expected_doc == "NONE":
            fb_ok += ans.startswith("I could not find"); continue
        d, f = r.expected_doc in srcs, r.expected_fact in ans
        doc_ok += d; fact_ok += f
        if not (d and f):
            fails.append((r.question, d, f, ans))
    say(f"\n--- {label}: chunks={len(index.chunks)} | expected_doc in top-3: {doc_ok}/12 | expected_fact in answer: {fact_ok}/12 | fallback on unanswerable: {fb_ok}/2")
    for q, d, f, a in fails:
        say(f"   FAILS: '{q}' (doc retrieved={d}, fact in answer={f}) -> answer was: {a[:110]!r}")
    return fails
say("\n=== RETRIEVAL + GROUNDING EVALUATION ===")
evaluate(idx, "overlap 10")
fails0 = evaluate(PolicyIndex(size=40, overlap=0), "overlap 0")
assert fails0, "expected at least one failure with zero overlap"

# ------------------------------------------------------------------ agent: seven scenarios
for p in (agent.LEDGER, "outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl"):
    if os.path.exists(p): os.remove(p)
shutil.rmtree(agent.STATE_DIR, ignore_errors=True)
sc = pd.read_csv("data/raw/agent_scenarios.csv")
say("\n=== SEVEN AGENT SCENARIOS (token_limit=6000, max_iterations=6) ===")
rows = []
with open("outputs/agent_runs.jsonl", "a") as f:
    for r in sc.itertuples():
        res = agent.run_agent(r.message, r.scenario_id, run_id=r.scenario_id)
        f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
        exp_ok = res["status"].startswith(r.expected_outcome) and abs(res["refund_inr"] - r.expected_refund_inr) <= 0.01
        rows.append((r.scenario_id, res["status"], f"{res['refund_inr']:.2f}", res["llm_calls"], res["tokens_used"], r.expected_outcome, "yes" if exp_ok else "NO"))
tab = pd.DataFrame(rows, columns=["scenario", "status", "refund_inr", "model_calls", "tokens_used", "expected", "matches"])
say(tab.to_string(index=False))
assert (tab["matches"] == "yes").all()

# ------------------------------------------------------------------ stop-safely demos
s1 = sc.iloc[0]
demos = []
d1 = agent.run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-demo")
d2 = agent.run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
say("\n=== STOP-SAFELY DEMOS ===")
say(f"S1 with token_limit=200      -> {d1['status']} (model calls={d1['llm_calls']}, tokens={d1['tokens_used']}, tools executed={d1['tools_executed']})")
say(f"   handoff: {d1['final_answer'][:140]}...")
say(f"S1 with time_limit_seconds=0 -> {d2['status']} (model calls={d2['llm_calls']}, tools executed={d2['tools_executed']})")
s7 = [json.loads(l) for l in open("outputs/agent_runs.jsonl")][6]
say(f"S7 with default limits       -> {s7['status']} (model calls={s7['llm_calls']}, tokens={s7['tokens_used']})")

# ------------------------------------------------------------------ crash and resume
say("\n=== CRASH AND RESUME (S1) ===")
try:
    agent.run_agent(s1.message, "S1", run_id="S1-crash-demo", crash_after=2)
    raise SystemExit("crash did not happen")
except agent.SimulatedCrash as e:
    ck = json.load(open("state/S1-crash-demo.json"))
    say(f"crash raised: {e}; checkpoint holds tools_executed={ck['tools_executed']}, tokens_used={ck['tokens_used']}, step={ck['step']}")
rs = agent.resume_agent("S1-crash-demo")
say(f"resumed run status={rs['status']} refund=INR {rs['refund_inr']:.2f}; tools executed across both parts: {rs['tools_executed']}")
assert rs["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert rs["status"] == "REFUND" and abs(rs["refund_inr"] - 128.25) < 0.01
with open("outputs/agent_demos.jsonl", "a") as f:
    for name, r in (("token_limit_200", d1), ("timeout_0", d2), ("crash_resume", rs)):
        f.write(json.dumps({"demo": name, **{k: r[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "tools_executed")}}) + "\n")

led = [json.loads(l) for l in open(agent.LEDGER)]
say(f"\nledger lines: {len(led)} -> {[(e['tool'], e['order_id'], e['refund_inr']) for e in led]}")
say("checkpoint files in state/:", sorted(os.path.basename(p) for p in glob.glob('state/*.json')))
open("outputs/part_d_report.txt", "w").write("\n".join(REPORT) + "\n")
