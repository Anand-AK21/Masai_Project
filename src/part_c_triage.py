"""Part C -- support-ticket triage with guardrails around the offline mock LLM.
Guardrails: empty/long input skip, PII redaction, retries with exponential backoff + jitter,
output validation (JSON, then keyword fallback, else human review), token budget, segment-aware routing.
Run: python src/part_c_triage.py"""
import json, os, random, re, sys, time
import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS = 400
BASE_DELAY, MAX_RETRIES = 0.05, 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
RETRY_LOG = "outputs/triage_retry_log.txt"
REPORT = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); REPORT.append(s)

# ------------------------------------------------------------------ redaction (order matters)
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
def redact(text, counts=None):
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        if counts is not None:
            counts[name] = counts.get(name, 0) + n
    return text
def has_pii(text):
    return any(p.search(text) for _, p, _ in PATTERNS)

# ------------------------------------------------------------------ prompts
JSON_RULE = ('Reply with JSON only, exactly in the form {"category": "...", "reason": "..."}. '
             'Allowed categories: billing, delivery, technical, fraud.')
ZERO_SHOT = "You CLASSIFY support tickets. " + JSON_RULE
FEW_SHOT = ("You CLASSIFY support tickets. " + JSON_RULE + "\nEXAMPLES:\n"
            'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The rider never arrived with [ORDER_ID]. -> {"category": "delivery", "reason": "late order"}\n'
            'Ticket: The app freezes on the cart page. -> {"category": "technical", "reason": "app problem"}\n'
            'Ticket: Unauthorized transaction on my card for [ORDER_ID]. -> {"category": "fraud", "reason": "unauthorized payment"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

# ------------------------------------------------------------------ resilient call
logf = open(RETRY_LOG, "w")
retry_stats = {"errors": 0, "waits": []}
def call_with_retry(messages, ticket_id):
    for retry in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            retry_stats["errors"] += 1
            if retry == MAX_RETRIES:
                raise                                   # permanent failure (never happens with this mock)
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            retry_stats["waits"].append(wait)
            logf.write(f"RETRY ticket={ticket_id} retry={retry} wait={wait:.4f}s\n")
            time.sleep(wait)

# ------------------------------------------------------------------ validation
def validate(text):
    """returns (status, category). Never guesses."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        found = {w for w in words if w in CATEGORIES}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""
    if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""

# ------------------------------------------------------------------ pipeline
def run_pipeline(tickets, system_prompt, token_budget=None):
    mock_llm.reset()                                    # fresh counters for every full run
    retry_stats["errors"] = 0; retry_stats["waits"] = []
    rows, total, sent = [], 0, []
    for t in tickets.itertuples():
        text = t.text
        row = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, true_category=t.true_category,
                   status="", category="", tokens_used=0)
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            assert not any(has_pii(m["content"]) for m in msgs), "PII reached a prompt!"
            sent.append(msgs[1]["content"])
            resp = call_with_retry(msgs, t.ticket_id)
            used = resp.usage.total_tokens
            total += used; row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), total, sent

def summarize(label, df, total):
    ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (ok["category"] == ok["true_category"]).mean() * 100
    say(f"\n--- {label} ---")
    say("status counts:", df["status"].value_counts().to_dict())
    say(f"classified={len(ok)}  accuracy={acc:.1f}% ({int((ok['category']==ok['true_category']).sum())}/{len(ok)})  "
        f"total tokens={total:,}  tokens per classified ticket={total/len(ok):.1f}")
    say(f"TransientLLMErrors={retry_stats['errors']}  retries logged={len(retry_stats['waits'])}  "
        f"wait range={min(retry_stats['waits']):.4f}-{max(retry_stats['waits']):.4f}s")
    return ok, acc

def route(df, seg):
    prem = dict(zip(seg["customer_id"], seg["is_premium"]))
    sname = dict(zip(seg["customer_id"], seg["segment_name"]))
    out = []
    for r in df.itertuples():
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            rt, pr = "human_review", "NORMAL"
        elif r.category == "fraud":
            rt, pr = "security_team", "HIGH"
        elif bool(prem.get(r.customer_id, False)):
            rt, pr = "premium_desk", "HIGH"
        else:
            rt, pr = "support_queue", "NORMAL"
        out.append(dict(ticket_id=r.ticket_id, customer_id=r.customer_id, status=r.status, category=r.category,
                        segment_name=sname.get(r.customer_id, ""), priority=pr, route=rt, tokens_used=r.tokens_used))
    return pd.DataFrame(out)

# ================================================================== main
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)
seg = pd.read_csv("outputs/customer_segments.csv")
say(f"tickets={len(tickets)}")

say("\n=== REDACTION ===")
counts = {}
red = [redact(t, counts) for t in tickets["text"]]
say("substitutions:", counts, "total =", sum(counts.values()))
left = sum(len(p.findall(t)) for t in red for _, p, _ in PATTERNS)
say(f"patterns re-applied to redacted texts find: {left}")
assert left == 0
empty = (tickets["text"].str.strip() == "").sum()
long_ = sum(est_tokens(t) > MAX_INPUT_TOKENS for t in tickets["text"])
say(f"empty tickets={empty}; long tickets={long_} (est tokens of a long ticket: {est_tokens(tickets[tickets.ticket_id.isin(tickets.loc[tickets['text'].str.len()>500,'ticket_id'])]['text'].iloc[0])})")
say(f"prompt sizes: zero-shot system = {est_tokens(ZERO_SHOT)} est tokens; few-shot system = {est_tokens(FEW_SHOT)} est tokens")

zs, zs_tok, zs_sent = run_pipeline(tickets, ZERO_SHOT)
ok_z, acc_z = summarize("ZERO-SHOT (no budget)", zs, zs_tok)
say("confusion table (rows = true category, columns = predicted), zero-shot classified tickets:")
say(pd.crosstab(ok_z["true_category"], ok_z["category"]).to_string())
fs, fs_tok, fs_sent = run_pipeline(tickets, FEW_SHOT)
ok_f, acc_f = summarize("FEW-SHOT (no budget)", fs, fs_tok)
say(f"extra cost of few-shot: {fs_tok/len(ok_f) - zs_tok/len(ok_z):.1f} tokens per ticket "
    f"({(fs_tok/zs_tok-1)*100:.0f}% more) for +{acc_f-acc_z:.1f} accuracy points")
assert all(not has_pii(s) for s in zs_sent + fs_sent)
say(f"assertion passed: none of the {len(zs_sent)+len(fs_sent)} user messages sent to the model contains an email, phone or order id")

# budget demo (zero-shot, TOKEN_BUDGET=6000)
bd, bd_tok, _ = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
route(bd, seg).to_csv("outputs/triage_budget_demo.csv", index=False)
summarize("BUDGET DEMO zero-shot TOKEN_BUDGET=6000", bd, bd_tok)
say("budget_exhausted tickets:", int((bd["status"] == "budget_exhausted").sum()))

# final run (few-shot, TOKEN_BUDGET=100000) + routing
fin, fin_tok, _ = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
res = route(fin, seg)
res.to_csv("outputs/triage_results.csv", index=False)
summarize("FINAL few-shot TOKEN_BUDGET=100000", fin, fin_tok)
say("\nrouting counts:"); say(res.groupby(["route", "priority"]).size().to_string())
say("rows in triage_results.csv:", len(res))
logf.close()
say(f"retry log lines written to {RETRY_LOG}: {sum(1 for _ in open(RETRY_LOG))} (all runs). first lines:")
say("".join(open(RETRY_LOG).readlines()[:3]).rstrip())
open("outputs/part_c_report.txt", "w").write("\n".join(REPORT) + "\n")
