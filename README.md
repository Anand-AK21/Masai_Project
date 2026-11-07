
# Swiggy Instamart: Customer Intelligence & Support Copilot

**In one line:** A small, offline AI project that cleans messy shop data, groups customers, predicts who will stop buying, sorts support tickets safely, and handles refunds.

-All data is synthetic (made up by `generate_data.py`), all money is in Indian Rupees (INR), and every "AI model" call uses the offline stand-in `mock_llm.py`. No API key, internet connection or payment is needed.

### The Problem
Instamart delivers groceries and gadgets in minutes. Management has three problems:

1. **Nobody trusts the order data.** It has duplicates, spelling mix-ups and impossible values.
2. **Nobody knows which customers are about to leave.**
3. **Support is flooded with free-text tickets** (refunds, fraud, app crashes, late deliveries).

This project builds one tool, the Copilot, that tackles all three. It has four connected parts, and each part uses what the previous one produced:
| Part | What It Does |
| --- | --- |
| **A** | Clean data + save in database |
| **B** | Group customers + predict who will leave |
| **C** | Sort support tickets safely |
| **D** | Answer policy questions + handle refunds |

Part A must be correct first. B, C, D all use its output.

### Words To Know

| Word | Simple Meaning |
| --- | --- |
| **Cleaning** | Fixing wrong / duplicate / missing data |
| **RFM** | Recency (days since last order), Frequency (how many orders), Monetary (how much spent) |
| **Churn / Lapse** | Customer stops ordering |
| **Data Leakage** | Letting model see the answer while learning |
| **RAG** | First find policy page, then answer using only that page |
| **Redaction** | Hiding email, phone, order ID before sending to AI |
| **Agent** | AI that thinks, uses a tool, sees result, thinks again |

### How To Run

Open terminal in project folder (where `generate_data.py` is):

```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_project.py
python check_submission.py
`run_project.py` runs everything. `check_submission.py` should say `ALL GOOD`.

Always run from project root. Order matters.
---
### Part A: Cleaning Data

*What was wrong in raw data:*
- 13,084 rows, but only 12,309 unique order IDs
- 119 exact duplicates, 656 pending duplicate copies
- 120 amounts were `N/A` or `error`
- 106 negative delivery times, 2,541 missing delivery times
- 24 spellings of city, 10 spellings of gender

*How we fixed it:*
- *City:* Fix spaces + Title Case. 24 spellings became 6 cities: Pune, Delhi, Bengaluru, Chennai, Hyderabad, Mumbai
- *Gender:* Map all to `male`, `female`, `unknown`
- *Age:* Ages outside 10-100 treated as missing, then filled with median age of that city. Final ages: 18 to 60
- *Orders:* Remove duplicates, keep only latest copy per order ID. Rebuild bad amounts as `price x quantity x (1 - discount/100)`
- *Delivery time:* Negative set to missing. Missing filled with median for that category. Cancelled orders left blank on purpose, because they were never delivered.

*Final result:* 12,309 orders = 11,110 completed + 1,199 cancelled. Total completed revenue = *INR 2,998,668.80*. `validate.py` prints `ALL CHECKPOINTS PASSED`.

*7 SQL Questions:*
- Q1: Spent > 8,000 on orders > 500: *30 customers*
- Q2: Revenue by city + category: *30 rows*, total matches INR 2,998,668.80
- Q3: Products above average price: *11 products*
- Q4: Never completed an order: *25 customers*
- Q5: Spent more than average: *550 customers*, avg INR 2,552.06
- Q6: Rank in each city: Top is Mumbai C0693 at INR 14,766.55
- Q7: Daily revenue + running total: *183 days*, final INR 2,998,668.80

*Order amounts:* Mean 269.91, Median 135.00. Mean is double median, so most orders are small, few big electronics orders pull average up. 897 orders above 645.30 are real high-value orders, not errors.
---
### Part B: Sort customers into groups by behaviour (segmentation). Then we train a model that predicts which customers will stop ordering (lapse prediction).

*RFM Table:* As of 2025-10-01, completed orders only. 1,175 customers. Recency median 19 days, Frequency mean 9.46, Monetary mean INR 2,552.06.

*Groups (K-Means):* Tried K = 2 to 8. Best was *K = 4* — highest silhouette 0.526 + clear elbow.
Segment | Name | Size | Behaviour | Action
1 | Premium Big Spenders | 138 | Spend INR 7,129 avg | Premium desk, early access
2 | Loyal Regulars | 419 | Order 19 times avg | Loyalty rewards
0 | Occasional Low-Value | 436 | Order rarely, spend low | Coupons, nudges
3 | Lapsing / At-Risk | 182 | Last order 109 days ago | Win-back offer
*Predicting Lapse:*
- *Lapsed =* No completed order in September 2025
- 1,156 customers, 443 lapsed (38.32%). Split: 924 train, 232 test
- Baseline (always say not lapsed): 61.6% accuracy but F1 = 0, so useless
Model | Precision | Recall | F1
Logistic Regression | 0.644 | 0.652 | 0.648
**Random Forest (Selected)** | **0.659** | **0.652** | **0.655**
Random Forest chosen before seeing test set, because of higher CV F1. It finds same lapsers with fewer false alarms.

*If we balance data:* Recall jumps 0.652 to 0.854, precision falls to 0.589. Good if coupon is cheap and losing customer is expensive.

*Top reasons for lapsing:* Frequency (0.28), Monetary (0.27), Recency (0.11), Avg delivery time, Cancel rate.

*No leakage:* No September data in features. Scaler / encoder fitted only on training data. Segment labels not used as features.

---

### Part C: Sorting Support Tickets Safely

We never send raw tickets to AI. Safety layers:

1. *Guardrails:* Skip 4 empty tickets. Flag 4 very long tickets (>400 tokens)
2. *Hide personal data:* Email to `[EMAIL]` (55), Phone to `[PHONE]` (91), Order ID to `[ORDER_ID]` (132). Total 278. 0 leaks.
3. *Retries:* Mock AI fails every 9th call. Retry up to 3 times. All 19 failures recovered.
4. *Validate output:* Valid JSON = accept. One category word = accept as keyword. Else = send to human.

*Zero-shot vs Few-shot:*
 | Zero-shot | Few-shot
Accuracy | 85.7% (126/147) | **100% (147/147)**
Tokens per ticket | 60.8 | 137.2
Few-shot costs 126% more tokens but fixes all 21 errors. 11 of those were fraud tickets wrongly sent to billing. For fraud, that is dangerous, so few-shot is worth it.

*Budget Demo:* With 6000 token budget, 51 tickets hit `budget_exhausted` and go to human.

*Final Routing:* Security team (fraud) 34 HIGH, Premium desk 11 HIGH, Support queue 102 NORMAL, Human review 17 NORMAL.

---

### Part D: Policy Assistant + Refund Agent

*Policy Assistant (RAG):*
- 5 policy files cut into 40-word chunks with 10-word overlap = 15 chunks
- Find top 3 chunks with TF-IDF, answer using only those chunks
- Test: 12/12 correct document, 12/12 correct facts, 2/2 unanswerable correctly said "I could not find"
- Without overlap, 2 questions break because sentences get cut in half. Overlap fixes it.

*Refund Agent Tools:* `get_order`, `search_policy`, `calculate_refund` (>45 mins = 100% refund, else 0), `issue_refund` (up to INR 1000), `escalate_to_human` (above INR 1000).
Scenario | Result | Refund
S1 | REFUND | INR 128.25
S2 | REFUND | INR 135.00
S3 | ESCALATE | INR 1,999.00
S4 / S5 | DENIED | INR 0
S6 | NOT_FOUND | INR 0
S7 | FALLBACK_MAX_ITERATIONS | INR 0
*Safety:* Agent stops at max 6 steps, token limit, or time limit. Always returns hand-off summary for human.

*Crash + Resume:* Agent saves checkpoint after every tool. If it crashes after 2 tools, resume continues without repeating tools. Final result still REFUND INR 128.25.

*Why ReAct?* ReAct checks each tool result before next step. For money decisions, that is safer, even if it uses more tokens (5 calls for S1). Plan-and-execute is cheaper but must plan before seeing the order.

### Limitations

- All data is fake, mock AI is not real LLM. 100% accuracy will not happen in real life.
- Test set is small (232 customers), so small F1 differences are noise.
- Retention list uses whole-window Premium flag, so it is illustrative only.
- Agent does not check if order belongs to the person asking, or the 7-day window. Production needs both.
- TF-IDF matches words, not meaning. Embeddings would be better.

