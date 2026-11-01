# Swiggy Instamart: Customer Intelligence & Support Copilot

A four-part, fully offline capstone: a data pipeline, customer segmentation and lapse prediction, a guarded support-ticket triage pipeline, and a policy RAG assistant plus a refund agent. All data is synthetic (produced by `generate_data.py`), all money is in INR, and every model call goes through the offline `mock_llm.py`: no API key, no network, no cost.

## Overview

**Business problem.** Leadership does not trust the raw order exports, retention cannot see who is about to stop ordering, and support is buried in free-text tickets. The Copilot answers all three in one repository, and each Part feeds the next.

| Part | What it does | Consumes | Produces |
|---|---|---|---|
| A | Audit, clean and store the raw exports; SQL and pandas analysis | `data/raw/*.csv` | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | RFM + K-means segments, PCA, lapse model | clean orders and customers | `outputs/customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv` |
| C | Ticket triage: redaction, retries, validation, budget, routing | raw tickets + segments from B | `outputs/triage_results.csv`, `triage_budget_demo.csv` |
| D | Policy RAG assistant + ReAct refund agent with safe stopping and resume | policies + database from A | `outputs/agent_runs.jsonl`, `refund_ledger.jsonl`, `state/` |

**Code layout**

| File | Purpose |
|---|---|
| `generate_data.py`, `mock_llm.py` | Supplied files, copied verbatim and unmodified |
| `src/part_a_clean.py` | Part A audit, cleaning rules, CSV and SQLite storage |
| `src/validate.py` | Asserts 15 reference checkpoints, prints `ALL CHECKPOINTS PASSED` |
| `src/part_a_sql.py` | Runs the 7 queries in `sql/queries.sql`, pandas cross-check, descriptive statistics |
| `src/part_b_segments.py` | RFM, K-means for K=2..8, segment profile and names, PCA |
| `src/part_b_lapse.py` | Lapse label and features, pipelines, grid searches, oversampling experiment, persistence, retention list |
| `src/part_c_triage.py` | Triage pipeline (guardrails, redaction, retry, validation, budget, routing) |
| `src/rag.py` | Chunking, TF-IDF retrieval, grounded policy assistant |
| `src/agent.py` | Tools, `TOOLS_SPEC`, `TOOL_MAP`, ReAct loop, limits, checkpoints, `resume_agent` |
| `src/part_d_run.py` | RAG evaluation, seven scenarios, stop-safely demos, crash-and-resume demo |
| `run_project.py` | Runs every stage above in order (optional shortcut) |
| `outputs/*_report.txt`, `outputs/part_a_audit.txt` | Plain-text logs of what each stage printed |

Environment used for every number below: Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, joblib 1.5.3, SQLite 3.45.1. Running the whole pipeline twice produced identical outputs apart from the jittered retry waits and ledger timestamps.

## Reproduce

Run from the repository root, on branch `main`:

1. `python -m venv venv`
2. `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a_clean.py`
6. `python src/validate.py` (prints `ALL CHECKPOINTS PASSED`)
7. `python src/part_a_sql.py`
8. `python src/part_b_segments.py`
9. `python src/part_b_lapse.py`
10. `python src/part_c_triage.py`
11. `python src/part_d_run.py`

Steps 4 to 11 can also be run in one go with `python run_project.py`. The order matters: Part C reads `outputs/customer_segments.csv` from step 8, and Part D reads `data/instamart.db` from step 5. `requirements.txt` was written with `pip freeze` (filtered to the five allowed libraries and their dependencies). No `.env` or key is used anywhere.

## Part A

### Audit of the raw data

Pandas 3 labels text columns `str`; older versions show `object`.

| Table | Shape | Columns with missing values (count, %) |
|---|---|---|
| customers.csv | 1,200 x 5 | age 61 (5.08%), gender 30 (2.50%); customer_id, city, signup_date 0 |
| products.csv | 40 x 4 | none |
| orders.csv | 13,084 x 9 | order_amount 53 (0.41%), delivery_minutes 2,541 (19.42%); all others 0 |

Column types: customers has age as float64 (the blanks) and everything else as text; products has price_inr as int64; orders has quantity and discount_pct as int64, delivery_minutes as float64 and order_amount as text (it holds the strings `N/A` and `error`). `read_csv` turns the 53 `N/A` entries into missing values, but the 67 `error` strings stay as text, so 120 amounts are non-numeric in total.

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| order_amount entries not convertible to a number | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and what each changed

**Customers**

| Rule | Effect |
|---|---|
| (a) city: strip spaces, title case | 476 of 1,200 values changed; 24 spellings became 6 cities |
| (b) gender: map to male / female / unknown | 950 values rewritten (e.g. `M`, `FEMALE`); 30 blanks became `unknown`; 10 spellings became 3 values |
| (c) ages outside 10-100 set to missing, then filled with the median age of the customer's city (`groupby` + `transform`) | 6 impossible ages removed; 67 ages filled in total (61 blank + 6 impossible); 0 missing afterwards, all ages 18 to 60 |
| (d) signup_date to datetime | all 1,200 parsed (2024-01-01 to 2025-03-31) |

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30.

**Orders**, applied in this order

| # | Rule | Rows / values changed | Rows left |
|---|---|---|---|
| 1 | `order_ts` to datetime | all 13,084 parsed | 13,084 |
| 2 | remove exact duplicate rows | 119 rows removed | 12,965 |
| 3 | per `order_id` keep the record with the latest `order_ts` (drops the earlier pending record) | 656 superseded rows removed | 12,309 |
| 4 | `order_amount` to numeric (bad entries become missing), then fill from `round(price_inr x quantity x (1 - discount_pct/100), 2)` | 112 amounts rebuilt from the product price (the other 8 of the 120 bad entries sat on rows removed by rules 2 and 3) | 12,309 |
| 5 | negative `delivery_minutes` to missing; for completed orders fill missing times with the median of the order's product category among completed orders (`groupby` + `transform`); cancelled orders stay missing | 105 negatives set to missing (the 106th was on a removed row); 774 completed orders filled; 1,199 cancelled orders deliberately left missing, because a cancelled order was never delivered and a filled value would be invented data | 12,309 |

Result: 12,309 rows (one per order_id); 11,110 completed and 1,199 cancelled; no pending rows; 0 missing order_amount; 0 missing delivery_minutes among completed orders and 1,199 among cancelled; sum of completed order_amount = INR 2,998,668.80.

### Storage and validation

The clean tables are in `data/clean/customers.csv`, `products.csv`, `orders.csv` and in the SQLite file `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `python src/validate.py` asserts 15 checkpoints (row counts, status counts, missing-value counts, the INR 2,998,668.80 revenue, city and gender counts, age range, database tables and database totals) and prints `ALL CHECKPOINTS PASSED`.

### SQL results (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers |
| Q2 | three-table join | 30 rows; revenues sum to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | scalar subquery | 11 products above the average catalogue price |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | two CTEs | 550 customers above the average spend of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER per city | top customer per city below; 8 rows have RANK different from ROW_NUMBER |
| Q7 | running total | 183 daily rows; last cumulative revenue INR 2,998,668.80 |

Q6 top customers: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

Ties cause the 8 differing rows because RANK() gives customers with identical total spend the same rank while ROW_NUMBER() must give every row a different number, so the second member of each tied pair gets a larger ROW_NUMBER than RANK.

### pandas cross-check

`pd.pivot_table(..., margins=True)` of completed revenue by city and category gives a grand total of INR 2,998,668.80, equal to the SQL total of Q2. Per city, named aggregation of completed order_amount:

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Distribution of completed order_amount

| Statistic | Value |
|---|---|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| 1.5 x IQR fences | lower -283.50, upper 645.30 |
| outliers below / above | 0 / 897 |

Correlation matrix (completed orders): order_amount vs discount_pct -0.095; order_amount vs quantity +0.087; order_amount vs delivery_minutes -0.003; discount_pct vs quantity -0.000; discount_pct vs delivery_minutes -0.010; quantity vs delivery_minutes -0.013. Largest in absolute value: order_amount with discount_pct at -0.095.

**Four questions (under 200 words).** *What is this data about?* Six months of quick-commerce orders (April to September 2025) from 1,175 buying customers across six cities, with 11,110 completed orders worth INR 2,998,668.80. *Distributions and statistics?* Order values are strongly right-skewed: the mean of 269.91 is double the median of 135.00, with skewness 4.61. Most baskets are small, and a few large electronics orders stretch the tail. *Outliers or anomalies?* 897 orders exceed the upper fence of 645.30 and none fall below the lower one; these are genuine high-priced items, not errors. The real anomalies were in the raw export: duplicates, pending twins, negative delivery times, bad amounts. *Relationships?* All correlations are weak. The largest is order_amount with discount_pct at -0.095, then quantity at +0.087. Correlation of this size does not establish causation; it may only reflect that discounts were handed out more often on cheap items.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19, minimum 0, maximum 180 days. Frequency: mean 9.46, maximum 33. Monetary: mean INR 2,552.06, maximum INR 14,766.55.

### K-means on standardised RFM (`KMeans(n_init=10, random_state=42)`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The elbow is at K=4: going from 3 to 4 clusters cuts inertia by 41% (1268 to 743), while 4 to 5 cuts only 17%. K=4 also has the highest silhouette score of the whole table (0.526), and it gives four groups the business can act on. The inertia at K=8 is always below that at K=2 because inertia is the sum of squared distances to the nearest centre, and more centres can only shrink it (it reaches zero when every customer is its own cluster). A lower inertia therefore cannot select K by itself; it has to be balanced against the elbow, the silhouette and interpretability.

### Segment profile (K = 4) and recommended actions

Cluster numbers are only labels with no ordering; the names below come from each segment's profile, and the Premium flag goes to the segment with the highest mean monetary value.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium Big Spenders** (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | Dedicated premium desk, early access to electronics launches, personal outreach when inactive |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards and grocery bundles; protect with on-time delivery |
| 0 | Occasional Low-Value | 436 | 28.86 | 3.77 | 1,057.62 | Coupons and basket-building nudges to raise order frequency |
| 3 | Lapsing / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Time-limited win-back offer on a low-cost channel |

### PCA

`PCA(n_components=2)` on the scaled RFM matrix: explained_variance_ratio_ = 0.634 and 0.218, total 0.852. Two components keep about 85% of the variance. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium (138 customers are Premium).

### Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with at least one completed order before the cutoff. Label `lapsed` = 1 if there is no completed order in September 2025: 443 lapsed (38.32%). Features (only orders before the cutoff plus the clean customer table): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category (ties broken alphabetically), city, age, gender, tenure in days. Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 training rows (354 lapsed, 570 not) and 232 test rows. Baseline of always predicting "not lapsed": accuracy 61.6%, lapsed-class F1 0.00.

**No-leakage statement.** No order dated 2025-09-01 or later entered any feature. The segment labels from the K-means step were not used as features. No encoder, scaler or resampler saw a test row: all are fitted inside a scikit-learn `Pipeline` on training rows only (`StandardScaler` for numeric columns, `OneHotEncoder(drop="first")` for city, gender and favourite category). A code assertion confirms no feature column contains a missing value.

The scaler fitted on the training rows learned a mean frequency of 8.1147, while the full-cohort mean of frequency is 8.0285; they differ because the scaler only sees the 924 training rows, never the 232 test rows.

### Models (5-fold CV, scoring F1)

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.6506 |
| RandomForestClassifier(n_estimators=100, random_state=42) | max_depth {3, 5, 8} x min_samples_leaf {5, 10, 20} (45 fits) | max_depth 8, min_samples_leaf 10 | 0.6811 |

Untouched test set, lapsed class (confusion matrix as [[TN, FP], [FN, TP]]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (never lapsed) | all predicted 0 | 0.000 | 0.000 | 0.000 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model: random forest.** It had the higher cross-validated F1 (0.681 against 0.651), chosen before looking at the test set, and the test F1 of 0.655 confirms it and clears the 0.55 requirement. Missing a customer who is about to lapse costs the future revenue of that customer (average spend is INR 2,552 over the six months), while wrongly sending a retention offer to a loyal customer costs only the price of a coupon, so recall deserves at least as much weight as precision. The forest finds the same share of lapsers as the logistic regression (recall 0.652) with fewer false alarms (30 against 32). If the retention team wants even more recall, the over-sampled logistic regression below is available.

### Class-balance experiment

Training set over-sampled with duplicates of lapsed rows (random_state=42) to 570 versus 570; best logistic regression refitted; test set unchanged.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic regression, original | 0.644 | 0.652 | 0.648 |
| Logistic regression, over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved most, from 0.652 to 0.854, while precision did not rise (it fell to 0.589). For the retention team this means that the balanced model catches about 85% of the customers who will lapse, at the price of contacting more customers who would have stayed anyway; that is a good trade when an offer is cheap relative to a lost customer.

### Interpretation

Five most important features of the random forest:

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.2796 | Customers who ordered often before the cutoff rarely stop |
| monetary | 0.2712 | Total spend marks the high-value customers whose loss would hurt most |
| recency | 0.1152 | A long gap since the last order warns of drifting away |
| avg_delivery | 0.0612 | Slow deliveries nudge people away |
| cancel_rate | 0.0525 | Frequent cancellations signal a poor experience |

Logistic-regression coefficients on standardised features: **frequency -1.697** (more past orders, lower lapse risk, the strongest effect); **monetary +0.283** (once frequency is held fixed, big-ticket buyers who order rarely are slightly more likely to skip a month); **recency +0.039** (a longer gap raises risk, but weakly because frequency already carries most of this signal).

### Persistence and retention list

The selected model is saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading it with `joblib.load` reproduces the original test predictions and probabilities exactly (checked in code). `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed) holds the 10 Premium customers in the test set with the highest predicted lapse probability; 1 of the 10 truly lapsed. The Premium flag comes from whole-window RFM, which includes September (the window that defines the label), so this retention list is illustrative and not a leakage-free evaluation.

## Part C

### Guardrails and redaction

4 empty tickets get status `skipped_empty` and 4 long tickets (734 estimated tokens each, above `MAX_INPUT_TOKENS = 400`) get `flagged_long`; neither is ever sent to the model. Redaction uses the three prescribed patterns in order (email, phone, order id):

| Type | Substitutions |
|---|---|
| `[EMAIL]` | 55 |
| `[PHONE]` | 91 |
| `[ORDER_ID]` | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0. The user message sent to the model is exactly the redacted text, and an assertion before every call (plus a final check over all 312 messages sent in the two comparison runs) shows that no email, phone number or order id ever reached a prompt.

### Prompts

Both contain the word CLASSIFY, the four categories and the JSON-only reply format. The zero-shot prompt is 29 estimated tokens and has no `EXAMPLES:` line. The few-shot prompt is 101 estimated tokens and has an `EXAMPLES:` line with four labelled examples (billing, delivery, technical, fraud).

### Retries and validation

Every call goes through a retry wrapper: at most 3 retries, wait = 0.05 x 2^n x uniform(0.9, 1.15) seconds, and one log line per retry (`outputs/triage_retry_log.txt`, format `RETRY ticket=T0002 retry=0 wait=0.0508s`). `mock_llm.reset()` is called before each full run. A full run over the 156 eligible tickets hit exactly 19 TransientLLMErrors, all recovered by a retry (every one was retry number 0, so each wait lies between 0.045 and 0.0575 s; observed 0.0453 to 0.0572 s); no ticket ended with a permanent error.

Answers are validated, never guessed: valid JSON with an allowed category is `classified_json`; non-JSON text is searched with `re.findall(r"[a-z]+", ...)` and accepted as `classified_keyword` only if exactly one allowed category word appears; everything else, including valid JSON with the category `shipping`, is `flagged_invalid_output` and goes to human review.

Status counts for one full run (identical for both prompts): 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output, 4 skipped_empty, 4 flagged_long.

### Zero-shot versus few-shot (no budget)

| | Zero-shot | Few-shot |
|---|---|---|
| Tickets classified | 147 | 147 |
| Accuracy over classified | 85.7% (126/147) | 100% (147/147) |
| Total tokens | 8,932 | 20,164 |
| Tokens per classified ticket | 60.8 | 137.2 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs 76.4 more tokens per ticket (126% more, 11,232 tokens in total) and removes all 21 zero-shot errors, which is about 535 extra tokens per corrected ticket. It is worth paying when a wrong label is expensive: 11 of the 34 fraud tickets were sent to billing, and the escalation policy requires fraud to reach the security team within 2 hours. It is not worth paying for low-stakes categories where a misrouted ticket is simply re-queued. Here all 21 errors sat in the two categories the zero-shot keyword order confuses with billing, so a few cheap examples fixed the whole problem.

### Token budget demo

Zero-shot run with `TOKEN_BUDGET = 6000` (budget used: 6000, prompt about 29 estimated tokens): 91 classified_json, 8 classified_keyword, 6 flagged_invalid_output, 4 skipped_empty, 4 flagged_long and **51 budget_exhausted** (those go to human review). The run used 6,023 tokens in total because the budget is checked before each call, so the last permitted call can cross the line. Saved as `outputs/triage_budget_demo.csv`.

### Routing and final output

`outputs/triage_results.csv` (164 rows) comes from the few-shot run with `TOKEN_BUDGET = 100000`, routed with `is_premium` from `outputs/customer_segments.csv` (a customer without a segment row counts as not premium; none were missing):

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

**What was rule-validated and what went to humans.** 147 tickets (135 JSON plus 12 keyword answers) passed the output rules and were routed automatically. 17 tickets went to human review: 4 empty, 4 too long and 9 whose model answer failed validation. The few-shot run reached 100% accuracy on the 147 tickets it classified, so every residual risk sits in the 17 human-reviewed ones.

## Part D

### Chunking and retrieval

40-word chunks with 10-word overlap give 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on the chunks, the query is transformed with the same vectorizer and chunks are ranked by cosine similarity; `retrieve(query, k=3)` returns dictionaries with source, score and text. For the query `refund policy late delivery` all three top chunks come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, distinctive words count most. One limitation: TF-IDF matches words, not meaning, so a question phrased as "how long until my money is back" misses a chunk that says "refunds are credited within 5 working days".

### Grounded answers and evaluation

`answer(question)` retrieves three chunks and sends the mock a system message containing POLICY ASSISTANT and a user message of the form `CONTEXT:` newline, the chunks each prefixed with `[<source file>]`, newline `QUESTION:` newline, the question. Evaluation on `policy_eval.csv`:

| Run | Chunks | expected_doc in top 3 (of 12) | expected_fact in answer (of 12) | Fallback on unanswerable (of 2) |
|---|---|---|---|---|
| overlap 10 | 15 | 12 | 12 | 2 |
| overlap 0 | 13 | 12 | 10 | 2 |

With overlap 0 two questions break. "What delivery time makes a completed order eligible for a full refund?" returns only the sentence fragment "receives a full refund of the order amount.", because the sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund..." is cut at the chunk boundary, so the fact "more than 45 minutes" sits in the neighbouring chunk. "How soon is money returned when a payment fails but is deducted?" is cut the same way, so the phrase "48 hours" falls into the next chunk. The 10-word overlap repeats the words around each boundary so that sentences stay whole in at least one chunk.

### Tools

`src/agent.py` defines five tools in `TOOLS_SPEC` (name, one-sentence description, JSON-style parameters) and the dictionary `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (the same retrieval as above), `calculate_refund` (delivered in more than 45 minutes means 100% of the amount, otherwise 0), `issue_refund` and `escalate_to_human` (each appends one JSON line with a timestamp and tool name to `outputs/refund_ledger.jsonl`). Both ledger tools are idempotent: an entry for the same tool and order id is written once, so re-running a demo or resuming a crashed run cannot pay the same refund twice. The ledger holds 3 lines: S1 issued 128.25, S2 issued 135.00, S3 escalated 1,999.00.

### Seven scenarios (`max_iterations=6`, `token_limit=6000`)

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,047 |
| S2 | REFUND | 135.00 | 5 | 1,047 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,034 |
| S4 | DENIED | 0.00 | 4 | 725 |
| S5 | DENIED | 0.00 | 3 | 427 |
| S6 | NOT_FOUND | 0.00 | 2 | 132 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,180 |

Every refund amount equals `expected_refund_inr` within 0.01 (asserted in code). Each run is one line in `outputs/agent_runs.jsonl` with scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool name, arguments and observation. S7 finishes all six model calls under the default token limit.

### Stop-safely demonstrations (S1 unless noted)

| Demo | Status | Detail |
|---|---|---|
| `token_limit=200` | FALLBACK_TOKEN_LIMIT | stopped on the 3rd model call at 444 tokens, before executing the tool it requested; tools executed: get_order, search_policy |
| `time_limit_seconds=0` | FALLBACK_TIMEOUT | stopped before the first model call (0 calls, 0 tools) |
| S7 with default limits | FALLBACK_MAX_ITERATIONS | 6 model calls, 3,180 tokens |

Every fallback returns the steps done so far and a human-handoff summary in `final_answer`. A token-limit check is applied after every model call, so even a call that would have produced a final answer stops the run once the limit is exceeded; that is the strict reading of the rule. The demo runs are logged in `outputs/agent_demos.jsonl`, separately from the seven scenario runs.

### Crash and resume (S1)

After every tool observation a checkpoint `state/<run_id>.json` is written (run id, scenario id, full message list, tokens_used, step number, tools executed so far, and the steps). A simulated crash was raised right after the second tool execution (after its checkpoint was saved); `resume_agent("S1-crash-demo")` then loaded the checkpoint and continued. Tools executed across both parts, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated). Final status REFUND with INR 128.25.

### ReAct versus plan-and-execute

My ReAct agent needed 5 model calls and 1,047 tokens for S1, 3 calls and 427 tokens for the cancelled order S5, and 2 calls and 132 tokens for the missing order S6. Each call resends the growing transcript, so tokens rise faster than calls. A plan-and-execute design asks the model for the whole step list once, so a clean refund needs about 2 calls (plan and final answer) and fewer tokens, but the plan must be written before the order is seen. The next steps depend on the observations: whether the order is completed, whether it was late, whether the amount exceeds INR 1000. An unexpected result such as NOT_FOUND is a visible error and triggers a replan, costing a third call, where ReAct simply stops in 2. A cancelled order is no error at all, so a fixed plan would carry on towards a refund unless the executor adds checks. For this refund workflow I would keep ReAct: it is at most 5 calls long, every decision moves money, and reading each observation before acting is the safer behaviour, so the extra tokens are cheap insurance.

## Limitations and Next Steps

- **Synthetic data and mock model.** All numbers describe generated data and a deterministic mock, not real customers or a real language model; accuracy such as 100% few-shot reflects the mock's keyword behaviour and would not carry over to a hosted model.
- **The retention list is illustrative.** The Premium flag is a post-hoc whole-window label, because it uses RFM over the full April to September window, which includes the September period that defines the lapse label; the list is therefore illustrative and not a leakage-free evaluation. The lapse model's own features use only orders before 2025-09-01.
- **Small test set.** The lapse test set has 232 customers, so F1 differences between the two models (0.648 against 0.655) are within noise. A larger sample or repeated cross-validation would be needed to separate them.
- **No decision-threshold tuning.** The models use the default 0.5 threshold; picking the threshold from the real cost of a missed lapser versus a wasted offer would be a better next step than over-sampling.
- **Refund agent safety.** The agent does not check that the order belongs to the customer who wrote the message, and it does not enforce the 7-day request window (the brief treats all requests as inside it). A production version needs both checks, plus a durable idempotency key for refunds.
- **Retrieval.** TF-IDF is lexical; embeddings with a real vector store would handle paraphrases better, and the evaluation set of 14 questions is small.
- **Plan-and-execute was compared on paper, not built.**
