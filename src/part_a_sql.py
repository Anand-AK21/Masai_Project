"""Part A -- run the seven SQL queries, cross-check with pandas, exploratory statistics.
Run: python src/part_a_sql.py  (writes outputs/part_a_sql_report.txt)"""
import re, sqlite3
import numpy as np
import pandas as pd

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# split sql/queries.sql into its seven queries (each starts with a "-- Qn:" comment)
text = open("sql/queries.sql").read()
parts = re.split(r"(?m)^-- (Q\d):", text)[1:]
Q = {parts[i]: parts[i + 1].split("\n", 1)[1].strip().rstrip(";") for i in range(0, len(parts), 2)}
con = sqlite3.connect("data/instamart.db")
R = {k: pd.read_sql_query(v, con) for k, v in Q.items()}

say("=== SQL RESULTS ===")
say(f"Q1 customers with big-order total > 8000: {len(R['Q1'])}")
q2 = R["Q2"]
say(f"Q2 rows={len(q2)}; total revenue = INR {q2['revenue'].sum():,.2f}; largest cell = {q2.iloc[0]['city']} / {q2.iloc[0]['category']} INR {q2.iloc[0]['revenue']:,.2f}")
say(f"Q3 products above average price: {len(R['Q3'])}")
say(f"Q4 customers with no completed order: {len(R['Q4'])}")
say(f"Q5 customers above average spend: {len(R['Q5'])}; average = INR {R['Q5']['avg_total'].iloc[0]:,.2f}")
q6 = R["Q6"]
say("Q6 top customer per city (rank 1):")
for _, r in q6[q6["rnk"] == 1].iterrows():
    say(f"   {r['city']:10s} {r['customer_id']}  INR {r['total_spend']:,.2f}")
say(f"Q6 rows where RANK() != ROW_NUMBER(): {int((q6['rnk'] != q6['row_num']).sum())}")
q7 = R["Q7"]
say(f"Q7 daily rows={len(q7)}; last cumulative revenue = INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")

# ---------------- pandas cross-check
o = pd.read_csv("data/clean/orders.csv"); c = pd.read_csv("data/clean/customers.csv"); p = pd.read_csv("data/clean/products.csv")
done = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")
pt = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
say("\n=== PANDAS PIVOT (margins=True) ===")
say(pt.round(2).to_string())
say(f"pivot grand total = {pt.loc['Total','Total']:,.2f}; SQL total = {q2['revenue'].sum():,.2f}; equal: {abs(pt.loc['Total','Total']-q2['revenue'].sum())<0.01}")
g = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")).round(2)
say("\nper-city named aggregation:"); say(g.to_string())

# ---------------- EDA of completed order_amount
x = done["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
say("\n=== DISTRIBUTION OF completed order_amount ===")
say(f"mean={x.mean():.2f} median={x.median():.2f} skew={x.skew():.2f} Q1={q1:.2f} Q3={q3:.2f}")
say(f"IQR fences: lower={lo:.2f} upper={hi:.2f}; outliers below={(x<lo).sum()} above={(x>hi).sum()}")
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
say(corr.to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
best = pairs.abs().idxmax()
say(f"largest |corr| pair: {best} = {pairs[best]:.3f}")
open("outputs/part_a_sql_report.txt", "w").write("\n".join(LOG) + "\n")
