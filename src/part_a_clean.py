"""Part A -- audit the raw exports, clean them with explicit rules, store them as CSV + SQLite.
Run:  python src/part_a_clean.py
Writes data/clean/{customers,products,orders}.csv, data/instamart.db and outputs/part_a_audit.txt"""
import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True); os.makedirs("outputs", exist_ok=True)
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

cust = pd.read_csv(f"{RAW}/customers.csv")
prod = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")

# ------------------------------------------------------------------ 1. AUDIT
say("=== AUDIT OF RAW TABLES ===")
for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    say(f"\n--- {name}: shape={df.shape}")
    audit = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                          "missing_pct": (df.isna().mean() * 100).round(2)})
    say(audit.to_string())

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(cust["age"], errors="coerce")
a = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status pending": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((deliv_num < 0).sum()),
    "blank delivery_minutes": int(deliv_num.isna().sum()),
    "blank age": int(age_num.isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": int(cust["city"].nunique()),
    "distinct non-blank raw gender spellings": int(cust["gender"].dropna().nunique()),
    "blank gender": int(cust["gender"].isna().sum()),
}
say("\n--- key audit figures")
for k, v in a.items():
    say(f"{k:42s}{v:>8,}")

# ------------------------------------------------------------------ 2. CLEAN CUSTOMERS
say("\n=== CLEANING CUSTOMERS ===")
c = cust.copy()
before_city = c["city"].nunique()
c["city"] = c["city"].str.strip().str.title()
say(f"(a) city: {before_city} spellings -> {c['city'].nunique()} cities")
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
say(f"(b) gender counts: {c['gender'].value_counts().to_dict()}")
c["age"] = pd.to_numeric(c["age"], errors="coerce")
bad_age = (c["age"] < 10) | (c["age"] > 100)
c.loc[bad_age, "age"] = np.nan
n_missing_age = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["age"] = c["age"].round().astype(int)
say(f"(c) ages set missing (outside 10-100): {int(bad_age.sum())}; total missing filled with city median: {n_missing_age}; "
    f"missing after: {int(c['age'].isna().sum())}; min/max age: {c['age'].min()}/{c['age'].max()}")
c["signup_date"] = pd.to_datetime(c["signup_date"])
say(f"(d) signup_date converted to datetime ({c['signup_date'].min().date()} .. {c['signup_date'].max().date()})")
say("city counts:", c["city"].value_counts().to_dict())

# ------------------------------------------------------------------ 3. CLEAN ORDERS
say("\n=== CLEANING ORDERS ===")
o = orders.copy()
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])                                    # rule 1
say(f"rule 1: order_ts -> datetime (all {n0} rows parsed)")
n = len(o); o = o.drop_duplicates(); say(f"rule 2: removed {n-len(o)} exact duplicate rows -> {len(o)} rows")
n = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")   # rule 3 (latest ts wins)
say(f"rule 3: kept latest record per order_id, dropped {n-len(o)} superseded (pending) rows -> {len(o)} rows")
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")           # rule 4
n_bad = int(o["order_amount"].isna().sum())
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
fill = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(fill)
say(f"rule 4: {n_bad} unparseable/missing amounts rebuilt from price*qty*(1-disc/100); missing after: {int(o['order_amount'].isna().sum())}")
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # rule 5
n_neg = int((o["delivery_minutes"] < 0).sum())
o.loc[o["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
miss_before = int((comp & o["delivery_minutes"].isna()).sum())
med = o[comp].groupby("category")["delivery_minutes"].transform("median")
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med)
say(f"rule 5: {n_neg} negative delivery times -> missing; {miss_before} completed orders with missing delivery "
    f"filled with category median; cancelled orders left missing ({int((~comp & o['delivery_minutes'].isna()).sum())})")
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
o["delivery_minutes"] = o["delivery_minutes"].astype(float)

done = o[o["status"] == "completed"]
say("\n--- after cleaning")
say(f"rows={len(o)}; status counts={o['status'].value_counts().to_dict()}")
say(f"missing order_amount={int(o['order_amount'].isna().sum())}; missing delivery (completed)={int(done['delivery_minutes'].isna().sum())}; "
    f"missing delivery (cancelled)={int(o.loc[o['status']=='cancelled','delivery_minutes'].isna().sum())}")
say(f"sum of completed order_amount = INR {done['order_amount'].sum():,.2f}")

# ------------------------------------------------------------------ 4. STORE
c_out = c.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    c_out.to_sql("customers", con, index=False)
    prod.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
say(f"\nstored 3 CSVs in {CLEAN}/ and 3 tables in {DB}")
with open("outputs/part_a_audit.txt", "w") as f:
    f.write("\n".join(LOG) + "\n")
