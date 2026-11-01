"""Part A validation: asserts reference checkpoints against the clean tables and the SQLite database."""
import sqlite3
import pandas as pd

o = pd.read_csv("data/clean/orders.csv"); c = pd.read_csv("data/clean/customers.csv")
done = o[o["status"] == "completed"]
with sqlite3.connect("data/instamart.db") as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    db_rows = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    db_sum = con.execute("SELECT ROUND(SUM(order_amount),2) FROM orders WHERE status='completed'").fetchone()[0]

checks = {
    "12,309 order rows": len(o) == 12309,
    "one row per order_id": o["order_id"].is_unique,
    "11,110 completed": len(done) == 11110,
    "1,199 cancelled": int((o["status"] == "cancelled").sum()) == 1199,
    "no pending rows": int((o["status"] == "pending").sum()) == 0,
    "0 missing order_amount": int(o["order_amount"].isna().sum()) == 0,
    "0 missing delivery among completed": int(done["delivery_minutes"].isna().sum()) == 0,
    "1,199 missing delivery among cancelled": int(o.loc[o["status"] == "cancelled", "delivery_minutes"].isna().sum()) == 1199,
    "completed revenue INR 2,998,668.80": round(done["order_amount"].sum(), 2) == 2998668.80,
    "six cities": c["city"].nunique() == 6,
    "gender female/male/unknown = 610/560/30": c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages, all within 18-60": c["age"].isna().sum() == 0 and c["age"].between(18, 60).all(),
    "DB has customers/products/orders": {"customers", "products", "orders"} <= tables,
    "DB orders rows = 12,309": db_rows == 12309,
    "DB revenue = 2,998,668.80": db_sum == 2998668.80,
}
for name, ok in checks.items():
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
assert all(checks.values()), "some checkpoints failed"
print("ALL CHECKPOINTS PASSED")
