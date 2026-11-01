"""Part B (tasks 1-3): RFM table, K-means model selection, segment profile, PCA.
Run: python src/part_b_segments.py  -> outputs/customer_segments.csv, outputs/part_b_segments_report.txt"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = o[o["status"] == "completed"]

# ---- task 1: RFM
rfm = done.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum"))
rfm["recency"] = (SNAPSHOT - rfm["last"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
say("=== RFM ===")
say(f"customers={len(rfm)}; recency median={rfm.recency.median():.0f} min={rfm.recency.min()} max={rfm.recency.max()}; "
    f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}; monetary mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}")

# ---- task 2: K-means for K = 2..8
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
tab = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
say("\n=== K-MEANS (standardised RFM) ===")
say(tab.round(3).to_string(index=False))

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---- task 3: profile + naming (names are assigned from the profile, never from the arbitrary label number)
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
premium = prof["monetary"].idxmax()
rest = prof.drop(index=premium)
loyal = rest["frequency"].idxmax()
rest2 = rest.drop(index=loyal)
lapsing = rest2["recency"].idxmax()
names = {premium: "Premium Big Spenders", loyal: "Loyal Regulars", lapsing: "Lapsing / At-Risk"}
for s in prof.index:
    names.setdefault(s, "Occasional Low-Value")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium
prof["segment_name"] = prof.index.map(names)
say(f"\n=== SEGMENT PROFILE (K={FINAL_K}) ===")
say(prof.to_string())
say("premium segment id:", premium, "| members:", int(rfm["is_premium"].sum()))

pca = PCA(n_components=2).fit(X)
say(f"\nPCA explained_variance_ratio_ = {pca.explained_variance_ratio_.round(3).tolist()} total={pca.explained_variance_ratio_.sum():.3f}")

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)
open("outputs/part_b_segments_report.txt", "w").write("\n".join(LOG) + "\n")
