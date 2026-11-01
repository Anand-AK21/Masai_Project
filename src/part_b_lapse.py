"""Part B (tasks 4-8): lapse label, leakage-free features, pipelines, grid searches, balance experiment,
interpretation, persistence and retention list.
Run: python src/part_b_lapse.py  -> outputs/lapse_model.joblib, outputs/retention_list.csv, outputs/part_b_lapse_report.txt"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

CUTOFF = pd.Timestamp("2025-09-01"); END = pd.Timestamp("2025-10-01")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
o = o.merge(prod[["product_id", "category"]], on="product_id")

# ---------------------------------------------------------------- task 4: label + features
before = o[o["order_ts"] < CUTOFF]                      # ONLY orders before the cutoff feed the features
b_done = before[before["status"] == "completed"]
feat = b_done.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                         monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                         avg_delivery=("delivery_minutes", "mean"))
feat["recency"] = (CUTOFF - feat["last"]).dt.days
# favourite category = most frequent one; ties broken alphabetically so the result is deterministic
feat["fav_category"] = b_done.groupby("customer_id")["category"].agg(lambda s: s.value_counts().sort_index().idxmax())
n_final = before.groupby("customer_id")["order_id"].count()
n_canc = before[before["status"] == "cancelled"].groupby("customer_id")["order_id"].count()
feat["cancel_rate"] = (n_canc.reindex(feat.index).fillna(0) / n_final.reindex(feat.index)).fillna(0)
feat = feat.drop(columns="last").reset_index().merge(cust, on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
# label uses the September window only (the future relative to the cutoff)
sept_buyers = set(o[(o["status"] == "completed") & (o["order_ts"] >= CUTOFF) & (o["order_ts"] < END)]["customer_id"])
feat["lapsed"] = (~feat["customer_id"].isin(sept_buyers)).astype(int)
feat = feat.sort_values("customer_id").reset_index(drop=True)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[["customer_id"] + NUM + CAT], feat["lapsed"]
say(f"cohort={len(feat)}; lapsed={int(y.sum())} ({y.mean()*100:.2f}%)")
assert not X[NUM + CAT].isna().any().any(), "a feature column has missing values"
say("confirmed: no feature column contains a missing value")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
say(f"train={len(Xtr)} test={len(Xte)}; train lapsed={int(ytr.sum())} / not lapsed={int((ytr==0).sum())}")
ids_te = Xte["customer_id"].values
Xtr, Xte = Xtr.drop(columns="customer_id"), Xte.drop(columns="customer_id")

def pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

def metrics(model, X_, y_):
    p = model.predict(X_)
    return dict(precision=precision_score(y_, p, zero_division=0), recall=recall_score(y_, p, zero_division=0),
                f1=f1_score(y_, p, zero_division=0), cm=confusion_matrix(y_, p))

# ---------------------------------------------------------------- task 5: scaler fitted on train only
chk = pre().fit(Xtr)
say(f"\nscaler mean of frequency (train only) = {chk.named_transformers_['num'].mean_[NUM.index('frequency')]:.4f}"
    f" | full-cohort mean of frequency = {X['frequency'].mean():.4f}")
say("they differ because the scaler learns its mean from the 924 training rows only; the 232 test rows were never shown to it.")

# ---------------------------------------------------------------- task 6: baseline + grid searches
base_pred = np.zeros(len(yte), dtype=int)
say(f"\nBASELINE (always 'not lapsed'): accuracy={accuracy_score(yte, base_pred)*100:.1f}%  lapsed F1={f1_score(yte, base_pred, zero_division=0):.2f}")

lr_gs = GridSearchCV(Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=1000))]),
                     {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_gs = GridSearchCV(Pipeline([("pre", pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
say(f"\nLogReg  best params={lr_gs.best_params_}  best CV F1={lr_gs.best_score_:.4f}")
say(f"RForest best params={rf_gs.best_params_}  best CV F1={rf_gs.best_score_:.4f}")
res = {"LogisticRegression": metrics(lr_gs.best_estimator_, Xte, yte), "RandomForest": metrics(rf_gs.best_estimator_, Xte, yte)}
for n, m in res.items():
    say(f"\n{n} TEST: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")
    say(f"confusion matrix [[TN FP],[FN TP]] =\n{m['cm']}")
# model selection uses CROSS-VALIDATED F1 (not the test set), the test set only confirms
best_name = "RandomForest" if rf_gs.best_score_ >= lr_gs.best_score_ else "LogisticRegression"
best_model = rf_gs.best_estimator_ if best_name == "RandomForest" else lr_gs.best_estimator_
say(f"\nSELECTED MODEL (higher CV F1): {best_name}; test F1={res[best_name]['f1']:.3f}")
assert res[best_name]["f1"] >= 0.55

# ---------------------------------------------------------------- task 7: over-sampling experiment (train only)
rng = np.random.RandomState(42)
minority = np.where(ytr.values == 1)[0]; majority_n = int((ytr == 0).sum())
extra = rng.choice(minority, size=majority_n - len(minority), replace=True)
idx = np.concatenate([np.arange(len(ytr)), extra])
Xbal, ybal = Xtr.iloc[idx], ytr.iloc[idx]
say(f"\nOVERSAMPLING: balanced training set = {int((ybal==0).sum())} not-lapsed / {int((ybal==1).sum())} lapsed")
lr_bal = Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))]).fit(Xbal, ybal)
m_un, m_bal = res["LogisticRegression"], metrics(lr_bal, Xte, yte)
say(f"unbalanced LR : precision={m_un['precision']:.3f} recall={m_un['recall']:.3f} F1={m_un['f1']:.3f}")
say(f"balanced   LR : precision={m_bal['precision']:.3f} recall={m_bal['recall']:.3f} F1={m_bal['f1']:.3f}")

# ---------------------------------------------------------------- task 8: interpretation, persistence, retention list
names = best_model.named_steps["pre"].get_feature_names_out()
rf = rf_gs.best_estimator_.named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
say("\nTop-5 random-forest feature importances:"); say(imp.head(5).round(4).to_string())
lr_names = lr_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
say("\nLogistic-regression coefficients (standardised features):")
say(coef.loc[["num__recency", "num__frequency", "num__tenure_days", "num__monetary", "num__avg_discount"]].round(3).to_string())

joblib.dump(best_model, "outputs/lapse_model.joblib")
re_model = joblib.load("outputs/lapse_model.joblib")
same = bool((re_model.predict(Xte) == best_model.predict(Xte)).all()) and \
       bool(np.allclose(re_model.predict_proba(Xte), best_model.predict_proba(Xte)))
say(f"\nreloaded model reproduces test predictions exactly: {same}")
assert same

seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg.loc[seg["is_premium"], "customer_id"])
out = pd.DataFrame({"customer_id": ids_te, "lapse_probability": best_model.predict_proba(Xte)[:, 1].round(4),
                    "actually_lapsed": yte.values})
out = out[out["customer_id"].isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
out.to_csv("outputs/retention_list.csv", index=False)
say(f"\nretention list: {len(out)} premium test customers, {int(out['actually_lapsed'].sum())} truly lapsed")
say(out.to_string(index=False))
say("NOTE: the Premium flag uses whole-window RFM (includes September), so this list is illustrative, not a leakage-free evaluation.")
open("outputs/part_b_lapse_report.txt", "w").write("\n".join(LOG) + "\n")
