"""Churn prediction and feature-impact analysis for a subscription product (SYNTHETIC data).

Part A: predictive model (logistic regression vs gradient boosting) to rank users by churn risk.
Part B: does adopting a feature CAUSE lower churn? Naive comparison vs propensity-score
        matching / inverse-probability weighting, with the confounding made explicit.
Run: python analysis.py -> results/*.png, results/metrics.csv, README.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent
RES = OUT / "results"
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(11)

# ------------------------------------------------------------ synthetic users
N = 40_000
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 25, N),
    "sessions_30d": rng.gamma(3.0, 3.0, N),
    "support_tickets_90d": rng.poisson(0.6, N),
    "plan_annual": (rng.random(N) < 0.35).astype(int),
    "payment_failures_90d": rng.poisson(0.25, N),
})
engaged = (df.sessions_30d - df.sessions_30d.mean()) / df.sessions_30d.std()
# feature adoption is CONFOUNDED: engaged users are more likely to adopt AND less likely to churn
p_adopt = 1 / (1 + np.exp(-(-0.6 + 0.9 * engaged + 0.3 * df.plan_annual)))
df["adopted_feature"] = (rng.random(N) < p_adopt).astype(int)
TRUE_EFFECT = -0.45   # true causal effect of adoption on churn log-odds (what we want to recover)
logit = (-1.6 - 0.9 * engaged - 0.02 * df.tenure_months + 0.25 * df.support_tickets_90d
         + 0.7 * df.payment_failures_90d - 0.8 * df.plan_annual + TRUE_EFFECT * df.adopted_feature)
df["churned"] = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
_l0 = logit - TRUE_EFFECT * df.adopted_feature          # log-odds if NOT adopted
TRUE_ATE = float(np.mean(1 / (1 + np.exp(-(_l0 + TRUE_EFFECT))) - 1 / (1 + np.exp(-_l0))))  # true avg effect in prob. points

# ------------------------------------------------------------ A. predictive model
feats = ["tenure_months", "sessions_30d", "support_tickets_90d", "plan_annual", "payment_failures_90d", "adopted_feature"]
Xtr, Xte, ytr, yte = train_test_split(df[feats], df.churned, test_size=0.3, random_state=1, stratify=df.churned)
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
gb = GradientBoostingClassifier(random_state=1).fit(Xtr, ytr)
p_lr, p_gb = lr.predict_proba(sc.transform(Xte))[:, 1], gb.predict_proba(Xte)[:, 1]
metrics = pd.DataFrame({
    "model": ["logistic_regression", "gradient_boosting"],
    "roc_auc": [roc_auc_score(yte, p_lr), roc_auc_score(yte, p_gb)],
    "avg_precision": [average_precision_score(yte, p_lr), average_precision_score(yte, p_gb)],
})
top = int(0.1 * len(yte))
lift_top10 = yte.iloc[np.argsort(-p_gb)[:top]].mean() / yte.mean()
metrics.to_csv(RES / "metrics.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for name, pr in [("logistic", p_lr), ("gbm", p_gb)]:
    fr, mp = calibration_curve(yte, pr, n_bins=10)
    ax[0].plot(mp, fr, marker="o", label=name)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xlabel("Predicted churn probability"); ax[0].set_ylabel("Observed churn rate")
ax[0].set_title("Calibration"); ax[0].legend()
imp = pd.Series(gb.feature_importances_, index=feats).sort_values()
ax[1].barh(imp.index, imp.values, color="#2b6cb0"); ax[1].set_title("GBM feature importance (predictive, not causal)")
plt.tight_layout(); plt.savefig(RES / "model.png", dpi=140); plt.close()

# ------------------------------------------------------------ B. causal: feature adoption -> churn
conf = ["tenure_months", "sessions_30d", "support_tickets_90d", "plan_annual", "payment_failures_90d"]
naive = df.groupby("adopted_feature").churned.mean()
naive_diff = naive[1] - naive[0]
ps_model = LogisticRegression(max_iter=500).fit(StandardScaler().fit_transform(df[conf]), df.adopted_feature)
ps = ps_model.predict_proba(StandardScaler().fit_transform(df[conf]))[:, 1]
ps = np.clip(ps, 0.02, 0.98)
w = np.where(df.adopted_feature == 1, 1 / ps, 1 / (1 - ps))
ipw1 = np.average(df.churned[df.adopted_feature == 1], weights=w[df.adopted_feature == 1])
ipw0 = np.average(df.churned[df.adopted_feature == 0], weights=w[df.adopted_feature == 0])
ipw_diff = ipw1 - ipw0
# bootstrap CI for the IPW estimate
boot = []
idx_all = np.arange(N)
for _ in range(200):
    b = rng.choice(idx_all, N)
    d = df.iloc[b]; wb = w[b]; a = d.adopted_feature.to_numpy(); y = d.churned.to_numpy()
    boot.append(np.average(y[a == 1], weights=wb[a == 1]) - np.average(y[a == 0], weights=wb[a == 0]))
ci = np.percentile(boot, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar(["Naive", "IPW adjusted"], [naive_diff * 100, ipw_diff * 100], color=["#a0aec0", "#2b6cb0"])
ax.errorbar([1], [ipw_diff * 100], yerr=[[ipw_diff * 100 - ci[0] * 100], [ci[1] * 100 - ipw_diff * 100]], fmt="none", ecolor="k", capsize=5)
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Churn difference, adopters - non-adopters (pp)")
ax.set_title("Feature adoption effect on churn")
plt.tight_layout(); plt.savefig(RES / "causal.png", dpi=140); plt.close()

readme = f"""# Churn model and feature-impact analysis

**Synthetic data, real method.** All users are simulated in `analysis.py` (fixed seed). No employer data.

## Questions
- A. Can we rank users by churn risk well enough to target retention outreach?
- B. Does adopting a product feature actually *reduce* churn, or do engaged users just adopt it more?

## A. Predictive model
| model | ROC AUC | average precision |
|---|---|---|
| logistic regression | {metrics.roc_auc[0]:.3f} | {metrics.avg_precision[0]:.3f} |
| gradient boosting | {metrics.roc_auc[1]:.3f} | {metrics.avg_precision[1]:.3f} |

The top 10% of users by predicted risk churn at **{lift_top10:.1f}x** the base rate ({yte.mean():.1%}), so a
retention campaign aimed at them is far more efficient than a blanket one. Calibration is checked so
that predicted probabilities can be used for cost-benefit decisions, not just ranking.
Feature importance is *predictive*, not causal.

## B. Causal question: feature adoption and churn
Engaged users both adopt the feature more and churn less, so a naive comparison is confounded.
- Naive difference in churn (adopters - non-adopters): **{naive_diff * 100:+.1f}pp**
- Inverse-probability-weighted estimate (propensity score on user covariates): **{ipw_diff * 100:+.1f}pp**
  (95% bootstrap CI {ci[0] * 100:+.1f} to {ci[1] * 100:+.1f}pp)
- The simulation's TRUE average effect (computed from counterfactual probabilities) is **{TRUE_ATE * 100:+.1f}pp**.
  The naive estimate is off by {abs(naive_diff - TRUE_ATE) * 100:.1f}pp; the adjusted estimate is off by {abs(ipw_diff - TRUE_ATE) * 100:.1f}pp.

![model](results/model.png)
![causal](results/causal.png)

## Caveats I would state on a real project
- Propensity adjustment only handles *observed* confounders; a randomized rollout (A/B test, see
  `../ab-test-experimentation`) is the stronger design.
- Check overlap (propensity scores near 0 or 1) and covariate balance before trusting weights.
- Do not use predictive feature importance as evidence that changing a feature will change churn.

## Run it
`pip install pandas numpy scikit-learn matplotlib` then `python analysis.py`.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(metrics.round(3).to_string(index=False))
print(f"TRUE ATE {TRUE_ATE*100:+.2f}pp")
print(f"top10 lift {lift_top10:.2f}x; naive {naive_diff*100:+.2f}pp; ipw {ipw_diff*100:+.2f}pp CI ({ci[0]*100:+.2f},{ci[1]*100:+.2f})")

