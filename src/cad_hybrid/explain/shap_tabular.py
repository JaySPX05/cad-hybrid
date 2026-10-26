"""SHAP explanations for the lab branch: which values pushed a patient's risk up or down.

What is explained is the saved model's *calibrated probability* (the number the app
shows), so each patient's contributions add up exactly to:

    probability = baseline + sum(contributions)

where the baseline is the model's average probability over the training data
(approximated by the background). A contribution of +0.12 means "this value raised
the risk by 12 percentage points compared with an average patient".

The final model is a stack that includes TabICL, which isn't a tree model, so a
model-agnostic method is used (Kernel SHAP). It needs many model calls per patient:
roughly ``nsamples x background_size`` rows. TabICL is slow on CPU, so both numbers are
set in ``configs/tabular.yaml`` (``explain``) and kept small.

The background is a few k-means cluster centres of the training data, weighted by
cluster size, not real patients, so no patient rows are stored in the model file for
explanations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..tabular.artifact import predict_calibrated


def make_background(X: pd.DataFrame, categorical: list[str], size: int = 10,
                    seed: int = 42) -> dict:
    """Summarise the training data as ``size`` typical (synthetic) patients.

    Missing values are filled with the median, features are standardised so large
    units (platelets) don't dominate, k-means finds ``size`` cluster centres, and the
    centres are converted back to original units. Binary features are rounded to 0/1.

    Returns ``{"centres": DataFrame, "weights": array}``; weights are the share of
    training patients in each cluster.
    """
    from sklearn.cluster import KMeans

    filled = X.fillna(X.median(numeric_only=True))
    mean, std = filled.mean(), filled.std(ddof=0).replace(0, 1)
    km = KMeans(n_clusters=min(size, len(filled)), n_init=10, random_state=seed)
    km.fit(((filled - mean) / std).to_numpy())
    centres = pd.DataFrame(km.cluster_centers_, columns=X.columns) * std + mean
    for col in categorical:
        if col in centres:
            centres[col] = centres[col].round().clip(0, 1)
    weights = np.bincount(km.labels_, minlength=len(centres)).astype(float)
    return {"centres": centres.astype(float), "weights": weights / weights.sum()}


def _probability_function(artifact: dict):
    features = artifact["features"]

    def f(rows: np.ndarray) -> np.ndarray:
        return predict_calibrated(artifact, pd.DataFrame(rows, columns=features))

    return f


def shap_values(artifact: dict, X: pd.DataFrame, nsamples: int | None = None,
                seed: int = 42) -> tuple[np.ndarray, float]:
    """SHAP values (one row per patient, one column per feature) and the baseline.

    Uses the background stored in the artifact (``explain_background``). Kernel SHAP
    samples feature combinations, so results vary slightly with ``nsamples``; ``seed``
    makes them repeatable.
    """
    import shap
    from shap.utils._legacy import DenseData  # weighted background (as shap.kmeans makes)

    background = artifact.get("explain_background")
    if background is None:
        raise ValueError("This lab model has no explanation background. "
                         "Add one with: python scripts/explain_tabular.py")
    features = artifact["features"]
    nsamples = nsamples or artifact.get("explain_nsamples") or 200
    data = DenseData(background["centres"][features].to_numpy(dtype=float), features,
                     None, np.asarray(background["weights"], dtype=float).copy())
    explainer = shap.KernelExplainer(_probability_function(artifact), data)
    state = np.random.get_state()
    np.random.seed(seed)  # Kernel SHAP samples with numpy's global generator
    try:
        values = explainer.shap_values(X[features].to_numpy(dtype=float), nsamples=nsamples,
                                       l1_reg=False, silent=True)
    finally:
        np.random.set_state(state)
    return np.asarray(values).reshape(len(X), len(features)), float(explainer.expected_value)


def explain_patient(artifact: dict, X: pd.DataFrame, nsamples: int | None = None) -> dict:
    """Contributions for one patient (one-row ``X``), largest effect first.

    Returns ``{"baseline", "probability", "contributions"}``, where contributions is a
    DataFrame with columns feature, value, contribution, given (False = value was blank
    or out of range, so the training median was used).
    """
    from ..data.tabular_loader import apply_plausible_ranges

    values, baseline = shap_values(artifact, X, nsamples)
    shown = apply_plausible_ranges(X[artifact["features"]], artifact["plausible_ranges"])
    table = pd.DataFrame({
        "feature": artifact["features"],
        "value": shown.iloc[0].to_numpy(),
        "contribution": values[0],
    })
    table["given"] = table["value"].notna()
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    return {
        "baseline": baseline,
        "probability": baseline + float(values[0].sum()),
        "contributions": table.reset_index(drop=True),
    }


def global_importance(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Average size of each feature's effect across patients, and its usual direction.

    ``direction`` is the rank correlation between a feature's value and its SHAP value:
    positive = higher values raise risk, negative = higher values lower it.
    """
    rows = []
    for i, feature in enumerate(X.columns):
        x, s = X[feature], pd.Series(values[:, i], index=X.index)
        ok = x.notna()
        corr = x[ok].corr(s[ok], method="spearman") if x[ok].nunique() > 1 else np.nan
        rows.append({"feature": feature, "mean_abs_shap": float(np.abs(values[:, i]).mean()),
                     "direction": float(corr) if pd.notna(corr) else np.nan})
    return (pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def direction_label(corr: float) -> str:
    if pd.isna(corr) or abs(corr) < 0.2:
        return "mixed / weak"
    return "higher value -> higher risk" if corr > 0 else "higher value -> lower risk"


# ---------- figure and report ----------

RAISES, LOWERS = "#e34948", "#2a78d6"   # red / blue diverging pair (validated)
INK, MUTED = "#0b0b0b", "#52514e"


def plot_importance(importance: pd.DataFrame, path: Path, top: int = 15) -> Path:
    """Horizontal bars of mean |SHAP|, coloured by usual direction, values labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = importance.head(top).iloc[::-1]
    colours = [RAISES if d > 0.2 else LOWERS if d < -0.2 else "#a3a29d"
               for d in data["direction"].fillna(0)]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(data) + 1.2), dpi=150)
    ax.barh(data["feature"], data["mean_abs_shap"] * 100, color=colours, height=0.6)
    for y, v in enumerate(data["mean_abs_shap"] * 100):
        ax.text(v, y, f" {v:.1f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Average effect on predicted risk (percentage points)", color=MUTED)
    ax.set_title("Lab model: which values matter most", loc="left", color=INK, fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RAISES, label="higher value -> higher risk"),
                       Patch(color=LOWERS, label="higher value -> lower risk"),
                       Patch(color="#a3a29d", label="mixed / weak")],
              loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def plot_patient(explanation: dict, top: int = 10):
    """Bar chart of one patient's largest contributions (red raises risk, blue lowers it).

    Returns a matplotlib figure; blank fields are labelled "(not given)".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = explanation["contributions"].head(top).iloc[::-1]
    labels = [f"{r.feature} = {r.value:g}" if r.given else f"{r.feature} (not given)"
              for r in data.itertuples()]
    pp = data["contribution"] * 100
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(data) + 1.3), dpi=150)
    ax.barh(labels, pp, color=[RAISES if v > 0 else LOWERS for v in pp], height=0.6)
    for y, v in enumerate(pp):
        ax.text(v, y, f" {v:+.1f} " if v >= 0 else f" {v:+.1f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    lim = max(abs(pp).max() * 1.3, 1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Effect on predicted risk (percentage points): "
                  "red raises it, blue lowers it", color=MUTED)
    ax.set_title(f"Average risk {explanation['baseline']:.0%} -> this patient "
                 f"{explanation['probability']:.0%}", loc="left", color=INK, fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def write_report(importance: pd.DataFrame, path: Path, model_name: str, dataset_id: str,
                 n_explained: int, settings: dict, figure: str | None, run_date: str) -> Path:
    """Markdown table of feature importance with plain-language direction."""
    lines = [
        "# Lab branch: what the model relies on (SHAP)",
        "",
        f"Model: `{model_name}`. Dataset: `{dataset_id}`. Patients explained: {n_explained} "
        f"(random sample, stratified by outcome). Run date: {run_date}.",
        f"Method: Kernel SHAP on the calibrated probability; background of "
        f"{settings['background_size']} weighted k-means centres; {settings['nsamples']} "
        "samples per patient.",
        "",
    ]
    if figure:
        lines += [f"![Feature importance]({figure})", ""]
    lines += [
        "| Rank | Feature | Average effect (percentage points) | Usual direction |",
        "|---|---|---|---|",
    ]
    for rank, row in importance.iterrows():
        lines.append(f"| {rank + 1} | {row['feature']} | {row['mean_abs_shap'] * 100:.1f} "
                     f"| {direction_label(row['direction'])} |")
    lines += [
        "",
        "- Average effect: how far, on average, a feature moves a patient's predicted risk "
        "away from the baseline (the model's average predicted risk).",
        "- These describe the model, not causes of disease. Directions that contradict "
        "clinical knowledge (for example higher blood pressure -> lower risk) or heavy "
        "reliance on values like hemoglobin usually reflect how this hospital dataset was "
        "collected (very sick admissions vs relatively healthy patients), not biology; see "
        "`reports/data_audit_tabular.md`. Check this table again on CARRS data.",
        "- Blank or out-of-range values are filled with the training median before "
        "prediction, so they still get a (usually small) contribution.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
