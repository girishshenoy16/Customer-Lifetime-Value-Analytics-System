import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from typing import Tuple, List

# Utility: convert fig -> png bytes (for download)
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf.getvalue()


# 1) Prediction distribution overlay
def prediction_distribution_overlay(baseline_preds: np.ndarray, current_preds: np.ndarray, nbins: int = 50):
    """
    Returns plotly figure showing baseline vs current predicted CLV distributions.
    """
    baseline = pd.DataFrame({"pred": baseline_preds, "set": "baseline"})
    current = pd.DataFrame({"pred": current_preds, "set": "current"})
    df = pd.concat([baseline, current], axis=0)

    # Log scale helpful for skewed CLV — show toggle at UI level if desired
    fig = px.histogram(
        df,
        x="pred",
        color="set",
        barmode="overlay",
        nbins=nbins,
        marginal="rug",
        histnorm="probability density",
        title="Predicted CLV Distribution — Baseline vs Current",
        labels={"pred": "Predicted CLV (₹)"},
    )
    fig.update_traces(opacity=0.6)
    fig.update_layout(legend=dict(title="Dataset"), template="plotly_dark")
    return fig


# 2) Prediction scatter (actual vs predicted) — if y_true available
def prediction_scatter(actual: np.ndarray, predicted: np.ndarray, sample:int=2000):
    n = min(len(predicted), sample)
    idx = np.random.default_rng(42).choice(len(predicted), size=n, replace=False)
    df = pd.DataFrame({"y_true": np.take(actual, idx), "y_pred": np.take(predicted, idx)})
    fig = px.scatter(df, x="y_true", y="y_pred", opacity=0.6, trendline="ols", title="Actual vs Predicted (sample)")
    fig.add_shape(type="line",
                  x0=df["y_true"].min(), y0=df["y_true"].min(),
                  x1=df["y_true"].max(), y1=df["y_true"].max(),
                  line=dict(color="white", dash="dash"))
    fig.update_layout(template="plotly_dark")
    return fig


# 3) PSI bar chart for features
def psi_bar_chart(drift_df: pd.DataFrame):
    """
    Accepts DataFrame with columns: ['feature', 'psi', 'ks_stat', 'ks_p_value', 'drift']
    Returns horizontal bar chart sorted by psi.
    """
    if drift_df is None or drift_df.empty:
        return go.Figure()

    df = drift_df.copy()
    df = df.sort_values("psi", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["psi"],
        orientation="h",
        marker=dict(color=df["psi"], colorscale="RdYlGn_r", showscale=True),
        hovertemplate="<b>%{y}</b><br>PSI=%{x:.4f}<br>KS p=%{customdata[0]:.4f}<extra></extra>",
        customdata=df[["ks_p_value"]].values
    ))
    fig.update_layout(title="Feature PSI (higher -> more drift)", template="plotly_dark", xaxis_title="PSI")
    return fig


# 4) Feature distribution overlay (single feature)
def feature_distribution_overlay(baseline: pd.Series, current: pd.Series, feature_name: str, nbins:int=40):
    b = pd.to_numeric(baseline, errors="coerce").fillna(0)
    c = pd.to_numeric(current, errors="coerce").fillna(0)
    df = pd.DataFrame({
        feature_name: np.concatenate([b.values, c.values]),
        "set": ["baseline"] * len(b) + ["current"] * len(c)
    })
    fig = px.histogram(df, x=feature_name, color="set", barmode="overlay", nbins=nbins, histnorm="probability density", title=f"{feature_name} — Baseline vs Current")
    fig.update_traces(opacity=0.65)
    fig.update_layout(template="plotly_dark")
    return fig


# 5) PSI table -> add highlight for drift (helper)
def psi_table_with_flags(drift_df: pd.DataFrame, psi_threshold: float = 0.2, ks_p_threshold: float = 0.05):
    if drift_df is None:
        return pd.DataFrame()
    df = drift_df.copy()
    df["psi_flag"] = np.where(df["psi"] > psi_threshold, "DRIFT", "OK")
    df["ks_flag"] = np.where(df["ks_p_value"] < ks_p_threshold, "DRIFT", "OK")
    df["overall"] = np.where((df["psi_flag"]=="DRIFT") | (df["ks_flag"]=="DRIFT"), "Drift", "No Drift")
    return df[["feature","psi","ks_stat","ks_p_value","overall"]]