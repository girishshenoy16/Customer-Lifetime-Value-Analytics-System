import io
import sys
import time
import zipfile
import base64
import subprocess
import re
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from app.data_loader import load_transactions
from app.preprocess import preprocess_all
from app.rfm_segmentation import build_rfm_dataset, build_rfm_segments
from app.clv_model import train_clv_model
from app.cohort_analysis import build_cohort_retention

from app.retention_recommender import (
    build_clv_with_segments, 
    generate_recommendations
)

from app.model_evaluation import evaluate_clv_model

from app.model_monitoring import (
    monitor_drift, 
    prediction_drift, 
    load_baseline_snapshot, 
    safe_load_feature_store,
    save_baseline_snapshot
)

from app.monitoring_viz import (
    prediction_distribution_overlay,
    prediction_scatter,
    psi_bar_chart,
    feature_distribution_overlay,
    psi_table_with_flags,
    fig_to_png_bytes,
)

from app.model_registry import (
    list_versions, 
    get_active_version
)

from app.auto_retrain import retrain_if_needed
from app.logs_utils import read_last_lines, filter_log_lines, highlight_block
from app.logging_config import get_logger as _get_logger

from app.logging_config import (
    list_log_files, zip_all_logs,
    LOG_DIR
)

from app.logs_utils import (
    read_last_lines, filter_log_lines, 
    highlight_block
)

from app.log_analytics import load_all_logs

from app.system_health import (
    get_system_metrics, get_process_table
)

from app.drift_simulator import (
    simulate_drift, restore_backup, 
    evolve_synthetic_data
)


# =========================
# ADDED: UI LOGGER (Depth C)
# =========================
LOG = _get_logger(__name__, filename="ui.log")
# =========================


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Lifetime Value & Retention Analytics",
    layout="wide",
)

LOG.info("Streamlit app starting - page config set")


# =========================
# GLOBAL STYLES (SAFE)
# =========================
st.markdown(
    """
<style>
/* NAVBAR */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 58px;
    background: linear-gradient(90deg, #1F1F21, #2C2C2E);
    color: white;
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    z-index: 9999;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    font-family: "Segoe UI", sans-serif;
}

.nav-title { font-size: 1.1rem; font-weight: 600; }
.nav-spacer { flex-grow: 1; }

.nav-icon {
    font-size: 1.2rem;
    margin-right: 1rem;
    cursor: pointer;
}
.nav-icon:hover { transform: scale(1.1); }

.nav-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 1px solid #555;
    cursor: pointer;
}

/* Sidebar styling */
.sidebar-section-title {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 1.4rem;
    margin-bottom: 0.4rem;
    color: #e5e5e7;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 0.9rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
    color: #ddd;
    font-size: 0.88rem;
    line-height: 1.3rem;
}

.sidebar-card:hover {
    background: rgba(255,255,255,0.09);
    transition: 0.2s ease;
}

.status-good { color: #2ecc71; font-weight: 600; }
.status-bad  { color: #e74c3c; font-weight: 600; }
.status-warn { color: #f1c40f; font-weight: 600; }

/* Push main content below navbar */
.main-container {
    margin-top: 70px;
}
</style>
""",
    unsafe_allow_html=True,
)

LOG.debug("Loaded global CSS styles")


# =========================
# CACHED DATA LOADERS
# =========================
@st.cache_data(show_spinner=False)
def get_transactions() -> pd.DataFrame:
    LOG.debug("Cache miss: get_transactions() called")
    df = load_transactions()
    LOG.info(f"get_transactions loaded {len(df):,} rows")
    return df


@st.cache_data(show_spinner=False)
def get_rfm() -> pd.DataFrame:
    LOG.debug("Cache miss: get_rfm() called")
    df = get_transactions()
    rfm, _ = build_rfm_dataset(df)
    LOG.info(f"RFM prepared with {len(rfm):,} customers")
    return rfm


@st.cache_data(show_spinner=False)
def get_rfm_segments() -> pd.DataFrame:
    LOG.debug("Cache miss: get_rfm_segments() called")
    rfm = get_rfm()
    segs = build_rfm_segments(rfm)
    LOG.info(f"RFM segments ready: {segs['rfm_segment'].nunique()} segments")
    return segs


@st.cache_data(show_spinner=False)
def get_cohort_tables():
    LOG.debug("Cache miss: get_cohort_tables() called")
    df = get_transactions()
    out = build_cohort_retention(df)
    LOG.info("Cohort tables generated")
    return out


@st.cache_data(show_spinner=False)
def get_clv_with_segments() -> pd.DataFrame:
    LOG.debug("Cache miss: get_clv_with_segments() called")
    out = build_clv_with_segments()
    LOG.info(f"CLV with segments loaded: {len(out):,} rows")
    return out


@st.cache_data(show_spinner=False)
def get_recommendations() -> pd.DataFrame:
    LOG.debug("Cache miss: get_recommendations() called")
    df = get_clv_with_segments()
    out = generate_recommendations(df)
    LOG.info(f"Generated recommendations for {len(out):,} customers")
    return out


@st.cache_data(show_spinner=False)
def get_model_eval():
    LOG.debug("Cache miss: get_model_eval() called")
    out = evaluate_clv_model(shap_sample_frac=0.3)
    LOG.info("Model evaluation completed (cached)")
    return out


# =========================
# MAIN APP
# =========================
def main():
    LOG.info("Entering main()")


    # ---------- NAVBAR ----------
    st.markdown(
        """
        <div class="navbar">
            <div class="nav-title">📊 Customer Lifetime Value Analytics</div>
            <div class="nav-spacer"></div>
            <div class="nav-icon">🔔</div>
            <img class="nav-avatar" src="https://i.ibb.co/5R7V2q1/user-avatar.png" alt="User">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Wrap main content in a container to push below navbar
    st.markdown('<div class="main-container">', unsafe_allow_html=True)


    st.title("📈 Customer Lifetime Value (CLV) & Strategic Retention System")

    st.caption(
        "End-to-end CLV prediction, RFM segmentation, cohort analytics, and ROI-based retention playbook."
    )


    LOG.info("Rendered title and caption")


    # ---------- SIDEBAR ----------
    with st.sidebar:
        LOG.info("Rendering sidebar controls")

        st.markdown(
            "<div class='sidebar-section-title'>⚙️ Controls</div>",
            unsafe_allow_html=True,
        )

        # Preprocess button
        if st.button(
            "🧹 Preprocess Data (Clean + Feature Store)",
            key="preprocess_btn",
            use_container_width=True,
        ):
            LOG.info("User clicked: Preprocess Data")

            with st.spinner("Running preprocessing pipeline..."):
                try:
                    preprocess_all()
                    LOG.info("Preprocessing pipeline completed successfully")
                except Exception as e:
                    LOG.exception("Preprocessing failed", exc_info=True)
                    st.error(f"Preprocessing failed: {e}")

            st.success("Data successfully preprocessed!")
            st.cache_data.clear()


        # Train model button (handle 4 or 5-return versions safely)
        if st.button(
            "🤖 Train / Retrain CLV Model",
            key="train_btn",
            use_container_width=True,
        ):
            LOG.info("User clicked: Train / Retrain CLV Model")

            with st.spinner("Training CLV model..."):
                try:
                    result = train_clv_model()
                    LOG.info("train_clv_model() returned a result")
                except Exception as e:
                    LOG.exception("Model training failed", exc_info=True)
                    st.error(f"Model training failed: {e}")
                    result = None

            # Flexible unpacking
            mae = rmse = r2 = None


            if isinstance(result, tuple):

                if len(result) == 4:
                    model, scaler, mae, r2 = result
                    LOG.debug("Unpacked 4-tuple from train result")

                elif len(result) == 5:
                    model, scaler, mae, rmse, r2 = result
                    LOG.debug("Unpacked 5-tuple from train result")

                else:
                    model, scaler = result[0], result[1]
                    LOG.debug("Unpacked result (fallback)")


            st.success("Model trained!")

            if mae is not None:
                st.info(f"MAE: {mae:,.2f}")
                LOG.info(f"Model MAE: {mae:.4f}")

            if rmse is not None:
                st.info(f"RMSE: {rmse:,.2f}")
                LOG.info(f"Model RMSE: {rmse:.4f}")

            if r2 is not None:
                st.info(f"R²: {r2:.3f}")
                LOG.info(f"Model R2: {r2:.4f}")

            st.cache_data.clear()


            # -------------------------
            # AUTO-GENERATE REPORTS
            # -------------------------
            try:
                gen_script = Path("app/generate_reports.py")
                subprocess.run(
                    [sys.executable, str(gen_script)], 
                    capture_output=True, text=True
                )
                st.success("📄 Reports regenerated automatically.")
            except Exception as e:
                st.error(f"❌ Could not regenerate reports: {e}")


        st.markdown("---")

        
        st.markdown("### 📝 Reports Generator")


        if st.button("📄 Generate All Reports", key="regen_reports"):
            try:
                st.info("Generating all reports… Please wait.")

                # Path to the reporting script
                gen_script = Path("app/generate_reports.py")

                # Run the script with the same Python environment
                result = subprocess.run(
                    [sys.executable, str(gen_script)],
                    capture_output=True,
                    text=True
                )


                if result.returncode == 0:
                    st.success("Reports regenerated successfully! Check the /reports folder.")
                    st.text(result.stdout[-1000:])  # Show last part of logs
                else:
                    st.error("Failed to generate reports.")
                    st.error(result.stderr)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            

            # ---------------------------
            # 📦 DOWNLOAD ALL REPORTS ZIP
            # ---------------------------
            st.markdown("### 📦 Export Reports")


            def zip_reports():
                reports_dir = Path("reports")
                mem_zip = io.BytesIO()

                with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in reports_dir.rglob("*"):
                        if file.is_file():
                            zf.write(file, file.relative_to(reports_dir))
                mem_zip.seek(0)
                return mem_zip


            if st.button("⬇ Download All Reports (ZIP)", key="download_reports"):
                reports_zip = zip_reports()
                st.download_button(
                    "Download Reports ZIP",
                    data=reports_zip,
                    file_name="reports_bundle.zip",
                    mime="application/zip"
                )



        st.markdown("---")

        # Pipeline status
        st.markdown(
            "<div class='sidebar-section-title'>📌 Pipeline Status</div>",
            unsafe_allow_html=True,
        )

        feature_store_exists = Path("data/processed/feature_store.csv").exists()
        model_exists = Path("models/clv_xgboost.pkl").exists()
        tx_exists = Path("data/raw/customers_transactions.csv").exists()


        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)


        st.markdown(
            f"Feature Store: <span class='{'status-good' if feature_store_exists else 'status-bad'}'>"
            f"{'READY' if feature_store_exists else 'MISSING'}</span>",
            unsafe_allow_html=True,
        )

    
        st.markdown(
            f"CLV Model: <span class='{'status-good' if model_exists else 'status-warn'}'>"
            f"{'TRAINED' if model_exists else 'NOT TRAINED'}</span>",
            unsafe_allow_html=True,
        )


        st.markdown(
            f"Transactions File: <span class='{'status-good' if tx_exists else 'status-bad'}'>"
            f"{'FOUND' if tx_exists else 'NOT FOUND'}</span>",
            unsafe_allow_html=True,
        )


        st.markdown("</div>", unsafe_allow_html=True)


        st.markdown("---")


        # Data source info
        st.markdown(
            "<div class='sidebar-section-title'>📦 Data Source</div>",
            unsafe_allow_html=True,
        )


        st.markdown(
            """
            <div class='sidebar-card'>
                Synthetic customer behavior dataset used for:<br>
                • CLV forecasting<br>
                • RFM segmentation<br>
                • Cohort retention<br>
                • Personas & churn<br>
                • Cross-sell modelling
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown("---")


    # ---------- TABS ----------
    tabs = st.tabs(
        [
            "🧾 Executive Summary",
            "📊 Overview",
            "💰 CLV Prediction",
            "🎯 RFM Segmentation",
            "🧠 Retention Playbook",
            "👤 Customer 360°",
            "📆 Cohort Analysis",
            "🧪 Model Evaluation",
            "📡 Model Monitoring",
            "📈 Log Analytics", 
            "📜 Logs Viewer",
            "🖥 System Health",
            "📄 Reports Preview"
        ]
    )

    LOG.info("Tabs created")



    # -------------------------
    # 1) 🧾 EXECUTIVE SUMMARY TAB
    # -------------------------
    with tabs[0]:
        st.subheader("🧾 Executive Summary Report")

        report_path = Path("reports/executive_summary.md")

        if not report_path.exists():
            st.warning("Executive Summary report not found. Generate it using the **'📄 Generate Reports'** button in the sidebar.")
        else:

            # Load content
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()

            # Display markdown
            st.markdown(report_text)

            # Show key numbers visually (optional extraction)
            try:

                # Extract metrics from text
                cust = re.search(r"Total Customers \| \*\*(.*?)\*\*", report_text)
                txs = re.search(r"Total Transactions \| \*\*(.*?)\*\*", report_text)
                rev = re.search(r"Total Revenue \| \*\₹(.*?)\*\*", report_text)
                aov = re.search(r"Average Order Value \| \*\₹(.*?)\*\*", report_text)


                if cust and txs and rev and aov:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Customers", cust.group(1))
                    c2.metric("Transactions", txs.group(1))
                    c3.metric("Revenue", "₹" + rev.group(1))
                    c4.metric("AOV", "₹" + aov.group(1))
            except Exception as e:
                st.info("Metrics could not be extracted — report still displayed normally.")

            st.markdown("---")

            # Download buttons
            st.download_button(
                "⬇ Download Executive Summary (Markdown)",
                data=report_text,
                file_name="executive_summary.md",
                mime="text/markdown",
                key="download_exec_md"
            )

            # PDF support (optional later)
            
            # --------------------------------------
            # 📄 PDF PREVIEW (Embedded Viewer)
            # --------------------------------------
            pdf_path = Path("reports/executive_summary.pdf")

            if pdf_path.exists():
                st.markdown("### 📄 Executive Summary Preview")

                try:
                    # Read PDF as base64 for inline display

                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

                    pdf_embed = f"""
                        <iframe 
                            src="data:application/pdf;base64,{base64_pdf}" 
                            width="100%" 
                            height="700px"
                            type="application/pdf">
                        </iframe>
                    """

                    st.markdown(pdf_embed, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Could not render PDF preview: {e}")

            else:
                st.info("PDF not found. Generate reports first.")
            

            # -------------------------
            # 📦 Download All Reports (ZIP)
            # Paste this in your Executive Summary tab (after PDF preview / download buttons)
            # -------------------------

            REPORT_DIR = Path("reports")
            ZIP_PATH = REPORT_DIR / "reports_bundle.zip"

            st.markdown("### 📦 Export: Download Full Report Bundle")

            # Options
            include_images = st.checkbox("Include persona images (reports/img/)", value=True, key="zip_include_imgs")
            include_pdf = st.checkbox("Include executive_summary.pdf (if present)", value=True, key="zip_include_pdf")

            if st.button("⬇ Create & Download ZIP", key="create_zip_btn"):
                # Collect files
                file_list = []

                # core markdown reports
                if REPORT_DIR.exists():
                    for p in REPORT_DIR.glob("*.md"):
                        file_list.append(p)
                else:
                    st.error("Reports folder not found (reports/). Generate reports first.")
                    st.stop()

                # optional PDF
                if include_pdf:
                    pdff = REPORT_DIR / "executive_summary.pdf"
                    if pdff.exists():
                        file_list.append(pdff)

                # optional images
                if include_images:
                    img_dir = REPORT_DIR / "img"
                    if img_dir.exists():
                        for p in img_dir.rglob("*"):
                            if p.is_file():
                                file_list.append(p)

                if not file_list:
                    st.warning("No report files found to include in ZIP.")
                else:
                    # create in-memory ZIP (so we don't depend on disk locks), but also write to reports/reports_bundle.zip
                    try:
                        # remove old zip if exists
                        if ZIP_PATH.exists():
                            ZIP_PATH.unlink()

                        # write zip to disk (smaller memory usage for large images) and also a bytes buffer for download
                        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
                            for fp in file_list:
                                # preserve folder structure inside zip (reports/... or reports/img/...)
                                arcname = fp.relative_to(REPORT_DIR.parent)  # include 'reports/...' root inside zip
                                zf.write(fp, arcname=str(arcname))

                        # read zip bytes
                        with open(ZIP_PATH, "rb") as f:
                            zip_bytes = f.read()

                        st.success(f"ZIP created: {ZIP_PATH.name} ({len(file_list)} files)")
                        st.download_button(
                            label="⬇ Download Reports Bundle (ZIP)",
                            data=zip_bytes,
                            file_name=ZIP_PATH.name,
                            mime="application/zip",
                            key="download_zip"
                        )

                    except Exception as e:
                        st.error(f"Failed to create ZIP: {e}")



    # =====================================================
    # 2) OVERVIEW TAB
    # =====================================================
    with tabs[1]:
        LOG.info("Rendering Overview tab")
        st.subheader("📊 Business Overview")

        df = get_transactions().copy()
        LOG.debug(f"Overview df shape: {df.shape}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{df['customer_id'].nunique():,}")
        c2.metric("Transactions", f"{len(df):,}")
        c3.metric("Total Revenue", f"₹{df['amount'].sum():,.0f}")
        c4.metric("Avg Order Value", f"₹{df['amount'].mean():,.0f}")

        # Additional metrics
        repeat_rate = (
            df[df["is_repeat"] == 1]["customer_id"].nunique()
            / df["customer_id"].nunique()
        )

        rfm = get_rfm()
        LOG.debug(f"RFM preview: {rfm.shape}")
        avg_tenure = rfm["tenure_days"].mean() / 30

        c5, c6, c7 = st.columns(3)
        c5.metric("Repeat Purchase Rate", f"{repeat_rate*100:.1f}%")
        c6.metric("Avg Tenure", f"{avg_tenure:.1f} months")
        c7.metric("Orders / Customer", f"{rfm['frequency'].mean():.0f}")

        # Revenue by Channel
        st.markdown("### Revenue by Channel")
        channel_rev = df.groupby("channel")["amount"].sum().reset_index()
        fig_channel = px.bar(
            channel_rev,
            x="channel",
            y="amount",
            title="Revenue by Channel",
        )
        st.plotly_chart(fig_channel, use_container_width=True, key="overview_channel")

        # Revenue by Category
        st.markdown("### Revenue by Category")
        cat_rev = df.groupby("category")["amount"].sum().reset_index()
        fig_cat = px.bar(
            cat_rev,
            x="category",
            y="amount",
            title="Revenue by Category",
        )
        st.plotly_chart(fig_cat, use_container_width=True, key="overview_category")

        # Monthly Revenue Trend
        st.markdown("### Monthly Revenue Trend")
        df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby("order_month")["amount"].sum().reset_index()
        fig_month = px.line(
            monthly,
            x="order_month",
            y="amount",
            markers=True,
            title="Monthly Revenue Trend",
        )
        st.plotly_chart(fig_month, use_container_width=True, key="overview_month")


    # =====================================================
    # 3) CLV PREDICTION TAB
    # =====================================================
    with tabs[2]:
        LOG.info("Rendering CLV Prediction tab")
        st.subheader("💰 CLV Prediction (XGBoost)")

        clv_df = get_clv_with_segments()
        LOG.debug(f"clv_df head: {clv_df.head(2).to_dict()}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg CLV", f"₹{clv_df['predicted_clv'].mean():,.0f}")
        c2.metric("Median CLV", f"₹{clv_df['predicted_clv'].median():,.0f}")
        c3.metric(
            "Top 5% CLV",
            f"₹{clv_df['predicted_clv'].quantile(0.95):,.0f}",
        )

        st.markdown("### CLV Distribution")
        fig_hist = px.histogram(
            clv_df,
            x="predicted_clv",
            nbins=40,
            title="Distribution of Predicted CLV",
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="clv_hist")

        st.markdown("### High-Value Customers")
        top_n = st.slider("Show Top N Customers", 10, 200, 50)
        st.dataframe(
            clv_df.sort_values("predicted_clv", ascending=False).head(top_n),
            use_container_width=True,
        )


    # =====================================================
    # 4) RFM SEGMENTATION TAB
    # =====================================================
    with tabs[3]:
        LOG.info("Rendering RFM Segmentation tab")
        st.subheader("🎯 RFM Segmentation")

        rfm_segments = get_rfm_segments()
        LOG.debug(f"rfm_segments sample: {rfm_segments.head(2).to_dict()}")

        seg_counts = rfm_segments["rfm_segment"].value_counts().reset_index()
        seg_counts.columns = ["rfm_segment", "count"]

        fig_seg = px.bar(
            seg_counts,
            x="rfm_segment",
            y="count",
            title="Customers per Segment",
        )
        st.plotly_chart(fig_seg, use_container_width=True, key="rfm_seg")

        rfm_tmp = rfm_segments.copy()
        rfm_tmp["clv_proxy"] = rfm_tmp["monetary"] * rfm_tmp["frequency"]

        fig_scatter = px.scatter(
            rfm_tmp,
            x="frequency",
            y="monetary",
            size="clv_proxy",
            color="rfm_segment",
            hover_data=["customer_id"],
            title="RFM Segmentation Map",
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="rfm_map")

        st.markdown("### RFM Table (Sample)")
        st.dataframe(rfm_segments.head(50), use_container_width=True)


    # =====================================================
    # 5) RETENTION PLAYBOOK TAB
    # =====================================================
    with tabs[4]:
        LOG.info("Rendering Retention Playbook tab")
        st.subheader("🧠 Retention Playbook")

        recs = get_recommendations()
        LOG.debug(f"recommendations sample: {recs.head(1).to_dict()}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"₹{recs['campaign_cost'].sum():,.0f}")
        c2.metric("Total Uplift", f"₹{recs['expected_uplift'].sum():,.0f}")
        c3.metric("Net ROI", f"₹{recs['expected_roi'].sum():,.0f}")

        st.markdown("### Recommendations Table")
        st.dataframe(
            recs[
                [
                    "customer_id",
                    "rfm_segment",
                    "value_tier",
                    "predicted_clv",
                    "campaign_cost",
                    "expected_uplift",
                    "expected_roi",
                    "recommended_action",
                    "suggested_offer",
                ]
            ],
            use_container_width=True,
        )


    # =====================================================
    # 6) CUSTOMER 360 TAB
    # =====================================================
    with tabs[5]:
        LOG.info("Rendering Customer 360 tab")
        st.subheader("👤 Customer 360° View")

        tx = get_transactions()
        clv_df = get_clv_with_segments()
        recs = get_recommendations()

        customer_ids = sorted(clv_df["customer_id"].unique().tolist())
        selected = st.selectbox("Select Customer", customer_ids)
        LOG.info(f"Customer selected: {selected}")

        cust = clv_df[clv_df["customer_id"] == selected].iloc[0]
        cust_rec = recs[recs["customer_id"] == selected].iloc[0]
        cust_tx = tx[tx["customer_id"] == selected]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted CLV", f"₹{cust['predicted_clv']:,.0f}")
        c2.metric("Historical Revenue", f"₹{cust['total_revenue']:,.0f}")
        c3.metric("Orders", int(cust["frequency"]))

        st.markdown("### Retention Strategy")
        st.write(f"**Value Tier:** {cust_rec['value_tier']}")
        st.write(f"**Recommended Action:** {cust_rec['recommended_action']}")
        st.write(f"**Suggested Offer:** {cust_rec['suggested_offer']}")
        st.write(
            f"**Campaign Cost:** ₹{cust_rec['campaign_cost']:,.0f} | "
            f"**Expected Uplift:** ₹{cust_rec['expected_uplift']:,.0f} | "
            f"**Expected ROI:** ₹{cust_rec['expected_roi']:,.0f}"
        )

        st.markdown("### Purchase Timeline")
        fig_tx = px.bar(
            cust_tx,
            x="order_date",
            y="amount",
            color="category",
            title="Transaction History",
        )
        st.plotly_chart(fig_tx, use_container_width=True, key="cust_history")

        st.markdown("### Raw Transactions")
        st.dataframe(cust_tx, use_container_width=True)


    # =====================================================
    # 7) COHORT ANALYSIS TAB
    # =====================================================
    with tabs[6]:
        LOG.info("Rendering Cohort Analysis tab")
        st.subheader("📆 Cohort Analysis")

        cohort_counts, cohort_ret = get_cohort_tables()
        LOG.debug(f"cohort_counts shape: {cohort_counts.shape}")

        st.markdown("### Cohort Size")
        base_sizes = cohort_counts.iloc[:, 0].reset_index()
        base_sizes.columns = ["cohort_month", "customers"]
        fig_size = px.bar(
            base_sizes,
            x="cohort_month",
            y="customers",
            title="Cohort Size (Month 1 Customers)",
        )
        st.plotly_chart(fig_size, use_container_width=True, key="cohort_size")

        st.markdown("### Retention Heatmap")
        fig_ret = px.imshow(
            cohort_ret,
            text_auto=True,
            aspect="auto",
            title="Cohort Retention Matrix",
        )
        st.plotly_chart(fig_ret, use_container_width=True, key="cohort_matrix")


    # =====================================================
    # 8) MODEL EVALUATION TAB
    # =====================================================
    with tabs[7]:
        LOG.info("Rendering Model Evaluation tab")
        st.subheader("🧪 Model Evaluation")

        metrics, eval_df, fi_df, shap_df = get_model_eval()
        LOG.debug(f"Evaluation metrics: {metrics}")

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"₹{metrics['MAE']:,.0f}")
        c2.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
        c3.metric("R²", f"{metrics['R2']:.3f}")

        st.markdown("### Actual vs Predicted CLV")
        fig_sc = px.scatter(
            eval_df.sample(min(1500, len(eval_df)), random_state=42),
            x="y_true",
            y="y_pred",
            opacity=0.45,
            title="Actual vs Predicted CLV",
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="eval_scatter")

        st.markdown("### Residual Distribution")
        fig_res = px.histogram(
            eval_df,
            x="residual",
            nbins=50,
            title="Residual Distribution",
        )
        st.plotly_chart(fig_res, use_container_width=True, key="eval_residual")

        if not fi_df.empty:
            st.markdown("### Feature Importance")
            fig_fi = px.bar(
                fi_df,
                x="feature",
                y="importance",
                title="Model Feature Importance",
            )
            st.plotly_chart(fig_fi, use_container_width=True, key="eval_fi")

        if not shap_df.empty:
            st.markdown("### SHAP Values")
            fig_shap = px.bar(
                shap_df,
                x="feature",
                y="mean_abs_shap",
                title="Mean |SHAP| Values",
            )
            st.plotly_chart(fig_shap, use_container_width=True, key="eval_shap")


    # =====================================================
    # 9) MODEL MONITORING TAB
    # =====================================================
    with tabs[8]:
        LOG.info("Rendering Model Monitoring tab")

        st.subheader("📡 Model Monitoring & Auto-Retrain")

        c1, c2, c3 = st.columns(3)
        
        
        with c1:
            st.markdown("### ⚙️ Retrain Controls")

            if st.button("Run Auto-Retrain Check"):
                LOG.info("User requested: Run Auto-Retrain Check")

                with st.spinner("Running auto-retrain check..."):
                    out = retrain_if_needed(force=False)

                if out.get("retrained"):
                    st.success(f"Retrained and registered model_v{out.get('version')}")
                    st.json(out.get("metrics", {}))
                    LOG.info(f"Auto-retrain performed: {out}")
                else:
                    st.info("No retrain performed: " + out.get("reason", "no reason"))
                    LOG.info(f"Auto-retrain skipped: {out.get('reason')}")


            if st.button("Force Retrain Now"):
                LOG.info("User requested: Force Retrain Now")

                with st.spinner("Forcing retrain..."):
                    out = retrain_if_needed(force=True)

                st.success(f"Retrain forced. New version: {out.get('version')}")
                LOG.info(f"Forced retrain completed: {out.get('version')}")


        with c2:
            st.markdown("### ⚙️ Model Auto-Retrain")

            # show current model info
            try:
                active = get_active_version()
                LOG.debug(f"Active model info: {active}")
            except Exception as e:
                LOG.exception("Failed to fetch active model version", exc_info=True)
                active = None

            if active:
                st.markdown(f"**Active Model:** v{active['version']}  —  trained at {active['timestamp']}")
                st.json(active.get("metrics", {}))
            else:
                st.info("No active model registered yet.")


        with c3:
            st.markdown("### 🔍 Prediction Drift Summary")

            # Auto-baseline fix inside UI
            if not Path("data/processed/baseline_feature_store.csv").exists():
                LOG.warning("Baseline missing — creating from Streamlit UI")
                
                try:
                    save_baseline_snapshot()
                    st.info("Baseline snapshot was missing — created automatically!")
                except Exception as e:
                    st.error(f"Could not create baseline: {e}")

            try:
                pred_report = prediction_drift()
                LOG.info(f"Prediction drift report: {pred_report}")
                st.metric("Prediction PSI", f"{pred_report['psi']:.4f}", delta=None)
                st.metric("Prediction KS p-value", f"{pred_report['ks_p_value']:.6f}", delta=None)
                st.write(f"Drift Status: **{pred_report['drift']}**")
            except Exception as e:
                LOG.exception("Prediction drift check failed", exc_info=True)
                st.error(f"Prediction drift check failed: {e}")


        st.markdown("### 📋 Feature Drift Report (PSI + KS)")

        try:
            feat_df = monitor_drift()
            LOG.debug("Feature drift df calculated")
            if feat_df.empty:
                st.info("No feature drift detected or baseline missing.")
            else:
                st.dataframe(feat_df, use_container_width=True)
        except Exception as e:
            LOG.exception("Feature drift check failed", exc_info=True)
            st.error(f"Feature drift check failed: {e}")
        
        # assume pred_report, feature_feat_df already computed by your code
        # pred_report: dict with psi, baseline_mean, current_mean
        # feat_df: DataFrame from monitor_drift()


        st.markdown("### ⚡ Drift Simulation Tools")

        c4, c5, c6 = st.columns(3)

        with c4:
            if st.button("🔥 Simulate Drift"):
                simulate_drift(intensity=0.25)
                st.warning("Drift simulated! Re-check Prediction Drift & Feature Drift.")

        with c5:
            if st.button("♻️ Restore Original Data"):
                restore_backup()
                st.success("Restored original feature_store!")

        with c6:
            if st.button("🌀 Evolve Synthetic Data Naturally"):
                evolve_synthetic_data()
                st.info("Synthetic dataset evolved (slow drift).")


        st.markdown("### 🔬 Drift Visualizations")

        # Top row: summary + controls
        c7, c8, = st.columns(2)


        with c7:
            st.subheader("Prediction Drift")

            if pred_report:
                st.json(pred_report)
            else:
                st.info("Prediction drift not available.")


        with c8:
            st.subheader("Export")

            # 1) Verify pred_report exists
            if not pred_report:
                st.info("No prediction drift report available to export.")
            else:
                
                # Safely extract data
                baseline_pred = pred_report.get("baseline_preds")
                current_pred  = pred_report.get("current_preds")

                # -------------------------------------------------------------
                # CASE 1: If prediction arrays exist → allow PNG download
                # -------------------------------------------------------------
                if isinstance(baseline_pred, (list, np.ndarray)) and isinstance(current_pred, (list, np.ndarray)):
                    try:
                        fig = prediction_distribution_overlay(
                            np.array(baseline_pred),
                            np.array(current_pred),
                            nbins=sample_size
                        )
                        png = fig_to_png_bytes(fig)

                        st.download_button(
                            label="⬇ Download Prediction Distribution (PNG)",
                            data=png,
                            file_name="prediction_distribution.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Could not generate PNG: {e}")

                else:
                    st.warning("Prediction distributions are missing — skipping PNG export.")

                # -------------------------------------------------------------
                # CASE 2: Drift table exists → allow CSV download
                # -------------------------------------------------------------
                try:
                    if feat_df is not None and not feat_df.empty:
                        csv = feat_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇ Download Feature Drift Report (CSV)",
                            data=csv,
                            file_name="feature_drift_report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No drift table available to export.")
                except:
                    st.info("Drift table not found or not generated.")
            
            
        st.subheader("Controls")
            
        psi_threshold = st.slider("PSI threshold (alert)", 0.01, 1.0, 0.20, 0.01)
        ks_p_threshold = st.slider("KS p-value threshold", 0.001, 0.2, 0.05, 0.001)
        sample_size = st.number_input("Histogram bins", min_value=10, max_value=200, value=50)


        # Main charts
        if pred_report and "baseline_preds" in pred_report and "current_preds" in pred_report:
            fig_pred = prediction_distribution_overlay(np.array(pred_report["baseline_preds"]), np.array(pred_report["current_preds"]), nbins=sample_size)
            st.plotly_chart(fig_pred, use_container_width=True, key="viz_pred_dist")

        # PSI bar chart
        if feat_df is not None:
            fig_psi = psi_bar_chart(feat_df)
            st.plotly_chart(fig_psi, use_container_width=True, key="viz_psi_bar")

        # Feature selection for deep-dive
        st.markdown("### Feature Deep Dive")
        features = [c for c in (feat_df["feature"].tolist() if feat_df is not None else [])]
        sel_feat = st.selectbox("Select feature", options=features)
        if sel_feat:
            # show overlay for selected feature
            baseline = load_baseline_snapshot()
            current = safe_load_feature_store() 
            fig_feat = feature_distribution_overlay(baseline[sel_feat], current[sel_feat], sel_feat, nbins=sample_size)
            st.plotly_chart(fig_feat, use_container_width=True, key=f"viz_feat_{sel_feat}")

        # PSI table with flags
        if feat_df is not None:
            psi_tbl = psi_table_with_flags(feat_df, psi_threshold, ks_p_threshold)
            st.dataframe(psi_tbl, use_container_width=True, height=350)


    # ================================
    # 10) 📈 LOG ANALYTICS TAB) 
    # ================================
    with tabs[9]:
        LOG.info("Rendering Log Analytics tab")
        st.subheader("📈 Log Analytics Dashboard")

        df_logs = load_all_logs()
        LOG.debug(f"Log analytics rows: {len(df_logs)}")

        if df_logs.empty:
            st.info("No structured logs available yet.")
        else:
            col1, col2 = st.columns(2)

            # Level Distribution
            level_counts = df_logs["level"].value_counts().reset_index()
            col1.plotly_chart(
                px.bar(level_counts, x="level", y="count", title="Log Level Distribution"),
                use_container_width=True
            )

            # Timeline chart
            df_logs["minute"] = df_logs["timestamp"].dt.floor("min")
            timeline = df_logs.groupby(["minute", "level"]).size().reset_index(name="count")

            col2.plotly_chart(
                px.line(
                    timeline,
                    x="minute",
                    y="count",
                    color="level",
                    title="Log Frequency Timeline",
                    markers=True,
                ),
                use_container_width=True,
            )

            # Message word cloud (optional later)
            st.markdown("### 🔍 View Raw Logs Table")
            st.dataframe(df_logs, use_container_width=True)


    # ==========================
    # 11) 📜 ADVANCED LOGS VIEWER (Live Tail + Color)
    # ==========================
    with tabs[10]:

        LOG.info("Rendering Logs Viewer tab")
        st.subheader("📜 Real-Time Logs Viewer (Advanced)")

        LOG_DIR.mkdir(exist_ok=True)
        files = list_log_files()

        if not files:
            st.info("No log files yet — run any pipeline to generate logs.")
        else:
            colA, colB = st.columns([2, 1])

            with colA:
                log_choice = st.selectbox("Select Log File", files)
                LOG.info(f"Selected log file: {log_choice}")

            with colB:
                n_lines = st.slider("Tail (Lines)", 50, 2000, 300, 50)

            # --- Logs Viewer Filter Section ---

            keyword = st.text_input("Search logs (keyword / severity):", "")
            level = st.selectbox("Filter by Level", ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"])

            auto_refresh = st.checkbox("Live Tail (Auto-refresh every 2s)", value=False)

            log_path = LOG_DIR / log_choice

            if log_path.exists():
                # Read lines
                raw_lines = read_last_lines(log_path, n_lines=n_lines)
                LOG.debug(f"Read {len(raw_lines)} lines from {log_path.name}")

                # Apply filters
                filtered = filter_log_lines(raw_lines, keyword, level)
                LOG.debug(f"Filtered lines count: {len(filtered)}")

                # -----------------------------------------
                # NEW: If no results → Show clean message
                # -----------------------------------------
                if len(filtered) == 0:
                    st.warning(f"No matching logs found for filter: "
                            f"{'level=' + level if level != 'ALL' else ''} "
                            f"{'keyword=' + keyword if keyword else ''}".strip())
                else:
                    # Highlight
                    html_block = highlight_block(filtered)

                    # Display HTML block
                    st.markdown("### 🔍 Log Output")
                    st.markdown(
                        f"<div style='background:#111; padding:15px; border-radius:8px; height:450px; overflow-y:scroll;'>{html_block}</div>",
                        unsafe_allow_html=True
                    )

                # AUTO-REFRESH
                if auto_refresh:
                    LOG.debug("Logs Viewer auto-refresh enabled")
                    time.sleep(2)
                    st.rerun()


            # DOWNLOADS
            st.markdown("### 📥 Download")
            col1, col2 = st.columns(2)

            with col1:
                LOG.info("User clicked: Download selected log (if pressed)")
                st.download_button(
                    "⬇ Download Selected Log",
                    data=open(log_path, "rb").read(),
                    file_name=log_choice,
                    mime="text/plain"
                )

            with col2:
                LOG.info("User clicked: Download ALL Logs (if pressed)")
                zip_path = zip_all_logs()
                st.download_button(
                    "⬇ Download ALL Logs (ZIP)",
                    data=open(zip_path, "rb").read(),
                    file_name="all_logs.zip",
                    mime="application/zip"
                )


    # ================================
    # 12) 🖥 SYSTEM HEALTH TAB
    # ================================
    with tabs[11]:
        LOG.info("Rendering System Health tab")
        st.subheader("🖥 System Health Monitoring")

        metrics = get_system_metrics()
        LOG.debug(f"System metrics: {metrics}")

        c1, c2, c3 = st.columns(3)
        c1.metric("CPU Usage (%)", f"{metrics['cpu_percent']}%")
        c2.metric("RAM Usage (%)", f"{metrics['memory_percent']}%")
        c3.metric("Disk Usage (%)", f"{metrics['disk_percent']}%")

        st.caption(f"Platform: {metrics['platform']} | Python {metrics['python_version']}")

        # Live auto-refresh
        auto = st.checkbox("🔄 Auto-refresh every 2s", value=False)
        if auto:
            LOG.debug("System Health auto-refresh enabled")
            time.sleep(2)
            st.rerun()

        st.markdown("### 🔥 Top Processes by CPU Usage")
        proc = get_process_table()
        LOG.debug(f"Top processes fetched: {len(proc)}")
        st.dataframe(proc, use_container_width=True)


    # ================================
    # 13) 📄 REPORTS PREVIEW TAB
    # ================================
    with tabs[12]: 
        st.subheader("📄 Reports Preview")

        reports_dir = Path("reports")
        md_files = sorted([p for p in reports_dir.glob("*.md")])

        if not md_files:
            st.info("No reports found. Generate reports first.")
        else:
            selected = st.selectbox(
                "Select Report to Preview",
                [f.name for f in md_files],
                index=0
            )

            md_path = reports_dir / selected
            content = md_path.read_text(encoding="utf-8")

            # nice display
            st.markdown(f"### 📝 {selected}")
            st.markdown(content, unsafe_allow_html=True)

            # download button
            st.download_button(
                "Download This Report",
                data=content,
                file_name=selected,
                mime="text/markdown"
            )

        # Close main-container div
        st.markdown("</div>", unsafe_allow_html=True)
        LOG.info("Exiting main() and rendered all tabs")


if __name__ == "__main__":
    LOG.info("Running streamlit_app as __main__")
    main()