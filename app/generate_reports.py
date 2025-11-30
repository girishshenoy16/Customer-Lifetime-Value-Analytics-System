"""
generate_reports.py
Creates professional markdown reports inside /reports/ for:
- Executive Summary (NEW)
- Business overview
- RFM segmentation
- CLV summary
- Persona insights
- Cohort retention
"""
import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import pandas as pd
from datetime import datetime
from app.pdf_generator import generate_pdf_summary
from app.rfm_segmentation import build_rfm_segments

REPORT_DIR = Path("reports")
IMG_DIR = REPORT_DIR / "img"


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        # replace ALL unicode characters with ascii equivalents
        ascii_text = text.encode("ascii", "replace").decode()
        print(ascii_text)




def write_md(path, content):
    """Write Markdown file into reports/ directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------
# 🔥 EXECUTIVE SUMMARY REPORT  (NEW)
# ---------------------------------------------------------
def generate_executive_summary(tx, rfm, fs):

    total_customers = tx["customer_id"].nunique()
    total_tx = len(tx)
    total_revenue = tx["amount"].sum()
    aov = tx["amount"].mean()

    top_persona = fs["persona"].value_counts().idxmax()
    churn_rate = fs["is_churned"].mean()
    avg_clv = fs["predicted_clv"].mean()

    if "rfm_segment" in rfm.columns:
        strongest_segment = rfm["rfm_segment"].value_counts().idxmax()
    else:
        strongest_segment = "Unknown"


    md = f"""
# 🧾 Executive Summary
Generated: **{datetime.now().strftime("%Y-%m-%d %H:%M")}**

This report summarizes the most important insights from all analytics modules:
personas, CLV modeling, RFM segmentation, and cohort retention.

---

## ⭐ 1. Business Snapshot

| Metric | Value |
|-------|-------|
| Total Customers | **{total_customers:,}** |
| Total Transactions | **{total_tx:,}** |
| Total Revenue | **₹{total_revenue:,.0f}** |
| Average Order Value | **₹{aov:,.0f}** |

**Interpretation:**  
Your business revenue is driven by a moderate customer base, with solid order volume
and healthy average order value.

---

## ⭐ 2. Customer Personas (Who Your Customers Really Are)

### Most frequent persona: **{top_persona.capitalize()}**

Other persona insights:
- Personas defined from purchase behavior & attributes  
- Loyalists = high spend, low churn  
- One-timers = majority of churn risk  
- Subscribers = strong lifetime revenue potential  

**Business takeaway:**  
Focus retention campaigns on high-value personas (Loyalists, Subscribers) while
running reactivation campaigns for One-Timers.

---

## ⭐ 3. Customer Lifetime Value (CLV)

- **Average Predicted CLV:** ₹{avg_clv:,.0f}
- High CLV customers show:
  - High Monetary value  
  - Longer tenure  
  - Lower churn probability  
  - Strong subscription adoption  

**Business takeaway:**  
A small % of customers drive a large % of CLV → invest in VIP retention campaigns.

---

## ⭐ 4. RFM Segmentation

- **Top segment by count:** `{strongest_segment}`
- Segments include: Champions, Loyal Customers, At Risk, Lost, Hibernating, etc.

**Business takeaway:**  
RFM reveals renewal opportunities — especially customers slipping from Loyal → At Risk.

---

## ⭐ 5. Churn & Retention

- Overall churn rate: **{churn_rate*100:.1f}%**
- Persona-level churn helps identify exact churn drivers
- Cohort analysis shows retention decay month over month

**Business takeaway:**  
Focus on reducing early lifecycle churn (months 1–3).

---

## ⭐ 6. Recommended Actions

### 🎯 Short-Term Actions  
- Run reactivation campaigns for **One-Timers**
- Launch upsell/cross-sell offers for **Subscribers**
- Incentivize repeat purchases for **Loyalists**

### 🎯 Long-Term Strategy  
- Build lifecycle journeys tailored to each persona  
- Increase customer tenure → improves overall CLV  
- Launch loyalty program for Champions & Loyalists  

---

## ⭐ 7. Supporting Visualizations (Dashboard Only)
See Streamlit Tabs:
- Overview  
- RFM Segmentation  
- CLV Prediction  
- Persona Insights  
- Cohort Analysis  
- Retention Playbook  

---

**End of Executive Summary Report**
"""

    write_md(REPORT_DIR / "executive_summary.md", md)


# ---------------------------------------------------------
# BUSINESS REPORT
# ---------------------------------------------------------
def generate_business_report(tx):
    total_customers = tx["customer_id"].nunique()
    total_tx = len(tx)
    total_revenue = tx["amount"].sum()
    aov = tx["amount"].mean()

    md = f"""
# 📊 Business Overview
Generated: **{datetime.now().strftime("%Y-%m-%d %H:%M")}**

## Key Metrics
- Customers: {total_customers:,}
- Total Transactions: {total_tx:,}
- Total Revenue: ₹{total_revenue:,.0f}
- Average Order Value: ₹{aov:,.0f}

---
"""

    write_md(REPORT_DIR / "business_overview.md", md)


# ---------------------------------------------------------
# RFM REPORT
# ---------------------------------------------------------
def generate_rfm_report(rfm):
    seg_counts = rfm["rfm_segment"].value_counts()
    md = f"""
# 🎯 RFM Segmentation Report

## Segment Counts
```

{seg_counts.to_string()}

```

---
"""
    write_md(REPORT_DIR / "rfm_report.md", md)


# ---------------------------------------------------------
# CLV REPORT
# ---------------------------------------------------------
def generate_clv_report(fs):
    md = f"""
# 💰 CLV Model Summary

- Average CLV: ₹{fs['predicted_clv'].mean():,.0f}
- Median CLV: ₹{fs['predicted_clv'].median():,.0f}
- Top 5% CLV: ₹{fs['predicted_clv'].quantile(0.95):,.0f}

---
"""
    write_md(REPORT_DIR / "clv_summary.md", md)


# ---------------------------------------------------------
# PERSONA REPORT
# ---------------------------------------------------------
def generate_persona_report(fs):
    md = f"""
# 🧠 Persona Insights

## Personas
```

{fs['persona'].value_counts().to_string()}

```

## Subscriber Share
```

{fs.groupby("persona")["is_subscriber"].mean().round(3).to_string()}

```

## Churn Rate
```

{fs.groupby("persona")["is_churned"].mean().round(3).to_string()}

```

---
"""
    write_md(REPORT_DIR / "persona_insights.md", md)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("Generating reports...")

    tx = pd.read_csv("data/raw/customers_transactions.csv")
    
    rfm_raw = pd.read_csv("data/processed/rfm_dataset.csv")

    # Always regenerate proper RFM segmentation (scores + segment label)
    try:
        rfm = build_rfm_segments(rfm_raw.copy())
    except Exception as e:
        print("\n❌ Failed to build RFM segments:", e)
        print("Using raw RFM dataset without segments.")
        rfm = rfm_raw.copy()




    # LOAD FEATURE STORE
    fs_base = pd.read_csv("data/processed/feature_store.csv")

    # LOAD CLV WITH SEGMENTS (preferred)
    clv_path = Path("data/processed/clv_with_segments.csv")

    if clv_path.exists():
        fs = pd.read_csv(clv_path)
    else:
        print("WARNING: clv_with_segments.csv not found — using feature_store.")
        fs = fs_base.copy()
        fs["predicted_clv"] = 0  # prevent KeyError in summary

    print("Generating reports...")
    
    # Now generate reports
    generate_executive_summary(tx, rfm, fs)
    generate_business_report(tx)
    generate_rfm_report(rfm)
    generate_clv_report(fs)
    generate_persona_report(fs)

    safe_print("Reports generated -> reports/")

    print("Generating PDF...")

    generate_pdf_summary()

    print("Executive Summary PDF created -> reports/executive_summary.pdf")


if __name__ == "__main__":
    main()