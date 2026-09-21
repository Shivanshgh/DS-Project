"""
Supermarket Sales Analysis — Streamlit Application
====================================================
Dataset : supermarket_sales.csv  (Kaggle Supermarket Sales, 1 000 rows)
Columns : Invoice ID · Branch · City · Customer type · Gender ·
          Product line · Unit price · Quantity · Tax 5% · Total ·
          Date · Time · Payment · cogs · gross margin percentage ·
          gross income · Rating

Single-file application that covers:
  • Data loading & cleaning
  • Total-sales recalculation  (Quantity × Unit Price × 1.05)
  • Exploratory Data Analysis (EDA)
  • Interactive Streamlit front-end with charts & KPIs
  • Business recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supermarket Sales Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .metric-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 18px 20px;
            text-align: center;
        }
        .metric-card h3 { margin: 0; font-size: 1.7rem; color: #3b82d4; }
        .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #57606a; }
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1f2328;
            border-left: 4px solid #3b82d4;
            padding-left: 10px;
            margin-top: 30px;
            margin-bottom: 10px;
        }
        .insight-box {
            background: #f0f6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
            font-size: 0.92rem;
            color: #1f2328;
        }
        .rec-box {
            background: #f0fdf4;
            border: 1px solid #86efac;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
            font-size: 0.92rem;
            color: #1f2328;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAP  (new dataset → internal names used throughout the app)
# ─────────────────────────────────────────────────────────────────────────────
# The Kaggle dataset uses "Product line" and "Customer type" (lowercase t).
# We normalise to "Category" and "Customer Type" once during cleaning so the
# rest of the code stays identical.

RAW_CATEGORY    = "Product line"    # → renamed to "Category"
RAW_CUST_TYPE   = "Customer type"   # → renamed to "Customer Type"
RAW_TOTAL       = "Total"           # existing column (cogs + Tax 5%); verified


# ─────────────────────────────────────────────────────────────────────────────
# BACK-END: DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & cleaning data…")
def load_and_clean(path: str = "supermarket_sales.csv") -> tuple[pd.DataFrame, dict]:
    """
    Load the CSV, perform data-cleaning steps, recalculate Sales, and return
    the clean DataFrame together with a cleaning-report dictionary.
    """
    df_raw = pd.read_csv(path)
    report = {}

    # ── 1. Record initial shape ──────────────────────────────────────────────
    report["rows_initial"] = len(df_raw)
    report["cols_initial"] = len(df_raw.columns)

    # ── 2. Strip whitespace from string columns ──────────────────────────────
    str_cols = df_raw.select_dtypes(include=["object", "string"]).columns
    df_raw[str_cols] = df_raw[str_cols].apply(lambda s: s.str.strip())

    # ── 3. Rename columns to internal standard names ─────────────────────────
    rename_map = {}
    if RAW_CATEGORY in df_raw.columns:
        rename_map[RAW_CATEGORY] = "Category"
    if RAW_CUST_TYPE in df_raw.columns:
        rename_map[RAW_CUST_TYPE] = "Customer Type"
    if rename_map:
        df_raw.rename(columns=rename_map, inplace=True)

    # ── 4. Missing-value audit ───────────────────────────────────────────────
    report["missing_before"] = int(df_raw.isnull().sum().sum())

    # Drop rows where key identifiers are missing
    df_raw.dropna(
        subset=["Invoice ID", "Date", "Branch", "Category", "Quantity", "Unit price"],
        inplace=True,
    )

    # Fill remaining missing numerics with column median
    for col in ["Quantity", "Unit price", "Rating", "Tax 5%", "Total", "cogs", "gross income"]:
        if col in df_raw.columns:
            median_val = pd.to_numeric(df_raw[col], errors="coerce").median()
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(median_val)

    # Fill remaining missing categoricals with mode
    for col in ["Category", "Customer Type", "Gender", "Payment", "City", "Time"]:
        if col in df_raw.columns:
            mode_val = df_raw[col].mode()[0]
            df_raw[col].fillna(mode_val, inplace=True)

    report["missing_after"] = int(df_raw.isnull().sum().sum())

    # ── 5. Duplicate rows ────────────────────────────────────────────────────
    report["duplicates"] = int(df_raw.duplicated(subset="Invoice ID").sum())
    df_raw.drop_duplicates(subset="Invoice ID", inplace=True)

    # ── 6. Type corrections ──────────────────────────────────────────────────
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw.dropna(subset=["Date"], inplace=True)   # drop unparseable dates

    df_raw["Quantity"]   = pd.to_numeric(df_raw["Quantity"],   errors="coerce")
    df_raw["Unit price"] = pd.to_numeric(df_raw["Unit price"], errors="coerce")
    df_raw["Rating"]     = pd.to_numeric(df_raw["Rating"],     errors="coerce")

    # ── 7. Negative / zero guard ─────────────────────────────────────────────
    report["negative_qty"]   = int((df_raw["Quantity"]   <= 0).sum())
    report["negative_price"] = int((df_raw["Unit price"] <= 0).sum())
    df_raw = df_raw[(df_raw["Quantity"] > 0) & (df_raw["Unit price"] > 0)]

    # Rating must be in [1, 10] for this dataset (Kaggle uses 1–10 scale)
    df_raw["Rating"] = df_raw["Rating"].clip(1, 10)

    # ── 8. Recalculate / verify Sales ────────────────────────────────────────
    # Formula: Sales = Quantity × Unit price × 1.05  (5 % tax included in Total)
    df_raw["Calculated_Sales"] = (df_raw["Quantity"] * df_raw["Unit price"] * 1.05).round(4)
    if RAW_TOTAL in df_raw.columns:
        df_raw[RAW_TOTAL] = pd.to_numeric(df_raw[RAW_TOTAL], errors="coerce")
        mismatch = (df_raw[RAW_TOTAL].round(2) != df_raw["Calculated_Sales"].round(2)).sum()
        report["sales_mismatch"] = int(mismatch)
        df_raw["Sales"] = df_raw["Calculated_Sales"]   # authoritative
    else:
        df_raw["Sales"] = df_raw["Calculated_Sales"]
        report["sales_mismatch"] = 0

    df_raw.drop(columns=["Calculated_Sales"], inplace=True)

    # ── 9. Derived time features ─────────────────────────────────────────────
    df_raw["Month"]      = df_raw["Date"].dt.month
    df_raw["Month_Name"] = df_raw["Date"].dt.strftime("%b")
    df_raw["Day_Name"]   = df_raw["Date"].dt.strftime("%a")
    df_raw["Quarter"]    = df_raw["Date"].dt.quarter

    report["rows_final"]   = len(df_raw)
    report["rows_removed"] = report["rows_initial"] - report["rows_final"]

    return df_raw.reset_index(drop=True), report


# ─────────────────────────────────────────────────────────────────────────────
# BACK-END: EDA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def eda_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary of key EDA metrics."""
    top_category  = df.groupby("Category")["Sales"].sum().idxmax()
    top_category_v= df.groupby("Category")["Sales"].sum().max()

    top_branch    = df.groupby("Branch")["Sales"].sum().idxmax()
    top_branch_v  = df.groupby("Branch")["Sales"].sum().max()

    top_payment   = df["Payment"].value_counts().idxmax()
    top_payment_v = df["Payment"].value_counts().max()

    avg_rating    = df["Rating"].mean()
    total_sales   = df["Sales"].sum()
    total_orders  = len(df)
    avg_basket    = df["Sales"].mean()

    best_month    = df.groupby("Month_Name")["Sales"].sum().idxmax()
    best_day      = df.groupby("Day_Name")["Sales"].sum().idxmax()

    top_cust_type = df.groupby("Customer Type")["Sales"].sum().idxmax()

    avg_gross_margin = df["gross margin percentage"].mean() if "gross margin percentage" in df.columns else None
    total_gross_income = df["gross income"].sum() if "gross income" in df.columns else None

    return {
        "total_sales":         total_sales,
        "total_orders":        total_orders,
        "avg_basket":          avg_basket,
        "avg_rating":          avg_rating,
        "top_category":        (top_category,  top_category_v),
        "top_branch":          (top_branch,    top_branch_v),
        "top_payment":         (top_payment,   top_payment_v),
        "best_month":          best_month,
        "best_day":            best_day,
        "top_cust_type":       top_cust_type,
        "avg_gross_margin":    avg_gross_margin,
        "total_gross_income":  total_gross_income,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRONT-END HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Set2

def metric_card(col, label: str, value: str):
    col.markdown(
        f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def recommendation(text: str):
    st.markdown(f'<div class="rec-box">✅ {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🛒 Supermarket Sales Analysis")
    st.markdown(
        "An interactive EDA dashboard built with **Streamlit** and **Plotly**. "
        "Use the sidebar filters to slice the data."
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    df, report = load_and_clean()
    eda = eda_summary(df)

    # ── Sidebar: Filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔎 Filters")

        branches = sorted(df["Branch"].unique())
        sel_branch = st.multiselect("Branch", branches, default=branches)

        categories = sorted(df["Category"].unique())
        sel_cat = st.multiselect("Product Line", categories, default=categories)

        payments = sorted(df["Payment"].unique())
        sel_pay = st.multiselect("Payment Method", payments, default=payments)

        cust_types = sorted(df["Customer Type"].unique())
        sel_cust = st.multiselect("Customer Type", cust_types, default=cust_types)

        genders = sorted(df["Gender"].unique())
        sel_gender = st.multiselect("Gender", genders, default=genders)

        date_min = df["Date"].min().date()
        date_max = df["Date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

        st.divider()
        st.caption("📊 Dataset: 1,000 transactions · 17 columns · 3 branches")

    # ── Apply filters ─────────────────────────────────────────────────────────
    fdf = df[
        df["Branch"].isin(sel_branch)
        & df["Category"].isin(sel_cat)
        & df["Payment"].isin(sel_pay)
        & df["Customer Type"].isin(sel_cust)
        & df["Gender"].isin(sel_gender)
    ]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        fdf = fdf[(fdf["Date"] >= start_d) & (fdf["Date"] <= end_d)]

    if fdf.empty:
        st.warning("No data matches the current filters. Adjust the sidebar selections.")
        return

    feda = eda_summary(fdf)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ═══════════════════════════════════════════════════════════════════════
    tab_overview, tab_products, tab_branches, tab_customers, tab_time, tab_financials, tab_data, tab_cleaning = st.tabs([
        "📈 Overview",
        "📦 Product Lines",
        "🏪 Branches",
        "👥 Customers",
        "📅 Time Trends",
        "💰 Financials",
        "🗃️ Raw Data",
        "🧹 Data Cleaning",
    ])

    # ───────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ───────────────────────────────────────────────────────────────────────
    with tab_overview:
        section("Key Performance Indicators")
        k1, k2, k3, k4 = st.columns(4)
        metric_card(k1, "Total Revenue ($)",   f"${feda['total_sales']:,.2f}")
        metric_card(k2, "Total Orders",         f"{feda['total_orders']:,}")
        metric_card(k3, "Avg Basket Size ($)",  f"${feda['avg_basket']:,.2f}")
        metric_card(k4, "Avg Customer Rating",  f"{feda['avg_rating']:.2f} / 10")

        st.markdown("<br>", unsafe_allow_html=True)
        k5, k6, k7, k8 = st.columns(4)
        metric_card(k5, "Top Product Line",   feda["top_category"][0])
        metric_card(k6, "Best Branch",        f"Branch {feda['top_branch'][0]}")
        metric_card(k7, "Popular Payment",    feda["top_payment"][0])
        metric_card(k8, "Best Sales Month",   feda["best_month"])

        if feda["avg_gross_margin"] is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            kg1, kg2, kg3 = st.columns(3)
            metric_card(kg1, "Gross Margin %",       f"{feda['avg_gross_margin']:.2f}%")
            metric_card(kg2, "Total Gross Income ($)",f"${feda['total_gross_income']:,.2f}")
            metric_card(kg3, "Top Customer Type",     feda["top_cust_type"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Sales by Branch (bar) + Payment mix (pie) side by side
        section("Sales Overview")
        col_a, col_b = st.columns(2)

        branch_sales = (
            fdf.groupby(["Branch", "City"])["Sales"].sum().reset_index()
            .sort_values("Sales", ascending=False)
        )
        fig_branch = px.bar(
            branch_sales, x="Branch", y="Sales",
            color="City", color_discrete_sequence=PALETTE,
            title="Total Revenue by Branch & City",
            labels={"Sales": "Revenue ($)"},
            text_auto=".2s",
        )
        fig_branch.update_layout(plot_bgcolor="white")
        col_a.plotly_chart(fig_branch, use_container_width=True)

        pay_counts = fdf["Payment"].value_counts().reset_index()
        pay_counts.columns = ["Payment", "Count"]
        fig_pay = px.pie(
            pay_counts, names="Payment", values="Count",
            color_discrete_sequence=PALETTE,
            title="Payment Method Distribution",
            hole=0.4,
        )
        fig_pay.update_traces(textposition="inside", textinfo="percent+label")
        col_b.plotly_chart(fig_pay, use_container_width=True)

        # Revenue by Product Line
        section("Revenue by Product Line")
        cat_sales = (
            fdf.groupby("Category")["Sales"].sum().reset_index()
            .sort_values("Sales", ascending=False)
        )
        fig_cat = px.bar(
            cat_sales, x="Sales", y="Category",
            orientation="h", color="Category",
            color_discrete_sequence=PALETTE,
            title="Total Revenue by Product Line",
            labels={"Sales": "Revenue ($)", "Category": "Product Line"},
            text_auto=".2s",
        )
        fig_cat.update_layout(showlegend=False, plot_bgcolor="white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_cat, use_container_width=True)

        # Key insights
        section("Key Insights")
        insight(f"**{feda['top_category'][0]}** is the highest-revenue product line with ${feda['top_category'][1]:,.2f} in total sales.")
        insight(f"**Branch {feda['top_branch'][0]}** leads all branches with ${feda['top_branch'][1]:,.2f} in revenue.")
        insight(f"**{feda['top_payment'][0]}** is the most preferred payment method ({feda['top_payment'][1]:,} transactions).")
        insight(f"Best sales month: **{feda['best_month']}** | Best sales day: **{feda['best_day']}**.")
        if feda["avg_gross_margin"] is not None:
            insight(f"Average gross margin is **{feda['avg_gross_margin']:.4f}%** with total gross income of **${feda['total_gross_income']:,.2f}**.")

    # ───────────────────────────────────────────────────────────────────────
    # TAB 2 — PRODUCT LINES
    # ───────────────────────────────────────────────────────────────────────
    with tab_products:
        section("Product Line Revenue Ranking")
        pl_sales = (
            fdf.groupby("Category")["Sales"].sum()
            .sort_values(ascending=False).reset_index()
        )
        fig_pl = px.bar(
            pl_sales, x="Sales", y="Category",
            orientation="h", color="Sales",
            color_continuous_scale="Blues",
            title="Product Lines by Total Revenue",
            labels={"Sales": "Revenue ($)", "Category": "Product Line"},
            text_auto=".2s",
        )
        fig_pl.update_layout(plot_bgcolor="white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_pl, use_container_width=True)

        col1, col2 = st.columns(2)

        # Average Rating by Product Line
        rat_pl = (
            fdf.groupby("Category")["Rating"].mean()
            .sort_values(ascending=False).reset_index()
        )
        fig_rat = px.bar(
            rat_pl, x="Rating", y="Category",
            orientation="h", color="Rating",
            color_continuous_scale="RdYlGn",
            title="Avg Customer Rating by Product Line",
            labels={"Rating": "Avg Rating (1–10)", "Category": "Product Line"},
            text_auto=".2f",
        )
        fig_rat.update_layout(plot_bgcolor="white", yaxis={"categoryorder": "total ascending"})
        col1.plotly_chart(fig_rat, use_container_width=True)

        # Quantity Sold by Product Line
        qty_pl = (
            fdf.groupby("Category")["Quantity"].sum()
            .sort_values(ascending=False).reset_index()
        )
        fig_qty = px.bar(
            qty_pl, x="Quantity", y="Category",
            orientation="h", color="Quantity",
            color_continuous_scale="Purples",
            title="Units Sold by Product Line",
            labels={"Quantity": "Units Sold", "Category": "Product Line"},
            text_auto="d",
        )
        fig_qty.update_layout(plot_bgcolor="white", yaxis={"categoryorder": "total ascending"})
        col2.plotly_chart(fig_qty, use_container_width=True)

        section("Product Line Performance Summary")
        cat_detail = fdf.groupby("Category").agg(
            Total_Sales=("Sales", "sum"),
            Avg_Rating=("Rating", "mean"),
            Total_Orders=("Invoice ID", "count"),
            Avg_Basket=("Sales", "mean"),
            Total_Qty=("Quantity", "sum"),
        ).reset_index().sort_values("Total_Sales", ascending=False)

        fig_scatter = px.scatter(
            cat_detail,
            x="Total_Orders", y="Total_Sales",
            size="Avg_Basket", color="Category",
            hover_name="Category",
            color_discrete_sequence=PALETTE,
            title="Product Line: Orders vs Revenue (bubble = avg basket)",
            labels={
                "Total_Orders": "Number of Transactions",
                "Total_Sales": "Total Revenue ($)",
                "Category": "Product Line",
            },
        )
        fig_scatter.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.dataframe(
            cat_detail.rename(columns={
                "Category":     "Product Line",
                "Total_Sales":  "Total Revenue ($)",
                "Avg_Rating":   "Avg Rating",
                "Total_Orders": "Transactions",
                "Avg_Basket":   "Avg Basket ($)",
                "Total_Qty":    "Units Sold",
            }).style.format({
                "Total Revenue ($)": "${:,.2f}",
                "Avg Rating":        "{:.2f}",
                "Avg Basket ($)":    "${:,.2f}",
            }),
            use_container_width=True,
        )

    # ───────────────────────────────────────────────────────────────────────
    # TAB 3 — BRANCHES
    # ───────────────────────────────────────────────────────────────────────
    with tab_branches:
        section("Branch-Level Performance")
        branch_detail = fdf.groupby(["Branch", "City"]).agg(
            Total_Sales=("Sales", "sum"),
            Avg_Rating=("Rating", "mean"),
            Transactions=("Invoice ID", "count"),
            Avg_Basket=("Sales", "mean"),
            Gross_Income=("gross income", "sum") if "gross income" in fdf.columns else ("Sales", "count"),
        ).reset_index().sort_values("Total_Sales", ascending=False)

        col1, col2 = st.columns(2)
        fig_bbar = px.bar(
            branch_detail, x="Branch", y="Total_Sales",
            color="City", barmode="group",
            color_discrete_sequence=PALETTE,
            title="Total Revenue by Branch & City",
            labels={"Total_Sales": "Revenue ($)"},
            text_auto=".2s",
        )
        fig_bbar.update_layout(plot_bgcolor="white")
        col1.plotly_chart(fig_bbar, use_container_width=True)

        fig_brat = px.bar(
            branch_detail, x="Branch", y="Avg_Rating",
            color="City", barmode="group",
            color_discrete_sequence=PALETTE,
            title="Average Customer Rating by Branch",
            labels={"Avg_Rating": "Avg Rating"},
            text_auto=".2f",
        )
        fig_brat.update_layout(plot_bgcolor="white")
        col2.plotly_chart(fig_brat, use_container_width=True)

        section("Product Line Mix per Branch")
        branch_cat = fdf.groupby(["Branch", "Category"])["Sales"].sum().reset_index()
        fig_sunburst = px.sunburst(
            branch_cat, path=["Branch", "Category"], values="Sales",
            color_discrete_sequence=PALETTE,
            title="Revenue Breakdown: Branch → Product Line",
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)

        section("Payment Methods per Branch")
        branch_pay = fdf.groupby(["Branch", "Payment"])["Sales"].sum().reset_index()
        fig_bpay = px.bar(
            branch_pay, x="Branch", y="Sales",
            color="Payment", barmode="stack",
            color_discrete_sequence=PALETTE,
            title="Revenue by Branch and Payment Method",
            labels={"Sales": "Revenue ($)"},
        )
        fig_bpay.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_bpay, use_container_width=True)

        st.dataframe(
            branch_detail.rename(columns={
                "Total_Sales":  "Total Revenue ($)",
                "Avg_Rating":   "Avg Rating",
                "Avg_Basket":   "Avg Basket ($)",
                "Gross_Income": "Gross Income ($)",
            }).style.format({
                "Total Revenue ($)": "${:,.2f}",
                "Avg Rating":        "{:.2f}",
                "Avg Basket ($)":    "${:,.2f}",
                "Gross Income ($)":  "${:,.2f}",
            }),
            use_container_width=True,
        )

    # ───────────────────────────────────────────────────────────────────────
    # TAB 4 — CUSTOMERS
    # ───────────────────────────────────────────────────────────────────────
    with tab_customers:
        section("Customer Type Analysis")
        col1, col2 = st.columns(2)

        cust_sales = fdf.groupby("Customer Type")["Sales"].sum().reset_index()
        fig_cust = px.pie(
            cust_sales, names="Customer Type", values="Sales",
            color_discrete_sequence=PALETTE,
            title="Revenue Share: Member vs Normal",
            hole=0.4,
        )
        fig_cust.update_traces(textposition="inside", textinfo="percent+label")
        col1.plotly_chart(fig_cust, use_container_width=True)

        cust_cat = fdf.groupby(["Customer Type", "Category"])["Sales"].sum().reset_index()
        fig_cust_cat = px.bar(
            cust_cat, x="Category", y="Sales",
            color="Customer Type", barmode="group",
            color_discrete_sequence=PALETTE,
            title="Spending by Product Line & Customer Type",
            labels={"Sales": "Revenue ($)", "Category": "Product Line"},
        )
        fig_cust_cat.update_layout(plot_bgcolor="white")
        col2.plotly_chart(fig_cust_cat, use_container_width=True)

        section("Gender Analysis")
        col3, col4 = st.columns(2)

        gender_sales = fdf.groupby("Gender")["Sales"].sum().reset_index()
        fig_gen = px.pie(
            gender_sales, names="Gender", values="Sales",
            color_discrete_sequence=[PALETTE[2], PALETTE[3]],
            title="Revenue Share by Gender",
            hole=0.4,
        )
        fig_gen.update_traces(textposition="inside", textinfo="percent+label")
        col3.plotly_chart(fig_gen, use_container_width=True)

        gender_cat = fdf.groupby(["Gender", "Category"])["Sales"].sum().reset_index()
        fig_gen_cat = px.bar(
            gender_cat, x="Category", y="Sales",
            color="Gender", barmode="group",
            color_discrete_sequence=[PALETTE[2], PALETTE[3]],
            title="Spending by Product Line & Gender",
            labels={"Sales": "Revenue ($)", "Category": "Product Line"},
        )
        fig_gen_cat.update_layout(plot_bgcolor="white")
        col4.plotly_chart(fig_gen_cat, use_container_width=True)

        section("Customer Ratings Distribution")
        fig_hist = px.histogram(
            fdf, x="Rating", nbins=20,
            color_discrete_sequence=[PALETTE[0]],
            title="Distribution of Customer Ratings",
            labels={"Rating": "Rating (1–10)", "count": "Frequency"},
        )
        fig_hist.update_layout(plot_bgcolor="white", bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

        col5, col6 = st.columns(2)
        fig_rbox = px.box(
            fdf, x="Branch", y="Rating", color="Branch",
            color_discrete_sequence=PALETTE,
            title="Rating Distribution by Branch",
            points="all",
        )
        fig_rbox.update_layout(plot_bgcolor="white", showlegend=False)
        col5.plotly_chart(fig_rbox, use_container_width=True)

        fig_rcat = px.box(
            fdf, x="Category", y="Rating", color="Category",
            color_discrete_sequence=PALETTE,
            title="Rating Distribution by Product Line",
            points="all",
        )
        fig_rcat.update_layout(plot_bgcolor="white", showlegend=False)
        col6.plotly_chart(fig_rcat, use_container_width=True)

    # ───────────────────────────────────────────────────────────────────────
    # TAB 5 — TIME TRENDS
    # ───────────────────────────────────────────────────────────────────────
    with tab_time:
        section("Monthly Revenue Trend")
        monthly = (
            fdf.groupby(fdf["Date"].dt.to_period("M"))["Sales"]
            .sum().reset_index()
        )
        monthly["Date"] = monthly["Date"].dt.to_timestamp()
        fig_monthly = px.line(
            monthly, x="Date", y="Sales",
            markers=True,
            color_discrete_sequence=[PALETTE[0]],
            title="Monthly Revenue Trend",
            labels={"Sales": "Revenue ($)", "Date": "Month"},
        )
        fig_monthly.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_monthly, use_container_width=True)

        col1, col2 = st.columns(2)

        section("Sales by Day of Week")
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_sales = (
            fdf.groupby("Day_Name")["Sales"].sum()
            .reindex(day_order, fill_value=0).reset_index()
        )
        fig_day = px.bar(
            day_sales, x="Day_Name", y="Sales",
            color="Sales", color_continuous_scale="Blues",
            title="Total Revenue by Day of Week",
            labels={"Sales": "Revenue ($)", "Day_Name": "Day"},
            text_auto=".2s",
        )
        fig_day.update_layout(plot_bgcolor="white")
        col1.plotly_chart(fig_day, use_container_width=True)

        qtr_sales = fdf.groupby("Quarter")["Sales"].sum().reset_index()
        qtr_sales["Quarter"] = "Q" + qtr_sales["Quarter"].astype(str)
        fig_qtr = px.bar(
            qtr_sales, x="Quarter", y="Sales",
            color="Quarter",
            color_discrete_sequence=PALETTE,
            title="Revenue by Quarter",
            labels={"Sales": "Revenue ($)"},
            text_auto=".2s",
        )
        fig_qtr.update_layout(plot_bgcolor="white", showlegend=False)
        col2.plotly_chart(fig_qtr, use_container_width=True)

        section("Monthly Revenue by Branch")
        monthly_branch = (
            fdf.groupby([fdf["Date"].dt.to_period("M"), "Branch"])["Sales"]
            .sum().reset_index()
        )
        monthly_branch["Date"] = monthly_branch["Date"].dt.to_timestamp()
        fig_mb = px.line(
            monthly_branch, x="Date", y="Sales",
            color="Branch",
            color_discrete_sequence=PALETTE,
            markers=True,
            title="Monthly Revenue by Branch",
            labels={"Sales": "Revenue ($)", "Date": "Month"},
        )
        fig_mb.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_mb, use_container_width=True)

        section("Monthly Revenue by Product Line")
        monthly_cat = (
            fdf.groupby([fdf["Date"].dt.to_period("M"), "Category"])["Sales"]
            .sum().reset_index()
        )
        monthly_cat["Date"] = monthly_cat["Date"].dt.to_timestamp()
        fig_mc = px.line(
            monthly_cat, x="Date", y="Sales",
            color="Category",
            color_discrete_sequence=PALETTE,
            markers=True,
            title="Monthly Revenue by Product Line",
            labels={"Sales": "Revenue ($)", "Date": "Month", "Category": "Product Line"},
        )
        fig_mc.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_mc, use_container_width=True)

    # ───────────────────────────────────────────────────────────────────────
    # TAB 6 — FINANCIALS  (new — uses cogs, Tax 5%, gross income)
    # ───────────────────────────────────────────────────────────────────────
    with tab_financials:
        section("Financial Performance Breakdown")

        if "gross income" in fdf.columns and "cogs" in fdf.columns:
            fin_branch = fdf.groupby("Branch").agg(
                Revenue=("Sales", "sum"),
                COGS=("cogs", "sum"),
                Tax=("Tax 5%", "sum"),
                Gross_Income=("gross income", "sum"),
            ).reset_index()

            col1, col2 = st.columns(2)

            fig_fin = px.bar(
                fin_branch.melt(id_vars="Branch", value_vars=["COGS", "Tax", "Gross_Income"],
                                var_name="Component", value_name="Amount"),
                x="Branch", y="Amount", color="Component", barmode="stack",
                color_discrete_sequence=PALETTE,
                title="Revenue Components by Branch (COGS + Tax + Gross Income)",
                labels={"Amount": "Amount ($)"},
            )
            fig_fin.update_layout(plot_bgcolor="white")
            col1.plotly_chart(fig_fin, use_container_width=True)

            fin_cat = fdf.groupby("Category").agg(
                Revenue=("Sales", "sum"),
                Gross_Income=("gross income", "sum"),
            ).reset_index()
            fin_cat["Gross_Margin_%"] = (fin_cat["Gross_Income"] / fin_cat["Revenue"] * 100).round(4)
            fig_gm = px.bar(
                fin_cat, x="Gross_Margin_%", y="Category",
                orientation="h", color="Gross_Margin_%",
                color_continuous_scale="Greens",
                title="Gross Margin % by Product Line",
                labels={"Gross_Margin_%": "Gross Margin %", "Category": "Product Line"},
                text_auto=".4f",
            )
            fig_gm.update_layout(plot_bgcolor="white", yaxis={"categoryorder": "total ascending"})
            col2.plotly_chart(fig_gm, use_container_width=True)

            section("Gross Income Trend Over Time")
            gi_monthly = (
                fdf.groupby(fdf["Date"].dt.to_period("M"))["gross income"]
                .sum().reset_index()
            )
            gi_monthly["Date"] = gi_monthly["Date"].dt.to_timestamp()
            fig_gi = px.line(
                gi_monthly, x="Date", y="gross income",
                markers=True,
                color_discrete_sequence=[PALETTE[1]],
                title="Monthly Gross Income Trend",
                labels={"gross income": "Gross Income ($)", "Date": "Month"},
            )
            fig_gi.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_gi, use_container_width=True)

            section("COGS vs Gross Income by Product Line")
            fin_cat2 = fdf.groupby("Category").agg(
                COGS=("cogs", "sum"),
                Gross_Income=("gross income", "sum"),
            ).reset_index().melt(id_vars="Category", var_name="Metric", value_name="Amount")
            fig_cogs = px.bar(
                fin_cat2, x="Category", y="Amount", color="Metric",
                barmode="group",
                color_discrete_sequence=PALETTE,
                title="COGS vs Gross Income by Product Line",
                labels={"Amount": "Amount ($)", "Category": "Product Line"},
            )
            fig_cogs.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_cogs, use_container_width=True)

            st.dataframe(
                fin_branch.style.format({
                    "Revenue":      "${:,.2f}",
                    "COGS":         "${:,.2f}",
                    "Tax":          "${:,.2f}",
                    "Gross_Income": "${:,.2f}",
                }),
                use_container_width=True,
            )
        else:
            st.info("Financial columns (cogs, Tax 5%, gross income) not found in the current filtered data.")

    # ───────────────────────────────────────────────────────────────────────
    # TAB 7 — RAW DATA
    # ───────────────────────────────────────────────────────────────────────
    with tab_data:
        section("Filtered Dataset")
        st.caption(f"Showing {len(fdf):,} rows after filters.")

        display_cols = [c for c in [
            "Invoice ID", "Date", "Time", "Branch", "City",
            "Customer Type", "Gender", "Category",
            "Quantity", "Unit price", "Tax 5%", "Sales",
            "cogs", "gross margin percentage", "gross income",
            "Payment", "Rating",
        ] if c in fdf.columns]

        fmt = {
            "Sales":                   "${:,.2f}",
            "Unit price":              "${:,.2f}",
            "Tax 5%":                  "${:,.4f}",
            "cogs":                    "${:,.2f}",
            "gross income":            "${:,.4f}",
            "gross margin percentage": "{:.4f}%",
        }
        st.dataframe(
            fdf[display_cols].style.format(fmt),
            use_container_width=True,
            height=420,
        )

        section("Descriptive Statistics")
        num_cols = [c for c in ["Quantity", "Unit price", "Tax 5%", "Sales", "cogs", "gross income", "Rating"]
                    if c in fdf.columns]
        st.dataframe(
            fdf[num_cols].describe().T.style.format("{:.4f}"),
            use_container_width=True,
        )

        section("Correlation Heatmap")
        corr = fdf[num_cols].corr()
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr.values, 3),
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig_corr.update_layout(title="Correlation Matrix — Numeric Features", height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

        csv_bytes = fdf[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered CSV",
            data=csv_bytes,
            file_name="filtered_sales.csv",
            mime="text/csv",
        )

    # ───────────────────────────────────────────────────────────────────────
    # TAB 8 — DATA CLEANING REPORT
    # ───────────────────────────────────────────────────────────────────────
    with tab_cleaning:
        section("Data Cleaning Report")

        cl1, cl2, cl3 = st.columns(3)
        metric_card(cl1, "Original Rows",    f"{report['rows_initial']:,}")
        metric_card(cl2, "Rows After Clean", f"{report['rows_final']:,}")
        metric_card(cl3, "Rows Removed",     f"{report['rows_removed']:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        cl4, cl5, cl6 = st.columns(3)
        metric_card(cl4, "Missing Values (before)", f"{report['missing_before']:,}")
        metric_card(cl5, "Missing Values (after)",  f"{report['missing_after']:,}")
        metric_card(cl6, "Duplicate Invoices",       f"{report['duplicates']:,}")

        st.markdown("<br>", unsafe_allow_html=True)
        cl7, cl8 = st.columns(2)
        metric_card(cl7, "Rows with Qty ≤ 0 (removed)",   f"{report['negative_qty']:,}")
        metric_card(cl8, "Sales Mismatch (auto-corrected)", f"{report['sales_mismatch']:,}")

        section("Cleaning Steps Performed")
        steps = [
            ("Strip whitespace",         "Trimmed leading/trailing spaces from all string columns."),
            ("Column normalisation",     "Renamed 'Product line' → 'Category' and 'Customer type' → 'Customer Type' for consistency."),
            ("Missing values",           "Dropped rows with missing Invoice ID, Date, Branch, Category, Quantity, or Unit price. "
                                         "Filled numeric NaNs with column medians; categorical NaNs with column modes."),
            ("Duplicate removal",        "Removed duplicate rows based on Invoice ID."),
            ("Type correction",          "Parsed Date as datetime; cast Quantity, Unit price, Rating, Tax 5%, Total, cogs, "
                                         "and gross income to numeric."),
            ("Negative/zero guard",      "Removed rows where Quantity ≤ 0 or Unit price ≤ 0."),
            ("Rating clipping",          "Clipped Rating to valid range [1, 10] (Kaggle dataset uses a 1–10 scale)."),
            ("Sales recalculation",      "Recomputed Sales = Quantity × Unit price × 1.05 (includes 5% tax) for every row; "
                                         "auto-corrected any mismatches against the existing Total column."),
            ("Time features",            "Derived Month, Month_Name, Day_Name, Quarter from Date column."),
        ]
        for title_, desc in steps:
            st.markdown(f"**{title_}** — {desc}")
            st.divider()

        section("Numeric Column Distributions After Cleaning")
        num_cols_dist = [c for c in ["Quantity", "Unit price", "Sales", "Rating"] if c in df.columns]
        fig_dist = make_subplots(rows=1, cols=len(num_cols_dist), subplot_titles=num_cols_dist)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(num_cols_dist))]
        for i, col in enumerate(num_cols_dist, 1):
            fig_dist.add_trace(
                go.Histogram(x=df[col], name=col, marker_color=colors[i - 1], showlegend=False),
                row=1, col=i,
            )
        fig_dist.update_layout(title="Distribution of Numeric Columns (Full Cleaned Dataset)", height=320)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # BUSINESS RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    with st.expander("💼 Business Recommendations", expanded=False):
        section("Data-Driven Business Recommendations")

        top_cat  = feda["top_category"][0]
        top_br   = feda["top_branch"][0]
        top_pay  = feda["top_payment"][0]
        low_day  = (
            fdf.groupby("Day_Name")["Sales"].sum()
            .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fill_value=0)
            .idxmin()
        )
        low_branch = fdf.groupby("Branch")["Sales"].sum().idxmin()
        low_rated_cat = fdf.groupby("Category")["Rating"].mean().idxmin()

        recommendation(
            f"**Expand {top_cat} inventory.** This is the highest-revenue product line. "
            "Consider wider SKU variety, bundle deals, and prominent shelf placement to maximise revenue."
        )
        recommendation(
            f"**Replicate Branch {top_br}'s best practices.** Analyse its staffing, layout, and promotions and roll "
            f"them out to underperforming Branch {low_branch} to close the revenue gap."
        )
        recommendation(
            f"**Double down on {top_pay}.** It is the most popular payment method. Negotiate lower transaction fees "
            "with the provider and offer exclusive cashback to further incentivise usage."
        )
        recommendation(
            f"**Drive footfall on {low_day}s.** Sales are lowest on {low_day}s. Introduce flash sales, day-specific "
            "loyalty points, or 'happy hour' discounts to smooth out weekly revenue variance."
        )
        recommendation(
            f"**Improve customer satisfaction for '{low_rated_cat}'.** It has the lowest average rating. "
            "Gather qualitative feedback, review product quality and pricing, and consider assortment refresh."
        )
        recommendation(
            "**Personalise for Members.** Members drive significant recurring revenue. "
            "Implement a tiered loyalty programme with personalised offers based on purchase history."
        )
        recommendation(
            "**Leverage gross income data.** Use the gross income and COGS breakdown in the Financials tab to "
            "identify which product lines and branches generate the healthiest margins, and prioritise them."
        )
        recommendation(
            "**Seasonal planning.** Use the monthly trend chart to pre-position stock for peak months and avoid "
            "overstocking in troughs — reducing wastage and improving margin."
        )


if __name__ == "__main__":
    main()
