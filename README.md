# 🛒 Supermarket Sales Analysis

An end-to-end **Data Analysis & Interactive Dashboard** project built with Python, Pandas, Plotly, and Streamlit.

---

## 📌 Project Overview

This project performs a complete data analysis pipeline on the Kaggle Supermarket Sales dataset:

| Stage | What happens |
|---|---|
| **Data Cleaning** | Whitespace strip, column normalisation, missing-value imputation, duplicate removal, type correction, negative-value guard |
| **Sales Calculation** | `Sales = Quantity × Unit price × 1.05` — verified against the existing `Total` column |
| **Exploratory Data Analysis (EDA)** | KPIs, product-line/branch/customer/payment/time/financial breakdowns, correlation analysis |
| **Interactive Dashboard** | Streamlit app with sidebar filters, 8 tabs, 25+ Plotly charts |
| **Business Recommendations** | Data-driven, auto-generated actionable insights |

---

## 📂 Project Structure

```
Supermarket_Sales_Analysis/
├── app.py                  # Single-file Streamlit application (back-end + front-end)
├── supermarket_sales.csv   # Dataset (1,000 transactions — Kaggle Supermarket Sales)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📊 Dataset

### Source
**Kaggle Supermarket Sales Dataset**
> Aung Pyae. (2019). *Supermarket sales*. Kaggle.
> https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales

### Schema

| Column | Type | Description |
|---|---|---|
| `Invoice ID` | string | Unique transaction identifier |
| `Branch` | string | Store branch code (A, B, C) |
| `City` | string | City of the branch (Yangon, Mandalay, Naypyitaw) |
| `Customer type` | string | `Member` or `Normal` |
| `Gender` | string | `Male` or `Female` |
| `Product line` | string | Product category / line (6 categories) |
| `Unit price` | float | Price per unit ($) |
| `Quantity` | int | Units purchased (1–10) |
| `Tax 5%` | float | 5% tax applied to the sale |
| `Total` | float | Total price including tax (`cogs + Tax 5%`) |
| `Date` | date | Transaction date (Jan–Mar 2019) |
| `Time` | string | Transaction time (HH:MM) |
| `Payment` | string | Payment method (Cash, Credit card, Ewallet) |
| `cogs` | float | Cost of goods sold (`Quantity × Unit price`) |
| `gross margin percentage` | float | Fixed at 4.761904762% |
| `gross income` | float | Gross profit (`Total − cogs`) |
| `Rating` | float | Customer satisfaction score (1.0–10.0) |

> **Note:** The app normalises `Product line` → `Category` and `Customer type` → `Customer Type` during the cleaning step for internal consistency.

### Statistics
- **Rows:** 1,000 transactions
- **Columns:** 17
- **Branches:** 3 (A — Yangon, B — Mandalay, C — Naypyitaw)
- **Product Lines:** 6 (Health and beauty, Electronic accessories, Home and lifestyle, Sports and travel, Food and beverages, Fashion accessories)
- **Payment Methods:** Cash, Credit card, Ewallet
- **Date Range:** January 2019 – March 2019
- **Rating Scale:** 1.0 – 10.0

---

## 🧹 Data Cleaning Steps

1. **Strip whitespace** — trim leading/trailing spaces from all string columns.
2. **Column normalisation** — rename `Product line` → `Category`; `Customer type` → `Customer Type`.
3. **Missing values** — drop rows with missing `Invoice ID`, `Date`, `Branch`, `Category`, `Quantity`, or `Unit price`; fill numeric NaNs with column medians; categorical NaNs with column modes.
4. **Duplicate removal** — drop rows with duplicate `Invoice ID`.
5. **Type correction** — parse `Date` as `datetime64`; cast `Quantity`, `Unit price`, `Rating`, `Tax 5%`, `Total`, `cogs`, and `gross income` to numeric.
6. **Negative/zero guard** — remove rows where `Quantity ≤ 0` or `Unit price ≤ 0`.
7. **Rating clipping** — clip `Rating` to valid range [1, 10].
8. **Sales recalculation** — recompute `Sales = Quantity × Unit price × 1.05`; auto-correct any mismatches with the existing `Total` column.
9. **Time feature engineering** — derive `Month`, `Month_Name`, `Day_Name`, and `Quarter` from `Date`.

---

## 📈 Dashboard Features

### Sidebar Filters
- **Branch** — filter by one or more branches (A / B / C)
- **Product Line** — filter by product category
- **Payment Method** — Cash / Credit card / Ewallet
- **Customer Type** — Member / Normal
- **Gender** — Male / Female
- **Date Range** — interactive date picker

### Tabs

| Tab | Contents |
|---|---|
| 📈 **Overview** | 11 KPI cards (incl. gross margin + gross income), branch bar, payment donut, product-line bar, key insights |
| 📦 **Product Lines** | Revenue/qty/rating bars per product line, bubble scatter, summary table |
| 🏪 **Branches** | Branch vs city bars, sunburst product-line mix, stacked payment bar, branch table |
| 👥 **Customers** | Member vs Normal pie, gender analysis, product-line spend by segment & gender, rating histograms + box plots |
| 📅 **Time Trends** | Monthly revenue line, day-of-week bar, quarterly bar, monthly by branch & product line |
| 💰 **Financials** | COGS + Tax + Gross Income stacked bar, gross margin % bar, gross income trend, COGS vs gross income by product line |
| 🗃️ **Raw Data** | Filterable table, descriptive stats, correlation heatmap (7 numeric cols), CSV download |
| 🧹 **Data Cleaning** | Cleaning report cards, 9-step log, numeric distribution histograms |

### Business Recommendations *(expandable panel)*
Auto-generated from live filtered data — covering inventory, branch benchmarking, payment optimisation, low-traffic day campaigns, product satisfaction improvement, loyalty programme, financial margin focus, and seasonal planning.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher

### Installation

```bash
# 1. Clone / download the project
cd Supermarket_Sales_Analysis

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io/) | ≥ 1.35 | Web front-end & interactivity |
| [Pandas](https://pandas.pydata.org/) | ≥ 2.1 | Data loading, cleaning, aggregation |
| [NumPy](https://numpy.org/) | ≥ 1.26 | Numerical operations |
| [Plotly](https://plotly.com/python/) | ≥ 5.22 | Interactive charts & visualisations |

---

## 🗺️ Analysis Highlights

- **Top Product Line** — identified via grouped `Sales` sum across 6 product lines.
- **Best-Performing Branch** — ranked by total revenue with city-level breakdown.
- **Gross Margin Analysis** — COGS vs gross income breakdown per branch and product line (unique to this dataset).
- **Payment Methods** — ranked by transaction count (Cash / Credit card / Ewallet).
- **Customer Ratings** — distribution analysis (1–10 scale) per branch and product line.
- **Time Patterns** — monthly (Jan–Mar 2019), day-of-week, and quarterly revenue trends.
- **Financial Deep-Dive** — dedicated Financials tab surfacing tax, COGS, gross income, and gross margin %.

---

## 📜 Dataset Citation

> Aung Pyae. (2019). *Supermarket sales* [Data set]. Kaggle.
> https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales
>
> File: `supermarket_sales.csv` · 1,000 records · 17 features
> Transactions span January 2019 – March 2019 across three branches in Yangon, Mandalay, and Naypyitaw, Myanmar.

---

## 📄 License

This project is released for educational use. Dataset is sourced from Kaggle under its original licence terms.
