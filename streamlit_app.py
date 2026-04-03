import io
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Quarter Goal Planner", layout="wide")
st.title("Quarter Goal Planner")
st.caption(
    "Build quarterly and monthly store goals using prior-year baselines, "
    "YoY benchmarking, scenario growth targets, company rollups, "
    "profit-center allocations, and bonus reference tables."
)

# ============================================================
# CONSTANTS
# ============================================================
CENTER_NAMES = {
    "1504": "Yucaipa",
    "5027": "Beaumont",
    "5052": "Ontario",
    "5255": "Summit",
    "5778": "Citrus",
    "6176": "Sierra",
    "6769": "Loma Linda",
    "7261": "Ayala",
}
CENTER_ORDER = ["1504", "5027", "5052", "5255", "5778", "6176", "6769", "7261"]

DEFAULT_SCENARIOS = [4.0, 6.0, 8.0, 10.0]

DEFAULT_PC_ALLOC = {
    "UPS Shipping": 25.0,
    "Packing": 15.0,
    "Mailbox": 15.0,
    "Notary": 25.0,
    "Print": 15.0,
}

AREA_MANAGER_BONUS = {
    4.0: {"Sales": 300, "Labor Company (20%)": 300, "Total": 600},
    6.0: {"Sales": 600, "Labor Company (20%)": 300, "Total": 900},
    8.0: {"Sales": 1200, "Labor Company (20%)": 300, "Total": 1500},
    10.0: {"Sales": 1500, "Labor Company (20%)": 300, "Total": 1800},
}

GENERAL_MANAGER_BONUS = {
    4.0: {"Sales": 250, "PC Growth": 300, "Total": 550},
    6.0: {"Sales": 500, "PC Growth": 300, "Total": 800},
    8.0: {"Sales": 950, "PC Growth": 300, "Total": 1250},
    10.0: {"Sales": 1250, "PC Growth": 300, "Total": 1550},
}

CENTER_MANAGER_BONUS = {
    4.0: {"Sales": 200, "Labor (Individual)": 300, "Total": 500},
    6.0: {"Sales": 400, "Labor (Individual)": 300, "Total": 700},
    8.0: {"Sales": 950, "Labor (Individual)": 300, "Total": 1250},
    10.0: {"Sales": 1250, "Labor (Individual)": 300, "Total": 1550},
}

# ============================================================
# HELPERS
# ============================================================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()

def money_to_float(s: str) -> float:
    return float(str(s).replace("$", "").replace(",", "").strip())

def fmt_money(x):
    return "—" if x is None or pd.isna(x) else f"${x:,.2f}"

def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:.1%}"

def safe_div(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return a / b

def split_big_paste(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    seps = ["\n-----\n", "\n=====\n", "\n---\n"]
    for sep in seps:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if parts:
                return parts

    return [raw]

def build_currency_format_map(columns):
    out = {}
    for c in columns:
        if any(
            key in c.lower()
            for key in [
                "sales", "goal", "variance", "increase", "shipping", "packing",
                "mailbox", "notary", "print", "total", "base"
            ]
        ) and "%" not in c:
            out[c] = "${:,.2f}"
    return out

# ============================================================
# PARSERS
# ============================================================
@dataclass
class ParsedFRS:
    center: Optional[str]
    total_sales: Optional[float]
    psp_sales: float
    net_sales: Optional[float]
    raw_text: str

def parse_frs_worker_sales_report(text: str) -> ParsedFRS:
    """
    Pulls:
    - Center
    - Totals income
    - Public Service Payments income
    - Net Sales = Totals - PSP
    """
    text = normalize_text(text)

    center = None
    m = re.search(r"Center:\s*(\d{3,5})", text, flags=re.IGNORECASE)
    if m:
        center = m.group(1)

    if not center:
        m = re.search(r"Center:\s*\n\s*(\d{3,5})", text, flags=re.IGNORECASE)
        if m:
            center = m.group(1)

    total_sales = None
    m = re.search(
        r"^Totals\s+\d+\s+\d+\s+\$([\d,]+\.\d{2})\s+\$[\d,]+\.\d{2}\s+\$[\d,]+\.\d{2}\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        total_sales = money_to_float(m.group(1))

    psp_sales = 0.0
    m = re.search(
        r"^[\+\-]\s*Public Service Payments\s+\d+\s+\d+\s+\$([\d,]+\.\d{2})\s+\$[\d,]+\.\d{2}\s+\$[\d,]+\.\d{2}\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        psp_sales = money_to_float(m.group(1))

    net_sales = None
    if total_sales is not None:
        net_sales = total_sales - psp_sales

    return ParsedFRS(
        center=center,
        total_sales=total_sales,
        psp_sales=psp_sales,
        net_sales=net_sales,
        raw_text=text,
    )

def parse_simple_monthly_sales(text: str) -> pd.DataFrame:
    """
    Accepts lines like:
    1504,78742,78122,79753
    5027,75411,81194,76831

    or with store name:
    1504,Yucaipa,78742,78122,79753

    Returns: Center, M1, M2, M3, Quarter Total
    """
    text = normalize_text(text)
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and re.fullmatch(r"\d{3,5}", parts[0]):
            center = parts[0]
            try:
                m1 = money_to_float(parts[1])
                m2 = money_to_float(parts[2])
                m3 = money_to_float(parts[3])
                rows.append({
                    "Center": center,
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "Quarter Total": m1 + m2 + m3
                })
            except:
                pass
        elif len(parts) >= 5 and re.fullmatch(r"\d{3,5}", parts[0]):
            center = parts[0]
            try:
                m1 = money_to_float(parts[-3])
                m2 = money_to_float(parts[-2])
                m3 = money_to_float(parts[-1])
                rows.append({
                    "Center": center,
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "Quarter Total": m1 + m2 + m3
                })
            except:
                pass

    return pd.DataFrame(rows)

def parse_simple_quarter_sales(text: str) -> pd.DataFrame:
    """
    Accepts:
    1504,236617
    5027,233436
    """
    text = normalize_text(text)
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and re.fullmatch(r"\d{3,5}", parts[0]):
            try:
                rows.append({
                    "Center": parts[0],
                    "Quarter Total": money_to_float(parts[-1])
                })
            except:
                pass

    return pd.DataFrame(rows)

# ============================================================
# CORE BUILDERS
# ============================================================
def build_base_table(input_df: pd.DataFrame, comparison_mode: str) -> pd.DataFrame:
    """
    comparison_mode:
      - 'Quarter Total'
      - 'Month-by-Month Total'
    """
    out_rows = []

    for center in CENTER_ORDER:
        row = input_df[input_df["Center"] == center]
        if row.empty:
            out_rows.append({
                "Center": center,
                "Store": CENTER_NAMES.get(center, ""),
                "Base M1": None,
                "Base M2": None,
                "Base M3": None,
                "Base Quarter": None,
                "Current M1": None,
                "Current M2": None,
                "Current M3": None,
                "Current Quarter": None,
                "YoY $ Variance": None,
                "YoY % Variance": None,
            })
            continue

        r = row.iloc[0]
        base_q = r.get("Base Quarter")
        curr_q = r.get("Current Quarter")

        yoy_dollar = None
        yoy_pct = None
        if pd.notna(base_q) and pd.notna(curr_q):
            yoy_dollar = curr_q - base_q
            yoy_pct = safe_div(yoy_dollar, base_q)

        out_rows.append({
            "Center": center,
            "Store": CENTER_NAMES.get(center, ""),
            "Base M1": r.get("Base M1"),
            "Base M2": r.get("Base M2"),
            "Base M3": r.get("Base M3"),
            "Base Quarter": base_q,
            "Current M1": r.get("Current M1"),
            "Current M2": r.get("Current M2"),
            "Current M3": r.get("Current M3"),
            "Current Quarter": curr_q,
            "YoY $ Variance": yoy_dollar,
            "YoY % Variance": yoy_pct,
        })

    return pd.DataFrame(out_rows)

def add_scenario_columns(df: pd.DataFrame, scenarios: list[float], target_month_weights: tuple[float, float, float]) -> pd.DataFrame:
    out = df.copy()
    w1, w2, w3 = target_month_weights

    for scen in scenarios:
        pct = scen / 100.0
        qcol = f"Goal {scen:.1f}% Quarter"
        m1col = f"Goal {scen:.1f}% M1"
        m2col = f"Goal {scen:.1f}% M2"
        m3col = f"Goal {scen:.1f}% M3"

        out[qcol] = out["Base Quarter"] * (1 + pct)
        out[m1col] = out[qcol] * w1
        out[m2col] = out[qcol] * w2
        out[m3col] = out[qcol] * w3

    return out

def add_benchmark_projection(df: pd.DataFrame, benchmark_adjustment_pct: float, target_month_weights: tuple[float, float, float]) -> pd.DataFrame:
    out = df.copy()
    w1, w2, w3 = target_month_weights

    out["Benchmark %"] = out["YoY % Variance"] + (benchmark_adjustment_pct / 100.0)
    out["Benchmark Quarter Goal"] = out["Base Quarter"] * (1 + out["Benchmark %"])
    out["Benchmark M1 Goal"] = out["Benchmark Quarter Goal"] * w1
    out["Benchmark M2 Goal"] = out["Benchmark Quarter Goal"] * w2
    out["Benchmark M3 Goal"] = out["Benchmark Quarter Goal"] * w3
    return out

def build_company_rollup(df: pd.DataFrame, scenarios: list[float]) -> pd.DataFrame:
    rows = []

    base_total = df["Base Quarter"].sum(min_count=1)
    curr_total = df["Current Quarter"].sum(min_count=1)
    yoy_dollar = None
    yoy_pct = None
    if pd.notna(base_total) and pd.notna(curr_total):
        yoy_dollar = curr_total - base_total
        yoy_pct = safe_div(yoy_dollar, base_total)

    rows.append({
        "Metric": "Base Quarter Total",
        "Value": base_total,
    })
    rows.append({
        "Metric": "Current Quarter Total",
        "Value": curr_total,
    })
    rows.append({
        "Metric": "YoY Dollar Variance",
        "Value": yoy_dollar,
    })
    rows.append({
        "Metric": "YoY % Variance",
        "Value": yoy_pct,
    })

    if "Benchmark Quarter Goal" in df.columns:
        bench_total = df["Benchmark Quarter Goal"].sum(min_count=1)
        bench_increase = None if pd.isna(bench_total) or pd.isna(base_total) else bench_total - base_total
        rows.append({"Metric": "Benchmark Quarter Goal", "Value": bench_total})
        rows.append({"Metric": "Benchmark Dollar Increase", "Value": bench_increase})

    for scen in scenarios:
        qcol = f"Goal {scen:.1f}% Quarter"
        if qcol in df.columns:
            scen_total = df[qcol].sum(min_count=1)
            scen_increase = None if pd.isna(scen_total) or pd.isna(base_total) else scen_total - base_total
            rows.append({"Metric": f"{scen:.1f}% Scenario Quarter Goal", "Value": scen_total})
            rows.append({"Metric": f"{scen:.1f}% Scenario Dollar Increase", "Value": scen_increase})

    return pd.DataFrame(rows)

def build_profit_center_breakdown(df: pd.DataFrame, selected_goal_col: str, alloc: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_rows = []
    m_rows = []

    for _, r in df.iterrows():
        center = r["Center"]
        store = r["Store"]
        qgoal = r.get(selected_goal_col)

        m1 = r.get("Selected M1 Goal")
        m2 = r.get("Selected M2 Goal")
        m3 = r.get("Selected M3 Goal")

        q_entry = {
            "Center": center,
            "Store": store,
            "Profit Center Sales Goal": qgoal,
        }
        m_entry = {
            "Center": center,
            "Store": store,
            "Monthly Profit Center Sales Goal": (qgoal / 3.0 if pd.notna(qgoal) else None),
        }

        for pc_name, pc_pct in alloc.items():
            pct = pc_pct / 100.0
            q_entry[pc_name] = qgoal * pct if pd.notna(qgoal) else None
            m_entry[pc_name] = (qgoal / 3.0) * pct if pd.notna(qgoal) else None

        # richer monthly detail in separate fields
        for pc_name, pc_pct in alloc.items():
            pct = pc_pct / 100.0
            m_entry[f"{pc_name} M1"] = m1 * pct if pd.notna(m1) else None
            m_entry[f"{pc_name} M2"] = m2 * pct if pd.notna(m2) else None
            m_entry[f"{pc_name} M3"] = m3 * pct if pd.notna(m3) else None

        q_rows.append(q_entry)
        m_rows.append(m_entry)

    q_df = pd.DataFrame(q_rows)
    m_df = pd.DataFrame(m_rows)
    return q_df, m_df

def add_selected_goal_columns(df: pd.DataFrame, selected_goal_type: str) -> pd.DataFrame:
    out = df.copy()

    if selected_goal_type == "Benchmark":
        out["Selected Quarter Goal"] = out["Benchmark Quarter Goal"]
        out["Selected M1 Goal"] = out["Benchmark M1 Goal"]
        out["Selected M2 Goal"] = out["Benchmark M2 Goal"]
        out["Selected M3 Goal"] = out["Benchmark M3 Goal"]
    else:
        # selected_goal_type expected like "8.0"
        scen = float(selected_goal_type)
        out["Selected Quarter Goal"] = out[f"Goal {scen:.1f}% Quarter"]
        out["Selected M1 Goal"] = out[f"Goal {scen:.1f}% M1"]
        out["Selected M2 Goal"] = out[f"Goal {scen:.1f}% M2"]
        out["Selected M3 Goal"] = out[f"Goal {scen:.1f}% M3"]

    return out

def build_bonus_reference_tables():
    def to_df(d: dict[float, dict[str, float]], label: str):
        rows = []
        for pct, vals in d.items():
            rows.append({
                "Role": label,
                "Target %": pct / 100.0,
                **vals
            })
        return pd.DataFrame(rows)

    return (
        to_df(AREA_MANAGER_BONUS, "Area Manager"),
        to_df(GENERAL_MANAGER_BONUS, "General Manager"),
        to_df(CENTER_MANAGER_BONUS, "Center Manager"),
    )

# ============================================================
# SIDEBAR / SETTINGS
# ============================================================
st.sidebar.header("Planner Settings")

quarter_label = st.sidebar.text_input("Quarter Label", value="Q2 2026")
base_label = st.sidebar.text_input("Base Period Label", value="Q2 2025")
current_label = st.sidebar.text_input("Comparison Period Label", value="Q1 2026")

comparison_input_mode = st.sidebar.radio(
    "Input Style",
    options=[
        "Monthly Store Tables",
        "Quarter Totals Only",
        "FRS Quarter Reports (aggregate net sales)",
    ]
)

dist_mode = st.sidebar.radio(
    "Target Monthly Distribution",
    options=["Equal Split", "Custom Weights"],
    index=0
)

if dist_mode == "Equal Split":
    target_weights = (0.3333, 0.3333, 0.3334)
else:
    mw1 = st.sidebar.number_input("Month 1 Weight %", min_value=0.0, max_value=100.0, value=33.3, step=0.1)
    mw2 = st.sidebar.number_input("Month 2 Weight %", min_value=0.0, max_value=100.0, value=33.3, step=0.1)
    mw3 = st.sidebar.number_input("Month 3 Weight %", min_value=0.0, max_value=100.0, value=33.4, step=0.1)
    target_weights = (mw1 / 100.0, mw2 / 100.0, mw3 / 100.0)

if abs(sum(target_weights) - 1.0) > 0.0001:
    st.sidebar.warning(f"Monthly weights total {sum(target_weights):.2%}, not 100.0%.")

scenarios_csv = st.sidebar.text_input("Scenario % Targets", value="4,6,8,10")
try:
    scenario_values = [float(x.strip()) for x in scenarios_csv.split(",") if x.strip()]
    scenario_values = sorted(set(scenario_values))
except:
    scenario_values = DEFAULT_SCENARIOS

benchmark_adjustment_pct = st.sidebar.number_input(
    "Benchmark Uplift Adjustment %",
    value=0.0,
    step=0.1,
    help="Adds an extra uplift on top of the YoY benchmark %."
)

selected_goal_type = st.sidebar.selectbox(
    "Selected Goal for Profit Center Breakdown",
    options=["Benchmark"] + [f"{x:.1f}" for x in scenario_values],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Profit Center Allocation %**")

pc_shipping = st.sidebar.number_input("UPS Shipping %", value=DEFAULT_PC_ALLOC["UPS Shipping"], step=0.1)
pc_packing = st.sidebar.number_input("Packing %", value=DEFAULT_PC_ALLOC["Packing"], step=0.1)
pc_mailbox = st.sidebar.number_input("Mailbox %", value=DEFAULT_PC_ALLOC["Mailbox"], step=0.1)
pc_notary = st.sidebar.number_input("Notary %", value=DEFAULT_PC_ALLOC["Notary"], step=0.1)
pc_print = st.sidebar.number_input("Print %", value=DEFAULT_PC_ALLOC["Print"], step=0.1)

pc_alloc = {
    "UPS Shipping": pc_shipping,
    "Packing": pc_packing,
    "Mailbox": pc_mailbox,
    "Notary": pc_notary,
    "Print": pc_print,
}

pc_total = sum(pc_alloc.values())
if abs(pc_total - 95.0) < 0.0001:
    st.sidebar.info("Profit center allocation currently totals 95.0%, matching the example mix shown in the PDF.")
elif abs(pc_total - 100.0) > 0.0001:
    st.sidebar.warning(f"Profit center allocation totals {pc_total:.1f}%.")

# ============================================================
# INPUT AREA
# ============================================================
st.subheader("1) Input Sales Data")

diagnostics = []

if comparison_input_mode == "Monthly Store Tables":
    c1, c2 = st.columns(2)
    with c1:
        base_monthly_text = st.text_area(
            f"{base_label} Monthly Sales by Store",
            height=260,
            placeholder="1504,78742,78122,79753\n5027,75411,81194,76831"
        )
    with c2:
        current_monthly_text = st.text_area(
            f"{current_label} Monthly Sales by Store",
            height=260,
            placeholder="1504,80632,79997,81667\n5027,83857,90288,85436"
        )

elif comparison_input_mode == "Quarter Totals Only":
    c1, c2 = st.columns(2)
    with c1:
        base_quarter_text = st.text_area(
            f"{base_label} Quarter Totals by Store",
            height=260,
            placeholder="1504,236617\n5027,233436"
        )
    with c2:
        current_quarter_text = st.text_area(
            f"{current_label} Quarter Totals by Store",
            height=260,
            placeholder="1504,242296\n5027,259581"
        )

else:
    c1, c2 = st.columns(2)
    with c1:
        base_frs_text = st.text_area(
            f"{base_label} FRS Reports",
            height=260,
            placeholder="Paste one or more full Worker Sales by Product Category reports separated by -----"
        )
    with c2:
        current_frs_text = st.text_area(
            f"{current_label} FRS Reports",
            height=260,
            placeholder="Paste one or more full Worker Sales by Product Category reports separated by -----"
        )

process = st.button("Build Quarter Goal Plan", type="primary", use_container_width=True)

# ============================================================
# PROCESSING
# ============================================================
if process:
    # -----------------------------------
    # Build unified input dataframe
    # -----------------------------------
    base_df = pd.DataFrame({"Center": CENTER_ORDER})
    current_df = pd.DataFrame({"Center": CENTER_ORDER})

    if comparison_input_mode == "Monthly Store Tables":
        base_parsed = parse_simple_monthly_sales(base_monthly_text)
        current_parsed = parse_simple_monthly_sales(current_monthly_text)

        if base_parsed.empty:
            diagnostics.append(f"{base_label} monthly table could not be parsed.")
        if current_parsed.empty:
            diagnostics.append(f"{current_label} monthly table could not be parsed.")

        if not base_parsed.empty:
            base_df = base_df.merge(base_parsed, on="Center", how="left")
            base_df = base_df.rename(columns={
                "M1": "Base M1",
                "M2": "Base M2",
                "M3": "Base M3",
                "Quarter Total": "Base Quarter",
            })
        else:
            base_df["Base M1"] = None
            base_df["Base M2"] = None
            base_df["Base M3"] = None
            base_df["Base Quarter"] = None

        if not current_parsed.empty:
            current_df = current_df.merge(current_parsed, on="Center", how="left")
            current_df = current_df.rename(columns={
                "M1": "Current M1",
                "M2": "Current M2",
                "M3": "Current M3",
                "Quarter Total": "Current Quarter",
            })
        else:
            current_df["Current M1"] = None
            current_df["Current M2"] = None
            current_df["Current M3"] = None
            current_df["Current Quarter"] = None

    elif comparison_input_mode == "Quarter Totals Only":
        base_parsed = parse_simple_quarter_sales(base_quarter_text)
        current_parsed = parse_simple_quarter_sales(current_quarter_text)

        if base_parsed.empty:
            diagnostics.append(f"{base_label} quarter table could not be parsed.")
        if current_parsed.empty:
            diagnostics.append(f"{current_label} quarter table could not be parsed.")

        if not base_parsed.empty:
            base_df = base_df.merge(base_parsed, on="Center", how="left")
            base_df = base_df.rename(columns={"Quarter Total": "Base Quarter"})
        else:
            base_df["Base Quarter"] = None

        base_df["Base M1"] = None
        base_df["Base M2"] = None
        base_df["Base M3"] = None

        if not current_parsed.empty:
            current_df = current_df.merge(current_parsed, on="Center", how="left")
            current_df = current_df.rename(columns={"Quarter Total": "Current Quarter"})
        else:
            current_df["Current Quarter"] = None

        current_df["Current M1"] = None
        current_df["Current M2"] = None
        current_df["Current M3"] = None

    else:
        # FRS quarter reports - aggregate one report per store for each comparison set
        base_reports = split_big_paste(base_frs_text)
        current_reports = split_big_paste(current_frs_text)

        base_rows = []
        for i, rpt in enumerate(base_reports, start=1):
            parsed = parse_frs_worker_sales_report(rpt)
            if not parsed.center:
                diagnostics.append(f"{base_label} FRS report #{i}: center not found.")
            if parsed.net_sales is None:
                diagnostics.append(f"{base_label} FRS report #{i} ({parsed.center or 'Unknown'}): totals not found.")
            if parsed.center:
                base_rows.append({"Center": parsed.center, "Base Quarter": parsed.net_sales})

        current_rows = []
        for i, rpt in enumerate(current_reports, start=1):
            parsed = parse_frs_worker_sales_report(rpt)
            if not parsed.center:
                diagnostics.append(f"{current_label} FRS report #{i}: center not found.")
            if parsed.net_sales is None:
                diagnostics.append(f"{current_label} FRS report #{i} ({parsed.center or 'Unknown'}): totals not found.")
            if parsed.center:
                current_rows.append({"Center": parsed.center, "Current Quarter": parsed.net_sales})

        base_parsed = pd.DataFrame(base_rows)
        current_parsed = pd.DataFrame(current_rows)

        if not base_parsed.empty:
            base_df = base_df.merge(base_parsed, on="Center", how="left")
        else:
            base_df["Base Quarter"] = None

        if not current_parsed.empty:
            current_df = current_df.merge(current_parsed, on="Center", how="left")
        else:
            current_df["Current Quarter"] = None

        base_df["Base M1"] = None
        base_df["Base M2"] = None
        base_df["Base M3"] = None
        current_df["Current M1"] = None
        current_df["Current M2"] = None
        current_df["Current M3"] = None

    unified = (
        pd.DataFrame({"Center": CENTER_ORDER})
        .merge(base_df, on="Center", how="left")
        .merge(current_df, on="Center", how="left")
    )

    if unified.empty:
        st.error("No usable input data was found.")
        st.stop()

    base_table = build_base_table(unified, comparison_input_mode)
    scenario_table = add_scenario_columns(base_table, scenario_values, target_weights)
    scenario_table = add_benchmark_projection(scenario_table, benchmark_adjustment_pct, target_weights)
    scenario_table = add_selected_goal_columns(scenario_table, selected_goal_type)
    company_rollup = build_company_rollup(scenario_table, scenario_values)
    quarterly_pc_df, monthly_pc_df = build_profit_center_breakdown(
        scenario_table,
        selected_goal_col="Selected Quarter Goal",
        alloc=pc_alloc
    )

    # -----------------------------------
    # TABS
    # -----------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Benchmark Builder",
        "Scenario Goals",
        "Profit Center Breakdown",
        "Company Rollup",
        "Bonus Reference",
    ])

    with tab1:
        st.subheader("Benchmark Builder")
        bench_cols = [
            "Center", "Store",
            "Base M1", "Base M2", "Base M3", "Base Quarter",
            "Current M1", "Current M2", "Current M3", "Current Quarter",
            "YoY $ Variance", "YoY % Variance",
            "Benchmark %", "Benchmark Quarter Goal",
            "Benchmark M1 Goal", "Benchmark M2 Goal", "Benchmark M3 Goal",
        ]
        bench_df = scenario_table[[c for c in bench_cols if c in scenario_table.columns]].copy()
        st.dataframe(bench_df, use_container_width=True)

        summary_lines = []
        for _, r in scenario_table.iterrows():
            summary_lines.append(
                f"{r['Center']} {r['Store']}: "
                f"{base_label} {fmt_money(r['Base Quarter'])} | "
                f"{current_label} {fmt_money(r['Current Quarter'])} | "
                f"YoY {fmt_money(r['YoY $ Variance'])} ({fmt_pct(r['YoY % Variance'])}) | "
                f"Benchmark {fmt_pct(r['Benchmark %'])} | "
                f"{quarter_label} Benchmark Goal {fmt_money(r['Benchmark Quarter Goal'])}"
            )
        st.text_area("Copy / Paste Benchmark Summary", "\n".join(summary_lines), height=240)

    with tab2:
        st.subheader("Scenario Goals")
        scen_cols = ["Center", "Store", "Base Quarter", "Current Quarter", "YoY % Variance"]
        for scen in scenario_values:
            scen_cols += [
                f"Goal {scen:.1f}% Quarter",
                f"Goal {scen:.1f}% M1",
                f"Goal {scen:.1f}% M2",
                f"Goal {scen:.1f}% M3",
            ]
        scen_df = scenario_table[[c for c in scen_cols if c in scenario_table.columns]].copy()
        st.dataframe(scen_df, use_container_width=True)

        st.caption(
            f"Selected goal for profit-center breakdown: "
            f"{'Benchmark' if selected_goal_type == 'Benchmark' else selected_goal_type + '% scenario'}"
        )

    with tab3:
        st.subheader("Profit Center Breakdown")
        st.markdown("**Quarterly Profit Center Breakdown**")
        st.dataframe(quarterly_pc_df, use_container_width=True)

        st.markdown("**Monthly Profit Center Breakdown**")
        st.dataframe(monthly_pc_df, use_container_width=True)

    with tab4:
        st.subheader("Company Rollup")
        st.dataframe(company_rollup, use_container_width=True)

        # Company totals by scenario in a compact grid
        compact_rows = []
        base_total = company_rollup.loc[company_rollup["Metric"] == "Base Quarter Total", "Value"]
        base_total = base_total.iloc[0] if not base_total.empty else None

        for scen in scenario_values:
            q_metric = f"{scen:.1f}% Scenario Quarter Goal"
            inc_metric = f"{scen:.1f}% Scenario Dollar Increase"
            q_val = company_rollup.loc[company_rollup["Metric"] == q_metric, "Value"]
            inc_val = company_rollup.loc[company_rollup["Metric"] == inc_metric, "Value"]
            compact_rows.append({
                "Scenario": f"{scen:.1f}%",
                "Quarter Goal": q_val.iloc[0] if not q_val.empty else None,
                "Dollar Increase": inc_val.iloc[0] if not inc_val.empty else None,
            })

        if "Benchmark Quarter Goal" in company_rollup["Metric"].values:
            bq = company_rollup.loc[company_rollup["Metric"] == "Benchmark Quarter Goal", "Value"]
            bi = company_rollup.loc[company_rollup["Metric"] == "Benchmark Dollar Increase", "Value"]
            compact_rows.insert(0, {
                "Scenario": "Benchmark",
                "Quarter Goal": bq.iloc[0] if not bq.empty else None,
                "Dollar Increase": bi.iloc[0] if not bi.empty else None,
            })

        st.markdown("**Company Scenario Summary**")
        st.dataframe(pd.DataFrame(compact_rows), use_container_width=True)

    with tab5:
        st.subheader("Bonus Reference")
        am_df, gm_df, cm_df = build_bonus_reference_tables()

        st.markdown("**Area Manager Bonus Reference**")
        st.dataframe(am_df, use_container_width=True)

        st.markdown("**General Manager Bonus Reference**")
        st.dataframe(gm_df, use_container_width=True)

        st.markdown("**Center Manager Bonus Reference**")
        st.dataframe(cm_df, use_container_width=True)

    # -----------------------------------
    # Diagnostics
    # -----------------------------------
    if diagnostics:
        with st.expander("Diagnostics", expanded=False):
            for msg in diagnostics:
                st.write(f"- {msg}")

    # -----------------------------------
    # Export
    # -----------------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        scenario_table.to_excel(writer, sheet_name="Scenario Goals", index=False)
        quarterly_pc_df.to_excel(writer, sheet_name="Quarterly PC Breakdown", index=False)
        monthly_pc_df.to_excel(writer, sheet_name="Monthly PC Breakdown", index=False)
        company_rollup.to_excel(writer, sheet_name="Company Rollup", index=False)

        am_df, gm_df, cm_df = build_bonus_reference_tables()
        am_df.to_excel(writer, sheet_name="Bonus Area Manager", index=False)
        gm_df.to_excel(writer, sheet_name="Bonus General Manager", index=False)
        cm_df.to_excel(writer, sheet_name="Bonus Center Manager", index=False)

    st.download_button(
        "Download Excel Workbook",
        data=output.getvalue(),
        file_name="quarter_goal_planner.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # -----------------------------------
    # Guidance
    # -----------------------------------
    # -----------------------------------
    # Guidance
    # -----------------------------------
    with st.expander("How to use this planner", expanded=False):
        st.markdown(
            f"""
1. Enter your baseline sales for **{base_label}** by store.
2. Enter your comparison sales for **{current_label}** by store.
3. The app calculates **YoY dollar** and **YoY % variance**.
4. The app then builds:
   - a **Benchmark** goal based on actual YoY performance plus any uplift
   - fixed **scenario goals** like {", ".join([f"{x:.1f}%" for x in scenario_values])}
5. Choose which goal type should drive the **profit-center breakdown**.
6. Review:
   - Benchmark Builder
   - Scenario Goals
   - Profit Center Breakdown
   - Company Rollup
   - Bonus Reference

This mirrors the structure of the quarter planning PDF: store-level scenarios, company totals, profit-center allocation, and bonus thresholds.
"""
        )
