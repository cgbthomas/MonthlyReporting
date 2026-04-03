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
    "Build quarterly store goals using prior-year baselines, "
    "YoY benchmarking, scenario growth targets, company rollups, "
    "actual FRS profit-center comparisons, goal allocations, and future quarter projections."
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
DEFAULT_FUTURE_SCENARIOS = [2.0, 4.0, 6.0, 8.0]

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

def get_status_icon(actual: Optional[float], goal: Optional[float]) -> str:
    if actual is None or goal is None or pd.isna(actual) or pd.isna(goal):
        return ""
    if actual >= goal:
        return "🟢"
    gap_pct = safe_div(goal - actual, goal)
    if gap_pct is not None and gap_pct <= 0.02:
        return "🟡"
    return "🔴"

def get_category_amount(categories: dict[str, float], name: str) -> float:
    return float(categories.get(name, 0.0) or 0.0)

def preserve_month_mix_or_weights(df: pd.DataFrame, qcol: str, m1col: str, m2col: str, m3col: str, weights: tuple[float, float, float]) -> pd.DataFrame:
    out = df.copy()
    w1, w2, w3 = weights

    has_months = all(col in out.columns for col in ["Base M1", "Base M2", "Base M3"])

    if has_months:
        month_total = out[["Base M1", "Base M2", "Base M3"]].sum(axis=1, min_count=1)
        use_mix = month_total.notna() & (month_total != 0)

        out[m1col] = None
        out[m2col] = None
        out[m3col] = None

        out.loc[use_mix, m1col] = out.loc[use_mix, qcol] * (out.loc[use_mix, "Base M1"] / month_total[use_mix])
        out.loc[use_mix, m2col] = out.loc[use_mix, qcol] * (out.loc[use_mix, "Base M2"] / month_total[use_mix])
        out.loc[use_mix, m3col] = out.loc[use_mix, qcol] * (out.loc[use_mix, "Base M3"] / month_total[use_mix])

        out.loc[~use_mix, m1col] = out.loc[~use_mix, qcol] * w1
        out.loc[~use_mix, m2col] = out.loc[~use_mix, qcol] * w2
        out.loc[~use_mix, m3col] = out.loc[~use_mix, qcol] * w3
    else:
        out[m1col] = out[qcol] * w1
        out[m2col] = out[qcol] * w2
        out[m3col] = out[qcol] * w3

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
    categories: dict[str, float]
    raw_text: str

def parse_frs_worker_sales_report(text: str) -> ParsedFRS:
    """
    Parses FRS Worker Sales by Product Category reports in either:
    1. multiline pasted format
    2. single-line exported format

    Pulls:
    - Center
    - Totals income
    - Public Service Payments income
    - Net Sales = Totals - PSP
    - Product category income lines
    """
    raw_text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    center = None
    total_sales = None
    categories = {}

    # Center
    for i, line in enumerate(lines):
        if line.lower() == "center:" and i + 1 < len(lines):
            if re.fullmatch(r"\d{3,5}", lines[i + 1]):
                center = lines[i + 1]
                break

    if not center:
        m = re.search(r"Center:\s*(\d{3,5})", text, flags=re.IGNORECASE)
        if m:
            center = m.group(1)

    # Pass 1: multiline format
    i = 0
    while i < len(lines):
        line = lines[i]

        if (line.startswith("+") or line.startswith("-")) and i + 3 < len(lines):
            cat_name = re.sub(r"^[\+\-]\s*", "", line).strip()
            income_candidate = lines[i + 3] if i + 3 < len(lines) else None

            if income_candidate and re.fullmatch(r"\$[\d,]+\.\d{2}", income_candidate):
                categories[cat_name] = money_to_float(income_candidate)
                i += 6
                continue

        if line.lower() == "totals" and i + 3 < len(lines):
            income_candidate = lines[i + 3]
            if re.fullmatch(r"\$[\d,]+\.\d{2}", income_candidate):
                total_sales = money_to_float(income_candidate)
                i += 6
                continue

        i += 1

    # Pass 2: single-line export fallback
    if not categories:
        category_matches = re.finditer(
            r"^[\+\-]\s*(.*?)\s+\d+\s+\d+\s+\$([\d,]+\.\d{2})\s+\$[\d,]+\.\d{2}\s+\$[\d,]+\.\d{2}\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for match in category_matches:
            cat_name = re.sub(r"\s+", " ", match.group(1)).strip()
            categories[cat_name] = money_to_float(match.group(2))

    if total_sales is None:
        m = re.search(
            r"^Totals\s+\d+\s+\d+\s+\$([\d,]+\.\d{2})\s+\$[\d,]+\.\d{2}\s+\$[\d,]+\.\d{2}\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if m:
            total_sales = money_to_float(m.group(1))

    psp_sales = categories.get("Public Service Payments", 0.0)

    net_sales = None
    if total_sales is not None:
        net_sales = total_sales - psp_sales

    return ParsedFRS(
        center=center,
        total_sales=total_sales,
        psp_sales=psp_sales,
        net_sales=net_sales,
        categories=categories,
        raw_text=raw_text,
    )

def parse_simple_monthly_sales(text: str) -> pd.DataFrame:
    """
    Accepts:
    1504,78742,78122,79753
    or
    1504,Yucaipa,78742,78122,79753
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
                    "Base M1": m1,
                    "Base M2": m2,
                    "Base M3": m3,
                    "Base Quarter": m1 + m2 + m3,
                })
            except Exception:
                pass
        elif len(parts) >= 5 and re.fullmatch(r"\d{3,5}", parts[0]):
            center = parts[0]
            try:
                m1 = money_to_float(parts[-3])
                m2 = money_to_float(parts[-2])
                m3 = money_to_float(parts[-1])
                rows.append({
                    "Center": center,
                    "Base M1": m1,
                    "Base M2": m2,
                    "Base M3": m3,
                    "Base Quarter": m1 + m2 + m3,
                })
            except Exception:
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
                    "Base Quarter": money_to_float(parts[-1])
                })
            except Exception:
                pass

    return pd.DataFrame(rows)

# ============================================================
# ACTUAL FRS PROFIT CENTER LOGIC
# ============================================================
def extract_actual_profit_centers(categories: dict[str, float]) -> dict[str, float]:
    """
    Packing = Retail Shipping Supplies + Packaging Materials +
              Packaging Service Fee + Office Supplies

    Print = Printing only
    """
    ups_shipping = get_category_amount(categories, "Shipping Charges (UPS)")
    mailbox = get_category_amount(categories, "Mailbox Service")
    notary = get_category_amount(categories, "Notary")

    packing = (
        get_category_amount(categories, "Retail Shipping Supplies")
        + get_category_amount(categories, "Packaging Materials")
        + get_category_amount(categories, "Packaging Service Fee")
        + get_category_amount(categories, "Office Supplies")
    )

    printing = get_category_amount(categories, "Printing")

    return {
        "UPS Shipping": ups_shipping,
        "Packing": packing,
        "Mailbox": mailbox,
        "Notary": notary,
        "Print": printing,
    }

def build_profit_center_comparison(base_pc_map: dict, current_pc_map: dict) -> pd.DataFrame:
    rows = []

    for center in CENTER_ORDER:
        store = CENTER_NAMES.get(center, "")
        base_pcs = base_pc_map.get(center, {})
        curr_pcs = current_pc_map.get(center, {})

        for pc in ["UPS Shipping", "Packing", "Mailbox", "Notary", "Print"]:
            base_val = float(base_pcs.get(pc, 0.0) or 0.0)
            curr_val = float(curr_pcs.get(pc, 0.0) or 0.0)
            dollar_var = curr_val - base_val
            pct_var = safe_div(dollar_var, base_val)

            rows.append({
                "Center": center,
                "Store": store,
                "Profit Center": pc,
                "Base Quarter": base_val,
                "Current Quarter": curr_val,
                "Variance $": dollar_var,
                "Variance %": pct_var,
            })

    return pd.DataFrame(rows)

# ============================================================
# CORE BUILDERS
# ============================================================
def build_base_table(input_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    for center in CENTER_ORDER:
        row = input_df[input_df["Center"] == center]
        if row.empty:
            out_rows.append({
                "Center": center,
                "Store": CENTER_NAMES.get(center, ""),
                "Base Quarter": None,
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
            "Base Quarter": base_q,
            "Current Quarter": curr_q,
            "YoY $ Variance": yoy_dollar,
            "YoY % Variance": yoy_pct,
        })

    return pd.DataFrame(out_rows)

def add_scenario_columns(df: pd.DataFrame, scenarios: list[float]) -> pd.DataFrame:
    out = df.copy()

    for scen in scenarios:
        pct = scen / 100.0
        qcol = f"Goal {scen:.1f}% Quarter"
        status_col = f"Goal {scen:.1f}% Status"

        out[qcol] = out["Base Quarter"] * (1 + pct)
        out[status_col] = out.apply(
            lambda r: get_status_icon(r.get("Current Quarter"), r.get(qcol)),
            axis=1
        )

    return out

def add_benchmark_projection(df: pd.DataFrame, benchmark_adjustment_pct: float) -> pd.DataFrame:
    out = df.copy()

    out["Benchmark %"] = out["YoY % Variance"] + (benchmark_adjustment_pct / 100.0)
    out["Benchmark Quarter Goal"] = out["Base Quarter"] * (1 + out["Benchmark %"])
    out["Benchmark Status"] = out.apply(
        lambda r: get_status_icon(r.get("Current Quarter"), r.get("Benchmark Quarter Goal")),
        axis=1
    )
    return out

def add_selected_goal_columns(df: pd.DataFrame, selected_goal_type: str) -> pd.DataFrame:
    out = df.copy()

    if selected_goal_type == "Benchmark":
        out["Selected Quarter Goal"] = out["Benchmark Quarter Goal"]
    else:
        scen = float(selected_goal_type)
        out["Selected Quarter Goal"] = out[f"Goal {scen:.1f}% Quarter"]

    return out

def build_profit_center_goal_allocation(df: pd.DataFrame, alloc: dict[str, float]) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        center = r["Center"]
        store = r["Store"]
        qgoal = r.get("Selected Quarter Goal")

        entry = {
            "Center": center,
            "Store": store,
            "Profit Center Sales Goal": qgoal,
        }

        for pc_name, pc_pct in alloc.items():
            pct = pc_pct / 100.0
            entry[pc_name] = qgoal * pct if pd.notna(qgoal) else None

        rows.append(entry)

    return pd.DataFrame(rows)

def build_company_rollup(df: pd.DataFrame, scenarios: list[float]) -> pd.DataFrame:
    rows = []

    base_total = df["Base Quarter"].sum(min_count=1)
    curr_total = df["Current Quarter"].sum(min_count=1)
    yoy_dollar = None
    yoy_pct = None
    if pd.notna(base_total) and pd.notna(curr_total):
        yoy_dollar = curr_total - base_total
        yoy_pct = safe_div(yoy_dollar, base_total)

    rows.append({"Metric": "Base Quarter Total", "Value": base_total})
    rows.append({"Metric": "Current Quarter Total", "Value": curr_total})
    rows.append({"Metric": "YoY Dollar Variance", "Value": yoy_dollar})
    rows.append({"Metric": "YoY % Variance", "Value": yoy_pct})

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

def build_future_goal_projection(df: pd.DataFrame, scenarios: list[float], month_weights: tuple[float, float, float]) -> pd.DataFrame:
    out = df.copy()

    for scen in scenarios:
        pct = scen / 100.0
        qcol = f"Goal {scen:.1f}% Quarter"
        m1col = f"Goal {scen:.1f}% M1"
        m2col = f"Goal {scen:.1f}% M2"
        m3col = f"Goal {scen:.1f}% M3"

        out[qcol] = out["Base Quarter"] * (1 + pct)
        out = preserve_month_mix_or_weights(out, qcol, m1col, m2col, m3col, month_weights)

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
    "Comparison Input Style",
    options=[
        "Monthly Store Tables",
        "Quarter Totals Only",
        "FRS Quarter Reports (one box per store)",
    ]
)

dist_mode = st.sidebar.radio(
    "Future Planner Monthly Distribution",
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

scenarios_csv = st.sidebar.text_input("Comparison Scenario % Targets", value="4,6,8,10")
try:
    scenario_values = [float(x.strip()) for x in scenarios_csv.split(",") if x.strip()]
    scenario_values = sorted(set(scenario_values))
except Exception:
    scenario_values = DEFAULT_SCENARIOS

future_scenarios_csv = st.sidebar.text_input("Future Planner % Targets", value="2,4,6,8")
try:
    future_scenario_values = [float(x.strip()) for x in future_scenarios_csv.split(",") if x.strip()]
    future_scenario_values = sorted(set(future_scenario_values))
except Exception:
    future_scenario_values = DEFAULT_FUTURE_SCENARIOS

benchmark_adjustment_pct = st.sidebar.number_input(
    "Benchmark Uplift Adjustment %",
    value=0.0,
    step=0.1,
    help="Adds an extra uplift on top of the YoY benchmark %."
)

selected_goal_type = st.sidebar.selectbox(
    "Selected Goal for Goal Allocation",
    options=["Benchmark"] + [f"{x:.1f}" for x in scenario_values],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Goal Allocation %**")

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
    st.sidebar.info("Goal allocation currently totals 95.0%.")
elif abs(pc_total - 100.0) > 0.0001:
    st.sidebar.warning(f"Goal allocation totals {pc_total:.1f}%.")

# ============================================================
# INPUT AREA
# ============================================================
st.subheader("1) Comparison Builder Input")

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
    with st.expander(f"{base_label} FRS Reports", expanded=True):
        st.markdown("Paste one full Worker Sales by Product Category report into each store box.")
        base_frs_inputs = {}
        base_cols = st.columns(2)

        for i, center in enumerate(CENTER_ORDER):
            with base_cols[i % 2]:
                base_frs_inputs[center] = st.text_area(
                    f"{center} - {CENTER_NAMES.get(center, '')}",
                    height=220,
                    key=f"base_frs_{center}",
                    placeholder=f"Paste the full Worker Sales by Product Category report for {center} here..."
                )

    with st.expander(f"{current_label} FRS Reports", expanded=True):
        st.markdown("Paste one full Worker Sales by Product Category report into each store box.")
        current_frs_inputs = {}
        current_cols = st.columns(2)

        for i, center in enumerate(CENTER_ORDER):
            with current_cols[i % 2]:
                current_frs_inputs[center] = st.text_area(
                    f"{center} - {CENTER_NAMES.get(center, '')}",
                    height=220,
                    key=f"current_frs_{center}",
                    placeholder=f"Paste the full Worker Sales by Product Category report for {center} here..."
                )

st.subheader("2) Future Goal Planner Input")

future_input_mode = st.radio(
    "Future Planner Input Style",
    options=["Monthly Store Tables", "Quarter Totals Only"],
    horizontal=True,
    key="future_input_mode"
)

if future_input_mode == "Monthly Store Tables":
    future_monthly_text = st.text_area(
        "Future Planner Base Monthly Sales by Store",
        height=220,
        placeholder="1504,78742,78122,79753\n5027,75411,81194,76831",
        key="future_monthly_text"
    )
else:
    future_quarter_text = st.text_area(
        "Future Planner Base Quarter Totals by Store",
        height=220,
        placeholder="1504,236617\n5027,233436",
        key="future_quarter_text"
    )

process = st.button("Build Quarter Goal Plan", type="primary", use_container_width=True)

# ============================================================
# PROCESSING
# ============================================================
if process:
    # ----------------------------
    # Comparison Builder
    # ----------------------------
    base_df = pd.DataFrame({"Center": CENTER_ORDER})
    current_df = pd.DataFrame({"Center": CENTER_ORDER})
    actual_pc_comparison_df = pd.DataFrame()

    if comparison_input_mode == "Monthly Store Tables":
        base_parsed_raw = parse_simple_monthly_sales(base_monthly_text)
        current_parsed_raw = parse_simple_monthly_sales(current_monthly_text)

        if base_parsed_raw.empty:
            diagnostics.append(f"{base_label} monthly table could not be parsed.")
        if current_parsed_raw.empty:
            diagnostics.append(f"{current_label} monthly table could not be parsed.")

        if not base_parsed_raw.empty:
            base_df = base_df.merge(
                base_parsed_raw[["Center", "Base Quarter"]],
                on="Center",
                how="left"
            )
        else:
            base_df["Base Quarter"] = None

        if not current_parsed_raw.empty:
            current_df = current_df.merge(
                current_parsed_raw.rename(columns={"Base Quarter": "Current Quarter"})[["Center", "Current Quarter"]],
                on="Center",
                how="left"
            )
        else:
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
        else:
            base_df["Base Quarter"] = None

        if not current_parsed.empty:
            current_df = current_df.merge(
                current_parsed.rename(columns={"Base Quarter": "Current Quarter"}),
                on="Center",
                how="left"
            )
        else:
            current_df["Current Quarter"] = None

    else:
        base_rows = []
        current_rows = []
        base_pc_map = {}
        current_pc_map = {}

        for center in CENTER_ORDER:
            rpt = (base_frs_inputs.get(center) or "").strip()
            if not rpt:
                continue

            parsed = parse_frs_worker_sales_report(rpt)

            diagnostics.append(f"{base_label} {center}: parsed {len(parsed.categories)} categories")

            if not parsed.center:
                diagnostics.append(f"{base_label} {center}: center not found in pasted report.")
            elif parsed.center != center:
                diagnostics.append(f"{base_label} {center}: pasted report appears to be for center {parsed.center}.")

            if parsed.net_sales is None:
                diagnostics.append(f"{base_label} {center}: totals not found.")

            base_rows.append({
                "Center": center,
                "Base Quarter": parsed.net_sales
            })

            base_pc_map[center] = extract_actual_profit_centers(parsed.categories)

        for center in CENTER_ORDER:
            rpt = (current_frs_inputs.get(center) or "").strip()
            if not rpt:
                continue

            parsed = parse_frs_worker_sales_report(rpt)

            diagnostics.append(f"{current_label} {center}: parsed {len(parsed.categories)} categories")

            if not parsed.center:
                diagnostics.append(f"{current_label} {center}: center not found in pasted report.")
            elif parsed.center != center:
                diagnostics.append(f"{current_label} {center}: pasted report appears to be for center {parsed.center}.")

            if parsed.net_sales is None:
                diagnostics.append(f"{current_label} {center}: totals not found.")

            current_rows.append({
                "Center": center,
                "Current Quarter": parsed.net_sales
            })

            current_pc_map[center] = extract_actual_profit_centers(parsed.categories)

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

        actual_pc_comparison_df = build_profit_center_comparison(base_pc_map, current_pc_map)

    unified = (
        pd.DataFrame({"Center": CENTER_ORDER})
        .merge(base_df, on="Center", how="left")
        .merge(current_df, on="Center", how="left")
    )

    if unified.empty:
        st.error("No usable comparison input data was found.")
        st.stop()

    base_table = build_base_table(unified)
    scenario_table = add_scenario_columns(base_table, scenario_values)
    scenario_table = add_benchmark_projection(scenario_table, benchmark_adjustment_pct)
    scenario_table = add_selected_goal_columns(scenario_table, selected_goal_type)
    company_rollup = build_company_rollup(scenario_table, scenario_values)
    quarterly_goal_alloc_df = build_profit_center_goal_allocation(scenario_table, pc_alloc)

    # ----------------------------
    # Future Goal Planner
    # ----------------------------
    future_base_df = pd.DataFrame({"Center": CENTER_ORDER})

    if future_input_mode == "Monthly Store Tables":
        future_parsed = parse_simple_monthly_sales(future_monthly_text)
        if future_parsed.empty:
            diagnostics.append("Future planner monthly table could not be parsed.")
            future_base_df["Base Quarter"] = None
            future_base_df["Base M1"] = None
            future_base_df["Base M2"] = None
            future_base_df["Base M3"] = None
        else:
            future_base_df = future_base_df.merge(future_parsed, on="Center", how="left")
    else:
        future_parsed = parse_simple_quarter_sales(future_quarter_text)
        if future_parsed.empty:
            diagnostics.append("Future planner quarter table could not be parsed.")
            future_base_df["Base Quarter"] = None
        else:
            future_base_df = future_base_df.merge(future_parsed, on="Center", how="left")
        future_base_df["Base M1"] = None
        future_base_df["Base M2"] = None
        future_base_df["Base M3"] = None

    future_base_df["Store"] = future_base_df["Center"].map(CENTER_NAMES)
    future_base_df = future_base_df[["Center", "Store", "Base M1", "Base M2", "Base M3", "Base Quarter"]]
    future_goal_df = build_future_goal_projection(future_base_df, future_scenario_values, target_weights)

    # ----------------------------
    # TABS
    # ----------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Benchmark Builder",
        "Scenario Goals",
        "Profit Center Breakdown",
        "Company Rollup",
        "Bonus Reference",
        "Future Goal Planner",
    ])

    with tab1:
        st.subheader("Benchmark Builder")
        bench_cols = [
            "Center", "Store",
            "Base Quarter",
            "Current Quarter",
            "YoY $ Variance", "YoY % Variance",
            "Benchmark %", "Benchmark Quarter Goal", "Benchmark Status",
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
                f"{quarter_label} Benchmark Goal {fmt_money(r['Benchmark Quarter Goal'])} {r.get('Benchmark Status', '')}"
            )
        st.text_area("Copy / Paste Benchmark Summary", "\n".join(summary_lines), height=240)

    with tab2:
        st.subheader("Scenario Goals")
        scen_cols = ["Center", "Store", "Base Quarter", "Current Quarter", "YoY % Variance"]
        for scen in scenario_values:
            scen_cols += [
                f"Goal {scen:.1f}% Quarter",
                f"Goal {scen:.1f}% Status",
            ]
        scen_df = scenario_table[[c for c in scen_cols if c in scenario_table.columns]].copy()
        st.dataframe(scen_df, use_container_width=True)

        st.caption(
            f"Selected goal for goal allocation: "
            f"{'Benchmark' if selected_goal_type == 'Benchmark' else selected_goal_type + '% scenario'}"
        )

    with tab3:
        st.subheader("Profit Center Breakdown")

        if comparison_input_mode == "FRS Quarter Reports (one box per store)" and not actual_pc_comparison_df.empty:
            st.markdown("**Quarterly Profit Center Comparison (Actual FRS Data)**")
            st.dataframe(actual_pc_comparison_df, use_container_width=True)

        st.markdown("**Quarterly Goal Allocation**")
        st.dataframe(quarterly_goal_alloc_df, use_container_width=True)

    with tab4:
        st.subheader("Company Rollup")
        st.dataframe(company_rollup, use_container_width=True)

        compact_rows = []
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

    with tab6:
        st.subheader("Future Goal Planner")
        st.caption("Use last year's quarter data to project future quarter goals at 2%, 4%, 6%, and 8% (or your selected targets).")

        future_cols = ["Center", "Store", "Base Quarter"]
        has_months = future_input_mode == "Monthly Store Tables"
        if has_months:
            future_cols += ["Base M1", "Base M2", "Base M3"]

        for scen in future_scenario_values:
            future_cols += [f"Goal {scen:.1f}% Quarter"]
            if has_months:
                future_cols += [f"Goal {scen:.1f}% M1", f"Goal {scen:.1f}% M2", f"Goal {scen:.1f}% M3"]

        st.dataframe(future_goal_df[[c for c in future_cols if c in future_goal_df.columns]], use_container_width=True)

        summary_lines = []
        for _, r in future_goal_df.iterrows():
            pieces = [f"{r['Center']} {r['Store']}: Base {fmt_money(r['Base Quarter'])}"]
            for scen in future_scenario_values:
                pieces.append(f"{scen:.1f}% {fmt_money(r.get(f'Goal {scen:.1f}% Quarter'))}")
            summary_lines.append(" | ".join(pieces))
        st.text_area("Copy / Paste Future Goal Summary", "\n".join(summary_lines), height=240)

    if diagnostics:
        with st.expander("Diagnostics", expanded=False):
            for msg in diagnostics:
                st.write(f"- {msg}")

    # ----------------------------
    # EXCEL EXPORT
    # ----------------------------
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            scenario_table.to_excel(writer, sheet_name="Scenario Goals", index=False)
            quarterly_goal_alloc_df.to_excel(writer, sheet_name="Quarterly Goal Allocation", index=False)
            company_rollup.to_excel(writer, sheet_name="Company Rollup", index=False)
            future_goal_df.to_excel(writer, sheet_name="Future Goal Planner", index=False)

            if not actual_pc_comparison_df.empty:
                actual_pc_comparison_df.to_excel(writer, sheet_name="PC YoY Comparison", index=False)

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
    except ModuleNotFoundError:
        st.warning("Excel export is unavailable because openpyxl is not installed. Add openpyxl to requirements.txt.")

    with st.expander("How to use this planner", expanded=False):
        st.markdown(
            f"""
### Comparison Builder
1. Enter your baseline sales for **{base_label}** by store.
2. Enter your comparison sales for **{current_label}** by store.
3. The app calculates **YoY dollar** and **YoY % variance**.
4. The app then builds:
   - a **Benchmark** goal based on actual YoY performance plus any uplift
   - fixed **scenario goals** like {", ".join([f"{x:.1f}%" for x in scenario_values])}
5. Choose which goal type should drive the **quarterly goal allocation**.

### FRS Profit Center Comparison
If you use **FRS Quarter Reports**, the app also builds an **actual profit-center YoY comparison**:
- UPS Shipping = Shipping Charges (UPS)
- Packing = Retail Shipping Supplies + Packaging Materials + Packaging Service Fee + Office Supplies
- Mailbox = Mailbox Service
- Notary = Notary
- Print = Printing

### Future Goal Planner
Use last year's quarter data to project future goals.
- Supports **Monthly Store Tables**
- Supports **Quarter Totals Only**
- Projects **{", ".join([f"{x:.1f}%" for x in future_scenario_values])}**
- If monthly data exists, the app preserves the original month mix
- Otherwise, it uses the selected monthly distribution
"""
        )
