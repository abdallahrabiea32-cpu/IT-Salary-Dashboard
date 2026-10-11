from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st



st.set_page_config(
    page_title="IT Survey | Abdullah Rabea",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp {
        background: linear-gradient(135deg, #F4F3F0 0%, #ECEDEF 48%, #F8F7F4 100%);
        color: #292A2D;
      }
      [data-testid="stHeader"] { background: rgba(244,243,240,0.92); }
      .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}
      h1, h2, h3, h4 { color: #292A2D; letter-spacing: -0.025em; }
      p, label, span, li { color: #414247; }
      h1 { border-bottom: 2px solid #C6A15B; padding-bottom: 0.45rem; }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, #FFFFFF 0%, #E9EAEC 100%);
        border: 1px solid #C7C9CC;
        border-top: 3px solid #C6A15B;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(40, 41, 45, 0.055);
      }
      [data-testid="stMetricLabel"] { color: #696A70 !important; }
      [data-testid="stMetricValue"] { color: #292A2D !important; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E7E8EA 0%, #F2F1EE 100%);
        border-right: 1px solid #C6A15B;
      }
      div[data-testid="stTabs"] button {font-weight: 600; color: #62636A;}
      div[data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom-color: #C6A15B !important; color: #292A2D !important;
      }
      .small-note {font-size: 0.88rem; opacity: 0.82;}
      div[data-testid="stAlert"] { border-radius: 10px; }
      .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #34353A 0%, #55575D 100%);
        color: #FFFFFF; border: 1px solid #C6A15B; border-radius: 9px;
      }
      .stButton button:hover, .stDownloadButton button:hover {
        background: linear-gradient(135deg, #C6A15B 0%, #A98543 100%);
        color: #FFFFFF; border-color: #A98543;
      }
      [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #D0D1D3; border-radius: 10px; overflow: hidden;
      }
      hr { border-color: #D1D2D4; }
    </style>
    """,
    unsafe_allow_html=True,
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#C6A15B", "#8F969D", "#34353A", "#D5D7DA", "#A98543", "#666A70", "#E8DDBF"]
px.defaults.color_continuous_scale = [[0, "#E7E8EA"], [0.48, "#A9ADB2"], [0.76, "#C6A15B"], [1, "#34353A"]]

TECH_COLUMN = "Your main technology / programming language"
OTHER_TECH_COLUMN = "Other technologies/programming languages you use often"
TOTAL_EXP = "Total years of experience"
GERMANY_EXP = "Years of experience in Germany"
REQUIRED_COLUMNS = [
    "Timestamp", "Age", "Gender", "City", "Position", TOTAL_EXP,
    GERMANY_EXP, "Seniority level", TECH_COLUMN, OTHER_TECH_COLUMN
]


# -----------------------------
# Data loading
# -----------------------------
def read_csv_safely(uploaded_file):
    """Read a CSV with a few common encodings."""
    if uploaded_file is None:
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def find_local_csv():

    base_dir = Path(__file__).resolve().parent
    names = [
        "IT Salary Survey EU  2020-selected-columns.csv",
        "IT Salary Survey EU 2020-selected-columns.csv",
        "IT Salary Survey EU  2020-selected-columns(1).csv",
        "IT Salary Survey EU 2020-selected-columns(1).csv",
    ]
    candidates = []
    for name in names:
        candidates.extend([Path(name), base_dir / name, base_dir / "data" / name])
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


# -----------------------------
# Cleaning helpers
# -----------------------------
TECH_ALIASES = {
    "python": "Python",
    "py": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "c# .net": "C# .NET",
    "c#.net": "C# .NET",
}


def clean_experience_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    if text in {"", "nan", "none", "unknown"}:
        return np.nan
    if "less than year" in text or "less than 1 year" in text:
        return 0.5
    month_match = re.fullmatch(r"(?:about\s*)?(\d+(?:\.\d+)?)\s*months?", text)
    if month_match:
        return float(month_match.group(1)) / 12
    year_match = re.fullmatch(r"(?:about\s*)?(\d+(?:\.\d+)?)\s*(?:years?|year)?", text)
    if year_match:
        return float(year_match.group(1))
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_technology(value):
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return np.nan
    if text.lower() in TECH_ALIASES:
        return TECH_ALIASES[text.lower()]
    parts = re.split(r"\s*[,/;|]+\s*", text)
    if len(parts) > 1:
        cleaned = []
        for part in parts:
            part = part.strip()
            if part:
                cleaned.append(TECH_ALIASES.get(part.lower(), part))
        cleaned = list(dict.fromkeys(cleaned))
        return " / ".join(cleaned) if cleaned else np.nan
    return text


def clean_dataset(raw_df):
    """Apply the cleaning approach documented in the project notebook."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Support the known technology column even if spacing differs slightly.
    tech_candidates = [c for c in df.columns if "main technology" in c.lower() or "programming language" in c.lower() and "other" not in c.lower()]
    tech_column = TECH_COLUMN if TECH_COLUMN in df.columns else (tech_candidates[0] if tech_candidates else None)
    other_candidates = [c for c in df.columns if "other technologies" in c.lower()]
    other_column = OTHER_TECH_COLUMN if OTHER_TECH_COLUMN in df.columns else (other_candidates[0] if other_candidates else None)

    if tech_column is None:
        raise ValueError(
            "Could not find the main technology column. Please upload the CSV used in the project notebook."
        )

    # Trim text values.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Parse timestamp, with day-first format as used in the notebook.
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")

    # Parse numeric fields.
    for col in ["Age", TOTAL_EXP, GERMANY_EXP]:
        if col in df.columns:
            if col in (TOTAL_EXP, GERMANY_EXP):
                df[col] = df[col].apply(clean_experience_value)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize technology spelling and seniority variants.
    df[tech_column] = df[tech_column].apply(normalize_technology)
    if other_column:
        df[other_column] = df[other_column].apply(normalize_technology)
    if "Seniority level" in df.columns:
        seniority_mapping = {
            "C-Level": "C-level",
            "C-level executive manager": "C-level",
            "c-level": "C-level",
            "intern": "Intern",
            "student": "Student",
            "No level": "No level",
            "no level": "No level",
        }
        df["Seniority level"] = df["Seniority level"].replace(seniority_mapping)

    # The project notebook excludes the fully empty Position column.
    if "Position" in df.columns and df["Position"].isna().all():
        df = df.drop(columns=["Position"])

    # Flag implausible values before filling missing values.
    if TOTAL_EXP in df.columns:
        df.loc[df[TOTAL_EXP] > 50, TOTAL_EXP] = np.nan
    if TOTAL_EXP in df.columns and GERMANY_EXP in df.columns:
        bad_germany = df[GERMANY_EXP] > df[TOTAL_EXP]
        df.loc[bad_germany, GERMANY_EXP] = np.nan

    # Impute numeric missing values with median, and categorical missing values as Unknown.
    numeric_cols = [c for c in ["Age", TOTAL_EXP, GERMANY_EXP] if c in df.columns]
    for col in numeric_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = [
        c for c in ["Gender", "City", "Seniority level", tech_column, other_column]
        if c and c in df.columns
    ]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
        df[col] = df[col].replace("", "Unknown")

    # Store the detected column names for downstream charts.
    df.attrs["tech_column"] = tech_column
    df.attrs["other_tech_column"] = other_column
    return df


def style_chart(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=65, b=12),
        title=dict(x=0.02, xanchor="left"),
        legend_title_text="",
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#D9DADD")
    return fig


# -----------------------------
# Header and sidebar
# -----------------------------
st.title("IT Salary Survey EU 2020")
st.subheader("Respondent Profile & Exploratory Data Analysis")
st.caption(
    "Data Science Mid Project • Prepared by Abdullah Rabea Mohamed • "
    "This dashboard analyzes respondent profiles, experience, location, seniority and technology."
)

with st.sidebar:
    st.header("Dashboard Controls")
    uploaded = st.file_uploader("Upload the project CSV", type=["csv"])
    st.caption("Expected: the selected-columns CSV used in the notebook.")
    st.divider()
    st.markdown("**Explore**")
    st.caption("Use the filters below after the dataset is loaded.")
    st.markdown("---")
    st.caption("Important: the available dataset has no salary column, so this dashboard does not estimate or compare salaries.")

raw = None
source_label = None
if uploaded is not None:
    try:
        raw = read_csv_safely(uploaded)
        source_label = uploaded.name
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
elif find_local_csv() is not None:
    local_path = find_local_csv()
    try:
        raw = pd.read_csv(local_path)
        source_label = str(local_path)
    except Exception as exc:
        st.error(f"Could not read {local_path}: {exc}")

if raw is None:
    st.info(
        "Upload the CSV file from your project to open the interactive dashboard. "
        "The notebook was provided, but the CSV dataset itself was not included with the uploads."
    )
    st.markdown("### What this dashboard will show")
    st.markdown(
        "- Summary KPIs and respondent profile\n"
        "- Age and experience distributions\n"
        "- City, gender, seniority and technology breakdowns\n"
        "- Relationships between age, experience and seniority\n"
        "- Data-quality notes, methodology and CSV export"
    )
    st.stop()

try:
    df = clean_dataset(raw)
except Exception as exc:
    st.error(f"Data cleaning failed: {exc}")
    st.write("Columns detected in the uploaded file:", list(raw.columns))
    st.stop()

tech_column = df.attrs.get("tech_column", TECH_COLUMN)
other_tech_column = df.attrs.get("other_tech_column", OTHER_TECH_COLUMN)

required_for_dashboard = ["Age", "Gender", "City", TOTAL_EXP, GERMANY_EXP, "Seniority level", tech_column]
missing_required = [c for c in required_for_dashboard if c not in df.columns]
if missing_required:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing_required))
    st.stop()

# Raw-data quality summary, before cleaning.
raw_missing = raw.isna().sum().sort_values(ascending=False)
clean_missing = df.isna().sum().sort_values(ascending=False)
raw_duplicates = int(raw.duplicated().sum())
clean_duplicates = int(df.duplicated().sum())

with st.sidebar:
    st.subheader("Filters")
    cities = sorted(df["City"].dropna().astype(str).unique().tolist())
    selected_cities = st.multiselect("City", cities, default=[])
    genders = sorted(df["Gender"].dropna().astype(str).unique().tolist())
    selected_genders = st.multiselect("Gender", genders, default=[])
    seniorities = sorted(df["Seniority level"].dropna().astype(str).unique().tolist())
    selected_seniorities = st.multiselect("Seniority level", seniorities, default=[])
    min_age = int(np.floor(df["Age"].min()))
    max_age = int(np.ceil(df["Age"].max()))
    age_range = st.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    exp_min = float(df[TOTAL_EXP].min())
    exp_max = float(df[TOTAL_EXP].max())
    experience_range = st.slider(
        "Total experience (years)",
        min_value=float(np.floor(exp_min)),
        max_value=float(np.ceil(exp_max)),
        value=(float(np.floor(exp_min)), float(np.ceil(exp_max))),
        step=1.0,
    )

filtered = df.copy()
if selected_cities:
    filtered = filtered[filtered["City"].isin(selected_cities)]
if selected_genders:
    filtered = filtered[filtered["Gender"].isin(selected_genders)]
if selected_seniorities:
    filtered = filtered[filtered["Seniority level"].isin(selected_seniorities)]
filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]
filtered = filtered[filtered[TOTAL_EXP].between(experience_range[0], experience_range[1])]

if filtered.empty:
    st.warning("No respondents match the current filters. Adjust the filters in the sidebar.")
    st.stop()

# -----------------------------
# KPI row
# -----------------------------
respondents = len(filtered)
mean_age = filtered["Age"].mean()
median_exp = filtered[TOTAL_EXP].median()
top_city = filtered["City"].mode().iloc[0] if not filtered["City"].mode().empty else "N/A"
top_tech = filtered[tech_column].mode().iloc[0] if not filtered[tech_column].mode().empty else "N/A"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Respondents", f"{respondents:,}", help="Number of survey responses after applying the filters.")
m2.metric("Average age", f"{mean_age:.1f} years")
m3.metric("Median experience", f"{median_exp:.1f} years")
m4.metric("Most common city", str(top_city))

st.caption(f"Data source: `{source_label}` • Showing {respondents:,} of {len(df):,} cleaned respondents.")

tab_overview, tab_demographics, tab_experience, tab_technology, tab_quality = st.tabs(
    ["Overview", "Demographics & Cities", "Experience & Seniority", "Technology", "Data Quality & Methodology"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.markdown("### Respondent snapshot")
    left, right = st.columns(2)
    with left:
        age_fig = px.histogram(
            filtered, x="Age", nbins=20,
            title="Age distribution", labels={"Age": "Age (years)"}
        )
        st.plotly_chart(style_chart(age_fig, 390), use_container_width=True)
    with right:
        gender_display = filtered["Gender"].astype(str).replace({
            "Unknown": "Other / not specified",
            "Diverse": "Other / not specified",
        })
        gender_counts = gender_display.value_counts().rename_axis("Gender").reset_index(name="Respondents")
        gender_fig = px.pie(
            gender_counts, names="Gender", values="Respondents", hole=0.55,
            title="Gender composition", color="Gender",
            color_discrete_map={
                "Male": "#34353A", "Female": "#C6A15B",
                "Other / not specified": "#8F969D",
            }
        )
        gender_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_chart(gender_fig, 390), use_container_width=True)
        st.caption("Unknown and Diverse are combined into “Other / not specified” because these groups are very small.")

    st.markdown("### Key findings in the current selection")
    most_senior = filtered["Seniority level"].value_counts()
    most_technology = filtered[tech_column].value_counts()
    col_a, col_b, col_c = st.columns(3)
    col_a.info(f"**Largest city group**\n\n{top_city} ({filtered['City'].value_counts().iloc[0]:,} respondents)")
    col_b.info(f"**Most common seniority**\n\n{most_senior.index[0]} ({most_senior.iloc[0]:,} respondents)")
    col_c.info(f"**Most common main technology**\n\n{most_technology.index[0]} ({most_technology.iloc[0]:,} respondents)")
    st.warning(
        "Interpretation note: the dataset describes survey respondents, not necessarily the full IT workforce. "
        "Relationships in this dashboard are associations and do not prove causation."
    )

# -----------------------------
# Demographics
# -----------------------------
with tab_demographics:
    st.markdown("### Where are respondents based?")
    c1, c2 = st.columns(2)
    with c1:
        city_counts = filtered["City"].value_counts().head(10).sort_values().reset_index()
        city_counts.columns = ["City", "Respondents"]
        city_fig = px.bar(
            city_counts, x="Respondents", y="City", orientation="h",
            text="Respondents", title="Top 10 cities by respondent count"
        )
        st.plotly_chart(style_chart(city_fig, 460), use_container_width=True)
    with c2:
        gender_seniority = filtered.copy()
        gender_seniority["Gender display"] = gender_seniority["Gender"].astype(str).replace({
            "Unknown": "Other / not specified",
            "Diverse": "Other / not specified",
        })
        gender_seniority = (
            gender_seniority.groupby(["Seniority level", "Gender display"], dropna=False)
            .size().reset_index(name="Respondents")
        )
        gender_seniority["Within-level %"] = (
            gender_seniority["Respondents"] /
            gender_seniority.groupby("Seniority level")["Respondents"].transform("sum") * 100
        )
        # Horizontal 100% stacked bars make the comparison easier to read.
        gender_fig = px.bar(
            gender_seniority, x="Within-level %", y="Seniority level",
            color="Gender display", barmode="stack", orientation="h",
            title="Gender composition within each seniority level",
            labels={"Within-level %": "Share within level (%)", "Gender display": "Gender"},
            category_orders={"Gender display": ["Male", "Female", "Other / not specified"]},
            color_discrete_map={
                "Male": "#34353A", "Female": "#C6A15B",
                "Other / not specified": "#8F969D",
            }
        )
        gender_fig.update_layout(
            xaxis_range=[0, 100],
            xaxis_ticksuffix="%",
            legend_title_text="Gender",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(style_chart(gender_fig, 460), use_container_width=True)
        st.caption("Each horizontal bar totals 100% for that seniority level. Unknown and Diverse are combined; interpret small groups cautiously.")
    st.caption("Small groups can produce unstable percentages. Check the respondent count before comparing categories.")

# -----------------------------
# Experience
# -----------------------------
with tab_experience:
    st.markdown("### Experience patterns")
    c1, c2 = st.columns(2)
    with c1:
        exp_fig = px.histogram(
            filtered, x=TOTAL_EXP, nbins=20,
            title="Total professional experience",
            labels={TOTAL_EXP: "Total experience (years)"}
        )
        st.plotly_chart(style_chart(exp_fig, 400), use_container_width=True)
    with c2:
        seniority_medians = (
            filtered.groupby("Seniority level", as_index=False)[TOTAL_EXP]
            .median().sort_values(TOTAL_EXP)
        )
        seniority_medians = seniority_medians.rename(columns={TOTAL_EXP: "Median experience (years)"})
        seniority_fig = px.bar(
            seniority_medians, x="Seniority level", y="Median experience (years)",
            text="Median experience (years)", title="Typical experience by seniority",
            labels={"Median experience (years)": "Median total experience (years)"},
            color_discrete_sequence=["#C6A15B"]
        )
        seniority_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        seniority_fig.update_layout(yaxis_rangemode="tozero")
        seniority_fig.update_xaxes(tickangle=25)
        st.plotly_chart(style_chart(seniority_fig, 400), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        scatter_fig = px.scatter(
            filtered, x="Age", y=TOTAL_EXP, color="Seniority level",
            hover_data=["City", "Gender", tech_column],
            title="Age vs total experience",
            labels={"Age": "Age (years)", TOTAL_EXP: "Total experience (years)"}
        )
        st.plotly_chart(style_chart(scatter_fig, 420), use_container_width=True)
    with c4:
        germany_by_seniority = (
            filtered.groupby("Seniority level", as_index=False)[GERMANY_EXP]
            .median().sort_values(GERMANY_EXP)
        )
        germany_by_seniority = germany_by_seniority.rename(
            columns={GERMANY_EXP: "Median Germany experience (years)"}
        )
        germany_fig = px.bar(
            germany_by_seniority, x="Seniority level", y="Median Germany experience (years)",
            text="Median Germany experience (years)",
            title="Typical experience in Germany by seniority",
            labels={"Median Germany experience (years)": "Median experience in Germany (years)"},
            color_discrete_sequence=["#8F969D"]
        )
        germany_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        germany_fig.update_layout(yaxis_rangemode="tozero")
        germany_fig.update_xaxes(tickangle=25)
        st.plotly_chart(style_chart(germany_fig, 420), use_container_width=True)

    corr_value = filtered[["Age", TOTAL_EXP]].corr().iloc[0, 1] if len(filtered) > 1 else np.nan
    if pd.notna(corr_value):
        st.metric("Age–experience correlation (Pearson r)", f"{corr_value:.3f}")
    st.caption("Correlation describes a linear association; it does not establish that one variable causes another.")

# -----------------------------
# Technology
# -----------------------------
with tab_technology:
    st.markdown("### Main technologies reported")
    tech_counts = filtered[tech_column].value_counts().head(12).sort_values().reset_index()
    tech_counts.columns = ["Main technology", "Respondents"]
    tech_counts["Share of filtered respondents (%)"] = tech_counts["Respondents"] / len(filtered) * 100
    tech_fig = px.bar(
        tech_counts, x="Respondents", y="Main technology", orientation="h",
        text="Share of filtered respondents (%)",
        title="Top 12 main technology categories",
        labels={"Respondents": "Respondents"}
    )
    tech_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(style_chart(tech_fig, 520), use_container_width=True)

    top_technologies = filtered[tech_column].value_counts().head(10).index
    tech_seniority = (
        filtered[filtered[tech_column].isin(top_technologies)]
        .groupby([tech_column, "Seniority level"])
        .size().reset_index(name="Respondents")
    )
    tech_seniority_fig = px.bar(
        tech_seniority, x=tech_column, y="Respondents", color="Seniority level",
        barmode="group", title="Top technologies by seniority level"
    )
    tech_seniority_fig.update_xaxes(tickangle=35)
    st.plotly_chart(style_chart(tech_seniority_fig, 470), use_container_width=True)

    st.dataframe(tech_counts.sort_values("Respondents", ascending=False), use_container_width=True, hide_index=True)
    st.caption("Multi-technology answers may appear as combined categories, and Unknown means no main technology was reported.")

# -----------------------------
# Data quality and methodology
# -----------------------------
with tab_quality:
    st.markdown("### Data quality checks")
    q1, q2, q3 = st.columns(3)
    q1.metric("Raw rows", f"{len(raw):,}")
    q2.metric("Duplicate raw rows", f"{raw_duplicates:,}")
    q3.metric("Duplicate cleaned rows", f"{clean_duplicates:,}")

    qleft, qright = st.columns(2)
    with qleft:
        st.markdown("**Missing values before cleaning**")
        raw_missing_df = raw_missing[raw_missing > 0].rename("Missing values").reset_index()
        raw_missing_df.columns = ["Column", "Missing values"]
        if raw_missing_df.empty:
            st.success("No missing values found in the raw file.")
        else:
            st.dataframe(raw_missing_df, use_container_width=True, hide_index=True)
    with qright:
        st.markdown("**Missing values after cleaning**")
        clean_missing_df = clean_missing[clean_missing > 0].rename("Missing values").reset_index()
        clean_missing_df.columns = ["Column", "Missing values"]
        if clean_missing_df.empty:
            st.success("No missing values remain after the documented cleaning and imputation.")
        else:
            st.dataframe(clean_missing_df, use_container_width=True, hide_index=True)

    st.markdown("### Methodology used")
    st.markdown(
        """
        1. Loaded the selected-columns CSV and kept the original data for quality checks.
        2. Trimmed surrounding whitespace and converted the timestamp to a datetime value.
        3. Converted experience values (including month-based answers) to years.
        4. Standardized common technology aliases and selected seniority labels.
        5. Treated total experience above 50 years as invalid and flagged Germany experience greater than total experience.
        6. Imputed missing numeric values with each column's median and labeled missing categorical values as `Unknown`.
        7. Used descriptive statistics and interactive Plotly visualizations to explore distributions and associations.
        """
    )
    st.markdown("### Limitations")
    st.markdown(
        """
        - The uploaded project dataset contains no salary column; salary analysis is not possible from this file.
        - City names and technology combinations may still have spelling or grouping inconsistencies.
        - Median imputation keeps rows in the analysis but can affect distributions and correlations.
        - Small groups should be interpreted cautiously.
        - The sample describes survey respondents and should not be treated as a representative census of all IT professionals.
        """
    )

    st.markdown("### Preview cleaned data")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download filtered, cleaned data (CSV)",
        data=csv_bytes,
        file_name="it_survey_cleaned_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("IT Survey Dashboard --> Abdullah Rabea")
