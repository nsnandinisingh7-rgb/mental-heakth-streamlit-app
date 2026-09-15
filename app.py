"""
Mental Health in Tech Workplace — Interactive EDA Dashboard
Run with: streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Mental Health in Tech — EDA",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # Age: keep a realistic working-age range
    df_clean = df_clean[(df_clean["Age"] >= 15) & (df_clean["Age"] <= 80)]

    # Gender: normalize free-text responses into 3 buckets
    def normalize_gender(g):
        g = str(g).strip().lower()
        male_set = {
            "male", "m", "man", "cis male", "cis man", "malr", "mail", "maile",
            "make", "msle", "male-ish", "male (cis)", "guy (-ish) ^_^",
        }
        female_set = {
            "female", "f", "woman", "cis female", "cis-female/femme",
            "female (cis)", "femail", "femake", "trans-female", "trans woman",
        }
        if g in male_set:
            return "Male"
        if g in female_set:
            return "Female"
        return "Other/Non-binary"

    df_clean["Gender_clean"] = df_clean["Gender"].apply(normalize_gender)

    if "self_employed" in df_clean.columns:
        df_clean["self_employed"] = df_clean["self_employed"].fillna("Unknown")
    if "work_interfere" in df_clean.columns:
        df_clean["work_interfere"] = df_clean["work_interfere"].fillna("Not applicable")

    drop_cols = [c for c in ["state", "comments"] if c in df_clean.columns]
    df_clean = df_clean.drop(columns=drop_cols)

    if "Timestamp" in df_clean.columns:
        df_clean["Timestamp"] = pd.to_datetime(df_clean["Timestamp"], errors="coerce")
        df_clean["SurveyYear"] = df_clean["Timestamp"].dt.year

    return df_clean


# ---------------------------------------------------------------------------
# Sidebar: data source + filters
# ---------------------------------------------------------------------------

st.sidebar.title("🧠 Mental Health EDA")
st.sidebar.caption("OSMI Tech Workplace Survey")

uploaded = st.sidebar.file_uploader("Upload survey.csv (optional)", type=["csv"])
data_path = uploaded if uploaded is not None else "survey.csv"

try:
    raw_df = load_data(data_path)
except FileNotFoundError:
    st.error(
        "Couldn't find `survey.csv` next to `app.py`. "
        "Upload a CSV using the sidebar to continue."
    )
    st.stop()

df = clean_data(raw_df)

st.sidebar.markdown("### Filters")

genders = sorted(df["Gender_clean"].unique().tolist())
sel_genders = st.sidebar.multiselect("Gender", genders, default=genders)

countries = df["Country"].value_counts().index.tolist()
top_default = countries[:5]
sel_countries = st.sidebar.multiselect(
    "Country (default: top 5 by count)", countries, default=top_default
)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
sel_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

filtered = df[
    df["Gender_clean"].isin(sel_genders)
    & df["Country"].isin(sel_countries)
    & df["Age"].between(sel_age[0], sel_age[1])
]

st.sidebar.markdown(f"**{len(filtered):,}** respondents match filters")

if filtered.empty:
    st.warning("No rows match the current filters — widen your selection in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------

st.title("🧠 Mental Health in Tech Workplace — EDA Dashboard")
st.caption(
    "Explore the OSMI survey on mental health attitudes and treatment-seeking "
    "behavior among tech workers. Use the sidebar to filter by gender, country, and age."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Respondents (filtered)", f"{len(filtered):,}")
k2.metric("Median age", f"{int(filtered['Age'].median())}")
treated_pct = (filtered["treatment"] == "Yes").mean() * 100
k3.metric("Sought treatment", f"{treated_pct:.1f}%")
fam_pct = (filtered["family_history"] == "Yes").mean() * 100
k4.metric("Family history", f"{fam_pct:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_demo, tab_workplace, tab_treatment, tab_data = st.tabs(
    ["📊 Overview", "👥 Demographics", "🏢 Workplace Factors", "💊 Treatment Analysis", "📄 Raw Data"]
)

# --- Overview ---------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered, x="Age", nbins=30, color="treatment", barmode="overlay",
            title="Age Distribution by Treatment-Seeking",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        counts = filtered["treatment"].value_counts().reset_index()
        counts.columns = ["treatment", "count"]
        fig = px.pie(
            counts, names="treatment", values="count", hole=0.45,
            title="Sought Mental Health Treatment",
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        st.plotly_chart(fig, use_container_width=True)

    country_counts = filtered["Country"].value_counts().reset_index().head(15)
    country_counts.columns = ["Country", "Respondents"]
    fig = px.bar(
        country_counts, x="Respondents", y="Country", orientation="h",
        title="Respondents by Country (Top 15 in current filter)",
        color="Respondents", color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# --- Demographics -------------------------------------------------------------
with tab_demo:
    c1, c2 = st.columns(2)

    with c1:
        gcounts = filtered["Gender_clean"].value_counts().reset_index()
        gcounts.columns = ["Gender", "Count"]
        fig = px.bar(
            gcounts, x="Count", y="Gender", orientation="h",
            title="Gender Distribution (Normalized)",
            color="Gender", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered, x="Gender_clean", y="Age", color="Gender_clean",
            title="Age Distribution by Gender",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.violin(
        filtered, x="treatment", y="Age", color="treatment", box=True, points="all",
        title="Age vs Treatment-Seeking (Violin)",
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Workplace Factors --------------------------------------------------------
with tab_workplace:
    c1, c2 = st.columns(2)

    with c1:
        order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
        present = [o for o in order if o in filtered["no_employees"].unique()]
        fig = px.histogram(
            filtered, x="no_employees", color="treatment", barmode="group",
            category_orders={"no_employees": present},
            title="Treatment-Seeking by Company Size",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            filtered, x="remote_work", color="treatment", barmode="group",
            title="Treatment-Seeking by Remote Work Status",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

    policy_col = st.selectbox(
        "Explore a workplace policy variable",
        ["benefits", "care_options", "wellness_program", "seek_help", "anonymity", "leave"],
    )
    ct = pd.crosstab(filtered[policy_col], filtered["treatment"], normalize="index").mul(100)
    fig = px.bar(
        ct, barmode="stack",
        title=f"Treatment-Seeking Rate (%) by '{policy_col}'",
        color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
    )
    fig.update_layout(yaxis_title="% of respondents")
    st.plotly_chart(fig, use_container_width=True)

# --- Treatment Analysis --------------------------------------------------------
with tab_treatment:
    c1, c2 = st.columns(2)

    with c1:
        ct = pd.crosstab(filtered["family_history"], filtered["treatment"], normalize="index").mul(100)
        fig = px.bar(
            ct, barmode="stack", title="Treatment-Seeking by Family History (%)",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        order = ["Never", "Rarely", "Sometimes", "Often", "Not applicable"]
        present = [o for o in order if o in filtered["work_interfere"].unique()]
        ct = pd.crosstab(filtered["work_interfere"], filtered["treatment"], normalize="index").mul(100)
        ct = ct.loc[[i for i in present if i in ct.index]]
        fig = px.bar(
            ct, barmode="stack", title="Treatment-Seeking by Work Interference (%)",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

    sun_df = filtered.groupby(["Gender_clean", "treatment"]).size().reset_index(name="count")
    fig = px.sunburst(
        sun_df, path=["Gender_clean", "treatment"], values="count",
        title="Gender → Treatment-Seeking Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig, use_container_width=True)

    top10 = filtered["Country"].value_counts().head(10).index
    sub = filtered[filtered["Country"].isin(top10)]
    if not sub.empty:
        ct = pd.crosstab(sub["Country"], sub["treatment"], normalize="index").mul(100)
        if "Yes" in ct.columns:
            ct = ct.sort_values("Yes", ascending=True)
        fig = px.bar(
            ct, orientation="h", barmode="stack",
            title="Treatment-Seeking Rate by Country (Top 10 by Respondent Count)",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Raw Data ------------------------------------------------------------------
with tab_data:
    st.subheader("Filtered dataset")
    st.dataframe(filtered, use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv, "filtered_survey.csv", "text/csv")

st.divider()
st.caption(
    "Data: OSMI Mental Health in Tech Survey (self-reported, observational). "
    "Associations shown here are not causal, and the sample skews toward US-based tech workers."
)
