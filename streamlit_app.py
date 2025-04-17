"""Streamlit SEM Explorer – no‑code UI
------------------------------------
Upload a CSV, pick indicators for each construct, define hypotheses with
simple drop‑downs, view three key result tables, and download them as CSV.
"""

from __future__ import annotations

import io
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
from pingouin import cronbach_alpha
from semopy import Model, Optimizer, calc_stats

# -------------------- helper functions --------------------

def build_measurement_syntax(constructs: list[dict]) -> str:
    """Return lavaan/semopy measurement block from UI dict."""
    return "\n".join(
        f"{c['name']} =~ " + " + ".join(c['items']) for c in constructs
    )


def build_structural_syntax(paths: list[dict]) -> str:
    """Return structural paths block."""
    if not paths:
        return ""
    lines = [f"{p['dv']} ~ " + " + ".join(p['ivs']) for p in paths]
    return "\n".join(lines)


def cronbach_table(data: pd.DataFrame, constructs: list[dict]) -> pd.DataFrame:
    rows = []
    for c in constructs:
        if len(c['items']) < 2:
            alpha = np.nan
        else:
            alpha = cronbach_alpha(data[c['items']])[0]
        rows.append({"Construct": c['name'], "Cronbach_Alpha": round(alpha, 3)})
    return pd.DataFrame(rows)


def path_table(params: pd.DataFrame) -> pd.DataFrame:
    """Return nice table of ~ paths with beta & p."""
    df = params[params['op'] == '~'][['lval', 'rval', 'Estimate', 'p-value']].copy()
    df.columns = ["DV", "IV", "Beta", "p"]
    df.insert(0, "Hypothesis", [f"H{i+1}" for i in range(len(df))])
    df['Beta'] = df['Beta'].round(3)
    df['p'] = df['p'].apply(lambda x: "<.001" if x < .001 else round(x, 3))
    return df

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="SEM Explorer", layout="wide")
st.title("📊 Structural Equation Modeling Explorer (No‑Code)")

if 'constructs' not in st.session_state:
    st.session_state.constructs: list[dict] = []
if 'paths' not in st.session_state:
    st.session_state.paths: list[dict] = []

uploaded_file = st.file_uploader("1️⃣ Upload your CSV dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded **{data.shape[0]}** rows × **{data.shape[1]}** columns.")
    with st.expander("Preview data", expanded=False):
        st.dataframe(data.head())

    # --------------- construct builder ---------------
    st.markdown("---")
    st.header("2️⃣ Define latent constructs")
    with st.form("construct_form"):
        c_name = st.text_input("Construct name (e.g., CSR)")
        c_items = st.multiselect(
            "Select indicator variables", options=list(data.columns)
        )
        add_c = st.form_submit_button("Add / Update Construct")
        if add_c and c_name and c_items:
            # replace if exists
            st.session_state.constructs = [c for c in st.session_state.constructs if c['name'] != c_name]
            st.session_state.constructs.append({"name": c_name, "items": c_items})

    if st.session_state.constructs:
        st.subheader("Current constructs")
        st.dataframe(pd.DataFrame(st.session_state.constructs))

    # --------------- structural paths builder ---------------
    if st.session_state.constructs:
        st.markdown("---")
        st.header("3️⃣ Define hypotheses (structural paths)")
        construct_names = [c['name'] for c in st.session_state.constructs]
        with st.form("path_form"):
            dv = st.selectbox("Dependent variable (DV)", construct_names)
            ivs = st.multiselect(
                "Predictor(s) (IV)", construct_names, default=[]
            )
            add_p = st.form_submit_button("Add Path")
            if add_p and ivs:
                st.session_state.paths.append({"dv": dv, "ivs": ivs})

        if st.session_state.paths:
            st.subheader("Current hypotheses")
            st.dataframe(pd.DataFrame(st.session_state.paths))

    # --------------- run SEM ---------------
    if st.session_state.constructs and st.button("🚀 Run SEM Analysis"):
        meas_syntax = build_measurement_syntax(st.session_state.constructs)
        struct_syntax = build_structural_syntax(st.session_state.paths)
        sem_desc = textwrap.dedent(meas_syntax + "\n" + struct_syntax)

        try:
            model = Model(sem_desc)
            opt = Optimizer(model)
            opt.optimize(data)
            stats = calc_stats(model)
            st.success("Model estimated successfully!")

            # --- Fit indices
            st.subheader("Fit indices")
            fit_df = stats.fit.round(3)
            st.dataframe(fit_df)
            st.download_button("Download fit indices CSV", fit_df.to_csv(index=False), "fit_indices.csv")

            # --- Reliability
            st.subheader("Reliability (Cronbach's α)")
            alpha_df = cronbach_table(data, st.session_state.constructs)
            st.dataframe(alpha_df)
            st.download_button("Download reliability CSV", alpha_df.to_csv(index=False), "reliability.csv")

            # --- Path coefficients
            st.subheader("Path coefficients")
            path_df = path_table(stats.parameters)
            st.dataframe(path_df)
            st.download_button("Download path coefficients CSV", path_df.to_csv(index=False), "path_coefficients.csv")

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
