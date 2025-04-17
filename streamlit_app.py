"""
Streamlit App: CFA & SEM Builder
================================
This app lets **anyone** (no coding required!) build and test a
Confirmatory‑Factor‑Analysis (CFA) and full Structural‑Equation Model (SEM)
workflow directly in the browser:

1. **Upload** any tidy, numeric‐only CSV (wide format: columns = observed
   variables / items, rows = respondents).  
2. **Define constructs** (latent variables) interactively by choosing which
   columns load on each construct.  
3. **Build hypotheses / structural relations** between constructs with a
   point‑and‑click interface.  
4. **Run** the model – fit indices, reliability, validity, and hypothesis
   tests are computed with **semopy** under the hood.  
5. **Download** publication‑ready CSV tables of every result.

---
Dependencies
------------
```bash
pip install streamlit semopy pandas numpy pingouin
```

Run the app:
```bash
streamlit run streamlit_app.py
```
"""

from __future__ import annotations

import io
import textwrap
from typing import Dict, List

import numpy as np
import pandas as pd
import pingouin as pg
import streamlit as st
from semopy import Model, Optimizer
from semopy.inspector import inspect
from semopy.stats import calc_stats

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def cronbach_alpha(df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of items."""
    corr = df.corr()
    k = len(df.columns)
    if k < 2:
        return np.nan
    alpha = (k / (k - 1)) * (1 - corr.values.sum().trace() / k)
    return alpha


def composite_reliability(loadings: np.ndarray) -> float:
    """Raykov & Shrout composite reliability."""
    sq_load = np.square(loadings)
    return sq_load.sum() / (sq_load.sum() + (1 - sq_load).sum())


def average_variance_extracted(loadings: np.ndarray) -> float:
    """Fornell & Larcker AVE."""
    return np.square(loadings).mean()


def build_measurement_model(constructs: Dict[str, List[str]]) -> str:
    """Convert construct mapping to lavaan/semopy syntax."""
    lines = [f"  {name} =~ " + " + ".join(items) for name, items in constructs.items()]
    return "\n".join(lines)


def build_structural_model(hypotheses: List[Dict[str, List[str]]]) -> str:
    """Convert hypotheses list to lavaan/semopy syntax."""
    lines: List[str] = []
    for h in hypotheses:
        dv = h["dv"]
        ivs = h["ivs"]
        if dv and ivs:
            lines.append(f"  {dv} ~ " + " + ".join(ivs))
    return "\n".join(lines)


def fit_sem(full_model: str, data: pd.DataFrame):
    model = Model(textwrap.dedent(full_model))
    opt = Optimizer(model)
    opt.optimize(data)
    return model


def get_fit_indices(model) -> pd.DataFrame:
    stats = calc_stats(model)
    wanted = ["Chi-Squared", "DoF", "p-value", "CFI", "TLI", "RMSEA", "SRMR"]
    nice = {"Chi-Squared": "chisq", "DoF": "df", "p-value": "pvalue"}
    rows = {nice.get(k, k.lower()): round(v, 3) for k, v in stats.items() if k in wanted}
    return pd.DataFrame(rows, index=["SEM"])


def get_loadings(model) -> pd.DataFrame:
    lambdas = inspect(model, what="lambdas", std_est=True)
    records = []
    for lv, items in lambdas.items():
        for item, est in items.items():
            records.append({"Construct": lv, "Item": item, "Loading": round(est, 3)})
    return pd.DataFrame(records)


def latent_correlation_matrix(model, ave_df: pd.DataFrame) -> pd.DataFrame:
    phi = inspect(model, what="phi", std_est=True)  # latent corr
    lvs = list(phi.keys())
    mat = pd.DataFrame(index=lvs, columns=lvs)
    for i in lvs:
        for j in lvs:
            if i == j:
                mat.loc[i, j] = ave_df.loc[ave_df.Construct == i, "AVE_Sqrt"].values[0]
            else:
                mat.loc[i, j] = round(phi[i][j], 3)
    mat.insert(0, "Factor", mat.index)
    return mat.reset_index(drop=True)


def hypothesis_table(model) -> pd.DataFrame:
    paths = inspect(model, what="beta", std_est=True)
    records = []
    for dv, ivs in paths.items():
        for iv, est in ivs.items():
            z = est / model.stderr[dv][iv]
            p = 2 * (1 - pg.distributions.normal_dist.cdf(np.abs(z)))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"{p:.3f}"
            records.append({
                "Path": f"{dv} ~ {iv}",
                "β": round(est, 3),
                "z": round(z, 3),
                "p": sig,
            })
    df = pd.DataFrame(records)
    df.index = [f"H{i+1}" for i in range(len(df))]
    df.index.name = "Hypothesis"
    return df.reset_index()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SEM Builder", layout="wide")
st.title("📐 SEM Builder – CFA & Structural Modelling without Coding")

st.sidebar.header("1. Upload Data")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])  # type: ignore

if csv_file is not None:
    data = pd.read_csv(csv_file)
    st.write("### Preview of Uploaded Data", data.head())

    if "constructs" not in st.session_state:
        st.session_state.constructs: Dict[str, List[str]] = {}

    # ---------------------------------------------------------
    # Measurement model builder
    # ---------------------------------------------------------
    st.sidebar.header("2. Build Measurement Model")
    with st.sidebar.form("construct_form", clear_on_submit=True):
        c_name = st.text_input("Construct name (latent variable)")
        c_items = st.multiselect("Observed items", options=list(data.columns))
        add_const = st.form_submit_button("➕ Add Construct")
        if add_const and c_name and c_items:
            st.session_state.constructs[c_name] = c_items
            st.success(f"Added construct {c_name} with {len(c_items)} items.")

    if st.session_state.constructs:
        st.write("### Current Constructs", pd.DataFrame([
            {"Construct": k, "Items": ", ".join(v)} for k, v in st.session_state.constructs.items()
        ]))

    # ---------------------------------------------------------
    # Hypotheses / structural relations
    # ---------------------------------------------------------
    if "hypotheses" not in st.session_state:
        st.session_state.hypotheses: List[Dict[str, List[str]]] = []

    st.sidebar.header("3. Build Hypotheses")
    with st.sidebar.form("hyp_form", clear_on_submit=True):
        dep = st.selectbox("Dependent (DV)", options=list(st.session_state.constructs.keys()))
        indep = st.multiselect("Independent (IVs)", options=[c for c in st.session_state.constructs.keys() if c != dep])
        add_hyp = st.form_submit_button("➕ Add Hypothesis")
        if add_hyp and dep and indep:
            st.session_state.hypotheses.append({"dv": dep, "ivs": indep})
            st.success(f"Added hypothesis: {dep} ~ {' + '.join(indep)}")

    if st.session_state.hypotheses:
        st.write("### Current Hypotheses")
        st.table(pd.DataFrame([{"DV": h["dv"], "IVs": ", ".join(h["ivs"])} for h in st.session_state.hypotheses]))

    # ---------------------------------------------------------
    # Run Model
    # ---------------------------------------------------------
    st.sidebar.header("4. Run Model")
    run_btn = st.sidebar.button("🚀 Run CFA / SEM")

    if run_btn:
        if not st.session_state.constructs:
            st.error("Add at least one construct before running the model.")
        else:
            st.info("Fitting model… this may take a moment.")
            meas = build_measurement_model(st.session_state.constructs)
            stru = build_structural_model(st.session_state.hypotheses)
            full_model = meas + ("\n" + stru if stru else "")

            try:
                model = fit_sem(full_model, data)
            except Exception as e:
                st.exception(e)
            else:
                st.success("Model fitted!")

                # -------------------------------------------------
                # Results tabs
                # -------------------------------------------------
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Fit Indices", "Reliability & Validity", "Loadings", "Latent Corr", "Hypotheses", "Syntax"])

                # Fit Indices
                with tab1:
                    fit_df = get_fit_indices(model)
                    st.dataframe(fit_df, use_container_width=True)
                    csv = fit_df.to_csv(index=False).encode()
                    st.download_button("Download fit indices", csv, "fit_indices.csv")

                # Reliability & Validity
                with tab2:
                    load_df = get_loadings(model)
                    rel_records = []
                    for lv, g in load_df.groupby("Construct"):
                        cr = composite_reliability(g.Loading.values)
                        ave = average_variance_extracted(g.Loading.values)
                        rel_records.append({
                            "Construct": lv,
                            "Cronbach_α": round(cronbach_alpha(data[g.Item].dropna()), 3),
                            "CR": round(cr, 3),
                            "AVE": round(ave, 3),
                            "AVE_Sqrt": round(np.sqrt(ave), 3)
                        })
                    rel_df = pd.DataFrame(rel_records)
                    st.dataframe(rel_df, use_container_width=True)
                    csv = rel_df.to_csv(index=False).encode()
                    st.download_button("Download reliability table", csv, "reliability_validity.csv")

                # Loadings table
                with tab3:
                    st.dataframe(load_df, use_container_width=True)
                    csv = load_df.to_csv(index=False).encode()
                    st.download_button("Download loadings", csv, "factor_loadings.csv")

                # Latent correlation
                with tab4:
                    lat_df = latent_correlation_matrix(model, rel_df)
                    st.dataframe(lat_df, use_container_width=True)
                    csv = lat_df.to_csv(index=False).encode()
                    st.download_button("Download latent correlations", csv, "latent_corr_matrix.csv")

                # Hypothesis results
                with tab5:
                    if st.session_state.hypotheses:
                        hyp_df = hypothesis_table(model)
                        st.dataframe(hyp_df, use_container_width=True)
                        csv = hyp_df.to_csv(index=False).encode()
                        st.download_button("Download hypotheses", csv, "hypothesis_results.csv")
                    else:
                        st.info("No structural relations specified – CFA only.")

                # Syntax
                with tab6:
                    st.code(full_model, language="lavaan")

