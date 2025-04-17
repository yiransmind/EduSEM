"""streamlit_app.py – No‑Code CFA & SEM Builder
================================================
A **single‑file** Streamlit application that lets *anyone* upload a tidy CSV,
create a measurement model, specify hypotheses (structural paths), run CFA/SEM
with **semopy**, and download results tables – all without writing code.

▶️ **How to run**
```bash
pip install streamlit pandas semopy openpyxl
streamlit run streamlit_app.py
```

------------------------------------------------------------
"""

import io
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

# Try importing semopy; if not present, inform the user gracefully
try:
    import semopy
except ImportError:
    semopy = None

st.set_page_config(page_title="CFA & SEM Builder", page_icon="📐", layout="centered")
st.title("📐 No‑Code CFA & SEM Builder")

# ----------------------------------------------------------------------------
# Session‑state helpers -------------------------------------------------------
# ----------------------------------------------------------------------------

def _init_session():
    st.session_state.setdefault("df", None)           # Uploaded dataframe
    st.session_state.setdefault("constructs", {})     # {latent: [items]}
    st.session_state.setdefault("paths", [])          # List[(dep, pred)]

_init_session()

# ----------------------------------------------------------------------------
# STEP 1 – Upload CSV ---------------------------------------------------------
# ----------------------------------------------------------------------------

st.header("1. Upload Your Data 📄")
file = st.file_uploader("Upload a tidy, numeric‑only CSV (wide format)", type=["csv"])

if file:
    try:
        st.session_state.df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()

    st.success("Dataset loaded! Preview below (first 10 rows):")
    st.dataframe(st.session_state.df.head(10), height=200)
else:
    st.info("Awaiting CSV upload…")
    st.stop()

# Ensure dataframe is numeric‑only (semopy requirement)
if not st.session_state.df.select_dtypes(include="number").shape[1] == st.session_state.df.shape[1]:
    st.warning("⚠️ All columns must be numeric for CFA/SEM. Non‑numeric columns were detected.")

# ----------------------------------------------------------------------------
# STEP 2 – Define Measurement Model ------------------------------------------
# ----------------------------------------------------------------------------

st.header("2. Define Latent Constructs 🏗️")
cols = list(st.session_state.df.columns)

with st.expander("Add a new construct"):
    c1, c2 = st.columns([1, 2])
    with c1:
        new_latent = st.text_input("Construct name", key="latent_name")
    with c2:
        new_items = st.multiselect("Observed variables", cols, key="latent_items")
    if st.button("➕ Add Construct"):
        if not new_latent or not new_items:
            st.warning("Provide a name **and** at least one item.")
        elif new_latent in st.session_state.constructs:
            st.warning("A construct with that name already exists.")
        else:
            st.session_state.constructs[new_latent] = new_items
            st.success(f"Added construct **{new_latent}**")

if st.session_state.constructs:
    st.subheader("Current constructs")
    for latent, items in st.session_state.constructs.items():
        st.markdown(f"- **{latent}** ← {', '.join(items)}")
    if st.button("❌ Clear all constructs"):
        st.session_state.constructs.clear()
        st.session_state.paths.clear()
else:
    st.info("Add at least one construct to continue…")

# ----------------------------------------------------------------------------
# STEP 3 – Build Hypotheses (Structural Paths) -------------------------------
# ----------------------------------------------------------------------------

if st.session_state.constructs:
    st.header("3. Build Hypotheses / Structural Paths 🔀")

    # Helper to avoid stale options after construct changes
    construct_names = list(st.session_state.constructs.keys())

    with st.form("add_path_form"):
        col1, col2 = st.columns(2)
        with col1:
            dep = st.selectbox("Outcome (dependent construct)", construct_names, key="dep_select")
        with col2:
            preds = st.multiselect(
                "Predictor construct(s)",
                options=[c for c in construct_names if c != dep],
                key="pred_select",
            )
        submitted = st.form_submit_button("Add path(s)")

        if submitted:
            if not preds:
                st.warning("Select at least one predictor.")
            else:
                added_any = False
                for pred in preds:
                    path = (dep, pred)
                    if path not in st.session_state.paths:
                        st.session_state.paths.append(path)
                        added_any = True
                if added_any:
                    st.success("Path(s) added!")
                else:
                    st.info("No new paths were added (possible duplicates).")

    # Display current hypotheses table
    if st.session_state.paths:
        path_df = pd.DataFrame(
            {
                "Outcome": [p[0] for p in st.session_state.paths],
                "Predictor": [p[1] for p in st.session_state.paths],
            }
        )
        st.dataframe(path_df, height=min(400, 35 * len(path_df) + 40))
        if st.button("❌ Clear all hypotheses"):
            st.session_state.paths.clear()
            st.success("All hypotheses cleared.")
    else:
        st.info("No paths yet – add at least one above.")

# ----------------------------------------------------------------------------
# STEP 4 – Run CFA/SEM --------------------------------------------------------
# ----------------------------------------------------------------------------

if st.session_state.constructs and st.session_state.paths:
    st.header("4. Estimate Model & Download Results 📊")

    if semopy is None:
        st.error(
            "`semopy` is not installed. Run `pip install semopy` in your environment "
            "and restart the app."
        )
        st.stop()

    if st.button("🚀 Run CFA + SEM"):
        with st.spinner("Fitting model… this may take a moment"):
            # ------------------------------------------------------------------
            # Build semopy model description
            # ------------------------------------------------------------------
            lines = []
            for latent, items in st.session_state.constructs.items():
                lines.append(f"{latent} =~ " + " + ".join(items))
            for dep, pred in st.session_state.paths:
                lines.append(f"{dep} ~ {pred}")
            model_desc = "\n".join(lines)

            # Fit model
            try:
                mod = semopy.Model(model_desc)
                mod.fit(st.session_state.df.dropna())  # drop rows with missing
            except Exception as e:
                st.error(f"Model failed to converge or is misspecified: {e}")
                st.stop()

            # ------------------------------------------------------------------
            # Collect outputs
            # ------------------------------------------------------------------
            estimates = semopy.inspect(mod)
            stat_res = mod.calc_stats()

            # Fit indices summary
            fit_summary = {
                "Chi2": stat_res.chi2,
                "df": stat_res.dof,
                "p": stat_res.p_value,
                "CFI": stat_res.cfi,
                "TLI": stat_res.tli,
                "RMSEA": stat_res.rmsea,
                "SRMR": stat_res.srmr,
            }
            fit_df = pd.DataFrame(fit_summary, index=["SEM"])

            # Standardized paths only
            std_paths = estimates.loc[estimates["op"] == "~"][[
                "lval", "rval", "Est", "SE", "Z", "pval", "StdEst"]
            ].rename(columns={
                "lval": "Outcome",
                "rval": "Predictor",
                "Est": "β (unstd)",
                "StdEst": "β (std)",
                "Z": "z",
                "pval": "p",
            })

            # Reliability (Cronbach's alpha) – simple average inter‑item corr.
            reliab_rows = []
            for latent, items in st.session_state.constructs.items():
                sub = st.session_state.df[items]
                # Avg. inter‑item correlation (r_bar)
                r_bar = sub.corr().where(~pd.eye(len(items), dtype=bool)).stack().mean()
                alpha = (len(items) * r_bar) / (1 + (len(items) - 1) * r_bar) if len(items) > 1 else 0
                reliab_rows.append({"Construct": latent, "Cronbach α": round(alpha, 3)})
            alpha_df = pd.DataFrame(reliab_rows)

            # ------------------------------------------------------------------
            # Show results in the UI
            # ------------------------------------------------------------------
            st.subheader("Model Fit Indices")
            st.table(fit_df.round(3))

            st.subheader("Standardized Path Estimates (Hypotheses)")
            st.dataframe(std_paths.round(3))

            st.subheader("Reliability (Cronbach's α)")
            st.table(alpha_df)

            # ------------------------------------------------------------------
            # Download buttons
            # ------------------------------------------------------------------
            def _csv_download(df: pd.DataFrame, label: str, fname: str):
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    label=f"⬇️ Download {label}",
                    data=csv,
                    file_name=fname,
                    mime="text/csv",
                )

            _csv_download(fit_df, "fit indices", "model_fit_indices.csv")
            _csv_download(std_paths, "hypothesis tests", "hypothesis_results.csv")
            _csv_download(alpha_df, "reliability table", "reliability.csv")

            st.success("Done! Scroll up for tables and use the buttons to download.")
