"""
Streamlit App: No‑Code CFA & SEM Builder
=======================================
This single‑file Streamlit application lets **anyone**—even without coding
skills—run a full measurement‑and‑structural model workflow on **any** tidy
CSV dataset:

1. **Upload Dataset** (wide format, one row per respondent).  
2. **Define Latent Constructs** by pointing‑and‑clicking observed variables.  
3. **Build Hypotheses / Structural Paths** interactively.  
4. **Estimate** CFA ➜ (optional) SEM with `semopy`.  
5. **Download** every results table (fit indices, loadings, paths, …) as CSV.

The file is totally self‑contained—just install the deps and run:  
```bash
pip install streamlit semopy pandas
streamlit run streamlit_app.py
```
"""

import streamlit as st
import pandas as pd
import base64
from semopy import Model, Optimizer

st.set_page_config(page_title="CFA‑SEM Builder", layout="wide")

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_state():
    """Initialize session‑state keys the first time the app runs."""
    for key, default in (
        ("data", None),
        ("constructs", {}),        # {"SE": ["SE1", "SE2", ...], ...}
        ("hypotheses", []),        # [{"lhs": "CSR", "rhs": "SE"}, ...]
    ):
        if key not in st.session_state:
            st.session_state[key] = default

_init_state()

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _b64_download(df: pd.DataFrame, filename: str, label: str):
    """Return an HTML download‑link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'

# ---------------------------------------------------------------------------
# Sidebar – Upload data
# ---------------------------------------------------------------------------

st.sidebar.header("1. Upload CSV")
file = st.sidebar.file_uploader("Select a tidy CSV file", type=["csv"])
if file:
    try:
        st.session_state.data = pd.read_csv(file)
        st.sidebar.success(f"Loaded **{st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns**.")
    except Exception as e:
        st.session_state.data = None
        st.sidebar.error(f"Failed to read CSV: {e}")
else:
    st.sidebar.info("Awaiting CSV file…")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🔧 No‑Code CFA & SEM Builder")
st.markdown("Design latent constructs, specify hypotheses, and estimate a full model—**all without writing code**.")

if st.session_state.data is None:
    st.stop()  # Wait until data uploaded

columns = list(st.session_state.data.columns)

# =============================================================================
# Step 2 – Define Constructs
# =============================================================================

st.header("2️⃣ Define Latent Constructs")
name = st.text_input("Construct name", placeholder="e.g., SE")
items = st.multiselect("Observed items (columns)", columns, key="item_select")
add_col1, add_col2 = st.columns([1,3])
with add_col1:
    if st.button("Add/Update"):
        if not name or not items:
            st.warning("Please supply both a construct name and at least one item.")
        else:
            st.session_state.constructs[name.strip()] = items
            st.success(f"Construct **{name.upper()}** saved.")
with add_col2:
    if st.button("Clear Constructs"):
        st.session_state.constructs = {}
        st.session_state.hypotheses = []

if st.session_state.constructs:
    st.subheader("Current constructs")
    st.dataframe(pd.DataFrame(
        [(k, ", ".join(v)) for k, v in st.session_state.constructs.items()],
        columns=["Construct", "Items"],
    ), hide_index=True, height=200)

# =============================================================================
# Step 3 – Build Hypotheses (Structural Paths)
# =============================================================================

st.header("3️⃣ Build Hypotheses / Structural Paths")
if not st.session_state.constructs:
    st.info("Define at least one construct first.")
    st.stop()

construct_list = list(st.session_state.constructs.keys())
col_dep, col_indep = st.columns(2)
dep = col_dep.selectbox("Outcome (dependent construct)", construct_list, key="dep")
indep_opts = [c for c in construct_list if c != dep]
indeps = col_indep.multiselect("Predictor(s) (independent constructs)", indep_opts, key="indeps")

hypo_cols = st.columns([1,5])
with hypo_cols[0]:
    if st.button("➕ Add Path(s)"):
        new_paths = [
            {"lhs": dep, "rhs": rhs}
            for rhs in indeps
            if not any(h["lhs"] == dep and h["rhs"] == rhs for h in st.session_state.hypotheses)
        ]
        st.session_state.hypotheses.extend(new_paths)
        st.experimental_rerun()
with hypo_cols[1]:
    if st.button("🗑️ Clear All Paths"):
        st.session_state.hypotheses = []
        st.experimental_rerun()

if st.session_state.hypotheses:
    st.subheader("Hypothesis list")
    st.dataframe(pd.DataFrame(st.session_state.hypotheses), hide_index=True, height=180)
else:
    st.info("No paths defined yet – add some above.")

# =============================================================================
# Step 4 – Estimate CFA/SEM
# =============================================================================

st.header("4️⃣ Estimate Model & Download Results")
est_btn = st.button("🚀 Run Model", disabled=not st.session_state.constructs)

if est_btn:
    # Build lavaan/semopy‑style syntax
    meas_lines = [f"{k} =~ " + " + ".join(v) for k, v in st.session_state.constructs.items()]
    struct_lines = [f"{h['lhs']} ~ {h['rhs']}" for h in st.session_state.hypotheses]
    model_desc = "\n".join(meas_lines + struct_lines)

    st.subheader("Model specification")
    st.code(model_desc, language="text")

    try:
        model = Model(model_desc)
        optim = Optimizer(model)
        optim.optimize(st.session_state.data)

        st.success("Model estimated successfully!")
        stats = model.calc_stats()

        # Fit indices
        st.subheader("Fit indices")
        fit_df = stats[["n", "chisq", "df", "p-value", "cfi", "tli", "rmsea", "aic"]]
        st.dataframe(fit_df, hide_index=True)
        st.markdown(_b64_download(fit_df.reset_index(), "fit_indices.csv", "⬇️ Download Fit CSV"), unsafe_allow_html=True)

        # Standardized loadings
        st.subheader("Standardized loadings")
        load_df = model.inspect()["lambda"]
        st.dataframe(load_df.style.format("{:.3f}"))
        st.markdown(_b64_download(load_df.reset_index(), "loadings.csv", "⬇️ Download Loadings CSV"), unsafe_allow_html=True)

        # Path estimates (only if structural part exists)
        if struct_lines:
            st.subheader("Path coefficients")
            path_df = model.inspect()["beta"]
            st.dataframe(path_df.style.format("{:.3f}"))
            st.markdown(_b64_download(path_df.reset_index(), "path_coeffs.csv", "⬇️ Download Paths CSV"), unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"Estimation failed: {exc}")

