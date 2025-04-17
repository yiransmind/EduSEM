
import streamlit as st
import pandas as pd
from semopy import Model, calc_stats, semplot

st.set_page_config(page_title="No‑Code SEM", layout="wide")
st.title("📐 No‑Code Structural Equation Modeling (SEM)")

st.markdown(
    """
    **What can this app do?**
    1. Fit Structural Equation Models (SEM) without writing code.
    2. Report fit indices and parameter estimates formatted for academic journals.
    3. Visualize the path diagram.

    **How to use it**
    1. Upload a rectangular CSV data file (rows = cases, columns = variables).
    2. Specify your model in lavaan / semopy syntax (examples below).
    3. Click **Run SEM**.

    *Example model syntax*  
    ```text
    # A simple CFA
    F1 =~ x1 + x2 + x3
    F2 =~ x4 + x5 + x6
    F1 ~~ F2

    # Structural paths
    Y1  ~  F1 + F2
    Y2  ~  Y1
    ```
    """
)

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
standardize = st.sidebar.checkbox("Standardize variables before fitting", value=False)
show_residuals = st.sidebar.checkbox("Show residuals", value=False)
dec = st.sidebar.number_input("Decimal places", 1, 6, 3)

# --- Main UI ---
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
model_spec = st.text_area("Model specification", height=220)

run = st.button("🚀 Run SEM")

if run:
    if uploaded is None or not model_spec.strip():
        st.error("Please upload a dataset and enter a model specification.")
        st.stop()

    # Read data
    data = pd.read_csv(uploaded)
    if standardize:
        data = (data - data.mean()) / data.std(ddof=0)

    # Fit model
    st.info("Fitting model…")
    model = Model(model_spec)
    try:
        model.fit(data)
    except Exception as e:
        st.exception(e)
        st.stop()

    # --- Output section ---
    st.success("Model fitted successfully!")

    # Fit indices
    st.subheader("Fit Indices")
    try:
        fit_stats = calc_stats(model, data).loc["stats"]
        # Select commonly reported indices
        keep = ["n_obs", "df", "chi2", "p_value", "gfi", "agfi", "cfi", "tli", "rmsea", "srmr", "aic", "bic"]
        fit_stats = fit_stats[keep].transpose().reset_index()
        fit_stats.columns = ["Statistic", "Value"]
        fit_stats["Value"] = fit_stats["Value"].round(dec)
        st.dataframe(fit_stats, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute fit indices: {e}")

    # Parameter estimates
    st.subheader("Parameter Estimates")
    try:
        params = model.inspect()
        params = params.round(dec)
        st.dataframe(params, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not retrieve parameter estimates: {e}")

    # Residuals
    if show_residuals:
        st.subheader("Residuals")
        try:
            resids = model.residuals(data)
            st.dataframe(resids.round(dec), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute residuals: {e}")

    # Path diagram
    st.subheader("Path Diagram")
    try:
        g = semplot(model, show=False)
        # semplot returns a graphviz.Digraph
        st.graphviz_chart(g.source, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate diagram: {e}")
