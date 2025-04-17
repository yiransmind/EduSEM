
"""Streamlit SEM Explorer
-------------------------
Upload a CSV, specify measurement & structural models (lavaan/semopy syntax),
inspect model‑fit and reliability tables, and download them as CSV files.
"""

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pingouin import cronbach_alpha
from semopy import Model, Optimizer, calc_stats

st.set_page_config(layout="wide")

st.title("Structural Equation Modeling Explorer")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    st.write(data.head())

    st.subheader("Measurement Model (lavaan-style syntax)")
    meas_model_text = st.text_area("Define measurement model", height=250, value=textwrap.dedent("""        CSR  =~ CSR1 + CSR2 + CSR3
        MSR  =~ MSR1 + MSR2 + MSR3
        SE   =~ SE1 + SE2 + SE3
    """))
    
    st.subheader("Structural Model (lavaan-style syntax)")
    struct_model_text = st.text_area("Define structural model (optional)", height=150, value=textwrap.dedent("""        MSR ~ CSR + SE
    """))
    
    full_model = meas_model_text + "\n" + struct_model_text
    
    if st.button("Run SEM Analysis"):
        try:
            model = Model(description=full_model)
            opt = Optimizer(model)
            opt.optimize(data)
            stats = calc_stats(model)
            
            st.success("Model estimation complete.")
            st.subheader("Model Fit Indices")
            fit_df = stats.fit.copy()
            st.dataframe(fit_df)
            csv_fit = fit_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Fit Indices", csv_fit, "fit_indices.csv", "text/csv")

            st.subheader("Reliability: Cronbach's Alpha")
            constructs = [line.split("=~")[0].strip() for line in meas_model_text.strip().split("\n")]
            alpha_data = []
            for con in constructs:
                items = full_model.split(f"{con} =~")[-1].split("\n")[0].strip().split(" + ")
                alpha = cronbach_alpha(data[items])[0]
                alpha_data.append({"Construct": con, "Cronbach_Alpha": round(alpha, 3)})
            alpha_df = pd.DataFrame(alpha_data)
            st.dataframe(alpha_df)
            csv_alpha = alpha_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Reliability Table", csv_alpha, "reliability.csv", "text/csv")

            st.subheader("Path Coefficients")
            estimates_df = stats.parameters.loc[stats.parameters['op'] == '~'][["lval", "op", "rval", "Estimate", "p-value"]]
            estimates_df.columns = ["DV", "", "IV", "Beta", "p"]
            st.dataframe(estimates_df)
            csv_paths = estimates_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Path Coefficients", csv_paths, "path_coefficients.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
