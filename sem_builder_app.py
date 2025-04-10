import streamlit as st
import pandas as pd
from semopy import Model, Optimizer
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import tempfile
import base64
import os

# Helper to download files
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    return f"<a href='data:application/octet-stream;base64,{b64}' download='{download_filename}'>{download_link_text}</a>"

# App title
st.title("🔧 Structural Equation Modeling (SEM) Builder")

# 1. Data Upload
st.header("📥 Upload CSV Data")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    # 2. Model Specification
    st.header("🧩 SEM Model Specification")

    st.subheader("Measurement Model Syntax")
    default_measurement = """
    # Example:
    # latent1 =~ x1 + x2 + x3
    # latent2 =~ x4 + x5 + x6
    """
    measurement_model = st.text_area("Enter measurement model (lavaan-style)", default_measurement)

    st.subheader("Structural Model Syntax")
    default_structural = """
    # Example:
    # latent2 ~ latent1
    """
    structural_model = st.text_area("Enter structural model (lavaan-style)", default_structural)

    model_description = measurement_model + "\n" + structural_model

    # 3. Model Estimation
    st.header("⚙️ Model Estimation")
    if st.button("Estimate Model"):
        try:
            model = Model(model_description)
            opt = Optimizer(model)
            opt.optimize(df)

            st.success("Model estimated successfully!")

            st.subheader("Model Fit Indices")
            fit_stats = model.calc_stats()
            st.write({
                "Chi-square": fit_stats.chi2,
                "CFI": fit_stats.cfi,
                "TLI": fit_stats.tli,
                "RMSEA": fit_stats.rmsea,
                "SRMR": fit_stats.srmr
            })

            st.subheader("Path Coefficients")
            est = model.inspect()
            st.dataframe(est)

            # 4. Visualization
            st.header("📊 Model Visualization")
            st.subheader("Path Diagram")
            dot = model.inspect_graphviz()
            st.graphviz_chart(dot.source)

            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # 5. Report Generation
            st.header("📝 Report Generation")
            if st.button("Generate Report"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmpfile:
                    report_content = f"""
# SEM Analysis Report

## Model Specification
```
{model_description}
```

## Model Fit Indices
- Chi-square: {fit_stats.chi2:.3f}
- CFI: {fit_stats.cfi:.3f}
- TLI: {fit_stats.tli:.3f}
- RMSEA: {fit_stats.rmsea:.3f}
- SRMR: {fit_stats.srmr:.3f}

## Standardized Coefficients
{est.to_markdown()}
"""
                    tmpfile.write(report_content.encode())
                    tmpfile.flush()
                    with open(tmpfile.name, "rb") as f:
                        st.markdown(download_link(f.read(), "sem_report.md", "📥 Download Report (Markdown)"), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error estimating model: {e}")
else:
    st.info("Please upload a CSV file to begin.")
