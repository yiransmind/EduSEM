import streamlit as st
from utils import load_data, plot_path_diagram
from sem_analysis import run_sem

st.set_page_config(page_title="SEM Analysis App")

st.title("Structural Equation Modeling for Non-Coders")

uploaded = st.file_uploader("Upload your CSV data", type=['csv'])
model_spec = st.text_area(
    "Enter your SEM model specification",
    value="# Example measurement model\nF1 =~ x1 + x2 + x3\n# Example structural model\nF2 ~ F1"
)

if st.button("Run SEM"):
    if not uploaded:
        st.error("Please upload a CSV file first.")
    else:
        data = load_data(uploaded)
        with st.spinner("Running SEM..."):
            model, estimates, fit_stats = run_sem(data, model_spec)

        st.subheader("Model Fit Statistics")
        st.write(fit_stats)

        st.subheader("Parameter Estimates")
        st.dataframe(estimates)

        st.subheader("Path Diagram")
        fig = plot_path_diagram(model)
        st.plotly_chart(fig)
