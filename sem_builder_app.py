import streamlit as st
import pandas as pd
import semopy
import matplotlib.pyplot as plt
import seaborn as sns
import pdfkit
import os
from io import BytesIO
import base64
from semopy import semplot

st.title("SEM Builder and Estimator")

st.markdown("""
This app allows you to:
1. Upload a dataset (CSV).
2. Specify a measurement and structural model (in lavaan/semopy style).
3. Estimate the model and see fit indices, parameter estimates.
4. Visualize the path diagram and correlation heatmap.
5. Generate and download a report (PDF or Markdown).
""")

##########################
# 1. Data Upload Section #
##########################
uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()  # Stop execution until a file is uploaded.

############################
# 2. Model Specification   #
############################
st.subheader("Model Specification")

st.markdown("### Example Model Syntax (lavaan-style for semopy)")
example_model_syntax = """# Measurement model
# Let's say we have two latent constructs: 'Latent1' and 'Latent2'
# with indicators: x1, x2, x3 for Latent1 and x4, x5 for Latent2

Latent1 =~ x1 + x2 + x3
Latent2 =~ x4 + x5

# Structural relationships
# Let Latent2 be predicted by Latent1
Latent2 ~ Latent1
"""

st.markdown(
    "Below is an example you can modify. "
    "In semopy/lavaan style, use `Latent =~ indicator1 + indicator2` for measurement, "
    "and `DV ~ IV1 + IV2` for structural paths."
)

model_syntax = st.text_area("Input SEM Syntax", value=example_model_syntax, height=200)

##########################
# 3. Model Estimation    #
##########################
if st.button("Estimate Model"):
    try:
        # Build the semopy model
        model = semopy.Model(model_syntax)
        
        # Fit the model
        model.fit(df)
        
        st.success("Model estimation completed successfully.")
        
        # Extract Fit Indices
        stats = semopy.calc_stats(model)
        
        # Common fit indices
        rmsea = stats.loc['RMSEA', 'Value'] if 'RMSEA' in stats.index else None
        cfi   = stats.loc['CFI', 'Value'] if 'CFI' in stats.index else None
        tli   = stats.loc['TLI', 'Value'] if 'TLI' in stats.index else None
        srmr  = stats.loc['SRMR', 'Value'] if 'SRMR' in stats.index else None
        chi2  = stats.loc['Chi2', 'Value'] if 'Chi2' in stats.index else None
        
        st.subheader("Model Fit Indices")
        fit_dict = {
            "RMSEA": rmsea,
            "CFI": cfi,
            "TLI": tli,
            "SRMR": srmr,
            "Chi-square": chi2
        }
        fit_df = pd.DataFrame(list(fit_dict.items()), columns=["Fit Index", "Value"])
        st.table(fit_df)

        # Standardized parameter estimates
        st.subheader("Parameter Estimates (Standardized)")
        param_estimates = model.inspect(std_est=True)
        st.dataframe(param_estimates)

        ##########################
        # 4. Visualization       #
        ##########################
        st.subheader("Visualization")

        # 4a. SEM Path Diagram
        st.markdown("**SEM Path Diagram**")
        try:
            graph = semplot(model, show=False)
            dot_source = graph.source
            st.graphviz_chart(dot_source)
        except Exception as e:
            st.warning(f"Could not generate path diagram. Check Graphviz installation. Error: {e}")

        # 4b. Correlation Heatmap
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots()
        corr = df.corr()
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu", center=0)
        st.pyplot(fig)

        ##########################
        # 5. Report Generation   #
        ##########################
        st.subheader("Generate Report")

        report_md = f"""# SEM Report

## Model Specification

## Fit Indices
| Index     | Value   |
|-----------|--------:|
| RMSEA     | {rmsea} |
| CFI       | {cfi}   |
| TLI       | {tli}   |
| SRMR      | {srmr}  |
| Chi-square| {chi2}  |

## Standardized Parameter Estimates
{param_estimates.to_markdown(index=False)}

*(Path Diagram and Correlation Heatmap omitted in the markdown file – see interactive app for visualizations.)*
"""

        # 5a. Download Markdown
        st.download_button(
            label="Download Report as Markdown",
            data=report_md,
            file_name="SEM_report.md",
            mime="text/markdown"
        )

        # 5b. Download PDF (requires pdfkit & wkhtmltopdf installed)
        if st.button("Generate and Download PDF"):
            try:
                pdf_file = "SEM_report.pdf"
                import markdown
                html_text = markdown.markdown(report_md)
                pdfkit.from_string(html_text, pdf_file)

                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()
                os.remove(pdf_file)
                
                b64 = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="SEM_report.pdf">Download PDF file</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating PDF. Check if wkhtmltopdf is installed. Details: {e}")

    except Exception as e:
        st.error(f"Error during model estimation: {e}")

