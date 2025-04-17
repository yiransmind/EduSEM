# Streamlit SEM App

This app lets non-coders upload data, specify a SEM model via a simple text notation, and run the analysis. Outputs include:

- Model fit indices (CFI, TLI, RMSEA, SRMR)
- Parameter estimates (loadings, regressions, covariances)
- Standard errors and p-values
- Diagrams of the SEM path model
- Publication-ready tables and plots

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Upload a CSV file containing your data.
2. Enter your SEM model specification (e.g., `# measurement
F1 =~ x1 + x2 + x3`).
3. Click **Run SEM** to see results.
