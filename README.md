
# No‑Code SEM Streamlit App

This starter kit lets you run Structural Equation Modeling (SEM) from an intuitive web UI—perfect for researchers who don't want to wrestle with code.

## Quick start

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit_app.py
```

Open the local URL that Streamlit prints (usually http://localhost:8501).

## Features

* CSV data upload  
* Lavaan‑style model syntax  
* Common fit indices (CFI, TLI, RMSEA, SRMR, etc.)  
* Parameter estimates table  
* Optional residuals table  
* Auto‑generated path diagram via Graphviz  

## License

MIT
