
# SEM Explorer (Streamlit)

A lightweight Streamlit app to run SEM models using `semopy`. Users can:

- Upload their own dataset (.csv)
- Specify measurement and structural models using lavaan-style syntax
- View model fit indices, reliability (Cronbach's alpha), and path coefficients
- Download results as CSV tables

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- Only continuous indicators supported
- Syntax compatible with lavaan/semopy style (no higher-order constructs)
