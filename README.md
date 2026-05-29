# Auckland Land Price Estimator
### COMP 717 — Assignment 2, Question 5

An interactive tool for town planners to estimate residential land prices across Auckland,
built with Streamlit and an RBF (Radial Basis Function) spatial model.

---

## 🚀 Live Demo

> [Add your Streamlit Cloud URL here after deploying]

---

## Running Locally

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/auckland-land-price-tool
cd auckland-land-price-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select this repo → set main file to `app.py`
5. Click **Deploy** — done. The app auto-redeploys on every push.

---

## Model

The spatial price model uses **Radial Basis Function interpolation** (thin-plate spline kernel)
fitted to median house price data for ~30 Auckland suburbs.

Given a latitude/longitude, the model:
1. Checks if the point falls within a known non-residential zone (sea, reserve)
2. If residential, interpolates price from surrounding suburb data points
3. Returns an estimated median price in NZD

### Upgrading the dataset

The current dataset uses Appendix A from the assignment brief. To improve accuracy:

- Download property valuation data from [LINZ Data Service](https://data.linz.govt.nz)
- Or use Auckland Council's open data portal
- Replace `SUBURB_DATA` in `app.py` with your richer dataset
- The model will automatically refit on startup

---

## Project Structure

```
auckland-land-price-tool/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## References

- Russell, S. & Norvig, P. (2018). *Artificial Intelligence: A Modern Approach* (4th ed.)
- Blick, G., Li, C., & Stewart, J. (2025). *Where Auckland Wants to Live*. Auckland Council.
- LINZ Data Service: https://data.linz.govt.nz
