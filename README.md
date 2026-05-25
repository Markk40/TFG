# Controlling Dynamics from Data Centers: A Data-Driven Approach

**Bachelor's Thesis — Mathematical Engineering and Artificial Intelligence**  
Pontifical Comillas University (ICAI) · Madrid, 2026

**Author:** Marcos Alconchel Fernández  
**Supervisor:** Pablo Dueñas Martínez  
**Co-Supervisor:** Deepjyoti Deka (MIT)

---

## Overview

AI data centers consume rapidly growing amounts of electricity and generate
abrupt power fluctuations that can threaten grid frequency stability. During
the checkpointing and collective communication phases of distributed training
jobs, the power consumption of a GPU node can drop by several megawatts within
seconds, disrupting the generation-load balance on which grid frequency depends.

This repository contains the full implementation of a probabilistic pipeline
that characterises these fluctuations from real GPU power traces and generates
realistic synthetic time series for use in grid impact simulation studies.

---

## Pipeline

The generative model consists of three chained stages:

1. **ARIMA** — captures the temporal autocorrelation structure of the mean signal
2. **GARCH(1,1)** with Student-t innovations — models the conditional volatility produced by checkpointing events
3. **Extreme Value Theory (EVT)** via Peaks-Over-Threshold — fits a Generalised Pareto Distribution (GPD) independently to the upper and lower tails of the residual distribution

The pipeline is wrapped in a **Streamlit web application** that guides the user
from raw data upload through model fitting to synthetic trace generation and
export, without requiring any coding.

---

## Repository Structure

```
project/
├── app.py                  # Streamlit application entry point
├── page_modules/           # Application pages
│   ├── load_data.py
│   ├── identify_model.py
│   ├── fit_GARCH_EVT.py
│   └── generate.py
├── src/
│   ├── config.py           # Configuration and scenario presets
│   ├── preprocessing.py    # Data loading, warm-up removal, resampling
│   ├── arima.py            # ADF test, ACF/PACF, ARIMA fitting
│   ├── garch.py            # GARCH(1,1) volatility modelling
│   ├── evt.py              # GPD fitting and KS validation
│   ├── generation.py       # Synthetic trace generation
│   └── plotting.py         # Plotly visualisation functions
├── notebooks/
│   ├── statistical_analysis.ipynb
│   ├── pipeline_development.ipynb
│   └── rf_baseline.ipynb
├── requirements.txt
└── README.md
```

---

## Usage

### Streamlit Application

```bash
streamlit run app.py
```

The application guides you through four steps:

1. **Load Data** — upload a CSV or XLSX file with GPU power measurements, select columns, and remove warm-up and end-of-job periods
2. **Identify Model** — inspect ACF/PACF plots and fit an ARIMA model
3. **Fit GARCH + EVT** — fit the volatility model and select GPD thresholds via an interactive Mean Excess Plot
4. **Generate** — adjust parameters via sliders and download synthetic traces in CSV or XLSX format

### Input Format

The application expects a file with at least one timestamp column and one or
more numeric power columns (one per GPU). Multiple GPU columns are summed
automatically to obtain total node power.

---

## Results

Applied to a GPT-2 124M training run on 8 NVIDIA H100 GPUs:

| Metric | Random Forest | ARIMA-GARCH-EVT |
|---|---|---|
| Global Wasserstein distance (W) | 1,729 | 24 |
| Tail KS p-value | ~0 | 0.485 |
| Min power reproduced (W) | 2,085 | 1,298 (original: 1,252) |
| Mean error (W) | 1,728 | 16 |

---

## Data

The power consumption traces used in this project were provided by researchers
at MIT and collected on the MIT SuperCloud infrastructure. The datasets cover
training, fine-tuning, and inference workloads on NVIDIA H100 and A100 GPUs.
The raw data is not included in this repository.

---

## Citation

```
Alconchel Fernández, M. (2026). Controlling Dynamics from Data Centers:
A Data-Driven Approach. Bachelor's Thesis, Pontifical Comillas University (ICAI).
```

---

## License

This project is released for academic and research purposes.
