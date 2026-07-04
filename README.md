# ◎ ExoDetect — AI-Powered Exoplanet Transit Detection & Analysis

An end-to-end web application that fetches **real TESS satellite data** from NASA's MAST archive, runs a Box Least Squares (BLS) periodogram to detect periodic transit signals, and classifies them using an **Ensemble Model (Dual-Branch 1D CNN + TransitFormer)** built with PyTorch. It features an interactive dark-mode dashboard, Explainable AI (XAI) attention heatmaps, automated full-sector batch processing, async MCMC parameter fitting, and downloadable PDF reports.

---

## ✨ Features

- **Real TESS Data Retrieval** — Fetches 2-minute cadence PDCSAP light curves via `lightkurve` from the MAST archive.
- **BLS Periodogram Search** — Searches a grid of 5,000 trial periods (0.51–27 days) to identify the strongest periodic signal.
- **AI Ensemble Classification** — Soft-votes between a **Dual-Branch 1D CNN** (Kepler-trained) and a **TransitFormer** (1D Patch-based Transformer) to classify signals into Exoplanet Transits or False Positives (Eclipsing Binaries, Stellar Blends, Starspots).
- **Explainable AI (XAI)** — Displays a visual attention heatmap of the `TransitFormer`'s self-attention weights to explain what regions of the light curve drove its classification.
- **Async MCMC Parameter Fitting** — Runs a full Markov Chain Monte Carlo (MCMC) using `emcee` to sample transit parameters (Period, Rp/R★, Inc, a/R★) and generate posteriors.
- **Full TESS Sector Batch Processing** — Automatically downloads target lists from MAST for a given sector, scans them concurrently using the ensemble, and logs candidates.
- **PDF Report Generation** — Dynamically generates professional PDF analysis reports (with phase-folded light curves, BLS power, and MCMC parameter statistics) for individual candidates or full-sector batches.
- **Live Backend Status & Device Detection** — Real-time backend connectivity check and automatic device mapping (MPS for Apple Silicon, CUDA for Nvidia, CPU fallback).

---

## 🏗️ Project Structure

```
Exodetect/
├── server.py                 # Flask backend — API endpoints & ensemble inference
├── tasks.py                  # Celery tasks for async MCMC fitting (DB 0)
├── batch_pipeline.py         # Celery tasks for async full-sector scanning (DB 1)
├── transit_transformer.py    # TransitFormer model architecture & ExoEnsemble
├── train_transformer.py      # TransitFormer training script with early stopping & AUC logging
├── report_generator.py       # PDF generator using ReportLab and Matplotlib
├── mcmc_fitter.py            # MCMC transit modeling using emcee and batman
├── exodetect_cnn.pt          # Exported PyTorch CNN weights
├── exodetect_transformer.pt  # Exported PyTorch TransitFormer weights
├── data/
│   └── kepler_lcs/           # Directory for training data (dataset.npz)
├── env/                      # Python virtual environment (not tracked)
└── exodetect/                # React + Vite frontend
    ├── src/
    │   ├── App.jsx           # Main dashboard UI
    │   ├── main.jsx          # React entry point
    │   └── index.css
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- **Redis** (required for Celery task queuing)

### 1. Clone the Repository

```bash
git clone https://github.com/rahul02-dat/AI-Exodetect.git
cd AI-Exodetect
```

### 2. Set Up the Python Backend

```bash
python3 -m venv env
source env/bin/activate        # macOS/Linux
pip install flask flask-cors numpy lightkurve astropy scikit-learn torch pandas scipy celery redis emcee batman-package corner reportlab matplotlib
```

Ensure Redis is installed and running:
```bash
# On macOS
brew install redis
brew services start redis
```

### 3. Train the TransitFormer Model (Optional)

To train the new attention-based `TransitFormer` on the local Kepler dataset:
```bash
python train_transformer.py
```
This saves `exodetect_transformer.pt` and outputs a training curve plot to `transformer_training.png`.

### 4. Start the Services

#### A. Start the Flask API:
```bash
python server.py
```
The API starts on `http://localhost:8000`. You should see output confirming both the CNN and TransitFormer models were loaded successfully on your device (e.g. `mps` for Apple Silicon).

#### B. Start the MCMC Worker:
Open a new terminal tab and run:
```bash
./env/bin/celery -A tasks worker --loglevel=info --concurrency=2
```

#### C. Start the Batch Sector Worker:
Open a new terminal tab and run:
```bash
./env/bin/celery -A batch_pipeline worker --loglevel=info --concurrency=2
```

### 5. Set Up the React Frontend

Open a new terminal tab:
```bash
cd exodetect
npm install
npm run dev
```

The frontend dashboard will be available at `http://localhost:5173`.

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                                          |
|--------|-----------------------|------------------------------------------------------|
| GET    | `/api/health`         | Backend health check                                 |
| GET    | `/api/model-info`     | Returns active models (Ensemble, CNN, TF) & device   |
| GET    | `/api/targets`        | Pre-configured stellar targets                       |
| GET    | `/api/analyse`        | Run full analysis pipeline on a target (`?target=`)  |
| POST   | `/api/mcmc/start`     | Start an async MCMC fitting job                      |
| GET    | `/api/mcmc/status`    | Poll progress of an MCMC job (`?job_id=`)            |
| GET    | `/api/mcmc/result`    | Fetch completed MCMC results (`?job_id=`)            |
| POST   | `/api/batch/start`    | Start a sector batch scan (`sector`, `max_targets`)  |
| GET    | `/api/batch/status`   | Poll sector scan progress (`?job_id=`)               |
| GET    | `/api/batch/results`  | Fetch results of a batch scan (`?job_id=`)           |
| GET    | `/api/report/candidate`| Download dynamic PDF candidate report (`?target=`)  |
| GET    | `/api/report/sector`  | Download dynamic PDF sector summary (`?job_id=`)     |

---

## 🎯 Pre-Configured Targets

| Target           | Type                 | Notes                              |
|------------------|----------------------|------------------------------------|
| L 98-59          | Exoplanet Transit    | 3 terrestrial planets              |
| TOI-700          | Exoplanet Transit    | Habitable-zone Earth-size planet   |
| WASP-18          | Exoplanet Transit    | Hot Jupiter, ~1 day period         |
| TIC 286923464    | Exoplanet Transit    | HD 118203 b — eccentric orbit      |
| HD 21749         | Exoplanet Transit    | Sub-Neptune + super-Earth          |

---

## 🛠️ Tech Stack

| Layer     | Technology                                              |
|-----------|---------------------------------------------------------|
| Frontend  | React 19, Vite 8, HTML Canvas API                      |
| Backend   | Flask, Flask-CORS, Celery, Redis                        |
| Deep Learning | PyTorch, TransitFormer, 1D CNN                      |
| Astrophysics | lightkurve, Astropy, emcee, batman                   |
| Reporting | ReportLab, Matplotlib                                   |
| Source    | NASA TESS / MAST Archive & Kepler DR25 Catalog          |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
