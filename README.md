# ◎ ExoDetect — AI-Powered Exoplanet Transit Detection

An end-to-end web application that fetches **real TESS satellite data** from NASA's MAST archive, runs a Box Least Squares (BLS) periodogram to detect periodic transit signals, and classifies them using a **Dual-Branch 1D Convolutional Neural Network (CNN)** built with PyTorch — all visualised through an interactive dark-mode dashboard.

---

## ✨ Features

- **Real TESS Data** — Fetches 2-minute cadence PDCSAP light curves via `lightkurve` from the MAST archive
- **BLS Periodogram** — Searches a grid of 5,000 trial periods (0.5–27 days) to identify the strongest periodic signal
- **Dual-Branch 1D CNN** — Uses a custom PyTorch CNN (trained on Kepler DR25 TCE data) processing both global (full period) and local (zoomed) transit views to classify signals with high precision
- **Transit Parameter Estimation** — Extracts orbital period, transit depth, duration, epoch (T₀), Rp/R★, SNR, and transit count
- **Phase-Folded View** — Folds the light curve on the best-fit period to visually confirm the transit shape
- **AI Classification** — The CNN distinguishes Exoplanet Transits from False Positives (Eclipsing Binaries, Stellar Blends, Starspots)
- **MCMC Parameter Fitting** — Runs a full Markov Chain Monte Carlo (MCMC) using `emcee` to sample transit parameters and generate posteriors
- **Data Quality Metrics** — Reports completeness, CDPP noise, and systematic noise levels
- **Live Backend Status** — Frontend displays real-time connection status to the Flask backend
- **Interactive Light Curve** — Canvas-rendered plot with crosshair hover showing time and flux values

---

## 🏗️ Project Structure

```
Exodetect/
├── server.py                 # Flask backend — TESS data pipeline & CNN inference
├── tasks.py                  # Celery worker for async MCMC fitting
├── mcmc_fitter.py            # MCMC transit modeling using emcee and batman
├── exodetect_cnn.pt          # Exported PyTorch model weights (generated after training)
├── ExoDetect_Phase1_CNN.ipynb# Jupyter notebook for training the CNN 
├── data/
│   └── kepler_lcs/           # Directory for Kepler DR25 training data
├── env/                      # Python virtual environment (not tracked)
└── exodetect/                # React + Vite frontend
    ├── src/
    │   ├── App.jsx           # Main application component
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
- **Redis** (required for async MCMC background tasks)

### 1. Clone the Repository

```bash
git clone https://github.com/rahul02-dat/AI-Exodetect.git
cd AI-Exodetect
```

### 2. Set Up the Python Backend

```bash
python3 -m venv env
source env/bin/activate        # macOS/Linux
pip install flask flask-cors numpy lightkurve astropy scikit-learn torch pandas scipy celery redis emcee batman-package corner
```

Ensure Redis is installed and running:
```bash
# On macOS
brew install redis
brew services start redis
```

### 3. (Optional) Train the CNN Model

If you want to train the model from scratch using the Kepler dataset:
```bash
# 1. Download light curves and build dataset.npz

# 2. Train the CNN and export exodetect_cnn.pt
python train_model.py

# 3. Ensure the model is in the root directory for the server
cp data/kepler_lcs/exodetect_cnn.pt ./exodetect_cnn.pt
```
*(If `exodetect_cnn.pt` is not found, the backend automatically falls back to a basic heuristic classifier).*

### 4. Start the Backend Server

Start the Flask API:
```bash
python server.py
```
The API will be available at `http://localhost:8000`. You should see `[INFO] CNN model loaded` in the console.

Open a new terminal, activate the environment, and start the Celery worker for MCMC processing:
```bash
source env/bin/activate
celery -A tasks worker --loglevel=info --concurrency=2
```

### 5. Set Up the React Frontend

Open a new terminal tab:
```bash
cd exodetect
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## 🔌 API Endpoints

| Method | Endpoint          | Description                               |
|--------|-------------------|-------------------------------------------|
| GET    | `/api/health`     | Backend health check                      |
| GET    | `/api/model-info` | Returns information about the loaded CNN  |
| GET    | `/api/targets`    | List of pre-configured stellar targets    |
| GET    | `/api/analyse`    | Run full pipeline on a target (`?target=`) |
| POST   | `/api/mcmc/start` | Start an async MCMC fitting job           |
| GET    | `/api/mcmc/status`| Poll progress of an MCMC job (`?job_id=`) |
| GET    | `/api/mcmc/result`| Fetch completed MCMC results (`?job_id=`) |

### Example

```bash
curl "http://localhost:8000/api/analyse?target=L%2098-59"
```

---

## 🎯 Pre-Configured Targets

| Target           | Type                 | Notes                              |
|------------------|----------------------|------------------------------------|
| L 98-59          | Exoplanet Transit    | 3 terrestrial planets              |
| TOI-700          | Exoplanet Transit    | Habitable-zone Earth-size planet   |
| WASP-18          | Exoplanet Transit    | Hot Jupiter, ~1 day period         |
| TIC 286923464    | Exoplanet Transit    | HD 118203 b — eccentric orbit      |
| HD 21749         | Exoplanet Transit    | Sub-Neptune + super-Earth          |
| Beta Pictoris    | Stellar Variability  | Debris disk + direct imaging planet|

> You can also type any valid TIC ID or star name in the search bar.

---

## 🛠️ Tech Stack

| Layer     | Technology                                        |
|-----------|---------------------------------------------------|
| Frontend  | React 19, Vite 8, Canvas API                      |
| Backend   | Flask, Flask-CORS, Celery, Redis                  |
| Machine Learning | PyTorch, Scikit-Learn, Pandas              |
| Astrophysics | lightkurve, Astropy, emcee, batman, NumPy      |
| Source    | NASA TESS / MAST Archive & Kepler DR25 Catalog    |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
