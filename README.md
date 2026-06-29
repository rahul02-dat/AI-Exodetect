# ◎ ExoDetect — AI-Powered Exoplanet Transit Detection

An end-to-end web application that fetches **real TESS satellite data** from NASA's MAST archive, runs a Box Least Squares (BLS) periodogram to detect periodic transit signals, and classifies them using a heuristic AI ensemble — all visualised through an interactive dark-mode dashboard.

---

## ✨ Features

- **Real TESS Data** — Fetches 2-minute cadence PDCSAP light curves via `lightkurve` from the MAST archive
- **BLS Periodogram** — Searches a grid of 5,000 trial periods (0.5–27 days) to identify the strongest periodic signal
- **Transit Parameter Estimation** — Extracts orbital period, transit depth, duration, epoch (T₀), Rp/R★, SNR, and transit count
- **Phase-Folded View** — Folds the light curve on the best-fit period to visually confirm the transit shape
- **AI Classification** — Heuristic ensemble classifier that distinguishes Exoplanet Transits, Eclipsing Binaries, Stellar Blends, Starspots, and Stellar Variability
- **Data Quality Metrics** — Reports completeness, CDPP noise, and systematic noise levels
- **Live Backend Status** — Frontend displays real-time connection status to the Flask backend
- **Interactive Light Curve** — Canvas-rendered plot with crosshair hover showing time and flux values

---

## 🏗️ Project Structure

```
Exodetect/
├── server.py              # Flask backend — TESS data pipeline
├── .gitignore
├── README.md
├── env/                   # Python virtual environment (not tracked)
└── exodetect/             # React + Vite frontend
    ├── src/
    │   ├── App.jsx        # Main application component
    │   ├── main.jsx       # React entry point
    │   ├── App.css
    │   └── index.css
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`

### 1. Clone the Repository

```bash
git clone https://github.com/rahul02-dat/AI-Exodetect.git
cd AI-Exodetect
```

### 2. Set Up the Python Backend

```bash
python -m venv env
source env/bin/activate        # macOS/Linux
pip install flask flask-cors numpy lightkurve astropy scikit-learn
```

### 3. Start the Backend Server

```bash
./env/bin/python server.py
```

The API will be available at `http://localhost:8000`.

### 4. Set Up the React Frontend

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
| GET    | `/api/targets`    | List of pre-configured stellar targets    |
| GET    | `/api/analyse`    | Run full pipeline on a target (`?target=`) |

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
| Frontend  | React 19, Vite 8, Canvas API                     |
| Backend   | Flask, Flask-CORS                                 |
| Data      | lightkurve, Astropy (BoxLeastSquares), NumPy      |
| Source     | NASA TESS / MAST Archive                          |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
