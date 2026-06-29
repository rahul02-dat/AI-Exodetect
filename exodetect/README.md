# ExoDetect — Frontend

Interactive React dashboard for the ExoDetect transit detection pipeline. Built with **React 19** and **Vite 8**, using the Canvas API for high-performance scientific visualisations.

## Setup

```bash
npm install
npm run dev
```

## Scripts

| Command           | Description                    |
|-------------------|--------------------------------|
| `npm run dev`     | Start dev server (port 5173)   |
| `npm run build`   | Production build to `dist/`    |
| `npm run preview` | Preview production build       |
| `npm run lint`    | Run ESLint                     |

## Key Components

- **LightCurveCanvas** — Renders the TESS light curve with interactive crosshair hover
- **PhaseFoldedCanvas** — Phase-folded transit view on the best-fit period
- **PeriodogramCanvas** — BLS periodogram with peak period annotation
- **PipelineStatus** — Animated step-by-step pipeline progress indicator
- **ConfBar** — Confidence bar for AI classification probabilities

## Backend Dependency

This frontend requires the Flask backend (`server.py` in the project root) to be running on `http://localhost:8000`. See the [root README](../README.md) for setup instructions.
