---
title: Cyclone Predict AI
emoji: 🌀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: dashboard/app.py
pinned: false
---

# 🌀 Cyclone Predict AI: North Indian Basin Forecasting

**Cyclone Predict AI** is a state-of-the-art deep learning dashboard designed to track and forecast tropical cyclones in the North Indian (NI) Ocean basin, with a primary focus on the Bay of Bengal. 

Inspired by **Google DeepMind's Weather Lab**, this application provides a professional-grade interface for meteorologists and researchers to visualize AI-driven trajectory and intensity predictions.

### 🧪 Key AI Features (Weather Lab Emulation)
*   **Deep Learning Ensembles**: Generates a 50-member probabilistic ensemble forecast to visualize the "cone of uncertainty" based on LSTM and Hybrid (ConvLSTM + Transformer) architectures.
*   **Expert Mode**: Visualizes early-stage **Cyclogenesis Hotspots** (2% formation probability clusters) and high-resolution atmospheric data.
*   **Official Baseline Comparison**: Toggle between AI-predicted tracks and simulated official baseline models for performance benchmarking.
*   **Rapid Intensification (RI) Alerts**: Automatic detection and alerting for storms likely to intensify by ≥35 knots within 24 hours.

### 🛠️ Technology Stack
*   **Models**: PyTorch-based LSTM and Hybrid models (ConvLSTM + Transformer layers).
*   **Dashboard**: Streamlit with custom glassmorphic CSS.
*   **Visualizations**: Folium (Leaflet) for maps and Plotly for high-speed intensity charting.
*   **Data Source**: Near real-time ERA5 atmospheric data and historical IBTrACS archives.

## 🚀 Getting Started
```bash
# Clone the repository
git clone https://github.com/subrata0xo/vortex-predict-ai.git
cd vortex-predict-ai

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python -m streamlit run dashboard/app.py
```
