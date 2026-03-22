"""
Plotly charts for the cyclone prediction dashboard.
"""

import plotly.graph_objects as go


def wind_forecast_chart(current_wind: float,
                        forecast_wind: dict,
                        history_wind: list = None) -> go.Figure:
    """
    Wind speed timeline with forecast overlay.

    Args:
        current_wind: current wind speed (kt)
        forecast_wind: {"24h_kt": ..., "48h_kt": ..., "72h_kt": ...}
        history_wind: optional list of historical wind speeds
    """
    fig = go.Figure()

    # Historical wind
    if history_wind and len(history_wind) > 0:
        hours_back = list(range(-6 * (len(history_wind) - 1), 1, 6))
        fig.add_trace(go.Scatter(
            x=hours_back, y=history_wind,
            mode="lines+markers",
            name="Observed",
            line=dict(color="#06b6d4", width=2),
            marker=dict(size=4),
        ))

    # Current + forecast
    forecast_hours = [0, 24, 48, 72]
    forecast_vals = [
        current_wind,
        forecast_wind.get("24h_kt", 0),
        forecast_wind.get("48h_kt", 0),
        forecast_wind.get("72h_kt", 0),
    ]

    fig.add_trace(go.Scatter(
        x=forecast_hours, y=forecast_vals,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#fbbf24", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))

    # Intensity thresholds
    fig.add_hline(y=33, line_dash="dot", line_color="#6ee7b7",
                  annotation_text="TS (33 kt)", annotation_position="top right")
    fig.add_hline(y=63, line_dash="dot", line_color="#fde68a",
                  annotation_text="Hurricane (63 kt)", annotation_position="top right")

    fig.update_layout(
        title="Wind Speed Forecast",
        xaxis_title="Hours from now",
        yaxis_title="Wind speed (knots)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6"),
        height=350,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def track_error_chart(test_results: dict) -> go.Figure:
    """
    Bar chart of track forecast error at 24h / 48h / 72h.

    Args:
        test_results: dict with track_24h_km, track_48h_km, track_72h_km
    """
    horizons = ["24h", "48h", "72h"]
    values = [
        test_results.get("track_24h_km", 0),
        test_results.get("track_48h_km", 0),
        test_results.get("track_72h_km", 0),
    ]

    colors = ["#06b6d4", "#8b5cf6", "#f43f5e"]

    fig = go.Figure(data=[
        go.Bar(
            x=horizons, y=values,
            marker_color=colors,
            text=[f"{v:.0f} km" for v in values],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        )
    ])

    fig.update_layout(
        title="Track Forecast Error",
        xaxis_title="Forecast Horizon",
        yaxis_title="Error (km)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6"),
        height=350,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
    )

    return fig


def ri_gauge(ri_probability: float) -> go.Figure:
    """
    RI probability gauge indicator.

    Args:
        ri_probability: 0-1 probability value
    """
    color = "#ef4444" if ri_probability > 0.5 else (
        "#fbbf24" if ri_probability > 0.2 else "#6ee7b7"
    )
    label = "HIGH RISK" if ri_probability > 0.5 else (
        "MODERATE" if ri_probability > 0.2 else "LOW"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ri_probability * 100,
        title={"text": f"Rapid Intensification — {label}",
               "font": {"color": "#e2e8f0", "size": 14}},
        number={"suffix": "%", "font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color},
            "bgcolor": "rgba(30,58,95,0.5)",
            "borderwidth": 1,
            "bordercolor": "#1e3a5f",
            "steps": [
                {"range": [0, 20], "color": "rgba(110,231,183,0.15)"},
                {"range": [20, 50], "color": "rgba(251,191,36,0.15)"},
                {"range": [50, 100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6"),
        height=250,
        margin=dict(l=30, r=30, t=60, b=20),
    )

    return fig


def model_comparison_chart(lstm_results: dict,
                           hybrid_results: dict) -> go.Figure:
    """
    Side-by-side comparison of LSTM vs Hybrid model metrics.
    """
    metrics = ["track_24h_km", "track_48h_km", "track_72h_km", "wind_mae_kt"]
    labels  = ["Track 24h (km)", "Track 48h (km)", "Track 72h (km)", "Wind MAE (kt)"]

    lstm_vals   = [lstm_results.get(m, 0) for m in metrics]
    hybrid_vals = [hybrid_results.get(m, 0) for m in metrics]

    fig = go.Figure(data=[
        go.Bar(name="LSTM Baseline", x=labels, y=lstm_vals,
               marker_color="#06b6d4"),
        go.Bar(name="Hybrid (LSTM+ConvLSTM+Transformer)", x=labels, y=hybrid_vals,
               marker_color="#8b5cf6"),
    ])

    fig.update_layout(
        title="Model Comparison",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6"),
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.98),
    )

    return fig
