"""
Streamlit Web Dashboard for Fantasy Football Evaluation
Run via: .\.venv\Scripts\streamlit.exe run app.py
"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_data
def load_data():
    signals_path = os.path.join(DATA_DIR, "player_valuation_signals.csv")
    if not os.path.exists(signals_path):
        st.error("Valuation data file not found! Click 'Refresh Data' in the sidebar or run 04_generate_buy_sell_signals.py.")
        st.stop()
    df = pd.read_csv(signals_path)
    return df

st.set_page_config(
    page_title="Fantasy Football Undervalued Player Engine",
    page_icon="🏈",
    layout="wide"
)

# Sidebar Data Refresh Button
st.sidebar.title("⚡ Pipeline Controls")
if st.sidebar.button("🔄 Fetch & Update Latest NFL Data"):
    with st.spinner("Fetching latest NFL stats, injury reports, and retraining models..."):
        try:
            python_exe = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = "python"
                
            subprocess.run([python_exe, "01_fetch_data.py"], check=True)
            subprocess.run([python_exe, "02_feature_engineering.py"], check=True)
            subprocess.run([python_exe, "04_generate_buy_sell_signals.py"], check=True)
            
            st.cache_data.clear()
            st.sidebar.success("Pipeline executed successfully! Data refreshed.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Data refresh failed: {e}")

st.sidebar.divider()
st.sidebar.title("🏈 Valuation Filters")

df = load_data()

seasons = sorted(df['season'].unique(), reverse=True)
selected_season = st.sidebar.selectbox("Select Season", seasons)

weeks = sorted(df[df['season'] == selected_season]['week'].unique(), reverse=True)
selected_week = st.sidebar.selectbox("Select Week", weeks)

positions = ["ALL", "WR", "RB", "TE", "QB"]
selected_pos = st.sidebar.selectbox("Position Filter", positions)

signals = [
    "ALL", "FANTASY ALPHA", "ELITE BUY TARGET", "TRUE BUY LOW",
    "HIGH-VALUE PRODUCER", "SOLID STARTER", "TRUE SELL HIGH", "PERPETUAL TRAP"
]
selected_signal = st.sidebar.selectbox("Valuation Signal", signals)

# Filter Dataset
filtered_df = df[(df['season'] == selected_season) & (df['week'] == selected_week)].copy()

if selected_pos != "ALL":
    filtered_df = filtered_df[filtered_df['position'] == selected_pos]

if selected_signal != "ALL":
    filtered_df = filtered_df[filtered_df['valuation_signal'] == selected_signal]

# Header Title
st.title("🏈 Fantasy Football Player Valuation & Trade Target Engine")
st.markdown(f"**Season {selected_season} — Week {selected_week} Analysis**")

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Players Filtered", len(filtered_df))
col2.metric("Elite & True Buy Lows", len(filtered_df[filtered_df['valuation_signal'].isin(['ELITE BUY TARGET', 'TRUE BUY LOW'])]))
col3.metric("True Sell High Targets", len(filtered_df[filtered_df['valuation_signal'] == 'TRUE SELL HIGH']))
col4.metric("Fantasy Alphas", len(filtered_df[filtered_df['valuation_signal'] == 'FANTASY ALPHA']))

st.divider()

# Ensure DvP and lineup projection columns exist if reading an older CSV
for col_name, default_val in [('effective_next_opponent', 'UNK'), ('next_opp_dvp_total_adj', 0.0)]:
    if col_name not in filtered_df.columns:
        filtered_df[col_name] = default_val
if 'next_week_lineup_projection' not in filtered_df.columns:
    filtered_df['next_week_lineup_projection'] = np.where(
        filtered_df['effective_next_opponent'] == 'BYE', 0.0, filtered_df['predicted_next_week_ppr']
    )

# Tab Navigation
tab1, tab2, tab3 = st.tabs([
    "📊 Interactive Plotly Quadrant Analysis",
    "📋 Position-Filtered Valuation Leaderboard",
    "📈 Individual Player Trend Analysis"
])

with tab1:
    st.subheader("Volume Opportunity vs. Performance Quadrant")
    st.caption("Hover over any dot to see player stats, upcoming DvP matchup adjustment, injury designations, and Bayesian confidence!")
    
    if len(filtered_df) == 0:
        st.warning("No players match the current filter selection.")
    else:
        fig = px.scatter(
            filtered_df,
            x="recent_expected_3wk",
            y="recent_actual_3wk",
            color="valuation_signal",
            symbol="position",
            hover_name="player_name",
            hover_data={
                "position": True,
                "recent_team": True,
                "effective_next_opponent": True,
                "next_opp_dvp_total_adj": ":+.2f",
                "recent_expected_3wk": ":.1f",
                "recent_actual_3wk": ":.1f",
                "personal_efficiency_baseline": ":.2f",
                "predicted_next_week_ppr": ":.1f",
                "next_week_lineup_projection": ":.1f",
                "injury_status": True,
                "valuation_signal": True
            },
            labels={
                "recent_expected_3wk": "3-Week Schedule-Adjusted Expected Points (xFP)",
                "recent_actual_3wk": "3-Week Actual PPR Points (Output)",
                "effective_next_opponent": "Next Opponent",
                "next_opp_dvp_total_adj": "Next Opp DvP Boost",
                "predicted_next_week_ppr": "Per-Game Projection (Neutral/Matchup)",
                "next_week_lineup_projection": "Next Week Lineup Projection",
                "valuation_signal": "Valuation Signal",
                "position": "Position",
                "injury_status": "Injury Status"
            },
            title=f"Schedule-Adjusted Opportunity vs. Output ({selected_season} W{selected_week})"
        )
        
        max_val = max(filtered_df['recent_expected_3wk'].max(), filtered_df['recent_actual_3wk'].max()) + 2
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Parity Line (Expected = Actual)"
            )
        )
        
        fig.update_traces(marker=dict(size=12, opacity=0.85))
        fig.update_layout(
            height=600,
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=1),
            title_x=0.5,
            title_y=0.01,
            title_xanchor="center",
            title_yanchor="bottom",
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Valuation Leaderboard (Position & Signal Filtered)")
    
    pos_tab2 = st.multiselect(
        "Refine Positions for Table",
        options=["WR", "RB", "TE", "QB"],
        default=[selected_pos] if selected_pos != "ALL" else ["WR", "RB", "TE", "QB"]
    )
    
    table_df = filtered_df[filtered_df['position'].isin(pos_tab2)].copy()
    
    display_cols = [
        'player_name', 'position', 'recent_team', 'effective_next_opponent',
        'next_opp_dvp_total_adj', 'valuation_signal',
        'injury_status', 'predicted_next_week_ppr', 'next_week_lineup_projection',
        'recent_actual_3wk', 'recent_expected_3wk',
        'personal_efficiency_baseline', 'undervalued_index'
    ]
    
    # Ensure injury_status exists
    if 'injury_status' not in table_df.columns:
        table_df['injury_status'] = 'Healthy'
        
    leaderboard = table_df[display_cols].sort_values(by='undervalued_index', ascending=False)
    
    st.dataframe(
        leaderboard.style.format({
            'next_opp_dvp_total_adj': '{:+.2f}',
            'predicted_next_week_ppr': '{:.1f}',
            'next_week_lineup_projection': '{:.1f}',
            'recent_actual_3wk': '{:.1f}',
            'recent_expected_3wk': '{:.1f}',
            'personal_efficiency_baseline': '{:.2f}',
            'undervalued_index': '{:.1f}'
        }),
        use_container_width=True,
        height=500
    )

with tab3:
    st.subheader("Individual Player Season Trajectory")
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        all_players = sorted(df['player_name'].dropna().unique())
        selected_player = st.selectbox("Select Player to Inspect", all_players)
    with col_p2:
        player_season = st.selectbox("Select Season to Inspect", seasons, index=seasons.index(selected_season))
    
    player_df = df[(df['player_name'] == selected_player) & (df['season'] == player_season)].sort_values(by='week')
    
    if len(player_df) == 0:
        st.info(f"No records found for {selected_player} in the {player_season} season.")
    else:
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=player_df['week'],
            y=player_df['fantasy_points_ppr'],
            mode='lines+markers',
            name='Actual PPR Points',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8)
        ))
        
        fig2.add_trace(go.Scatter(
            x=player_df['week'],
            y=player_df['expected_fantasy_points_ppr'],
            mode='lines+markers',
            name='Expected Fantasy Points (xFP)',
            line=dict(color='firebrick', dash='dash', width=2),
            marker=dict(size=8)
        ))
        
        fig2.add_trace(go.Scatter(
            x=player_df['week'],
            y=player_df['predicted_next_week_ppr'],
            mode='lines+markers',
            name='Model Prediction (Next Week)',
            line=dict(color='forestgreen', dash='dot', width=2),
            marker=dict(size=8)
        ))
        
        fig2.update_layout(
            title=f"{selected_player} — Weekly Performance vs. Opportunity ({player_season})",
            xaxis_title="Week Number",
            yaxis_title="PPR Fantasy Points",
            height=450,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        st.write(f"**Personal Efficiency Baseline Ratio**: `{player_df['personal_efficiency_baseline'].iloc[0]:.2f}`")
