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
    st.subheader("Player Season Trajectory & Multi-Player Comparison")
    
    col_p1, col_p2, col_p3 = st.columns([2, 1, 2])
    with col_p1:
        player_season = st.selectbox("Select Season to Inspect", seasons, index=seasons.index(selected_season))
    with col_p2:
        pos_filter_tab3 = st.selectbox("Filter Player Pool by Position", ["ALL", "WR", "RB", "TE", "QB"], index=0)
        
    season_pool_df = df[df['season'] == player_season]
    if pos_filter_tab3 != "ALL":
        season_pool_df = season_pool_df[season_pool_df['position'] == pos_filter_tab3]
        
    all_players = sorted(season_pool_df['player_name'].dropna().unique())
    default_players = [all_players[0]] if len(all_players) > 0 else []
    
    with col_p1:
        selected_players = st.multiselect(
            "Select Player(s) to Compare",
            options=all_players,
            default=default_players
        )
    with col_p3:
        selected_metrics = st.multiselect(
            "Metrics to Display on Chart",
            options=["Actual PPR Points", "Expected Fantasy Points (xFP)", "Model Prediction"],
            default=["Actual PPR Points", "Expected Fantasy Points (xFP)", "Model Prediction"]
        )
    
    if len(selected_players) == 0:
        st.info("Please select at least one player above to display their trajectory.")
    elif len(selected_metrics) == 0:
        st.info("Please select at least one metric to display on the chart.")
    else:
        fig2 = go.Figure()
        player_palette = [
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#ff7f0e", "#17becf", "#e377c2", "#bcbd22"
        ]
        is_multi = len(selected_players) > 1
        comparison_rows = []
        
        for idx, p_name in enumerate(selected_players):
            p_df = df[(df['player_name'] == p_name) & (df['season'] == player_season)].sort_values(by='week')
            if len(p_df) == 0:
                continue
                
            comparison_rows.append(p_df.iloc[-1])
            base_color = player_palette[idx % len(player_palette)]
            
            last_played_week = int(p_df['week'].iloc[-1])
            next_target_week = last_played_week + 1
            
            if 'projected_next_week_xfp' in p_df.columns:
                upcoming_xfp = float(p_df['projected_next_week_xfp'].iloc[-1])
            else:
                next_xfp_boost = float(p_df['next_opp_dvp_xfp_adj'].iloc[-1]) if 'next_opp_dvp_xfp_adj' in p_df.columns else 0.0
                upcoming_xfp = max(0.0, float(p_df['recent_expected_3wk'].iloc[-1]) + next_xfp_boost)
                
            # 1. Actual PPR Points (Completed Weeks)
            if "Actual PPR Points" in selected_metrics:
                fig2.add_trace(go.Scatter(
                    x=p_df['week'],
                    y=p_df['fantasy_points_ppr'],
                    mode='lines+markers',
                    name=f"{p_name} — Actual PPR" if is_multi else "Actual PPR Points",
                    legendgroup=p_name,
                    line=dict(color=base_color if is_multi else 'royalblue', width=3, dash='solid'),
                    marker=dict(size=8, symbol='circle')
                ))
            
            # 2. Expected Fantasy Points (xFP): Realized xFP + Projected Pre-Game xFP for Next Week (t+1)
            if "Expected Fantasy Points (xFP)" in selected_metrics:
                x_xfp = list(p_df['week']) + [next_target_week]
                y_xfp = list(p_df['expected_fantasy_points_ppr']) + [upcoming_xfp]
                
                fig2.add_trace(go.Scatter(
                    x=x_xfp,
                    y=y_xfp,
                    mode='lines+markers',
                    name=f"{p_name} — Expected (xFP)" if is_multi else "Expected Fantasy Points (xFP)",
                    legendgroup=p_name,
                    line=dict(color=base_color if is_multi else 'firebrick', width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
            
            # 3. Model PPR Prediction: Pre-Week 1 Prior + Target-Week Aligned + Next Week (t+1)
            if "Model Prediction" in selected_metrics:
                x_pred = list(p_df['week'].iloc[1:]) + [next_target_week]
                y_pred = list(p_df['predicted_next_week_ppr'])
                if 'pre_week1_pred_ppr' in p_df.columns and pd.notna(p_df['pre_week1_pred_ppr'].iloc[0]):
                    x_pred = [int(p_df['week'].iloc[0])] + x_pred
                    y_pred = [float(p_df['pre_week1_pred_ppr'].iloc[0])] + y_pred
                
                fig2.add_trace(go.Scatter(
                    x=x_pred,
                    y=y_pred,
                    mode='lines+markers',
                    name=f"{p_name} — Model Pred" if is_multi else "Model Prediction (Target Week Aligned)",
                    legendgroup=p_name,
                    line=dict(color=base_color if is_multi else 'forestgreen', width=2, dash='dot'),
                    marker=dict(size=8, symbol='square')
                ))
        
        title_str = (
            f"Multi-Player Comparison ({ ', '.join(selected_players) }) — {player_season}"
            if is_multi else
            f"{selected_players[0]} — Weekly Performance vs. Opportunity ({player_season})"
        )
        
        fig2.update_layout(
            title=title_str,
            xaxis_title="Week Number",
            yaxis_title="PPR Fantasy Points",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        if len(comparison_rows) > 0:
            st.markdown("#### 📋 Selected Player(s) Head-to-Head Summary (Latest Week in Season)")
            comp_df = pd.DataFrame(comparison_rows)
            if 'projected_next_week_xfp' not in comp_df.columns:
                comp_df['projected_next_week_xfp'] = comp_df['recent_expected_3wk']
            if 'injury_status' not in comp_df.columns:
                comp_df['injury_status'] = 'Healthy'
                
            summary_cols = [
                'player_name', 'position', 'recent_team', 'week',
                'effective_next_opponent', 'next_opp_dvp_total_adj',
                'valuation_signal', 'predicted_next_week_ppr', 'projected_next_week_xfp',
                'recent_actual_3wk', 'recent_expected_3wk',
                'personal_efficiency_baseline', 'undervalued_index'
            ]
            valid_summary_cols = [c for c in summary_cols if c in comp_df.columns]
            comp_display = comp_df[valid_summary_cols].sort_values(by='undervalued_index', ascending=False)
            
            st.dataframe(
                comp_display.style.format({
                    'next_opp_dvp_total_adj': '{:+.2f}',
                    'predicted_next_week_ppr': '{:.1f}',
                    'projected_next_week_xfp': '{:.1f}',
                    'recent_actual_3wk': '{:.1f}',
                    'recent_expected_3wk': '{:.1f}',
                    'personal_efficiency_baseline': '{:.2f}',
                    'undervalued_index': '{:.1f}'
                }),
                use_container_width=True
            )
