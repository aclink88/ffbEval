"""
Data Ingestion Script using Polars & nflreadpy
Dynamically fetches historical seasons + current active live NFL season (2026 Week 2+).
Includes Next-Week Scheduled Opponent lookup via nflreadpy.load_schedules().
"""

import sys
import os
import polars as pl
import nflreadpy as nfl

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_target_seasons(years_back: int = 3) -> list[int]:
    """Dynamically calculates target seasons based on current active NFL season (default: 3 years back)."""
    current_season = nfl.get_current_season()
    seasons = list(range(current_season - years_back, current_season + 1))
    return seasons

def fetch_weekly_data(years: list[int]) -> pl.DataFrame:
    """Fetch player weekly stats for given seasons natively as a Polars DataFrame."""
    print(f"Fetching live weekly player stats for seasons {years} via nflreadpy...")
    polars_df = nfl.load_player_stats(years)
    return polars_df

def fetch_injury_data(years: list[int]) -> pl.DataFrame:
    """Fetch live weekly injury reports using nflreadpy."""
    print(f"Fetching injury reports for seasons {years} via nflreadpy...")
    inj_df = nfl.load_injuries(years)
    return inj_df

def fetch_player_draft_metadata() -> pl.DataFrame:
    """Fetch player draft capital and rookie season metadata via nflreadpy.load_players()."""
    print("Fetching player draft capital metadata via nflreadpy...")
    players_df = nfl.load_players()
    draft_meta = players_df.select([
        pl.col("gsis_id").alias("player_id"),
        pl.col("rookie_season").cast(pl.Int64),
        pl.col("draft_year").cast(pl.Int64),
        pl.col("draft_round").cast(pl.Int64),
        pl.col("draft_pick").cast(pl.Float64)
    ]).unique(subset=["player_id"])
    return draft_meta

def fetch_next_week_schedule_map(years: list[int]) -> pl.DataFrame:
    """
    Fetches NFL schedule for given seasons and creates a lookup mapping:
    (season, week, recent_team) -> next_opponent_team (who recent_team plays in week + 1).
    """
    print(f"Fetching NFL schedules for seasons {years} via nflreadpy...")
    sched = nfl.load_schedules(years)
    
    # Unpivot home and away perspectives to get (season, week, team, opponent)
    home_view = sched.select([
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("home_team").alias("recent_team"),
        pl.col("away_team").alias("scheduled_opp")
    ])
    away_view = sched.select([
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("away_team").alias("recent_team"),
        pl.col("home_team").alias("scheduled_opp")
    ])
    
    team_sched = pl.concat([home_view, away_view]).unique(subset=["season", "week", "recent_team"])
    
    # Shift schedule back by 1 week so row at week `t` knows who `recent_team` plays in week `t + 1`
    next_opp_lookup = team_sched.with_columns(
        (pl.col("week") - 1).alias("week")
    ).rename({"scheduled_opp": "next_opponent_team"})
    
    return next_opp_lookup

def main():
    ensure_data_dir()
    
    # Dynamically detect active live season + 3 historical backtest seasons (e.g. 2023, 2024, 2025, 2026)
    seasons = get_target_seasons(years_back=3)
    print(f"Detected active NFL seasons to process: {seasons} (Current Week: {nfl.get_current_week()})")
    
    # 1. Fetch live weekly player stats
    weekly_df = fetch_weekly_data(seasons)
    
    # Standardize column mappings (team -> recent_team)
    if "team" in weekly_df.columns and "recent_team" not in weekly_df.columns:
        weekly_df = weekly_df.with_columns(pl.col("team").alias("recent_team"))
        
    weekly_df = weekly_df.with_columns([
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64)
    ])
        
    # 2. Fetch schedule and merge next_opponent_team
    try:
        next_opp_lookup = fetch_next_week_schedule_map(seasons)
        weekly_df = weekly_df.join(next_opp_lookup, on=["season", "week", "recent_team"], how="left")
        weekly_df = weekly_df.with_columns(
            pl.col("next_opponent_team").fill_null("BYE")
        )
    except Exception as e:
        print(f"Warning: Next opponent schedule merge skipped due to: {e}")
        weekly_df = weekly_df.with_columns(pl.lit("BYE").alias("next_opponent_team"))
        
    # 3. Fetch injury reports and join on player & week
    try:
        inj_df = fetch_injury_data(seasons)
        if "full_name" in inj_df.columns and "player_name" not in inj_df.columns:
            inj_df = inj_df.with_columns(pl.col("full_name").alias("player_name"))
            
        inj_clean = inj_df.select([
            pl.col("player_name"),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("report_status").alias("injury_status"),
            pl.col("report_primary_injury").alias("primary_injury")
        ]).unique(subset=["player_name", "season", "week"])
        
        # Left join injury info onto weekly stats
        weekly_df = weekly_df.join(inj_clean, on=["player_name", "season", "week"], how="left")
    except Exception as e:
        print(f"Warning: Injury data merge skipped due to: {e}")
        weekly_df = weekly_df.with_columns([
            pl.lit("Healthy").alias("injury_status"),
            pl.lit(None).alias("primary_injury")
        ])

    # 4. Fetch and merge player draft capital & rookie season metadata
    try:
        draft_meta = fetch_player_draft_metadata()
        weekly_df = weekly_df.join(draft_meta, on="player_id", how="left")
    except Exception as e:
        print(f"Warning: Draft metadata merge skipped due to: {e}")

    weekly_path = os.path.join(DATA_DIR, "weekly_data_raw.csv")
    weekly_df.write_csv(weekly_path)
    print(f"Saved live weekly data to {weekly_path} ({weekly_df.height} rows)")

if __name__ == "__main__":
    main()
