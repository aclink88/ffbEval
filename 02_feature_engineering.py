"""
Feature Engineering Engine with Bayesian Shrinkage, Position-Specific Expected Points (xFP),
and Zero-Leakage Defense vs. Position (DvP) Matchup & Schedule-Adjusted Metrics.
Calculates position-specific Expected Fantasy Points and DvP adjustments for QBs, RBs, WRs, and TEs.
"""

import sys
import os
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

EMPIRICAL_M_CONSTANTS = {
    'QB': 1.8,
    'RB': 1.1,
    'WR': 1.4,
    'TE': 1.5
}

DEFENSIVE_PRIOR_GAMES_M = 3.0

def load_data() -> pl.DataFrame:
    raw_path = os.path.join(DATA_DIR, "weekly_data_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run 01_fetch_data.py first.")
    return pl.read_csv(raw_path, ignore_errors=True)

def clean_and_prep(df: pl.DataFrame) -> pl.DataFrame:
    """Filter skill positions and fill missing numeric values."""
    skill_positions = ['QB', 'RB', 'WR', 'TE']
    df = df.filter(pl.col('position').is_in(skill_positions))
    
    stat_cols = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_air_yards',
        'carries', 'rushing_yards', 'rushing_tds',
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'receiving_air_yards', 'receiving_yards_after_catch', 'fantasy_points_ppr'
    ]
    
    fill_exprs = [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in stat_cols if c in df.columns]
    df = df.with_columns(fill_exprs)
    df = df.with_columns([
        pl.col('season').cast(pl.Int64),
        pl.col('week').cast(pl.Int64)
    ])
    return df

def calculate_expected_points(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes Position-Specific Expected Fantasy Points (xFP):
    - QB: Pass Attempts (0.20 + 0.18 TD), Passing Air Yards (0.025), Rushing Carries (0.65 + 0.18 TD).
    - RB: Carries (0.65 + 0.18 TD), Targets (0.60 + 0.30 TD), Air Yards (0.08).
    - WR/TE: Targets (0.60 + 0.30 TD), Air Yards (0.08).
    """
    df = df.with_columns([
        # QB Passing Expected Points
        ((pl.col('attempts') * 0.38) + (pl.col('passing_air_yards') * 0.025)).alias('xfp_passing'),
        
        # Skill Receiving Expected Points (WR/TE/RB)
        (pl.col('targets') * 0.90 + pl.col('receiving_air_yards') * 0.08).alias('xfp_receiving'),
        
        # Rushing Expected Points (RB/QB)
        (pl.col('carries') * 0.83).alias('xfp_rushing'),
    ])
    
    # Combined Expected Points per Position
    df = df.with_columns(
        pl.when(pl.col('position') == 'QB')
          .then(pl.col('xfp_passing') + pl.col('xfp_rushing'))
          .otherwise(pl.col('xfp_receiving') + pl.col('xfp_rushing'))
          .alias('expected_fantasy_points_ppr')
    )
    
    df = df.with_columns(
        (pl.col('fantasy_points_ppr') - pl.col('expected_fantasy_points_ppr')).alias('actual_minus_expected')
    )
    return df

def calculate_dvp_and_schedule_adjustments(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes Zero-Leakage Defense vs. Position (DvP) profiles and joins both:
    1) Current Opponent DvP (to compute Schedule-Adjusted player metrics)
    2) Next Opponent DvP (to predict next week's matchup impact)
    """
    # 1. Aggregate total points and xFP allowed by each defense per (season, week, opponent_team, position)
    def_game = (
        df.filter(pl.col('opponent_team').is_not_null())
        .group_by(['season', 'week', 'opponent_team', 'position'])
        .agg([
            pl.col('fantasy_points_ppr').sum().alias('team_pos_actual_allowed'),
            pl.col('expected_fantasy_points_ppr').sum().alias('team_pos_xfp_allowed')
        ])
        .with_columns(
            (pl.col('team_pos_actual_allowed') - pl.col('team_pos_xfp_allowed')).alias('team_pos_eff_allowed')
        )
        .sort(['opponent_team', 'position', 'season', 'week'])
    )
    
    # 2. Compute league-wide positional averages per game as Bayesian priors
    def_game = def_game.with_columns([
        pl.col('team_pos_xfp_allowed').mean().over(['season', 'position']).alias('league_pos_xfp_prior'),
        pl.col('team_pos_eff_allowed').mean().over(['season', 'position']).alias('league_pos_eff_prior')
    ])
    
    # 3. Compute strictly lagged (pre-game) rolling 4-week defensive averages & sample counts within each season
    def_game = def_game.with_columns(
        pl.lit(1.0).alias('is_def_game')
    )
    def_game = def_game.with_columns([
        pl.col('team_pos_xfp_allowed').shift(1).rolling_mean(window_size=4, min_samples=1).over(['opponent_team', 'position', 'season']).alias('lag_roll4_xfp_allowed'),
        pl.col('team_pos_eff_allowed').shift(1).rolling_mean(window_size=4, min_samples=1).over(['opponent_team', 'position', 'season']).alias('lag_roll4_eff_allowed'),
        pl.col('is_def_game').shift(1).rolling_sum(window_size=4, min_samples=1).over(['opponent_team', 'position', 'season']).cast(pl.Float64).alias('lag_games_count')
    ])
    
    # Fill Week 1 nulls with 0 games count and league prior
    def_game = def_game.with_columns([
        pl.col('lag_games_count').fill_null(0.0),
        pl.col('lag_roll4_xfp_allowed').fill_null(pl.col('league_pos_xfp_prior')),
        pl.col('lag_roll4_eff_allowed').fill_null(pl.col('league_pos_eff_prior'))
    ])
    
    # 4. Apply Empirical Bayesian Shrinkage (m = 3.0 games) and compute differential vs. league average
    def_game = def_game.with_columns(
        (pl.col('lag_games_count') / (pl.col('lag_games_count') + DEFENSIVE_PRIOR_GAMES_M)).alias('def_credibility')
    )
    
    def_game = def_game.with_columns([
        (pl.col('def_credibility') * (pl.col('lag_roll4_xfp_allowed') - pl.col('league_pos_xfp_prior'))).alias('opp_dvp_xfp_diff'),
        (pl.col('def_credibility') * (pl.col('lag_roll4_eff_allowed') - pl.col('league_pos_eff_prior'))).alias('opp_dvp_eff_diff')
    ]).with_columns(
        (pl.col('opp_dvp_xfp_diff') + pl.col('opp_dvp_eff_diff')).alias('opp_dvp_total_diff')
    )
    
    # Also compute post-game (inclusive of Week t) shrunk DvP so Week t+1 (even if unplayed in live season) has updated DvP
    def_game = def_game.with_columns([
        pl.col('team_pos_xfp_allowed').rolling_mean(window_size=4, min_samples=1).over(['opponent_team', 'position', 'season']).alias('post_roll4_xfp_allowed'),
        pl.col('team_pos_eff_allowed').rolling_mean(window_size=4, min_samples=1).over(['opponent_team', 'position', 'season']).alias('post_roll4_eff_allowed'),
        pl.col('is_def_game').rolling_sum(window_size=4, min_samples=1).over(['opponent_team', 'position', 'season']).cast(pl.Float64).alias('post_games_count')
    ]).with_columns(
        (pl.col('post_games_count') / (pl.col('post_games_count') + DEFENSIVE_PRIOR_GAMES_M)).alias('post_def_credibility')
    ).with_columns([
        (pl.col('post_def_credibility') * (pl.col('post_roll4_xfp_allowed') - pl.col('league_pos_xfp_prior'))).alias('post_opp_dvp_xfp_diff'),
        (pl.col('post_def_credibility') * (pl.col('post_roll4_eff_allowed') - pl.col('league_pos_eff_prior'))).alias('post_opp_dvp_eff_diff')
    ]).with_columns(
        (pl.col('post_opp_dvp_xfp_diff') + pl.col('post_opp_dvp_eff_diff')).alias('post_opp_dvp_total_diff')
    )
    
    # 5. Join Current Week Opponent DvP onto player rows (strictly lagged entering Week t)
    curr_dvp_lookup = def_game.select([
        'season', 'week', 'opponent_team', 'position',
        pl.col('opp_dvp_xfp_diff').alias('curr_opp_dvp_xfp_diff'),
        pl.col('opp_dvp_eff_diff').alias('curr_opp_dvp_eff_diff'),
        pl.col('opp_dvp_total_diff').alias('curr_opp_dvp_total_diff')
    ])
    
    df = df.join(curr_dvp_lookup, on=['season', 'week', 'opponent_team', 'position'], how='left')
    df = df.with_columns([
        pl.col('curr_opp_dvp_xfp_diff').fill_null(0.0),
        pl.col('curr_opp_dvp_eff_diff').fill_null(0.0),
        pl.col('curr_opp_dvp_total_diff').fill_null(0.0)
    ])
    
    # 6. Determine each player's Next-Game Opponent (for training on target_next_week_ppr) and Next-Week Scheduled Opponent
    df = df.sort(['player_id', 'season', 'week'])
    df = df.with_columns(
        pl.col('opponent_team').shift(-1).over(['player_id', 'season']).alias('next_game_opponent')
    )
    if 'next_opponent_team' in df.columns:
        df = df.with_columns(
            pl.when(pl.col('next_game_opponent').is_not_null())
              .then(pl.col('next_game_opponent'))
              .otherwise(pl.col('next_opponent_team'))
              .alias('effective_next_opponent')
        )
    else:
        df = df.with_columns(pl.col('next_game_opponent').alias('effective_next_opponent'))
        
    # Build latest available defensive profile per (season, week, defense, position) for upcoming opponent lookup
    # At the end of week `t`, a defense's latest profile entering their next game is their latest `post_opp_dvp_*` as of week `t`
    latest_def_entering_next = (
        def_game.select([
            'season', 'week',
            pl.col('opponent_team').alias('effective_next_opponent'),
            'position',
            pl.col('post_opp_dvp_xfp_diff').alias('next_opp_dvp_xfp_diff'),
            pl.col('post_opp_dvp_eff_diff').alias('next_opp_dvp_eff_diff'),
            pl.col('post_opp_dvp_total_diff').alias('next_opp_dvp_total_diff')
        ])
    )
    
    df = df.join(
        latest_def_entering_next,
        on=['season', 'week', 'effective_next_opponent', 'position'],
        how='left'
    )
    
    # If the upcoming opponent was on a bye in week `t`, fall back to their lagged profile from week `t-1`
    fallback_def = (
        def_game.with_columns((pl.col('week') + 1).alias('week'))
        .select([
            'season', 'week',
            pl.col('opponent_team').alias('effective_next_opponent'),
            'position',
            pl.col('post_opp_dvp_xfp_diff').alias('fb_next_opp_dvp_xfp_diff'),
            pl.col('post_opp_dvp_eff_diff').alias('fb_next_opp_dvp_eff_diff'),
            pl.col('post_opp_dvp_total_diff').alias('fb_next_opp_dvp_total_diff')
        ])
    )
    df = df.join(
        fallback_def,
        on=['season', 'week', 'effective_next_opponent', 'position'],
        how='left'
    )
    
    df = df.with_columns([
        pl.coalesce(['next_opp_dvp_xfp_diff', 'fb_next_opp_dvp_xfp_diff', pl.lit(0.0)]).alias('next_opp_dvp_xfp_diff'),
        pl.coalesce(['next_opp_dvp_eff_diff', 'fb_next_opp_dvp_eff_diff', pl.lit(0.0)]).alias('next_opp_dvp_eff_diff'),
        pl.coalesce(['next_opp_dvp_total_diff', 'fb_next_opp_dvp_total_diff', pl.lit(0.0)]).alias('next_opp_dvp_total_diff')
    ]).drop(['fb_next_opp_dvp_xfp_diff', 'fb_next_opp_dvp_eff_diff', 'fb_next_opp_dvp_total_diff', 'next_game_opponent'])
    
    # 7. Compute Schedule-Adjusted Single-Week Player Metrics
    # Note: Team-level defensive diff is scaled by positional share (~0.45 for RB/WR, ~0.85 for QB, ~0.65 for TE)
    # so we don't over-credit a backup player for the entire team's positional DvP differential.
    df = df.with_columns(
        pl.when(pl.col('position') == 'QB').then(0.85)
          .when(pl.col('position') == 'TE').then(0.65)
          .otherwise(0.45)
          .alias('pos_share_scalar')
    )
    
    df = df.with_columns([
        (pl.col('fantasy_points_ppr') - (pl.col('curr_opp_dvp_total_diff') * pl.col('pos_share_scalar'))).alias('sched_adj_actual_ppr'),
        (pl.col('expected_fantasy_points_ppr') - (pl.col('curr_opp_dvp_xfp_diff') * pl.col('pos_share_scalar'))).alias('sched_adj_xfp'),
        (pl.col('next_opp_dvp_xfp_diff') * pl.col('pos_share_scalar')).alias('next_opp_dvp_xfp_adj'),
        (pl.col('next_opp_dvp_eff_diff') * pl.col('pos_share_scalar')).alias('next_opp_dvp_eff_adj'),
        (pl.col('next_opp_dvp_total_diff') * pl.col('pos_share_scalar')).alias('next_opp_dvp_total_adj')
    ]).drop('pos_share_scalar')
    
    return df

def calculate_bayesian_rolling_metrics(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(['player_id', 'season', 'week'])
    
    df = df.with_columns(
        pl.when(pl.col('fantasy_points_ppr') > 0.0).then(1.0).otherwise(0.0).alias('is_active_week')
    )
    
    df = df.with_columns([
        pl.col('is_active_week').rolling_sum(window_size=3, min_samples=1).over(['player_id', 'season']).alias('active_weeks_count'),
        pl.col('fantasy_points_ppr').rolling_mean(window_size=3, min_samples=1).over(['player_id', 'season']).alias('roll3_actual_ppr'),
        pl.col('expected_fantasy_points_ppr').rolling_mean(window_size=3, min_samples=1).over(['player_id', 'season']).alias('roll3_expected_ppr'),
        pl.col('sched_adj_actual_ppr').rolling_mean(window_size=3, min_samples=1).over(['player_id', 'season']).alias('roll3_sched_adj_actual_ppr'),
        pl.col('sched_adj_xfp').rolling_mean(window_size=3, min_samples=1).over(['player_id', 'season']).alias('roll3_sched_adj_xfp'),
        pl.col('targets').rolling_mean(window_size=3, min_samples=1).over(['player_id', 'season']).alias('roll3_targets'),
        pl.col('carries').rolling_mean(window_size=3, min_samples=1).over(['player_id', 'season']).alias('roll3_carries')
    ])
    
    # Bayesian Credibility Weight calculation
    df = df.with_columns(
        pl.when(pl.col('position') == 'QB').then(pl.col('active_weeks_count') / (pl.col('active_weeks_count') + EMPIRICAL_M_CONSTANTS['QB']))
          .when(pl.col('position') == 'RB').then(pl.col('active_weeks_count') / (pl.col('active_weeks_count') + EMPIRICAL_M_CONSTANTS['RB']))
          .when(pl.col('position') == 'WR').then(pl.col('active_weeks_count') / (pl.col('active_weeks_count') + EMPIRICAL_M_CONSTANTS['WR']))
          .otherwise(pl.col('active_weeks_count') / (pl.col('active_weeks_count') + EMPIRICAL_M_CONSTANTS['TE']))
          .alias('bayesian_credibility_weight')
    )
    
    df = df.with_columns(
        pl.col('fantasy_points_ppr').mean().over(['position']).alias('position_prior_mean')
    )
    
    df = df.with_columns(
        (pl.col('bayesian_credibility_weight') * pl.col('roll3_actual_ppr') + 
        (1.0 - pl.col('bayesian_credibility_weight')) * pl.col('position_prior_mean')).alias('bayesian_shrunk_ppr')
    )
    
    df = df.with_columns(
        (pl.col('roll3_actual_ppr') - pl.col('roll3_expected_ppr')).alias('efficiency_delta_roll3')
    )
    return df

def main():
    print("Loading raw weekly data into Polars...")
    df = load_data()
    
    print("Cleaning and filtering skill positions...")
    df = clean_and_prep(df)
    
    print("Calculating Corrected Position-Specific Expected Points (xFP) in Polars...")
    df = calculate_expected_points(df)
    
    print("Computing Zero-Leakage Defense vs. Position (DvP) & Schedule-Adjusted Metrics...")
    df = calculate_dvp_and_schedule_adjustments(df)
    
    print("Applying Empirical Bayesian Rate Shrinkage & Rolling Metrics...")
    df = calculate_bayesian_rolling_metrics(df)
    
    output_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    df.write_csv(output_path)
    print(f"Featured dataset successfully saved to {output_path} ({df.height} rows)")

if __name__ == "__main__":
    main()
