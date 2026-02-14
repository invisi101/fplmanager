"""Build the feature matrix: one row per player per gameweek."""

import numpy as np
import pandas as pd


# --- Player rolling stat columns from playermatchstats ---
PLAYER_ROLLING_COLS = [
    "xg", "xa", "xgot", "total_shots", "shots_on_target",
    "touches_opposition_box", "chances_created", "successful_dribbles",
    "accurate_crosses", "tackles_won", "interceptions", "recoveries",
    "clearances", "minutes_played", "goals", "assists",
    "big_chances_missed", "accurate_passes", "final_third_passes",
    "blocks", "aerial_duels_won", "saves",
]
PLAYER_ROLLING_WINDOWS = [3, 5]

OPPONENT_ROLLING_WINDOWS = [3, 5]


def _add_gameweek_to_pms(pms: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Add gameweek column to playermatchstats by joining on match_id."""
    if "gameweek" in pms.columns:
        return pms
    match_gw = matches[["match_id", "gameweek"]].drop_duplicates()
    return pms.merge(match_gw, on="match_id", how="left")


def _build_player_rolling_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages of player stats over recent gameweeks.

    Returns a DataFrame with player_id, gameweek, and rolling features.
    Each row represents the rolling stats *as of* that gameweek (using data
    from prior gameweeks only — no data leakage).
    """
    # Ensure numeric and fill NaN with 0
    for col in PLAYER_ROLLING_COLS:
        if col in pms.columns:
            pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

    available_cols = [c for c in PLAYER_ROLLING_COLS if c in pms.columns]

    # Aggregate per player per gameweek (handles double GWs)
    agg = pms.groupby(["player_id", "gameweek"])[available_cols].mean().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    # Compute rolling averages (shift by 1 to avoid leakage — only past data)
    result_frames = [agg[["player_id", "gameweek"]]]
    for window in PLAYER_ROLLING_WINDOWS:
        for col in available_cols:
            feat_name = f"player_{col}_last{window}"
            rolled = (
                agg.groupby("player_id")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    # BPS from playerstats gets added separately
    return pd.concat(result_frames, axis=1)


def _build_ewm_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Compute exponentially weighted means of key raw stats.

    Applies shift(1) + ewm(span=5) directly on per-GW aggregated raw stats
    so the EWM operates on actual match data rather than already-smoothed
    rolling averages (which would double-smooth the signal).

    Returns DataFrame with player_id, gameweek, and ewm features.
    """
    raw_cols = ["xg", "xa", "xgot", "chances_created", "shots_on_target"]
    available = [c for c in raw_cols if c in pms.columns]
    if not available:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # Aggregate per player per GW (handles DGW)
    agg = pms.groupby(["player_id", "gameweek"])[available].mean().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    result = agg[["player_id", "gameweek"]].copy()
    for col in available:
        # Map raw stat name to the feature name used in DEFAULT_FEATURES
        feat_name = f"ewm_player_{col}_last3"
        result[feat_name] = (
            agg.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
        )

    return result


def _build_upside_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Compute features that capture explosive/upside potential.

    - xg_volatility_last5: std of xG over last 5 matches (high = explosive)
    - form_acceleration: xG last 3 minus xG last 5 (positive = upward trend)
    - big_chance_frequency_last5: rolling mean of (goals + big_chances_missed)

    Returns DataFrame with player_id, gameweek, and upside features.
    """
    needed = ["xg", "goals", "big_chances_missed"]
    available = [c for c in needed if c in pms.columns]
    if "xg" not in available:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    for col in available:
        pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

    # Aggregate per player per GW (handles DGW)
    agg = pms.groupby(["player_id", "gameweek"])[available].mean().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    result = agg[["player_id", "gameweek"]].copy()

    # xG volatility: std of last 5 shifted xG values
    result["xg_volatility_last5"] = (
        agg.groupby("player_id")["xg"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    )

    # Form acceleration: difference between short and long rolling xG
    xg_last3 = agg.groupby("player_id")["xg"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    xg_last5 = agg.groupby("player_id")["xg"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    result["form_acceleration"] = xg_last3 - xg_last5

    # Big chance frequency: rolling mean of (goals + big_chances_missed)
    if "goals" in available and "big_chances_missed" in available:
        agg["_big_chance_total"] = agg["goals"] + agg["big_chances_missed"]
        result["big_chance_frequency_last5"] = (
            agg.groupby("player_id")["_big_chance_total"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    return result


def _build_team_match_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert matches into per-team-per-gameweek rows with defensive/offensive stats.

    For each match, creates two rows: one for the home team, one for the away team.
    """
    rows = []
    for _, m in matches.iterrows():
        gw = m.get("gameweek")
        if pd.isna(gw):
            continue
        gw = int(gw)
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.isna(home) or pd.isna(away):
            continue

        # Home team perspective
        rows.append({
            "team_code": int(home), "gameweek": gw, "is_home": True,
            "goals_scored": _safe(m, "home_score"),
            "goals_conceded": _safe(m, "away_score"),
            "xg": _safe(m, "home_expected_goals_xg"),
            "xg_conceded": _safe(m, "away_expected_goals_xg"),
            "big_chances": _safe(m, "home_big_chances"),
            "big_chances_allowed": _safe(m, "away_big_chances"),
            "shots_inside_box": _safe(m, "home_shots_inside_box"),
            "shots_inside_box_allowed": _safe(m, "away_shots_inside_box"),
            "shots_on_target": _safe(m, "home_shots_on_target"),
            "accurate_crosses": _safe(m, "home_accurate_crosses"),
            "accurate_crosses_allowed": _safe(m, "away_accurate_crosses"),
            "clean_sheet": 1 if _safe(m, "away_score") == 0 else 0,
            "opponent_xg": _safe(m, "away_expected_goals_xg"),
            "opponent_big_chances": _safe(m, "away_big_chances"),
            "opponent_shots_on_target": _safe(m, "away_shots_on_target"),
        })
        # Away team perspective
        rows.append({
            "team_code": int(away), "gameweek": gw, "is_home": False,
            "goals_scored": _safe(m, "away_score"),
            "goals_conceded": _safe(m, "home_score"),
            "xg": _safe(m, "away_expected_goals_xg"),
            "xg_conceded": _safe(m, "home_expected_goals_xg"),
            "big_chances": _safe(m, "away_big_chances"),
            "big_chances_allowed": _safe(m, "home_big_chances"),
            "shots_inside_box": _safe(m, "away_shots_inside_box"),
            "shots_inside_box_allowed": _safe(m, "home_shots_inside_box"),
            "shots_on_target": _safe(m, "away_shots_on_target"),
            "accurate_crosses": _safe(m, "away_accurate_crosses"),
            "accurate_crosses_allowed": _safe(m, "home_accurate_crosses"),
            "clean_sheet": 1 if _safe(m, "home_score") == 0 else 0,
            "opponent_xg": _safe(m, "home_expected_goals_xg"),
            "opponent_big_chances": _safe(m, "home_big_chances"),
            "opponent_shots_on_target": _safe(m, "home_shots_on_target"),
        })

    return pd.DataFrame(rows)


def _safe(row, col, default=0):
    val = row.get(col, default)
    return default if pd.isna(val) else val


def _build_opponent_rolling_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling team stats for opponent difficulty assessment.

    Returns per-team-per-gameweek rolling stats using only past data.
    """
    team_stats = team_stats.sort_values(["team_code", "gameweek"])

    roll_cols = [
        "goals_conceded", "xg_conceded", "big_chances_allowed",
        "shots_inside_box_allowed", "accurate_crosses_allowed",
        "clean_sheet", "opponent_xg", "opponent_big_chances",
        "opponent_shots_on_target",
    ]

    # Aggregate per team per GW (in case of double GWs)
    agg = team_stats.groupby(["team_code", "gameweek"])[roll_cols].mean().reset_index()
    agg = agg.sort_values(["team_code", "gameweek"])

    result_frames = [agg[["team_code", "gameweek"]]]
    for window in OPPONENT_ROLLING_WINDOWS:
        for col in roll_cols:
            feat_name = f"opp_{col}_last{window}"
            rolled = (
                agg.groupby("team_code")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    return pd.concat(result_frames, axis=1)


def _build_own_team_rolling_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling own-team attacking/defensive stats.

    Returns per-team-per-gameweek rolling stats using only past data.
    """
    team_stats = team_stats.sort_values(["team_code", "gameweek"])
    own_cols = ["goals_scored", "xg", "big_chances", "shots_on_target", "clean_sheet"]
    available = [c for c in own_cols if c in team_stats.columns]

    agg = team_stats.groupby(["team_code", "gameweek"])[available].mean().reset_index()
    agg = agg.sort_values(["team_code", "gameweek"])

    result_frames = [agg[["team_code", "gameweek"]]]
    for window in [3, 5]:
        for col in available:
            feat_name = f"team_{col}_last{window}"
            rolled = (
                agg.groupby("team_code")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            result_frames.append(rolled.rename(feat_name))

    return pd.concat(result_frames, axis=1)


def _build_opponent_history_features(pms: pd.DataFrame, matches: pd.DataFrame,
                                     players: pd.DataFrame) -> pd.DataFrame:
    """Compute player's historical performance vs each specific opponent.

    Uses expanding mean (not rolling) since matches vs a specific opponent
    are rare (2-4 per 2 seasons). shift(1) prevents leakage.

    Returns DataFrame with player_id, gameweek, opponent_code,
    vs_opponent_xg_avg, vs_opponent_goals_avg, vs_opponent_matches.
    """
    if pms.empty or matches.empty:
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    if not all(c in pms.columns for c in ["player_id", "match_id", "gameweek"]):
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    # Build match -> (team -> opponent) mapping
    match_opponents = []
    for _, m in matches.iterrows():
        mid = m.get("match_id")
        if pd.isna(mid):
            continue
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.notna(home) and pd.notna(away):
            match_opponents.append({"match_id": mid, "team_code": int(home), "opponent_code": int(away)})
            match_opponents.append({"match_id": mid, "team_code": int(away), "opponent_code": int(home)})

    if not match_opponents:
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    opp_df = pd.DataFrame(match_opponents)

    pm = pms[["player_id", "match_id", "gameweek"]].copy()
    for col in ["xg", "goals"]:
        if col in pms.columns:
            pm[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)
        else:
            pm[col] = 0

    # Get player's team from players table
    if not players.empty and "player_id" in players.columns and "team_code" in players.columns:
        player_team = players[["player_id", "team_code"]].drop_duplicates(subset=["player_id"])
        pm = pm.merge(player_team, on="player_id", how="left")
    else:
        return pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

    pm = pm.dropna(subset=["team_code"])
    pm["team_code"] = pm["team_code"].astype(int)

    # Merge to get opponent per match
    pm = pm.merge(opp_df, on=["match_id", "team_code"], how="left")
    pm = pm.dropna(subset=["opponent_code"])
    pm["opponent_code"] = pm["opponent_code"].astype(int)
    pm = pm.sort_values(["player_id", "opponent_code", "gameweek"])

    # Expanding mean EXCLUDING current match per player-opponent pair.
    # shift(1) ensures GW N features only include data from GW N-1 and before,
    # consistent with all other rolling features in the file.
    pm["vs_opponent_xg_avg"] = (
        pm.groupby(["player_id", "opponent_code"])["xg"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    pm["vs_opponent_goals_avg"] = (
        pm.groupby(["player_id", "opponent_code"])["goals"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    # Total matches played against this opponent (excluding current)
    pm["vs_opponent_matches"] = (
        pm.groupby(["player_id", "opponent_code"]).cumcount()
    )

    result = pm[["player_id", "gameweek", "opponent_code",
                 "vs_opponent_xg_avg", "vs_opponent_goals_avg",
                 "vs_opponent_matches"]].copy()

    return result


def _build_home_away_form(pms: pd.DataFrame, matches: pd.DataFrame,
                          players: pd.DataFrame) -> pd.DataFrame:
    """Compute separate home/away rolling xG form per player.

    Joins PMS with matches to determine venue, then computes rolling 5-match
    xG averages separately for home and away appearances. Forward-fills each
    so that rows always have a value. Creates venue_matched_form that picks
    the appropriate one based on next fixture's is_home.

    Returns DataFrame with player_id, gameweek, home_xg_form, away_xg_form.
    (venue_matched_form is built during assembly when is_home is available.)
    """
    if pms.empty or matches.empty:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    if "xg" not in pms.columns or "match_id" not in pms.columns:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # Get venue (is_home) for each match from the match data
    match_venue = []
    for _, m in matches.iterrows():
        mid = m.get("match_id")
        if pd.isna(mid):
            continue
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.notna(home):
            match_venue.append({"match_id": mid, "team_code": int(home), "is_home_venue": 1})
        if pd.notna(away):
            match_venue.append({"match_id": mid, "team_code": int(away), "is_home_venue": 0})

    if not match_venue:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    venue_df = pd.DataFrame(match_venue)

    # Merge PMS with venue info — need player's team to determine venue
    pm = pms[["player_id", "match_id", "gameweek", "xg"]].copy()
    pm["xg"] = pd.to_numeric(pm["xg"], errors="coerce").fillna(0)

    # Get player's team from players table
    if not players.empty and "player_id" in players.columns and "team_code" in players.columns:
        player_team = players[["player_id", "team_code"]].drop_duplicates(subset=["player_id"])
        pm = pm.merge(player_team, on="player_id", how="left")
    else:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    pm = pm.dropna(subset=["team_code"])
    pm["team_code"] = pm["team_code"].astype(int)

    pm = pm.merge(venue_df, on=["match_id", "team_code"], how="left")
    pm = pm.dropna(subset=["is_home_venue"])

    # Aggregate per player per GW per venue
    agg = pm.groupby(["player_id", "gameweek", "is_home_venue"])["xg"].mean().reset_index()
    agg = agg.sort_values(["player_id", "gameweek"])

    # Compute rolling xG form for home games
    home = agg[agg["is_home_venue"] == 1].copy()
    home["home_xg_form"] = (
        home.groupby("player_id")["xg"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Compute rolling xG form for away games
    away = agg[agg["is_home_venue"] == 0].copy()
    away["away_xg_form"] = (
        away.groupby("player_id")["xg"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Merge home/away forms back per player-GW and forward-fill
    all_pgw = pm[["player_id", "gameweek"]].drop_duplicates().sort_values(["player_id", "gameweek"])
    all_pgw = all_pgw.merge(
        home[["player_id", "gameweek", "home_xg_form"]],
        on=["player_id", "gameweek"], how="left",
    )
    all_pgw = all_pgw.merge(
        away[["player_id", "gameweek", "away_xg_form"]],
        on=["player_id", "gameweek"], how="left",
    )
    all_pgw = all_pgw.sort_values(["player_id", "gameweek"])
    all_pgw["home_xg_form"] = all_pgw.groupby("player_id")["home_xg_form"].ffill()
    all_pgw["away_xg_form"] = all_pgw.groupby("player_id")["away_xg_form"].ffill()

    return all_pgw[["player_id", "gameweek", "home_xg_form", "away_xg_form"]]


def _build_rest_days_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute days rest and fixture congestion per team per gameweek.

    Uses kickoff_time from matches. days_rest is shifted so the value at
    GW N reflects rest before GW N (not after). fixture_congestion is the
    inverse of rolling 3-match average rest.

    Returns DataFrame with team_code, gameweek, days_rest, fixture_congestion.
    """
    if "kickoff_time" not in matches.columns:
        return pd.DataFrame(columns=["team_code", "gameweek"])

    m = matches.copy()
    m["kickoff_dt"] = pd.to_datetime(m["kickoff_time"], errors="coerce")
    m = m.dropna(subset=["kickoff_dt", "gameweek"])

    # Build per-team rows from both home and away perspectives
    rows = []
    for _, row in m.iterrows():
        gw = int(row["gameweek"])
        dt = row["kickoff_dt"]
        home = row.get("home_team")
        away = row.get("away_team")
        if pd.notna(home):
            rows.append({"team_code": int(home), "gameweek": gw, "kickoff_dt": dt})
        if pd.notna(away):
            rows.append({"team_code": int(away), "gameweek": gw, "kickoff_dt": dt})

    if not rows:
        return pd.DataFrame(columns=["team_code", "gameweek"])

    team_matches = pd.DataFrame(rows)
    # For DGWs, take the latest kickoff per team per GW
    team_matches = (
        team_matches.groupby(["team_code", "gameweek"])["kickoff_dt"]
        .max()
        .reset_index()
        .sort_values(["team_code", "kickoff_dt"])
    )

    # Days since previous match (shifted: value at GW N = rest before GW N)
    team_matches["days_rest"] = (
        team_matches.groupby("team_code")["kickoff_dt"]
        .diff()
        .dt.total_seconds() / 86400.0
    )
    team_matches["days_rest"] = team_matches["days_rest"].fillna(7.0)

    # Fixture congestion = inverse of rolling 3-match average rest
    team_matches["fixture_congestion"] = (
        team_matches.groupby("team_code")["days_rest"]
        .transform(lambda s: 1.0 / s.rolling(3, min_periods=1).mean())
    )

    return team_matches[["team_code", "gameweek", "days_rest", "fixture_congestion"]]


def _build_fixture_map(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping of team_code -> gameweek -> opponent_code, is_home."""
    rows = []
    for _, m in matches.iterrows():
        gw = m.get("gameweek")
        if pd.isna(gw):
            continue
        gw = int(gw)
        home = m.get("home_team")
        away = m.get("away_team")
        if pd.isna(home) or pd.isna(away):
            continue
        rows.append({"team_code": int(home), "gameweek": gw, "opponent_code": int(away), "is_home": 1})
        rows.append({"team_code": int(away), "gameweek": gw, "opponent_code": int(home), "is_home": 0})
    return pd.DataFrame(rows)


def _build_playerstats_features(playerstats: pd.DataFrame) -> pd.DataFrame:
    """Extract per-player-per-GW features from the FPL playerstats snapshot."""
    feature_cols = {
        "event_points": "event_points",
        "form": "player_form",
        "bonus": "player_bonus",
        "bps": "player_bps",
        "ep_next": "ep_next",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "ict_index": "ict_index",
        "now_cost": "cost",
        "chance_of_playing_next_round": "chance_of_playing",
        "selected_by_percent": "ownership",
        "minutes": "cumulative_minutes",
        "clean_sheets_per_90": "clean_sheets_per_90",
        "starts_per_90": "starts_per_90",
        "yellow_cards": "yellow_cards",
        "transfers_in_event": "transfers_in_event",
        "transfers_out_event": "transfers_out_event",
        "expected_goals_conceded_per_90": "xgc_per_90",
        "saves_per_90": "saves_per_90",
        "total_points": "total_points",
    }

    # Set piece involvement
    set_piece_cols = {
        "penalties_order": "penalties_order",
        "corners_and_indirect_freekicks_order": "corners_order",
        "direct_freekicks_order": "freekicks_order",
    }

    available = {}
    for src, dst in {**feature_cols, **set_piece_cols}.items():
        if src in playerstats.columns:
            available[src] = dst

    result = playerstats[["id", "gw"]].copy()
    result = result.rename(columns={"id": "player_id", "gw": "gameweek"})

    # Carry web_name through if available
    if "web_name" in playerstats.columns:
        result["web_name"] = playerstats["web_name"]

    for src, dst in available.items():
        result[dst] = pd.to_numeric(playerstats[src], errors="coerce")

    # Players without injury flags are fully fit — NaN means 100%
    if "chance_of_playing" in result.columns:
        result["chance_of_playing"] = result["chance_of_playing"].fillna(100)

    # Set piece involvement flag (1 if any set piece role <= 2)
    sp_cols = [v for k, v in set_piece_cols.items() if k in playerstats.columns]
    if sp_cols:
        for c in sp_cols:
            result[c] = result[c].fillna(99)
        result["set_piece_involvement"] = (result[sp_cols].min(axis=1) <= 2).astype(int)

    # Convert cumulative season totals to per-GW deltas
    for cum_col in ["influence", "creativity", "threat", "ict_index", "player_bps"]:
        if cum_col in result.columns:
            result = result.sort_values(["player_id", "gameweek"])
            result[f"gw_{cum_col}"] = result.groupby("player_id")[cum_col].diff()
            first_mask = result[f"gw_{cum_col}"].isna()
            result.loc[first_mask, f"gw_{cum_col}"] = result.loc[first_mask, cum_col]
            result[f"gw_{cum_col}"] = result[f"gw_{cum_col}"].clip(lower=0)

    # Availability consistency: fraction of recent GWs where player featured
    if "cumulative_minutes" in result.columns:
        result = result.sort_values(["player_id", "gameweek"])
        gw_mins = result.groupby("player_id")["cumulative_minutes"].diff()
        first = gw_mins.isna()
        gw_mins[first] = result.loc[first, "cumulative_minutes"]
        result["availability_rate_last5"] = (
            (gw_mins > 0).astype(float)
            .groupby(result["player_id"])
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    # Net transfers and transfer momentum
    if "transfers_in_event" in result.columns and "transfers_out_event" in result.columns:
        result["net_transfers"] = (
            result["transfers_in_event"].fillna(0) - result["transfers_out_event"].fillna(0)
        )
        result = result.sort_values(["player_id", "gameweek"])
        result["transfer_momentum"] = (
            result.groupby("player_id")["net_transfers"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

    return result


def _build_targets(playerstats: pd.DataFrame) -> pd.DataFrame:
    """Build target variables: next_gw_points and next_3gw_points.

    Uses explicit gameweek-number lookup instead of row-shift so that
    gaps in a player's gameweek sequence (missed GWs, blanks) don't
    misalign the target.
    """
    ps = playerstats[["id", "gw", "event_points"]].copy()
    ps = ps.rename(columns={"id": "player_id", "gw": "gameweek"})
    ps["event_points"] = pd.to_numeric(ps["event_points"], errors="coerce").fillna(0)
    ps = ps.sort_values(["player_id", "gameweek"])

    # Build a lookup: (player_id, gameweek) -> event_points
    pts_lookup = ps.set_index(["player_id", "gameweek"])["event_points"]

    # next_gw_points: points scored in gameweek + 1
    ps["next_gw_points"] = ps.apply(
        lambda r: pts_lookup.get((r["player_id"], r["gameweek"] + 1)), axis=1
    )

    # next_3gw_points: sum of points in gameweek+1, +2, +3
    for offset in [1, 2, 3]:
        ps[f"pts_gw_plus{offset}"] = ps.apply(
            lambda r, o=offset: pts_lookup.get((r["player_id"], r["gameweek"] + o)),
            axis=1,
        )
    shift_cols = ["pts_gw_plus1", "pts_gw_plus2", "pts_gw_plus3"]
    # Only valid if all 3 future GWs exist for this player
    ps["next_3gw_points"] = ps[shift_cols].sum(axis=1)
    ps.loc[ps[shift_cols].isna().any(axis=1), "next_3gw_points"] = np.nan
    ps = ps.drop(columns=shift_cols)

    return ps[["player_id", "gameweek", "next_gw_points", "next_3gw_points"]]


def _build_decomposed_targets(
    pms: pd.DataFrame, playerstats: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-component targets for decomposed scoring models.

    Creates next-GW targets for each FPL point source from PMS (per-match data),
    playerstats (cumulative data), and matches (team results for clean sheets).
    Targets are aligned so that a row at gameweek=N contains what happens in
    gameweek N+1.

    Returns DataFrame with player_id, gameweek, and component targets:
      next_gw_minutes, next_gw_goals, next_gw_assists, next_gw_cs,
      next_gw_bonus, next_gw_goals_conceded, next_gw_saves
    """
    # --- Per-GW stats from PMS (available both seasons) ---
    pms_cols = {
        "goals": "gw_goals",
        "assists": "gw_assists",
        "minutes_played": "gw_minutes",
        "saves": "gw_saves",
    }
    available_pms = {k: v for k, v in pms_cols.items() if k in pms.columns}

    if available_pms and "gameweek" in pms.columns:
        for col in available_pms:
            pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

        # Aggregate per player per GW (sum across matches for DGW)
        pms_agg = pms.groupby(["player_id", "gameweek"])[
            list(available_pms.keys())
        ].sum().reset_index()
        pms_agg = pms_agg.rename(columns=available_pms)

        # --- Clean sheet: derive from match results + player minutes ---
        # PMS goals_conceded is only populated for GKP, so instead we check
        # whether the player's team conceded 0 goals in each match and the
        # player played 60+ minutes.
        if (
            matches is not None
            and not matches.empty
            and players is not None
            and not players.empty
            and "minutes_played" in pms.columns
            and "match_id" in pms.columns
        ):
            # Build per-match team goals conceded from match results
            match_gc_rows = []
            for _, m in matches.iterrows():
                mid = m.get("match_id")
                if pd.isna(mid):
                    continue
                home_team = m.get("home_team")
                away_team = m.get("away_team")
                home_score = m.get("home_score", 0)
                away_score = m.get("away_score", 0)
                if pd.notna(home_team) and pd.notna(away_score):
                    match_gc_rows.append(
                        {"match_id": mid, "team_code": int(home_team),
                         "team_goals_conceded": int(away_score)}
                    )
                if pd.notna(away_team) and pd.notna(home_score):
                    match_gc_rows.append(
                        {"match_id": mid, "team_code": int(away_team),
                         "team_goals_conceded": int(home_score)}
                    )

            if match_gc_rows:
                match_gc = pd.DataFrame(match_gc_rows)
                # Map player_id -> team_code
                pid_to_team = dict(zip(players["player_id"], players["team_code"]))
                pms_cs = pms[["player_id", "match_id", "gameweek", "minutes_played"]].copy()
                pms_cs["team_code"] = pms_cs["player_id"].map(pid_to_team)
                pms_cs = pms_cs.merge(match_gc, on=["match_id", "team_code"], how="left")
                pms_cs["team_goals_conceded"] = pms_cs["team_goals_conceded"].fillna(0)
                pms_cs["_match_cs"] = (
                    (pms_cs["team_goals_conceded"] == 0)
                    & (pms_cs["minutes_played"] >= 60)
                ).astype(int)
                cs_agg = pms_cs.groupby(["player_id", "gameweek"]).agg(
                    gw_cs=("_match_cs", "sum"),
                    gw_goals_conceded=("team_goals_conceded", "sum"),
                ).reset_index()
                pms_agg = pms_agg.merge(cs_agg, on=["player_id", "gameweek"], how="left")
                pms_agg["gw_cs"] = pms_agg["gw_cs"].fillna(0).astype(int)
                pms_agg["gw_goals_conceded"] = pms_agg["gw_goals_conceded"].fillna(0)
    else:
        pms_agg = pd.DataFrame(columns=["player_id", "gameweek"])

    # --- Bonus from playerstats cumulative diff (available both seasons) ---
    if "bonus" in playerstats.columns:
        ps_bonus = playerstats[["id", "gw", "bonus"]].copy()
        ps_bonus = ps_bonus.rename(columns={"id": "player_id", "gw": "gameweek"})
        ps_bonus["bonus"] = pd.to_numeric(ps_bonus["bonus"], errors="coerce").fillna(0)
        ps_bonus = ps_bonus.sort_values(["player_id", "gameweek"])
        ps_bonus["gw_bonus"] = ps_bonus.groupby("player_id")["bonus"].diff()
        # First GW diff is NaN — use the raw value (it IS the per-GW value for GW1)
        first_mask = ps_bonus["gw_bonus"].isna()
        ps_bonus.loc[first_mask, "gw_bonus"] = ps_bonus.loc[first_mask, "bonus"]
        ps_bonus["gw_bonus"] = ps_bonus["gw_bonus"].clip(lower=0)

        if not pms_agg.empty:
            pms_agg = pms_agg.merge(
                ps_bonus[["player_id", "gameweek", "gw_bonus"]],
                on=["player_id", "gameweek"], how="outer",
            )
        else:
            pms_agg = ps_bonus[["player_id", "gameweek", "gw_bonus"]].copy()

    if pms_agg.empty:
        return pd.DataFrame(columns=["player_id", "gameweek"])

    # --- Expand to full player-GW universe from playerstats ---
    # playerstats includes ALL players each GW (including non-starters with
    # event_points=0).  We need zeros for players who didn't play so that the
    # minutes model can learn to predict 0 for non-starters.
    all_player_gws = playerstats[["id", "gw"]].copy()
    all_player_gws = all_player_gws.rename(columns={"id": "player_id", "gw": "gameweek"})
    all_player_gws = all_player_gws.drop_duplicates(subset=["player_id", "gameweek"])

    pms_agg = all_player_gws.merge(pms_agg, on=["player_id", "gameweek"], how="left")
    gw_cols = [c for c in pms_agg.columns if c.startswith("gw_")]
    for col in gw_cols:
        pms_agg[col] = pms_agg[col].fillna(0)

    pms_agg = pms_agg.sort_values(["player_id", "gameweek"])

    # --- Build next-GW targets via lookup (same approach as _build_targets) ---
    for col in gw_cols:
        target_name = col.replace("gw_", "next_gw_")
        lookup = pms_agg.set_index(["player_id", "gameweek"])[col]
        pms_agg[target_name] = pms_agg.apply(
            lambda r, lu=lookup, tn=target_name: lu.get(
                (r["player_id"], r["gameweek"] + 1)
            ),
            axis=1,
        )

    target_cols = [c for c in pms_agg.columns if c.startswith("next_gw_")]
    return pms_agg[["player_id", "gameweek"] + target_cols]


def _build_elo_features(teams: pd.DataFrame) -> pd.DataFrame:
    """Extract Elo ratings per team."""
    cols = ["code", "elo"]
    available = [c for c in cols if c in teams.columns]
    if "elo" not in available:
        return pd.DataFrame()
    result = teams[available].copy()
    result = result.rename(columns={"code": "team_code", "elo": "team_elo"})
    return result


def _build_fdr_map(api_fixtures: list) -> pd.DataFrame:
    """Build FDR (Fixture Difficulty Rating) from FPL API fixtures."""
    rows = []
    for fx in api_fixtures:
        event = fx.get("event")
        if event is None:
            continue
        # Home team
        rows.append({
            "team_id": fx["team_h"],
            "gameweek": event,
            "fdr": fx.get("team_h_difficulty", 3),
            "opponent_team_id": fx["team_a"],
        })
        # Away team
        rows.append({
            "team_id": fx["team_a"],
            "gameweek": event,
            "fdr": fx.get("team_a_difficulty", 3),
            "opponent_team_id": fx["team_h"],
        })
    return pd.DataFrame(rows)


def _build_team_id_to_code_map(teams: pd.DataFrame) -> dict:
    """Map FPL team id -> team code."""
    return dict(zip(teams["id"], teams["code"]))


def _build_next3_features(fixture_map: pd.DataFrame, fdr_map: pd.DataFrame,
                           elo: pd.DataFrame) -> pd.DataFrame:
    """Compute lookahead features over the next 3 GWs for the 3-GW target.

    Returns DataFrame with team_code, gameweek, and:
      - fixture_count_next3: total fixtures in next 3 GWs
      - home_pct_next3: fraction of those fixtures at home
      - avg_fdr_next3: average FDR across next 3 GWs
      - avg_opponent_elo_next3: average opponent Elo across next 3 GWs
    """
    if fixture_map.empty:
        return pd.DataFrame(columns=["team_code", "gameweek"])

    fm = fixture_map.copy()
    if not fdr_map.empty and "team_code" in fdr_map.columns and "opponent_code" in fdr_map.columns:
        # Use per-fixture FDR keyed by opponent so DGW fixtures get
        # their individual difficulty ratings, not just the first one
        fdr_lookup = fdr_map[["team_code", "gameweek", "opponent_code", "fdr"]].copy()
        fdr_lookup = fdr_lookup.dropna(subset=["opponent_code"])
        fdr_lookup["opponent_code"] = fdr_lookup["opponent_code"].astype(int)
        fdr_lookup = fdr_lookup.drop_duplicates(
            subset=["team_code", "gameweek", "opponent_code"], keep="first"
        )
        fm = fm.merge(fdr_lookup, on=["team_code", "gameweek", "opponent_code"], how="left")

    if not elo.empty and "team_elo" in elo.columns:
        elo_dict = dict(zip(elo["team_code"], elo["team_elo"]))
        fm["opponent_elo"] = fm["opponent_code"].map(elo_dict)

    rows = []
    for team, tf in fm.groupby("team_code"):
        tf = tf.sort_values("gameweek")
        gws = sorted(tf["gameweek"].unique())
        for gw in gws:
            ahead = tf[(tf["gameweek"] > gw) & (tf["gameweek"] <= gw + 3)]
            if ahead.empty:
                continue
            entry = {"team_code": int(team), "gameweek": int(gw)}
            entry["fixture_count_next3"] = len(ahead)
            entry["home_pct_next3"] = float(ahead["is_home"].mean())
            if "fdr" in ahead.columns:
                fdr_vals = ahead["fdr"].fillna(3.0)
                entry["avg_fdr_next3"] = float(fdr_vals.mean())
            if "opponent_elo" in ahead.columns and ahead["opponent_elo"].notna().any():
                entry["avg_opponent_elo_next3"] = float(ahead["opponent_elo"].mean())
            rows.append(entry)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["team_code", "gameweek"])


def build_features(data: dict) -> pd.DataFrame:
    """Main entry point: build the full feature matrix from raw data.

    Args:
        data: dict from load_all_data() with season keys, 'api', 'current_season', 'seasons'

    Returns:
        DataFrame with one row per player per gameweek, including features and targets.
    """
    all_frames = []

    current_season = data.get("current_season", "2025-2026")
    seasons = data.get("seasons", [current_season])

    for season_label in seasons:
        season = data.get(season_label, {})
        if not season:
            continue

        players = season.get("players", pd.DataFrame())
        pms = season.get("playermatchstats", pd.DataFrame())
        matches = season.get("matches", pd.DataFrame())
        playerstats = season.get("playerstats", pd.DataFrame())
        teams = season.get("teams", pd.DataFrame())

        if players.empty or playerstats.empty:
            print(f"  Skipping {season_label}: missing core data")
            continue

        print(f"  Building features for {season_label}...")

        # Filter to Premier League matches only — the data source may include
        # Champions League, Europa League, EFL Cup etc. which would pollute
        # features and create phantom double gameweeks
        if "tournament" in matches.columns:
            non_prem = len(matches) - (matches["tournament"] == "prem").sum()
            if non_prem > 0:
                print(f"    Filtering out {non_prem} non-PL matches (CL, EFL Cup, etc.)")
                matches = matches[matches["tournament"] == "prem"].copy()

        # Filter matches to finished ones only for historical data
        if "finished" in matches.columns:
            finished_matches = matches[matches["finished"].astype(str).str.lower() == "true"].copy()
        else:
            finished_matches = matches.copy()

        # Filter playermatchstats to PL matches only
        if not pms.empty and not finished_matches.empty and "match_id" in pms.columns:
            prem_match_ids = set(matches["match_id"].dropna())
            pms_before = len(pms)
            pms = pms[pms["match_id"].isin(prem_match_ids)].copy()
            filtered_out = pms_before - len(pms)
            if filtered_out > 0:
                print(f"    Filtering out {filtered_out} non-PL player match stat rows")

        # 1. Add gameweek to playermatchstats
        if not pms.empty and not finished_matches.empty:
            pms = _add_gameweek_to_pms(pms, finished_matches)
            pms = pms.dropna(subset=["gameweek"])
            pms["gameweek"] = pms["gameweek"].astype(int)

        # 2. Player rolling features
        if not pms.empty:
            player_rolling = _build_player_rolling_features(pms.copy())
        else:
            player_rolling = pd.DataFrame(columns=["player_id", "gameweek"])

        # 2b. EWM features (built from raw PMS stats, not pre-rolled)
        if not pms.empty:
            ewm_features = _build_ewm_features(pms.copy())
        else:
            ewm_features = pd.DataFrame(columns=["player_id", "gameweek"])

        # 2c. Upside features (volatility, acceleration, big chances)
        if not pms.empty:
            upside_features = _build_upside_features(pms.copy())
        else:
            upside_features = pd.DataFrame(columns=["player_id", "gameweek"])

        # 2d. Home/away form splits
        if not pms.empty and not finished_matches.empty:
            home_away_form = _build_home_away_form(pms, finished_matches, players)
        else:
            home_away_form = pd.DataFrame(columns=["player_id", "gameweek"])

        # 2e. Opponent-specific history
        if not pms.empty and not finished_matches.empty:
            opponent_history = _build_opponent_history_features(pms, finished_matches, players)
        else:
            opponent_history = pd.DataFrame(columns=["player_id", "gameweek", "opponent_code"])

        # 3. Team match stats + opponent rolling features + own-team rolling features
        if not finished_matches.empty:
            team_stats = _build_team_match_stats(finished_matches)
            opp_rolling = _build_opponent_rolling_features(team_stats)
            own_team_rolling = _build_own_team_rolling_features(team_stats)
        else:
            opp_rolling = pd.DataFrame(columns=["team_code", "gameweek"])
            own_team_rolling = pd.DataFrame(columns=["team_code", "gameweek"])

        # 3b. Rest days / fixture congestion
        if not finished_matches.empty:
            rest_days = _build_rest_days_features(finished_matches)
        else:
            rest_days = pd.DataFrame(columns=["team_code", "gameweek"])

        # Team id -> code mapping (FPL API uses team id, GitHub data uses team code)
        team_id_to_code = _build_team_id_to_code_map(teams) if not teams.empty else {}

        # 4. Fixture map (team -> GW -> opponent, is_home)
        #    Built from played matches + FPL API fixtures for upcoming GWs
        if not matches.empty:
            fixture_map = _build_fixture_map(matches)
        else:
            fixture_map = pd.DataFrame(columns=["team_code", "gameweek", "opponent_code", "is_home"])

        # Supplement fixture map with future fixtures from FPL API so that
        # the next-GW shift has data at the latest gameweek.
        # Only for current season — API data is for the current season only.
        api_fixtures_raw = data["api"].get("fixtures", [])
        if api_fixtures_raw and team_id_to_code and season_label == current_season:
            future_rows = []
            existing_keys = set()
            if not fixture_map.empty:
                # Use (team, gw, opponent) so partially-played DGWs still get
                # the second fixture added from the API
                existing_keys = set(zip(
                    fixture_map["team_code"], fixture_map["gameweek"],
                    fixture_map["opponent_code"]
                ))
            for fx in api_fixtures_raw:
                event = fx.get("event")
                if event is None:
                    continue
                h_code = team_id_to_code.get(fx.get("team_h"))
                a_code = team_id_to_code.get(fx.get("team_a"))
                if h_code is None or a_code is None:
                    continue
                if (int(h_code), int(event), int(a_code)) not in existing_keys:
                    future_rows.append({"team_code": int(h_code), "gameweek": int(event),
                                        "opponent_code": int(a_code), "is_home": 1})
                if (int(a_code), int(event), int(h_code)) not in existing_keys:
                    future_rows.append({"team_code": int(a_code), "gameweek": int(event),
                                        "opponent_code": int(h_code), "is_home": 0})
            if future_rows:
                fixture_map = pd.concat([fixture_map, pd.DataFrame(future_rows)], ignore_index=True)

        # 5. Playerstats features (form, BPS, ICT, cost, etc.)
        ps_features = _build_playerstats_features(playerstats)

        # 6. Targets
        targets = _build_targets(playerstats)

        # 7. Elo ratings
        elo = _build_elo_features(teams)

        # 8. FDR from API (current season only)
        if season_label == current_season:
            api_fixtures = data["api"].get("fixtures", [])
            fdr_map = _build_fdr_map(api_fixtures)
        else:
            fdr_map = pd.DataFrame()

        # Map FDR from team_id to team_code
        if not fdr_map.empty and team_id_to_code:
            fdr_map["team_code"] = fdr_map["team_id"].map(team_id_to_code)
            fdr_map["opponent_code"] = fdr_map["opponent_team_id"].map(team_id_to_code)
            fdr_map = fdr_map.dropna(subset=["team_code"])
            fdr_map["team_code"] = fdr_map["team_code"].astype(int)

        # --- Assemble ---
        # Start from playerstats features (one row per player per GW)
        df = ps_features.copy()

        # Add player info (team, position)
        df = df.merge(
            players[["player_id", "team_code", "position"]],
            on="player_id", how="left"
        )

        # Add NEXT fixture info (opponent, is_home) — shifted so that a row
        # at gameweek=N gets the fixture for gameweek N+1, aligning with the
        # target (next_gw_points = points scored in GW N+1).
        if not fixture_map.empty:
            # Count fixtures per team per GW to detect DGWs (2+) and BGWs (0)
            fixture_counts = (
                fixture_map.groupby(["team_code", "gameweek"])
                .size()
                .reset_index(name="next_gw_fixture_count")
            )
            fixture_counts["gameweek"] = fixture_counts["gameweek"] - 1

            next_fixture = fixture_map[["team_code", "gameweek", "opponent_code", "is_home"]].copy()
            next_fixture["gameweek"] = next_fixture["gameweek"] - 1  # GW N+1 fixture → attach to GW N row

            # Add per-fixture FDR so DGW fixtures keep their individual difficulty
            if not fdr_map.empty and "opponent_code" in fdr_map.columns:
                fdr_for_fixture = fdr_map[["team_code", "gameweek", "opponent_code", "fdr"]].copy()
                fdr_for_fixture = fdr_for_fixture.dropna(subset=["opponent_code"])
                fdr_for_fixture["opponent_code"] = fdr_for_fixture["opponent_code"].astype(int)
                fdr_for_fixture["gameweek"] = fdr_for_fixture["gameweek"] - 1
                fdr_for_fixture = fdr_for_fixture.drop_duplicates(
                    subset=["team_code", "gameweek", "opponent_code"], keep="first"
                )
                next_fixture = next_fixture.merge(
                    fdr_for_fixture,
                    on=["team_code", "gameweek", "opponent_code"],
                    how="left"
                )

            # No dedup — DGW players get one row per fixture with correct
            # opponent data so the model can predict each match separately
            df = df.merge(next_fixture, on=["team_code", "gameweek"], how="left")

            # Merge DGW fixture count
            df = df.merge(fixture_counts, on=["team_code", "gameweek"], how="left")
            df["next_gw_fixture_count"] = df["next_gw_fixture_count"].fillna(1).astype(int)

        # Add player rolling features
        if not player_rolling.empty:
            df = df.merge(player_rolling, on=["player_id", "gameweek"], how="left")

            # Forward-fill rolling features: for players missing PMS at current GW,
            # carry forward their latest available rolling values
            rolling_cols = [c for c in player_rolling.columns if c.startswith("player_") and c not in ("player_id",)]
            if rolling_cols:
                df = df.sort_values(["player_id", "gameweek"])
                df[rolling_cols] = df.groupby("player_id")[rolling_cols].ffill()

        # Add EWM features
        if not ewm_features.empty:
            df = df.merge(ewm_features, on=["player_id", "gameweek"], how="left")
            ewm_cols = [c for c in ewm_features.columns if c.startswith("ewm_")]
            if ewm_cols:
                df = df.sort_values(["player_id", "gameweek"])
                df[ewm_cols] = df.groupby("player_id")[ewm_cols].ffill()

        # Add upside features
        if not upside_features.empty:
            df = df.merge(upside_features, on=["player_id", "gameweek"], how="left")
            upside_cols = [c for c in upside_features.columns
                           if c not in ("player_id", "gameweek")]
            if upside_cols:
                df = df.sort_values(["player_id", "gameweek"])
                df[upside_cols] = df.groupby("player_id")[upside_cols].ffill()

        # Add home/away form
        if not home_away_form.empty:
            df = df.merge(home_away_form, on=["player_id", "gameweek"], how="left")
            for col in ["home_xg_form", "away_xg_form"]:
                if col in df.columns:
                    df = df.sort_values(["player_id", "gameweek"])
                    df[col] = df.groupby("player_id")[col].ffill()

        # Add opponent rolling features for the NEXT-GW opponent.
        # opponent_code now points to the GW N+1 opponent, so we look up that
        # opponent's rolling stats at GW N (their most recent historical stats).
        if not opp_rolling.empty and "opponent_code" in df.columns:
            opp_feats = opp_rolling.rename(columns={"team_code": "opponent_code"})
            df = df.merge(opp_feats, on=["opponent_code", "gameweek"], how="left")

            # Forward-fill opponent rolling features by opponent
            opp_cols = [c for c in opp_feats.columns if c.startswith("opp_")]
            if opp_cols:
                df = df.sort_values(["opponent_code", "gameweek"])
                df[opp_cols] = df.groupby("opponent_code")[opp_cols].ffill()

        # Add own-team rolling features (team attacking/defensive strength)
        if not own_team_rolling.empty:
            df = df.merge(own_team_rolling, on=["team_code", "gameweek"], how="left")

        # Add rest days / fixture congestion
        if not rest_days.empty:
            df = df.merge(rest_days, on=["team_code", "gameweek"], how="left")
            df["days_rest"] = df["days_rest"].fillna(7.0)
            df["fixture_congestion"] = df["fixture_congestion"].fillna(1.0 / 7.0)

        # Add opponent-specific history via merge_asof: for each row at GW N
        # (predicting GW N+1 vs opponent X), find the latest match this player
        # played against X at GW <= N.  This avoids future-data leakage.
        if not opponent_history.empty and "opponent_code" in df.columns:
            df["opponent_code"] = df["opponent_code"].astype("Int64")
            opponent_history["opponent_code"] = opponent_history["opponent_code"].astype("Int64")
            pre_sort_cols = df.columns.tolist()
            df = df.reset_index(drop=True)
            df["_orig_order"] = df.index
            df = df.sort_values("gameweek")
            opponent_history = opponent_history.sort_values("gameweek")
            df = pd.merge_asof(
                df,
                opponent_history,
                on="gameweek",
                by=["player_id", "opponent_code"],
                direction="backward",
            )
            df = df.sort_values("_orig_order").drop(columns=["_orig_order"])

        # Build venue_matched_form: pick home or away xG form based on next fixture
        if "home_xg_form" in df.columns and "away_xg_form" in df.columns and "is_home" in df.columns:
            df["venue_matched_form"] = np.where(
                df["is_home"] == 1, df["home_xg_form"], df["away_xg_form"]
            )

        # Add opponent Elo (keyed by opponent_code only — now correctly the next-GW opponent)
        if not elo.empty and "opponent_code" in df.columns:
            opp_elo = elo.rename(columns={"team_code": "opponent_code", "team_elo": "opponent_elo"})
            df = df.merge(opp_elo, on="opponent_code", how="left")

        # FDR is now merged per-fixture above (inside the fixture merge).
        # Fill missing FDR with 3.0 (neutral) — FDR uses a 1-5 scale, so 0
        # would be misleading. Missing values occur for seasons without API data.
        if "fdr" in df.columns:
            df["fdr"] = df["fdr"].fillna(3.0)

        # Multi-GW lookahead features (for next_3gw_points target)
        next3 = _build_next3_features(fixture_map, fdr_map, elo)
        if not next3.empty:
            df = df.merge(next3, on=["team_code", "gameweek"], how="left")

        # Add targets
        df = df.merge(targets, on=["player_id", "gameweek"], how="left")

        # Add decomposed targets (per-component: goals, assists, CS, bonus, etc.)
        if not pms.empty:
            decomposed = _build_decomposed_targets(
                pms, playerstats, matches=finished_matches, players=players,
            )
            if not decomposed.empty:
                df = df.merge(decomposed, on=["player_id", "gameweek"], how="left")

        # Add season label
        df["season"] = season_label

        # Season progress indicator (helps model weight early-season noise)
        df["season_progress"] = df["gameweek"] / 38.0

        # --- Team form (rolling avg of team total event_points) ---
        # Deduplicate before summing so DGW players aren't double-counted
        if "event_points" in df.columns and "team_code" in df.columns:
            team_pts = (
                df.drop_duplicates(subset=["player_id", "gameweek"], keep="first")
                .groupby(["team_code", "gameweek"])["event_points"]
                .sum()
                .reset_index()
                .sort_values(["team_code", "gameweek"])
            )
            team_pts["team_form_5"] = (
                team_pts.groupby("team_code")["event_points"]
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )
            df = df.merge(
                team_pts[["team_code", "gameweek", "team_form_5"]],
                on=["team_code", "gameweek"], how="left",
            )

        # --- Opponent-adjusted interaction features ---
        if "player_xg_last3" in df.columns and "opp_goals_conceded_last3" in df.columns:
            df["xg_x_opp_goals_conceded"] = df["player_xg_last3"] * df["opp_goals_conceded_last3"]
        if "player_chances_created_last3" in df.columns and "opp_big_chances_allowed_last3" in df.columns:
            df["chances_x_opp_big_chances"] = df["player_chances_created_last3"] * df["opp_big_chances_allowed_last3"]
        if "opp_opponent_xg_last3" in df.columns:
            # Lower opponent xG = better for clean sheets
            df["cs_opportunity"] = 1.0 / (df["opp_opponent_xg_last3"] + 0.1)

        # --- Position one-hot encoding ---
        pos_map = {"Goalkeeper": "GKP", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}
        df["position_clean"] = df["position"].map(pos_map).fillna("UNK")
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            df[f"pos_{pos}"] = (df["position_clean"] == pos).astype(int)

        # --- Minutes availability proxy ---
        if "player_minutes_played_last3" in df.columns:
            df["minutes_availability"] = df["player_minutes_played_last3"] / 90.0

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Carry forward rolling features across seasons so early-season rows
    # inherit end-of-prior-season values instead of starting from NaN.
    # Apply exponential decay (factor per GW) so stale carry-over values
    # fade out as fresh in-season data becomes available.
    if len(all_frames) > 1:
        CROSS_SEASON_DECAY = 0.90  # per-GW decay factor for carried-over values

        def _ffill_with_decay(group_df, cols, group_col):
            """Forward-fill across seasons with per-GW decay on carried values."""
            group_df = group_df.sort_values([group_col, "season", "gameweek"])

            for col in cols:
                filled = group_df.groupby(group_col)[col].ffill()
                # Identify rows where the value was NaN but got filled (carry-over)
                was_nan = group_df[col].isna()
                is_filled = filled.notna()
                carried = was_nan & is_filled

                if carried.any():
                    # Count consecutive NaN-then-filled rows per group
                    # to compute decay distance
                    gw_of_last_real = group_df[col].copy()
                    gw_of_last_real[group_df[col].isna()] = np.nan
                    # For each row, get the gameweek of the last real (non-NaN) value
                    last_real_gw = group_df.groupby(group_col)["gameweek"].transform(
                        lambda s: s.where(group_df.loc[s.index, col].notna()).ffill()
                    )
                    distance = group_df["gameweek"] - last_real_gw
                    decay = CROSS_SEASON_DECAY ** distance.clip(lower=0)
                    # Only apply decay to carried-over values, keep real values intact
                    group_df[col] = np.where(carried, filled * decay, filled)
                else:
                    group_df[col] = filled

            return group_df

        combined = combined.sort_values(["player_id", "season", "gameweek"])
        player_rolling_cols = [c for c in combined.columns
                               if c.startswith("player_") and "_last" in c]
        if player_rolling_cols:
            combined = _ffill_with_decay(combined, player_rolling_cols, "player_id")

        opp_rolling_cols = [c for c in combined.columns
                            if c.startswith("opp_") and "_last" in c]
        if opp_rolling_cols and "opponent_code" in combined.columns:
            combined = _ffill_with_decay(combined, opp_rolling_cols, "opponent_code")

        if "team_form_5" in combined.columns:
            combined = _ffill_with_decay(combined, ["team_form_5"], "team_code")

        # Forward-fill own-team rolling features across seasons
        own_team_cols = [c for c in combined.columns
                         if c.startswith("team_") and "_last" in c]
        if own_team_cols:
            combined = _ffill_with_decay(combined, own_team_cols, "team_code")

        # Forward-fill upside features across seasons
        upside_cols = [c for c in combined.columns
                       if c in ("xg_volatility_last5", "form_acceleration",
                                "big_chance_frequency_last5")]
        if upside_cols:
            combined = _ffill_with_decay(combined, upside_cols, "player_id")

        # Forward-fill EWM features across seasons
        ewm_cols = [c for c in combined.columns if c.startswith("ewm_")]
        if ewm_cols:
            combined = _ffill_with_decay(combined, ewm_cols, "player_id")

        # Forward-fill home/away form across seasons
        ha_form_cols = [c for c in ["home_xg_form", "away_xg_form"] if c in combined.columns]
        if ha_form_cols:
            combined = _ffill_with_decay(combined, ha_form_cols, "player_id")

        # Recompute venue_matched_form after cross-season ffill — the
        # per-season computation (line ~994) runs before home/away form is
        # forward-filled, leaving NaN for players without in-season data.
        if "home_xg_form" in combined.columns and "away_xg_form" in combined.columns and "is_home" in combined.columns:
            combined["venue_matched_form"] = np.where(
                combined["is_home"] == 1, combined["home_xg_form"], combined["away_xg_form"]
            )

        # Forward-fill transfer momentum across seasons
        if "transfer_momentum" in combined.columns:
            combined = _ffill_with_decay(combined, ["transfer_momentum"], "player_id")

    # Drop rows with no target (last few GWs of season)
    print(f"  Total rows before target filter: {len(combined)}")
    print(f"  Rows with next_gw_points: {combined['next_gw_points'].notna().sum()}")

    # Filter out players with zero minutes in the last 5 GWs — these are
    # trivial zero-point predictions that dilute model training signal.
    if "player_minutes_played_last5" in combined.columns:
        before = len(combined)
        combined = combined[combined["player_minutes_played_last5"] > 0].copy()
        print(f"  Filtered out {before - len(combined)} non-playing rows (0 mins in last 5 GWs)")

    # Defragment — many sequential merges cause internal fragmentation
    combined = combined.copy()

    return combined


def get_fixture_context(data: dict) -> dict:
    """Extract fixture context needed for multi-GW predictions.

    Returns dict with:
      - fixture_map: DataFrame(team_code, gameweek, opponent_code, is_home)
      - fdr_map: DataFrame(team_code, gameweek, opponent_code, fdr)
      - elo: DataFrame(team_code, team_elo)
      - opp_rolling: DataFrame(opponent_code, gameweek, opp_* columns)
    """
    current_season = data.get("current_season", "2025-2026")
    season = data.get(current_season, {})
    matches = season.get("matches", pd.DataFrame())
    teams_df = season.get("teams", pd.DataFrame())
    team_id_to_code = _build_team_id_to_code_map(teams_df) if not teams_df.empty else {}

    # Filter to PL matches
    if not matches.empty and "tournament" in matches.columns:
        matches = matches[matches["tournament"] == "prem"].copy()

    # Build fixture_map from played matches
    if not matches.empty:
        fixture_map = _build_fixture_map(matches)
    else:
        fixture_map = pd.DataFrame(columns=["team_code", "gameweek", "opponent_code", "is_home"])

    # Add future fixtures from API
    api_fixtures_raw = data["api"].get("fixtures", [])
    if api_fixtures_raw and team_id_to_code:
        future_rows = []
        existing_keys = set()
        if not fixture_map.empty:
            existing_keys = set(zip(
                fixture_map["team_code"], fixture_map["gameweek"],
                fixture_map["opponent_code"]
            ))
        for fx in api_fixtures_raw:
            event = fx.get("event")
            if event is None:
                continue
            h_code = team_id_to_code.get(fx.get("team_h"))
            a_code = team_id_to_code.get(fx.get("team_a"))
            if h_code is None or a_code is None:
                continue
            if (int(h_code), int(event), int(a_code)) not in existing_keys:
                future_rows.append({"team_code": int(h_code), "gameweek": int(event),
                                    "opponent_code": int(a_code), "is_home": 1})
            if (int(a_code), int(event), int(h_code)) not in existing_keys:
                future_rows.append({"team_code": int(a_code), "gameweek": int(event),
                                    "opponent_code": int(h_code), "is_home": 0})
        if future_rows:
            fixture_map = pd.concat([fixture_map, pd.DataFrame(future_rows)], ignore_index=True)

    # Build FDR map
    fdr_map = _build_fdr_map(api_fixtures_raw) if api_fixtures_raw else pd.DataFrame()
    if not fdr_map.empty and team_id_to_code:
        fdr_map["team_code"] = fdr_map["team_id"].map(team_id_to_code)
        fdr_map["opponent_code"] = fdr_map["opponent_team_id"].map(team_id_to_code)
        fdr_map = fdr_map.dropna(subset=["team_code"])
        fdr_map["team_code"] = fdr_map["team_code"].astype(int)

    # Build Elo
    elo = _build_elo_features(teams_df) if not teams_df.empty else pd.DataFrame()

    # Build opponent rolling features
    if not matches.empty:
        finished_matches = matches.copy()
        if "finished" in finished_matches.columns:
            finished_matches = finished_matches[
                finished_matches["finished"].astype(str).str.lower() == "true"
            ].copy()
        team_stats = _build_team_match_stats(finished_matches)
        opp_rolling = _build_opponent_rolling_features(team_stats)
    else:
        opp_rolling = pd.DataFrame(columns=["team_code", "gameweek"])

    return {
        "fixture_map": fixture_map,
        "fdr_map": fdr_map,
        "elo": elo,
        "opp_rolling": opp_rolling,
    }


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excluding targets, IDs, metadata)."""
    exclude = {
        "player_id", "gameweek", "season", "team_code", "opponent_code",
        "position", "position_clean", "next_gw_points", "next_3gw_points",
        "event_points", "web_name", "cumulative_minutes", "ep_next",
    }
    # Also exclude set piece order raw columns (we use the binary flag)
    exclude.update({"penalties_order", "corners_order", "freekicks_order",
                     "transfers_out_event", "total_points"})

    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
    return sorted(feature_cols)
