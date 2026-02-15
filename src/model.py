"""Phase 2: Train XGBoost prediction models using selected features."""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent
else:
    _BASE = Path(__file__).resolve().parent.parent

MODEL_DIR = _BASE / "models"
OUTPUT_DIR = _BASE / "output"
_N_JOBS = 1 if getattr(sys, "frozen", False) else -1
POSITION_GROUPS = ["GKP", "DEF", "MID", "FWD"]
TARGETS = ["next_gw_points", "next_3gw_points"]

MIN_TRAIN_GWS = 10

from src.data_fetcher import detect_current_season
CURRENT_SEASON = detect_current_season()


def _season_weight(season: str, current: str) -> float:
    """Weight: 1.0 for current, 0.5 for previous, 0.25 for two back, etc."""
    cur_year = int(current.split("-")[0])
    s_year = int(season.split("-")[0])
    age = cur_year - s_year  # 0 for current, 1 for prev, 2 for two back
    return 0.5 ** age

# Default feature sets per position if feature selection hasn't been run
# These will be overridden by actual feature selection results
DEFAULT_FEATURES = {
    "GKP": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "opp_opponent_xg_last3", "opp_goals_conceded_last3",
        "player_minutes_played_last3", "chance_of_playing", "cs_opportunity",
        "opp_opponent_shots_on_target_last3", "gw_influence", "ownership",
        "team_form_5", "next_gw_fixture_count", "season_progress",
        "avg_fdr_next3", "home_pct_next3", "avg_opponent_elo_next3",
        # GKP-specific save data
        "player_saves_last3", "player_saves_last5", "saves_per_90",
        # Clean sheet and starter data
        "clean_sheets_per_90", "starts_per_90", "xgc_per_90",
        # Own-team defensive strength
        "team_clean_sheet_last3", "team_goals_scored_last3",
        # Longer windows
        "player_minutes_played_last5",
        # Transfer momentum
        "transfer_momentum",
        # Rest / congestion
        "days_rest", "fixture_congestion",
        # Venue form
        "venue_matched_form",
        # Rotation risk
        "availability_rate_last5",
        # Opponent history
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        # Market sentiment
        "transfers_in_event", "net_transfers",
        # Opponent attacking threat
        "opp_big_chances_allowed_last3",
    ],
    "DEF": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "player_xg_last3", "player_xa_last3",
        "player_chances_created_last3", "player_clearances_last3",
        "player_interceptions_last3", "player_tackles_won_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "cs_opportunity", "player_minutes_played_last3",
        "chance_of_playing", "gw_influence", "gw_threat", "gw_creativity",
        "set_piece_involvement", "team_form_5",
        "next_gw_fixture_count", "season_progress",
        "avg_fdr_next3", "home_pct_next3", "avg_opponent_elo_next3",
        # Clean sheet and starter data
        "clean_sheets_per_90", "starts_per_90", "xgc_per_90",
        # Card risk
        "yellow_cards",
        # Own-team strength
        "team_clean_sheet_last3", "team_goals_scored_last3", "team_xg_last3",
        # Longer windows for key stats
        "player_xg_last5", "player_xa_last5", "player_minutes_played_last5",
        # EWM features
        "ewm_player_xg_last3", "ewm_player_xa_last3",
        # Transfer momentum
        "transfer_momentum",
        # Rest / congestion
        "days_rest", "fixture_congestion",
        # Venue form
        "venue_matched_form",
        # Opponent history
        "vs_opponent_matches", "vs_opponent_goals_avg", "vs_opponent_xg_avg",
        # Market sentiment
        "transfers_in_event", "net_transfers", "ownership",
        # FPL index
        "gw_ict_index",
        # Opponent attacking threat
        "opp_big_chances_allowed_last3",
        # Rotation risk
        "availability_rate_last5",
    ],
    "MID": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "player_xg_last3", "player_xa_last3",
        "player_xgot_last3", "player_shots_on_target_last3",
        "player_chances_created_last3", "player_touches_opposition_box_last3",
        "player_successful_dribbles_last3", "player_accurate_crosses_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "opp_big_chances_allowed_last3",
        "xg_x_opp_goals_conceded", "chances_x_opp_big_chances",
        "player_minutes_played_last3", "chance_of_playing",
        "gw_influence", "gw_threat", "gw_creativity", "gw_ict_index",
        "set_piece_involvement", "team_form_5",
        "next_gw_fixture_count", "season_progress",
        "avg_fdr_next3", "home_pct_next3", "avg_opponent_elo_next3",
        # Starter and transfer data
        "starts_per_90", "transfers_in_event",
        # Card risk
        "yellow_cards",
        # Own-team attacking strength
        "team_goals_scored_last3", "team_xg_last3", "team_big_chances_last3",
        # Longer windows for key stats
        "player_xg_last5", "player_xa_last5", "player_shots_on_target_last5",
        "player_minutes_played_last5",
        "player_touches_opposition_box_last5", "player_total_shots_last3",
        # EWM features
        "ewm_player_xg_last3", "ewm_player_xa_last3",
        "ewm_player_xgot_last3", "ewm_player_chances_created_last3",
        "ewm_player_shots_on_target_last3",
        # Transfer momentum
        "net_transfers", "transfer_momentum",
        # Rest / congestion
        "days_rest", "fixture_congestion",
        # Venue form
        "home_xg_form", "away_xg_form", "venue_matched_form",
        # Opponent history
        "vs_opponent_xg_avg", "vs_opponent_goals_avg", "vs_opponent_matches",
        # Upside / explosive potential
        "xg_volatility_last5", "form_acceleration", "big_chance_frequency_last5",
        # Rotation risk
        "availability_rate_last5",
    ],
    "FWD": [
        "player_form", "cost", "gw_player_bps", "is_home", "fdr",
        "opponent_elo", "player_xg_last3", "player_xa_last3",
        "player_xgot_last3", "player_total_shots_last3",
        "player_shots_on_target_last3", "player_touches_opposition_box_last3",
        "player_big_chances_missed_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "opp_big_chances_allowed_last3",
        "xg_x_opp_goals_conceded",
        "player_minutes_played_last3", "chance_of_playing",
        "gw_influence", "gw_threat", "gw_creativity", "gw_ict_index",
        "set_piece_involvement", "chances_x_opp_big_chances",
        "team_form_5", "next_gw_fixture_count", "season_progress",
        "avg_fdr_next3", "home_pct_next3", "avg_opponent_elo_next3",
        # Starter and transfer data
        "starts_per_90", "transfers_in_event",
        # Own-team attacking strength
        "team_goals_scored_last3", "team_xg_last3", "team_big_chances_last3",
        # Longer windows for key stats
        "player_xg_last5", "player_xa_last5", "player_shots_on_target_last5",
        "player_touches_opposition_box_last5", "player_minutes_played_last5",
        # EWM features
        "ewm_player_xg_last3", "ewm_player_xa_last3",
        "ewm_player_xgot_last3", "ewm_player_chances_created_last3",
        "ewm_player_shots_on_target_last3",
        # Transfer momentum
        "net_transfers", "transfer_momentum",
        # Rest / congestion
        "days_rest", "fixture_congestion",
        # Venue form
        "home_xg_form", "away_xg_form", "venue_matched_form",
        # Opponent history
        "vs_opponent_xg_avg", "vs_opponent_goals_avg", "vs_opponent_matches",
        # Upside / explosive potential
        "xg_volatility_last5", "form_acceleration", "big_chance_frequency_last5",
        # Rotation risk
        "availability_rate_last5",
    ],
}

def get_features_for_position(position: str) -> list[str]:
    """Get feature list for a position, using selected_features.json if available.

    Falls back to DEFAULT_FEATURES if the JSON file doesn't exist.
    """
    json_path = OUTPUT_DIR / "selected_features.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                selected = json.load(f)
            if position in selected and selected[position]:
                return selected[position]
        except (json.JSONDecodeError, KeyError):
            pass
    return DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"])


# XGBoost parameter grid for tuning
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def _prepare_position_data(df: pd.DataFrame, position: str, target: str, feature_cols: list[str]):
    """Filter and prepare data for a specific position/target."""
    pos_df = df[df["position_clean"] == position].copy()
    pos_df = pos_df.dropna(subset=[target])

    # DGW handling: keep per-fixture rows (aligned with inference).
    # Divide target by fixture count so each row represents a single
    # fixture's contribution to total GW points.  At inference time,
    # per-fixture predictions are summed back to the full GW total.
    group_keys = ["player_id", "season", "gameweek"]
    if pos_df.duplicated(subset=group_keys, keep=False).any():
        if "next_gw_fixture_count" in pos_df.columns:
            fc = pos_df["next_gw_fixture_count"].clip(lower=1)
            pos_df[target] = pos_df[target] / fc
    else:
        pos_df = pos_df.drop_duplicates(subset=group_keys, keep="first")

    available_feats = [c for c in feature_cols if c in pos_df.columns]
    # Require at least half the features to be non-null
    pos_df = pos_df.dropna(subset=available_feats, thresh=(len(available_feats) + 1) // 2)
    # Use semantically correct defaults for features where 0 is misleading
    _fill_defaults = {"opponent_elo": 1500.0, "fdr": 3.0, "avg_fdr_next3": 3.0,
                      "avg_opponent_elo_next3": 1500.0}
    for c in available_feats:
        pos_df[c] = pos_df[c].fillna(_fill_defaults.get(c, 0))

    return pos_df, available_feats


def _walk_forward_splits(df: pd.DataFrame, min_train_gws: int = MIN_TRAIN_GWS):
    """Walk-forward validation splits across seasons."""
    df = df.copy()
    season_order = sorted(df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    df["_seq_gw"] = df["season"].map(season_map) * 100 + df["gameweek"]

    seq_gws = sorted(df["_seq_gw"].unique())
    if len(seq_gws) < min_train_gws + 1:
        return

    for i in range(min_train_gws, len(seq_gws)):
        train_gws = set(seq_gws[:i])
        test_gw = seq_gws[i]
        train_mask = df["_seq_gw"].isin(train_gws)
        test_mask = df["_seq_gw"] == test_gw
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            yield train_mask, test_mask


def train_model(
    df: pd.DataFrame,
    position: str,
    target: str,
    feature_cols: list[str] | None = None,
    tune: bool = True,
) -> dict:
    """Train an XGBoost model for a position/target combination.

    Returns dict with: model, features, mae, position, target.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    if feature_cols is None:
        feature_cols = get_features_for_position(position)

    pos_df, available_feats = _prepare_position_data(df, position, target, feature_cols)

    if len(pos_df) < 50:
        print(f"    {position}/{target}: insufficient data ({len(pos_df)} rows)")
        return {}

    print(f"    {position}/{target}: {len(pos_df)} rows, {len(available_feats)} features")

    # Season-based sample weights: graduated decay (current=1.0, prev=0.5, etc.)
    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    # Sort by time for temporal ordering
    season_order = sorted(pos_df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    pos_df = pos_df.copy()
    pos_df["_seq_gw"] = pos_df["season"].map(season_map) * 100 + pos_df["gameweek"]
    pos_df = pos_df.sort_values("_seq_gw")

    X_all = pos_df[available_feats].values
    y_all = pos_df[target].values
    w_all = pos_df["_sample_weight"].values

    # Tune hyperparameters first (if requested) so walk-forward uses tuned params
    if tune:
        print(f"    Tuning hyperparameters...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42, verbosity=0,
            )
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(
                base_model, PARAM_GRID, cv=tscv, scoring="neg_mean_absolute_error",
                n_jobs=_N_JOBS, verbose=0,
            )
            grid.fit(X_all, y_all, sample_weight=w_all)
            best_params = grid.best_params_
            print(f"    Best params: {best_params}")
    else:
        best_params = {
            "n_estimators": 150, "max_depth": 5, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }

    # Walk-forward validation using the selected hyperparameters
    maes = []
    spearmans = []
    all_pred_resid = []  # (prediction, residual) pairs for conditional intervals
    n_splits = 0
    for train_mask, test_mask in _walk_forward_splits(pos_df):
        X_train = pos_df.loc[train_mask, available_feats].values
        y_train = pos_df.loc[train_mask, target].values
        w_train = pos_df.loc[train_mask, "_sample_weight"].values
        X_test = pos_df.loc[test_mask, available_feats].values
        y_test = pos_df.loc[test_mask, target].values

        model = XGBRegressor(
            **best_params, objective="reg:squarederror",
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        residuals = y_test - preds
        for pred_val, resid_val in zip(preds, residuals):
            all_pred_resid.append((float(pred_val), float(resid_val)))
        if len(y_test) >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(y_test, preds).correlation
            if not np.isnan(rho):
                spearmans.append(float(rho))
        n_splits += 1
        if n_splits >= 20:
            break

    walk_forward_mae = np.mean(maes) if maes else float("nan")
    avg_spearman = np.mean(spearmans) if spearmans else float("nan")
    print(f"    Walk-forward MAE: {walk_forward_mae:.3f}, Spearman: {avg_spearman:.3f} (over {n_splits} splits)")

    # Residual percentiles for prediction intervals (80% PI)
    # Global fallback
    all_resid_vals = [r for _, r in all_pred_resid]
    residual_q10 = float(np.percentile(all_resid_vals, 10)) if all_resid_vals else 0.0
    residual_q90 = float(np.percentile(all_resid_vals, 90)) if all_resid_vals else 0.0

    # Conditional (heteroscedastic) intervals binned by predicted value (Fix 5)
    residual_bins = {}
    bin_edges = []
    if len(all_pred_resid) >= 30:
        pr = np.array(all_pred_resid)
        pred_vals, resid_vals = pr[:, 0], pr[:, 1]
        bin_edges = np.percentile(pred_vals, [33, 67]).tolist()
        bins = np.digitize(pred_vals, bin_edges)
        for b in range(3):
            mask = bins == b
            if mask.sum() >= 10:
                residual_bins[b] = {
                    "q10": float(np.percentile(resid_vals[mask], 10)),
                    "q90": float(np.percentile(resid_vals[mask], 90)),
                }

    # Holdout: last 3 sequential GWs — realistic estimate of deployed model accuracy
    seq_gws = sorted(pos_df["_seq_gw"].unique())
    holdout_gws = set(seq_gws[-3:])
    holdout_mask = pos_df["_seq_gw"].isin(holdout_gws)
    train_mask = ~holdout_mask
    if train_mask.sum() >= 50 and holdout_mask.sum() >= 10:
        ho_model = XGBRegressor(
            **best_params, objective="reg:squarederror",
            random_state=42, verbosity=0,
        )
        ho_model.fit(
            pos_df.loc[train_mask, available_feats].values,
            pos_df.loc[train_mask, target].values,
            sample_weight=pos_df.loc[train_mask, "_sample_weight"].values,
        )
        ho_preds = ho_model.predict(pos_df.loc[holdout_mask, available_feats].values)
        holdout_mae = mean_absolute_error(pos_df.loc[holdout_mask, target].values, ho_preds)
        print(f"    Holdout MAE (last 3 GWs): {holdout_mae:.3f}")

    # Train final model on all data with the selected hyperparameters
    final_model = XGBRegressor(
        **best_params, objective="reg:squarederror",
        random_state=42, verbosity=0,
    )
    final_model.fit(X_all, y_all, sample_weight=w_all)

    # Save model with residual percentiles for prediction intervals
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{position}_{target}.joblib"
    joblib.dump({
        "model": final_model,
        "features": available_feats,
        "residual_q10": residual_q10,
        "residual_q90": residual_q90,
        "residual_bins": residual_bins,
        "bin_edges": bin_edges,
    }, model_path)
    print(f"    Model saved to {model_path}")

    return {
        "model": final_model,
        "features": available_feats,
        "mae": walk_forward_mae,
        "spearman": avg_spearman,
        "position": position,
        "target": target,
    }


def train_all_models(df: pd.DataFrame, tune: bool = True) -> list[dict]:
    """Train 1-GW models for all positions.

    Only trains ``next_gw_points`` models — 3-GW predictions are derived at
    inference time by summing three 1-GW predictions with per-GW opponent data.
    """
    results = []
    for position in POSITION_GROUPS:
        target = "next_gw_points"
        print(f"\n  Training {position} — {target}...")
        result = train_model(df, position, target, tune=tune)
        if result:
            results.append(result)
    return results


def load_model(position: str, target: str, suffix: str = "") -> dict | None:
    """Load a trained model from disk.

    Args:
        suffix: optional model variant suffix, e.g. "_q80" for quantile models.
    """
    model_path = MODEL_DIR / f"xgb_{position}_{target}{suffix}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def predict_for_position(
    df: pd.DataFrame, position: str, target: str,
    model_dict: dict | None = None, suffix: str = "",
) -> pd.DataFrame:
    """Generate predictions for all players of a given position.

    Args:
        suffix: model variant suffix (e.g. "_q80"). Used to load model from
                disk when model_dict is None, and to name the prediction column.

    Returns DataFrame with player_id, predicted points, and player info.
    """
    if model_dict is None:
        model_dict = load_model(position, target, suffix=suffix)
    if model_dict is None:
        return pd.DataFrame()

    model = model_dict["model"]
    features = model_dict["features"]

    pos_df = df[df["position_clean"] == position].copy()
    available_feats = [c for c in features if c in pos_df.columns]

    if not available_feats:
        return pd.DataFrame()

    # Use same semantically correct defaults as training (_prepare_position_data)
    _fill_defaults = {"opponent_elo": 1500.0, "fdr": 3.0, "avg_fdr_next3": 3.0,
                      "avg_opponent_elo_next3": 1500.0}
    for c in available_feats:
        pos_df[c] = pos_df[c].fillna(_fill_defaults.get(c, 0))

    # Handle missing features (pad with zeros)
    X = np.zeros((len(pos_df), len(features)))
    for i, f in enumerate(features):
        if f in pos_df.columns:
            X[:, i] = pos_df[f].values

    pred_col = f"predicted_{target}{suffix}"
    pos_df[pred_col] = model.predict(X).clip(min=0)

    # DGW players have two rows (one per fixture) with different opponent
    # features.  Sum per-fixture predictions — each fixture contributes
    # additive points.
    if pos_df.duplicated(subset=["player_id"], keep=False).any():
        agg_pred = pos_df.groupby("player_id")[pred_col].sum()
        meta_cols = [c for c in pos_df.columns if c != pred_col]
        deduped = pos_df[meta_cols].drop_duplicates(subset=["player_id"], keep="first")
        deduped = deduped.set_index("player_id")
        deduped[pred_col] = agg_pred
        pos_df = deduped.reset_index()

    return pos_df


def train_quantile_model(
    df: pd.DataFrame,
    position: str,
    target: str = "next_gw_points",
    feature_cols: list[str] | None = None,
    quantile_alpha: float = 0.80,
) -> dict:
    """Train an XGBoost quantile regression model for captain picking.

    Uses reg:quantileerror objective to predict the 80th percentile of
    outcomes rather than the mean. This surfaces players with explosive
    upside potential.

    Only intended for MID and FWD (captain-relevant positions).
    Returns dict with: model, features, mae, position, target, calibration.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    if feature_cols is None:
        feature_cols = get_features_for_position(position)

    pos_df, available_feats = _prepare_position_data(df, position, target, feature_cols)

    if len(pos_df) < 50:
        print(f"    {position}/{target} q{int(quantile_alpha*100)}: insufficient data ({len(pos_df)} rows)")
        return {}

    suffix = f"_q{int(quantile_alpha * 100)}"
    print(f"    {position}/{target}{suffix}: {len(pos_df)} rows, {len(available_feats)} features")

    # Season-based sample weights: graduated decay
    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    # Sort by time for temporal ordering (required for TimeSeriesSplit)
    season_order = sorted(pos_df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    pos_df = pos_df.copy()
    pos_df["_seq_gw"] = pos_df["season"].map(season_map) * 100 + pos_df["gameweek"]
    pos_df = pos_df.sort_values("_seq_gw")

    # Walk-forward validation
    maes = []
    calibration_vals = []
    n_splits = 0
    for train_mask, test_mask in _walk_forward_splits(pos_df):
        X_train = pos_df.loc[train_mask, available_feats].values
        y_train = pos_df.loc[train_mask, target].values
        w_train = pos_df.loc[train_mask, "_sample_weight"].values
        X_test = pos_df.loc[test_mask, available_feats].values
        y_test = pos_df.loc[test_mask, target].values

        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile_alpha,
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        preds = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        # Calibration: fraction of actuals below quantile prediction
        calibration_vals.append(float((y_test <= preds).mean()))
        n_splits += 1
        if n_splits >= 20:
            break

    walk_forward_mae = np.mean(maes) if maes else float("nan")
    avg_calibration = np.mean(calibration_vals) if calibration_vals else float("nan")
    print(f"    Walk-forward MAE: {walk_forward_mae:.3f} (over {n_splits} splits)")
    print(f"    Calibration: {avg_calibration:.1%} of actuals below q{int(quantile_alpha*100)} prediction (target: {quantile_alpha:.0%})")

    # Train final model on all data (pos_df already sorted with _seq_gw)
    X_all = pos_df[available_feats].values
    y_all = pos_df[target].values
    w_all = pos_df["_sample_weight"].values

    final_model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=quantile_alpha,
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0,
    )
    final_model.fit(X_all, y_all, sample_weight=w_all)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{position}_{target}{suffix}.joblib"
    joblib.dump({"model": final_model, "features": available_feats}, model_path)
    print(f"    Model saved to {model_path}")

    return {
        "model": final_model,
        "features": available_feats,
        "mae": walk_forward_mae,
        "calibration": avg_calibration,
        "position": position,
        "target": target,
    }


def train_all_quantile_models(
    df: pd.DataFrame, quantile_alpha: float = 0.80
) -> list[dict]:
    """Train quantile models for MID and FWD (captain-relevant positions only)."""
    results = []
    for position in ["MID", "FWD"]:
        print(f"\n  Training {position} — quantile q{int(quantile_alpha*100)}...")
        result = train_quantile_model(df, position, quantile_alpha=quantile_alpha)
        if result:
            results.append(result)
    return results


# ── Decomposed sub-model definitions ──────────────────────────────────────

# Sub-model components and their targets.  Each component gets a separate
# XGBoost model per position trained on the corresponding target column.
SUB_MODEL_COMPONENTS = {
    "goals": "next_gw_goals",           # regression: expected goals
    "assists": "next_gw_assists",       # regression: expected assists
    "cs": "next_gw_cs",                 # regression: clean sheet count (0 or 1, treat as P(CS))
    "bonus": "next_gw_bonus",           # regression: expected bonus (0-3)
    "goals_conceded": "next_gw_goals_conceded",  # regression: expected GC
    "saves": "next_gw_saves",           # regression: expected saves (GKP only)
}

# XGBoost objective per sub-model component.
# Poisson for count data, squarederror for clean sheets (fractional DGW targets).
SUB_MODEL_OBJECTIVES = {
    "goals": "count:poisson",
    "assists": "count:poisson",
    "cs": "reg:squarederror",
    "bonus": "count:poisson",
    "goals_conceded": "count:poisson",
    "saves": "count:poisson",
}

# FPL scoring multipliers by position
FPL_SCORING = {
    "GKP": {"appearance": 2, "goal": 10, "assist": 3, "cs": 4, "gc_per_2": -1, "save_per_3": 1},
    "DEF": {"appearance": 2, "goal": 6, "assist": 3, "cs": 4, "gc_per_2": -1, "save_per_3": 0},
    "MID": {"appearance": 2, "goal": 5, "assist": 3, "cs": 1, "gc_per_2": 0, "save_per_3": 0},
    "FWD": {"appearance": 2, "goal": 4, "assist": 3, "cs": 0, "gc_per_2": 0, "save_per_3": 0},
}

# Feature sets tailored to each sub-model — smaller and more focused than the
# full points model to avoid diluting signal with irrelevant features.
SUB_MODEL_FEATURES = {
    "goals": [
        "player_xg_last3", "player_xg_last5", "ewm_player_xg_last3",
        "player_xgot_last3", "player_shots_on_target_last3",
        "player_shots_on_target_last5", "player_touches_opposition_box_last3",
        "opp_goals_conceded_last3", "opp_xg_conceded_last3",
        "xg_x_opp_goals_conceded", "is_home", "fdr", "opponent_elo",
        "set_piece_involvement", "big_chance_frequency_last5",
        "player_form", "gw_threat", "next_gw_fixture_count",
        "player_minutes_played_last3", "starts_per_90",
        # Opponent history
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        # Team attacking context
        "cost", "team_goals_scored_last3",
    ],
    "assists": [
        "player_xa_last3", "player_xa_last5", "ewm_player_xa_last3",
        "player_chances_created_last3", "ewm_player_chances_created_last3",
        "player_successful_dribbles_last3", "player_accurate_crosses_last3",
        "opp_big_chances_allowed_last3", "chances_x_opp_big_chances",
        "is_home", "fdr", "opponent_elo", "gw_creativity",
        "player_form", "next_gw_fixture_count",
        "player_minutes_played_last3", "starts_per_90",
    ],
    "cs": [
        "opp_opponent_xg_last3", "opp_opponent_shots_on_target_last3",
        "opp_goals_conceded_last3",
        "team_clean_sheet_last3", "cs_opportunity",
        "is_home", "fdr", "opponent_elo",
        "clean_sheets_per_90", "xgc_per_90",
        "player_minutes_played_last3", "starts_per_90",
        "next_gw_fixture_count",
        # Opponent history
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        "opp_big_chances_allowed_last3",
    ],
    "bonus": [
        "gw_player_bps", "gw_ict_index", "gw_influence", "gw_threat", "gw_creativity",
        "player_form", "player_xg_last3", "player_xa_last3",
        "player_goals_last3", "player_assists_last3",
        "is_home", "cost", "next_gw_fixture_count",
        "player_minutes_played_last3", "starts_per_90",
    ],
    "goals_conceded": [
        "opp_opponent_xg_last3", "opp_goals_conceded_last3",
        "opp_big_chances_allowed_last3",
        "team_clean_sheet_last3", "cs_opportunity",
        "is_home", "fdr", "opponent_elo",
        "xgc_per_90", "next_gw_fixture_count",
        # Opponent history
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
    ],
    "saves": [
        "player_saves_last3", "player_saves_last5", "saves_per_90",
        "opp_opponent_xg_last3", "opp_opponent_shots_on_target_last3",
        "opp_goals_conceded_last3",
        "is_home", "fdr", "opponent_elo",
        "next_gw_fixture_count",
        # Opponent history
        "vs_opponent_goals_avg", "vs_opponent_xg_avg", "vs_opponent_matches",
        "opp_big_chances_allowed_last3",
    ],
}

# Which sub-models to train for each position
SUB_MODELS_FOR_POSITION = {
    "GKP": ["cs", "goals_conceded", "saves", "bonus"],
    "DEF": ["goals", "assists", "cs", "goals_conceded", "bonus"],
    "MID": ["goals", "assists", "cs", "bonus"],
    "FWD": ["goals", "assists", "bonus"],
}


def train_sub_model(
    df: pd.DataFrame,
    position: str,
    component: str,
    tune: bool = False,
) -> dict:
    """Train a single decomposed sub-model for a position/component.

    Faster than the main model — uses fixed hyperparameters by default and
    no grid search, since each sub-model has a narrower target.
    """
    from xgboost import XGBRegressor

    target = SUB_MODEL_COMPONENTS[component]
    feature_cols = SUB_MODEL_FEATURES.get(component, [])

    pos_df, available_feats = _prepare_position_data(
        df, position, target, feature_cols,
    )

    # Train only on rows where the player actually played.
    # This gives E[goals | played] rather than E[goals | all],
    # which better differentiates starters.  P(plays) gates at prediction time.
    if "next_gw_minutes" in pos_df.columns:
        pos_df = pos_df[pos_df["next_gw_minutes"] > 0].copy()

    if len(pos_df) < 30:
        print(f"    {position}/sub_{component}: insufficient data ({len(pos_df)} rows)")
        return {}

    print(f"    {position}/sub_{component}: {len(pos_df)} rows, {len(available_feats)} features")

    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    # Sort by time for temporal ordering (required for TimeSeriesSplit)
    season_order = sorted(pos_df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    pos_df = pos_df.copy()
    pos_df["_seq_gw"] = pos_df["season"].map(season_map) * 100 + pos_df["gameweek"]
    pos_df = pos_df.sort_values("_seq_gw")

    X_all = pos_df[available_feats].values
    y_all = pos_df[target].values
    w_all = pos_df["_sample_weight"].values

    objective = SUB_MODEL_OBJECTIVES.get(component, "reg:squarederror")
    obj_params = {"objective": objective}
    if objective == "binary:logistic":
        obj_params["eval_metric"] = "logloss"

    params = {
        "n_estimators": 150, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
    }

    if tune:
        print(f"    Tuning {component}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = XGBRegressor(**obj_params, random_state=42, verbosity=0)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(
                base, PARAM_GRID, cv=tscv, scoring="neg_mean_absolute_error",
                n_jobs=_N_JOBS, verbose=0,
            )
            grid.fit(X_all, y_all, sample_weight=w_all)
            params = grid.best_params_

    # Walk-forward MAE + Spearman
    maes = []
    spearmans = []
    n_splits = 0
    for train_mask, test_mask in _walk_forward_splits(pos_df):
        X_tr = pos_df.loc[train_mask, available_feats].values
        y_tr = pos_df.loc[train_mask, target].values
        w_tr = pos_df.loc[train_mask, "_sample_weight"].values
        X_te = pos_df.loc[test_mask, available_feats].values
        y_te = pos_df.loc[test_mask, target].values

        m = XGBRegressor(**params, **obj_params, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        preds = m.predict(X_te)
        maes.append(mean_absolute_error(y_te, preds))
        if len(y_te) >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(y_te, preds).correlation
            if not np.isnan(rho):
                spearmans.append(float(rho))
        n_splits += 1
        if n_splits >= 15:
            break

    wf_mae = np.mean(maes) if maes else float("nan")
    avg_spearman = np.mean(spearmans) if spearmans else float("nan")
    print(f"    Walk-forward MAE: {wf_mae:.4f}, Spearman: {avg_spearman:.3f} ({n_splits} splits)")

    # Holdout: last 3 sequential GWs (pos_df already sorted with _seq_gw)
    seq_gws = sorted(pos_df["_seq_gw"].unique())
    holdout_gws = set(seq_gws[-3:])
    ho_mask = pos_df["_seq_gw"].isin(holdout_gws)
    tr_mask = ~ho_mask
    if tr_mask.sum() >= 30 and ho_mask.sum() >= 5:
        ho_m = XGBRegressor(**params, **obj_params, random_state=42, verbosity=0)
        ho_m.fit(
            pos_df.loc[tr_mask, available_feats].values,
            pos_df.loc[tr_mask, target].values,
            sample_weight=pos_df.loc[tr_mask, "_sample_weight"].values,
        )
        ho_preds = ho_m.predict(pos_df.loc[ho_mask, available_feats].values)
        holdout_mae = mean_absolute_error(pos_df.loc[ho_mask, target].values, ho_preds)
        print(f"    Holdout MAE (last 3 GWs): {holdout_mae:.4f}")

    # Final model on all data
    final = XGBRegressor(**params, **obj_params, random_state=42, verbosity=0)
    final.fit(X_all, y_all, sample_weight=w_all)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"xgb_{position}_sub_{component}.joblib"
    save_dict = {"model": final, "features": available_feats}
    joblib.dump(save_dict, path)
    print(f"    Saved → {path}")

    return {
        "model": final, "features": available_feats,
        "mae": wf_mae, "position": position, "component": component,
    }


def train_all_sub_models(df: pd.DataFrame, tune: bool = False) -> list[dict]:
    """Train all decomposed sub-models for all positions."""
    results = []
    for position in POSITION_GROUPS:
        components = SUB_MODELS_FOR_POSITION.get(position, [])
        print(f"\n  Training {position} sub-models ({', '.join(components)})...")
        for comp in components:
            result = train_sub_model(df, position, comp, tune=tune)
            if result:
                results.append(result)
    return results


def load_sub_model(position: str, component: str) -> dict | None:
    """Load a trained sub-model from disk."""
    path = MODEL_DIR / f"xgb_{position}_sub_{component}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def predict_decomposed(
    df: pd.DataFrame, position: str,
) -> pd.DataFrame:
    """Generate decomposed predictions for a position and combine using FPL rules.

    Loads each sub-model, predicts per-component expected values, then applies
    FPL scoring multipliers to produce a total predicted_next_gw_points.

    Handles DGW players the same way as predict_for_position: each fixture row
    gets predicted independently, then summed per player.
    """
    components = SUB_MODELS_FOR_POSITION.get(position, [])
    if not components:
        return pd.DataFrame()

    scoring = FPL_SCORING[position]
    pos_df = df[df["position_clean"] == position].copy()
    if pos_df.empty:
        return pd.DataFrame()

    # Predict each component
    component_preds = {}
    loaded_models = {}  # cache to avoid double disk loads
    for comp in components:
        model_dict = load_sub_model(position, comp)
        if model_dict is None:
            continue
        loaded_models[comp] = model_dict

        model = model_dict["model"]
        features = model_dict["features"]
        available = [c for c in features if c in pos_df.columns]
        if not available:
            continue

        _fill_defaults = {
            "opponent_elo": 1500.0, "fdr": 3.0, "avg_fdr_next3": 3.0,
            "avg_opponent_elo_next3": 1500.0,
        }
        for c in available:
            pos_df[c] = pos_df[c].fillna(_fill_defaults.get(c, 0))

        X = np.zeros((len(pos_df), len(features)))
        for i, f in enumerate(features):
            if f in pos_df.columns:
                X[:, i] = pos_df[f].values

        pos_df[f"sub_{comp}"] = model.predict(X).clip(min=0)
        component_preds[comp] = True

    if not component_preds:
        return pd.DataFrame()

    # --- Combine using FPL scoring rules ---
    # P(plays) = chance_of_playing/100 * availability_rate_last5.
    # chance_of_playing (0-100) is "available to play" from FPL API — but it
    # doesn't distinguish starters from bench warmers (backup GKPs get 100%).
    # Multiplying by availability_rate_last5 (fraction of recent GWs with
    # minutes) filters out players who are fit but don't actually get selected.
    cop = pos_df["chance_of_playing"].fillna(100) / 100.0 if "chance_of_playing" in pos_df.columns else 0.8
    avail = pos_df["availability_rate_last5"].fillna(0.75) if "availability_rate_last5" in pos_df.columns else pd.Series(0.75, index=pos_df.index)
    pos_df["p_plays"] = (cop * avail).clip(0, 1)
    pos_df["p_60plus"] = pos_df["p_plays"] * 0.85

    # Appearance points: E[appearance] = P(60+)*2 + (P(plays)-P(60+))*1
    pos_df["pts_appearance"] = (
        pos_df["p_60plus"] * 2 + (pos_df["p_plays"] - pos_df["p_60plus"]).clip(lower=0) * 1
    )

    # Goals: E[goals_pts] = P(plays) * E[goals | plays] * multiplier
    # Sub-models are trained on played-only data → predict E[component | plays]
    if "sub_goals" in pos_df.columns:
        pos_df["pts_goals"] = pos_df["p_plays"] * pos_df["sub_goals"] * scoring["goal"]
    else:
        pos_df["pts_goals"] = 0.0

    # Assists
    if "sub_assists" in pos_df.columns:
        pos_df["pts_assists"] = pos_df["p_plays"] * pos_df["sub_assists"] * scoring["assist"]
    else:
        pos_df["pts_assists"] = 0.0

    # Clean sheets (only if player plays 60+)
    if "sub_cs" in pos_df.columns and scoring["cs"] > 0:
        pos_df["pts_cs"] = pos_df["p_60plus"] * pos_df["sub_cs"] * scoring["cs"]
    else:
        pos_df["pts_cs"] = 0.0

    # Goals conceded penalty (GKP/DEF only)
    # Use continuous E[GC]/2 instead of floor(E[GC]/2) — avoids Jensen's
    # inequality bias (Fix 3).
    if "sub_goals_conceded" in pos_df.columns and scoring["gc_per_2"] != 0:
        pos_df["pts_gc"] = pos_df["p_60plus"] * (pos_df["sub_goals_conceded"] / 2) * scoring["gc_per_2"]
    else:
        pos_df["pts_gc"] = 0.0

    # Saves (GKP only) — continuous E[saves]/3 (Fix 3)
    if "sub_saves" in pos_df.columns and scoring["save_per_3"] > 0:
        pos_df["pts_saves"] = pos_df["p_plays"] * (pos_df["sub_saves"] / 3) * scoring["save_per_3"]
    else:
        pos_df["pts_saves"] = 0.0

    # Bonus
    if "sub_bonus" in pos_df.columns:
        pos_df["pts_bonus"] = pos_df["p_plays"] * pos_df["sub_bonus"]
    else:
        pos_df["pts_bonus"] = 0.0

    # Total: gate everything (except maybe bonus) by P(plays)
    pos_df["predicted_next_gw_points"] = (
        pos_df["pts_appearance"]
        + pos_df["pts_goals"]
        + pos_df["pts_assists"]
        + pos_df["pts_cs"]
        + pos_df["pts_gc"]
        + pos_df["pts_saves"]
        + pos_df["pts_bonus"]
    ).clip(lower=0)

    # Soft calibration cap: when component predictions stack up (especially for
    # GKPs where CS + saves + bonus can inflate), dampen extreme values.
    # Above the threshold, predictions are blended 50/50 with the cap to avoid
    # hard clipping while still pulling back unrealistic expectations.
    _position_soft_cap = {"GKP": 7.0, "DEF": 8.0, "MID": 10.0, "FWD": 10.0}
    cap = _position_soft_cap.get(position, 10.0)
    pred_col = "predicted_next_gw_points"
    over = pos_df[pred_col] > cap
    if over.any():
        pos_df.loc[over, pred_col] = cap + (pos_df.loc[over, pred_col] - cap) * 0.5

    # DGW: sum per-fixture predictions
    if pos_df.duplicated(subset=["player_id"], keep=False).any():
        pred_col = "predicted_next_gw_points"
        agg_pred = pos_df.groupby("player_id")[pred_col].sum()
        # Also sum component predictions for diagnostics
        sub_cols = [c for c in pos_df.columns if c.startswith("sub_") or c.startswith("pts_")]
        meta_cols = [c for c in pos_df.columns if c not in sub_cols + [pred_col]]
        deduped = pos_df[meta_cols].drop_duplicates(subset=["player_id"], keep="first")
        deduped = deduped.set_index("player_id")
        deduped[pred_col] = agg_pred
        pos_df = deduped.reset_index()

    return pos_df


if __name__ == "__main__":
    from src.data_fetcher import load_all_data
    from src.feature_engineering import build_features, get_feature_columns

    data = load_all_data()
    df = build_features(data)
    results = train_all_models(df, tune=False)

    print("\n=== Training Summary ===")
    for r in results:
        print(f"  {r['position']:3s} / {r['target']:18s}: MAE = {r['mae']:.3f}")
