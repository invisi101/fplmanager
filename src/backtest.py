"""Backtest module: evaluate model accuracy against historical gameweeks."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import ndcg_score
from xgboost import XGBRegressor

from src.model import (
    CURRENT_SEASON,
    DEFAULT_FEATURES,
    FPL_SCORING,
    POSITION_GROUPS,
    SUB_MODEL_COMPONENTS,
    SUB_MODEL_FEATURES,
    SUB_MODEL_OBJECTIVES,
    SUB_MODELS_FOR_POSITION,
    _prepare_position_data,
    _season_weight,
    predict_for_position,
)


def _bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Return (lower, upper) bootstrap confidence interval for the mean."""
    values = np.array(values, dtype=float)
    if len(values) < 2:
        m = float(np.mean(values))
        return (m, m)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return (round(lower, 4), round(upper, 4))


def _train_backtest_model(
    train_df: pd.DataFrame, position: str, target: str = "next_gw_points",
    quantile_alpha: float | None = None,
) -> dict | None:
    """Train a lightweight XGBoost model for backtesting (no tuning/grid search).

    Uses only the provided train_df — no future data leakage.
    When quantile_alpha is set, trains a quantile regression model instead
    of the default mean regression model.
    Returns dict with 'model' and 'features', or None if insufficient data.
    """
    feature_cols = DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"])
    pos_df, available_feats = _prepare_position_data(train_df, position, target, feature_cols)

    if len(pos_df) < 50:
        return None

    # Season-based sample weights
    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    X = pos_df[available_feats].values
    y = pos_df[target].values
    w = pos_df["_sample_weight"].values

    if quantile_alpha is not None:
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile_alpha,
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
    model.fit(X, y, sample_weight=w)

    return {"model": model, "features": available_feats}


def _train_backtest_sub_models(
    train_df: pd.DataFrame, position: str,
) -> dict[str, dict]:
    """Train all decomposed sub-models for backtesting.

    Returns dict mapping component name -> model_dict, or empty dict on failure.
    """
    components = SUB_MODELS_FOR_POSITION.get(position, [])
    models = {}
    for comp in components:
        target = SUB_MODEL_COMPONENTS[comp]
        feature_cols = SUB_MODEL_FEATURES.get(comp, [])
        pos_df, available_feats = _prepare_position_data(
            train_df, position, target, feature_cols,
        )
        # Train only on rows where the player actually played
        if "next_gw_minutes" in pos_df.columns:
            pos_df = pos_df[pos_df["next_gw_minutes"] > 0].copy()
        if len(pos_df) < 30:
            continue
        pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))
        X = pos_df[available_feats].values
        y = pos_df[target].values
        w = pos_df["_sample_weight"].values

        objective = SUB_MODEL_OBJECTIVES.get(comp, "reg:squarederror")
        obj_params = {"objective": objective}
        if objective == "binary:logistic":
            obj_params["eval_metric"] = "logloss"

        model = XGBRegressor(
            **obj_params,
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(X, y, sample_weight=w)
        models[comp] = {"model": model, "features": available_feats}
    return models


def _predict_decomposed_backtest(
    snapshot: pd.DataFrame, position: str,
    sub_models: dict[str, dict],
) -> pd.DataFrame:
    """Generate decomposed predictions for backtesting using provided models.

    Same logic as model.predict_decomposed but uses in-memory model dicts
    instead of loading from disk.
    """
    scoring = FPL_SCORING[position]
    pos_df = snapshot[snapshot["position_clean"] == position].copy()
    if pos_df.empty:
        return pd.DataFrame()

    _fill_defaults = {
        "opponent_elo": 1500.0, "fdr": 3.0, "avg_fdr_next3": 3.0,
        "avg_opponent_elo_next3": 1500.0,
    }

    for comp, model_dict in sub_models.items():
        model = model_dict["model"]
        features = model_dict["features"]
        available = [c for c in features if c in pos_df.columns]
        if not available:
            continue
        for c in available:
            pos_df[c] = pos_df[c].fillna(_fill_defaults.get(c, 0))
        X = np.zeros((len(pos_df), len(features)))
        for i, f in enumerate(features):
            if f in pos_df.columns:
                X[:, i] = pos_df[f].values
        pos_df[f"sub_{comp}"] = model.predict(X).clip(min=0)

    # Combine using FPL rules (same logic as model.predict_decomposed)
    # P(plays) = chance_of_playing/100 * availability_rate_last5
    cop = pos_df["chance_of_playing"].fillna(0) / 100.0 if "chance_of_playing" in pos_df.columns else 0.8
    avail = pos_df["availability_rate_last5"].fillna(0) if "availability_rate_last5" in pos_df.columns else 1.0
    pos_df["p_plays"] = (cop * avail).clip(0, 1)
    pos_df["p_60plus"] = pos_df["p_plays"] * 0.85

    pos_df["predicted_next_gw_points"] = (
        pos_df["p_60plus"] * 2 + (pos_df["p_plays"] - pos_df["p_60plus"]).clip(lower=0) * 1
    )

    # Sub-models trained on played-only -> predict E[component | plays]
    # Gate by P(plays) to get unconditional expectation
    if "sub_goals" in pos_df.columns:
        pos_df["predicted_next_gw_points"] += (
            pos_df["p_plays"] * pos_df["sub_goals"] * scoring["goal"]
        )
    if "sub_assists" in pos_df.columns:
        pos_df["predicted_next_gw_points"] += (
            pos_df["p_plays"] * pos_df["sub_assists"] * scoring["assist"]
        )
    if "sub_cs" in pos_df.columns and scoring["cs"] > 0:
        pos_df["predicted_next_gw_points"] += (
            pos_df["p_60plus"] * pos_df["sub_cs"] * scoring["cs"]
        )
    if "sub_goals_conceded" in pos_df.columns and scoring["gc_per_2"] != 0:
        # Continuous E[GC]/2 instead of floor — avoids Jensen's inequality bias (Fix 3)
        pos_df["predicted_next_gw_points"] += (
            pos_df["p_60plus"] * (pos_df["sub_goals_conceded"] / 2) * scoring["gc_per_2"]
        )
    if "sub_saves" in pos_df.columns and scoring["save_per_3"] > 0:
        # Continuous E[saves]/3 instead of floor (Fix 3)
        pos_df["predicted_next_gw_points"] += (
            pos_df["p_plays"] * (pos_df["sub_saves"] / 3) * scoring["save_per_3"]
        )
    if "sub_bonus" in pos_df.columns:
        pos_df["predicted_next_gw_points"] += pos_df["p_plays"] * pos_df["sub_bonus"]

    pos_df["predicted_next_gw_points"] = pos_df["predicted_next_gw_points"].clip(lower=0)

    # DGW: sum per-fixture
    if pos_df.duplicated(subset=["player_id"], keep=False).any():
        agg_pred = pos_df.groupby("player_id")["predicted_next_gw_points"].sum()
        meta_cols = [c for c in pos_df.columns
                     if not c.startswith("sub_") and not c.startswith("pts_")
                     and c != "predicted_next_gw_points"
                     and c not in ("p_60plus", "p_plays")]
        deduped = pos_df[meta_cols].drop_duplicates(subset=["player_id"], keep="first")
        deduped = deduped.set_index("player_id")
        deduped["predicted_next_gw_points"] = agg_pred
        pos_df = deduped.reset_index()

    return pos_df


def predict_single_gw(
    df: pd.DataFrame, predict_gw: int, season: str, print_fn=print,
) -> pd.DataFrame | None:
    """Run model prediction for a single gameweek (walk-forward, no leakage).

    Returns a DataFrame with columns: player_id, predicted_next_gw_points,
    actual, position, web_name, cost, team_code.  Returns None on failure.
    """
    season_df = df[df["season"] == season]
    available_gws = sorted(season_df["gameweek"].unique())
    season_year = int(season.split("-")[0])

    snapshot_gw = predict_gw - 1
    if snapshot_gw not in available_gws or predict_gw not in available_gws:
        return None

    snapshot = season_df[season_df["gameweek"] == snapshot_gw].copy()
    if snapshot.empty:
        return None

    # Actual points for predict_gw
    actuals_df = (
        season_df[season_df["gameweek"] == predict_gw][["player_id", "event_points"]]
        .drop_duplicates(subset="player_id", keep="first")
        .copy()
    )
    if actuals_df.empty:
        return None

    actuals_dict = dict(zip(actuals_df["player_id"], actuals_df["event_points"]))

    # Train fresh models using only data up to (but NOT including) snapshot_gw
    train_df = df[
        (df["season"].apply(lambda s: int(s.split("-")[0])) < season_year) |
        ((df["season"] == season) & (df["gameweek"] < snapshot_gw))
    ].copy()

    # Decomposed sub-model predictions
    all_preds = []
    for pos in POSITION_GROUPS:
        sub_models = _train_backtest_sub_models(train_df, pos)
        components = SUB_MODELS_FOR_POSITION.get(pos, [])
        if sub_models and len(sub_models) == len(components):
            preds = _predict_decomposed_backtest(snapshot, pos, sub_models)
        else:
            model_dict = _train_backtest_model(train_df, pos, "next_gw_points")
            if model_dict is None:
                continue
            preds = predict_for_position(
                snapshot, pos, "next_gw_points", model_dict,
            )
        if not preds.empty:
            all_preds.append(
                preds[["player_id", "predicted_next_gw_points"]].copy()
            )

    if not all_preds:
        return None

    pred_df = pd.concat(all_preds)

    pred_df["actual"] = pred_df["player_id"].map(actuals_dict)
    pred_df = pred_df.dropna(subset=["actual"])
    if pred_df.empty:
        return None

    # Map metadata from snapshot
    deduped = snapshot.drop_duplicates("player_id")
    for col, default in [("position_clean", "MID"), ("web_name", "?")]:
        col_map = dict(zip(deduped["player_id"], deduped[col] if col in deduped.columns else pd.Series()))
        target_col = "position" if col == "position_clean" else col
        pred_df[target_col] = pred_df["player_id"].map(col_map).fillna(default)

    if "now_cost" in deduped.columns:
        cost_map = dict(zip(deduped["player_id"], deduped["now_cost"].fillna(0)))
        pred_df["cost"] = pred_df["player_id"].map(cost_map).fillna(0) / 10.0
    elif "cost" in deduped.columns:
        cost_map = dict(zip(deduped["player_id"], deduped["cost"].fillna(0)))
        pred_df["cost"] = pred_df["player_id"].map(cost_map).fillna(0)
    else:
        pred_df["cost"] = 0.0

    if "team_code" in deduped.columns:
        tc_map = dict(zip(deduped["player_id"], deduped["team_code"]))
        pred_df["team_code"] = pred_df["player_id"].map(tc_map)

    return pred_df


def _run_season_backtest(
    df: pd.DataFrame,
    start_gw: int,
    end_gw: int,
    season: str,
    print_fn=print,
) -> list[dict]:
    """Run walk-forward backtest for a single season, returning raw GW results.

    For each gameweek in [start_gw, end_gw], trains a fresh model using
    only data up to the previous GW (no data leakage), then predicts the
    target GW.
    """
    season_df = df[df["season"] == season]
    available_gws = sorted(season_df["gameweek"].unique())
    season_year = int(season.split("-")[0])

    gameweek_results = []

    for predict_gw in range(start_gw, end_gw + 1):
        snapshot_gw = predict_gw - 1
        if snapshot_gw not in available_gws or predict_gw not in available_gws:
            continue

        snapshot = season_df[season_df["gameweek"] == snapshot_gw].copy()
        if snapshot.empty:
            continue

        # Actual points for predict_gw
        # Deduplicate first — DGW players have multiple rows but
        # event_points already contains the total for the whole GW.
        actuals_df = (
            season_df[season_df["gameweek"] == predict_gw][["player_id", "event_points"]]
            .drop_duplicates(subset="player_id", keep="first")
            .copy()
        )
        if actuals_df.empty:
            continue

        actuals_dict = dict(zip(actuals_df["player_id"], actuals_df["event_points"]))

        # --- Train fresh models using only data up to (but NOT including)
        # snapshot_gw, because snapshot_gw's target (next_gw_points) IS the
        # predict_gw actual points — including it would leak the answer.
        # For multi-season: only include seasons <= current season (not future).
        train_df = df[
            (df["season"].apply(lambda s: int(s.split("-")[0])) < season_year) |
            ((df["season"] == season) & (df["gameweek"] < snapshot_gw))
        ].copy()

        # --- Decomposed sub-model predictions ---
        all_preds = []
        for pos in POSITION_GROUPS:
            sub_models = _train_backtest_sub_models(train_df, pos)
            components = SUB_MODELS_FOR_POSITION.get(pos, [])
            if sub_models and len(sub_models) == len(components):
                preds = _predict_decomposed_backtest(snapshot, pos, sub_models)
            else:
                # Fall back to single model
                model_dict = _train_backtest_model(train_df, pos, "next_gw_points")
                if model_dict is None:
                    continue
                preds = predict_for_position(
                    snapshot, pos, "next_gw_points", model_dict,
                )
            if not preds.empty:
                all_preds.append(
                    preds[["player_id", "predicted_next_gw_points"]].copy()
                )

        if not all_preds:
            continue

        pred_df = pd.concat(all_preds)

        # --- Quantile predictions for captain scoring (MID/FWD only) ---
        q80_preds = []
        for pos in ["MID", "FWD"]:
            q_model_dict = _train_backtest_model(
                train_df, pos, "next_gw_points", quantile_alpha=0.80
            )
            if q_model_dict is None:
                continue
            q_preds = predict_for_position(
                snapshot, pos, "next_gw_points", q_model_dict, suffix="_q80"
            )
            if not q_preds.empty:
                q80_preds.append(
                    q_preds[["player_id", "predicted_next_gw_points_q80"]].copy()
                )

        if q80_preds:
            q80_df = pd.concat(q80_preds)
            pred_df = pred_df.merge(q80_df, on="player_id", how="left")

        # Composite captain score: blend mean + quantile for upside
        if "predicted_next_gw_points_q80" in pred_df.columns:
            pred_df["captain_score"] = (
                0.4 * pred_df["predicted_next_gw_points"]
                + 0.6 * pred_df["predicted_next_gw_points_q80"].fillna(
                    pred_df["predicted_next_gw_points"]
                )
            )
        else:
            pred_df["captain_score"] = pred_df["predicted_next_gw_points"]
        pred_df["actual"] = pred_df["player_id"].map(actuals_dict)
        pred_df = pred_df.dropna(subset=["actual"])

        if pred_df.empty:
            continue

        # --- Baselines ---
        # ep_next from snapshot
        ep_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id")["ep_next"].fillna(0),
            )
        )
        pred_df["ep_next"] = pred_df["player_id"].map(ep_map).fillna(0)

        # Naive form baseline (just use player_form as the prediction)
        form_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id")["player_form"].fillna(0),
            )
        )
        pred_df["form_pred"] = pred_df["player_id"].map(form_map).fillna(0)

        # Position-average baseline from training data
        pos_avg_map = train_df.groupby("position_clean")["next_gw_points"].mean().to_dict()

        # Last 3 GW average baseline
        prior_gws = [g for g in available_gws if g <= snapshot_gw]
        last3_gws = prior_gws[-3:] if len(prior_gws) >= 3 else prior_gws
        last3_df = season_df[season_df["gameweek"].isin(last3_gws)]
        last3_avg = (
            last3_df
            .drop_duplicates(subset=["player_id", "gameweek"], keep="first")
            .groupby("player_id")["event_points"]
            .mean()
            .to_dict()
        )
        pred_df["last3_avg_pred"] = pred_df["player_id"].map(last3_avg).fillna(0)

        # Position info
        pos_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id")["position_clean"],
            )
        )
        pred_df["position"] = pred_df["player_id"].map(pos_map)

        # Name info
        name_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id").get("web_name", pd.Series()),
            )
        )
        pred_df["web_name"] = pred_df["player_id"].map(name_map).fillna("?")

        # --- MAE ---
        model_mae = float(
            np.abs(pred_df["predicted_next_gw_points"] - pred_df["actual"]).mean()
        )
        ep_mae = float(np.abs(pred_df["ep_next"] - pred_df["actual"]).mean())
        form_mae = float(np.abs(pred_df["form_pred"] - pred_df["actual"]).mean())
        last3_mae = float(np.abs(pred_df["last3_avg_pred"] - pred_df["actual"]).mean())

        # Played-only MAE (excludes 0-point players who didn't feature)
        played_mask = pred_df["actual"] > 0
        model_mae_played = float(
            np.abs(
                pred_df.loc[played_mask, "predicted_next_gw_points"]
                - pred_df.loc[played_mask, "actual"]
            ).mean()
        ) if played_mask.any() else model_mae

        # Position-average MAE
        pred_df["pos_avg_pred"] = pred_df["position"].map(pos_avg_map).fillna(2.0)
        pos_avg_mae = float(np.abs(pred_df["pos_avg_pred"] - pred_df["actual"]).mean())

        # --- Ranking metrics ---
        _sp = spearmanr(pred_df["actual"], pred_df["predicted_next_gw_points"]).correlation
        spearman_rho = float(_sp) if not np.isnan(_sp) else 0.0
        # NDCG requires non-negative relevance — shift actuals so min is 0
        actual_shifted = pred_df["actual"].values - pred_df["actual"].values.min()
        ndcg_top20 = float(ndcg_score(
            np.array([actual_shifted]),
            np.array([pred_df["predicted_next_gw_points"].values]),
            k=20,
        ))
        _ep_sp = spearmanr(pred_df["actual"], pred_df["ep_next"]).correlation
        ep_spearman = float(_ep_sp) if not np.isnan(_ep_sp) else 0.0

        # --- Per-position MAE ---
        pos_maes = {}
        for pos in POSITION_GROUPS:
            pos_rows = pred_df[pred_df["position"] == pos]
            if not pos_rows.empty:
                pos_maes[pos] = {
                    "model": float(
                        np.abs(
                            pos_rows["predicted_next_gw_points"] - pos_rows["actual"]
                        ).mean()
                    ),
                    "ep": float(
                        np.abs(pos_rows["ep_next"] - pos_rows["actual"]).mean()
                    ),
                    "form": float(
                        np.abs(pos_rows["form_pred"] - pos_rows["actual"]).mean()
                    ),
                    "last3": float(
                        np.abs(pos_rows["last3_avg_pred"] - pos_rows["actual"]).mean()
                    ),
                    "n_players": len(pos_rows),
                }

        # --- Top 11 comparison ---
        model_top11 = pred_df.nlargest(11, "predicted_next_gw_points")
        ep_top11 = pred_df.nlargest(11, "ep_next")
        form_top11 = pred_df.nlargest(11, "form_pred")
        last3_top11 = pred_df.nlargest(11, "last3_avg_pred")
        actual_top11 = pred_df.nlargest(11, "actual")

        model_pts = float(model_top11["actual"].sum())
        ep_pts = float(ep_top11["actual"].sum())
        form_pts = float(form_top11["actual"].sum())
        last3_pts = float(last3_top11["actual"].sum())
        actual_best = float(actual_top11["actual"].sum())

        actual_ids = set(actual_top11["player_id"])
        model_overlap = int(len(set(model_top11["player_id"]) & actual_ids))
        ep_overlap = int(len(set(ep_top11["player_id"]) & actual_ids))

        # --- Captain pick accuracy ---
        # Captain = highest captain_score (composite of mean + quantile)
        captain = pred_df.nlargest(1, "captain_score").iloc[0]
        actual_top3_ids = set(pred_df.nlargest(3, "actual")["player_id"])
        captain_in_top3 = bool(captain["player_id"] in actual_top3_ids)
        captain_actual_rank = int(
            (pred_df["actual"] > captain["actual"]).sum() + 1
        )

        # Winner
        if model_pts > ep_pts:
            winner = "MODEL"
        elif model_pts < ep_pts:
            winner = "ep_next"
        else:
            winner = "TIE"

        # Capture percentage for this GW
        gw_capture_pct = round((model_pts / actual_best) * 100, 1) if actual_best > 0 else 0

        gw_result = {
            "gw": predict_gw,
            "season": season,
            "n_players": len(pred_df),
            "model_mae": round(model_mae, 3),
            "model_mae_played": round(model_mae_played, 3),
            "ep_mae": round(ep_mae, 3),
            "form_mae": round(form_mae, 3),
            "last3_mae": round(last3_mae, 3),
            "pos_avg_mae": round(pos_avg_mae, 3),
            "spearman_rho": round(spearman_rho, 3),
            "ndcg_top20": round(ndcg_top20, 3),
            "ep_spearman": round(ep_spearman, 3),
            "model_top11_pts": round(model_pts, 1),
            "ep_top11_pts": round(ep_pts, 1),
            "form_top11_pts": round(form_pts, 1),
            "last3_top11_pts": round(last3_pts, 1),
            "actual_best_pts": round(actual_best, 1),
            "model_overlap": model_overlap,
            "ep_overlap": ep_overlap,
            "captain_name": str(captain.get("web_name", "?")),
            "captain_predicted": round(float(captain["predicted_next_gw_points"]), 2),
            "captain_actual": round(float(captain["actual"]), 1),
            "captain_in_top3": captain_in_top3,
            "captain_actual_rank": captain_actual_rank,
            "winner": winner,
            "capture_pct": gw_capture_pct,
            "pos_mae": pos_maes,
        }

        gameweek_results.append(gw_result)
        print_fn(
            f"  [{season}] GW{predict_gw:2d}: MAE m={model_mae:.2f} pl={model_mae_played:.2f} ep={ep_mae:.2f} f={form_mae:.2f} l3={last3_mae:.2f}"
            f" rho={spearman_rho:.2f}"
            f" | Top11 m={model_pts:.0f} ep={ep_pts:.0f} best={actual_best:.0f}"
            f" | Cap: {captain.get('web_name','?')} ({captain['actual']:.0f}pts, rank {captain_actual_rank})"
            f" | {winner}"
        )

    return gameweek_results


def run_backtest(
    df: pd.DataFrame,
    start_gw: int = 5,
    end_gw: int = 25,
    season: str = "",
    seasons: list[str] | None = None,
    print_fn=print,
) -> dict:
    """Run walk-forward backtest with per-GW model retraining.

    For each gameweek in [start_gw, end_gw], trains a fresh model using
    only data up to the previous GW (no data leakage), then predicts the
    target GW. Compares model predictions against FPL's ep_next, a
    naive form baseline, and a last-3-GW average baseline.

    When `seasons` is provided, runs backtest across multiple seasons and
    aggregates results. Default None preserves single-season behavior.

    Returns a dict with summary stats, per-GW results, and per-position breakdown.
    """
    # Determine which seasons to backtest
    if seasons:
        season_list = seasons
    else:
        if not season:
            from src.data_fetcher import detect_current_season
            season = detect_current_season()
        season_list = [season]

    # Collect gameweek results across all seasons
    gameweek_results = []
    for s in season_list:
        print_fn(f"\n  --- Season {s} ---")
        gw_results = _run_season_backtest(df, start_gw, end_gw, s, print_fn)
        gameweek_results.extend(gw_results)

    if not gameweek_results:
        return {"error": "No gameweeks available for backtest."}

    # --- Aggregate summary ---
    n_gws = len(gameweek_results)
    model_wins = sum(1 for r in gameweek_results if r["winner"] == "MODEL")
    ep_wins = sum(1 for r in gameweek_results if r["winner"] == "ep_next")
    ties = sum(1 for r in gameweek_results if r["winner"] == "TIE")

    avg_model_mae = round(np.mean([r["model_mae"] for r in gameweek_results]), 3)
    avg_model_mae_played = round(np.mean([r["model_mae_played"] for r in gameweek_results]), 3)
    avg_ep_mae = round(np.mean([r["ep_mae"] for r in gameweek_results]), 3)
    avg_form_mae = round(np.mean([r["form_mae"] for r in gameweek_results]), 3)
    avg_last3_mae = round(np.mean([r["last3_mae"] for r in gameweek_results]), 3)
    avg_pos_avg_mae = round(np.mean([r["pos_avg_mae"] for r in gameweek_results]), 3)

    avg_spearman = round(np.mean([r["spearman_rho"] for r in gameweek_results]), 3)
    avg_ndcg_top20 = round(np.mean([r["ndcg_top20"] for r in gameweek_results]), 3)
    avg_ep_spearman = round(np.mean([r["ep_spearman"] for r in gameweek_results]), 3)

    avg_model_pts = round(np.mean([r["model_top11_pts"] for r in gameweek_results]), 1)
    avg_ep_pts = round(np.mean([r["ep_top11_pts"] for r in gameweek_results]), 1)
    avg_form_pts = round(np.mean([r["form_top11_pts"] for r in gameweek_results]), 1)
    avg_last3_pts = round(np.mean([r["last3_top11_pts"] for r in gameweek_results]), 1)
    avg_actual_pts = round(
        np.mean([r["actual_best_pts"] for r in gameweek_results]), 1
    )

    avg_model_overlap = round(
        np.mean([r["model_overlap"] for r in gameweek_results]), 1
    )
    avg_ep_overlap = round(
        np.mean([r["ep_overlap"] for r in gameweek_results]), 1
    )

    captain_hit_rate = round(
        sum(1 for r in gameweek_results if r["captain_in_top3"]) / n_gws, 2
    )

    # Paired Wilcoxon signed-rank tests: model vs ep_next
    model_maes = [r["model_mae"] for r in gameweek_results]
    ep_maes = [r["ep_mae"] for r in gameweek_results]
    model_spears = [r["spearman_rho"] for r in gameweek_results]
    ep_spears = [r["ep_spearman"] for r in gameweek_results]

    mae_diffs = [m - e for m, e in zip(model_maes, ep_maes)]
    if len(mae_diffs) >= 6 and any(d != 0 for d in mae_diffs):
        _, mae_pvalue = wilcoxon(mae_diffs, alternative='less')
    else:
        mae_pvalue = float('nan')

    spear_diffs = [m - e for m, e in zip(model_spears, ep_spears)]
    if len(spear_diffs) >= 6 and any(d != 0 for d in spear_diffs):
        _, spear_pvalue = wilcoxon(spear_diffs, alternative='greater')
    else:
        spear_pvalue = float('nan')

    # --- Bootstrap confidence intervals ---
    per_gw_capture_pcts = [r["capture_pct"] for r in gameweek_results]
    per_gw_win_indicators = [1.0 if r["winner"] == "MODEL" else 0.0 for r in gameweek_results]

    model_mae_ci = _bootstrap_ci(model_maes)
    ep_mae_ci = _bootstrap_ci(ep_maes)
    model_top11_pts_ci = _bootstrap_ci([r["model_top11_pts"] for r in gameweek_results])
    capture_pct_ci = _bootstrap_ci(per_gw_capture_pcts)
    win_rate_ci = _bootstrap_ci(per_gw_win_indicators)

    # Per-position aggregate
    pos_summary = {}
    for pos in POSITION_GROUPS:
        pos_maes_list = [
            r["pos_mae"][pos] for r in gameweek_results if pos in r["pos_mae"]
        ]
        if pos_maes_list:
            pos_model_maes = [p["model"] for p in pos_maes_list]
            pos_ep_maes = [p["ep"] for p in pos_maes_list]

            # Per-position Wilcoxon p-value
            pos_mae_diffs = [m - e for m, e in zip(pos_model_maes, pos_ep_maes)]
            if len(pos_mae_diffs) >= 6 and any(d != 0 for d in pos_mae_diffs):
                _, pos_p = wilcoxon(pos_mae_diffs, alternative='less')
                pos_pvalue = round(float(pos_p), 4)
            else:
                pos_pvalue = None

            pos_summary[pos] = {
                "model_mae": round(np.mean(pos_model_maes), 3),
                "ep_mae": round(np.mean(pos_ep_maes), 3),
                "form_mae": round(
                    np.mean([p["form"] for p in pos_maes_list]), 3
                ),
                "last3_mae": round(
                    np.mean([p["last3"] for p in pos_maes_list]), 3
                ),
                "avg_players": round(
                    np.mean([p["n_players"] for p in pos_maes_list]), 0
                ),
                "model_mae_ci": _bootstrap_ci(pos_model_maes),
                "ep_mae_ci": _bootstrap_ci(pos_ep_maes),
                "mae_pvalue": pos_pvalue,
            }

    # Model % of theoretical maximum
    capture_pct = round((avg_model_pts / avg_actual_pts) * 100, 1) if avg_actual_pts > 0 else 0

    summary = {
        "start_gw": start_gw,
        "end_gw": end_gw,
        "seasons": season_list,
        "n_gameweeks": n_gws,
        "model_avg_mae": avg_model_mae,
        "model_avg_mae_played": avg_model_mae_played,
        "ep_avg_mae": avg_ep_mae,
        "form_avg_mae": avg_form_mae,
        "last3_avg_mae": avg_last3_mae,
        "pos_avg_mae": avg_pos_avg_mae,
        "avg_spearman": avg_spearman,
        "avg_ndcg_top20": avg_ndcg_top20,
        "avg_ep_spearman": avg_ep_spearman,
        "model_wins": model_wins,
        "ep_wins": ep_wins,
        "ties": ties,
        "model_avg_top11_pts": avg_model_pts,
        "ep_avg_top11_pts": avg_ep_pts,
        "form_avg_top11_pts": avg_form_pts,
        "last3_avg_top11_pts": avg_last3_pts,
        "actual_avg_top11_pts": avg_actual_pts,
        "model_capture_pct": capture_pct,
        "model_avg_overlap": avg_model_overlap,
        "ep_avg_overlap": avg_ep_overlap,
        "captain_hit_rate": captain_hit_rate,
        "mae_pvalue": round(mae_pvalue, 4) if not np.isnan(mae_pvalue) else None,
        "spearman_pvalue": round(spear_pvalue, 4) if not np.isnan(spear_pvalue) else None,
        "model_mae_ci": list(model_mae_ci),
        "ep_mae_ci": list(ep_mae_ci),
        "model_top11_pts_ci": list(model_top11_pts_ci),
        "capture_pct_ci": list(capture_pct_ci),
        "win_rate_ci": list(win_rate_ci),
    }

    season_label = ", ".join(season_list) if len(season_list) > 1 else season_list[0]
    print_fn(f"\n  === Backtest Summary ({season_label}, GW{start_gw}-{end_gw}) ===")
    print_fn(f"  Model wins: {model_wins}/{n_gws}, ep_next: {ep_wins}/{n_gws}, Ties: {ties}/{n_gws}")
    print_fn(f"  Avg MAE — Model: {avg_model_mae} (played: {avg_model_mae_played}), ep_next: {avg_ep_mae}, Form: {avg_form_mae}, Last3: {avg_last3_mae}, PosAvg: {avg_pos_avg_mae}")
    print_fn(f"  Avg Spearman — Model: {avg_spearman}, ep_next: {avg_ep_spearman} | NDCG@20: {avg_ndcg_top20}")
    print_fn(f"  Avg Top 11 — Model: {avg_model_pts}, ep_next: {avg_ep_pts}, Last3: {avg_last3_pts}, Actual: {avg_actual_pts}")
    print_fn(f"  Model captures {capture_pct}% of theoretical maximum")
    print_fn(f"  Captain in actual top 3: {int(captain_hit_rate * 100)}% of GWs")
    print_fn(f"  95% CIs — MAE: [{model_mae_ci[0]}, {model_mae_ci[1]}], Top11: [{model_top11_pts_ci[0]}, {model_top11_pts_ci[1]}]")
    mae_p_str = f"{mae_pvalue:.4f}" if not np.isnan(mae_pvalue) else "n/a"
    spear_p_str = f"{spear_pvalue:.4f}" if not np.isnan(spear_pvalue) else "n/a"
    print_fn(f"  Significance — MAE p={mae_p_str}, Spearman p={spear_p_str} (Wilcoxon signed-rank, one-sided)")

    return {
        "summary": summary,
        "by_position": pos_summary,
        "gameweeks": gameweek_results,
    }
