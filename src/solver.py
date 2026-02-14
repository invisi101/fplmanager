"""MILP solvers for FPL squad selection and transfer optimization."""

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp


def scrub_nan(records: list[dict]) -> list[dict]:
    """Replace NaN/inf with None in a list of dicts for valid JSON."""
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
    return records


def solve_milp_team(
    player_df: pd.DataFrame, target_col: str,
    budget: float = 1000.0, team_cap: int = 3,
    captain_col: str | None = None,
) -> dict | None:
    """Solve two-tier MILP for optimal squad selection.

    When captain_col is provided, jointly optimizes captain selection
    alongside squad/starter decisions (3n variables instead of 2n).

    Returns {"starters": [...], "bench": [...], "total_cost": float,
    "starting_points": float, "captain_id": int | None} or None on failure.
    """
    from scipy.optimize import Bounds as ScipyBounds

    required = ["position", "cost", target_col]
    if not all(c in player_df.columns for c in required):
        return None

    df = player_df.dropna(subset=required).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None

    pred = df[target_col].values.astype(float)

    # Check if captain optimization is possible
    use_captain = captain_col and captain_col in df.columns
    if use_captain:
        captain_scores = df[captain_col].values.astype(float)
        captain_bonus = captain_scores - pred  # Extra value from captaining
        captain_bonus = np.maximum(captain_bonus, 0)  # Only positive bonus

    SUB_WEIGHT = 0.1
    if use_captain:
        # 3n variables: x_i (squad), s_i (starter), c_i (captain)
        c_obj = np.concatenate([
            -SUB_WEIGHT * pred,
            -(1 - SUB_WEIGHT) * pred,
            -captain_bonus,
        ])
        integrality = np.ones(3 * n)
        nvars = 3 * n
    else:
        c_obj = np.concatenate([
            -SUB_WEIGHT * pred,
            -(1 - SUB_WEIGHT) * pred,
        ])
        integrality = np.ones(2 * n)
        nvars = 2 * n

    A_rows = []
    lbs = []
    ubs = []

    def add_constraint(coeffs, lb, ub):
        # Pad to nvars if needed
        if len(coeffs) < nvars:
            coeffs = np.concatenate([coeffs, np.zeros(nvars - len(coeffs))])
        A_rows.append(coeffs)
        lbs.append(lb)
        ubs.append(ub)

    zeros = np.zeros(n)
    costs = df["cost"].values.astype(float)

    add_constraint(np.concatenate([costs, zeros]), 0, budget)

    squad_req = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, count in squad_req.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(np.concatenate([pos_mask, zeros]), count, count)

    if "team_code" in df.columns:
        for tc in df["team_code"].unique():
            team_mask = (df["team_code"] == tc).astype(float).values
            add_constraint(np.concatenate([team_mask, zeros]), 0, team_cap)

    add_constraint(np.concatenate([zeros, np.ones(n)]), 11, 11)

    start_min = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    for pos, (lo, hi) in start_min.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(np.concatenate([zeros, pos_mask]), lo, hi)

    for i in range(n):
        row = np.zeros(nvars)
        row[i] = -1.0      # -x_i
        row[n + i] = 1.0   # +s_i
        A_rows.append(row)
        lbs.append(-np.inf)
        ubs.append(0)

    # Captain constraints
    if use_captain:
        # sum(c_i) == 1
        cap_sum = np.zeros(nvars)
        cap_sum[2 * n:] = 1.0
        A_rows.append(cap_sum)
        lbs.append(1)
        ubs.append(1)

        # c_i <= s_i (captain must be a starter)
        for i in range(n):
            row = np.zeros(nvars)
            row[n + i] = -1.0      # -s_i
            row[2 * n + i] = 1.0   # +c_i
            A_rows.append(row)
            lbs.append(-np.inf)
            ubs.append(0)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, lbs, ubs)
    variable_bounds = ScipyBounds(lb=0, ub=1)

    result = milp(c_obj, integrality=integrality, bounds=variable_bounds, constraints=constraints)

    if not result.success:
        return None

    x_vals = result.x[:n]
    s_vals = result.x[n:2 * n]
    squad_mask = x_vals > 0.5
    starter_mask = s_vals > 0.5

    captain_id = None
    if use_captain:
        c_vals = result.x[2 * n:]
        cap_idx = np.where(c_vals > 0.5)[0]
        if len(cap_idx) > 0 and "player_id" in df.columns:
            captain_id = int(df.iloc[cap_idx[0]]["player_id"])

    team_df = df[squad_mask].copy()
    team_df["starter"] = starter_mask[squad_mask]

    float_cols = team_df.select_dtypes(include="float").columns
    team_df[float_cols] = team_df[float_cols].round(2)

    starters = team_df[team_df["starter"]]
    bench = team_df[~team_df["starter"]]

    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    team_df["_pos_order"] = team_df["position"].map(pos_order)
    team_df = team_df.sort_values(
        ["starter", "_pos_order", target_col], ascending=[False, True, False],
    )
    team_df = team_df.drop(columns=["_pos_order"])

    return {
        "starters": scrub_nan(starters.to_dict(orient="records")),
        "bench": scrub_nan(bench.to_dict(orient="records")),
        "total_cost": round(team_df["cost"].sum(), 1),
        "starting_points": round(starters[target_col].sum(), 2),
        "players": scrub_nan(team_df.to_dict(orient="records")),
        "captain_id": captain_id,
    }


def solve_transfer_milp(
    player_df: pd.DataFrame,
    current_player_ids: set[int],
    target_col: str,
    budget: float = 1000.0,
    max_transfers: int = 2,
    team_cap: int = 3,
    captain_col: str | None = None,
) -> dict | None:
    """Solve MILP for optimal squad reachable via at most max_transfers changes.

    Identical to solve_milp_team() but adds one extra constraint:
    at least (15 - max_transfers) players must come from the current squad.

    When captain_col is provided, jointly optimizes captain selection.
    """
    from scipy.optimize import Bounds as ScipyBounds

    required = ["position", "cost", target_col]
    if not all(c in player_df.columns for c in required):
        return None

    df = player_df.dropna(subset=required).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None

    pred = df[target_col].values.astype(float)

    # Mark which players are in the current squad
    is_current = df["player_id"].isin(current_player_ids).astype(float).values

    # Check if captain optimization is possible
    use_captain = captain_col and captain_col in df.columns
    if use_captain:
        captain_scores = df[captain_col].values.astype(float)
        captain_bonus = captain_scores - pred
        captain_bonus = np.maximum(captain_bonus, 0)

    SUB_WEIGHT = 0.1
    if use_captain:
        c_obj = np.concatenate([
            -SUB_WEIGHT * pred,
            -(1 - SUB_WEIGHT) * pred,
            -captain_bonus,
        ])
        integrality = np.ones(3 * n)
        nvars = 3 * n
    else:
        c_obj = np.concatenate([
            -SUB_WEIGHT * pred,
            -(1 - SUB_WEIGHT) * pred,
        ])
        integrality = np.ones(2 * n)
        nvars = 2 * n

    A_rows = []
    lbs = []
    ubs = []

    def add_constraint(coeffs, lb, ub):
        if len(coeffs) < nvars:
            coeffs = np.concatenate([coeffs, np.zeros(nvars - len(coeffs))])
        A_rows.append(coeffs)
        lbs.append(lb)
        ubs.append(ub)

    zeros = np.zeros(n)
    costs = df["cost"].values.astype(float)

    # Budget
    add_constraint(np.concatenate([costs, zeros]), 0, budget)

    # Position counts (squad)
    squad_req = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, count in squad_req.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(np.concatenate([pos_mask, zeros]), count, count)

    # Max 3 from same team
    if "team_code" in df.columns:
        for tc in df["team_code"].unique():
            team_mask = (df["team_code"] == tc).astype(float).values
            add_constraint(np.concatenate([team_mask, zeros]), 0, team_cap)

    # Exactly 11 starters
    add_constraint(np.concatenate([zeros, np.ones(n)]), 11, 11)

    # Formation constraints
    start_min = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    for pos, (lo, hi) in start_min.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(np.concatenate([zeros, pos_mask]), lo, hi)

    # s_i <= x_i (can only start if in squad)
    for i in range(n):
        row = np.zeros(nvars)
        row[i] = -1.0
        row[n + i] = 1.0
        A_rows.append(row)
        lbs.append(-np.inf)
        ubs.append(0)

    # TRANSFER CONSTRAINT: keep at least (15 - max_transfers) current players
    keep_min = max(0, 15 - max_transfers)
    add_constraint(np.concatenate([is_current, zeros]), keep_min, 15)

    # Captain constraints
    if use_captain:
        # sum(c_i) == 1
        cap_sum = np.zeros(nvars)
        cap_sum[2 * n:] = 1.0
        A_rows.append(cap_sum)
        lbs.append(1)
        ubs.append(1)

        # c_i <= s_i
        for i in range(n):
            row = np.zeros(nvars)
            row[n + i] = -1.0
            row[2 * n + i] = 1.0
            A_rows.append(row)
            lbs.append(-np.inf)
            ubs.append(0)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, lbs, ubs)
    variable_bounds = ScipyBounds(lb=0, ub=1)

    result = milp(c_obj, integrality=integrality, bounds=variable_bounds, constraints=constraints)

    if not result.success:
        return None

    x_vals = result.x[:n]
    s_vals = result.x[n:2 * n]
    squad_mask = x_vals > 0.5
    starter_mask = s_vals > 0.5

    captain_id = None
    if use_captain:
        c_vals = result.x[2 * n:]
        cap_idx = np.where(c_vals > 0.5)[0]
        if len(cap_idx) > 0 and "player_id" in df.columns:
            captain_id = int(df.iloc[cap_idx[0]]["player_id"])

    team_df = df[squad_mask].copy()
    team_df["starter"] = starter_mask[squad_mask]

    float_cols = team_df.select_dtypes(include="float").columns
    team_df[float_cols] = team_df[float_cols].round(2)

    # Sort by starter first, then position, then predicted points
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    team_df["_pos_order"] = team_df["position"].map(pos_order)
    team_df = team_df.sort_values(
        ["starter", "_pos_order", target_col], ascending=[False, True, False],
    )
    team_df = team_df.drop(columns=["_pos_order"])

    new_squad_ids = set(team_df["player_id"].tolist())
    transfers_out_ids = current_player_ids - new_squad_ids
    transfers_in_ids = new_squad_ids - current_player_ids

    starters = team_df[team_df["starter"]]
    bench = team_df[~team_df["starter"]]

    return {
        "starters": scrub_nan(starters.to_dict(orient="records")),
        "bench": scrub_nan(bench.to_dict(orient="records")),
        "players": scrub_nan(team_df.to_dict(orient="records")),
        "total_cost": round(team_df["cost"].sum(), 1),
        "starting_points": round(starters[target_col].sum(), 2),
        "transfers_in_ids": transfers_in_ids,
        "transfers_out_ids": transfers_out_ids,
        "captain_id": captain_id,
    }
