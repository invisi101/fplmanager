"""Fetch and cache data from FPL-Core-Insights GitHub repo and the FPL API."""

import io
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent
else:
    _BASE = Path(__file__).resolve().parent.parent

CACHE_DIR = _BASE / "cache"
CACHE_MAX_AGE_SECONDS = 6 * 3600  # 6 hours (static GitHub data)
API_CACHE_MAX_AGE_SECONDS = 30 * 60  # 30 minutes (FPL API — injury/price changes)

GITHUB_BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"

LAYOUT_FLAT = "flat"
LAYOUT_PER_GW = "per_gw"

# Only 2024-2025 uses the flat layout. All others use per-GW.
FLAT_LAYOUT_SEASONS = {"2024-2025"}

FLAT_FILES = {
    "players": "players/players.csv",
    "playermatchstats": "playermatchstats/playermatchstats.csv",
    "matches": "matches/matches.csv",
    "playerstats": "playerstats/playerstats.csv",
    "teams": "teams/teams.csv",
}

PER_GW_ROOT_FILES = {
    "players": "players.csv",
    "playerstats": "playerstats.csv",
    "teams": "teams.csv",
}

PER_GW_GW_FILES = ["playermatchstats.csv", "matches.csv"]

FPL_API_BASE = "https://fantasy.premierleague.com/api"
FPL_API_ENDPOINTS = {
    "bootstrap": f"{FPL_API_BASE}/bootstrap-static/",
    "fixtures": f"{FPL_API_BASE}/fixtures/",
}


def get_season_layout(season: str) -> str:
    """Return 'flat' for 2024-2025, 'per_gw' for everything else."""
    return LAYOUT_FLAT if season in FLAT_LAYOUT_SEASONS else LAYOUT_PER_GW


def detect_current_season(bootstrap: dict | None = None) -> str:
    """Detect the current FPL season from bootstrap GW1 deadline_time, or fallback to date."""
    if bootstrap is None:
        # Try cached bootstrap (only when no bootstrap provided, to avoid recursion)
        cache = CACHE_DIR / "fpl_api_bootstrap.json"
        if cache.exists():
            try:
                bootstrap = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass
    if bootstrap:
        events = bootstrap.get("events", [])
        if events:
            deadline = events[0].get("deadline_time", "")
            if len(deadline) >= 4 and deadline[:4].isdigit():
                y = int(deadline[:4])
                return f"{y}-{y+1}"
    # Date fallback
    from datetime import datetime
    now = datetime.now()
    y = now.year if now.month >= 6 else now.year - 1
    return f"{y}-{y+1}"


EARLIEST_SEASON = "2024-2025"  # Earliest season available on the data source
MAX_SEASONS = 2  # Train on current + previous season only


def get_all_seasons(current: str) -> list[str]:
    """Return seasons to fetch, oldest first. Capped at MAX_SEASONS, floored at EARLIEST_SEASON."""
    current_start = int(current.split("-")[0])
    earliest_start = int(EARLIEST_SEASON.split("-")[0])
    first = max(earliest_start, current_start - MAX_SEASONS + 1)
    return [f"{y}-{y+1}" for y in range(first, current_start + 1)]


def get_previous_season(current: str) -> str:
    """Return the season immediately before the given one."""
    start = int(current.split("-")[0])
    return f"{start-1}-{start}"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _is_cache_fresh(path: Path, max_age: int | None = None) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < (max_age if max_age is not None else CACHE_MAX_AGE_SECONDS)


def _fetch_url(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _fetch_csv(url: str, cache_file: Path, force: bool = False) -> pd.DataFrame:
    if not force and _is_cache_fresh(cache_file):
        return pd.read_csv(cache_file, encoding="utf-8")
    print(f"  Fetching {url}")
    try:
        resp = _fetch_url(url)
        # Bug 60 fix: validate as CSV before writing to cache to prevent
        # cache poisoning from non-CSV responses (e.g. rate-limit HTML pages)
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or len(df.columns) < 2:
            raise ValueError(f"Response does not look like valid CSV ({len(df.columns)} cols)")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(resp.text, encoding="utf-8")
        return df
    except (requests.RequestException, OSError, ValueError) as e:
        if cache_file.exists():
            print(f"  Warning: fetch failed ({e}), using stale cache for {cache_file.name}")
            return pd.read_csv(cache_file, encoding="utf-8")
        raise


def fetch_fpl_api(endpoint: str, force: bool = False) -> dict:
    """Fetch JSON from the FPL API, caching locally (30min TTL)."""
    cache_file = _cache_path(f"fpl_api_{endpoint}.json")
    if not force and _is_cache_fresh(cache_file, max_age=API_CACHE_MAX_AGE_SECONDS):
        return json.loads(cache_file.read_text(encoding="utf-8"))
    url = FPL_API_ENDPOINTS[endpoint]
    print(f"  Fetching {url}")
    try:
        resp = _fetch_url(url)
        data = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (requests.RequestException, OSError) as e:
        if cache_file.exists():
            print(f"  Warning: fetch failed ({e}), using stale cache for {cache_file.name}")
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise
    return data


_manager_cache: dict[str, tuple] = {}
_MANAGER_CACHE_TTL = 60  # 60 seconds
_manager_cache_lock = threading.Lock()


def _cached_manager_fetch(cache_key: str, fetch_fn):
    """Simple TTL cache for manager API calls (thread-safe with stale fallback)."""
    now = time.time()
    with _manager_cache_lock:
        if cache_key in _manager_cache:
            data, ts = _manager_cache[cache_key]
            if now - ts < _MANAGER_CACHE_TTL:
                return data
    try:
        data = fetch_fn()
    except requests.RequestException:
        # Bug 62 fix: fall back to stale cached data on network error
        with _manager_cache_lock:
            if cache_key in _manager_cache:
                return _manager_cache[cache_key][0]
        raise
    with _manager_cache_lock:
        _manager_cache[cache_key] = (data, now)
        # Bug 63 fix: evict stale entries when cache grows too large
        if len(_manager_cache) > 200:
            stale = [k for k, (_, ts) in _manager_cache.items()
                     if now - ts > _MANAGER_CACHE_TTL]
            for k in stale:
                del _manager_cache[k]
    return data


def fetch_manager_entry(manager_id: int) -> dict:
    """Fetch manager overview (name, bank, value, current_event)."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/"
    return _cached_manager_fetch(f"entry_{manager_id}", lambda: _fetch_url(url).json())


def fetch_manager_picks(manager_id: int, event: int) -> dict:
    """Fetch manager's 15 picks for a gameweek."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/event/{event}/picks/"
    return _cached_manager_fetch(f"picks_{manager_id}_{event}", lambda: _fetch_url(url).json())


def fetch_manager_history(manager_id: int) -> dict:
    """Fetch per-GW history (transfers, chips) for FT calculation."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/history/"
    return _cached_manager_fetch(f"history_{manager_id}", lambda: _fetch_url(url).json())


def fetch_manager_transfers(manager_id: int) -> list[dict]:
    """Fetch all transfers made by a manager this season."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/transfers/"
    return _cached_manager_fetch(f"transfers_{manager_id}", lambda: _fetch_url(url).json())


def _detect_max_gw(season: str, force: bool = False) -> int:
    """Detect the latest available gameweek for a per-GW layout season."""
    cache_file = _cache_path(f"{season}_playerstats.csv")
    url = f"{GITHUB_BASE}/{season}/playerstats.csv"
    df = _fetch_csv(url, cache_file, force=force)
    if df.empty or "gw" not in df.columns:
        return 0
    max_gw = df["gw"].max()
    if pd.isna(max_gw):
        return 0
    return int(max_gw)


def fetch_season_data(season: str, force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch data for any season, auto-detecting layout. Returns empty dict if season doesn't exist."""
    layout = get_season_layout(season)
    data = {}

    # Probe: try the first file to check if this season exists on the data source
    if layout == LAYOUT_FLAT:
        probe_path = list(FLAT_FILES.values())[0]
    else:
        probe_path = list(PER_GW_ROOT_FILES.values())[0]
    probe_url = f"{GITHUB_BASE}/{season}/{probe_path}"
    probe_cache = _cache_path(f"{season}_probe.csv")
    try:
        _fetch_csv(probe_url, probe_cache, force=force)
    except requests.RequestException:
        return {}

    if layout == LAYOUT_FLAT:
        for key, path in FLAT_FILES.items():
            url = f"{GITHUB_BASE}/{season}/{path}"
            cache_file = _cache_path(f"{season}_{key}.csv")
            try:
                data[key] = _fetch_csv(url, cache_file, force=force)
            except requests.RequestException as e:
                print(f"  Warning: could not fetch {season}/{key}: {e}")
    else:
        # Detect max GW first (also caches playerstats.csv)
        max_gw = _detect_max_gw(season, force=force)
        print(f"  {season}: detected {max_gw} gameweeks")

        for key, path in PER_GW_ROOT_FILES.items():
            url = f"{GITHUB_BASE}/{season}/{path}"
            cache_file = _cache_path(f"{season}_{key}.csv")
            try:
                data[key] = _fetch_csv(url, cache_file, force=force)
            except requests.RequestException as e:
                print(f"  Warning: could not fetch {season}/{key}: {e}")

        for filename in PER_GW_GW_FILES:
            key = filename.replace(".csv", "")
            frames = []
            for gw in range(1, max_gw + 1):
                url = f"{GITHUB_BASE}/{season}/By Gameweek/GW{gw}/{filename}"
                cache_file = _cache_path(f"{season}_gw{gw}_{filename}")
                try:
                    df = _fetch_csv(url, cache_file, force=force)
                    if "gameweek" not in df.columns:
                        df["gameweek"] = gw
                    frames.append(df)
                except requests.RequestException:
                    pass
            if frames:
                data[key] = pd.concat(frames, ignore_index=True)

    return data


def fetch_event_live(event: int, force: bool = False) -> dict:
    """Fetch per-player live stats for a specific gameweek."""
    cache_file = _cache_path(f"fpl_api_event_{event}_live.json")
    if not force and _is_cache_fresh(cache_file, max_age=API_CACHE_MAX_AGE_SECONDS):
        return json.loads(cache_file.read_text(encoding="utf-8"))
    url = f"{FPL_API_BASE}/event/{event}/live/"
    print(f"  Fetching {url}")
    try:
        resp = _fetch_url(url)
        data = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (requests.RequestException, OSError) as e:
        if cache_file.exists():
            print(f"  Warning: fetch failed ({e}), using stale cache")
            return json.loads(cache_file.read_text(encoding="utf-8"))
        raise
    return data


def load_all_data(force: bool = False) -> dict:
    """Main entry point: fetch everything.

    Returns dict keyed by season label (e.g. '2024-2025') plus:
        'api'            -> dict with 'bootstrap' and 'fixtures'
        'current_season' -> str like '2025-2026'
        'seasons'        -> list of all seasons, oldest first
    """
    print("Fetching FPL API data...")
    api_data = {ep: fetch_fpl_api(ep, force=force) for ep in FPL_API_ENDPOINTS}

    current = detect_current_season(api_data.get("bootstrap"))
    seasons = get_all_seasons(current)

    result = {"api": api_data, "current_season": current, "seasons": seasons}

    for season in seasons:
        print(f"Fetching {season} data...")
        try:
            sdata = fetch_season_data(season, force=force)
            if not sdata:
                print(f"  {season}: no data available, skipping")
            result[season] = sdata
        except Exception as e:
            print(f"  Skipping {season}: {e}")
            result[season] = {}

    print("Data loading complete.")
    return result
