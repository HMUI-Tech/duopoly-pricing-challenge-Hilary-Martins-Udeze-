from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque

# =========================
# Config (default values)
# =========================
P_MIN = 10.0
P_MAX = 90.0
W = 80                 # stats window for UCB
K = 31                 # discrete grid size
EPS = 0.05             # epsilon-greedy exploration rate
UCB_C = 1.0            # UCB exploration weight
EPS_DECAY = 0.999      # gentle decay to calm exploration over time
RB_SIZE = 128          # bounded ring buffer capacity (>= W; keep compact)
SAFE_PRIOR_B = -0.5    # prior slope for demand (keeps slope negative)
EPSILON_DEMAND_SLOPE = 1e-6  # guards division-by-zero / wrong-signed b
COST_MARGIN = 0.1      # ensure price >= cost + margin
COLD_START_PERIODS = 3 # safe fixed price at the very beginning
SAFE_DEFAULT_PRICE = (P_MIN + P_MAX) / 2.0

# Pre-computed grid (deterministic; used by bandit policy)
GRID = np.linspace(P_MIN, P_MAX, K)

# =========================
# Helpers
# =========================
# This function is essential to keep prices within their valid range.
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

# This helper function maps a continuous price from my OLS model to the nearest discrete price on the grid.
def nearest_grid_idx(price: float) -> int:
    idx = int(round((price - P_MIN) / (P_MAX - P_MIN) * (K - 1)))
    return int(np.clip(idx, 0, K - 1))

# I use this small rounding helper to keep state tiny and stable across serializations.
def _round4(x: float) -> float:
    return float(round(float(x), 4))

# I record a compact "Decision Card" each round to explain what the policy did.
def _record_decision(
    state: Dict[str, Any],
    *,
    t: int,
    grid_idx: Optional[int],
    grid_price: Optional[float],
    favored_idx: Optional[int],
    favored_price: Optional[float],
    raw_score: Optional[float],
    ucb_bonus: Optional[float],
    epsilon_used: bool,
    eps_value: float,
    caps_applied: list[tuple[str, float]],
) -> None:
    meta = state.setdefault("meta", {})
    deck = meta.get("decision_cards")
    if deck is None:
        # I keep the most recent 16 cards to bound the memory footprint.
        deck = deque(maxlen=16)
        meta["decision_cards"] = deck

    card = {
    "t": int(t),
    "choice": (
        None if (grid_idx is None or grid_price is None)
        else {"i": int(grid_idx), "p": _round4(grid_price)}
    ),
    "fav": (
        None if (favored_idx is None or favored_price is None)
        else {"i": int(favored_idx), "p": _round4(favored_price)}
    ),
    "score": None if raw_score is None else _round4(raw_score),
    "bonus_ucb": None if ucb_bonus is None else _round4(ucb_bonus),
    "eps": {"used": bool(epsilon_used), "val": _round4(eps_value)},
    "caps": [{"name": n, "to": _round4(v)} for (n, v) in caps_applied],
}

    deck.append(card)
    meta["decision_last"] = card

# =========================
# Data Structures
# =========================
# My solution to the memory constraint. This fixed-size buffer prevents memory leaks and keeps the state object lightweight.
class RingBuffer:
    def __init__(self, size: int):
        self.size = int(size)
        self.buffer = {
            'p': np.zeros(self.size, dtype=float),
            'q': np.zeros(self.size, dtype=float),
            'pc': np.zeros(self.size, dtype=float),
        }
        self.head = 0
        self.count = 0

    def add(self, price: float, sales: float, competitor_price: Optional[float] = None):
        # This function overwrites the oldest data point to make room for new ones.
        i = self.head % self.size
        self.buffer['p'][i] = price
        self.buffer['q'][i] = sales
        self.buffer['pc'][i] = competitor_price if competitor_price is not None else 0.0
        self.head += 1
        self.count = min(self.count + 1, self.size)

    def get_tail(self, n: int):
        # This allows me to grab the most recent 'n' data points for my calculations.
        n = int(min(n, self.count))
        if n == 0:
            return {k: np.zeros(0, dtype=float) for k in self.buffer}
        start = (self.head - n) % self.size
        if start + n <= self.size:
            return {k: v[start:start + n].copy() for k, v in self.buffer.items()}
        first_len = self.size - start
        res = {}
        for k, v in self.buffer.items():
            res[k] = np.concatenate([v[start:], v[:n - first_len]])
        return res

# I use this to track EWMA (Exponentially Weighted Moving Average) statistics for our key metrics.
class OnlineStats:
    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.mean = 0.0
        self.variance = 0.0
        self.n = 0.0

    def update(self, x: float):
        if self.n == 0:
            self.mean = float(x)
            self.n = 1.0
            return
        self.n = (1 - self.alpha) * self.n + 1.0
        delta = x - self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.variance = ((1 - self.alpha) * self.variance
                         + self.alpha * delta * (x - self.mean))

# My solution for online linear regression. Instead of re-fitting a model every period (which would be too slow), instead i just update a few key sufficient statistics in constant time
class DecayedOLS:
     def __init__(self, lam: float = 0.95):
        self.lam = float(lam)
        self.Sx = 0.0
        self.Sy = 0.0
        self.Sxx = 0.0
        self.Sxy = 0.0
        self.n_eff = 0.0

     def update(self, price: float, sales: float):
        lam = self.lam
        self.Sx = lam * self.Sx + price
        self.Sy = lam * self.Sy + sales
        self.Sxx = lam * self.Sxx + price * price
        self.Sxy = lam * self.Sxy + price * sales
        self.n_eff = lam * self.n_eff + 1.0
        
     # Calculate the beta (slope) and alpha (intercept) from the updated stats.
     def get_coeffs(self) -> Tuple[float, float]:
        if self.n_eff < 1e-9:
            return 0.0, 0.0
        denom = self.Sxx - (self.Sx * self.Sx) / self.n_eff
        if abs(denom) < 1e-9:
            return 0.0, 0.0
        b = (self.Sxy - (self.Sx * self.Sy) / self.n_eff) / denom
        a = (self.Sy - b * self.Sx) / self.n_eff
        return a, b

# =========================
# Core policy components
# =========================

# This is my bandit policy. It calculates the UCB score for each price arm on the grid.
def _ucb_from_ringbuffer(rb: RingBuffer, window: int, t: int):
    tail = rb.get_tail(window)
    p, q = tail['p'], tail['q']
    if len(p) == 0:
        return np.zeros(K), np.zeros(K)

    idxs = np.clip(np.rint((p - P_MIN) / (P_MAX - P_MIN) * (K - 1)).astype(int), 0, K - 1)
    rev = p * q

    counts = np.bincount(idxs, minlength=K).astype(float)
    sums = np.bincount(idxs, weights=rev, minlength=K)
    means = np.divide(sums, np.maximum(counts, 1.0))
    t = max(t, 1)
    bonus = UCB_C * np.sqrt(np.log(t + 1.0) / (counts + 1.0))
    scores = means + bonus
    return means, scores

# This is where I use the online regression model to predict the optimal price.
def _ols_target(ols: DecayedOLS) -> Optional[float]:
    a_hat, b_hat = ols.get_coeffs()
    b_hat = (b_hat * ols.n_eff + SAFE_PRIOR_B) / (ols.n_eff + 1.0)
    b_hat = min(b_hat, -EPSILON_DEMAND_SLOPE)
    if a_hat > 0 and b_hat < 0:
        p_star = -a_hat / (2.0 * b_hat)
        return clamp(p_star, P_MIN, P_MAX)
    return None

# =========================
# Pricing function (core)
# =========================

# The agent's central logic.
def _core_p(context: Dict[str, Any], info_dump: Optional[Dict[str, Any]] = None):
    if info_dump is None:
        # On the first call, the initial state is set up. The information dump is reset at the beginning of each competition.
        state: Dict[str, Any] = {
            'period': 0,
            'rb': {'size': RB_SIZE, 'head': 0, 'count': 0,
                   'p': np.zeros(RB_SIZE).tolist(),
                   'q': np.zeros(RB_SIZE).tolist(),
                   'pc': np.zeros(RB_SIZE).tolist()},
            'ew': {'p': {'m': 0.0, 'v': 0.0, 'n': 0.0},
                   'q': {'m': 0.0, 'v': 0.0, 'n': 0.0},
                   'pc': {'m': 0.0, 'v': 0.0, 'n': 0.0}},
            'ols': {'Sx': 0.0, 'Sy': 0.0, 'Sxx': 0.0, 'Sxy': 0.0, 'n_eff': 0.0},
            'meta': {'epsilon': EPS, 'last_price': None, 'last_sales': None},
            'cfg': {'P_MIN': P_MIN, 'P_MAX': P_MAX, 'W': W, 'K': K}
        }
    else:
        # On subsequent calls, I deserialize the state.
        state = info_dump
        
    # Re-instantiate the custom objects from the serialized state.
    rb = RingBuffer(state['rb']['size'])
    rb.head = state['rb']['head']
    rb.count = state['rb']['count']
    rb.buffer = {
        'p': np.array(state['rb']['p'], dtype=float),
        'q': np.array(state['rb']['q'], dtype=float),
        'pc': np.array(state['rb']['pc'], dtype=float),
    }

    ew_p = OnlineStats(); ew_p.mean = state['ew']['p']['m']; ew_p.variance = state['ew']['p']['v']; ew_p.n = state['ew']['p']['n']
    ew_q = OnlineStats(); ew_q.mean = state['ew']['q']['m']; ew_q.variance = state['ew']['q']['v']; ew_q.n = state['ew']['q']['n']
    ew_pc = OnlineStats(); ew_pc.mean = state['ew']['pc']['m']; ew_pc.variance = state['ew']['pc']['v']; ew_pc.n = state['ew']['pc']['n']

    ols = DecayedOLS()
    ols.Sx = state['ols']['Sx']; ols.Sy = state['ols']['Sy']
    ols.Sxx = state['ols']['Sxx']; ols.Sxy = state['ols']['Sxy']; ols.n_eff = state['ols']['n_eff']

    last_sales = context.get('sales', None)
    last_comp = context.get('competitor_price', None)
    last_cost = context.get('cost', None)

    if last_sales is not None and state['meta']['last_price'] is not None:
        # Update all my online statistics and ring buffer with the latest data.
        lp = float(state['meta']['last_price'])
        rb.add(lp, float(last_sales), float(last_comp) if last_comp is not None else None)
        ew_p.update(lp)
        ew_q.update(float(last_sales))
        if last_comp is not None:
            ew_pc.update(float(last_comp))
        ols.update(lp, float(last_sales))

    t = state['period']
    epsilon = float(state['meta'].get('epsilon', EPS))

    # I prepare decision-card fields (they stay None during cold-start).
    chosen_idx = None
    chosen_price = None
    favored_idx = None
    favored_price = None
    raw_score_for_choice = None
    ucb_bonus_for_choice = None
    epsilon_flag = False
    caps: list[tuple[str, float]] = []

    if t < COLD_START_PERIODS:
        # During the cold start, I use a safe, fixed price. This avoids making # wild guesses when there is no data to learn from.
        price = SAFE_DEFAULT_PRICE
    else:
        # This is the core of my hybrid policy. I get the scores from my UCB bandit and also a target price from my online regression model.
        means, scores = _ucb_from_ringbuffer(rb, W, t)
        p_star = _ols_target(ols)
        fav_idx = nearest_grid_idx(p_star) if p_star is not None else None
        favored_idx = fav_idx
        favored_price = None if fav_idx is None else float(GRID[fav_idx])

        if np.random.rand() < epsilon:
            # Exploration: I randomly select a price from the grid.
            idx = int(np.random.randint(0, K))
            epsilon_flag = True
        else:
            # Exploitation: I choose the best-performing price.
            idx = int(np.argmax(scores))
            if fav_idx is not None:
                best = scores[idx]
                # I give a slight preference to the OLS-predicted price if its UCB score is close to the best-performing arm.
                if scores[fav_idx] >= best - 1e-9:
                    idx = fav_idx

        # I record the raw grid choice before any business caps.
        chosen_idx = idx
        chosen_price = float(GRID[idx])
        raw_score_for_choice = float(scores[idx])
        ucb_bonus_for_choice = float(scores[idx] - means[idx])

        price = chosen_price
        state['meta']['epsilon'] = max(0.01, epsilon * EPS_DECAY)

    # I capture which caps actually modified the price for the Decision Card.
    p_prev = price
    price = min(price, P_MAX)
    if price != p_prev:
        caps.append(("max_cap", price))

    p_prev = price
    price = max(price, P_MIN)
    if price != p_prev:
        caps.append(("min_cap", price))

    if last_comp is not None:
        p_prev = price
        # I add a critical business rule: never price above the competitor.
        price = min(price, float(last_comp) - 0.01)
        if price != p_prev:
            caps.append(("competitor_cap", price))

    if not np.isfinite(price):
        # A final safety check to ensure the price is not NaN or infinity.
        price = SAFE_DEFAULT_PRICE
    price = clamp(price, P_MIN, P_MAX)
    
    if last_cost is not None:
        p_prev = price
        # Another business safeguard: never price below cost plus a margin.
        price = max(price, float(last_cost) + 0.1)
        if price != p_prev:
            caps.append(("cost_floor", price))
        price = clamp(price, P_MIN, P_MAX)

    # I update my state object with the latest information before returning it.
    state['period'] = t + 1
    state['meta']['last_price'] = price
    state['meta']['last_sales'] = last_sales

    # I also store a compact Decision Card to explain the action taken this round.
    _record_decision(
        state,
        t=t,
        grid_idx=chosen_idx,
        grid_price=chosen_price,
        favored_idx=favored_idx,
        favored_price=favored_price,
        raw_score=raw_score_for_choice,
        ucb_bonus=ucb_bonus_for_choice,
        epsilon_used=epsilon_flag,
        eps_value=epsilon,
        caps_applied=caps,
    )

    state['rb']['p'] = rb.buffer['p'].tolist()
    state['rb']['q'] = rb.buffer['q'].tolist()
    state['rb']['pc'] = rb.buffer['pc'].tolist()
    state['rb']['head'] = rb.head
    state['rb']['count'] = rb.count

    state['ew']['p'] = {'m': ew_p.mean, 'v': ew_p.variance, 'n': ew_p.n}
    state['ew']['q'] = {'m': ew_q.mean, 'v': ew_q.variance, 'n': ew_q.n}
    state['ew']['pc'] = {'m': ew_pc.mean, 'v': ew_pc.variance, 'n': ew_pc.n}

    state['ols'] = {'Sx': ols.Sx, 'Sy': ols.Sy, 'Sxx': ols.Sxx, 'Sxy': ols.Sxy, 'n_eff': ols.n_eff}

    return price, state

def p(*args, **kwargs):
    # This wrapper allows the core logic to be called with different function signatures.
    # It ensures the code is compatible with both local testing and the DPC platform's # specific API.
    
    def _extract_dump(_kw):
        return _kw.get('information_dump') if 'information_dump' in _kw else _kw.get('info_dump')

    if (len(args) in (1, 2)) and ('current_selling_season' not in kwargs):
        context = args[0] if len(args) >= 1 else {}
        info_dump = args[1] if len(args) == 2 else _extract_dump(kwargs)
        return _core_p(context, info_dump)

    if 'current_selling_season' in kwargs:
        sell_per = kwargs.get('selling_period_in_current_season')
        demand_hist = kwargs.get('demand_historical_in_current_season')
        info_dump = _extract_dump(kwargs)

        sales = None
        try:
            if sell_per and sell_per > 1 and demand_hist is not None:
                sales = float(demand_hist[-1])
        except Exception:
            sales = None

        context = {'sales': sales, 'competitor_price': None, 'cost': None}
        return _core_p(context, info_dump)

    raise TypeError("p() called with unexpected arguments; expected local or platform signature.")
