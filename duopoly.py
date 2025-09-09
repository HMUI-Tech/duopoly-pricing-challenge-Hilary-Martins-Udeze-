import numpy as np

# ===== Config =====
P_MIN, P_MAX = 10.0, 90.0
K = 31                         # grid size
W = 80                         # window for bandit stats
EPS0, EPS_DECAY = 0.05, 0.999  # epsilon-greedy
UCB_C = 1.0
RB_SIZE = 128
COLD_START = 3
SAFE_PRIOR_B = -0.5            # negative-demand prior
SAFE_DEFAULT = (P_MIN + P_MAX) / 2.0

GRID = np.linspace(P_MIN, P_MAX, K)

def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))

def grid_idx(p):
    i = int(round((p - P_MIN) / (P_MAX - P_MIN) * (K - 1)))
    return int(np.clip(i, 0, K - 1))

# ===== Bounded buffer =====
class RingBuffer:
    def __init__(self, size):
        self.size = int(size)
        self.p = np.zeros(self.size, float)
        self.q = np.zeros(self.size, float)
        self.pc = np.zeros(self.size, float)
        self.head = 0
        self.count = 0

    def add(self, price, qty, comp=None):
        i = self.head % self.size
        self.p[i] = float(price)
        self.q[i] = float(qty)
        self.pc[i] = 0.0 if comp is None else float(comp)
        self.head += 1
        self.count = min(self.count + 1, self.size)

    def tail(self, n):
        n = int(min(n, self.count))
        if n == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        start = (self.head - n) % self.size
        if start + n <= self.size:
            return self.p[start:start+n].copy(), self.q[start:start+n].copy(), self.pc[start:start+n].copy()
        k = self.size - start
        return (np.concatenate([self.p[start:], self.p[:n-k]]),
                np.concatenate([self.q[start:], self.q[:n-k]]),
                np.concatenate([self.pc[start:], self.pc[:n-k]]))

# ===== Online OLS with decay =====
class DecayedOLS:
    def __init__(self, lam=0.95):
        self.lam = float(lam)
        self.Sx = self.Sy = self.Sxx = self.Sxy = self.n = 0.0

    def update(self, x, y):
        l = self.lam
        self.Sx  = l*self.Sx  + x
        self.Sy  = l*self.Sy  + y
        self.Sxx = l*self.Sxx + x*x
        self.Sxy = l*self.Sxy + x*y
        self.n   = l*self.n   + 1.0

    def coeffs(self):
        if self.n < 1e-9:
            return 0.0, 0.0
        denom = self.Sxx - (self.Sx*self.Sx)/self.n
        if abs(denom) < 1e-9:
            return 0.0, 0.0
        b = (self.Sxy - (self.Sx*self.Sy)/self.n) / denom
        a = (self.Sy - b*self.Sx) / self.n
        return a, b

def _ols_target(ols):
    a, b = ols.coeffs()
    # blend with negative prior to avoid positive slope
    b = (b*ols.n + SAFE_PRIOR_B) / (ols.n + 1.0)
    b = min(b, -1e-6)
    if a > 0 and b < 0:
        p_star = -a / (2.0*b)
        return clamp(p_star, P_MIN, P_MAX)
    return None

def _ucb_scores(rb, window, tnow):
    p, q, _ = rb.tail(window)
    if p.size == 0:
        return np.zeros(K), np.zeros(K)
    idx = np.clip(np.rint((p - P_MIN) / (P_MAX - P_MIN) * (K - 1)).astype(int), 0, K-1)
    rev = p * q
    cnt = np.bincount(idx, minlength=K).astype(float)
    s = np.bincount(idx, weights=rev, minlength=K)
    mean = s / np.maximum(cnt, 1.0)
    tnow = max(int(tnow), 1)
    bonus = UCB_C * np.sqrt(np.log(tnow + 1.0) / (cnt + 1.0))
    return mean, mean + bonus

# ===== Core =====
def _core_p(ctx, dump=None):
    if dump is None:
        state = {
            "t": 0,
            "rb": {"size": RB_SIZE, "head": 0, "count": 0,
                   "p": np.zeros(RB_SIZE).tolist(),
                   "q": np.zeros(RB_SIZE).tolist(),
                   "pc": np.zeros(RB_SIZE).tolist()},
            "ols": {"Sx":0.0,"Sy":0.0,"Sxx":0.0,"Sxy":0.0,"n":0.0},
            "meta": {"eps": EPS0, "last_p": None},
        }
    else:
        state = dump

    # rebuild objects
    rb = RingBuffer(state["rb"]["size"])
    rb.head  = state["rb"]["head"]
    rb.count = state["rb"]["count"]
    rb.p = np.array(state["rb"]["p"], float)
    rb.q = np.array(state["rb"]["q"], float)
    rb.pc = np.array(state["rb"]["pc"], float)

    ols = DecayedOLS()
    ols.Sx = state["ols"]["Sx"];  ols.Sy = state["ols"]["Sy"]
    ols.Sxx= state["ols"]["Sxx"]; ols.Sxy= state["ols"]["Sxy"]; ols.n = state["ols"]["n"]

    last_sales = ctx.get("sales")
    comp = ctx.get("competitor_price")
    cost = ctx.get("cost")

    if last_sales is not None and state["meta"]["last_p"] is not None:
        lp = float(state["meta"]["last_p"])
        rb.add(lp, float(last_sales), float(comp) if comp is not None else None)
        ols.update(lp, float(last_sales))

    t = state["t"]
    eps = float(state["meta"]["eps"])

    if t < COLD_START:
        price = SAFE_DEFAULT
    else:
        _, scores = _ucb_scores(rb, W, t)
        p_star = _ols_target(ols)
        fav = grid_idx(p_star) if p_star is not None else None

        if np.random.rand() < eps:
            idx = int(np.random.randint(0, K))
        else:
            idx = int(np.argmax(scores))
            if fav is not None and scores[fav] >= scores[idx] - 1e-9:
                idx = fav

        price = float(GRID[idx])
        state["meta"]["eps"] = max(0.01, eps * EPS_DECAY)

    # bounds & simple caps
    price = clamp(price, P_MIN, P_MAX)
    if comp is not None:
        price = min(price, float(comp) - 0.01)
    if cost is not None:
        price = max(price, float(cost) + 0.1)
    price = clamp(price, P_MIN, P_MAX)

    # persist state
    state["t"] = t + 1
    state["meta"]["last_p"] = price
    state["rb"]["p"] = rb.p.tolist(); state["rb"]["q"] = rb.q.tolist(); state["rb"]["pc"] = rb.pc.tolist()
    state["rb"]["head"] = rb.head; state["rb"]["count"] = rb.count
    state["ols"] = {"Sx": ols.Sx, "Sy": ols.Sy, "Sxx": ols.Sxx, "Sxy": ols.Sxy, "n": ols.n}

    return price, state

def p(*args, **kwargs):
    def _extract_dump(kw):
        return kw.get("information_dump") if "information_dump" in kw else kw.get("info_dump")
    if (len(args) in (1, 2)) and ("current_selling_season" not in kwargs):
        ctx = args[0] if len(args) >= 1 else {}
        dump = args[1] if len(args) == 2 else _extract_dump(kwargs)
        return _core_p(ctx, dump)
    if "current_selling_season" in kwargs:
        sell_per = kwargs.get("selling_period_in_current_season")
        hist = kwargs.get("demand_historical_in_current_season")
        dump = _extract_dump(kwargs)
        y = None
        try:
            if sell_per and sell_per > 1 and hist is not None:
                y = float(hist[-1])
        except Exception:
            y = None
        ctx = {"sales": y, "competitor_price": None, "cost": None}
        return _core_p(ctx, dump)
    raise TypeError("unexpected signature")
