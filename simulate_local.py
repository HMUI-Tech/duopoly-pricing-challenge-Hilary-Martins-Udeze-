"""
Tiny local simulator for the pricing policy.

Modes
-----
1) Synthetic demand (default): q = max(0, a - b*p + c*p_comp + noise).
   - Competitor follows a slow random walk with optional shocks.
2) Replay competitor path from DPC CSV (no demand): shows what our policy would
   have priced given a chosen competition_id path; writes CSV for inspection.

Usage
-----
# Synthetic for 300 steps (with a price shock around step 200)
python simulate.py --steps 300 --shock_start 200 --shock_len 25

# Replay from CSV (no demand)
python simulate.py --csv "/path/to/duopoly_competition_details.csv" --competition_id YCGzvi --steps 200
"""
import argparse
import numpy as np
from pathlib import Path

# Import the algorithm
def load_algo(algo_path: str = "duopoly.py"):
    
    # This ensures the simulator can run the pricing policy even if it's not in the same directory.
    import importlib.util
    path = Path(algo_path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm file not found: {path}")
    spec = importlib.util.spec_from_file_location("duopoly_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def synthetic_env(a=50.0, b=0.8, c=0.2, noise=1.0, comp_init=50.0):
    # This function creates a simple, fake environment to test against the policy.
    # It models demand as a simple linear function with noise.
    state = {"a": a, "b": b, "c": c, "noise": noise, "pc": comp_init, "t": 0}
    rng = np.random.default_rng(42)
    def step(price, shock=False):
        # This is a key part of the simulator: it calculates sales and updates the competitor's price.
        drift = rng.normal(0, 0.3)
        state["pc"] = float(np.clip(state["pc"] + drift + (10 if shock else 0), 10, 90))
        mu = max(0.0, a - b*price + c*state["pc"])
        q = max(0.0, rng.normal(mu, noise))
        state["t"] += 1
        return q, state["pc"]
    return step

# This function runs the simulation in synthetic mode.
def run_synthetic(mod, steps=300, shock_start=None, shock_len=0):
    info = None
    step_fn = synthetic_env()
    out = []
    for t in range(steps):
        ctx = {} if t == 0 else {"sales": out[-1]["q"], "competitor_price": out[-1]["pc"]}
        price, info = mod.p(ctx, info_dump=info)
        in_shock = shock_start is not None and (shock_start <= t < shock_start + shock_len)
        q, pc = step_fn(price, shock=in_shock)
        out.append({"t": t+1, "price": price, "q": q, "rev": price*q, "pc": pc, "shock": int(in_shock)})
    return out

# This function runs the simulation in replay mode, using real-world competitor data.
def run_replay(mod, csv_path, competition_id, steps=None):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for --csv mode. Install it in your env.") from e
    df = pd.read_csv(csv_path)
    df = df[df["competition_id"] == competition_id].sort_values(["selling_season","selling_period"])
    if steps:
        df = df.head(steps)
    info = None
    out = []
    for _, row in df.iterrows():
        # Here, the competitor price is read from the CSV instead of simulating it
        comp = float(row.get("price_competitor", np.nan))
        ctx = {} if not out else {"sales": out[-1].get("q", np.nan), "competitor_price": comp}
        price, info = mod.p(ctx, info_dump=info)
        out.append({"t": int(row["selling_period"]), "price": price, "pc": comp})
    return out

def main():
    # This is the entry point for the script, handling command-line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--shock_start", type=int, default=None)
    ap.add_argument("--shock_len", type=int, default=0)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--competition_id", type=str, default=None)
    ap.add_argument("--algo", type=str, default="duopoly.py")  # always duopoly.py
    ap.add_argument("--export_csv", type=str, default=None)
    args = ap.parse_args()

    mod = load_algo(args.algo)

    if args.csv and args.competition_id:
        out = run_replay(mod, args.csv, args.competition_id, steps=args.steps)
        fields = ["t","price","pc"]
    else:
        out = run_synthetic(mod, steps=args.steps, shock_start=args.shock_start, shock_len=args.shock_len)
        fields = ["t","price","q","rev","pc","shock"]

    # Print a tiny textual summary
    if "rev" in out[0]:
        total_rev = sum(r["rev"] for r in out)
        avg_rev = total_rev / len(out)
        last50 = out[-50:]
        last50_avg = sum(r["rev"] for r in last50) / len(last50)
        print(f"Steps: {len(out)} | Total revenue: {total_rev:.2f} | Avg/step: {avg_rev:.2f} | Last50 avg: {last50_avg:.2f}")
    else:
        print(f"Replay rows: {len(out)} (no demand; showing policy actions given competitor path)")

    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        print(f"Exported to {args.export_csv}")

if __name__ == "__main__":
    main()
