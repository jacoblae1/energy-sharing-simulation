# Tests für Daten, Modell und Simulationen.

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data
import model as m
import plots
import sim_a

TOL = 1e-4
GAP_TOL = 0.05
RHO = config.RHO_BASELINE


class Recorder:
    def __enter__(self):
        self.calls, self.orig = [], m.solve_lp

        def rec(g, p, d, soc_start, sharing, rho, E, P):
            self.calls.append(dict(g=np.array(g), p=np.array(p), d=np.array(d), soc_start=soc_start,
                                   sharing=sharing, rho=rho, E=E, P=P))
            return self.orig(g, p, d, soc_start, sharing, rho, E, P)
        m.solve_lp = rec
        return self

    def __exit__(self, *exc):
        m.solve_lp = self.orig


def check_balance(name, r, g, d_real):
    avail = g + r["q"] - r["c"] - r["s"] - r["e"]
    assert avail.min() > -TOL, f"{name}: energy balance violated (min {avail.min():.6f})"
    assert (r["s"] - d_real).max() < TOL, f"{name}: more electricity shared than demanded"


def soc_path(r, soc0):
    return soc0 + np.cumsum(m.ETA * r["c"] - r["q"] / m.ETA)


def manual_profit(r, p, rho):
    pi_s = p + config.SHARING_BENEFIT_SHARE * rho * config.NEG_EUR_PER_KWH
    return ((pi_s * r["s"]).sum() + (p * r["e"]).sum()
            - config.DEGRADATION_COST_EUR_PER_KWH * (r["c"] + r["q"]).sum())


def load_inputs():
    g = data.load_pvgis().to_numpy()
    p = data.load_smard().to_numpy()
    d_plan = data.load_bdew_h0_profile().to_numpy()
    return g, p, d_plan


def realized(d_plan, seed):
    return data.realized_demand(pd.Series(d_plan), np.random.default_rng(seed)).to_numpy()


def test_data():
    g = data.load_pvgis()
    p = data.load_smard()
    d = data.load_bdew_h0_profile()
    assert len(g) == len(p) == len(d) == 8760
    assert abs(d.sum() - config.ANNUAL_DEMAND_KWH) < 1e-6 and (d > 0).all()
    assert abs(config.ANNUAL_DEMAND_KWH - config.N_HOUSEHOLDS * config.HOUSEHOLD_DEMAND_KWH) < 1e-9
    for name, bad in (("missing value", g.where(np.arange(len(g)) != 5)),
                      ("irregular index", g.drop(g.index[100])),
                      ("wrong length", g.iloc[:-1])):
        try:
            data._check_hourly(bad, name)
        except ValueError:
            continue
        raise AssertionError(f"input validation did not reject: {name}")
    eps = np.array([data.generate_ar1_noise(8760, np.random.default_rng(1000 + i)) for i in range(100)])
    std_theory = config.AR1_INNOVATION_STD / np.sqrt(1 - config.AR1_PHI ** 2)
    lag1 = np.mean([np.corrcoef(e[500:-1], e[501:])[0, 1] for e in eps])
    assert abs(eps[:, 500:].std() / std_theory - 1) < 0.02, "AR(1) standard deviation off"
    assert abs(lag1 - config.AR1_PHI) < 0.01, "AR(1) autocorrelation off"
    a, b = (data.realized_demand(d, np.random.default_rng(config.RANDOM_SEED)).to_numpy() for _ in range(2))
    assert np.array_equal(a, b), "same seed gives different demand"
    hour = np.arange(8760) % 24
    peak_hour = int(pd.Series(d.to_numpy()).groupby(hour).mean().idxmax())
    assert peak_hour == 18, f"evening demand peak at {peak_hour} UTC, expected 18"
    print(f"Data tests ok: 8,760 validated hourly values per series, input validation rejects bad data, "
          f"AR(1) std/autocorrelation match, seeds reproducible, demand peak {peak_hour}:00 UTC")


def test_sim_a(g, p, d_plan):
    d = realized(d_plan, config.RANDOM_SEED)
    neg, alpha = config.NEG_EUR_PER_KWH, config.SHARING_BENEFIT_SHARE
    potential = np.minimum(g, d).sum()
    gains = np.array([sim_a.gross_gain(g, p, d, r) for r in sim_a.RHO_GRID])
    for rho in (0.15, RHO, 0.75):
        pi_s = p + alpha * rho * neg
        independent = sum((pi_s[t] - max(p[t], 0.0)) * min(g[t], d[t]) for t in range(len(g)) if pi_s[t] >= 0)
        assert abs(independent - sim_a.gross_gain(g, p, d, rho)) < 1e-6, f"gross gain differs from independent loop at rho={rho}"
        assert sim_a.gross_gain(g, p, d, rho) <= alpha * rho * neg * potential + 1e-6
    assert abs(gains[0]) < 1e-9 and (np.diff(gains) >= -1e-9).all(), "gain curve not monotone or non-zero at rho = 0"
    assert (sim_a.shared_energy(g, p, d, RHO) <= np.minimum(g, d) + 1e-12).all()
    fixed = config.SHARING_FIXED_COST_EUR_PER_YEAR
    lo, hi = 0.0, config.RHO_SWEEP[1]
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sim_a.gross_gain(g, p, d, mid) < fixed else (lo, mid)
    interpolated = float(np.interp(fixed, gains, sim_a.RHO_GRID))
    assert abs(mid - interpolated) < 5e-4, f"interpolated break-even {interpolated:.5f} vs exact {mid:.5f}"
    print(f"Simulation A tests ok: gross gain = independent derivation, bounded by alpha*rho*NEG*S, monotone, "
          f"break-even interpolation exact ({interpolated * 100:.2f} % vs {mid * 100:.2f} %)")


def test_ratio_sensitivity(g, p):
    counts = config.RATIO_HOUSEHOLD_COUNTS
    n_base = config.N_HOUSEHOLDS
    assert n_base in counts
    assert sim_a.sharing_cost(n_base, "base") == config.SHARING_FIXED_COST_EUR_PER_YEAR
    assert (sim_a.sharing_cost(n_base, "low"), sim_a.sharing_cost(n_base, "high")) == config.SHARING_FIXED_COST_RANGE
    assert all(sim_a.sharing_cost(n + 1, s) > sim_a.sharing_cost(n, s) for n in counts for s in sim_a.SCENARIOS)
    assert all(sim_a.sharing_cost(n, "low") < sim_a.sharing_cost(n, "base") < sim_a.sharing_cost(n, "high") for n in counts)
    profiles = {n: data.load_bdew_h0_profile(annual_kwh=n * config.HOUSEHOLD_DEMAND_KWH) for n in counts}
    d_base = realized(profiles[n_base].to_numpy(), config.RANDOM_SEED)
    eps_base = d_base / profiles[n_base].to_numpy() - 1.0
    for n in counts:
        d = realized(profiles[n].to_numpy(), config.RANDOM_SEED)
        assert abs(profiles[n].sum() - n * config.HOUSEHOLD_DEMAND_KWH) < 1e-6, f"{n} households: annual demand wrong"
        assert np.allclose(d, d_base * n / n_base, rtol=1e-12, atol=1e-9), f"{n} households: demand does not scale linearly"
        assert np.allclose(d / profiles[n].to_numpy() - 1.0, eps_base, atol=1e-12), f"{n} households: relative noise changed"
    base = sim_a.ratio_sensitivity(config.N_RUNS_SIM_A, (n_base,)).iloc[0]
    stored = pd.read_csv(sim_a.RESULTS / "sim_a_fixkosten.csv")
    stored_be = float(stored.loc[stored.fixed_cost == config.SHARING_FIXED_COST_EUR_PER_YEAR, "rho_star_mean"].iloc[0]) * 100
    assert base.n_runs == config.N_RUNS_SIM_A and base.break_even_status_base == "ok"
    assert abs(base.break_even_rho_pct_base - stored_be) < 0.02, (base.break_even_rho_pct_base, stored_be)
    grid = np.linspace(0.0, config.BREAK_EVEN_RHO_MAX, 101)
    for n in counts:
        for seed in (42, 43, 44):
            d = realized(profiles[n].to_numpy(), seed)
            gains = np.array([sim_a.gross_gain(g, p, d, r) for r in grid])
            for s in sim_a.SCENARIOS:
                cost = sim_a.sharing_cost(n, s)
                exact = sim_a.break_even_rho(g, p, d, cost)
                if np.isnan(exact):
                    assert gains[-1] < cost, f"{n} households, {s}: NaN although break-even exists within the range"
                else:
                    assert 0.0 <= exact <= config.BREAK_EVEN_RHO_MAX
                    assert abs(sim_a.gross_gain(g, p, d, exact) - cost) < 1e-6, f"{n} households, {s}: not a root"
                    assert abs(exact - np.interp(cost, gains, grid)) < 5e-4, f"{n} households, {s}: interpolation differs from exact root"
    small = sim_a.ratio_sensitivity(10)
    shared = small[f"shared_kwh_rho{sim_a.RHO_PCT}"].to_numpy()
    revenue = small[f"sharing_revenue_rho{sim_a.RHO_PCT}"].to_numpy()
    assert (small.n_runs == 10).all()
    assert (np.diff(shared) >= -1e-9).all() and (np.diff(revenue) >= -1e-9).all(), "shared energy / revenue not increasing"
    assert (shared <= g.sum() + 1e-6).all() and (shared <= small.annual_demand_kwh * 1.01).all(), "shared energy implausible"
    for s in sim_a.SCENARIOS:
        ok = small[f"break_even_status_{s}"] == "ok"
        col = small[f"break_even_rho_pct_{s}"]
        assert col[~ok].isna().all() and col[ok].notna().all()
        assert col[ok].between(0.0, config.BREAK_EVEN_RHO_MAX * 100).all(), f"{s}: break-even outside 0-100 %"
        assert small.loc[~ok, f"break_even_status_{s}"].str.startswith("No break-even within 0–100%").all(), f"{s}: missing flag"
    for _, r in small.iterrows():
        vals = [r[f"break_even_rho_pct_{s}"] for s in sim_a.SCENARIOS]
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]) if not (np.isnan(a) or np.isnan(b)))
    original_cost = config.SHARING_COST_FIXED_EUR
    config.SHARING_COST_FIXED_EUR = (original_cost[0], original_cost[1], 20000.0)
    try:
        flagged = sim_a.ratio_sensitivity(3, (25, 400))
    finally:
        config.SHARING_COST_FIXED_EUR = original_cost
    assert flagged.break_even_rho_pct_high.isna().all()
    assert (flagged.break_even_status_high == "No break-even within 0–100%").all()
    assert flagged.break_even_rho_pct_low.notna().all()
    pd.testing.assert_frame_equal(small.iloc[:2], sim_a.ratio_sensitivity(10, counts[:2]))
    print(f"Ratio sensitivity tests ok: cost function c(N) = F + k*N, demand scales linearly with households (noise unchanged), "
          f"baseline break-even {base.break_even_rho_pct_base:.2f} % = main simulation, exact root = interpolation, "
          f"flag for no break-even works, reproducible")


def test_dispatch(g, p, d_plan):
    n = len(g)
    d42 = realized(d_plan, config.RANDOM_SEED)

    r0 = m.rolling_horizon(g, p, d_plan, d42, True, RHO, 1e-6, 1e-6)
    a_val, b_val = (np.maximum(p, 0) * g).sum() + sim_a.gross_gain(g, p, d42, RHO), m.profit(r0, p, RHO)
    assert abs(a_val - b_val) < 0.01, (a_val, b_val)
    print(f"Test 1 ok: battery model with E~0 = Simulation A = {b_val:,.2f} EUR")

    E, P = 500, config.C_RATE * 500
    soc0 = config.INITIAL_SOC_FRACTION * E
    with Recorder() as rec_rh:
        rh = m.rolling_horizon(g, p, d_plan, d42, True, RHO, E, P)
    with Recorder() as rec_pf:
        pf = m.rolling_horizon(g, p, d_plan, d42, True, RHO, E, P, perfect_foresight=True)
    n_windows = -(-n // config.RESOLVE_EVERY_HOURS)
    assert len(rec_rh.calls) == len(rec_pf.calls) == n_windows == 2190
    for rec in (rec_rh, rec_pf):
        lengths = [len(c["g"]) for c in rec.calls]
        starts = range(0, n, config.RESOLVE_EVERY_HOURS)
        assert lengths == [min(config.HORIZON_HOURS, n - s) for s in starts], "unexpected planning window lengths"
        assert lengths[:-5] == [config.HORIZON_HOURS] * (n_windows - 5) and lengths[-5:] == [20, 16, 12, 8, 4]
        assert rec.calls[0]["soc_start"] == soc0
    for k, call in enumerate(rec_pf.calls):
        start = config.RESOLVE_EVERY_HOURS * k
        end = min(start + config.HORIZON_HOURS, n)
        assert np.array_equal(call["d"], d42[start:end]), f"PF-RH window {k}: not the realized demand"
    print("Test 2 ok: 2,190 windows (2,186 x 24 h, then 20/16/12/8/4 h at the year end), no terminal condition, "
          "PF-RH plans with the realized demand of its 24-h horizon")

    pf_garbage = m.rolling_horizon(g, p, d_plan * 0.5, d42, True, RHO, E, P, perfect_foresight=True)
    for key in "secq":
        assert np.array_equal(pf[key], pf_garbage[key]), "PF-RH depends on the base profile / AR(1)"
    for k in (1, 100, 300):
        start = config.RESOLVE_EVERY_HOURS * k
        end = min(start + config.HORIZON_HOURS, n)
        eps = d42[start - 1] / d_plan[start - 1] - 1.0
        expected = d_plan[start:end] * (1.0 + config.AR1_PHI ** np.arange(1, end - start + 1) * eps)
        assert np.allclose(rec_rh.calls[k]["d"], expected), f"RH window {k}: AR(1) forecast changed"
        assert not np.allclose(rec_rh.calls[k]["d"], d42[start:end]), f"RH window {k}: knows the realized demand"
    print("Test 3 ok: PF-RH independent of base profile / AR(1); RH still uses BDEW profile + AR(1)")

    for a, b in zip(rec_rh.calls, rec_pf.calls):
        for key in ("g", "p"):
            assert np.array_equal(a[key], b[key])
        for key in ("sharing", "rho", "E", "P"):
            assert a[key] == b[key], f"constraint/parameter {key} differs between RH and PF-RH"
    print("Test 4 ok: RH and PF-RH call the optimizer with identical data, limits, battery and rho")

    rh_b = m.rolling_horizon(g, p, d_plan, d_plan.copy(), True, RHO, E, P)
    pf_b = m.rolling_horizon(g, p, d_plan, d_plan.copy(), True, RHO, E, P, perfect_foresight=True)
    for key in "secq":
        assert np.allclose(rh_b[key], pf_b[key], atol=1e-9), f"RH != PF-RH for demand = base profile ({key})"
    assert abs(m.profit(rh_b, p, RHO) - m.profit(pf_b, p, RHO)) < 1e-6
    print(f"Test 5 ok: demand = base profile -> RH = PF-RH ({m.profit(rh_b, p, RHO):,.2f} EUR)")

    rows = []
    for E in (100, 500, 2000):
        P, soc0 = config.C_RATE * E, config.INITIAL_SOC_FRACTION * E
        for seed in (42, 43, 44):
            d = realized(d_plan, seed)
            res = {"RH": m.rolling_horizon(g, p, d_plan, d, True, RHO, E, P),
                   "PF-RH": m.rolling_horizon(g, p, d_plan, d, True, RHO, E, P, perfect_foresight=True),
                   "Oracle": m.oracle(g, p, d, RHO, E, P)}
            for name, r in res.items():
                path = soc_path(r, soc0)
                assert path.min() > -TOL and path.max() < E + TOL, f"{name}: SOC outside [0, E]"
                assert abs(path[-1] - r["soc_end"]) < 0.01, f"{name}: terminal SOC does not match the SOC path"
                assert path[-1] < 0.01, f"{name}: battery not emptied at the year end ({path[-1]:.3f} kWh)"
                check_balance(name, r, g, d)
                assert not ((r["c"] > 1e-6) & (r["q"] > 1e-6)).any(), f"{name}: simultaneous charging and discharging"
                assert abs(m.profit(r, p, RHO) - manual_profit(r, p, RHO)) < 1e-6, f"{name}: profit inconsistent"
            assert abs(m.profit(res["Oracle"], p, RHO) - res["Oracle"]["objective"]) < 0.5
            pr, pp, po = (m.profit(res[k], p, RHO) for k in ("RH", "PF-RH", "Oracle"))
            assert po >= pr - GAP_TOL, f"Oracle < RH: E={E}, seed={seed}"
            assert po >= pp - GAP_TOL, f"Oracle < PF-RH: E={E}, seed={seed}"
            forecast, horizon, total = pp - pr, po - pp, po - pr
            assert abs(forecast + horizon - total) < 1e-9
            rows.append({"E": E, "seed": seed, "Forecast gap (PF-RH - RH)": forecast,
                         "Horizon gap (Oracle - PF-RH)": horizon, "Total gap (Oracle - RH)": total,
                         "End SOC RH/PF/Orc": "/".join(f"{res[k]['soc_end']:.2f}" for k in ("RH", "PF-RH", "Oracle"))})
    pd.set_option("display.width", 200)
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    print("Tests 6-9 ok: SOC path and limits, energy balance, no simultaneous charging/discharging, profit = independent "
          "recomputation, battery empty at the year end for all methods, Oracle >= RH and >= PF-RH, decomposition exact")
    neg = df[df["Forecast gap (PF-RH - RH)"] < 0]
    print("Test 10: forecast gap negative for: " +
          (", ".join(f"E={r.E}, seed={r.seed}" for _, r in neg.iterrows()) if len(neg) else "none of the tested configurations"))


def test_plots():
    required = ["sim_a_rho.csv", "sim_a_fixkosten.csv", "sim_b_summary.csv"]
    missing = [f for f in required if not (plots.RES / f).exists()]
    assert not missing, f"result files missing: {missing}"
    original = plots.OUT
    with tempfile.TemporaryDirectory() as tmp:
        plots.OUT = Path(tmp)
        try:
            plots.main()
        finally:
            plots.OUT = original
        files = sorted(f.name for f in Path(tmp).iterdir())
        assert len(files) == 8 and all(f.endswith(".pdf") and (Path(tmp) / f).stat().st_size > 0 for f in files), files
    print("Plot test ok: all 8 figures (PDF) are produced without titles from the stored results")


def main():
    test_data()
    g, p, d_plan = load_inputs()
    test_sim_a(g, p, d_plan)
    test_ratio_sensitivity(g, p)
    test_dispatch(g, p, d_plan)
    test_plots()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
