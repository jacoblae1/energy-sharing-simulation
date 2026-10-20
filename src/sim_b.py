# Simulation B: Lohnt sich eine Batterie, und wie groß sollte sie sein?
# Je Batteriegröße (100 bis 2.000 kWh, Leistung = 0,25 * Kapazität) wird verglichen:
#   - Wert der Batterie ohne Sharing und mit Sharing (gegenüber PV allein bzw. PV + Sharing ohne Batterie)
#   - jährliche Batteriekosten (Annuität + Betrieb), die erst nach dem Lauf abgezogen werden
#   - Kipppunkt rho* mit vorhandener Batterie (ohne Anschaffungskosten)
#   - Lücke zwischen den Verfahren: Forecast-Lücke = PF-RH - RH, Horizont-Lücke = Oracle - PF-RH,
#     Gesamtlücke = Oracle - RH = Forecast-Lücke + Horizont-Lücke
# Ergebnisse: results/sim_b_raw.csv (jeder Lauf) und results/sim_b_summary.csv (Mittelwerte je Batterie).

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import data
import model as m
import sim_a

RESULTS = data.ROOT / "results"
RHO_BASE = config.RHO_BASELINE
RHO_GRID = list(config.RHO_SUPPORT_POINTS)
assert RHO_BASE in RHO_GRID
CAPEX_LOW, CAPEX_HIGH = config.BATTERY_CAPEX_EUR_PER_KWH
COST_COLS = (f"cost_{CAPEX_LOW:.0f}", f"cost_{CAPEX_HIGH:.0f}")
NET_COLS = (f"net_{CAPEX_LOW:.0f}", f"net_{CAPEX_HIGH:.0f}")
MAIN_CONFIGS = [(E, config.C_RATE * E) for E in config.BATTERY_SIZES_KWH]
CONFIGS = MAIN_CONFIGS + [tuple(c) for c in config.EXTRA_POWER_CONFIGS]
BREAK_EVEN_FIXED_COSTS = (int(config.SHARING_FIXED_COST_RANGE[0]), int(config.SHARING_FIXED_COST_EUR_PER_YEAR),
                          int(config.SHARING_FIXED_COST_RANGE[1]))   # Sharing-Kosten (niedrig, Baseline, hoch) der 50-Haushalte-Community


def annual_cost(E, capex_per_kwh):
    """Jährliche Batteriekosten: Investition als Annuität (Zinssatz, Nutzungsdauer) plus Betriebskosten."""
    r, n = config.BATTERY_WACC, config.BATTERY_LIFETIME_YEARS
    annuity = r / (1.0 - (1.0 + r) ** -n)           # Annuitätenfaktor
    return E * capex_per_kwh * (annuity + config.BATTERY_OPEX_SHARE)


def one_run(args):
    """Ein Monte-Carlo-Lauf: eine zufällige Nachfrage ziehen und RH rechnen; beim Referenz-rho zusätzlich Oracle und PF-RH."""
    seed, E, P, rho, with_oracle, g, p, d_plan = args
    d_real = data.realized_demand(pd.Series(d_plan), np.random.default_rng(seed)).to_numpy()
    r = m.rolling_horizon(g, p, d_plan, d_real, True, rho, E, P)
    row = {"seed": seed, "E": E, "P": P, "rho": rho, "gross_rh": m.profit(r, p, rho),
           "soc_end_rh": r["soc_end"],
           "shared_share": r["s"].sum() / d_real.sum(),
           "gain_no_battery": sim_a.gross_gain(g, p, d_real, rho)}      # Zusatzgewinn durch Sharing ohne Batterie (Simulation A)
    if with_oracle:
        ro = m.oracle(g, p, d_real, rho, E, P)
        row["gross_oracle"], row["soc_end_oracle"] = m.profit(ro, p, rho), ro["soc_end"]
        rp = m.rolling_horizon(g, p, d_plan, d_real, True, rho, E, P, perfect_foresight=True)
        row["gross_pfrh"], row["soc_end_pfrh"] = m.profit(rp, p, rho), rp["soc_end"]
        row["forecast_gap"] = row["gross_pfrh"] - row["gross_rh"]
        row["horizon_gap"] = row["gross_oracle"] - row["gross_pfrh"]
        row["total_gap"] = row["gross_oracle"] - row["gross_rh"]
    return row


def main(n_runs):
    """Rechnet alle Batteriegrößen mit n_runs Läufen je Einstellung, prüft die Ergebnisse und speichert sie."""
    g = data.load_pvgis().to_numpy()
    p = data.load_smard().to_numpy()
    d_plan = data.load_bdew_h0_profile().to_numpy()
    assert len(g) == len(p) == len(d_plan) == 8760
    market_only = (np.maximum(p, 0.0) * g).sum()        # Gewinn mit PV allein (nur Börse, Abregeln bei negativem Preis)

    # Batterie ohne Sharing: braucht keine Zufallsnachfrage, deshalb nur einmal je Größe
    b0 = {}
    for E, P in CONFIGS:
        b0[(E, P)] = m.profit(m.rolling_horizon(g, p, d_plan, d_plan, False, 0.0, E, P), p, 0.0)
        print(f"Battery {E} kWh / {P:.0f} kW without sharing: value {b0[(E, P)] - market_only:,.0f} EUR/year")

    # Alle Läufe vorbereiten: Hauptgrößen für alle rho-Stützstellen, Extra-Konfigurationen nur beim Referenz-rho
    jobs = []
    for E, P in CONFIGS:
        rhos = RHO_GRID if (E, P) in MAIN_CONFIGS else [RHO_BASE]
        for rho in rhos:
            for i in range(n_runs):
                jobs.append((config.RANDOM_SEED + i, E, P, rho, rho == RHO_BASE, g, p, d_plan))
    print(f"{len(jobs)} runs ...")
    t0 = time.time()
    with Pool(min(config.N_WORKERS, os.cpu_count() or 1)) as pool:
        df = pd.DataFrame(pool.map(one_run, jobs))
    print(f"finished in {time.time() - t0:.0f}s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "sim_b_raw.csv", index=False)

    # Plausibilitätsprüfungen der Ergebnisse (Oracle muss mindestens so gut sein wie die anderen Verfahren)
    chk = df.dropna(subset=["gross_oracle"])
    n_bad = int((chk.gross_oracle < chk.gross_rh).sum())
    print(f"Check Oracle >= RH: {len(chk) - n_bad} of {len(chk)} runs ok"
          + (f" -- WARNING: {n_bad} runs with Oracle < RH, investigate!" if n_bad else ""))
    n_bad2 = int((chk.gross_oracle < chk.gross_pfrh - 0.05).sum())
    print(f"Check Oracle >= PF-RH: {len(chk) - n_bad2} of {len(chk)} runs ok"
          + (f" -- WARNING: {n_bad2} runs with Oracle < PF-RH, investigate!" if n_bad2 else ""))
    print(f"Year-end SOC (kWh), largest value over all runs: RH {df.soc_end_rh.max():.2f}, PF-RH {chk.soc_end_pfrh.max():.2f}, "
          f"Oracle {chk.soc_end_oracle.max():.2f}")
    ident = (chk.forecast_gap + chk.horizon_gap - chk.total_gap).abs().max()
    print(f"Check decomposition: max |forecast + horizon - total| = {ident:.2e} EUR")
    n_neg = int((chk.forecast_gap < 0).sum())
    print(f"Forecast gap negative in {n_neg} of {len(chk)} runs"
          + (": " + ", ".join(f"{int(E)} kWh/{P:g} kW: {int(k)}" for (E, P), k in
                              chk[chk.forecast_gap < 0].groupby(["E", "P"]).size().items()) if n_neg else ""))

    # Zusammenfassung: eine Zeile je Batteriegröße (Mittelwerte über die Läufe)
    rows = []
    for E, P in CONFIGS:
        sub = df[(df.E == E) & (df.P == P)]
        base = sub[sub.rho == RHO_BASE]             # nur die Läufe beim Referenz-rho
        gross_rh = base.gross_rh.mean()
        row = {"E_kWh": E, "P_kW": P, "C_rate": P / E,
               "value_no_sharing": b0[(E, P)] - market_only,                                   # Batterie ohne Sharing gegenüber PV allein
               "value_with_sharing": gross_rh - (market_only + base.gain_no_battery.mean()),   # Batterie gegenüber PV + Sharing ohne Batterie
               "shared_pct": base.shared_share.mean() * 100,
               COST_COLS[0]: annual_cost(E, CAPEX_LOW),
               COST_COLS[1]: annual_cost(E, CAPEX_HIGH)}
        row[NET_COLS[0]] = row["value_with_sharing"] - row[COST_COLS[0]]        # Wert minus Jahreskosten
        row[NET_COLS[1]] = row["value_with_sharing"] - row[COST_COLS[1]]
        row["forecast_gap"] = base.forecast_gap.mean()
        row["horizon_gap"] = base.horizon_gap.mean()
        row["total_gap"] = base.total_gap.mean()
        row["forecast_gap_min"] = base.forecast_gap.min()
        row["forecast_gap_std"] = base.forecast_gap.std()
        if (E, P) in MAIN_CONFIGS:
            # Kipppunkt rho*: Zusatzgewinn durch Sharing (mit bzw. ohne Batterie) gleich den Fixkosten; bei rho = 0 ist er 0
            gb = [0.0] + [sub[sub.rho == r_].gross_rh.mean() - b0[(E, P)] for r_ in RHO_GRID]   # mit Batterie
            ga = [0.0] + [sub[sub.rho == r_].gain_no_battery.mean() for r_ in RHO_GRID]         # ohne Batterie
            xs = [0.0] + RHO_GRID
            for c in BREAK_EVEN_FIXED_COSTS:
                row[f"rho_star_battery_c{c}"] = np.interp(c, gb, xs, right=np.nan) * 100
                row[f"rho_star_no_battery_c{c}"] = np.interp(c, ga, xs, right=np.nan) * 100
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sim_b_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(out.round(1).T.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else config.N_RUNS_SIM_B)
