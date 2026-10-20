# Simulation A: Energy Sharing ohne Batterie.
# Ohne Speicher gibt es nichts zu planen: Der Aggregator teilt in jeder Stunde, was die Community
# gerade braucht (höchstens die PV-Erzeugung), und verkauft den Rest an der Börse.
# Gefragt wird: Wie viel Zusatzgewinn bringt das Teilen gegenüber dem reinen Börsenverkauf,
# und ab welcher Netzentgelt-Reduktion rho (Kipppunkt rho*) decken die Zusatzerlöse die Fixkosten?
# Ergebnisse: results/sim_a_rho.csv (Zusatzgewinn über rho) und results/sim_a_fixkosten.csv (Kipppunkt über Fixkosten).
# Zusätzlich: Sensitivität zum Verhältnis PV-Leistung zu Haushaltszahl (results/sim_a_ratio_sensitivity.csv).
# Die Kosten des Sharings sind ein fester Anteil plus ein Betrag je Haushalt (siehe config.py); main() rechnet mit den
# Gesamtkosten der Baseline-Community (50 Haushalte).

import sys

import numpy as np
import pandas as pd

import config
import data
from model import sharing_price

RHO_GRID = np.linspace(*config.RHO_SWEEP, config.RHO_GRID_N)
FIXED_COST_GRID = np.arange(0.0, config.FIXED_COST_GRID_MAX_EUR + 1.0, config.FIXED_COST_GRID_STEP_EUR)
RESULTS = data.ROOT / "results"


def shared_energy(g, p, d, rho):
    """Geteilte Strommenge je Stunde: das Kleinere aus Erzeugung und Nachfrage, aber nur bei nicht negativem Sharing-Preis."""
    return np.where(sharing_price(p, rho) >= 0, np.minimum(g, d), 0.0)


def gross_gain(g, p, d, rho):
    """Zusatzgewinn durch Sharing im Jahr (vor Fixkosten): Erlös mit Sharing minus Erlös nur mit Börsenverkauf."""
    price_s = sharing_price(p, rho)
    s = shared_energy(g, p, d, rho)
    market_only = np.maximum(p, 0.0) * g                    # nur Börse: bei negativem Preis wird abgeregelt (Erlös 0)
    with_sharing = price_s * s + np.maximum(p, 0.0) * (g - s)    # geteilter Teil zum Sharing-Preis, Rest an die Börse
    return (with_sharing - market_only).sum()


def main(n_runs):
    """Rechnet n_runs Nachfrage-Realisierungen und speichert Zusatzgewinn-Kurve und Kipppunkte."""
    g = data.load_pvgis().to_numpy()
    p = data.load_smard().to_numpy()
    d_plan = data.load_bdew_h0_profile()
    assert len(g) == len(p) == len(d_plan) == 8760

    # Je Lauf eine zufällige Nachfrage ziehen und den Zusatzgewinn für jedes rho berechnen
    gains = np.zeros((n_runs, len(RHO_GRID)))
    potential = np.zeros(n_runs)        # technisch teilbare Menge: min(Erzeugung, Nachfrage)
    shared = np.zeros(n_runs)           # tatsächlich geteilte Menge bei rho = Referenzwert
    for i in range(n_runs):
        d = data.realized_demand(d_plan, np.random.default_rng(config.RANDOM_SEED + i)).to_numpy()
        gains[i] = [gross_gain(g, p, d, r) for r in RHO_GRID]
        potential[i] = np.minimum(g, d).sum()
        shared[i] = shared_energy(g, p, d, config.RHO_BASELINE).sum()

    # Kurve 1: Zusatzgewinn nach Abzug der Baseline-Fixkosten, Mittelwert und 5-/95-%-Band über die Läufe
    fixed_cost = config.SHARING_FIXED_COST_EUR_PER_YEAR
    net = gains - fixed_cost
    curve = pd.DataFrame({"rho": RHO_GRID, "mean": net.mean(0),
                          "p5": np.percentile(net, 5, axis=0), "p95": np.percentile(net, 95, axis=0),
                          "fixed_cost": fixed_cost})

    # Kurve 2: Kipppunkt rho* für jede Fixkostenhöhe = das rho, bei dem der Zusatzgewinn genau den Fixkosten entspricht
    rows = []
    for c in FIXED_COST_GRID:
        rho_star = np.array([np.interp(c, gains[i], RHO_GRID, right=np.nan) for i in range(n_runs)])
        rows.append({"fixed_cost": c, "rho_star_mean": np.nanmean(rho_star),
                     "p5": np.nanpercentile(rho_star, 5), "p95": np.nanpercentile(rho_star, 95)})
    tipping = pd.DataFrame(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    curve.to_csv(RESULTS / "sim_a_rho.csv", index=False)
    tipping.to_csv(RESULTS / "sim_a_fixkosten.csv", index=False)

    demand_total = d_plan.sum()
    print(f"{n_runs} runs. Technical sharing potential: {potential.mean():,.0f} kWh/year "
          f"({potential.mean() / demand_total * 100:.1f} % of demand); "
          f"actually shared at rho = {config.RHO_BASELINE:.0%}: {shared.mean():,.0f} kWh/year "
          f"({shared.mean() / demand_total * 100:.1f} %), std {shared.std():,.0f} kWh")
    print(f"\nIncremental profit after {fixed_cost:,.0f} EUR fixed costs (mean, 5 %-95 %):")
    print(curve[curve.rho.round(2).isin([0, .05, .10, .15, .30, .45, .57, .75, 1.0])]
          .drop(columns="fixed_cost").round(2).to_string(index=False))
    print("\nBreak-even rho* (%) by fixed costs:")
    print((tipping.set_index("fixed_cost") * 100).round(2).loc[[0, 1000, 2000, 3000, 4000]].to_string())


# --- Sensitivität: Verhältnis von PV-Leistung zu Haushaltszahl ---
# Die PV-Anlage bleibt bei 500 kWp, nur die Zahl der Haushalte ändert sich (25 bis 400, also 20 bis 1,25 kWp je Haushalt).
# Die Sharing-Kosten wachsen mit der Haushaltszahl: c(N) = fester Anteil + Betrag je Haushalt (drei Szenarien, siehe config).
# Ein Teil der Kosten ist also fix (z. B. Steuerberatung), ein Teil skaliert mit den Haushalten (Plattform, Abrechnung).
# Die PV-Investition wird weiterhin nicht berücksichtigt (die Anlage ist bereits vorhanden).
RHO_PCT = round(config.RHO_BASELINE * 100)          # Referenz-Reduktion in Prozent, für die Spaltennamen (57)
SCENARIOS = ("low", "base", "high")                 # Kostenszenarien: niedrig, Baseline, hoch


def sharing_cost(n_households, scenario):
    """Jährliche Sharing-Kosten einer Community: fester Anteil plus Betrag je Haushalt (Szenario "low", "base" oder "high")."""
    k = SCENARIOS.index(scenario)
    return config.SHARING_COST_FIXED_EUR[k] + config.SHARING_COST_PER_HOUSEHOLD_EUR[k] * n_households


def break_even_rho(g, p, d, fixed_cost, rho_max=config.BREAK_EVEN_RHO_MAX):
    """Exakter Break-even für eine Nachfrage-Realisierung: das rho, bei dem der Zusatzgewinn genau den Fixkosten entspricht.

    Intervallhalbierung auf [0, rho_max]. Wird der Break-even selbst bei rho_max nicht erreicht, ist das Ergebnis NaN
    (kein künstlicher Wert, keine Extrapolation über rho_max hinaus).
    """
    if gross_gain(g, p, d, rho_max) < fixed_cost:
        return np.nan
    lo, hi = 0.0, rho_max
    for _ in range(60):             # 60-mal halbieren: die Genauigkeit liegt an der Rechengrenze
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gross_gain(g, p, d, mid) < fixed_cost else (lo, mid)
    return hi


def ratio_sensitivity(n_runs, household_counts=config.RATIO_HOUSEHOLD_COUNTS):
    """Rechnet Simulation A für verschiedene Haushaltszahlen bei gleicher PV-Anlage und gibt eine Tabelle zurück.

    Je Haushaltszahl: durchschnittlich geteilte Energie und Zusatzerlös bei der Referenz-Reduktion. Für die drei Kostenszenarien
    zusätzlich die Kosten, der Zusatzgewinn nach Kosten und die nötige Reduktion für den Break-even (gesucht im Bereich
    0 bis 100 %). Gibt es dort keinen Break-even, steht das im Status.
    """
    g = data.load_pvgis().to_numpy()
    p = data.load_smard().to_numpy()
    rows = []
    for n_hh in household_counts:
        # Nachfrage skaliert mit der Haushaltszahl; PV-Erzeugung und Preise bleiben unverändert.
        # Die Zufallszahlen sind bei gleichem Seed identisch, das relative Rauschen (ca. 12 %) bleibt also gleich:
        # die Nachfrage jedes Laufs ist (bis auf Rundung) Haushaltszahl/50 mal die Nachfrage der Baseline.
        # Annahme: Die relative Unsicherheit bleibt bewusst für alle Haushaltszahlen gleich. In der Realität würde sie
        # bei mehr Haushalten sinken, weil sich einzelne Abweichungen teilweise ausgleichen.
        annual_demand = n_hh * config.HOUSEHOLD_DEMAND_KWH
        d_plan = data.load_bdew_h0_profile(annual_kwh=annual_demand)
        revenue = np.zeros(n_runs)          # Zusatzerlös durch Sharing bei der Referenz-Reduktion
        shared = np.zeros(n_runs)           # tatsächlich geteilte Menge bei der Referenz-Reduktion
        costs = {s: sharing_cost(n_hh, s) for s in SCENARIOS}       # Kosten dieser Community je Szenario
        rho_star = {s: np.zeros(n_runs) for s in SCENARIOS}         # Break-even je Lauf (NaN = im Bereich nicht erreicht)
        for i in range(n_runs):
            d = data.realized_demand(d_plan, np.random.default_rng(config.RANDOM_SEED + i)).to_numpy()
            revenue[i] = gross_gain(g, p, d, config.RHO_BASELINE)
            shared[i] = shared_energy(g, p, d, config.RHO_BASELINE).sum()
            for s in SCENARIOS:
                rho_star[s][i] = break_even_rho(g, p, d, costs[s])

        row = {"households": n_hh, "pv_kwp": config.PV_PEAK_KW, "kwp_per_household": config.PV_PEAK_KW / n_hh,
               "annual_demand_kwh": annual_demand, "n_runs": n_runs,
               f"shared_kwh_rho{RHO_PCT}": shared.mean(),
               f"sharing_revenue_rho{RHO_PCT}": revenue.mean()}
        # Nur wenn alle Läufe den Break-even im Bereich erreichen, wird ein Mittelwert ausgewiesen
        max_pct = config.BREAK_EVEN_RHO_MAX * 100
        for s in SCENARIOS:
            n_ok = int((~np.isnan(rho_star[s])).sum())
            if n_ok == n_runs:
                break_even, status = np.nanmean(rho_star[s]) * 100, "ok"
            elif n_ok == 0:
                break_even, status = np.nan, f"No break-even within 0–{max_pct:.0f}%"
            else:
                break_even, status = np.nan, f"Break-even reached in only {n_ok} of {n_runs} runs within 0–{max_pct:.0f}%"
            row.update({f"cost_{s}": costs[s], f"net_gain_rho{RHO_PCT}_{s}": revenue.mean() - costs[s],
                        f"break_even_rho_pct_{s}": break_even, f"runs_with_break_even_{s}": n_ok,
                        f"break_even_status_{s}": status})
        rows.append(row)
    return pd.DataFrame(rows)


def main_ratio_sensitivity(n_runs):
    """Rechnet die Haushaltszahl-Sensitivität, speichert sie in results/ und gibt die Tabelle aus."""
    df = ratio_sensitivity(n_runs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "sim_a_ratio_sensitivity.csv", index=False)
    print(f"\nRatio sensitivity ({n_runs} runs per scenario, PV {config.PV_PEAK_KW:.0f} kWp, costs = fixed part + amount per household):")
    shown = ["households", "kwp_per_household", f"shared_kwh_rho{RHO_PCT}", f"sharing_revenue_rho{RHO_PCT}"]
    shown += [c for s in SCENARIOS for c in (f"cost_{s}", f"break_even_rho_pct_{s}")]
    print(df[shown].round(2).to_string(index=False))
    for s in SCENARIOS:
        flagged = df[df[f"break_even_status_{s}"] != "ok"]
        for _, r in flagged.iterrows():
            print(f"  {s} costs, {int(r.households)} households: {r[f'break_even_status_{s}']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ratio":        # python src/sim_a.py ratio
        main_ratio_sensitivity(config.N_RUNS_SIM_A)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else config.N_RUNS_SIM_A)
