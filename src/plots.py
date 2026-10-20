# Erzeugt die Abbildungen der Arbeit aus den gespeicherten Ergebnissen.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

import config
import data
import sim_b

RES = data.ROOT / "results"
OUT = data.ROOT / "figures"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 200,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd8a2a", "#3a9a5b", "#777777", "#c0392b"
DARKGREEN = "#1e6b3a"
DARKBLUE = "#1f3f7a"
SOFT_ORANGE = "#e8b36e"
LIGHT_ORANGE = "#fdf1de"
LOW, HIGH = config.SHARING_FIXED_COST_RANGE
BASE = config.SHARING_FIXED_COST_EUR_PER_YEAR
CAPEX_LOW, CAPEX_HIGH = config.BATTERY_CAPEX_EUR_PER_KWH
RHO_REF_PCT = config.RHO_BASELINE * 100
eur = lambda v: f"€{v:,.0f}"
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
N_SIZES = len(config.BATTERY_SIZES_KWH)
BAR_LABEL_OFFSET_EUR = 2500


def save(fig, name):
    assert all(ax.get_title() == "" for ax in fig.axes) and fig._suptitle is None, "figure contains a title"
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved:", name)


def summary():
    return pd.read_csv(RES / "sim_b_summary.csv").iloc[:N_SIZES]


def fig_example_week():
    g, p, d = data.load_pvgis(), data.load_smard(), data.load_bdew_h0_profile()
    a = 24 * 190
    idx = d.index[a:a + 168]
    gg = pd.Series(g.to_numpy()[a:a + 168], index=idx)
    pp = pd.Series(p.to_numpy()[a:a + 168], index=idx)
    dd = d.iloc[a:a + 168]
    s = np.minimum(gg.to_numpy(), dd.to_numpy())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.fill_between(gg.index, gg, color=ORANGE, alpha=0.35, label="PV generation")
    a1.plot(dd.index, dd, color=BLUE, lw=1.4, label="Community demand")
    a1.fill_between(gg.index, s, color=GREEN, alpha=0.6, label="Potentially shared electricity")
    a1.set_ylabel("Energy per hour (kWh)")
    a1.set_ylim(0, 500)
    a1.legend(loc="upper right", fontsize=8)
    a2.plot(pp.index, pp * 100, color=GREY)
    a2.axhline(0, color="black", lw=0.8)
    a2.set_ylabel("Day-ahead electricity\nprice (ct/kWh)")
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a\n%d %b"))
    save(fig, "fig1_example_week")


def fig_additional_profit():
    df = pd.read_csv(RES / "sim_a_rho.csv")
    c_base = float(df["fixed_cost"].iloc[0])
    rho = df.rho.to_numpy() * 100
    gross = (df["mean"] + c_base).to_numpy()
    be = float(np.interp(c_base, gross, rho))
    tip = pd.read_csv(RES / "sim_a_fixkosten.csv")
    be_ref = float(tip.loc[tip.fixed_cost == c_base, "rho_star_mean"].iloc[0]) * 100
    assert abs(be - be_ref) < 0.1, (be, be_ref)
    print(f"  Fig. 2 break-even at baseline: {be:.2f}% (cross-check sim_a_fixkosten: {be_ref:.2f}%)")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rho, gross, color=BLUE, lw=2, label="Incremental sharing revenue")
    ax.fill_between(rho, df.p5 + c_base, df.p95 + c_base, color=BLUE, alpha=0.2)
    for c, col, lab, lw, weight in ((LOW, ORANGE, "", 1.2, "normal"), (c_base, GREEN, " (baseline)", 2.4, "bold"),
                                    (HIGH, RED, "", 1.2, "normal")):
        ax.axhline(c, color=col, ls="--", lw=lw)
        ax.text(1, c + 60, f"Annual costs {eur(c)}{lab}", color=col, fontsize=8, fontweight=weight)
    ax.axvline(RHO_REF_PCT, color=GREY, ls=":")
    ax.text(RHO_REF_PCT + 1, 1000, f"Reference scenario: {config.RHO_BASELINE:.0%}", ha="left", color=GREY, fontsize=8)
    ax.plot([be], [c_base], "o", ms=4.5, color=DARKBLUE, zorder=5)
    ax.text(min(be, RHO_REF_PCT) - 1.0, c_base + 0.11 * c_base, f"Break-even: {be:.1f}%", ha="right", va="bottom", fontsize=9,
            color="#222222")
    ax.set_xlabel("Grid-fee reduction ρ on shared electricity (%)")
    ax.set_ylabel("Incremental annual revenue (€)")
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig2_additional_profit_rho")


def fig_breakeven_fixed_costs():
    df = pd.read_csv(RES / "sim_a_fixkosten.csv")
    base_be = float(df.loc[df.fixed_cost == BASE, "rho_star_mean"].iloc[0]) * 100
    print(f"  Fig. A.1 break-even at baseline: {base_be:.2f}%")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(LOW, HIGH, color=ORANGE, alpha=0.18, label=f"Sensitivity range of annual costs ({eur(LOW)}–{HIGH:,.0f})")
    ax.fill_between(df.fixed_cost, df.p5 * 100, df.p95 * 100, color=BLUE, alpha=0.25)
    ax.plot(df.fixed_cost, df.rho_star_mean * 100, color=BLUE, lw=2, label="Break-even ρ* (mean over demand runs)")
    ax.axvline(BASE, color=GREEN, ls="--", lw=1.2, label=f"Baseline {eur(BASE)}")
    ax.axhline(RHO_REF_PCT, color=GREY, ls=":")
    ax.text(df.fixed_cost.max() - 60, RHO_REF_PCT + 1.0, f"Reference scenario: {config.RHO_BASELINE:.0%}", color=GREY, fontsize=8, ha="right")
    ax.set_xlabel("Annual costs of the sharing operation (€ per year)")
    ax.set_ylabel("Required grid-fee reduction ρ* (%)")
    ax.xaxis.set_major_formatter(thousands)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "figA1_breakeven_fixed_costs")


def fig_battery_value_cost():
    s = summary()
    x = np.arange(len(s))
    w = 0.40
    value, c_low, c_high = s.value_with_sharing, s[sim_b.COST_COLS[0]], s[sim_b.COST_COLS[1]]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.bar(x - w / 2, value, w, color=GREEN, label="Incremental operating profit (before battery costs)")
    ax.bar(x + w / 2, c_low, w, color=RED, alpha=0.5, label=f"Annualized battery costs at €{CAPEX_LOW:.0f}/kWh")
    with plt.rc_context({"hatch.linewidth": 0.5}):
        ax.bar(x + w / 2, c_high - c_low, w, bottom=c_low, color=LIGHT_ORANGE, edgecolor=SOFT_ORANGE, hatch="////", lw=0.6,
               label=f"Additional cost range up to €{CAPEX_HIGH:.0f}/kWh")
    for xi, v in zip(x, value):
        ax.text(xi - w / 2, v + BAR_LABEL_OFFSET_EUR, f"{v / 1000:,.1f}k", ha="center", fontsize=7.5, color=DARKGREEN)
    ax.set_xticks(x, [f"{int(e)} / {p:g}" for e, p in zip(s.E_kWh, s.P_kW)])
    ax.set_xlabel("Battery capacity (kWh) / Power rating (kW)")
    ax.set_ylabel("Annual operating value and battery costs (€)")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylim(0, c_high.max() * 1.1)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig4_battery_value_vs_cost")


def fig_battery_breakeven_price():
    s = summary()
    factor = sim_b.annual_cost(1.0, 1.0)
    be = s.value_with_sharing / (s.E_kWh * factor)
    print("  Fig. 5 break-even €/kWh:", [round(v) for v in be])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(CAPEX_LOW, CAPEX_HIGH, color=RED, alpha=0.15,
               label=f"Assumed investment cost range ({CAPEX_LOW:.0f}–{CAPEX_HIGH:.0f} €/kWh)")
    ax.plot(s.E_kWh, be, "o-", color=BLUE, lw=2, label="Break-even investment cost")
    for x, y in zip(s.E_kWh, be):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(s.E_kWh, [str(int(e)) for e in s.E_kWh])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Battery capacity (kWh)")
    ax.set_ylabel("Investment cost (€/kWh)")
    ax.set_ylim(0, 900)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig5_battery_breakeven_price")


def fig_breakeven_with_battery():
    s = summary()
    x = np.arange(len(s))
    lower, base, upper = (s[f"rho_star_battery_c{int(c)}"].to_numpy() for c in (LOW, BASE, HIGH))
    assert not (np.isnan(lower).any() or np.isnan(base).any() or np.isnan(upper).any()), "missing break-even values"
    nobatt = float(s[f"rho_star_no_battery_c{int(BASE)}"].iloc[0])
    print("  Fig. 6 baseline (2,000):", [round(v, 1) for v in base], "| band low:", [round(v, 1) for v in lower],
          "| band high:", [round(v, 1) for v in upper], "| no battery:", round(nobatt, 1))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(x, lower, upper, color=GREEN, alpha=0.22, lw=0, label=f"{eur(LOW)}–{eur(HIGH)} sensitivity range")
    ax.plot(x, base, "o-", color=DARKGREEN, lw=2.6, ms=6, label=f"{eur(BASE)} baseline")
    ax.axhline(nobatt, color="#444444", ls="--", lw=1.4, label=f"No battery ({eur(BASE)} annual costs)")
    ax.set_xticks(x, [f"{int(e)}" for e in s.E_kWh])
    ax.set_xlabel("Battery capacity (kWh)")
    ax.set_ylabel("Required grid-fee reduction (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig6_breakeven_with_battery")


def fig_gap_decomposition():
    s = summary()
    x = np.arange(len(s))
    f, h, t = s.forecast_gap, s.horizon_gap, s.total_gap
    assert np.allclose(f + h, t), "decomposition does not add up"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if (f >= 0).all() and (h >= 0).all():
        ax.bar(x, f, 0.55, color=BLUE, label="Forecast gap")
        ax.bar(x, h, 0.55, bottom=f, color=ORANGE, label="Horizon gap")
        for xi, ti in zip(x, t):
            ax.text(xi, ti + 0.02 * t.max(), eur(ti), ha="center", fontsize=8)
    else:
        w = 0.26
        ax.bar(x - w, f, w, color=BLUE, label="Forecast gap")
        ax.bar(x, h, w, color=ORANGE, label="Horizon gap")
        ax.bar(x + w, t, w, color=GREY, label="Total gap")
        ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"{int(e)}" for e in s.E_kWh])
    ax.set_xlabel("Battery capacity (kWh)")
    ax.set_ylabel("Annual optimality gap (€)")
    ax.set_ylim(top=t.max() * 1.15)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig7_optimality_gap_decomposition")


def fig_community_size():
    df = pd.read_csv(RES / "sim_a_ratio_sensitivity.csv")
    top = config.BREAK_EVEN_RHO_MAX * 100
    be = {s: df[f"break_even_rho_pct_{s}"] for s in ("low", "base", "high")}
    print("  Fig. 3 break-even by households (low / base / high costs):",
          {int(h): tuple(None if np.isnan(be[s][i]) else round(be[s][i], 1) for s in be) for i, h in enumerate(df.households)})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(df.households, be["low"].fillna(top), be["high"].fillna(top), color=BLUE, alpha=0.2, lw=0,
                    label="Cost sensitivity range")
    ok = be["base"].notna()
    ax.plot(df.households[ok], be["base"][ok], "o-", color=BLUE, lw=2, ms=6, label="Baseline cost scenario")
    for x, y in zip(df.households[ok], be["base"][ok]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(df.households, [str(int(h)) for h in df.households])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Number of households")
    ax.set_ylabel("Required grid-fee reduction (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig3_community_size_sensitivity")


def main():
    fig_example_week()
    fig_additional_profit()
    fig_community_size()
    fig_battery_value_cost()
    fig_battery_breakeven_price()
    fig_breakeven_with_battery()
    fig_gap_decomposition()
    fig_breakeven_fixed_costs()


if __name__ == "__main__":
    main()
