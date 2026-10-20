# Das Dispatch-Modell: Wie setzt der Aggregator seinen PV-Strom stundenweise ein?
# Pro Stunde t gilt die Energiebilanz:  g + q = s + c + e + a
#   g = PV-Erzeugung, q = Entladung der Batterie, s = geteilt mit der Community,
#   c = Ladung der Batterie, e = Verkauf an der Börse, a = abgeregelt (nicht genutzt)
# Gewinn = Erlös aus Teilen + Erlös aus Börsenverkauf - Verschleiß.
# Für den Ladestand am Jahresende gibt es keine Vorgabe und keine Strafe; er wird nur ausgewiesen (soc_end).
# Nach dem 31.12. hat gespeicherter Strom keinen Wert mehr, deshalb leeren alle Verfahren die Batterie am Jahresende.
# Fixkosten und Batteriekosten werden nicht hier, sondern erst nach dem Lauf abgezogen.
#
# Drei Verfahren:
#   rolling_horizon (RH):        plant alle 4 Stunden neu für 24 Stunden, kennt die Nachfrage nur als Prognose
#   rolling_horizon (PF-RH):     wie RH, kennt aber die tatsächliche Nachfrage der nächsten 24 Stunden
#   oracle:                      plant das ganze Jahr auf einmal und kennt alles (Obergrenze)

import numpy as np
import pulp

import config

ETA = config.ROUND_TRIP_EFFICIENCY ** 0.5       # Wirkungsgrad je Richtung (Laden bzw. Entladen); zusammen = 0,93
NEG = config.NEG_EUR_PER_KWH
C_DEG = config.DEGRADATION_COST_EUR_PER_KWH
ALPHA = config.SHARING_BENEFIT_SHARE


def sharing_price(p, rho):
    """Preis für geteilten Strom: Börsenpreis plus 80 % der Netzentgelt-Reduktion (alpha * rho * Netzentgelt)."""
    return p + ALPHA * rho * NEG


def solve_lp(g, p, d, soc_start, sharing, rho, E, P):
    """Löst die Optimierung für ein Zeitfenster aus n Stunden und gibt den besten Einsatzplan zurück.

    g, p, d: PV-Erzeugung, Börsenpreis und (erwartete) Nachfrage je Stunde.
    soc_start: Ladestand zu Beginn. sharing: darf geteilt werden? rho: Netzentgelt-Reduktion.
    E, P: Kapazität (kWh) und Leistung (kW) der Batterie.
    """
    n = len(g)
    prob = pulp.LpProblem("dispatch", pulp.LpMaximize)

    # Entscheidungsvariablen pro Stunde (alle >= 0); Laden/Entladen höchstens P, Ladestand höchstens E
    e = [pulp.LpVariable(f"e{t}", 0) for t in range(n)]             # Verkauf an der Börse
    a = [pulp.LpVariable(f"a{t}", 0) for t in range(n)]             # abgeregelt
    c = [pulp.LpVariable(f"c{t}", 0, P) for t in range(n)]          # Ladung
    q = [pulp.LpVariable(f"q{t}", 0, P) for t in range(n)]          # Entladung
    soc = [pulp.LpVariable(f"soc{t}", 0, E) for t in range(n)]      # Ladestand am Ende der Stunde
    s = [pulp.LpVariable(f"s{t}", 0) for t in range(n)] if sharing else [0.0] * n    # geteilt (ohne Sharing immer 0)

    # Zielfunktion: möglichst hoher Gewinn
    price_s = sharing_price(p, rho)
    prob += pulp.lpSum(
        (price_s[t] * s[t] if sharing else 0) + p[t] * e[t] - C_DEG * (c[t] + q[t])
        for t in range(n))

    # Nebenbedingungen pro Stunde
    for t in range(n):
        prob += g[t] + q[t] == s[t] + c[t] + e[t] + a[t]            # Energiebilanz
        if sharing:
            prob += s[t] <= d[t]                                    # es kann nicht mehr geteilt werden, als nachgefragt wird
        prev = soc_start if t == 0 else soc[t - 1]
        prob += soc[t] == prev + ETA * c[t] - q[t] / ETA            # Ladestand: vorher + geladen - entladen (jeweils mit Verlusten)

    status = prob.solve(pulp.HiGHS(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[status]}")

    def values(variables):
        return np.array([0.0 if isinstance(x, float) else pulp.value(x) for x in variables])

    return {"s": values(s), "e": values(e), "c": values(c), "q": values(q),
            "soc_end": pulp.value(soc[n - 1]), "objective": pulp.value(prob.objective)}


def rolling_horizon(g, p, d_plan, d_real, sharing, rho, E, P, perfect_foresight=False):
    """Rolling Horizon über das ganze Jahr: alle 4 Stunden neu planen und die ersten 4 Stunden des Plans umsetzen.

    d_plan: erwartete Nachfrage (Profil), d_real: tatsächliche Nachfrage.
    perfect_foresight=True ist das Verfahren PF-RH: Der Plan nutzt die tatsächliche Nachfrage.
    Gibt je Stunde s, e, c, q sowie den End-Ladestand zurück.
    """
    n = len(g)
    soc0 = config.INITIAL_SOC_FRACTION * E          # Start-Ladestand (leer)
    out = {k: np.zeros(n) for k in "secq"}
    soc = soc0
    for start in range(0, n, config.RESOLVE_EVERY_HOURS):       # alle 4 Stunden ein neues Fenster
        end = min(start + config.HORIZON_HOURS, n)              # Planungsfenster: 24 Stunden
        commit = min(start + config.RESOLVE_EVERY_HOURS, n)     # umgesetzt werden die ersten 4 Stunden

        # Nachfrage, mit der geplant wird
        if sharing and perfect_foresight:
            d_fc = d_real[start:end]                            # PF-RH: tatsächliche Nachfrage
        elif sharing:
            # RH: Profil, korrigiert um die letzte beobachtete Abweichung (klingt mit phi^k ab)
            eps = d_real[start - 1] / d_plan[start - 1] - 1.0 if start > 0 else 0.0
            k = np.arange(1, end - start + 1)
            d_fc = d_plan[start:end] * (1.0 + config.AR1_PHI ** k * eps)
        else:
            d_fc = d_plan[start:end]                            # ohne Sharing spielt die Nachfrage keine Rolle

        sol = solve_lp(g[start:end], p[start:end], d_fc, soc, sharing, rho, E, P)

        # Umsetzung: Batterie macht, was geplant war; abgerechnet wird mit der tatsächlichen Nachfrage
        m = commit - start
        c, q = sol["c"][:m], sol["q"][:m]
        avail = g[start:commit] + q - c                         # Strom, der nach der Batterie noch übrig ist
        if sharing:
            ok = sharing_price(p[start:commit], rho) >= 0       # geteilt wird nur, wenn der Preis nicht negativ ist
            s = np.where(ok, np.minimum(avail, d_real[start:commit]), 0.0)
        else:
            s = np.zeros(m)
        e = np.where(p[start:commit] >= 0, avail - s, 0.0)      # Rest an die Börse; bei negativem Preis abregeln
        out["c"][start:commit], out["q"][start:commit] = c, q
        out["s"][start:commit], out["e"][start:commit] = s, e
        soc += (ETA * c - q / ETA).sum()                        # Ladestand für das nächste Fenster fortschreiben
    out["soc_end"] = soc
    return out


def oracle(g, p, d_real, rho, E, P):
    """Obergrenze: ein einziger Plan für das ganze Jahr mit Kenntnis der tatsächlichen Nachfrage."""
    soc0 = config.INITIAL_SOC_FRACTION * E
    return solve_lp(g, p, d_real, soc0, True, rho, E, P)


def profit(r, p, rho):
    """Jahresgewinn (brutto) eines Ergebnisses: Erlös aus Teilen + Börsenerlös - Verschleiß.

    Ohne Fixkosten des Sharings und ohne Batteriekosten; die werden erst später abgezogen.
    """
    return (sharing_price(p, rho) * r["s"]).sum() + (p * r["e"]).sum() - C_DEG * (r["c"] + r["q"]).sum()
