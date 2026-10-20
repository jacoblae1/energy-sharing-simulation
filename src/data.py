# Lädt die drei Eingabereihen und erzeugt die Nachfrage-Unsicherheit:
#   g = PV-Erzeugung (PVGIS), p = Day-Ahead-Preis (SMARD), d = Nachfrage der Haushalte (BDEW H0).
# Alle Reihen sind stündlich, Zeitbezug UTC.

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import config

ROOT = Path(__file__).resolve().parent.parent
PVGIS_PATH = ROOT / "raw_data/pvgis/pvgis_wuerzburg_500kwp_2023.csv"
SMARD_PATH = ROOT / "raw_data/smard/smard_dayahead_DE_2023_hourly.csv"
BDEW_H0_PATH = ROOT / "raw_data/bdew_h0/Haushalt-Lastprofil.xls"

# Koeffizienten der BDEW-Dynamisierung (Polynom 4. Grades über den Tag im Jahr)
A4, A3, A2, A1, A0 = -3.92e-10, 3.2e-7, -7.02e-5, 2.1e-3, 1.24


def dynamization_factor(day_of_year: int) -> float:
    """Faktor für den Jahresgang des Verbrauchs (Tag 1 bis 365); mehr Verbrauch im Winter, weniger im Sommer."""
    t = day_of_year
    return A4 * t**4 + A3 * t**3 + A2 * t**2 + A1 * t + A0


def season_of(date: dt.date) -> str:
    """Jahreszeit nach BDEW: Winter 1.11.-20.3., Sommer 15.5.-14.9., sonst Übergangszeit."""
    md = (date.month, date.day)
    if md >= (11, 1) or md <= (3, 20):
        return "Winter"
    if (5, 15) <= md <= (9, 14):
        return "Sommer"
    return "Übergangszeit"


def daytype_of(date: dt.date) -> str:
    """Tagestyp nach BDEW: Werktag, Samstag oder Sonntag (Feiertage werden nicht gesondert behandelt)."""
    weekday = date.weekday()
    if weekday == 5:
        return "Samstag"
    if weekday == 6:
        return "Sonntag"
    return "Werktag"


def _check_hourly(series: pd.Series, name: str, year: int = config.YEAR) -> pd.Series:
    n_expected = 8784 if pd.Timestamp(year, 12, 31).is_leap_year else 8760
    if len(series) != n_expected:
        raise ValueError(f"{name}: expected {n_expected} hourly values, found {len(series)}")
    if series.isna().any():
        raise ValueError(f"{name}: contains missing values")
    steps = series.index.to_series().diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{name}: index is not a regular hourly series")
    return series


def _read_pvgis_peak_kw(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Nominal power"):
                return float(line.split(":")[1].strip())
    raise ValueError(f"{path}: nominal power not found in the PVGIS header")


def load_pvgis(path: Path = PVGIS_PATH) -> pd.Series:
    """PV-Erzeugung in kWh pro Stunde."""
    peak_kw = _read_pvgis_peak_kw(path)
    if abs(peak_kw - config.PV_PEAK_KW) > 1e-9:
        raise ValueError(f"PVGIS file is for {peak_kw} kWp, configuration expects {config.PV_PEAK_KW} kWp")
    df = pd.read_csv(path, skiprows=10, skipfooter=10, engine="python")
    idx = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
    g_t = pd.Series(df["P"].to_numpy() / 1000.0, index=idx, name="g_t_kwh")    # Quelle in W -> kWh pro Stunde
    return _check_hourly(g_t.sort_index(), "PVGIS generation")


def load_smard(path: Path = SMARD_PATH) -> pd.Series:
    """Day-Ahead-Preis in Euro pro kWh (die Quelle liefert Euro pro MWh); negative Preise bleiben erhalten."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["datetime_utc"])
    p_t = pd.Series(df["price_eur_mwh"].to_numpy() / 1000.0, index=idx, name="p_t_eur_kwh")
    return _check_hourly(p_t.sort_index(), "SMARD day-ahead price")


def _load_bdew_base_profiles(path: Path = BDEW_H0_PATH) -> dict[tuple[str, str], np.ndarray]:
    df = pd.read_excel(path, sheet_name="Profil", header=None)
    seasons = df.iloc[1, 1:10].ffill().tolist()
    daytypes = df.iloc[2, 1:10].tolist()
    values = df.iloc[3:99, 1:10].to_numpy(dtype=float)
    return {(s, d): values[:, i] for i, (s, d) in enumerate(zip(seasons, daytypes))}


def load_bdew_h0_profile(year: int = config.YEAR, annual_kwh: float = config.ANNUAL_DEMAND_KWH) -> pd.Series:
    """Erwartete Nachfrage der Community in kWh pro Stunde für das ganze Jahr (Profil ohne Zufall)."""
    base_profiles = _load_bdew_base_profiles()
    start = dt.date(year, 1, 1)
    n_days = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365

    # Pro Tag das passende Basisprofil wählen und mit dem Faktor für den Jahresgang multiplizieren
    quarter_hourly_w = []
    for day_idx in range(n_days):
        date = start + dt.timedelta(days=day_idx)
        key = (season_of(date), daytype_of(date))
        factor = dynamization_factor(day_idx + 1)
        quarter_hourly_w.append(base_profiles[key] * factor)
    quarter_hourly_w = np.concatenate(quarter_hourly_w)

    # Je 4 Viertelstunden zu einer Stunde mitteln (Watt -> kW, also kWh pro Stunde)
    hourly_kw = quarter_hourly_w.reshape(-1, 4).mean(axis=1) / 1000.0
    idx = pd.date_range(f"{year}-01-01", periods=len(hourly_kw), freq="h")
    d_t = pd.Series(hourly_kw, index=idx, name="d_t_bdew_kwh")

    # Das BDEW-Profil gilt in deutscher Ortszeit, PV und Preise in UTC: Nachfrage auf UTC verschieben
    # (Ortszeit = UTC + 1 Stunde im Winter, + 2 Stunden im Sommer)
    utc = pd.date_range(f"{year}-01-01", periods=len(d_t), freq="h", tz="UTC")
    offset = np.array([ts.utcoffset().total_seconds() / 3600 for ts in utc.tz_convert("Europe/Berlin")],
                      dtype=int)
    d_t = pd.Series(d_t.to_numpy()[np.minimum(np.arange(len(d_t)) + offset, len(d_t) - 1)],
                    index=d_t.index, name=d_t.name)

    d_t = d_t * (annual_kwh / d_t.sum())        # auf den Jahresverbrauch der 50 Haushalte skalieren
    if (d_t <= 0).any():
        raise ValueError("BDEW H0 profile contains non-positive hourly demand")
    return _check_hourly(d_t, "BDEW H0 demand profile", year)


def generate_ar1_noise(n_periods: int, rng: np.random.Generator,
                       phi: float = config.AR1_PHI,
                       innovation_std: float = config.AR1_INNOVATION_STD) -> np.ndarray:
    """Zufälliges Rauschen, das von Stunde zu Stunde zusammenhängt: eps_t = phi * eps_(t-1) + Zufall."""
    eps = np.zeros(n_periods)
    innovations = rng.normal(0.0, innovation_std, size=n_periods)       # der neue Zufallsanteil je Stunde
    eps[0] = innovations[0]
    for t in range(1, n_periods):
        eps[t] = phi * eps[t - 1] + innovations[t]
    return eps


def realized_demand(d_bdew: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Tatsächliche Nachfrage = erwartetes Profil * (1 + Rauschen); wird nie negativ."""
    eps = generate_ar1_noise(len(d_bdew), rng)
    d_t = d_bdew.to_numpy() * (1.0 + eps)
    d_t = np.clip(d_t, 0.0, None)
    return pd.Series(d_t, index=d_bdew.index, name="d_t_kwh")


if __name__ == "__main__":
    g = load_pvgis()
    p = load_smard()
    d_base = load_bdew_h0_profile()
    print("g_t:", len(g), "hours, annual sum:", round(g.sum()), "kWh")
    print("p_t:", len(p), "hours, mean:", round(p.mean(), 4), "EUR/kWh")
    print("d_t_BDEW:", len(d_base), "hours, annual sum:", round(d_base.sum()), "kWh")

    rng = np.random.default_rng(config.RANDOM_SEED)
    d = realized_demand(d_base, rng)
    print("d_t (one realization): annual sum:", round(d.sum()), "kWh")
