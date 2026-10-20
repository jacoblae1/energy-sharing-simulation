# --- Standort und PV-Anlage (die Erzeugungsdaten kommen aus PVGIS) ---

LAT, LON = 49.7913, 9.9534                                  # Würzburg
PV_PEAK_KW = 500.0                                          # Nennleistung der PV-Anlage
PV_SLOPE_DEG, PV_AZIMUTH_DEG = 36, -4                       # Neigung und Ausrichtung der Module (PVGIS-Optimum)
PV_SYSTEM_LOSS_PCT = 14.0                                   # Systemverluste der PV-Anlage
YEAR = 2023                                                 # Simulationsjahr für alle Daten

# --- Nachfrage der Community ---

N_HOUSEHOLDS = 50
HOUSEHOLD_DEMAND_KWH = 3_500.0                              # Jahresverbrauch pro Haushalt
ANNUAL_DEMAND_KWH = N_HOUSEHOLDS * HOUSEHOLD_DEMAND_KWH     # Jahresverbrauch der ganzen Community
RATIO_HOUSEHOLD_COUNTS = (25, 50, 100, 200, 400)            # Haushaltszahlen der Sensitivität (die PV-Leistung bleibt konstant)

# --- Batterie ---

C_RATE = 0.25                                               # Leistung = C_RATE * Kapazität (0,25 -> in 4 Stunden leer)
ROUND_TRIP_EFFICIENCY = 0.93                                # Wirkungsgrad für Laden und Entladen zusammen
DEGRADATION_COST_EUR_PER_KWH = 0.0                          # Verschleiß pro kWh im Dispatch; 0, weil die Anschaffung separat als Jahreskosten zählt
INITIAL_SOC_FRACTION = 0.0                                  # Start-Ladestand: leer; am Jahresende gibt es keine Vorgabe, die Batterie endet leer, ohne Startbonus
BATTERY_SIZES_KWH = (100, 250, 500, 1000, 2000)             # untersuchte Kapazitäten (Simulation B)
EXTRA_POWER_CONFIGS = ((500, 250.0), (500, 500.0))          # zusätzlich (Kapazität kWh, Leistung kW): 500 kWh mit 0,5 C und 1 C
BATTERY_CAPEX_EUR_PER_KWH = (450.0, 800.0)                  # Investitionskosten pro kWh: untere und obere Annahme
BATTERY_LIFETIME_YEARS = 15                                 # Nutzungsdauer
BATTERY_WACC = 0.037                                        # Zinssatz für die Jahreskosten (Annuität)
BATTERY_OPEX_SHARE = 0.02                                   # Betriebskosten pro Jahr als Anteil der Investition

# --- Netzentgelt und Sharing ---

NEG_EUR_PER_KWH = 0.0735                                    # Netzentgelt in Euro pro kWh (Stadtwerke Dettelbach 2026)
RHO_BASELINE = 0.57                                         # Referenzwert für die Netzentgelt-Reduktion (Österreich)
SHARING_BENEFIT_SHARE = 0.8                                 # Anteil der Reduktion für den Aggregator (80 %), 20 % bleiben bei den Haushalten
RHO_SWEEP = (0.0, 1.0)                                      # Bereich der Reduktion in Simulation A (0 bis 100 %)
RHO_GRID_N = 101
BREAK_EVEN_RHO_MAX = 1.0                                    # Obergrenze der Break-even-Suche in der Haushaltszahl-Sensitivität (100 %)
RHO_SUPPORT_POINTS = (0.05, 0.10, 0.20, 0.35, 0.57, 0.75, 1.0)   # Stützstellen für rho in Simulation B

# Kosten des Sharing-Betriebs pro Jahr: fester Anteil plus Betrag je Haushalt, also c(N) = F + k * N.
# Jeweils (niedrig, Baseline, hoch); die Baseline liegt in der Mitte der Spannen.
# Die Messstellenentgelte der Haushalte (MsbG § 30) zahlen Haushalt und Netzbetreiber, sie gehören nicht dazu.

SHARING_COST_FIXED_EUR = (500.0, 750.0, 1000.0)             # fester Anteil pro Jahr, z. B. Steuerberatung (500 bis 1.000 €)
SHARING_COST_PER_HOUSEHOLD_EUR = (30.0, 45.0, 60.0)         # Betrag je Haushalt und Jahr, z. B. Plattform und Abrechnung (30 bis 60 €)

# Gesamtkosten für die Baseline-Community (50 Haushalte): Baseline 3.000 €, Spanne 2.000 bis 4.000 €

SHARING_FIXED_COST_EUR_PER_YEAR = SHARING_COST_FIXED_EUR[1] + SHARING_COST_PER_HOUSEHOLD_EUR[1] * N_HOUSEHOLDS
SHARING_FIXED_COST_RANGE = (SHARING_COST_FIXED_EUR[0] + SHARING_COST_PER_HOUSEHOLD_EUR[0] * N_HOUSEHOLDS,
                            SHARING_COST_FIXED_EUR[2] + SHARING_COST_PER_HOUSEHOLD_EUR[2] * N_HOUSEHOLDS)
FIXED_COST_GRID_MAX_EUR = 4500.0
FIXED_COST_GRID_STEP_EUR = 100.0

# --- Unsicherheit der Nachfrage: tatsächliche Nachfrage = Profil * (1 + Rauschen) ---

AR1_PHI = 0.7                                               # Rauschen hängt zu 70 % vom Rauschen der Vorstunde ab (Autokorrelation)
AR1_INNOVATION_STD = 0.086                                  # Streuung des neuen Zufallsanteils je Stunde (ergibt ca. 12 % insgesamt)

# --- Rolling Horizon ---

HORIZON_HOURS = 24                                          # so viele Stunden plant der Optimierer voraus (Preise gelten im Fenster als bekannt)
RESOLVE_EVERY_HOURS = 4                                     # alle 4 Stunden neu planen; so viele Stunden des Plans werden umgesetzt

# --- Simulation ---

N_RUNS_SIM_A = 100                                          # Monte-Carlo-Läufe in Simulation A
N_RUNS_SIM_B = 20                                           # Monte-Carlo-Läufe je Einstellung in Simulation B
RANDOM_SEED = 42                                            # Startwert; Lauf i verwendet RANDOM_SEED + i
N_WORKERS = 8
