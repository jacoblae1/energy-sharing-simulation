# Startet den ganzen Ablauf: Simulationen, Tests, Abbildungen.

import config
import plots
import sim_a
import sim_b
import tests_model


def main():
    sim_a.main(config.N_RUNS_SIM_A)
    sim_a.main_ratio_sensitivity(config.N_RUNS_SIM_A)
    sim_b.main(config.N_RUNS_SIM_B)
    tests_model.main()
    plots.main()


if __name__ == "__main__":
    main()
