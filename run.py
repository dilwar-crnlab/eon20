import matplotlib.pyplot as plt
from eon_simulador import PLIAwareRMSA
import simpy
from random import seed
from config import *
import numpy as np

def main(args):
    blocking_ratios = {}

    for e in range(ERLANG_MIN, ERLANG_MAX+1, ERLANG_INC):
        print("-----------------------------")
        blocking_ratios[e] = []
        for rep in range(REP):
            rate = e / HOLDING_TIME
            seed(RANDOM_SEED[rep])
            env = simpy.Environment()
            simulador = PLIAwareRMSA(env)
            env.process(simulador.Run(rate))
            env.run()
            blocking_ratio = (simulador.NumReqBlocked / NUM_OF_REQUESTS) * 100
            blocking_ratios[e].append(blocking_ratio)
            print(f"Erlang {e}, Simulation... {rep}")
            print(f"#Total Request {NUM_OF_REQUESTS}, Blocked {simulador.NumReqBlocked}, B_% {blocking_ratio:.2f}")

    # Calculate average blocking ratio for each Erlang value
    avg_blocking_ratios = {e: np.mean(ratios) for e, ratios in blocking_ratios.items()}

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(list(avg_blocking_ratios.keys()), list(avg_blocking_ratios.values()), marker='o')
    plt.xlabel('Erlang')
    plt.ylabel('Blocking Ratio (%)')
    plt.title('Average Blocking Ratio vs Erlang')
    plt.grid(True)
    plt.savefig('blocking_ratio_vs_erlang.png')
    plt.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))