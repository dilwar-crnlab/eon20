# import matplotlib.pyplot as plt
# from eon_simulador import PLIAwareRMSA
# import simpy
# from random import seed
# from config import *
# import numpy as np

# def main(args):
#     blocking_ratios = {}

#     for e in range(ERLANG_MIN, ERLANG_MAX+1, ERLANG_INC):
#         print("-----------------------------")
#         blocking_ratios[e] = []
#         for rep in range(REP):
#             rate = e / HOLDING_TIME
#             seed(RANDOM_SEED[rep])
#             env = simpy.Environment()
#             simulador = PLIAwareRMSA(env)
#             env.process(simulador.Run(rate))
#             env.run()
#             blocking_ratio = (simulador.NumReqBlocked / NUM_OF_REQUESTS) * 100
#             blocking_ratios[e].append(blocking_ratio)
#             print(f"Erlang {e}, Simulation... {rep}")
#             print(f"#Total Request {NUM_OF_REQUESTS}, Blocked {simulador.NumReqBlocked}, B_% {blocking_ratio:.2f}")

#     # Calculate average blocking ratio for each Erlang value
#     avg_blocking_ratios = {e: np.mean(ratios) for e, ratios in blocking_ratios.items()}

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(list(avg_blocking_ratios.keys()), list(avg_blocking_ratios.values()), marker='o')
#     plt.xlabel('Erlang')
#     plt.ylabel('Blocking Ratio (%)')
#     plt.title('Average Blocking Ratio vs Erlang')
#     plt.grid(True)
#     plt.savefig('blocking_ratio_vs_erlang.png')
#     plt.show()

#     return 0

# if __name__ == '__main__':
#     import sys
#     sys.exit(main(sys.argv))

import multiprocessing
from eon_simulador import PLIAwareRMSA
import simpy
from random import seed
from config import *
import numpy as np
import matplotlib.pyplot as plt
import datetime


def run_simulation(e, rep):
    rate = e / HOLDING_TIME
    seed(RANDOM_SEED[rep])
    env = simpy.Environment()
    simulador = PLIAwareRMSA(env)
    env.process(simulador.Run(rate))
    env.run()
    blocking_ratio = (simulador.NumReqBlocked / NUM_OF_REQUESTS) * 100
    print(f"Erlang {e}, Simulation... {rep}")
    print(f"#Total Request {NUM_OF_REQUESTS}, Blocked {simulador.NumReqBlocked}, B_% {blocking_ratio:.2f}")
    return e, blocking_ratio

def main(args):
    pool = multiprocessing.Pool()
    results = []

    for e in range(ERLANG_MIN, ERLANG_MAX+1, ERLANG_INC):
        for rep in range(REP):
            results.append(pool.apply_async(run_simulation, (e, rep)))

    pool.close()
    pool.join()

    blocking_ratios = {}
    for result in results:
        e, ratio = result.get()
        if e not in blocking_ratios:
            blocking_ratios[e] = []
        blocking_ratios[e].append(ratio)

    # Calculate average blocking ratio for each Erlang value
    avg_blocking_ratios = {e: np.mean(ratios) for e, ratios in blocking_ratios.items()}

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(list(avg_blocking_ratios.keys()), list(avg_blocking_ratios.values()), marker='o')
    plt.xlabel('Erlang')
    plt.ylabel('Blocking Ratio (%)')
    plt.title('Average Blocking Ratio vs Erlang')
    plt.grid(True)
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'blocking_ratio_vs_erlang_{current_datetime}.png'
    plt.savefig(filename)
    plt.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))