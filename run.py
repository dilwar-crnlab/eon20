from eon_simulador import Simulador
import simpy
from random import *
from config import *
import numpy as np

def main(args):
	for e in range(ERLANG_MIN, ERLANG_MAX+1, ERLANG_INC):
		print("-----------------------------")
		for rep in range(10):
			rate = e / HOLDING_TIME
			seed(RANDOM_SEED[rep])
			env = simpy.Environment()
			simulador = Simulador(env)
			env.process(simulador.Run(rate))
			env.run()
			print("Erlang", e, "Simulation...", rep)
			print("#Total Request", NUM_OF_REQUESTS, "Blocked", simulador.NumReqBlocked, "B_%", (simulador.NumReqBlocked/NUM_OF_REQUESTS)*100)
			
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
