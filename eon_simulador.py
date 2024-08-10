import simpy
from random import *
from config import *
import numpy as np
import networkx as nx
import math
from itertools import islice

topology = nx.read_weighted_edgelist('topology/' + TOPOLOGY, nodetype=int)

class Desalocate(object):
	def __init__(self, env):
		self.env = env
	def Run(self, count, path, spectro, holding_time):
		global topology
		yield self.env.timeout(holding_time)
		for i in range(0, (len(path)-1)):
			for slot in range(spectro[0],spectro[1]+1):
				topology[path[i]][path[i+1]]['capacity'][slot] = 0

class Simulador(object):
	def __init__(self, env):
		self.env = env
		global topology
		for u, v in list(topology.edges):
			topology[u][v]['capacity'] = [0] * SLOTS
		self.nodes = list(topology.nodes())
		self.random = Random()
		self.NumReqBlocked = 0 
		self.cont_req = 0
		self.k_paths = {}

	def Run(self, rate):
		global topology
		for i in list(topology.nodes()):
			for j in list(topology.nodes()):
				if i!= j:
					self.k_paths[i,j] = self.k_shortest_paths(topology, i, j, N_PATH, weight='weight')

	# Calculates the path distance according to the edge weights            
	def Distance(self, path):
		global topology 
		soma = 0
		for i in range(0, (len(path)-1)):
			soma += topology[path[i]][path[i+1]]['weight']
		return (soma)

	#Calculates the k-shortest paths between o-d pairs
	def k_shortest_paths(self,G, source, target, k, weight='weight'):
		return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

	#Calculates the modulation format according to the path distance   
	def Modulation(self, dist, demand):
		if dist <= 500:
			return (float(demand) / float(4 * SLOT_SIZE))
		elif 500 < dist <= 1000:
			return (float(demand) / float(3 * SLOT_SIZE))
		elif 1000 < dist <= 2000:
			return (float(demand) / float(2 * SLOT_SIZE)) 
		else:
			return (float(demand) / float(1 * SLOT_SIZE))


class QLearningRouter(Simulador):
    def __init__(self, env):
        super().__init__(env)
        self.q_table = {}
        self.initialize_q_table()

    def initialize_q_table(self):
        for node in self.nodes:
            self.q_table[node] = {neighbor: 0 for neighbor in topology[node]}

    def get_action(self, state, possible_actions):
        if random() < EPSILON:
            return choice(possible_actions)
        else:
            return max(possible_actions, key=lambda a: self.q_table[state][a])

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_next_q)
        self.q_table[state][action] = new_q

    def find_k_paths(self, source, destination):
        paths = []
        for _ in range(Q_LEARNING_EPISODES):
            path = [source]
            current_node = source
            visited = set([source])
            while current_node != destination:
                possible_actions = [node for node in topology[current_node] if node not in visited]
                if not possible_actions:
                    break  # No valid moves, abandon this path
                action = self.get_action(current_node, possible_actions)
                reward = 1 / topology[current_node][action]['weight']
                self.update_q_value(current_node, action, reward, action)
                path.append(action)
                visited.add(action)
                current_node = action
            if current_node == destination:
                paths.append(tuple(path))
        return sorted(set(paths), key=len)[:K_PATHS]

class PLIAwareRMSA(QLearningRouter):
    def __init__(self, env):
        super().__init__(env)

    def calculate_fragmentation(self, path):
        total_free_slots = 0
        total_free_blocks = 0
        current_block = 0
        for i in range(len(path) - 1):
            link_capacity = topology[path[i]][path[i+1]]['capacity']
            for slot in link_capacity:
                if slot == 0:
                    total_free_slots += 1
                    current_block += 1
                else:
                    if current_block > 0:
                        total_free_blocks += 1
                        current_block = 0
            if current_block > 0:
                total_free_blocks += 1
        if total_free_slots == 0:
            return 1  # Fully fragmented
        return 1 - (total_free_blocks / total_free_slots)

    def find_feasible_spectrum(self, path, num_slots):
        for i in range(SLOTS - num_slots + 1):
            if all(topology[path[j]][path[j+1]]['capacity'][i:i+num_slots] == [0]*num_slots 
                   for j in range(len(path)-1)):
                return i, i + num_slots - 1
        return None

    def Run(self, rate):
        for count in range(1, NUM_OF_REQUESTS + 1):
            yield self.env.timeout(self.random.expovariate(rate))
            src, dst = self.random.sample(self.nodes, 2)
            bandwidth = self.random.choice(BANDWIDTH)
            holding_time = self.random.expovariate(HOLDING_TIME)
            print("Request", count, "(",src, dst, bandwidth, holding_time,")" )

            paths = self.find_k_paths(src, dst)
            print("paths",paths)
            for path in paths:
                distance = int(self.Distance(path))
                num_slots = int(math.ceil(self.Modulation(distance, bandwidth)))
                spectrum = self.find_feasible_spectrum(path, num_slots)

                if spectrum:
                    self.allocate_path(path, spectrum, count)
                    fragmentation = self.calculate_fragmentation(path)
                    reward = 1 - fragmentation  # Higher reward for less fragmentation
                    self.update_q_values(path, reward)
                    desalocate = Desalocate(self.env)
                    self.env.process(desalocate.Run(count, path, spectrum, holding_time))
                    print("Request", count, "established")
                    break
					
            else:
                self.NumReqBlocked += 1
                print("Request", count, "blocked")

    def allocate_path(self, path, spectrum, count):
        for i in range(len(path) - 1):
            for slot in range(spectrum[0], spectrum[1] + 1):
                topology[path[i]][path[i+1]]['capacity'][slot] = count

    def update_q_values(self, path, reward):
        for i in range(len(path) - 1):
            self.update_q_value(path[i], path[i+1], reward, path[i+1])