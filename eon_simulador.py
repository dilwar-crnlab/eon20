import simpy
from random import *
from config import *
import numpy as np
import networkx as nx
import math
from itertools import islice
from tqdm import tqdm
import heapq


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
        self.modulation_table = {
            (0, 560): 4,
            (561, 1360): 3,
            (1360, 2720): 2,
            (2721, 5520): 1
        }
        self.closeness_centrality = nx.closeness_centrality(topology)

    def Run(self, rate):
        global topology
        for i in list(topology.nodes()):
            for j in list(topology.nodes()):
                if i!= j:
                    self.k_paths[i,j] = self.k_shortest_paths(topology, i, j, N_PATH, weight='weight')

    def Distance(self, path):
    	return nx.path_weight(topology, path, weight='weight')

    def k_shortest_paths(self,G, source, target, k, weight='weight'):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

	#Calculates the modulation format according to the path distance   
    def Modulation(self, dist, demand):
        if dist <= 560:
            return (float(demand) / float(4 * SLOT_SIZE))
        elif 560 < dist <= 1360:
            return (float(demand) / float(3 * SLOT_SIZE))
        elif 1360 < dist <= 2720:
            return (float(demand) / float(2 * SLOT_SIZE)) 
        else:
            return (float(demand) / float(1 * SLOT_SIZE))


class QLearningRouter(Simulador):
    def __init__(self, env):
        super().__init__(env)
        self.q_table = {}
        self.initialize_q_table()
        self.distances = {}
        self.calculate_all_distances()
        self.c_table = {}  # New confidence table
        

    def initialize_q_table(self):
        for node in self.nodes:
            self.q_table[node] = {neighbor: 0 for neighbor in topology[node]}
            self.c_table[node] = {neighbor: 0.5 for neighbor in topology[node]}  # Initialize confidence to 0.5

    def get_action(self, state, possible_actions):
        if random() < EPSILON:
            return choice(possible_actions)
        else:
            return max(possible_actions, key=lambda a: self.q_table[state][a])

    def calculate_learning_rate(self, c_old, c_est):
        return max(c_est, 1 - c_old)

    def calculate_all_distances(self):
        self.distances = {}
        for source in self.nodes:
            for target in self.nodes:
                if source == target:
                    self.distances[(source, target)] = 0
                else:
                    self.distances[(source, target)] = nx.shortest_path_length(topology, source, target, weight='weight')

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        c_old = self.c_table[state][action]
        c_est = max(self.c_table[next_state].values())
        alpha = self.calculate_learning_rate(c_old, c_est)
        new_q = current_q + alpha * (reward + GAMMA * max_next_q - current_q)
        self.q_table[state][action] = new_q

        # Update confidence value
        self.c_table[state][action] = c_old + alpha * (1 - c_old)

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
                
                
                link_length = 1 / topology[current_node][action]['weight']
                link_fragmentation = self.calculate_link_fragmentation(current_node, action)
                link_load = self.calculate_link_load(current_node, action)

                closeness_factor = 1 - self.closeness_centrality[action]  # Invert the closeness centrality
                reward = link_length + (1-link_load) + (1-link_fragmentation) + closeness_factor   
                self.update_q_value(current_node, action, reward, action)
                path.append(action)
                visited.add(action)
                current_node = action
            if current_node == destination:
                paths.append(tuple(path))
        return sorted(set(paths), key=len)[:K_PATHS]



    
    # Combine distance and fragmentation
        return distance + (1 - avg_fragmentation)

    def calculate_link_load(self, node1, node2):
        link_capacity = topology[node1][node2]['capacity']
        return sum(1 for slot in link_capacity if slot != 0) / len(link_capacity)


    def calculate_link_fragmentation(self, node1, node2):
        link_capacity = topology[node1][node2]['capacity']
        S = len(link_capacity)  # Total number of spectrum slots
        free_fragments = []
        current_fragment = 0
        # Identify free fragments
        for slot in link_capacity:
            if slot == 0:
                current_fragment += 1
            else:
                if current_fragment > 0:
                    free_fragments.append(current_fragment)
                    current_fragment = 0
        if current_fragment > 0:
            free_fragments.append(current_fragment)
        # Calculate entropy-based fragmentation for this link
        link_fragmentation = 0
        for w in free_fragments:
            link_fragmentation += (w / S) * math.log(S / w)
        return link_fragmentation

class PLIAwareRMSA(QLearningRouter):
    def __init__(self, env):
        super().__init__(env)

    def calculate_fragmentation(self, path):
        total_fragmentation = 0
        num_links = len(path) - 1
        for i in range(num_links):
            link_capacity = topology[path[i]][path[i+1]]['capacity']
            S = len(link_capacity)  # Total number of spectrum slots
            free_fragments = []
            current_fragment = 0
            # Identify free fragments
            for slot in link_capacity:
                if slot == 0:
                    current_fragment += 1
                else:
                    if current_fragment > 0:
                        free_fragments.append(current_fragment)
                        current_fragment = 0
            if current_fragment > 0:
                free_fragments.append(current_fragment)
            # Calculate entropy-based fragmentation for this link
            link_fragmentation = 0
            for w in free_fragments:
                link_fragmentation += (w / S) * math.log(S / w)
            total_fragmentation += link_fragmentation
        # Average fragmentation across all links in the path
        return total_fragmentation / num_links if num_links > 0 else 0

    def find_feasible_spectrum(self, path, num_slots):
        for i in range(SLOTS - num_slots + 1):
            if all(topology[path[j]][path[j+1]]['capacity'][i:i+num_slots] == [0]*num_slots 
                   for j in range(len(path)-1)):
                return i, i + num_slots - 1
        return None

    def Run(self, rate):
        for count in tqdm(range(0, NUM_OF_REQUESTS), desc="Processing Requests"):
            yield self.env.timeout(self.random.expovariate(rate))
            src, dst = self.random.sample(self.nodes, 2)
            bandwidth = self.random.choice(BANDWIDTH)
            holding_time = self.random.expovariate(HOLDING_TIME)
            #print("Request", count, "(",src, dst, bandwidth, holding_time,")" )

            paths = self.find_k_paths(src, dst)
            #print("paths",paths)
            for path in paths:
                distance = int(self.Distance(path))
                num_slots = int(math.ceil(self.Modulation(distance, bandwidth)))
                spectrum = self.find_feasible_spectrum(path, num_slots)

                if spectrum:
                    self.allocate_path(path, spectrum, count)
                    fragmentation = self.calculate_fragmentation(path)
                    reward = 1 - fragmentation  # Higher reward for less fragmentation
                    #reward = 1
                    self.update_q_values(path, reward)
                    desalocate = Desalocate(self.env)
                    self.env.process(desalocate.Run(count, path, spectrum, holding_time))
                    #print("Request", count, "established")
                    break
                    
            else:
                self.NumReqBlocked += 1
                #print("Request", count, "blocked")

            #blocking_ratio = (self.NumReqBlocked / (count+1)) * 100
            #print(f"\rCurrent Blocking Ratio: {blocking_ratio:.2f}% ", end="", flush=True)
        #print()

    def allocate_path(self, path, spectrum, count):
        for i in range(len(path) - 1):
            for slot in range(spectrum[0], spectrum[1] + 1):
                topology[path[i]][path[i+1]]['capacity'][slot] = count

    def update_q_values(self, path, reward):
        for i in range(len(path) - 1):
            self.update_q_value(path[i], path[i+1], reward, path[i+1])
