import simpy
from random import *
from config import *
import numpy as np
import networkx as nx
import math
from itertools import islice
from tqdm import tqdm
import tensorflow as tf
from collections import deque
import random

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
            (0, 500): 4,
            (501, 1000): 3,
            (1001, 2000): 2,
            (2001, float('inf')): 1
        }

    def Modulation(self, dist, demand):
        for (lower, upper), multiplier in self.modulation_table.items():
            if lower <= dist <= upper:
                return math.ceil(float(demand) / float(multiplier * SLOT_SIZE))
        return math.ceil(float(demand) / float(SLOT_SIZE))  # Default case

    
    def Distance(self, path):
    	return nx.path_weight(topology, path, weight='weight')


class DQNRouter(Simulador):
    def __init__(self, env):
        super().__init__(env)
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.target_update_counter = 0

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def find_k_paths(self, source, destination):
        paths = []
        for _ in range(Q_LEARNING_EPISODES):
            path = [source]
            current_node = source
            state = np.zeros((1, self.state_size))
            state[0][current_node] = 1
            done = False
            while not done:
                action = self.act(state)
                next_node = action
                # Initialize reward with a default value
                reward = 0
                if next_node == destination:
                    reward = 1  # Positive reward for reaching the destination
                    done = True
                elif next_node in path:
                    reward = -1  # Negative reward for revisiting a node
                    done = True
                else:
                    # Reward based on the inverse of the link weight
                    reward = 1 / topology[current_node][next_node]['weight']
                next_state = np.zeros((1, self.state_size))
                next_state[0][next_node] = 1
                self.remember(state, action, reward, next_state, done)
                state = next_state
                path.append(next_node)
                current_node = next_node
                if len(self.memory) > BATCH_SIZE:
                    self.replay(BATCH_SIZE)
            if path[-1] == destination:
                paths.append(tuple(path))
            self.target_update_counter += 1
            if self.target_update_counter % TARGET_UPDATE_FREQUENCY == 0:
                self.update_target_model()
        return sorted(set(paths), key=len)[:K_PATHS]

    def calculate_link_load(self, node1, node2):
        link_capacity = topology[node1][node2]['capacity']
        return sum(1 for slot in link_capacity if slot != 0) / len(link_capacity)

    # def calculate_link_osnr(self, node1, node2):
    #     # This is a placeholder. In a real system, you would measure or estimate the OSNR.
    #     # For this example, we'll use a random value between 0 and 1.
    #     return random()

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

class PLIAwareRMSA(DQNRouter):
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
