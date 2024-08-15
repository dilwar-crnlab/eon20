RANDOM_SEED = [50,60,70,80,90,100,110,120,130,140]
MAX_TIME = 10000000  
ERLANG_MIN = 100
ERLANG_MAX = 180
ERLANG_INC = 20
REP = 3
NUM_OF_REQUESTS = 100000
BANDWIDTH = [100,200,400]
TOPOLOGY = 'nsfnet'
HOLDING_TIME = 1.0
SLOTS = 350
SLOT_SIZE = 12.5
# N_PATH = 1
# Add these lines to config.py
Q_LEARNING_EPISODES = 2000
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
K_PATHS = 5  # Number of paths to consider
# Q_LEARNING_EPISODES = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9


# DQN parameters
STATE_SIZE = 14  # Number of nodes in the network
ACTION_SIZE = 14  # Number of possible next nodes
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQUENCY = 100
