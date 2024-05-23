import gym
from stable_baselines3 import DQN
from environment import PokerEnv

# Vytvoření prostředí
env = PokerEnv()

# Inicializace modelu
model = DQN('CnnPolicy', env, verbose=1)

# Trénování modelu
model.learn(total_timesteps=10000)

# Uložení modelu
model.save("poker_agent")
