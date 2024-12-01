import numpy as np
from utils.e_greedy import e_greedy, max_a
import gym
from FeatureExtractor import FeatureExtractor

env = gym.make("MountainCar-v0")
feature_extractor = FeatureExtractor()
alpha = 0.5 / 8  # Learning rate
gamma = 1.0  # Discount factor
epsilon = 0.1  # Epsilon para e-greedy

# Inicialización de pesos
num_features = feature_extractor.num_of_features
weights = {a: np.zeros(num_features) for a in range(3)}  # Tres acciones: 0, 1, 2

# Función para calcular Q(s, a)
def q_value(state, action):
    features = feature_extractor.get_features(state, action)
    return np.dot(weights[action], features)

# Entrenamiento
for episode in range(1000):
    state, info = env.reset()
    for t in range(1000):  # Limitar duración del episodio
        action_values = {a: q_value(state, a) for a in range(3)}  # Calcula los valores Q(s, a) para todas las acciones
        action = e_greedy(action_values, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Actualización de pesos
        features = feature_extractor.get_features(state, action)
        if not terminated:
            next_action = max(range(3), key=lambda a: q_value(next_state, a))
            td_target = reward + gamma * q_value(next_state, next_action)
        else:
            td_target = reward

        td_error = td_target - q_value(state, action)
        weights[action] += alpha * td_error * features

        state = next_state
        if terminated or truncated:
            break

# Visualización del agente entrenado
env = gym.make("MountainCar-v0", render_mode="human")
state, info = env.reset()
for t in range(1000):
    action = max(range(3), key=lambda a: q_value(state, a))
    state, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("¡Meta alcanzada!")
        break
env.close()
