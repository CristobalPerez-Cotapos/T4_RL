import numpy as np
import gym
import json
from FeatureExtractor import FeatureExtractor
from utils.e_greedy import e_greedy

# Parámetros
alpha = 0.5 / 8  # Tasa de aprendizaje
gamma = 1.0  # Factor de descuento
epsilon = 0.1  # Probabilidad de exploración
episodes = 1000  # Número de episodios
epochs = 30

# Inicialización
env = gym.make("MountainCar-v0")
feature_extractor = FeatureExtractor()
num_features = feature_extractor.num_of_features

# Función para calcular Q(s, a)
def q_value(state, action):
    features = feature_extractor.get_features(state, action)
    return np.dot(weights[action], features)

mean_episode_length = {value: 0 for value in range(0, episodes, 10)}

# Entrenamiento con Sarsa
for epoch in range(epochs):
    print(f"Epoch {epoch}:")
    weights = {a: np.zeros(num_features) for a in range(3)}  # Tres acciones: 0, 1, 2
    for episode in range(episodes):
        state, _ = env.reset()
        action = e_greedy({a: q_value(state, a) for a in range(3)}, epsilon)
        for t in range(1000):  # Límite de pasos por episodio
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = e_greedy({a: q_value(next_state, a) for a in range(3)}, epsilon)
            
            # Actualización
            td_target = reward + gamma * q_value(next_state, next_action)
            td_error = td_target - q_value(state, action)
            features = feature_extractor.get_features(state, action)
            weights[action] += alpha * td_error * features
            
            state, action = next_state, next_action
            if terminated or truncated:
                if episode % 10 == 0:
                    print(f"Episodio {episode} terminado. pasos: {t}")
                    mean_episode_length[episode] += t/epochs
                break


with open("experiments/sarsa_mean_episode_length.json", "w") as archivo:
    json.dump(mean_episode_length, archivo, indent=4)

# Prueba del agente entrenado
# env = gym.make("MountainCar-v0", render_mode="human")
# state, _ = env.reset()
# for t in range(1000):
#     action = max(range(3), key=lambda a: q_value(state, a))
#     state, reward, terminated, truncated, _ = env.step(action)
#     if terminated:
#         print("¡Meta alcanzada!")
#         break
# env.close()
