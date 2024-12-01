import numpy as np
import gym
from FeatureExtractor import FeatureExtractor
from utils.e_greedy import e_greedy

# Parámetros
alpha = 0.5 / 8  # Tasa de aprendizaje
gamma = 1.0  # Factor de descuento
epsilon = 0.1  # Probabilidad de exploración
episodes = 500  # Número de episodios

# Inicialización
env = gym.make("MountainCar-v0")
feature_extractor = FeatureExtractor()
num_features = feature_extractor.num_of_features
weights = {a: np.zeros(num_features) for a in range(3)}  # Tres acciones: 0, 1, 2

# Función para calcular Q(s, a)
def q_value(state, action):
    features = feature_extractor.get_features(state, action)
    return np.dot(weights[action], features)

# Entrenamiento con Sarsa
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
            break
    print(f"Episodio {episode + 1} terminado.")

# Prueba del agente entrenado
env = gym.make("MountainCar-v0", render_mode="human")
state, _ = env.reset()
for t in range(1000):
    action = max(range(3), key=lambda a: q_value(state, a))
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated:
        print("¡Meta alcanzada!")
        break
env.close()
