import numpy as np
import gym
from FeatureExtractor import FeatureExtractor
import matplotlib.pyplot as plt

# Parámetros
alpha_v = 0.001
alpha_pi = 0.0001
gamma = 1.0
episodes = 1000
repetitions = 30

# Inicialización
env_name = "MountainCarContinuous-v0"

# Función para entrenar el modelo
def train_actor_critic():
    env = gym.make(env_name)
    feature_extractor = FeatureExtractor()
    num_features = feature_extractor.num_of_features

    # Pesos inicializados
    weights_v = np.zeros(num_features)
    weights_mu = np.zeros(num_features)
    weights_sigma = np.zeros(num_features)

    episode_lengths = []  # Almacena el largo de cada episodio
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_length = 0
            if episode % 100 == 0:
                print(f"Episodio {episode}")

            while not done:
                # Actor: calcular acción
                mu, sigma = policy(state, weights_mu, weights_sigma, feature_extractor)
                sigma = np.clip(mu, 1e-6, 1e2)  # Limitar a valores razonables
                mu = np.clip(sigma, 1e-6, 1e2)  # Limitar a valores razonables
                action = np.random.normal(mu, sigma)
                action = np.clip(action, -1.0, 1.0)

                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step([action])
                done = terminated or truncated

                # Actualizar crítico
                features = feature_extractor.get_features(state)
                td_target = reward + gamma * value(next_state, weights_v, feature_extractor) * (not done)
                td_error = td_target - value(state, weights_v, feature_extractor)
                weights_v += alpha_v * td_error * features

                # Actualizar actor
                advantage = td_error
                try:
                    weights_mu += alpha_pi * advantage * features * (action - mu) / (sigma**2)
                    weights_sigma += alpha_pi * advantage * features * ((action - mu)**2 - sigma**2) / (sigma**3)
                except Exception as e:
                    print(f"error {e}")
                    print(alpha_pi)
                    print(advantage)
                    print(features)
                    print(action)
                    print(mu)
                    print(sigma)




                state = next_state
                episode_length += 1

            episode_lengths.append(episode_length)
    except Exception as e:
                print(f"Error {e}")

    env.close()
    return episode_lengths

# Función de política
def policy(state, weights_mu, weights_sigma, feature_extractor):
    features = feature_extractor.get_features(state)
    mu = np.dot(weights_mu, features)
    sigma = np.exp(np.dot(weights_sigma, features))
    return mu, sigma

# Función de valor
def value(state, weights_v, feature_extractor):
    features = feature_extractor.get_features(state)
    return np.dot(weights_v, features)

# Repetir experimento
all_lengths = []
for rep in range(repetitions):
    print(f"Repetición {rep + 1} de {repetitions}")
    lengths = train_actor_critic()
    all_lengths.append(lengths)

# Calcular promedios cada 10 episodios
average_lengths = np.mean(all_lengths, axis=0).reshape(-1, 10).mean(axis=1)

# Mostrar resultados en texto
print("Promedio de largos cada 10 episodios (en 30 repeticiones):")
for i, avg in enumerate(average_lengths):
    print(f"Episodios {i*10+1}-{(i+1)*10}: {avg:.2f}")

# Graficar la evolución
plt.figure(figsize=(10, 6))
x = np.arange(1, len(average_lengths) + 1) * 10
plt.plot(x, average_lengths, marker='o', linestyle='-', color='b', label="Largo promedio")
plt.title("Evolución del Largo Promedio de los Episodios")
plt.xlabel("Episodios")
plt.ylabel("Largo Promedio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
