import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordEpisodeStatistics
import matplotlib.pyplot as plt
import numpy as np

# Parámetros del entrenamiento
gamma = 1.0
use_sde = True
train_freq = 32
repetitions = 30
episodes = 1000
steps_per_episode = 3000  # Número máximo de pasos por episodio
visualization_interval = 5  # Cada cuántas repeticiones visualizar

# Función para entrenar y recolectar métricas por episodios
def train_sac_by_episodes(env_name, episodes):
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    # Crear el modelo SAC
    model = SAC(
        "MlpPolicy",
        env,
        gamma=gamma,
        use_sde=use_sde,
        train_freq=train_freq,
        verbose=0  # Reducir detalles en logs
    )

    episode_lengths = []  # Almacenar el largo de cada episodio

    # Entrenamiento controlado por episodios
    for episode in range(episodes):
        print(episode)
        state, info = env.reset()
        done = False
        episode_length = 0

        while not done and episode_length < steps_per_episode:
            action, _ = model.predict(state, deterministic=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1

        episode_lengths.append(episode_length)
        model.learn(total_timesteps=episode_length, reset_num_timesteps=False)

    env.close()
    return episode_lengths, model

# Visualizar el agente entrenado
def visualize_agent(model):
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    frames = []
    state, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
    env.close()
    return frames

# Repetir experimento y recolectar métricas
all_lengths = []
for i in range(repetitions):
    print(f"Repetición {i + 1} de {repetitions}...")
    lengths, trained_model = train_sac_by_episodes("MountainCarContinuous-v0", episodes)
    all_lengths.append(lengths)

    # Visualizar el agente entrenado en el intervalo definido
    if (i + 1) % visualization_interval == 0:
        print(f"Visualizando el agente tras {i + 1} repeticiones...")
        frames = visualize_agent(trained_model)
        plt.imshow(frames[-1])  # Mostrar el último frame
        plt.axis("off")
        plt.title(f"Visualización tras repetición {i + 1}")
        plt.show()

# Calcular promedios cada 10 episodios
average_lengths = np.mean(all_lengths, axis=0).reshape(-1, 10).mean(axis=1)

# Mostrar resultados en texto
print("\nPromedio de largos cada 10 episodios (en 30 repeticiones):")
for i, avg in enumerate(average_lengths):
    print(f"Episodios {i*10+1}-{(i+1)*10}: {avg:.2f}")

# Graficar la evolución de los largos promedio
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
