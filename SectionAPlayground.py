import json
import matplotlib.pyplot as plt

file_paths = {
    "Q Learning": "qlearning_mean_episode_length.json",
    "Sarsa": "sarsa_mean_episode_length.json",
    "Sarsa(Lambda)": "n_step_sarsa_mean_episode_length.json"
}

# Diccionario para almacenar los datos
data = {}

# Cargar y parsear los datos JSON
for label, path in file_paths.items():
    with open("experiments/" + path, 'r') as f:
        data[label] = json.load(f)

# Configurar el gráfico
plt.figure(figsize=(16, 8))

# Colores específicos para cada curva
colors = {
    "Q Learning": "red",
    "Sarsa": "blue",
    "Sarsa(Lambda)": "yellow"
}

# Graficar cada conjunto de datos
for label, values in data.items():
    episodes = list(map(int, values.keys()))
    steps = list(values.values())
    plt.plot(episodes, steps, label=label, color=colors[label])

# Añadir etiquetas y título
plt.xlabel('Episode')
plt.ylabel('Mean Steps')
plt.title('Comparison of Mean steps per Episode for MountainCar')

# Establecer el límite superior del eje y a 400
# plt.ylim(top=400)

plt.legend()
plt.grid(True)
plt.show()