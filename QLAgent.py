import random
from utils.e_greedy import e_greedy, max_a
import gym

from FeatureExtractor import FeatureExtractor

def addapt_obs(obs, decimals_1 = 2, decimals_2 = 2):
    uno = round(obs[0], decimals_1)
    dos = round(obs[1], decimals_2)
    return tuple([uno, dos])

env = gym.make("MountainCar-v0")
feature_extractor = FeatureExtractor()
observation, info = env.reset()
observation = addapt_obs(observation)
obs_features = feature_extractor.get_features(observation)
alpha = 0.5/8
gamma = 1

action_space = [0, 1, 2]

Q = dict()
Q[observation] = {a:1 for a in action_space}


for _ in range(100000):
    action = e_greedy(Q, observation, 0)
    old_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    observation = addapt_obs(observation)
    next_obs_features = feature_extractor.get_features(observation)
    
    if observation not in Q.keys():
        Q[observation] = {a : 1 for a in action_space}
    
    Q[old_observation][action] = alpha * (reward + gamma * max_a(Q, observation) - Q[old_observation][action])


    if terminated or truncated:
        observation, info = env.reset()
        obs_features = feature_extractor.get_features(observation)
        observation = addapt_obs(observation)
        if observation not in Q.keys():
            Q[observation] = {a : 1 for a in action_space}
        if terminated:
            print("EPICOOOO")

env.close()




env = gym.make("MountainCar-v0", render_mode="human")
feature_extractor = FeatureExtractor()
observation, info = env.reset()
observation = addapt_obs(observation)
obs_features = feature_extractor.get_features(observation)
if observation not in Q.keys():
    Q[observation] = {a : 1 for a in action_space}

for _ in range(10000):
    action = e_greedy(Q, observation, 0.1)
    old_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    observation = addapt_obs(observation)
    next_obs_features = feature_extractor.get_features(observation)
    
    if observation not in Q.keys():
        Q[observation] = {a : 1 for a in action_space}
    
    Q[old_observation][action] = alpha * (reward + gamma * max_a(Q, observation) - Q[old_observation][action])


    if terminated or truncated:
        observation, info = env.reset()
        obs_features = feature_extractor.get_features(observation)
        observation = addapt_obs(observation)
        if observation not in Q.keys():
            Q[observation] = {a : 1 for a in action_space}
        if terminated:
            print("EPICOOOO")

env.close()


