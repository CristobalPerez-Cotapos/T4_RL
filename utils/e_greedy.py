import random

def e_greedy(action_values, epsilon):
    if epsilon > random.random():
        return random.choice(list(action_values.keys()))
    else:
        return max(action_values, key=action_values.get)
    

def max_a(action_values):
    return max(action_values.values())