import random

def e_greedy(Q, features, epsilon):
    if epsilon > random.random():
        # print("Elegi random")
        return random.choice([0, 1, 2])
    else:
        # print("Elegi greedy")
        max_a = max(Q[features])
        q_value_for_max_a = Q[features][max(Q[features])]
        for a in Q[features].keys():
            # print(f"Accion {a} con Q = {Q[features][a]}")
            # print(f"{Q[features][a]} > {q_value_for_max_a}")
            if Q[features][a] > q_value_for_max_a:
                # print("entre al if")
                q_value_for_max_a = Q[features][a]
                max_a = a
            # print(f"Accion {a} con Q = {Q[next_s][a]}")
        return max_a
    

def max_a(Q, features):
    max_a = max(Q[features])
    q_value_for_max_a = Q[features][max(Q[features])]
    for a in Q[features].keys():
        # print(f"Accion {a} con Q = {Q[next_s][a]}")
        # print(f"{Q[next_s][a]} > {q_value_for_max_a}")
        if Q[features][a] > q_value_for_max_a:
            # print("entre al if")
            q_value_for_max_a = Q[features][a]
            max_a = a
        # print(f"Accion {a} con Q = {Q[next_s][a]}")
    # print(f"La mejor accion de next state ({next_s}) es {max(Q[next_s])} = {Q[next_s][max(Q[next_s])]}")
    return q_value_for_max_a