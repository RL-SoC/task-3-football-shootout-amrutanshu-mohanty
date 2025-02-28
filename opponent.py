import numpy as np

def converge(x, y, state) -> int:
    position = x, y     # position to converge to 
    if position[0] == state[4] and position[1] == state[5]:
        movement = random_policy(state)
    elif position[0] == state[4]:
        movement = np.sign(position[1] - state[5]) + 1
    elif position[1] == state[5]:
        movement = np.sign(state[4] - position[0]) + 2
    else:
        movement = np.random.choice([np.sign(position[1] - state[5]) + 1, np.sign(state[4] - position[0]) + 2])
    return movement
    
def random_policy(state : np.ndarray):
    x, y = state[4], state[5]
    movements = [0, 1, 2, 3]
    if x == 0:
        movements.remove(3)
    elif x == 3:
        movements.remove(1)
    if y == 0:
        movements.remove(0)
    elif y == 3:
        movements.remove(2)
    return np.random.choice(movements)
    
def greedy_policy(state : np.ndarray):
    position = state[2*state[6]], state[2*state[6]+1]
    return converge(position[0], position[1], state)

def park_the_bus(state : np.ndarray):
    if state[4] == 3 and state[5] == 1:
        return 2
    elif state[4] == 3 and state[5] == 2:
        return 0
    else:
        if state[5] >= 2:
            return converge(3, 2, state)
        else:
            return converge(3, 1, state)