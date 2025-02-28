from players import player
import numpy as np

class opponent_policy(player):
    def __init__(self, x, y):
        super.__init__(x, y)
    
    def converge(self, x, y, state) -> int:
        position = x, y     # position to converge to 
        if position[0] == state[4] and position[1] == state[5]:
            movement = np.random.randint(0, 4)
        elif position[0] == state[4]:
            movement = np.sign(position[1] - state[5]) + 1
        elif position[1] == state[5]:
            movement = np.sign(position[0] - state[4]) + 2
        else:
            movement = np.random.choice([np.sign(position[1] - state[5]) + 1, np.sign(position[0] - state[4]) + 2])
        return movement
    
    def random_policy(self, state : np.ndarray):
        movement = np.random.randint(0, 4)
    
    def greedy_policy(self, state : np.ndarray):
        position = state[2*state[6]], state[2*state[6]+1]
        movement = self.converge(position[0], position[1], state)

    def park_the_bus(self, state : np.ndarray):
        if state[4] == 3 and state[5] == 1:
            movement = 2
        elif state[4] == 3 and state[5] == 2:
            movement = 0
        else:
            if state[5] >= 2:
                movement = self.converge(3, 2, state)
            else:
                movement = self.converge(3, 1, state)