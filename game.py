import numpy as np
import matplotlib.pyplot as plt
import opponent as opp
from players import player
import argparse
# from opponent_policies import policy_1, policy_2, policy_3

parser = argparse.ArgumentParser()
parser.add_argument("--p", help="p")
parser.add_argument("--q", help="q")
parser.add_argument("--opponent_policy", help="0: random, 1: park the bus, 2: greedy", \
                    choices=['0', '1', '2'])
args = parser.parse_args()

def area(x1,y1,x2,y2,x3,y3):
    a = np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
    return 0.5 * abs(np.linalg.det(a))
def manhattan_distance(x1, y1, x2, y2):
    return np.abs(x1-x2) + np.abs(y1-y2)


    
def shooting(state):
    global game_end
    global ball_possession
    reward = -0.5
    probability = q - 0.2 * (3 - state[2 * state[6]])
    if state[4] == 3 and state[5] % 3 != 0:
        probability /= 2
    if np.random.rand() < probability:
        reward = 10
    game_end = True
    return reward
    
def passing(state):
    global game_end
    global ball_possession
    reward = 0
    probability = q - 0.1 * max(abs(state[0] - state[2]), abs(state[1] - state[3]))
    if area(state[0], state[1], state[2], state[3], state[4], state[5]) == 0:
        probability /= 2
    if np.random.rand() < probability:
        ball_possession = (ball_possession + 1) % 2
    else:
        game_end = True
        reward = -0.5
    return reward

def moving(state, movement):
    global game_end
    global ball_possession
    reward = 0
    probability = 1 - p
    if movement >= 4 and movement <= 7:
        b = b2
        if ball_possession == 1:
            probability = 1 - 2*p
    else:
        b = b1
        if ball_possession == 0:
            probability = 1 - 2*p

    b.move(movement % 4)

    # out of bounds
    if min(b.x, b.y) < 0 or max(b.x, b.y) > 3:
        game_end = True
        return -2

    
    # tackle
    if ball_possession == 0:
        b = b1
    else:
        b = b2

    if b.x == r.x and b.y == r.y:
        # ending up on the same square
        probability /= 2

    elif b.x == state[4] and b.y == state[5] and r.x == state[2*state[6]] and r.y == state[2*state[6]+1]:
        # crossing each other
        probability /= 2

    if np.random.rand() < probability:
        pass
    else:
        game_end = True
        reward = 0
        
    return reward

# define the action space
action_space = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# defining p and q
p = float(args.p)
q = float(args.q)

# defining the opponent policy
if int(args.opponent_policy) == 0:
    opp_movement = opp.random_policy    # storing the function in an object, not calling it
elif int(args.opponent_policy) == 1:
    opp_movement = opp.park_the_bus
elif int(args.opponent_policy) == 2:
    opp_movement = opp.greedy_policy

# epsilon greedy
epsilon = 0.4
epsilon_decay = 0.00002

# defining the lookup table for the q-values
state_action_value_function = {}
for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                for e in range(4):
                    for f in range(4):
                        for g in range(2):
                            for h in range(10):
                                state_action_value_function[10000000*a + 1000000*b \
                                                     + 100000*c + 10000*d + 1000*e + 100*f + 10*g + h] = 0

def access_state_action_value_function(state_action):
    # print(state_action)
    return 10000000*state_action[0] + 1000000*state_action[1] + 100000*state_action[2] \
                                 + 10000*state_action[3] + 1000*state_action[4] \
                                      + 100*state_action[5] + 10*state_action[6] + state_action[7]

# let's do each episode and update the rewards backwards

def my_policy(state, state_action_value_function):
    if np.random.rand() < epsilon:
        return np.random.randint(0,10)
    else:
        all_possible_actions = np.array([np.append(state, [i], axis=None) for i in range(10)])
        values = []
        for element in all_possible_actions:
            values.append(access_state_action_value_function(element))
        return np.argmax(values)     # returns the first max index if multiple max values present, but an issue for later


# need to convert the subsequent lines into reusable episodic code
# the only thing retained after each episode is the updated value function we'll use
# need to do something about terminal state

# initialize starting positions
"""
b1 = player(0,1)
b2 = player(0,2)
r = player(3,1)
"""

# add -ve rewards to losing the ball and shit
# keep an array for rewards and another for goals

# discounting factor
gamma = 0.99999
# learning rate
alpha = 0.2
# number of episodes
num_of_episodes = 1000_000

rewards_array = [0]
goals = 0
goals_array = [0]
goals_after_x_episodes = [0]

for episode in range(num_of_episodes):
    # initialize the episode
    b1 = player(np.random.randint(0, 4), np.random.randint(0, 4))
    b2 = player(np.random.randint(0, 4), np.random.randint(0, 4))
    r = player(np.random.randint(0, 4), np.random.randint(0, 4))
    ball_possession = np.random.randint(0, 2)   # can be 0 or 1

    # define state at any time step
    state = np.array([b1.x, b1.y, b2.x, b2.y, r.x, r.y, ball_possession])
    game_end = False
    reward = 0
    own_policy = my_policy(state, state_action_value_function)

    while True:
        # opp policy
        # own policy with timestep
        # should we define a max time step?

        # actually put all this under policy or something and just call the policy
        r.move(opp_movement(state))   # r has to move first
        # rewards

        """
        # random policy for now
        own_policy = np.random.choice(action_space)
        """
        if own_policy == 9:
            reward = shooting(state)
        elif own_policy == 8:
            reward = passing(state)
        else:
            reward = moving(state, own_policy)
        
        # update epsilon
        if episode > 999800:
            print(episode, " ", own_policy)
            if reward == 10:
                goals_after_x_episodes.append(goals_after_x_episodes[-1]+1)
            else:
                goals_after_x_episodes.append(goals_after_x_episodes[-1]+0)
        epsilon -= epsilon_decay
        current_state = state.copy()
        current_action = own_policy
        current_q = state_action_value_function[access_state_action_value_function(np.append(state, [own_policy], axis=None))]
        # print(current_state, current_action)
        if game_end:
            state_action_value_function[access_state_action_value_function(
                np.append(current_state, [current_action], axis=None))] = current_q + alpha*(reward - current_q)
            break

        # update state
        state = np.array([b1.x, b1.y, b2.x, b2.y, r.x, r.y, ball_possession])
        # update action
        own_policy = my_policy(state, state_action_value_function)
        # update q-value function, i.e. the bellman equation
        future_q = state_action_value_function[access_state_action_value_function(np.append(state, [own_policy], axis=None))]
        current_q = current_q + alpha*(reward + gamma*future_q - current_q)
        state_action_value_function[access_state_action_value_function(np.append(current_state, [current_action], axis=None))] = current_q
    
    # after episode ends
    rewards_array.append(rewards_array[-1] + reward)
    if reward == 10:
        goals += 1
        goals_array.append(goals_array[-1] + 1)
    else:
        goals_array.append(goals_array[-1])
    
    # print(episode)
    # print(reward)


plt.plot(goals_after_x_episodes)
plt.show()
print(goals)
print(goals/episode)
