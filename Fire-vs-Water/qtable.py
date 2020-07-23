import numpy as np  # numpy for maintaining the q-table
import pygame as pg
from environment import GameEnv

# can be changed
height = 5
width = 5

env = GameEnv(height=height, width=width)  # calling the environment
done = False  # will be controlled by the environment

# making the q-table
q_table = {}
for x1 in range(-height + 1, height):
    for y1 in range(-width + 1, width):
        for x2 in range(-height + 1, height):
            for y2 in range(-width + 1, width):
                q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0)
                                                 for i in range(4)]

# some constants - can be tinkered with
no_episodes = 500000  # number of rounds to train the AI
# between 0 and 1, higher value = higher learning rate (in this case freezes if set very high)
learning_rate = 0.2
discount = 0.95  # more inclined to prefer future rewards than immediate rewards
epsilon = 0.9  # how much we want to explore the environment
# decrease exploration each step (multiplied by epilson)
epsilon_decay = 0.9998
show_every = 500  # at which frequency do we want to see the progress visually

state = env.get_rel_pos()  # get the starting position

for episode in range(no_episodes):
    while True:
        if np.random.random() > epsilon:
            # will be false in the beginning but will diminish towards the end
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 4)

        # getting the values to make q table
        new_state, reward, done = env.action(action)

        # IMPORTANT: new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        # formula above is the new value for the state the AI is currently in (or it is learning due to this formula)
        # action VALUE for the new state
        max_future_q = np.max(q_table[new_state])
        # VALUE for the state before the action
        current_q = q_table[state][action]

        # if we get reward after action then definitely do that
        if reward == 50:
            new_q = 50
        else:
            # magic going on here!
            new_q = (1 - learning_rate) * current_q + \
                learning_rate * (reward + discount * max_future_q)

        # new value is calculated and is replaced in the previous q table for the current state only
        # it takes state before action as we get reward based on that
        q_table[state][action] = new_q
        state = new_state  # the new state is the "state before action" for the new step

        if episode % show_every == 0:
            print(f"Episode - {episode}")  # just for testing purposes
            env.render()  # show the state progress of AI

        if done:
            env.reset()  # reset the environment
            break  # go to the next game

    epsilon *= epsilon_decay  # decaying or decreasing the exploration

print(q_table)  # again for testing purposes
pg.quit()  # end pygame
