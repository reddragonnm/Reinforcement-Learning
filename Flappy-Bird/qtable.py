from environment import GameEnv
import random
import numpy as np

env = GameEnv()
done = False

q_table = {}
for y in range(-500, 500):
    q_table[y] = [np.random.uniform(-5, 0) for i in range(2)]

no_episodes = 500000  # number of rounds to train the AI
learning_rate = 0.7  # between 0 and 1, higher value = higher learning rate (in this case freezes if set very high)
discount = 0.1  # more inclined to prefer future rewards than immediate rewards
epsilon = 0.9  # how much we want to explore the environment
epsilon_decay = 0.9998  # decrease exploration each step (multiplied by epilson)
show_every = 1

state = env.get_start_state()

for episode in range(no_episodes):
    total_reward = 0
    while True:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 2)

        new_state, reward, done = env.step(action)
        total_reward += reward

        max_future_q = np.max(q_table[new_state])  # action VALUE for the new state
        current_q = q_table[state][action]

        if reward > 5:
            new_q = 100
        else:
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

        q_table[state][action] = new_q
        state = new_state

        if episode % show_every == 0:
            env.render()

        if done:
            if env.score > 0:
                print(f"Episode: {episode}, Score: {env.score}")
            env.reset()
            break

    epsilon *= epsilon_decay
