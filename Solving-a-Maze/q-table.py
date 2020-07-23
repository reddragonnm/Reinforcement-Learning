from env import GameEnv
import numpy as np

env = GameEnv(type="q_table")
done = False

no_episodes = 500000  # number of rounds to train the AI
learning_rate = 0.2  # between 0 and 1, higher value = higher learning rate (in this case freezes if set very high)
discount = 0.95  # more inclined to prefer future rewards than immediate rewards
epsilon = 0.9  # how much we want to explore the environment
epsilon_decay = 0.9998  # decrease exploration each step (multiplied by epilson)
show_every = 100  # at which frequency do we want to see the progress visually

state = env.reset()

q_table = {}
for a in range(10):
    for b in range(10):
        q_table[(a, b)] = [np.random.uniform(-5, 0) for i in range(4)]  # number of actions

for episode in range(no_episodes):
    while True:
        if np.random.random() > epsilon:
            # will be false in the beginning but will diminish towards the end
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 4)

        new_state, reward, done = env.action(action)

        # IMPORTANT: new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        # formula above is the new value for the state the AI is currently in (or it is learning due to this formula)
        current_q = q_table[state][action]  # VALUE for the state before the action
        max_future_q = np.max(q_table[new_state])  # action VALUE for the new state

        # if we get reward after action then definitely do that
        if reward == 500:
            new_q = 500
        else:
            # magic going on here!
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

        # new value is calculated and is replaced in the previous q table for the current state only
        q_table[state][action] = new_q  # it takes state before action as we get reward based on that
        state = new_state  # the new state is the "state before action" for the new step

        if episode % show_every == 0:
            print(episode)
            env.render()

        if done:
            env.reset()
            break

        epsilon *= epsilon_decay  # decaying or decreasing the exploration

print(q_table)  # again for testing purposes
