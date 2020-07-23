import gym
import numpy as np
import matplotlib.pyplot as plt

'''
actions:
0 - left
1 - nothing
2 - right

observation_space and state is in the form of [position velocity]

q-table is a table to which the computer (agent) can refer to for any combination of position and velocity values and
get the most appropriate (highest value in q-table) action......In this case the q-table is 20x20

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
print(q_tab)

print(discrete_state)
print(q_table[discrete_state])

'''

env = gym.make("MountainCar-v0")
env.reset()

learning_rate = 0.1  # anything from 0 to 1
discount = 0.95
episodes = 25000  # number of rounds
show_every = 300  # show training after <- rounds and train in the meantime

epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

discrete_os_size = [20] * len(env.observation_space.high)
discrete_win_size = (env.observation_space.high -
                     env.observation_space.low) / discrete_os_size

q_table = np.random.uniform(
    low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {
    "ep": [],
    "avg": [],
    "min": [],
    "max": []
}


def get_discrete_state(state):
    d_state = (state - env.observation_space.low) / discrete_win_size
    return tuple(d_state.astype(np.int))


for epsds in range(episodes):
    episode_reward = 0

    if epsds % show_every == 0:
        print(epsds)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            # IMPORTANT: new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            new_q = (1 - learning_rate) * current_q + \
                learning_rate * (reward + discount * max_future_q)

            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We made it in episode {epsds}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if end_epsilon_decaying >= epsds >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not epsds % show_every:  # if episode number is divisible by show every
        np.save(f"qtables/{epsds}-qtable.npy", q_table)

        a = ep_rewards[-show_every:]
        average_reward = sum(a) / len(a)

        aggr_ep_rewards["ep"].append(epsds)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(a))
        aggr_ep_rewards["max"].append(max(a))

        print(
            f"episode: {epsds} avg: {average_reward} min: {min(a)} max: {max(a)}")

env.close()
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")
plt.legend(loc=4)
plt.show()
