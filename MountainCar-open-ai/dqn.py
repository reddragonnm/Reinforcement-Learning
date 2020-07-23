import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.discount = 0.85  # discount
        self.epsilon = 1.0  # epilson starting
        self.epsilon_min = 0.01  # minimum epilson
        self.epsilon_decay = 0.995  # epilson decay
        self.learning_rate = 0.005  # learning rate
        self.tau = .125  # plays the role of shifting from the prediction models to the target models gradually

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()

        state_shape = self.env.observation_space.shape
        print(self)

        model.add(Dense(24, input_shape=state_shape, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))

        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay  # decaying the epilson or exploration rate
        # prevent epilson to go below minimum epilson
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # just exploring! could also write random

        # print(self.model.predict(state))

        # only one item is present and returning action with highest q-value
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        # adding the details to the deque(list)
        self.memory.append([state, action, reward, new_state, done])

    def model_train(self):
        batch_size = 32

        if len(self.memory) < batch_size:
            return

        # generating random samples for ideal training
        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)

            if done:
                # it is sure that this action will bring a reward!
                target[0][action] = reward
            else:
                # prediction for state after action
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * \
                    self.discount  # the formula

            self.model.fit(state, target, epochs=1,
                           verbose=0)  # training the model

    def target_train(self):
        # updating the target model with progress of main model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + \
                target_weights[i] * (1 - self.tau)  # another formula

        # after generating new target weights update the
        self.target_model.set_weights(target_weights)
        # target model

    def save_model(self, fn):
        # fn -> file name
        # saving the full model
        self.model.save(fn)


def main():
    env = gym.make("MountainCar-v0")

    trials = 1000  # number of episodes
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)  # getting the agent

    for trial in range(trials):
        print(env.reset())
        cur_state = env.reset().reshape(1, 2)
        print(cur_state)

        for step in range(trial_len):

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward,
                               new_state, done)  # remember for training

            dqn_agent.model_train()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            # setting the current state for next episode as this episode's new_state
            cur_state = new_state

            if done:
                break

        if step >= 199:  # if exceed number of step then model failed

            print("Failed to complete in trial {}".format(trial))

            if step % 10 == 0:  # save every 10 episodes
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:

            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    main()
