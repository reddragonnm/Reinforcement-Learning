from env import GameEnv

import numpy as np
from collections import deque
import random
from tqdm import tqdm as prog_bar

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, env):
        self.env = env

        self.batch_size = 50
        self.memory_size = 2000
        self.memory = deque(maxlen=self.memory_size)

        self.learning_rate = 0.1
        self.discount = 0.9
        self.tau = 0.125

        self.epilson = 1
        self.eps_decay = 0.9
        self.min_epilson = 0.05

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()

        model.add(Dense(10, input_shape=(2,), activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(25, activation="relu"))
        model.add(Dense(4))

        model.compile(
            loss="mse",
            optimizer=Adam(lr=self.learning_rate)
        )

        return model

    def model_train(self):
        batch_size = 32

        if len(self.memory) < batch_size:
            return

        # generating random samples for ideal training
        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            state, action, reward, new_state, done = sample
            # bas target hi rakh bahut sataya he isne mujhe
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
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for count in range(len(target_weights)):
            target_weights[count] = weights[count] * \
                self.tau + target_weights[count] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def predict_action(self, state):
        self.epilson = max([self.epilson * self.eps_decay, self.min_epilson])

        if np.random.random() < self.epilson:
            return np.random.randint(0, 4)

        # state = np.array([state])
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def save_model(self, file_name):
        self.model.save(file_name)


show_every = 20


def main():
    env = GameEnv(show=False)
    agent = DQNAgent(env=env)
    no_episodes = 500

    for episode in prog_bar(range(no_episodes), ascii=False, unit="episodes"):
        state = env.reset().reshape(1, 2)

        while True:
            action = agent.predict_action(state)
            new_state, reward, done = env.action(action)

            new_state = new_state.reshape(1, 2)

            agent.remember(state, action, reward, new_state, done)
            agent.model_train()
            agent.target_train()

            state = new_state.reshape(1, 2)

            if done:
                if reward == 500:
                    print(f"Completed in episode {episode}")
                    agent.save_model(f"final-{episode}.model")

                break


if __name__ == '__main__':
    main()
