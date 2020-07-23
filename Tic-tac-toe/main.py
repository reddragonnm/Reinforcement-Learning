# import all modules
import numpy as np
import pygame as pg
import time
import random

pg.init()
font = pg.font.Font('freesansbold.ttf', 32)


class GameEnv:
    def __init__(self):
        # initialising board and character(X/O) the player and opponent uses
        self.board = ["-"] * 9
        self.char = "X"
        self.opp_char = "O"

        # initiating the pygame screen
        screen_dim = (150, 150)
        self.screen = pg.display.set_mode(screen_dim)
        pg.display.set_caption('TicTacToe using Reinforcement Learning')

    def play_opponent(self):
        # opponent plays random space from available spaces
        available = []

        for n, val in enumerate(self.board):
            if val == "-":
                available.append(n)

        try:
            self.board[random.choice(available)] = self.opp_char
        except:
            pass

    def are_same(self, lst):
        # check if all elements of list are same
        if len(list(set(lst))) == 1:
            return True
        else:
            return False

    def validate_board(self):
        # checking if player won, lost or game was drawn and giving done value and reward based on it
        win_reward = 50
        lose_penalty = -50
        draw_reward = 10
        move_penalty = -1

        row1 = self.board[:3]
        row2 = self.board[3:6]
        row3 = self.board[6:9]

        col1 = self.board[0::3]
        col2 = self.board[1::3]
        col3 = self.board[2::3]

        dia1 = self.board[0::4]
        dia2 = self.board[2:6:2]

        for thing in [row1, row2, row3, col1, col2, col3, dia1, dia2]:
            if self.are_same(thing) and thing[0] != "-":
                done = True
                if thing[0] == self.char:
                    reward = win_reward
                else:
                    reward = lose_penalty

                return reward, done

        if "-" not in self.board:
            done = True
            reward = draw_reward
        else:
            done = False
            reward = move_penalty

        return reward, done

    def action(self, action):
        assert 0 <= action <= 8

        ''' All valid actions
            0	1	2
            3	4	5
            6	7	8
        '''

        # checking action is played in valid place
        if self.board[action] == "-":
            self.board[action] = self.char

            reward, done = self.validate_board()

            if not done:
                self.play_opponent()
                reward, done = self.validate_board()

            return np.array([tuple(self.board), reward, done])

        # discourage the AI from performing invalid actions
        return np.array([tuple(self.board), -5, False])

    def reset(self):
        # reset the board
        self.board = ["-"] * 9

    def render(self):
        # displaying the board
        self.screen.fill((255, 255, 255))

        self.screen.blit(font.render(self.board[0], True, (0, 0, 0)), (10, 10))
        self.screen.blit(font.render(self.board[1], True, (0, 0, 0)), (10, 60))
        self.screen.blit(font.render(
            self.board[2], True, (0, 0, 0)), (10, 110))
        self.screen.blit(font.render(self.board[3], True, (0, 0, 0)), (60, 10))
        self.screen.blit(font.render(self.board[4], True, (0, 0, 0)), (60, 60))
        self.screen.blit(font.render(
            self.board[5], True, (0, 0, 0)), (60, 110))
        self.screen.blit(font.render(
            self.board[6], True, (0, 0, 0)), (110, 10))
        self.screen.blit(font.render(
            self.board[7], True, (0, 0, 0)), (110, 60))
        self.screen.blit(font.render(
            self.board[8], True, (0, 0, 0)), (110, 110))

        pg.display.update()
        time.sleep(0.1)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

    def play_with_user(self, q_table, rounds):
        for _ in range(rounds):
            play = False
            while not play:
                self.render()

                while True:
                    ai_choice = np.argmax(q_table[tuple(self.board)])
                    if self.board[ai_choice] == "-":
                        self.board[ai_choice] = self.char
                        self.render()
                        break

                choice = int(input("Enter a number: "))

                while True:
                    if self.board[choice] == "-":
                        self.board[choice] = self.opp_char
                        break

                _, play = self.validate_board()


env = GameEnv()
done = False

# some constants - can be tinkered with
no_episodes = 50000  # number of rounds to train the AI
# between 0 and 1, higher value = higher learning rate (in this case freezes if set very high)
learning_rate = 0.2
discount = 0.95  # more inclined to prefer future rewards than immediate rewards
epsilon = 0.9  # how much we want to explore the environment
# decrease exploration each step (multiplied by epilson)
epsilon_decay = 0.9998
show_every = 500  # at which frequency do we want to see the progress visually

lost = 0

state = tuple(["-"] * 9)

# making the q-table
lst = ["-", "X", "O"]
q_table = {}

# I know this can be simplified
for a in lst:
    for b in lst:
        for c in lst:
            for d in lst:
                for e in lst:
                    for f in lst:
                        for g in lst:
                            for h in lst:
                                for i in lst:
                                    q_table[(a, b, c, d, e, f, g, h, i)] = [
                                        np.random.uniform(-5, 0) for j in range(9)]


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

        if reward == -50:
            print(f"Lost in episode {episode}")
            lost += 1

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


print(f"Lost rate in {lost} episodes",
      f"Losing rate: {(lost/no_episodes)*100}")

env.play_with_user(q_table=q_table, rounds=10)

pg.quit()  # end pygame
