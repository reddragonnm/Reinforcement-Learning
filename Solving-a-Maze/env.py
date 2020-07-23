import pygame as pg
import time
import numpy as np

pg.init()


class GameEnv:
    def __init__(self, show=True, type="deep_q"):
        self.show = show
        self.type = type

        if show:
            self.block_size = 50

            self.screen_dim = (self.block_size * 10, self.block_size * 10)
            self.screen = pg.display.set_mode(self.screen_dim)
            pg.display.set_caption('MazeRunner using Reinforcement Learning')

        self.obs_pos = [
            (5, 0), (9, 0), (0, 1), (1, 1), (2, 1), (3, 1), (5, 1), (6, 0), (1, 2), (3, 2), (5, 2), (7, 2),
            (8, 2), (1, 3), (3, 3), (7, 3), (8, 3), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (1, 5),
            (6, 5), (1, 6), (3, 6), (4, 6), (8, 6), (1, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
            (9, 7), (2, 8), (6, 8), (0, 9), (4, 9), (8, 9)
        ]

        self.start_pos = (0, 0)
        self.end_pos = (9, 9)

        self.player_pos = self.start_pos

    def move_player(self, x=0, y=0):
        # move player by given x and y values
        self.player_pos = (self.player_pos[0] + x, self.player_pos[1] + y)

    def on_obstacle(self):
        for pos in self.obs_pos:
            if self.player_pos == (pos[1], pos[0]):
                return True

        return False

    def action(self, action):
        assert 0 <= action <= 3

        ''' movement key and action
              1
            0   2
              3
        '''

        if action == 0 and self.player_pos[0] != 0:
            self.move_player(x=-1)
        elif action == 1 and self.player_pos[1] != 0:
            self.move_player(y=-1)
        elif action == 2 and self.player_pos[0] != 9:
            self.move_player(x=1)
        elif action == 3 and self.player_pos[1] != 9:
            self.move_player(y=1)

        if self.on_obstacle():
            reward = -50
            done = True
        elif self.player_pos == self.end_pos:
            print("Solved!!")
            reward = 500
            done = True
        else:
            reward = -1
            done = False

        if self.type == "deep_q":
            return np.array([self.player_pos]).reshape(2), reward, done  # observation shape is (1, 2)
        elif self.type == "q_table":
            return self.player_pos, reward, done

    def render(self):
        if self.show:
            self.screen.fill((255, 255, 255))

            for pos in self.obs_pos:
                pg.draw.rect(self.screen, (0, 0, 0),
                             (pos[1] * self.block_size, pos[0] * self.block_size, self.block_size, self.block_size))

            pg.draw.rect(self.screen, (255, 0, 0), (self.end_pos[0] * self.block_size, self.end_pos[1] * self.block_size, self.block_size, self.block_size))

            pg.draw.rect(self.screen, (0, 255, 0),
                         (self.player_pos[0] * self.block_size, self.player_pos[1] * self.block_size, self.block_size,
                          self.block_size))

            pg.display.update()

            time.sleep(0.2)

    def reset(self):
        self.player_pos = self.start_pos

        if self.type == "deep_q":
            return np.array([self.player_pos]).reshape(2)
        elif self.type == "q_table":
            return self.player_pos

pg.quit()
