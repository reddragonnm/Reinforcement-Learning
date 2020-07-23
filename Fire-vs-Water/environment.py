# CREATED BY:
# Neelesh Om Jagadish Mishra

# importing all modules
import numpy as np  # numpy for maintaining the q-table
import pygame as pg  # pygame to show the visual for the environment
import time  # time to slow the steps of the computer visually

pg.init()
block_size = 100  # size of each box

# some images to make the environment visually appealing and to know what is happening
# "template_for_robots" in images folder is just in case you want to change the robot
player_image = pg.transform.scale(pg.image.load(
    "images\\player.png"), (block_size, block_size))
fire_image = pg.transform.scale(pg.image.load(
    "images\\fire.png"), (block_size, block_size))
water_image = pg.transform.scale(pg.image.load(
    "images\\water.jpg"), (block_size, block_size))


class GameEnv:
    def __init__(self, height=20, width=3):
        # dimensions of the environment
        self.height = height
        self.width = width

        # initiating player, fire and water positions
        self.player_pos = (np.random.randint(0, self.height),
                           np.random.randint(0, self.width))
        self.fire_pos = (np.random.randint(0, self.height),
                         np.random.randint(0, self.width))
        self.water_pos = (np.random.randint(0, self.height),
                          np.random.randint(0, self.width))

        # initiating the pygame screen
        screen_dim = (self.height * block_size, self.width * block_size)
        self.screen = pg.display.set_mode(screen_dim)
        pg.display.set_caption('Fire vs Water using Reinforcement Learning')

    def move_player(self, x=0, y=0):
        # move player by given x and y values
        self.player_pos = (self.player_pos[0] + x, self.player_pos[1] + y)

    def get_rel_pos(self):
        # relative position between fire/water and player
        self.fire_rel = (
            self.fire_pos[0] - self.player_pos[0], self.fire_pos[1] - self.player_pos[1])
        self.water_rel = (
            self.water_pos[0] - self.player_pos[0], self.water_pos[1] - self.player_pos[1])

        pos = (self.fire_rel, self.water_rel)
        return pos

    def action(self, action):
        assert 0 <= action <= 3  # ensure action is valid

        ''' movement key and action
              1
            0   2
              3
        '''

        # implementing the action
        if action == 0 and self.player_pos[0] != 0:
            self.move_player(x=-1)
        elif action == 1 and self.player_pos[1] != 0:
            self.move_player(y=-1)
        elif action == 2 and self.player_pos[0] != self.height - 1:
            self.move_player(x=1)
        elif action == 3 and self.player_pos[1] != self.width - 1:
            self.move_player(y=1)

        # giving reward based on the action
        if self.player_pos == self.fire_pos:
            reward = -50
            completed = True
        elif self.player_pos == self.water_pos:
            reward = 50
            completed = True
        else:
            reward = -2
            completed = False

        pos = self.get_rel_pos()
        # returning values
        return np.array([(pos[0], pos[1]), reward, completed])

    def make_sprite(self, pos, img):
        # drawing the images on the screen
        self.screen.blit(img, (pos[0] * block_size, pos[1] * block_size))
        pg.display.update()

    def render(self):
        # showing the progress of the AI visually
        self.screen.fill((0, 0, 0))

        self.make_sprite(self.fire_pos, fire_image)
        self.make_sprite(self.water_pos, water_image)
        self.make_sprite(self.player_pos, player_image)

        time.sleep(0.1)  # wait to make the movement of the AI "seeable"

    def reset(self):
        # reset the screen and position of the sprites(fire, water and player)
        self.fire_pos = (np.random.randint(0, self.height),
                         np.random.randint(0, self.width))
        self.water_pos = (np.random.randint(0, self.height),
                          np.random.randint(0, self.width))
        self.player_pos = (np.random.randint(0, self.height),
                           np.random.randint(0, self.width))
