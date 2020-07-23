import numpy as np
import time
import pygame as pg
import random

pg.init()
FPS = 1000  # frames per second setting
fpsClock = pg.time.Clock()
font = pg.font.Font('freesansbold.ttf', 32)


class GameEnv:
    def __init__(self):
        self.generation = 0

        self.screen_dim = 400, 600
        self.screen = pg.display.set_mode(self.screen_dim)

        self.background = pg.transform.scale(pg.image.load("Images\\background.gif"),
                                             (self.screen_dim[0], self.screen_dim[1] - 100))
        self.bird = pg.image.load("Images\\bird.gif")
        self.ground = pg.image.load("Images\\ground.gif")
        self.pipet = pg.image.load("Images\\pipet.gif")
        self.pipeb = pg.image.load("Images\\pipeb.gif")

        self.bird_x, self.bird_y = 150, 300
        self.gap = 450
        self.score = 0

        self.bird_pl = None
        self.gn = None
        self.pt = None
        self.pb = None

        y = random.randint(0, self.screen_dim[1] - self.gap - 100)
        self.pipet_bottom_y, self.pipet_bottom_x = y, self.screen_dim[0]
        self.pipeb_top_y, self.pipeb_top_x = y + self.gap, self.screen_dim[0]

    def step(self, action):
        # 0 - nothing
        # 1 - jump

        scored = self.update()

        if action == 1:
            self.fly_bird()

        done = self.check_collision()
        if done:
            reward = -100
        elif scored:
            reward = 15
        elif main_scored:
            reward = 200
        else:
            reward = -5

        state = self.pipet_bottom_y - self.bird_y
        return np.array([state, reward, done])

    def fly_bird(self):
        if self.bird_y - 100 > 0:
            self.bird_y -= 100

    def update(self):
        global FPS
        self.screen.blit(self.background, (0, 0))

        self.bird_pl = self.screen.blit(self.bird, (self.bird_x, self.bird_y))
        self.bird_fall()
        scored, main_scored = self.move_pipes()
        self.gn = self.screen.blit(self.ground, (0, self.screen_dim[1] - 100))

        self.screen.blit(font.render(f"Generation: {self.generation}", True, (255, 255, 255)), (20, 20))
        self.screen.blit(font.render(f"Score: {self.score}", True, (255, 255, 255)), (20, 50))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_f:
                    FPS += 5
                elif event.key == pg.K_g:
                    FPS -= 5

        fpsClock.tick(FPS)

        return scored, main_scored

    def render(self):
        pg.display.update()

    def move_pipes(self):
        scored = False
        main_scored = False

        if self.pipet_bottom_x <= 0:
            self.score += 1
            y = random.randint(0, self.screen_dim[1] - self.gap - 100)
            self.pipet_bottom_y, self.pipet_bottom_x = y, self.screen_dim[0]
            self.pipeb_top_y, self.pipeb_top_x = y + self.gap, self.screen_dim[0]
            main_scored = True

        if self.pipet_bottom_x <= self.bird_x:
            scored = True

        move_by = 5
        self.pipeb_top_x -= move_by
        self.pipet_bottom_x -= move_by

        self.pt = self.screen.blit(self.pipet, (self.pipet_bottom_x, self.pipet_bottom_y - self.pipet.get_height()))
        self.pb = self.screen.blit(self.pipeb, (self.pipeb_top_x, self.pipeb_top_y))

        return scored, main_scored

    def bird_fall(self):
        self.bird_y += 10

    def check_collision(self):
        if self.bird_pl.colliderect(self.gn) or self.bird_pl.colliderect(self.pt) or self.bird_pl.colliderect(self.pb):
            return True
        return False

    def get_start_state(self):
        state = self.pipet_bottom_y - self.bird_y
        return state

    def reset(self):
        self.generation += 1
        self.bird_x, self.bird_y = 150, 300
        self.gap = 200
        self.score = 0

        y = random.randint(0, self.screen_dim[1] - self.gap - 100)
        self.pipet_bottom_y, self.pipet_bottom_x = y, self.screen_dim[0]
        self.pipeb_top_y, self.pipeb_top_x = y + self.gap, self.screen_dim[0]
