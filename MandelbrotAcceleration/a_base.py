import pygame as pg
import numpy as np

res = width, height = 800, 600

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()

    def run(self):
        while True:
            self.screen.fill('black')
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps():.2f}')

if __name__ == '__main__':
    app = App()
    app.run()