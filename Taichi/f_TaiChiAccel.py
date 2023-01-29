import pygame as pg
import numpy as np
import taichi as ti

res = width, height = 800, 600
offset = np.array([1.3 * width, height]) // 2
max_itr = 300
zoom = 2.2/height
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size())-1
texture_array = pg.surfarray.array3d(texture)

@ti.data_oriented
class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((width,height,3), [0,0,0], dtype=np.uint8)
        #taichi archiecture
        ti.init(arch=ti.metal)
        #taichi fields 
        self.screen_field = ti.Vector.field(3, ti.uint8, (width, height))
        self.texture_field = ti.Vector.field(3, ti.uint8, texture.get_size())
        self.texture_field.from_numpy(texture_array)

    @ti.kernel
    def render(self):
        # for x in range(width):
        #     for y in range(height):
        for x, y in self.screen_field: #parallelization loop
            # c = (x-offset[0]) * zoom + 1j * (y-offset[1]) * zoom
            # z = 0
            c = ti.Vector([(x-offset[0]) * zoom, (y-offset[1]) * zoom])
            z = ti.Vector([0.0, 0.0])
            num_itr = 0
            for i in range(max_itr):
                # z = z**2 + c
                z = ti.Vector([(z.x ** 2 - z.y ** 2 + c.x), (2 * z.x * z.y + c.y)])
                # if abs(z) > 2:
                if z.dot(z) > 4:
                    break
                num_itr += 1
            col = int(texture_size * num_itr/max_itr)
            # self.screen_array[x,y] = texture_array[col,col]
            self.screen_field[x,y] = self.texture_field[col,col]
                

    def update(self):
        self.render() 
        self.screen_array = self.screen_field.to_numpy()

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps():.2f}')

if __name__ == '__main__':
    app = App()
    app.run()