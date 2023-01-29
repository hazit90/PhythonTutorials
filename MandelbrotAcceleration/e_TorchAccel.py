import pygame as pg
import numpy as np
import math
import torch
import time
computeDevice = torch.device("mps")
cpuDevide = torch.device("cpu")


res = width, height = 600, 600
offset = np.array([1.3 * width, height]) // 2
offsetT = torch.tensor(offset, dtype=torch.float16, device=computeDevice)
max_iter = 300
zoom = 2.2/height
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size())-1
texture_array = pg.surfarray.array3d(texture)


class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((width,height,3), [0,0,0], dtype=np.uint8)
        self.xT = torch.linspace(0, width, steps=width, dtype=torch.float16, device=cpuDevide)
        self.yT = torch.linspace(0, height, steps=height, dtype=torch.float16, device=cpuDevide)

        #create a width x height x 2 tensor
        #populate coordinates tensor with coordinate values
        # self.coordinatesT = torch.empty((width, height, 2), dtype=torch.float16, device=cpuDevide)
        self.coordinates = np.empty((width, height, 2), dtype=np.float16)
        self.coordinatesT = torch.from_numpy(self.coordinates)
        for x in range(width):
            for y in range(height):
                self.coordinatesT[x][y][0] = self.xT[x]
                self.coordinatesT[x][y][1] = self.yT[y]

        self.coordinatesT = self.coordinatesT.to(computeDevice)       
                

    def render(self):
        zT = (self.coordinatesT - (offsetT)) * zoom
        cT = torch.clone(zT)
        # [x + yi]^2 = x^2 + 2xyi - y^2
        # checks if we have hit the max iterations for a z element 
        num_iterT = torch.full((width, height), max_iter, dtype=torch.int32, device=computeDevice)
        realPart = 0
        fakePart = 0
        finalPart = 0
        for i in range(max_iter):
            maskT = (torch.eq(num_iterT, max_iter))
            #real part = x^2 - y^2
            # ztPreviewA = zT[:,:,0][maskT]
            ztTemp = zT[:,:,0][maskT]
            st = time.time()
            # zT[:,:,0][maskT] = torch.square(zT[:,:,0][maskT]) - torch.square(zT[:,:,1][maskT]) + cT[:,:,0][maskT]
            partA = torch.subtract(torch.square(zT[:,:,0][maskT]), torch.square(zT[:,:,1][maskT]))
            zT[:,:,0][maskT] = torch.add(partA, cT[:,:,0][maskT])
            et  = time.time()
            realPart += (et-st)*1000

            #imaginary part = 2xy
            # zT[:,:,1][maskT] =  2 * ztTemp * zT[:,:,1][maskT] + cT[:,:,1][maskT]
            st = time.time()
            zT[:,:,1][maskT] = torch.add(torch.multiply(torch.multiply(2, ztTemp), zT[:,:,1][maskT]), cT[:,:,1][maskT])
            et  = time.time()
            fakePart += (et-st)*1000
            # zTPreviewB  = zT[:,:,1]
            # check break condition and log iterations
            # 2 conditions, first condition is outside circle of radius 2, the other condition is that we exceed the max iterations
            # num_iterT[maskT & (zT[:,:,0] ** 2 + zT[:,:,1] ** 2 > 4.0)] = i + 1
            st = time.time()
            num_iterT[maskT & (torch.add(torch.square(zT[:,:,0]), torch.square(zT[:,:,1])) > 4.0)] = i + 1
            et  = time.time()
            finalPart += (et-st)*1000

            breakPoint = 0

        print("realPart: ", realPart, " fakePart: ", fakePart, " finalPart: ", finalPart)
        #print ms
        colT = (torch.multiply(torch.t(num_iterT), texture_size/max_iter)).type(torch.uint8)
        #transpose torch tensor
        # colT = colT.T
        #move to cpu
        col = colT.cpu().numpy()
        self.screen_array = texture_array[col,col]        
                

    def update(self):
        self.render() 

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