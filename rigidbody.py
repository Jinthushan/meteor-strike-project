import pygame, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 138, 22)
BLUE = (97, 194, 250)

class RigidBody:
    def __init__(self,x,y,vx,vy,radius,mass):
        self.crashed = False

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
        self.vt = 300
        
        self.radius = radius
        self.mass = mass
        #(1/4) * (self.mass) * (self.radius**2)
        Ixx = (1/2) * (self.mass) * (self.radius**2)
        Iyy = Ixx
        Izz = Ixx
        self.Ibody = np.identity(3)
        
        self.Ibody[0][0] = Ixx #inertia tensor 
        self.Ibody[1][1] = Iyy
        self.Ibody[2][2] = Izz

        self.body_matrix = np.array([-self.radius*2,0,0])

        self.IbodyInv = np.linalg.inv(self.Ibody)  # inverse of inertia tensor
        self.v = np.array([[self.vx,self.vy,0]])    
        #self.omega = np.array([0,0,0])   # angular velocity

        self.omega = np.array([0,0,1*Izz]) 

        self.state = np.zeros(19)
        self.state[0:3] = np.array([self.x,self.y,0])               # position
        self.state[3:12] = np.identity(3).reshape([1,9])  # rotation
        self.state[12:15] = self.mass * self.v            # linear momentum
        self.state[15:18] = self.omega                # angular momentum

        # Computed quantities
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

        # Setting up the solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_f_params(self.force, self.torque, self.IbodyInv)

    def f(self, t, state, force, torque, IbodyInv):
        rate = np.zeros(19)
        v = state[12:15] / self.mass

        _R = state[3:12].reshape([3,3])
        _R = self.orthonormalize(_R)

        Iinv = np.dot(_R, np.dot(IbodyInv, _R.T))    

        _L = state[15:18]
        omega = np.dot(Iinv, _L)

        center_mass = state[0:3]

        F_grav = np.array([0,self.mass*(-9.8),0])

        C2 = (self.mass * 9.8)/(self.vt**2)
        
        speed = np.linalg.norm(v)

        F_drag = -C2 * speed * v

        force = F_grav + F_drag

        point_worldframe = center_mass + _R @ self.body_matrix

        r = point_worldframe - center_mass

        torque = np.cross(r,F_drag)

        rate[0:3] = v
        rate[3:12] = np.dot(self.star(omega), _R).reshape([1,9])
        rate[12:15] = force
        rate[15:18] = torque
        return rate

    def star(self, v):
        vs = np.zeros([3,3])
        vs[0][0] = 0
        vs[1][0] = v[2]
        vs[2][0] = -v[1]
        vs[0][1] = -v[2]
        vs[1][1] = 0
        vs[2][1] = v[0]
        vs[0][2] = v[1] 
        vs[1][2] = -v[0]
        vs[2][2] = 0
        return vs;       

    def orthonormalize(self, m):
        mo = np.zeros([3,3])
        r0 = m[0,:]
        r1 = m[1,:]
        r2 = m[2,:]
        
        r0new = r0 / np.linalg.norm(r0)
        
        r2new = np.cross(r0new, r1)
        r2new = r2new / np.linalg.norm(r2new)

        r1new = np.cross(r2new, r0new)
        r1new = r1new / np.linalg.norm(r1new)

        mo[0,:] = r0new
        mo[1,:] = r1new
        mo[2,:] = r2new
        return mo

    def get_pos(self):
        return self.state[0:3]

    def get_rot(self):
        return self.state[3:12].reshape([3,3])

    def get_angle_2d(self):
        v1 = [1,0,0]
        v2 = np.dot(self.state[3:12].reshape([3,3]), v1)
        cosang = np.dot(v1, v2)
        axis = np.cross(v1, v2)
        return np.degrees(np.arccos(cosang)), axis

    def prn_state(self):
        print('Pos', self.state[0:3])
        print('Rot', self.state[3:12].reshape([3,3]))
        print('P', self.state[12:15])
        print('L', self.state[15:18])

class Box2d(pygame.sprite.Sprite):
    def __init__(self, screen_height, imgfile):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(imgfile).convert_alpha()
        self.rect = self.image.get_rect()
        self.pos = (0,0)
        self.image_rot = self.image
        self.screen_height = screen_height
        self.alpha = 255

    def rotate(self, angle):
        self.image_rot = pygame.transform.rotate(self.image, angle)

    def set_alpha(self,alpha):
        if alpha >= 0 and alpha <= 255:
            self.alpha = alpha

    def draw(self, surface):
        rect = self.image_rot.get_rect()
        rect.center = self.pos
        self.image_rot.set_alpha(self.alpha)
        surface.blit(self.image_rot, rect)

def main():
    pygame.init()
    clock = pygame.time.Clock()

    win_width = 1500
    win_height = 1000
    screen = pygame.display.set_mode((win_width, win_height))
    screen_center = np.array([win_width/2,win_height/2])
    pygame.display.set_caption('METEOR STRIKE')

    box = Box2d(win_height, 'meteor_no_trail_small.png')

    rb = RigidBody(x=350,y=900,vx=100,vy=-200,radius=20,mass=1000)

    cur_time = 0.0
    dt = 0.1

    rb.solver.set_initial_value(rb.state, cur_time)

    surface = pygame.Surface((win_width,win_height))
    surface.fill(BLUE)

    floor_height = 150
    grass_height = win_height - floor_height

    pygame.draw.rect(surface,GREEN, pygame.Rect(0,grass_height, win_width, floor_height))

    explosion_timer = 0
    density = 2500
    g = 9.8

    while True:
        # 30 fps
        clock.tick(30)

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)
        else:
            pass

        rb.state = rb.solver.integrate(cur_time)
        cur_time += dt

        angle, axis = rb.get_angle_2d()
        if axis[2] < 0:
            angle *= -1.

        pos = rb.get_pos()
        meteor_bottom = pos[1] - rb.radius

        if meteor_bottom <= floor_height:

            vel = rb.state[12:15] /rb.mass
            speed = np.linalg.norm(vel)
            KE = (0.5)*rb.mass*(speed**2)
            crater_dia = int(1.161 * (KE /(density*g))**(1/4))

            rb.state[1] = 100 + rb.radius
            rb.state[12:15] = np.zeros(3)
            rb.state[15:18] = np.zeros(3)
            box.image = pygame.image.load('explosion.png').convert_alpha()
            explosion_timer = 60

            rb.crashed = True

        if explosion_timer > 0:
            new_alpha = box.alpha - 5
            box.set_alpha(new_alpha)
            explosion_timer -= 1

        

        # clear the background, and draw the sprites
        screen.blit(surface, (0,0))

        if rb.crashed == True:
            crater_x = pos[0]
            crater_y = grass_height
            scaling = 50 

            crater_dia *= scaling

            pygame.draw.circle(surface,BLUE, (crater_x,crater_y), crater_dia // 2)

        box.rotate(angle)

        screen_x = pos[0]
        screen_y = win_height - pos[1]
        box.pos = (screen_x,screen_y)
        box.draw(screen)

        pygame.display.update()

if __name__ == '__main__':
    main()