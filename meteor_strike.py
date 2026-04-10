import pygame, sys
import numpy as np
from scipy.integrate import ode
import random
import math

NIGHT_SKY = (10, 14, 40)
ROCK_GRAY = (89, 89, 92)
ROCK_DARKGRAY = (60, 60, 61)

CRATER_LINE_BROWN = (61, 50, 40)
CRATER_BROWN = (43, 36, 29)
CRATER_LIGHT = (71, 59, 53)

g = 9.8
floor_density = 2000

win_width = 1200
win_height = 700

floor_height = 150
grass_height = win_height - floor_height

seed = 43
np.random.seed(seed)

class Meteor:
    def __init__(self,x,y,vx,vy,radius,mass):
        self.crashed = False
        self.isfragment = False

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
        self.vt = 300
        
        self.radius = radius
        self.mass = mass

        Ixx = (1/2) * (self.mass) * (self.radius**2)
        Iyy = Ixx
        Izz = Ixx
        self.Ibody = np.identity(3)
        
        self.Ibody[0][0] = Ixx #inertia tensor 
        self.Ibody[1][1] = Iyy
        self.Ibody[2][2] = Izz

        self.body_matrix = np.array([-self.radius*2,0,0])

        self.IbodyInv = np.linalg.inv(self.Ibody)  # inverse of inertia tensor
        self.v = np.array([self.vx,self.vy,0])    
        #self.omega = np.array([0,0,0])   # angular velocity

        self.omega = np.array([0,0,3*Izz]) 

        self.state = np.zeros(19)
        self.state[0:3] = np.array([self.x,self.y,0])               # position
        self.state[3:12] = np.identity(3).reshape([1,9])  # rotation
        self.state[12:15] = self.mass * self.v            # linear momentum
        self.state[15:18] = self.omega                # angular momentum

        self.force = np.zeros(3)
        self.torque = np.zeros(3)

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

    def get_angle_2d(self):
        v1 = [1,0,0]
        v2 = np.dot(self.state[3:12].reshape([3,3]), v1)

        angle = np.degrees(np.arctan2(v2[1],v2[0]))

        return angle
    
    def kinetic_energy(self):
        v = self.state[12:15] / self.mass
        speed = np.linalg.norm(v)
        KE_linear = 0.5 * self.mass * speed**2

        R = self.state[3:12].reshape([3,3])

        I_world = R @ self.Ibody @ R.T

        L = self.state[15:18]
        
        omega = np.linalg.inv(I_world) @ L

        KE_ang = 0.5 * np.dot(omega, (I_world @ omega))

        return KE_linear + KE_ang
    
    def step(self,dt):
        self.old_pos = self.state[0:3].copy()
        self.solver.integrate(self.solver.t + dt)
        self.state = self.solver.y

class Terrain:
    def __init__(self,width,height,ground_y):
        self.width = width
        self.height = height
        self.ground_y = ground_y
        
        self.heights = np.full(width,float(ground_y))
    
    def draw_crater(self, surface):
        for x in range(self.width):
            screen_y = int(self.heights[x])
            if screen_y > self.ground_y:
                pygame.draw.line(surface,CRATER_LIGHT, (x,self.ground_y), (x, screen_y))

    def make_crater(self, impact_x, diameter):
        radius = int(diameter/2)

        depth = max(6,(diameter) * 0.35)
        
        x0 = int(impact_x - radius)
        x1 = int(impact_x + radius)

        for x in range(max(0,x0), min(self.width,x1+1)):
            dx = x - impact_x 
            frac = dx / radius
            dip = depth * (1-frac**2)
            new_y = self.heights[x] + dip
            if new_y > self.heights[x]:
                self.heights[x] = new_y
    
    def surface_y_at(self,x):
        xi = int(np.clip(x,0,self.width - 1))
        return self.heights[xi]
    
    def draw(self, surface):
        points = []
        for x in range(self.width):
            points.append((x,int(self.heights[x])))
        points.append((self.width, self.height))
        points.append((0,self.height))

        pygame.draw.polygon(surface,CRATER_BROWN, points)
        
        edge = [(x, int(self.heights[x])) for x in range(self.width)]
        if len(edge) >= 2:
            pygame.draw.lines(surface, CRATER_LINE_BROWN, False, edge, 2)

def pos_to_screen(x,y):
    return int(x), int(grass_height - y)

def screen_to_pos(x,y):
    return float(x), float(grass_height - y)

def spawn_fragments(meteor, count):
    frags = []

    main_v = np.array([meteor.vx, meteor.vy,0.0])

    KE_total = meteor.kinetic_energy()
    ke_loss = 0.1
    r = meteor.radius / count

    fragment_mass = meteor.mass/count 

    for _ in range(count):
        angle = random.uniform(math.pi/6, math.pi)
        direction = np.random.choice([-1, 1])

        ke_frag = (KE_total * (1-ke_loss))/count

        speed = np.sqrt(2 * ke_frag / fragment_mass)

        vx = direction * speed * np.cos(angle) + main_v[0]
        vy = speed * np.sin(angle) + main_v[1]

        f = Meteor(meteor.state[0], meteor.state[1] + 75, vx, vy, r, fragment_mass)
        f_img = MeteorImage(win_height, r, is_fragment = True)

        f.isfragment = True
        f.solver.set_initial_value(f.state, meteor.solver.t)
        frags.append([f,f_img])

    return frags

class MeteorImage(pygame.sprite.Sprite):
    def __init__(self, screen_height, radius, is_fragment = False):
        pygame.sprite.Sprite.__init__(self)
        
        self.alpha = 255
        self.pos = (0,0)
        
        self.is_fragment = is_fragment
        self.radius = radius

        self.image = self.draw_fragment(radius) if is_fragment else self.draw_meteor(radius)
        self.image_rot = self.image.copy()

    def draw_meteor(self,radius):
        size = radius * 5
        surface = pygame.Surface((size,size), pygame.SRCALPHA)
    
        circle_x = size // 2
        circle_y = size // 2 

        pygame.draw.circle(surface,ROCK_GRAY, (circle_x,circle_y), radius)

        hole_x = int(radius * 0.25)
        hole_y = int(radius * 0.2)
        hole_r = int(radius * 0.45)

        hole_center = (circle_x + hole_x, circle_y + hole_y)

        pygame.draw.circle(surface, ROCK_DARKGRAY, hole_center, hole_r)

        return surface

    def draw_fragment(self,radius):
        size = radius * 4
        surf = pygame.Surface((size,size), pygame.SRCALPHA)

        x = size // 2
        y = size // 2

        points = [(x,y-radius), (x-radius, y + radius), (x+radius, y+radius)]

        pygame.draw.polygon(surf, ROCK_GRAY, points)

        return surf
    
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

def crater_dia(KE):
    raw = 1.161*((KE)/(floor_density*g))**(0.25)
    
    return raw*6

def step_body(body,dt,terrain):
    body.step(dt)
    pos = body.get_pos()
    x_screen ,y_screen = pos_to_screen(pos[0], pos[1])
    terrain_y = terrain.surface_y_at(x_screen)

    v = body.state[12:15] / body.mass
    
    hit = y_screen >= terrain_y and v[1] < 0

    return x_screen,y_screen,terrain_y, hit

def main():
    sim_time = 0

    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('METEOR STRIKE')

    meteors = []
    fragments = []
    terrain = Terrain(win_width,win_height, grass_height)

    sky_surface = pygame.Surface((win_width, grass_height))
    sky_surface.fill(NIGHT_SKY)

    meteor = Meteor(x=150,y=650,vx=50,vy=-100,radius=20,mass=100000)
    meteor_img = MeteorImage(win_height,20,False)

    meteor2 = Meteor(x=950,y=700,vx=0,vy=-100,radius=20,mass=100000)
    meteor_img2 = MeteorImage(win_height,20,False)

    meteors.append([meteor,meteor_img])
    meteors.append([meteor2,meteor_img2])

    for meteor,_ in meteors:
        meteor.solver.set_initial_value(meteor.state, sim_time)

    while True:
        # 30 fps
        dt = clock.tick(60) /1000
        sim_time += dt
        
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        to_remove_meteors = []
        new_frags = []

        for m, img in meteors:
            if m.crashed == True:
                to_remove_meteors.append([m,img])
                continue
            cur_x, cur_y , terrain_y,hit = step_body(m,dt,terrain)

            if hit == True:
                m.crashed = True
                KE = m.kinetic_energy()
                dia = crater_dia(KE)
                terrain.make_crater(cur_x,dia)
        
                n_frags = random.randint(4,9)
            
                screen_x, screen_y = screen_to_pos(cur_x,int(terrain_y))

                new_frags += spawn_fragments(m,n_frags)
                to_remove_meteors.append([m,img])
        
        for meteor in to_remove_meteors:
            if meteor in meteors:
                meteors.remove(meteor)

        fragments += new_frags    


        to_remove_f = []
        for f,img in fragments:
            if f.crashed:
                to_remove_f.append([f,img])
                continue
            
            cur_x,cur_y, terrain_y, hit = step_body(f,dt,terrain)
        
            if hit:
                f.crashed = True
                KE = f.kinetic_energy()
                dia = crater_dia(KE)
                dia *= 0.5
                terrain.make_crater(cur_x, dia)
                
                to_remove_f.append([f,img])
        
        for f in to_remove_f:
            if f in fragments:
                fragments.remove(f)

        screen.blit(sky_surface, (0,0))
        
        
        for m,img in meteors:
            pos = m.get_pos()

            screen_x,screen_y = pos_to_screen(pos[0],pos[1])

            img.pos = (screen_x,screen_y)

            angle = m.get_angle_2d()
            img.rotate(angle)

            img.draw(screen)

        terrain.draw_crater(screen)

        for f,img in fragments:
            pos = f.get_pos()
            x,y = pos_to_screen(pos[0],pos[1])

            img.pos = (x,y)

            angle = f.get_angle_2d()
            img.rotate(angle)
            img.draw(screen)
        
        terrain.draw(screen)
        
        pygame.display.update()

if __name__ == '__main__':
    main()