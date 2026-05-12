import pygame
from cso import *
from ui_elements import *
from utilities import *
CSO = CircularSolarObject

def setup_pygame_window(width: int = 1920, height: int = 1080):
    screen = pygame.display.set_mode((width, height))
    pygame.display.flip()
    pygame.display.set_caption('Simulation')
    return screen

def update(camera_offset):
    global running
    real_delta_time = clock.tick(60) / 1000
    sim_speed = 86000 * 100

    sub_steps = 100
    dt_per_step = (real_delta_time * sim_speed) / sub_steps

    screen.fill((0, 0, 0))

    camera_lock_position = cso_lock_on(mouse_position)
    if camera_lock_position is None: pass
    else: camera_offset = camera_lock_position
    
    camera_movement(zoom, camera_offset, 1)

    for _ in range(sub_steps):
         step_all_csos(dt_per_step)

    draw_all_csos(screen)
    for trail in trails:
         trail.draw(screen, zoom, camera_offset, 1)

    display_radial_texts(screen, Sun)
    display_radial_texts(screen, Jupiter)

    if Extend_Button.pressed == False:
         UI_Background.width = 350
         UI_Background.position = (1570, 0)
         Extend_Button.position = (1570, 540)
         Extend_Button.set_new_image("close_button.png")
    else:
        UI_Background.width = 10
        UI_Background.position = (1910, 0)
        Extend_Button.position = (1910, 540)
        Extend_Button.set_new_image("extend_button.png")

    UI_Background.draw(screen)
    Extend_Button.update(screen)
    Exit_Button.update(screen)

    if Exit_Button.pressed: running = False

    pygame.display.update()

screen = setup_pygame_window()
clock = pygame.time.Clock()
clock.tick()

Sun = CSO(density=1408, mass=(1.989*10**30), starting_position=(0, 0), color="YELLOW", acceleration_vector=(0, 0), speed_vector=(0, 0))
Mercury = CSO(density=5430, mass=(3.285*10**23), starting_position=(0.39 * AU_IN_METERS, 0), color="LIGHT GRAY", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (0.39 * AU_IN_METERS))))
Venus = CSO(density=5243, mass=(4.867*10**24), starting_position=(0.72 * AU_IN_METERS, 0), color="BEIGE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (0.72 * AU_IN_METERS))))
Earth = CSO(density=5515, mass=(5.972*10**24), starting_position=(AU_IN_METERS, 0), color="BLUE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / 1.496e11)))
Moon = CSO(density=3340, mass=(7.342*10**22), starting_position=(149982270700, 0), color="GRAY", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((6.67430e-11 * 1.989e30) / 1.496e11)) + (numpy.sqrt((6.67430e-11 * 5.972*10**24) / 384400000))))
###Moon2 = CSO(density=3340, mass=(7.342*10**22), starting_position=((AU_IN_METERS - 384400000), 0), color="GRAY", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((6.67430e-11 * 1.989e30) / 1.496e11)) - (numpy.sqrt((6.67430e-11 * 5.972*10**24) / 384400000))))
Mars = CSO(density=3933, mass=(6.39*10**23), starting_position=(2.279e11, 0), color="RED", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / 2.279e11)))
Jupiter = CSO(density=1330, mass=(1.898*10**27), starting_position=(5.2 * AU_IN_METERS, 0), color="ORANGE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (5.2 * AU_IN_METERS))))
Saturn = CSO(density=687, mass=(5.683*10**26), starting_position=(9.5 * AU_IN_METERS, 0), color="LIGHT BEIGE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (9.5 * AU_IN_METERS))))
Uranus = CSO(density=1270, mass=(8.681*10**25), starting_position=(19.2 * AU_IN_METERS, 0), color="LIGHT BLUE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (19.2 * AU_IN_METERS))))
Neptun = CSO(density=1760, mass=(1.024*10**26), starting_position=(30.05 * AU_IN_METERS, 0), color="TURQUOISE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (30.05 * AU_IN_METERS))))

Io = CSO(density=3528, mass=8.94e22, starting_position=((5.2 * AU_IN_METERS) + 421600000, 0), color="GRAY", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((GRAVITATIONAL_CONSTANT * 1.989e30) / (5.2 * AU_IN_METERS))) + numpy.sqrt((GRAVITATIONAL_CONSTANT * 1.898e27) / 421600000)))
Europa = CSO(density=3010, mass=4.8e22, starting_position=((5.2 * AU_IN_METERS) + 671000000, 0), color="LIGHT BLUE", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((6.67430e-11 * 1.989e30) / 777920000000)) + (numpy.sqrt((6.67430e-11 * 1.898e27) / 671000000))))
Ganymed = CSO(density=1940, mass=1.48e23, starting_position=((5.2 * AU_IN_METERS) + 1070000000, 0), color="LIGHT GRAY", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((6.67430e-11 * 1.989e30) / 777920000000)) + (numpy.sqrt((6.67430e-11 * 1.898e27) / 1070000000))))
Kallisto = CSO(density=1834, mass=1.0759e23, starting_position=(779802700000, 0), color="WHITE", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((6.67430e-11 * 1.989e30) / 777920000000)) + (numpy.sqrt((6.67430e-11 * 1.898e27) / 1882700000))))

Pluto = CSO(density=1860, mass=1.3e22, starting_position=(49 * AU_IN_METERS, 0), color="ORANGE", acceleration_vector=(0, 0), speed_vector=(0, 3657.6717322834224))

Ag37 = CSO(density=1000, mass=3.35e19, starting_position=((132.3 * AU_IN_METERS), 0), color="LIGHT BLUE", acceleration_vector=(0, 0), speed_vector=(0, 1540))

trails = []
for obj in CSO.objects:
     Trail = TrailParticles(obj, 5000)
     trails.append(Trail)

Exit_Button = Button((45, 45), "exit_button.png")
Extend_Button = Button((1570, 540), "close_button.png", toggle=True)
UI_Background = Rectangle(position=(1570, 0), width=350, height=1080, color="GRAY")

zoom = 2.437658971708419e-09
ZOOM_SPEED = 2

dragging = False
camera_center = pygame.math.Vector2(1729.65, 1943.64)

running: bool = True
while running and __name__ == "__main__":

    mouse_position = pygame.math.Vector2(pygame.mouse.get_pos())
    world_mouse_before = (mouse_position - pygame.math.Vector2(750, 500)) / zoom + camera_center

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEWHEEL and tuple(screen.get_at(pygame.mouse.get_pos())) == (0,0,0,255):
            world_mouse_before = (mouse_position - pygame.math.Vector2(750, 500)) / zoom + camera_center

            if event.y > 0 : zoom *= ZOOM_SPEED
            else: zoom /= ZOOM_SPEED

            camera_center = world_mouse_before - (mouse_position - pygame.math.Vector2(750, 500)) / zoom
        
        if event.type == pygame.MOUSEBUTTONDOWN:
             if event.button == 1 and tuple(screen.get_at(pygame.mouse.get_pos())) == (0,0,0,255):
                  dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
             if event.button == 1:
                  dragging = False

        if event.type == pygame.MOUSEMOTION:
             if dragging:
                  delta_mouse_position = pygame.math.Vector2(event.rel)
                  camera_center -= delta_mouse_position / zoom

    update(camera_center)