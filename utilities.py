import pygame
import numpy
from cso import CircularSolarObject
CSO = CircularSolarObject

GRAVITATIONAL_CONSTANT = 6.67430e-11
AU_IN_METERS = 149597870700
LIGHTSPEED = 299792458

def get_color_from_string(color: str) -> tuple:
    ### Converts string-color into rgb
    COLORS: dict = {
        "WHITE": (255, 255, 255),
        "RED": (255, 0, 0),
        "BLUE": (0, 0, 250),
        "GREEN": (0, 255, 0),
        "YELLOW": (255, 255, 0),
        "BLACK": (0, 0, 0),
        "GRAY": (50, 50, 50),
        "LIGHT GRAY": (229, 229, 229),
        "BEIGE": (238, 203, 139),
        "ORANGE": (201, 144, 57),
        "LIGHT BEIGE": (237, 219, 173),
        "LIGHT BLUE": (172, 229, 238),
        "TURQUOISE": (124, 183, 187)
    }
    rgb_value = COLORS.get(color)
    if rgb_value is not None:
        return rgb_value
    else:
        print(f"Invalid Color-String: {color}")

def draw_specific_csos(target_window, csos: tuple):
    ### Draws all objects from a tuple in given order
    for obj in csos:
        pygame.draw.circle(target_window, center=obj.position, color=obj.color, radius=obj.display_radius)

def draw_all_csos(target_window):
    ### Draws all objects starting with the biggest
    cso_objects = list(CSO.objects)
    cso_objects.sort(key=lambda obj: obj.radius_in_meters, reverse=True)
    for obj in cso_objects:
        obj.draw(target_window)

def camera_movement(zoom_factor, offset_vector, inflator):
        ### Moves the camera and operates the zoom (values are only updated when on black background)
        for obj in CSO.objects:
            obj.drawing_position = (obj.position_vector_in_pixels - offset_vector) * zoom_factor + (750, 500)
            obj.display_radius = max(2, (numpy.sqrt(obj.radius_in_pixels)* 5) * zoom_factor * inflator)
            
def cso_lock_on(mouse_position_vector):
    for obj in CSO.objects:
        distance_mouse_to_obj = mouse_position_vector.distance_to(obj.drawing_position)
        if distance_mouse_to_obj <= obj.display_radius and pygame.mouse.get_just_pressed()[0] == 1:
            if obj.locked_on: obj.locked_on = False
            else: obj.locked_on = True

        if obj.locked_on:
            return pygame.math.Vector2(obj.position_vector_in_pixels)

def display_radial_texts(target_window, object):
    from ui_elements import RadialText

    accelerations = sorted(object.accelerations, key=lambda x: x[2])
    last_angle = -999
    current_layer = 0

    for i, data in enumerate(accelerations):
        color = data[0].color
        value = data[1]
        angle = data[2]

        if abs(angle - last_angle) < 5: current_layer += 1
        else: current_layer = 0

        last_angle = angle

        text_string = f"{value:.2e} [N]"
        text_object = RadialText(text=text_string, cso=object, angle=angle, font="Arial", text_size=70, color=color)

        distance = 2 + (current_layer * 1.75)

        text_object.draw(target_window, distance)