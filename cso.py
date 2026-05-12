import numpy
import pygame

PIXEL_SCALING_FACTOR = 1
GRAVITATIONAL_CONSTANT = 6.67430e-11
SOFTENING = 1000

def step_all_csos(dt):

    for obj in CircularSolarObject.objects:
        obj.acceleration_vector = pygame.math.Vector2(0, 0)
        obj.accelerations = []

    number_of_cso = len(CircularSolarObject.objects)

    for j in range(number_of_cso):
        obj = CircularSolarObject.objects[j]
        pos_j = obj.position_vector_in_meters
        for i in range(j + 1, number_of_cso):
            obi = CircularSolarObject.objects[i]
            pos_i = obi.position_vector_in_meters

            r = pos_i - pos_j
            dist_squared = r.length_squared()

            if dist_squared > 0.1:
                dist = numpy.sqrt(dist_squared)

                force_value_j = GRAVITATIONAL_CONSTANT * (obi.mass / dist_squared) * obj.mass
                force_value_i = GRAVITATIONAL_CONSTANT * (obj.mass / dist_squared) * obi.mass

                softening_factor = (dist_squared + SOFTENING)**1.5

                acceleration_vector_j = (GRAVITATIONAL_CONSTANT * obi.mass / softening_factor) * r
                acceleration_vector_i = (GRAVITATIONAL_CONSTANT * obj.mass / softening_factor) * -r

                obj.acceleration_vector += acceleration_vector_j
                obi.acceleration_vector += acceleration_vector_i

                force_angle_j_i = numpy.degrees(numpy.atan2(r.y, r.x))
                force_angle_i_j = numpy.degrees(numpy.atan2(-r.y, -r.x))

                obj.accelerations.append([obi, force_value_j, force_angle_j_i])
                obi.accelerations.append([obj, force_value_i, force_angle_i_j])

    for obj in CircularSolarObject.objects:
        obj.speed_vector += obj.acceleration_vector * dt
        obj.position_vector_in_meters += obj.speed_vector * dt
        obj.position_vector_in_pixels = obj.position_vector_in_meters / PIXEL_SCALING_FACTOR

class CircularSolarObject():

    objects = []
    
    def __init__(self, density: float, mass: float, starting_position: tuple, color, speed_vector: tuple = (0, 0), acceleration_vector: tuple = (0, 0)):
        self.density: float = density  ### Unit in kg/m^3
        self.mass: float = mass         ### Unit in kg  ->  GUI with adjustiable num and 10th power (x*10^y)

        self.position_vector_in_pixels = pygame.math.Vector2(starting_position[0], starting_position[1])
        self.position_vector_in_meters = self.position_vector_in_pixels * PIXEL_SCALING_FACTOR
        self.drawing_position: tuple = self.position_vector_in_pixels
        self.speed_vector = pygame.math.Vector2(speed_vector[0], speed_vector[1])
        self.acceleration_vector = pygame.math.Vector2(acceleration_vector[0], acceleration_vector[1])
        self.accelerations: list = []

        if isinstance(color, tuple):
            self.color: tuple = color
        elif isinstance(color, str):
            from utilities import get_color_from_string
            self.color: tuple = get_color_from_string(color)
        else:
            print(f"Invalid instance: {color} -> {type(color)}; Expected String or Tuple")

        ### Converts mass and density to its respective radius in meters and pixels
        if self.mass and self.density > 0:
            self.volume: float = self.mass / self.density
            self.radius_in_meters: float = numpy.power(((3 * self.volume) / (4 * numpy.pi)), (1/3))
            self.radius_in_pixels: float = self.radius_in_meters / PIXEL_SCALING_FACTOR
            self.display_radius = self.radius_in_pixels
        else:
            print(f"Invalid value mass - {self.mass} or density - {self.density}; Expected > 0")

        self.locked_on = False
        
        CircularSolarObject.objects.append(self)
    
    def draw(self, target_window):
        ### Draws cso onto given window
        if self.locked_on:
            pygame.draw.circle(target_window, center=self.drawing_position, color=(255, 255, 255), radius=self.display_radius * 1.1)
        else: pass
        
        pygame.draw.circle(target_window, center=self.drawing_position, color=self.color, radius=self.display_radius)