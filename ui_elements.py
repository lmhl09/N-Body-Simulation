import pygame
from utilities import *
from cso import CircularSolarObject as CSO
import math

class Button():

    def __init__(self, position: tuple, image_name: str, scale: float=1, toggle: bool = False):
        self.position: tuple = position
        self.scale: float = scale
        self.image = pygame.image.load(image_name).convert_alpha()
        self.width: float = self.image.get_width()
        self.height: float = self.image.get_height()
        self.scaled_image = pygame.transform.scale(self.image, (int(self.width * self.scale), int(self.height * self.scale)))
        self.rectangle = self.scaled_image.get_rect()
        self.rectangle.center = self.position

        self.hover = False
        self.pressed = False     
        self.toggle = toggle
    
    def update(self, target_window):
        if self.pressed: scale = self.scale * 0.95
        else: scale = self.scale

        self.scaled_image = pygame.transform.scale(self.image, (int(self.width * scale), int(self.height * scale)))

        if self.hover:
            self.scaled_image.fill((50, 50, 50, 100), special_flags=pygame.BLEND_ADD)

            if pygame.mouse.get_just_pressed()[0] == 1 and self.toggle == False: self.pressed = True
            elif pygame.mouse.get_just_pressed()[0] == 1 and self.toggle:
                if self.pressed: self.pressed = False
                else: self.pressed = True
            elif pygame.mouse.get_just_pressed()[0] == 0 and self.toggle == False:
                self.pressed = False

        else: pass

        self.rectangle = self.scaled_image.get_rect()
        self.rectangle.center = self.position

        pos = pygame.mouse.get_pos()
        if self.rectangle.collidepoint(pos): self.hover = True
        else: self.hover = False

        target_window.blit(self.scaled_image, (self.rectangle.x, self.rectangle.y))

    def set_new_image(self, image_name: str):
        self.image = pygame.image.load(image_name).convert_alpha()

class Rectangle():

    def __init__(self, position: tuple, width: int, height: int, color, border_radius: int = 0):
        
        self.position: tuple = position
        self.width: int = width
        self.height: int = height
        self.border_radius: int = border_radius

        if isinstance(color, tuple):
            self.color: tuple = color
        elif isinstance(color, str):
            self.color: tuple = get_color_from_string(color)
        else:
            print(f"Invalid instance: {color} -> {type(color)}; Expected String or Tuple")

    def draw(self, target_window):
        pygame.draw.rect(target_window, self.color, rect=pygame.Rect(self.position[0], self.position[1], self.width, self.height), border_radius=self.border_radius)


class Arrow():
    def __init__(self, start_position, end_position, size: int, color):

        if isinstance(start_position, tuple): self.start_position_vector = pygame.math.Vector2(start_position)
        elif isinstance(start_position, pygame.math.Vector2): self.start_position_vector = start_position
        else: print(f"Invalid instance: {start_position} -> {type(start_position)}; Expected Vector2 or Tuple")

        if isinstance(end_position, tuple): self.end_position_vector = pygame.math.Vector2(end_position)
        elif isinstance(end_position, pygame.math.Vector2): self.end_position_vector = end_position
        else: print(f"Invalid instance: {end_position} -> {type(end_position)}; Expected Vector2 or Tuple")

        self.size: int = size
        
        if isinstance(color, tuple): self.color: tuple = color
        elif isinstance(color, str): self.color: tuple = get_color_from_string(color)
        else: print(f"Invalid instance: {color} -> {type(color)}; Expected String or Tuple")

    def draw(self, target_window):

        relative_vector = self.start_position_vector - self.end_position_vector
        if relative_vector.length() == 0: return

        relative_unit = relative_vector.normalize()

        shortend_end_position_vector = self.end_position_vector - relative_unit * (self.size / 2)

        left_arrow_head = self.end_position_vector + relative_unit.rotate(30) * (self.size**1.4)
        right_arrow_head = self.end_position_vector + relative_unit.rotate(-30) * (self.size**1.4)

        pygame.draw.line(target_window, self.color, start_pos=self.start_position_vector, end_pos=shortend_end_position_vector, width=self.size)
        pygame.draw.polygon(target_window, self.color, [self.end_position_vector, left_arrow_head, right_arrow_head])

class RadialText():
    def __init__(self, text, cso: CircularSolarObject, angle: float, font: str, text_size: int, color):

        self.text = text
        self.cso: CircularSolarObject = cso
        self.angle: float = angle
        
        if not pygame.font.get_init(): pygame.font.init()
        self.font_name: str = font
        self.text_size: int = text_size
        
        if isinstance(color, tuple): self.color: tuple = color
        elif isinstance(color, str): self.color: tuple = get_color_from_string(color)
        else: print(f"Invalid instance: {color} -> {type(color)}; Expected String or Tuple")
        
    def draw(self, target_window, distance_factor):
        
        if self.cso.display_radius >= 20:
            radial_angle = math.radians(self.angle)
            cso_center = self.cso.drawing_position
            radius = self.cso.display_radius * distance_factor
            scaling_factor = self.cso.display_radius / 337
            if scaling_factor <= 0: scaling_factor = 0.1

            font_size = max(1, int(self.text_size * scaling_factor))
            if font_size >= 150: font_size = 150

            font = pygame.font.SysFont(self.font_name, font_size)

            x = cso_center[0] + radius * math.cos(radial_angle)
            y = cso_center[1] + radius * math.sin(radial_angle)

            text_render = font.render(self.text, True, self.color)
            rotated_text = pygame.transform.rotate(text_render, -self.angle)

            text_rect = rotated_text.get_rect()
            text_rect.center = (x, y)

            target_window.blit(rotated_text, text_rect)

class TrailParticles():
    def __init__(self, tracking_object: CircularSolarObject, max_particles: int = 200, thickness: int = 1):
        
        self.tracking_object: CircularSolarObject = tracking_object
        self.max_particles: int = max_particles
        self.thickness: int = thickness

        self.color = self.tracking_object.color

        self.positions: list = []
    
    def draw(self, target_window, zoom_factor, offset_vector, inflator):
        position: pygame.math.Vector2 = self.tracking_object.position_vector_in_pixels

        if len(self.positions) > self.max_particles: self.positions.pop(0)

        self.positions.append(position)

        if len(self.positions) < 2: return

        points = []
        for position in self.positions[:-1]:
            drawing_position = (position - offset_vector) * zoom_factor + (750, 500)
            if -1500 < drawing_position[0] < 2500 and -1500 < drawing_position[1] < 2500: points.append(drawing_position)
        
        calc_display_width = (numpy.sqrt(self.thickness)* 5) * zoom_factor * inflator
        display_width = max(1, min(int(calc_display_width), 50))
        if len(points) > 1: pygame.draw.lines(target_window ,self.color, False, points, display_width)