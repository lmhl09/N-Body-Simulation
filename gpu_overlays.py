import math

import numpy as np
import pygame
import moderngl

from gpu_cso import CircularSolarObject, PLANET_BUFFER, GRAVITATIONAL_CONSTANT, color_for_shader
from shaders import (
    INSPECT_CALCULATION, ARROW_LAYOUT, MARKER_LAYOUT,
    ZONE_VERTEX_SHADER, ZONE_FRAGMENT_SHADER,
    ARROW_VERTEX_SHADER, ARROW_FRAGMENT_SHADER,
    LABEL_VERTEX_SHADER, LABEL_FRAGMENT_SHADER,
    MARKER_VERTEX_SHADER, MARKER_FRAGMENT_SHADER, MARKER_LABEL_VERTEX_SHADER,
    ROCHE_RING_VERTEX_SHADER, ROCHE_RING_FRAGMENT_SHADER,
    LABEL_CHARS, GLYPHS, TOP_ATTRACTORS, PROBE_INTS,
)

#hill spheres, habitable zones, barycenters, lagrange points, force arrows and roche limits
#the shaders calculate all of them from the body data that is on the gpu anyway
#from here they only get the list of which body has which overlay

PROBE_BUFFER = "overlay_probes"
ZONE_BUFFER = "overlay_zones"
ARROW_BUFFER = "overlay_arrows"
ARROW_LAYOUT_BUFFER = "overlay_arrow_layout"
MARKER_BUFFER = "overlay_markers"
MARKER_LAYOUT_BUFFER = "overlay_marker_layout"

ZONE_HILL, ZONE_HABITABLE = 0, 1
MARKER_BARYCENTER = 0       #1 to 5 are the lagrange points L1 to L5

#one zone or marker in its buffer (struct Zone and struct Marker in shaders.py)
OVERLAY_DTYPE = np.dtype([("info", "<i4", (4,)), ("color", "<f4", (4,))])


class Overlay():

    rgba = (1.0, 1.0, 1.0, 1.0)

    def __init__(self, object: CircularSolarObject):
        self.object = object
        Overlays.add(self)

    def delete(self):
        Overlays.remove(self)


class HillSphere(Overlay):
    # Inside this sphere the body can hold satellites against the pull of what it orbits
    rgba = color_for_shader((172, 229, 238), 0.22)


class HabitableZone(Overlay):
    # Where water could be liquid around a star
    rgba = color_for_shader((0, 200, 100), 0.3)


class Barycenter(Overlay):
    # Center of mass of the body and what it orbits (drawn with an "S" for Schwerpunkt)
    rgba = color_for_shader((235, 120, 110), 0.9)


class LagrangePoints(Overlay):
    # L1 to L5 of the body and what it orbits
    rgba = color_for_shader((150, 200, 235), 0.75)


def create_glyph_atlas(ctx, characters: str, font_px: int):
    # The label shaders cant draw text, so all characters they need are put next to each other into one texture
    if not pygame.font.get_init(): pygame.font.init()

    font = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", font_px)
    rendered = []
    for character in characters:
        rendered.append(font.render(character, True, (255, 255, 255)))

    padding = max(2, font_px // 8)
    cell_width = max(surface.get_width() for surface in rendered) + 2 * padding
    cell_height = max(font.get_height(), max(surface.get_height() for surface in rendered)) + 2 * padding

    atlas = pygame.Surface((cell_width * len(characters), cell_height), pygame.SRCALPHA)
    atlas.fill((0, 0, 0, 0))
    for k, surface in enumerate(rendered):
        atlas.blit(surface, (k * cell_width + (cell_width - surface.get_width()) // 2, padding))

    #only the alpha channel is needed (.T because pygame sorts its arrays by column)
    alpha = np.ascontiguousarray(pygame.surfarray.array_alpha(atlas).T)

    texture = ctx.texture((atlas.get_width(), cell_height), 1, alpha.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False

    advance = max(1, font.size("0")[0])

    return texture, cell_width, cell_height, advance


def set_uniform(program, name, value):
    # Not every program has every uniform (the glsl compiler removes the ones a shader doesnt use)
    uniform = program.get(name, None)
    if uniform is not None: uniform.value = value


def get_probe(body, body_index, probe_targets, probe_slots, max_probes):
    # A probe makes the gpu look up what a body orbits and what pulls on it the most
    # bodies with several overlays share one probe, returns its slot (-1 if all probes are taken)
    if id(body) in probe_slots: return probe_slots[id(body)]
    if len(probe_targets) >= max_probes: return -1

    slot = len(probe_targets)
    probe_slots[id(body)] = slot
    probe_targets.append(body_index)
    return slot


class Overlays():

    engine = None
    elements = []

    selected_object = None
    selection_arrows = 0            #how many force arrows the selected body shows

    probe_slots = {}                #id(body) -> probe slot
    probe_count = 0
    zone_count = 0
    arrow_count = 0
    marker_count = 0

    changed = True
    layout_version = -1
    warned = False

    @classmethod
    def setup(cls, engine, max_probes: int = 64, max_zones: int = 256, max_markers: int = 256, label_px: int = 11):
        cls.engine = engine
        cls.max_probes = max_probes
        cls.max_zones = max_zones
        cls.max_markers = max_markers

        ctx = engine.ctx

        engine.ReserveNewBuffer(PROBE_BUFFER, max_probes, PROBE_INTS, binding_point=5)
        engine.ReserveNewBuffer(ZONE_BUFFER, max_zones, 8, binding_point=8)
        engine.ReserveNewBuffer(ARROW_BUFFER, TOP_ATTRACTORS, 4, binding_point=9)
        engine.ReserveNewBuffer(ARROW_LAYOUT_BUFFER, TOP_ATTRACTORS, 16, binding_point=11)
        engine.ReserveNewBuffer(MARKER_BUFFER, max_markers, 8, binding_point=13)
        engine.ReserveNewBuffer(MARKER_LAYOUT_BUFFER, max_markers, 8, binding_point=15)

        engine.SetupNewComputeShader("INSPECT_CALCULATION", INSPECT_CALCULATION)
        engine.SetupNewComputeShader("MARKER_LAYOUT", MARKER_LAYOUT)
        cls.layout_shader = engine.SetupNewComputeShader("ARROW_LAYOUT", ARROW_LAYOUT)

        cls.zone_program = engine.SetupShaderProgram(ZONE_VERTEX_SHADER, ZONE_FRAGMENT_SHADER)
        cls.arrow_program = engine.SetupShaderProgram(ARROW_VERTEX_SHADER, ARROW_FRAGMENT_SHADER)
        cls.label_program = engine.SetupShaderProgram(LABEL_VERTEX_SHADER, LABEL_FRAGMENT_SHADER)
        cls.marker_program = engine.SetupShaderProgram(MARKER_VERTEX_SHADER, MARKER_FRAGMENT_SHADER)
        cls.marker_label_program = engine.SetupShaderProgram(MARKER_LABEL_VERTEX_SHADER, LABEL_FRAGMENT_SHADER)

        cls.roche_program = engine.SetupShaderProgram(ROCHE_RING_VERTEX_SHADER, ROCHE_RING_FRAGMENT_SHADER)
        set_uniform(cls.roche_program, "u_G", GRAVITATIONAL_CONSTANT)
        cls.vao_roche = ctx.vertex_array(cls.roche_program, [])

        cls.vao_zones = ctx.vertex_array(cls.zone_program, [])
        cls.vao_arrows = ctx.vertex_array(cls.arrow_program, [])
        cls.vao_labels = ctx.vertex_array(cls.label_program, [])
        cls.vao_markers = ctx.vertex_array(cls.marker_program, [])
        cls.vao_marker_labels = ctx.vertex_array(cls.marker_label_program, [])

        #the characters are rendered twice as big as they get shown, that looks smoother
        cls.glyph_texture, cell_width, cell_height, advance = create_glyph_atlas(ctx, GLYPHS, label_px * 2)
        screen_advance = max(1, round(advance * 0.5))
        scale = screen_advance / advance
        glyph_size = (cell_width * scale, cell_height * scale)

        for program in (cls.layout_shader, cls.arrow_program, cls.label_program, cls.marker_label_program):
            #the force between two bodies is far too big for a float, the shaders calculate with its logarithm
            set_uniform(program, "u_log10_G", math.log10(GRAVITATIONAL_CONSTANT))
            #shortest and longest arrow, width of the shaft, gap to the body (all in px)
            set_uniform(program, "u_arrow_style", (22.0, 64.0, 1.5, 4.0))
            set_uniform(program, "u_arrow_head", (7.0, 3.5))
            set_uniform(program, "u_glyph_size", glyph_size)
            set_uniform(program, "u_glyph_advance", float(screen_advance))

        for program in (cls.label_program, cls.marker_label_program):
            set_uniform(program, "u_glyphs", 1)
            set_uniform(program, "u_glyph_count", float(len(GLYPHS)))
            set_uniform(program, "u_shadow_offset",
                        (1.0 / glyph_size[0] / len(GLYPHS), 1.0 / glyph_size[1]))

        CircularSolarObject.delete_hooks.append(cls.remove_overlays_of)

    @classmethod
    def reset(cls):
        cls.elements.clear()
        cls.selected_object = None
        cls.probe_slots = {}
        cls.changed = True

    @classmethod
    def add(cls, element):
        cls.elements.append(element)
        cls.changed = True

    @classmethod
    def remove(cls, element):
        if element in cls.elements: cls.elements.remove(element)
        cls.changed = True

    @classmethod
    def set_selected_object(cls, object):
        if object is not cls.selected_object:
            cls.selected_object = object
            cls.changed = True

    @classmethod
    def set_selection_arrows(cls, count: int):
        count = max(0, min(int(count), TOP_ATTRACTORS))
        if count != cls.selection_arrows:
            cls.selection_arrows = count
            cls.changed = True

    @classmethod
    def get_elements_of(cls, body, kind):
        # Returns the overlays of one kind (HillSphere, Barycenter ...) the body has
        found = []
        for element in cls.elements:
            if isinstance(element, kind) and element.object is body: found.append(element)
        return found

    @classmethod
    def get_probe_slot(cls, body):
        return cls.probe_slots.get(id(body), -1)

    @classmethod
    def remove_overlays_of(cls, body):
        # Gets called when a body is deleted
        cls.elements[:] = [element for element in cls.elements if element.object is not body]
        if cls.selected_object is body: cls.selected_object = None
        cls.changed = True

    @classmethod
    def upload_overlay_list(cls):
        # Builds the lists of probes, zones, markers and arrows new and uploads them
        # only needed when an overlay or the selection changed or a body got another index
        engine = cls.engine
        if not cls.changed and engine.layout_version == cls.layout_version: return

        cls.changed = False
        cls.layout_version = engine.layout_version

        probe_targets = []
        probe_slots = {}

        #the selected body always has the first probe (the inspect window and the force arrows read it)
        arrows = []
        selected_index = None
        if cls.selected_object is not None:
            selected_index = engine.GetObjectIndex(id(cls.selected_object), PLANET_BUFFER)
        if selected_index is not None:
            slot = get_probe(cls.selected_object, selected_index, probe_targets, probe_slots, cls.max_probes)
            for rank in range(cls.selection_arrows):
                arrows.append([selected_index, slot, rank, 0])

        zone_infos, zone_colors = [], []
        marker_infos, marker_colors = [], []
        for element in cls.elements:
            body_index = engine.GetObjectIndex(id(element.object), PLANET_BUFFER)
            if body_index is None: continue

            #a habitable zone only depends on the star itself, it needs no probe
            if isinstance(element, HabitableZone):
                zone_infos.append((ZONE_HABITABLE, body_index, -1, -1))
                zone_colors.append(element.rgba)
                continue

            slot = get_probe(element.object, body_index, probe_targets, probe_slots, cls.max_probes)
            if slot < 0:
                cls.warn_once("probes")
            elif isinstance(element, HillSphere):
                zone_infos.append((ZONE_HILL, body_index, -1, slot))
                zone_colors.append(element.rgba)
            elif isinstance(element, Barycenter):
                marker_infos.append((MARKER_BARYCENTER, body_index, -1, slot))
                marker_colors.append(element.rgba)
            else:
                for point in range(1, 6):
                    marker_infos.append((point, body_index, -1, slot))
                    marker_colors.append(element.rgba)

        if len(zone_infos) > cls.max_zones:
            cls.warn_once("zones")
            zone_infos, zone_colors = zone_infos[:cls.max_zones], zone_colors[:cls.max_zones]

        if len(marker_infos) > cls.max_markers:
            cls.warn_once("markers")
            marker_infos, marker_colors = marker_infos[:cls.max_markers], marker_colors[:cls.max_markers]

        cls.probe_slots = probe_slots
        cls.upload_lists(probe_targets, zone_infos, zone_colors, arrows, marker_infos, marker_colors)

    @classmethod
    def upload_lists(cls, probe_targets, zone_infos, zone_colors, arrows, marker_infos, marker_colors):
        # Writes the finished lists into their buffers and remembers how long each one is (for the draw calls)
        engine = cls.engine
        cls.probe_count = len(probe_targets)
        cls.zone_count = len(zone_infos)
        cls.arrow_count = len(arrows)
        cls.marker_count = len(marker_infos)

        set_uniform(cls.layout_shader, "u_arrow_count", cls.arrow_count)
        set_uniform(cls.layout_shader, "u_marker_count", cls.marker_count)

        if probe_targets:
            #only the first int of a probe is the body, the gpu fills in the rest
            probes = np.full((len(probe_targets), PROBE_INTS), -1, dtype=np.int32)
            probes[:, 0] = probe_targets
            engine.WriteRawToBuffer(PROBE_BUFFER, probes.tobytes())

        if zone_infos:
            zone_data = np.zeros(len(zone_infos), dtype=OVERLAY_DTYPE)
            zone_data["info"] = zone_infos
            zone_data["color"] = zone_colors
            engine.WriteRawToBuffer(ZONE_BUFFER, zone_data.tobytes())

        if arrows:
            engine.WriteRawToBuffer(ARROW_BUFFER, np.asarray(arrows, dtype=np.int32).tobytes())

        if marker_infos:
            marker_data = np.zeros(len(marker_infos), dtype=OVERLAY_DTYPE)
            marker_data["info"] = marker_infos
            marker_data["color"] = marker_colors
            engine.WriteRawToBuffer(MARKER_BUFFER, marker_data.tobytes())

    @classmethod
    def warn_once(cls, what):
        if not cls.warned:
            print(f"Overlay capacity exceeded ({what}); the rest is not drawn")
            cls.warned = True

    @classmethod
    def run_shaders(cls):
        # First the probes (what orbits what), then the places of markers, arrows and their labels on screen
        if cls.probe_count > 0:
            cls.engine.RunComputeShader("INSPECT_CALCULATION", 0, {
                "u_count": cls.engine.GetCurrentBufferCount(PLANET_BUFFER),
            }, groups=cls.probe_count)

        if cls.marker_count > 0:
            cls.engine.RunComputeShader("MARKER_LAYOUT", 0, groups=cls.marker_count)

        if cls.arrow_count > 0:
            cls.engine.RunComputeShader("ARROW_LAYOUT", 0, groups=1)

    @classmethod
    def render_zones(cls):
        if cls.zone_count > 0:
            cls.vao_zones.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=cls.zone_count)

    @classmethod
    def render_roche_rings(cls, body_count: int):
        # The roche limit needs the tides the collision shader calculates, so it only shows while collisions are on
        if body_count > 0:
            cls.vao_roche.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=body_count)

    @classmethod
    def render_markers(cls):
        if cls.marker_count > 0:
            cls.vao_markers.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=cls.marker_count)

            #a marker label has up to two characters ("S", "L1")
            cls.glyph_texture.use(1)
            set_uniform(cls.marker_label_program, "u_mouse", tuple(map(float, pygame.mouse.get_pos())))
            cls.vao_marker_labels.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=cls.marker_count * 2)

    @classmethod
    def render_arrows(cls):
        if cls.arrow_count > 0:
            cls.vao_arrows.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=cls.arrow_count)

            cls.glyph_texture.use(1)
            set_uniform(cls.label_program, "u_mouse", tuple(map(float, pygame.mouse.get_pos())))
            cls.vao_labels.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=cls.arrow_count * LABEL_CHARS)
