import math

import numpy as np
import pygame

from gpu_cso import (
    CircularSolarObject, TrailParticles, ParticleCluster, ParticleRing, NEW_BODIES, create_new_body, create_body_ring,
    calculate_ring_orbits, GRAVITATIONAL_CONSTANT, PLANET_BUFFER, FLAG_BACKGROUND, TYPE_BLACKHOLE,
    F_VEL_X, F_VEL_Y, F_MASS, F_FLAGS, to_numpy, get_orbit_direction, get_tangent, calculate_circular_velocity,
)
from interface import get_fonts, set_wanted_cursor, format_grouped, format_distance, format_speed, TEXT, ACCENT

CLUSTER_PARTICLES = 2000
CLUSTER_RADIUS_PX = 30          #a new particle cluster is always this big on screen
PARTICLE_COLOR = (185, 190, 200)

RING_DRAG_PX = 12               #a right click that gets dragged further than this makes a ring
RING_PARTICLES_PER_PX = 0.35    #particles per pixel of ring area, so a ring looks equally dense at every zoom
RING_PARTICLES = (500, 200000)  #but never less or more than this
RING_COLOR = (150, 200, 235)

ARROW_PX = 90                   #length of the velocity arrow at the speed of a circular orbit
SNAP_PX = 9                     #the arrow tip snaps to "circular orbit" and to "no speed" within this distance


class Tools():
    # Everything the mouse does in the view besides selecting and moving the camera:
    # placing bodies and particles (right click) or rings of them (right drag),
    # sending a body onto an orbit, changing its velocity with an arrow, moving bodies while paused

    def __init__(self, engine, toolbar, inspector, select_body, get_free_view):
        self.engine = engine
        self.toolbar = toolbar
        self.inspector = inspector
        self.select_body = select_body
        self.get_free_view = get_free_view      #the part of the window no panel covers
        self.fonts = get_fonts()

        self.mode = None                #None, "orbit" or "velocity"
        self.body = None                #the body the mode is for
        self.arrow = None               #everything about the velocity arrow (see setup_velocity_arrow)
        self.drag_tip = None            #where the tip of the arrow is while it gets dragged
        self.was_paused = False

        self.moving = None              #body that is held with the right button
        self.grab_offset = np.zeros(2)

        self.press = None               #a held right click with a placing tool (see start_press)

        self.saved_camera = (None, None)
        self.drawn = None
        self.needs_redraw = False

    ### MODES ###

    def is_active(self):
        return self.mode is not None or self.toolbar.armed is not None or self.moving is not None

    def start_orbit(self, body):
        # The next click on another body sends this one onto an orbit around it
        self.toolbar.armed = None
        self.leave_mode()
        self.mode = "orbit"
        self.body = body
        self.update_hint()

    def start_velocity(self, body):
        # Shows the velocity of the body as an arrow that can be dragged (the sim stands still meanwhile)
        self.toolbar.armed = None
        self.leave_mode()
        self.was_paused = self.toolbar.paused
        self.toolbar.paused = True
        self.toolbar.dirty = True
        self.mode = "velocity"
        self.body = body
        self.setup_velocity_arrow()
        self.update_hint()

    def cancel(self):
        self.toolbar.armed = None
        self.leave_mode()
        self.update_hint()

    def leave_mode(self):
        #the velocity tool paused the sim, it goes on if it was running before
        if self.mode == "velocity" and self.toolbar.armed is None: self.toolbar.paused = self.was_paused
        self.mode = None
        self.body = None
        self.arrow = None
        self.drag_tip = None
        self.toolbar.dirty = True
        self.needs_redraw = True

    def update_hint(self):
        # Tells the toolbar what to write into the message line, and the inspect window which button to light up
        armed = self.toolbar.armed
        if armed is None: self.press = None

        if self.is_dragging_ring():
            hint = self.get_ring_text()
        elif armed:
            tool = self.toolbar.PLACING_TOOLS[armed]
            hint = f"Rechtsklick: {tool['placed']} · mit rechter Maustaste ziehen: Ring aus {tool['ring']}"
        elif self.mode == "orbit":
            hint = f"Körper anklicken, den {self.body.name} umkreisen soll"
        elif self.mode == "velocity":
            hint = "Ziehen legt die Geschwindigkeit fest"
            if self.arrow and self.arrow["reference"]: hint += f" (relativ zu {self.arrow['reference']})"
        else:
            hint = ""

        self.toolbar.set_tool_hint(hint)
        self.inspector.set_active_tool(self.mode)
        if not self.is_active(): set_wanted_cursor("tools", None)

    def update(self):
        # Once per frame: a mode ends by itself if a placing tool got picked, the sim runs again or its body is gone
        if self.toolbar.armed is not None and self.mode is not None: self.leave_mode()
        if self.mode == "velocity" and not self.toolbar.paused:
            self.was_paused = False
            self.leave_mode()
        for body in (self.body, self.moving):
            if body is not None and not body.alive:
                self.moving = None
                self.leave_mode()

        #the saved camera position is only good while nothing moves
        if not self.toolbar.paused: self.saved_camera = (None, None)
        self.update_hint()

    ### INPUT ###

    def handle_event(self, event, over_ui, now):
        # Returns True if the event was used by a tool
        if event.type == pygame.MOUSEMOTION: return self.mouse_moved(event, over_ui)
        if event.type == pygame.MOUSEBUTTONUP: return self.mouse_released(event, now)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.is_active():
            self.cancel()
            return True

        if event.type != pygame.MOUSEBUTTONDOWN or over_ui: return False

        if event.button == 3:
            if self.toolbar.armed is not None:
                self.start_press(self.toolbar.armed, event.pos)
                return True
            return self.start_moving(event.pos, now)

        if event.button != 1: return False

        if self.mode == "orbit":
            target = CircularSolarObject.get_body_at_screen(event.pos)
            if target is None or target is self.body:
                self.toolbar.notify("Dort ist kein anderer Körper", now)
            else:
                self.put_on_orbit(self.body, target, now)
            return True

        if self.mode == "velocity" and self.arrow is not None:
            self.drag_tip = event.pos
            self.needs_redraw = True
            return True

        return False

    def mouse_moved(self, event, over_ui):
        #a drag ends too if its button is not held anymore (it can be let go outside of the window)
        if self.moving is not None:
            if event.buttons[2]: self.move_body_to(event.pos)
            else: self.stop_moving()
            return True

        if self.press is not None:
            if event.buttons[2]: self.press["pointer"] = event.pos
            else: self.press = None
            self.needs_redraw = True
            return True

        if self.drag_tip is not None:
            if event.buttons[0]: self.drag_tip = event.pos
            else: self.drag_tip = None
            self.needs_redraw = True
            return True

        if self.is_active():
            if over_ui: set_wanted_cursor("tools", None)
            else: set_wanted_cursor("tools", pygame.SYSTEM_CURSOR_CROSSHAIR)
        return False

    def mouse_released(self, event, now):
        if event.button == 3 and self.moving is not None:
            self.stop_moving()
            return True

        if event.button == 3 and self.press is not None:
            self.press["pointer"] = event.pos
            if self.is_dragging_ring(): self.place_ring(now)
            else: self.place_single(now)
            return True

        if event.button == 1 and self.drag_tip is not None:
            self.apply_arrow_velocity(event.pos, now)
            return True

        return False

    ### PLACING ###

    def find_strongest_attractor(self, position):
        # Looks for the body that pulls the hardest at a position (m / r² is enough to compare)
        # a new body gets its orbit around that one, returns None in empty space
        records, positions = CircularSolarObject.read_all_records()
        if len(records) == 0: return None

        offsets = positions - np.asarray(position)
        distance_squared = offsets[:, 0] ** 2 + offsets[:, 1] ** 2

        #background bodies (ring stars) dont count, and nothing can orbit a body it sits exactly on
        background = (records[:, F_FLAGS].astype(int) & FLAG_BACKGROUND) != 0
        usable = (distance_squared > 0) & ~background
        pull = np.where(usable, records[:, F_MASS] / np.where(distance_squared > 0, distance_squared, 1.0), 0.0)

        best = int(np.argmax(pull))
        if pull[best] <= 0: return None

        return {
            "position": positions[best],
            "velocity": records[best, [F_VEL_X, F_VEL_Y]],
            "mu": GRAVITATIONAL_CONSTANT * records[best, F_MASS],
            "body": CircularSolarObject.by_id.get(self.engine.GetObjectIdAtIndex(best, PLANET_BUFFER)),
        }

    def start_press(self, kind, screen_position):
        # The right button went down with a placing tool, what gets placed is decided when it comes up again
        world = self.screen_to_world(screen_position)
        self.press = {"kind": kind, "start": screen_position, "pointer": screen_position, "world": world,
                      "attractor": self.find_strongest_attractor(world)}

    def place_single(self, now):
        press = self.press
        self.press = None
        kind, world, attractor = press["kind"], press["world"], press["attractor"]

        if kind == "particles":
            self.place_particle_cluster(world, attractor, now)
        else:
            #the new body starts on a circular orbit around what pulls on it the most
            velocity = np.zeros(2)
            if attractor is not None:
                mu = attractor["mu"] + GRAVITATIONAL_CONSTANT * NEW_BODIES[kind]["mass"]
                velocity = calculate_circular_velocity(world, attractor["position"], attractor["velocity"], mu)
            body = create_new_body(kind, (float(world[0]), float(world[1])), (float(velocity[0]), float(velocity[1])),
                                   self.toolbar.PLACING_TOOLS[kind]["name"])
            TrailParticles(body)
            self.select_body(body, follow=False)

        self.drop_tool()

    def drop_tool(self):
        self.toolbar.armed = None
        self.toolbar.dirty = True
        self.needs_redraw = True
        self.update_hint()

    def place_particle_cluster(self, position, attractor, now):
        # A round cloud of particles that stands still relative to the strongest body
        radius = CLUSTER_RADIUS_PX / self.engine.zoom
        count = CLUSTER_PARTICLES
        angles = np.random.uniform(0, 2 * np.pi, count)
        radii = radius * np.sqrt(np.random.uniform(0, 1, count))
        x = position[0] + radii * np.cos(angles)
        y = position[1] + radii * np.sin(angles)

        x_velocity, y_velocity = np.zeros(count), np.zeros(count)
        host = None
        if attractor is not None:
            if attractor["body"] is not None and ParticleCluster.can_attach():
                #attached to the body the particles are saved relative to it, that keeps their positions exact
                host = attractor["body"]
                x, y = x - attractor["position"][0], y - attractor["position"][1]
            else:
                x_velocity, y_velocity = x_velocity + attractor["velocity"][0], y_velocity + attractor["velocity"][1]

        try:
            ParticleCluster(x, y, x_velocity, y_velocity, color=PARTICLE_COLOR, opacity=0.9, host=host)
            self.toolbar.notify(f"{format_grouped(count)} Partikel platziert", now)
        except MemoryError:
            self.toolbar.notify("Kein Platz für weitere Partikel", now)

    ### RINGS ###

    def is_dragging_ring(self):
        if self.press is None: return False
        return math.dist(self.press["start"], self.press["pointer"]) > RING_DRAG_PX

    def get_ring(self):
        # Returns center, inner radius, outer radius, host and amount of the ring that is being dragged
        # the ring lies around the strongest body, press point and mouse give its two edges
        # in empty space it becomes a disc around the press point (host = None)
        press = self.press
        end = self.screen_to_world(press["pointer"])

        if press["attractor"] is None:
            center, host = press["world"], None
            inner = 0.0
            outer = float(np.hypot(*(end - center)))
        else:
            center, host = press["attractor"]["position"], press["attractor"]["body"]
            first = float(np.hypot(*(press["world"] - center)))
            second = float(np.hypot(*(end - center)))

            #the ring has to stay a bit above the surface (a black hole swallows everything within 3 radii)
            surface = 0.0
            if host is not None:
                surface = host.radius_in_meters
                if host.type == TYPE_BLACKHOLE: surface *= 3.0
            inner = max(min(first, second), 1.2 * surface)
            outer = max(max(first, second), inner)

        if press["kind"] == "particles":
            area = math.pi * (outer ** 2 - inner ** 2) * self.engine.zoom ** 2
            amount = int(min(max(RING_PARTICLES_PER_PX * area, RING_PARTICLES[0]), RING_PARTICLES[1]))
        else:
            amount = int(self.toolbar.settings.values["ring_bodies"])
        return center, inner, outer, host, amount

    def get_ring_text(self):
        center, inner, outer, host, amount = self.get_ring()
        unit = format_distance(outer)[1]
        what = f"{format_grouped(amount)} {self.toolbar.PLACING_TOOLS[self.press['kind']]['ring']}"
        if host is None: return f"Scheibe aus {what}, in Ruhe: Radius {format_distance(outer, unit)[0]} {unit}"
        return f"Ring aus {what} um {host.name}: {format_distance(inner, unit)[0]} bis {format_distance(outer, unit)[0]} {unit}"

    def place_ring(self, now):
        center, inner, outer, host, amount = self.get_ring()
        kind = self.press["kind"]
        self.press = None
        what = f"{format_grouped(amount)} {self.toolbar.PLACING_TOOLS[kind]['ring']}"
        if host is not None: shape = "Ring"
        else: shape = "Scheibe"

        if outer <= inner:
            self.toolbar.notify("Der Ring hat keine Breite", now)
        elif kind == "particles":
            try:
                if host is not None:
                    ParticleRing(amount, host, inner, outer, color=PARTICLE_COLOR, opacity=0.9)
                else:
                    x, y, x_velocity, y_velocity = calculate_ring_orbits(amount, inner, outer, 0.0)
                    ParticleCluster(x + center[0], y + center[1], x_velocity, y_velocity, color=PARTICLE_COLOR, opacity=0.9)
                self.toolbar.notify(f"{shape} aus {what} platziert", now)
            except MemoryError:
                self.toolbar.notify("Kein Platz für weitere Partikel", now)
        elif self.engine.GetCurrentBufferCount(PLANET_BUFFER) + amount > self.engine.GetBufferCapacity(PLANET_BUFFER):
            self.toolbar.notify(f"Kein Platz für {what}", now)
        else:
            create_body_ring(kind, amount, inner, outer, host=host, center=(float(center[0]), float(center[1])))
            self.toolbar.notify(f"{shape} aus {what} platziert", now)

        self.drop_tool()

    ### ORBIT ###

    def put_on_orbit(self, body, target, now):
        # Gives the body the velocity of a circular orbit around the target (its position stays)
        body.sync_from_gpu()
        target.sync_from_gpu()
        position = to_numpy(body.position_vector_in_meters)
        center = to_numpy(target.position_vector_in_meters)
        center_velocity = to_numpy(target.speed_vector)

        #it keeps going around in the direction it already does
        orbit_direction = get_orbit_direction(position - center, to_numpy(body.speed_vector) - center_velocity)

        mu = GRAVITATIONAL_CONSTANT * (target.mass + body.mass)
        velocity = calculate_circular_velocity(position, center, center_velocity, mu, orbit_direction)
        body.set_velocity((float(velocity[0]), float(velocity[1])))
        self.inspector.body_changed()
        self.toolbar.notify(f"{body.name} umkreist jetzt {target.name}", now)
        self.leave_mode()
        self.update_hint()

    ### VELOCITY ARROW ###

    def setup_velocity_arrow(self):
        # The arrow shows the velocity relative to what the body orbits (the inspect window knows what that is)
        # a body that orbits nothing gets its velocity shown as it is
        body, inspector = self.body, self.inspector
        body.sync_from_gpu()
        position = to_numpy(body.position_vector_in_meters)
        velocity = to_numpy(body.speed_vector)

        center = position
        reference_velocity = np.zeros(2)
        circular_speed = None
        reference_name = None
        if inspector.body is body and inspector.primary is not None and inspector.state is not None \
                and inspector.state.get("mu"):
            reference_position, reference_speed = inspector.get_reference()
            center, reference_velocity = to_numpy(reference_position), to_numpy(reference_speed)
            distance = float(np.hypot(*(position - center)))
            if distance > 0: circular_speed = math.sqrt(inspector.state["mu"] / distance)
            reference_name = inspector.reference_name

        relative = velocity - reference_velocity
        offset = position - center
        distance = float(np.hypot(*offset))
        #the direction a circular orbit would go in (the same way around as now)
        if distance > 0: tangent = get_tangent(offset / distance, get_orbit_direction(offset, relative))
        else: tangent = np.zeros(2)

        #how many m/s one arrow length stands for: the speed of a circular orbit if there is one,
        #but enough that the arrow of the current speed is not longer than 1.7 lengths
        speed = float(np.hypot(*relative))
        if circular_speed or speed > 0: speed_per_length = max(circular_speed or 0.0, speed / 1.7, 1.0)
        else: speed_per_length = 1000.0

        self.arrow = {
            "position": position,
            "reference_velocity": reference_velocity,
            "relative": relative,
            "circular": circular_speed,
            "tangent": tangent,
            "scale": ARROW_PX / speed_per_length,       #pixels per m/s
            "reference": reference_name,
        }
        self.needs_redraw = True

    def get_arrow_velocity(self, tip):
        # Returns the velocity an arrow tip at this screen position stands for, and what it snapped to (or None)
        arrow = self.arrow
        origin = self.world_to_screen(arrow["position"])
        drag = np.array([tip[0] - origin[0], tip[1] - origin[1]], dtype=np.float64)

        if arrow["circular"]:
            circular = arrow["tangent"] * arrow["circular"]
            if np.hypot(*(drag - circular * arrow["scale"])) < SNAP_PX: return circular, "circular"
        if np.hypot(*drag) < SNAP_PX * 0.7:
            return np.zeros(2), "rest"
        return drag / arrow["scale"], None

    def apply_arrow_velocity(self, tip, now):
        vector, snapped = self.get_arrow_velocity(tip)
        velocity = self.arrow["reference_velocity"] + vector
        self.body.set_velocity((float(velocity[0]), float(velocity[1])))
        self.arrow["relative"] = vector
        self.drag_tip = None
        self.needs_redraw = True
        self.inspector.body_changed()
        self.toolbar.notify(f"{self.body.name}: {self.get_speed_text(float(np.hypot(*vector)))}", now)

    ### MOVING BODIES ###

    def start_moving(self, screen_position, now):
        # A right click on a body grabs it (only while the sim is paused)
        body = CircularSolarObject.get_body_at_screen(screen_position)
        if body is None: return False
        if not self.toolbar.paused:
            self.toolbar.notify("Zum Verschieben erst pausieren (Leertaste)", now)
            return True

        #a camera that follows the body would run after it while it gets dragged
        if self.engine.anchor_object_id == id(body):
            self.engine.ReleaseAnchor()
            self.inspector.stop_following()
            self.saved_camera = (None, None)
        self.select_body(body, follow=False)

        body.sync_from_gpu()
        self.grab_offset = self.screen_to_world(screen_position) - to_numpy(body.position_vector_in_meters)
        self.moving = body
        #the trail starts new where the body gets dropped
        if body.trail is not None: body.trail.delete()
        set_wanted_cursor("tools", pygame.SYSTEM_CURSOR_SIZEALL)
        self.update_hint()
        return True

    def move_body_to(self, screen_position):
        position = self.screen_to_world(screen_position) - self.grab_offset
        self.moving.set_position((float(position[0]), float(position[1])))
        if self.arrow is not None and self.body is self.moving:
            self.arrow["position"] = position
            self.needs_redraw = True

    def stop_moving(self):
        body = self.moving
        self.moving = None
        if body.has_trail: TrailParticles(body)
        self.inspector.body_changed()
        if self.mode == "velocity" and self.body is body: self.setup_velocity_arrow()
        set_wanted_cursor("tools", None)
        self.update_hint()

    ### SCREEN AND WORLD ###

    def get_camera_center(self):
        # While the camera follows a body its position has to be read back from the gpu
        # during a pause the body doesnt move, so one read is enough until the sim runs again
        engine = self.engine
        if engine.anchor_object_id is None: return to_numpy(engine.camera_center)

        anchor, position = self.saved_camera
        if anchor != engine.anchor_object_id or not self.toolbar.paused:
            position = to_numpy(engine.CurrentCameraCenter())
            self.saved_camera = (engine.anchor_object_id, position)
        return position

    def world_to_screen(self, world):
        return tuple(self.engine.WorldToScreen(world, self.get_camera_center()))

    def screen_to_world(self, screen):
        return np.array(self.engine.ScreenToWorld(screen, self.get_camera_center()))

    ### DRAWING ###

    def render(self, surface):
        # Draws the velocity arrow or the ring that is being dragged, returns the areas that changed
        ring = self.is_dragging_ring()
        if not self.needs_redraw and self.arrow is None and not ring: return []
        self.needs_redraw = False

        regions = []
        if self.drawn is not None:
            surface.fill((0, 0, 0, 0), self.drawn)
            regions.append(self.drawn)
            self.drawn = None
        if self.arrow is None and not ring: return regions

        view = self.get_free_view()
        surface.set_clip(view)
        if ring: area = self.draw_ring(surface, view)
        else: area = self.draw_velocity_arrow(surface)
        surface.set_clip(None)

        self.drawn = area.clip(view)
        regions.append(self.drawn)
        return regions

    def draw_ring(self, surface, view):
        center, inner, outer, host, amount = self.get_ring()
        origin = self.world_to_screen(center)
        for radius in (inner, outer):
            self.draw_dashed_circle(surface, RING_COLOR, origin, radius * self.engine.zoom, view)
        pygame.draw.circle(surface, RING_COLOR, self.press["pointer"], 4, 1)

        #returns the area the ring covers (a ring far bigger than the window just counts as the whole view)
        reach = outer * self.engine.zoom + 6
        if reach > 1e6: return view.copy()
        return pygame.Rect(origin[0] - reach, origin[1] - reach, 2 * reach, 2 * reach)

    def get_speed_text(self, speed):
        # The speed with what it means for the orbit: from sqrt(2) times the circular speed on the body escapes
        text = " ".join(format_speed(speed))
        circular = None
        if self.arrow: circular = self.arrow["circular"]
        if circular:
            if abs(speed - circular) < 1e-6 * circular: text += " · Kreisbahn"
            elif speed >= circular * math.sqrt(2.0): text += " · entweicht"
            else: text += " · gebunden"
        return text

    def draw_velocity_arrow(self, surface):
        arrow = self.arrow
        fonts = self.fonts
        ox, oy = origin = self.world_to_screen(arrow["position"])
        scale = arrow["scale"]
        area = pygame.Rect(ox - 12, oy - 12, 24, 24)

        if self.drag_tip is not None:
            vector, snapped = self.get_arrow_velocity(self.drag_tip)
            #while dragging a faint circle shows how long the arrow of a circular orbit is, with a mark where it points
            if arrow["circular"]:
                radius = arrow["circular"] * scale
                self.draw_dashed_circle(surface, (80, 120, 160), origin, radius)
                mark = (ox + arrow["tangent"][0] * radius, oy + arrow["tangent"][1] * radius)
                pygame.draw.circle(surface, ACCENT, mark, 4, 1)
                area.union_ip(pygame.Rect(ox - radius - 2, oy - radius - 2, 2 * radius + 4, 2 * radius + 4))
        else:
            vector, snapped = arrow["relative"], None

        tip = (ox + vector[0] * scale, oy + vector[1] * scale)
        self.draw_arrow_line(surface, TEXT, origin, tip, 2)
        pygame.draw.circle(surface, ACCENT, tip, 5)
        pygame.draw.circle(surface, TEXT, tip, 5, 1)

        speed = float(np.hypot(*vector))
        if snapped == "rest": text = "in Ruhe"
        else: text = self.get_speed_text(speed)
        label = fonts.text(surface, text, (tip[0] + 10, tip[1] - 8), TEXT, fonts.small, anchor="bottomleft")
        area.union_ip(label.inflate(4, 4))
        area.union_ip(pygame.Rect(min(ox, tip[0]) - 12, min(oy, tip[1]) - 12,
                                  abs(tip[0] - ox) + 24, abs(tip[1] - oy) + 24))
        return area

    def draw_arrow_line(self, surface, color, start, end, width):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 2: return
        ux, uy = dx / length, dy / length

        #a line and a triangle as head (smaller on short arrows)
        head = min(10.0, length * 0.4)
        pygame.draw.line(surface, color, start, (end[0] - ux * head * 0.6, end[1] - uy * head * 0.6), width)
        pygame.draw.polygon(surface, color, [end, (end[0] - ux * head - uy * head * 0.5, end[1] - uy * head + ux * head * 0.5),
                                             (end[0] - ux * head + uy * head * 0.5, end[1] - uy * head - ux * head * 0.5)])

    def draw_dashed_circle(self, surface, color, center, radius, view=None):
        if radius < 3: return
        segments = max(12, int(radius / 4))
        first, last = 0, segments

        #a circle much bigger than the window would need thousands of dashes that nobody sees
        #with a view given only the piece of the circle that points towards the view gets drawn
        if view is not None and segments > 720:
            towards = math.atan2(view.centery - center[1], view.centerx - center[0])
            half = min(math.pi, 1.5 * math.hypot(view.width, view.height) / radius)
            first = int((towards - half) / (2 * math.pi) * segments)
            last = int((towards + half) / (2 * math.pi) * segments) + 1
            first -= first % 2

        #every second segment is a dash
        for k in range(first, last, 2):
            a0, a1 = 2 * math.pi * k / segments, 2 * math.pi * (k + 1) / segments
            pygame.draw.line(surface, color, (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0)),
                             (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1)))
