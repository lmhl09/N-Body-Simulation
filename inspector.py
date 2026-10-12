import colorsys
import math

import numpy as np
import pygame

from engine import Readback
from gpu_cso import (
    CircularSolarObject, TrailParticles, GRAVITATIONAL_CONSTANT, MAX_MASS, TYPE_STAR, TYPE_PLANET,
    PLANET_BUFFER, PLANET_STRIDE, F_VEL_X, F_VEL_Y, F_MASS, F_RADIUS,
    get_exact_position, calculate_orbit_shape, calculate_orbit_period,
)
from gpu_overlays import Overlays, HillSphere, LagrangePoints, Barycenter, HabitableZone
from shaders import INSPECT_SAMPLE
from interface import (
    get_fonts, Panel, TextEdit, set_wanted_cursor, draw_icon, draw_panel, draw_button, PLAIN_UNITS, format_number,
    format_grouped, parse_value_with_unit, EXAMPLES, DISTANCE_UNITS, get_distance_unit,
    format_distance, format_acceleration, format_speed, format_mass, format_radius, format_duration,
    TEXT, DIM, ACCENT, ERROR, BOX, BOX_BORDER, BOX_HOVER, TOGGLE_OFF, PLOT_BG, RIM,
    ICON_DELETE, ICON_CLOSE,
)

#about 30 times per second a shader copies everything about the selected body into a small buffer (a sample):
#the body, what it orbits and its four strongest attractors (struct InspectSampleBuffer in shaders.py)
SAMPLE_BINDING = 16
SAMPLE_BODIES = 6
SAMPLE_FLOATS = SAMPLE_BODIES * PLANET_STRIDE + 4 + 4 + 8
SAMPLE_INTERVAL = 1.0 / 30.0

#the tides of a body only count for the shown surface gravity if it pulls at least this hard (m/s²)
#(a bit less than the moon pulls on the earth)
TIDE_THRESHOLD = 2e-5

DISTANCE_COLOR = (120, 200, 255)
SPEED_COLOR = (255, 185, 95)
PLOT_LINE = (40, 48, 68)
UNUSABLE = (84, 92, 112)        #text of a toggle that cant be used right now

#the first part of the color slider is greys, the rest goes once through the rainbow
SLIDER_GREYS = 0.14


def slider_color(position):
    # Returns the color at a place of the color slider (0 = left end, 1 = right end)
    position = min(1.0, max(0.0, position))
    if position < SLIDER_GREYS:
        grey = int(round(255 * (0.24 + 0.74 * position / SLIDER_GREYS)))
        return (grey, grey, grey)

    hue = (position - SLIDER_GREYS) / (1.0 - SLIDER_GREYS) * 0.92      #from red to magenta
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return (int(round(255 * red)), int(round(255 * green)), int(round(255 * blue)))


def slider_position(color):
    # The other way around: looks for the place on the slider whose color is closest to the given one
    best_position = 0.0
    best_difference = None
    for k in range(241):
        position = k / 240.0
        sample = slider_color(position)
        difference = (sample[0] - color[0]) ** 2 + (sample[1] - color[1]) ** 2 + (sample[2] - color[2]) ** 2
        if best_difference is None or difference < best_difference:
            best_position = position
            best_difference = difference
    return best_position


class Refused(ValueError):
    # A typed value that could be read but is not allowed (its text says why)
    pass


#spectral classes with the temperature (K) they start at
STAR_CLASSES = (
    (30000.0, "Blauer Hauptreihenstern"),
    (10000.0, "Blauweißer Stern"),
    (7500.0, "Weißer Hauptreihenstern"),
    (6000.0, "Gelbweißer Zwerg"),
    (5200.0, "Gelber Zwerg"),
    (3700.0, "Oranger Zwerg"),
    (2400.0, "Roter Zwerg"),
)


def get_star_class(temperature):
    for lowest, name in STAR_CLASSES:
        if temperature >= lowest: return name
    return "Brauner Zwerg (keine Fusion)"


def get_circularity(eccentricity):
    # 100 % = a perfect circle, 0 % = the body escapes
    return max(0.0, 1.0 - eccentricity) * 100.0


def calculate_orbit_forecast(r_vec, v_vec, mu, times):
    # Where a body will be on its orbit: returns distance and speed for each time (in s from now)
    # only the body and its center count here (two-body problem), solved with Kepler's equation
    times = np.asarray(times, dtype=np.float64)
    r = float(np.hypot(*r_vec))
    v2 = float(np.dot(v_vec, v_vec))
    rv = float(np.dot(r_vec, v_vec))
    energy = v2 / 2.0 - mu / r

    if energy < 0:
        #bound: an ellipse (a = semi-major axis, n = mean motion, e = eccentricity)
        a = -mu / (2.0 * energy)
        n = math.sqrt(mu / a ** 3)
        e = math.hypot(1.0 - r / a, rv / math.sqrt(mu * a))
        if e < 1e-7:
            radius = np.full(len(times), a)
        else:
            anomaly = math.atan2(rv / math.sqrt(mu * a), 1.0 - r / a)
            mean = anomaly - e * math.sin(anomaly) + n * times
            #Kepler's equation E - e sin E = M cant be solved directly, Newton's method gets close in a few rounds
            eccentric = mean + e * np.sin(mean)
            for i in range(12):
                eccentric -= (eccentric - e * np.sin(eccentric) - mean) / (1.0 - e * np.cos(eccentric))
            radius = a * (1.0 - e * np.cos(eccentric))
        speed2 = mu * (2.0 / radius - 1.0 / a)
    else:
        #not bound: a hyperbola, the same with sinh and cosh
        a = mu / (2.0 * max(energy, 1e-12 * mu / r))
        n = math.sqrt(mu / a ** 3)
        e = math.sqrt(max((1.0 + r / a) ** 2 - rv * rv / (mu * a), 1.0 + 1e-12))
        anomaly = math.asinh(rv / math.sqrt(mu * a) / e)
        mean = e * math.sinh(anomaly) - anomaly + n * times
        hyperbolic = np.arcsinh(mean / e)
        for i in range(20):
            hyperbolic -= (e * np.sinh(hyperbolic) - hyperbolic - mean) / (e * np.cosh(hyperbolic) - 1.0)
        radius = a * (e * np.cosh(hyperbolic) - 1.0)
        speed2 = mu * (2.0 / radius + 1.0 / a)

    return radius, np.sqrt(np.maximum(speed2, 0.0))


class Inspector(Panel):
    # The window in the top right corner for the selected body
    # shows its values (the ones in a box can be typed over), two graphs of its orbit and switches for its overlays

    WIDTH = 332
    MARGIN = 12
    REFRESH = 1.0 / 30.0        #s between two redraws
    POINTS_PER_ORBIT = 300      #how many measured points the graphs keep per orbit
    AHEAD = 0.5                 #the graphs look this many orbits into the future

    #key, label, kind of unit (None = only shown, "color" = the color slider), body types that have it (None = all)
    FIELDS = (
        ("mass", "Masse", "mass", None),
        ("density", "Dichte", "density", (TYPE_PLANET,)),
        ("temperature", "Temperatur", None, (TYPE_STAR,)),
        ("class", "Klasse", None, (TYPE_STAR,)),
        ("radius", "Radius", None, None),
        ("gravity", "Schwerkraft", None, None),
        ("color", "Farbe", "color", (TYPE_PLANET,)),
        ("speed", "Geschw.", "speed", None),
        ("distance", "Abstand", "distance", None),
        ("orbit", "Umlauf", None, None),
        ("circularity", "Kreisförmigkeit", None, None),
    )

    #the unit kind of every field that is shown as text
    FIELD_KINDS = {}
    for field in FIELDS:
        if field[2] != "color": FIELD_KINDS[field[0]] = field[2]

    #these fields have a small button that switches their unit
    UNIT_TOGGLES = ("mass", "radius", "distance")

    TOGGLES = (
        ("follow", "Folgen"), ("trail", "Spur"), ("barycenter", "Schwerpunkt"),
        ("hill", "Hill-Sphäre"), ("lagrange", "Lagrange"),
    )
    STAR_TOGGLES = TOGGLES + (("habitable", "Habitable Zone"),)
    OVERLAYS = {"barycenter": Barycenter, "hill": HillSphere, "lagrange": LagrangePoints, "habitable": HabitableZone}

    def __init__(self, engine, on_select, on_follow, on_zoom_orbit, on_error):
        self.engine = engine
        #functions of main.py
        self.on_select = on_select
        self.on_follow = on_follow
        self.on_zoom_orbit = on_zoom_orbit
        self.on_error = on_error        #shows a text in the message line of the toolbar ("" takes it away)

        self.readback = Readback(engine, SAMPLE_FLOATS * 4, SAMPLE_BINDING)
        self.requests = 0               #every sample request gets a number
        self.last_request = -1.0
        engine.SetupNewComputeShader("INSPECT_SAMPLE", INSPECT_SAMPLE)

        self.fonts = get_fonts()
        self.font = self.fonts.body
        self.font_title = self.fonts.title
        self.font_small = self.fonts.small

        self.body = None
        self.primary = None             #the body it orbits
        self.partner = None             #a second heavy body in the middle of the orbit (the other sun of a double star)
        self.companion = False          #True if primary is only a partner of similar mass and not really orbited
        self.system = ()
        self.state = None               #everything calculated from the last sample (see take_sample)
        self.show_plots = True
        self.follow = True
        self.relative_units = {"mass": True, "radius": True}    #earth / sun units instead of kg and km
        self.chosen_distance_unit = None                        #None = the unit gets picked by the size of the orbit
        self.auto_distance_unit = "AE"

        self.rect = pygame.Rect(engine.width - self.WIDTH - self.MARGIN, self.MARGIN, self.WIDTH, 0)
        self.calculate_layout()

        self.sim_time = 0.0
        self.now = 0.0
        self.ignore_before = 0          #samples requested before this number are outdated
        self.history = []               #measured points for the graphs: (sim time, distance, speed)

        #for measuring one whole orbit: (sim time, angle turned so far, distance)
        self.track = []
        self.turned = 0.0
        self.last_angle = None
        self.last_track_time = 0.0

        self.focus = None               #key of the field that is being typed in
        self.edit = None
        self.error_key = None
        self.set_message("")

        self.color_drag = False
        self.color_at = (None, 0.0)     #the body color the slider knob was placed for, and its place
        self.color_strip = None

        self.tools = None               #gets set in main.py
        self.active_tool = None

        self.hover = None
        self.dirty = True
        self.last_draw = -1.0
        self.drawn = None

    ### SELECTION AND UPDATE ###

    @property
    def is_open(self):
        return self.body is not None

    @property
    def is_star(self):
        return self.body is not None and self.body.type == TYPE_STAR

    @property
    def solar_units(self):
        # Stars and black holes are measured in sun masses and sun radii, planets in those of the earth
        return self.body is not None and self.body.type != TYPE_PLANET

    @property
    def distance_unit(self):
        return self.chosen_distance_unit or self.auto_distance_unit

    @property
    def has_orbit(self):
        return self.state is not None and self.state["distance"] is not None

    @property
    def reference_name(self):
        # What distance and speed are measured against: the orbited body, or the center of mass if there are several
        if len(self.system) > 1: return "Schwerpunkt"
        return self.primary.name

    def set_message(self, text):
        self.message = text
        self.on_error(text)

    def open(self, body, follow=True):
        if body is self.body: return
        self.follow = follow
        self.end_edit()
        self.color_drag = False
        self.error_key = None
        self.set_message("")
        self.body = body
        self.primary = None
        self.partner = None
        self.companion = False
        self.system = ()
        self.state = None
        self.show_plots = True
        self.chosen_distance_unit = None
        self.calculate_layout()
        self.forget_measurements()

        #the window appeared under the mouse without the mouse moving
        self.hover = self.get_element_at(pygame.mouse.get_pos())
        self.update_cursor()

    def close(self):
        self.open(None)

    def stop_following(self):
        if self.follow:
            self.follow = False
            self.dirty = True

    def set_active_tool(self, tool):
        # The orbit and the velocity button stay lit while their tool waits for a click
        if tool != self.active_tool:
            self.active_tool = tool
            self.dirty = True

    def body_changed(self):
        # Something else changed the body (a tool, a merge), what was measured before doesnt fit anymore
        self.forget_measurements()

    def forget_measurements(self):
        self.history.clear()
        self.reset_track()
        self.ignore_before = self.requests
        self.dirty = True

    def reset_track(self):
        self.track.clear()
        self.turned = 0.0
        self.last_angle = None

    def frame(self, frame_sim_time, now):
        # Once per frame: takes a finished sample and asks for the next one
        self.sim_time += frame_sim_time

        sample = None
        finished = self.readback.get_finished()
        if finished is not None:
            buffer, note = finished
            sample = np.frombuffer(buffer.read(), dtype=np.float32)

        index = None
        if self.body is not None: index = self.body.index
        if index is not None and now - self.last_request >= SAMPLE_INTERVAL and self.readback.has_free_buffer():
            slot = self.readback.next_buffer()
            self.engine.RunComputeShader("INSPECT_SAMPLE", 0, {
                "u_target": index,
                "u_probe": Overlays.get_probe_slot(self.body),
                "u_count": self.engine.GetCurrentBufferCount(PLANET_BUFFER),
            }, groups=1)

            #the note says what the sample was requested for, it comes back a few frames later
            self.readback.mark_written(slot, (self.requests, self.sim_time, id(self.body), self.engine.layout_version))
            self.requests += 1
            self.last_request = now

        if sample is not None: self.take_sample(note, sample)

    def measure_orbit(self, sim_time, r_vec, distance, period_estimate):
        # Watches the body go around its center by adding up the angle it turns
        # after one full turn it returns the measured orbit time and eccentricity (None before that)
        angle = math.atan2(r_vec[1], r_vec[0])
        if self.last_angle is not None:
            step = (angle - self.last_angle + math.pi) % (2 * math.pi) - math.pi
            #if the samples are too far apart the turned angle cant be trusted, the measurement starts over
            too_sparse = period_estimate and sim_time - self.last_track_time > period_estimate / 8
            if abs(step) > 1.0 or too_sparse: self.reset_track()
            else: self.turned += step
        self.last_angle = angle
        self.last_track_time = sim_time

        track = self.track
        spacing = (period_estimate or 0.0) / 720.0
        if not track or (sim_time > track[-1][0] and sim_time - track[-1][0] >= spacing):
            track.append((sim_time, self.turned, distance))
            if len(track) > 4000: track.pop(0)

        #only the last full turn is kept
        full_turn = 2 * math.pi
        while len(track) > 2 and abs(self.turned - track[1][1]) >= full_turn:
            track.pop(0)
        if len(track) < 2 or abs(self.turned - track[0][1]) < full_turn: return None

        #the turn began somewhere between the first two points
        time_0, turned_0, distance_0 = track[0]
        time_1, turned_1, distance_1 = track[1]
        began = self.turned - math.copysign(full_turn, self.turned - turned_0)
        if turned_1 != turned_0: fraction = (began - turned_0) / (turned_1 - turned_0)
        else: fraction = 0.0
        period = sim_time - (time_0 + fraction * (time_1 - time_0))

        #eccentricity from the nearest and the farthest point of the turn
        distances = [point[2] for point in track[1:]] + [distance]
        nearest, farthest = min(distances), max(distances)
        if farthest + nearest > 0: return period, (farthest - nearest) / (farthest + nearest)
        return period, 0.0

    def take_sample(self, note, data):
        # Calculates everything the window shows from a sample of the gpu
        request, sim_time, body_id, layout_version = note
        if self.body is None or body_id != id(self.body) or request < self.ignore_before: return
        if layout_version != self.engine.layout_version: return

        #the sample: 6 body records, then 8 ints, then 8 floats
        records = data[:SAMPLE_BODIES * PLANET_STRIDE].astype(np.float64).reshape(SAMPLE_BODIES, PLANET_STRIDE)
        body, other = records[0], records[1]
        info_at = SAMPLE_BODIES * PLANET_STRIDE
        integers = data[info_at:info_at + 8].view(np.int32)
        top_attractors = [int(integers[4]), int(integers[5]), int(integers[6]), int(integers[7])]
        system_values = data[info_at + 8:info_at + 16].astype(np.float64)

        position = get_exact_position(body)
        velocity = np.array([body[F_VEL_X], body[F_VEL_Y]])

        state = {"absolute_speed": float(np.hypot(*velocity)), "distance": None,
                 "bound": False, "period": None, "eccentricity": None,
                 "tides": self.calculate_tides(position, body[F_RADIUS], top_attractors, records[2:])}

        self.update_orbited_bodies(int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3]))

        if self.primary is None: state["speed"] = state["absolute_speed"]
        else: self.calculate_orbit(state, sim_time, body, other, system_values)

        self.state = state
        self.now = sim_time

    def update_orbited_bodies(self, primary_index, heavy_bodies, partner_index, companion):
        # The gpu found out what the body orbits, here the python objects for it get looked up
        # heavy_bodies = how many other heavy bodies are inside the orbit, partner_index = one of them
        primary = partner = None
        if primary_index >= 0:
            primary = CircularSolarObject.by_id.get(self.engine.GetObjectIdAtIndex(primary_index, PLANET_BUFFER))
        if primary is not None and heavy_bodies == 1 and partner_index >= 0:
            partner = CircularSolarObject.by_id.get(self.engine.GetObjectIdAtIndex(partner_index, PLANET_BUFFER))

        #system = what is in the middle of the orbit, heaviest first
        #(two close suns swap places as the "primary" all the time, sorted by mass the window stays calm)
        system = []
        for member in (primary, partner):
            if member is not None: system.append(member)
        if len(system) == 2 and system[1].mass > system[0].mass: system.reverse()
        system = tuple(system)
        if heavy_bodies > 1: system += (None,)       #more than one other heavy body in there

        #it orbits something else now: the graphs and the orbit measurement start over
        if set(system) != set(self.system):
            self.history.clear()
            self.reset_track()
            self.dirty = True
        self.primary, self.partner, self.system = primary, partner, system
        self.companion = primary is not None and companion == 1

        #a body that orbits nothing has no graphs
        if (primary is not None) != self.show_plots:
            self.show_plots = primary is not None
            self.calculate_layout()
            self.dirty = True

    def calculate_orbit(self, state, sim_time, body, other, system_values):
        # Adds everything about the orbit to the state: distance, speed, shape and orbit time
        # body and other are the records of the body and of what it orbits
        # system_values = of everything inside the orbit: total mass (in masses of other), then the mass-weighted
        # position and velocity relative to other
        position = get_exact_position(body)
        velocity = np.array([body[F_VEL_X], body[F_VEL_Y]])

        #distance and speed are measured from the center of mass of everything inside the orbit
        #(for a planet around two suns that is the point both suns circle around)
        system_mass = max(system_values[0], 1.0)
        offset = np.array(system_values[1:3]) / system_mass
        system_velocity = system_values[4:6]
        state["reference_offset"] = (offset, system_velocity / system_mass)

        r_vec = position - get_exact_position(other) - offset
        v_vec = velocity - np.array([other[F_VEL_X], other[F_VEL_Y]]) - system_velocity / system_mass

        distance = float(np.hypot(*r_vec))
        speed = float(np.hypot(*v_vec))
        mu = GRAVITATIONAL_CONSTANT * (other[F_MASS] * system_mass + body[F_MASS])
        state.update(distance=distance, speed=speed, r_vec=r_vec, v_vec=v_vec, mu=mu)

        #the orbit it would have if only the two of them existed
        eccentricity, semi_major = calculate_orbit_shape(r_vec, v_vec, mu)
        state.update(eccentricity=eccentricity, bound=semi_major is not None)

        #window = how much time the graphs show behind "now" (one orbit)
        if semi_major is not None:
            state.update(period=calculate_orbit_period(semi_major, mu), apoapsis=semi_major * (1.0 + eccentricity))
            window = state["period"]
        else:
            #escaping: the time it needs to cover its current distance once
            state["apoapsis"] = distance
            if speed > 0: window = distance / speed
            else: window = 0.0

        #once a whole orbit was watched the measured values count instead of the calculated ones
        measured = self.measure_orbit(sim_time, r_vec, distance, state["period"])
        if measured is not None:
            state.update(measured_period=measured[0], measured_eccentricity=measured[1])
            window = measured[0]
        state["window"] = window

        #one more point for the graphs (not more than POINTS_PER_ORBIT per orbit, older than one orbit gets dropped)
        if self.history: last = self.history[-1][0]
        else: last = None
        if last is None or (sim_time > last and sim_time - last >= window / self.POINTS_PER_ORBIT):
            self.history.append((sim_time, distance, speed))
            if len(self.history) > 2000: self.history.pop(0)
        while self.history and self.history[0][0] < sim_time - 1.02 * window:
            self.history.pop(0)

        #the unit follows the size of the whole orbit, not the current distance (it would jump on eccentric orbits)
        unit = get_distance_unit(state["apoapsis"], self.auto_distance_unit)
        if unit != self.auto_distance_unit:
            self.auto_distance_unit = unit
            self.dirty = True

    def calculate_tides(self, position, radius, top, attractors):
        # Returns how much other bodies change the gravity on the surface (m/s²) and the names of these bodies
        # measured at the place where the body with the strongest tides stands directly overhead
        if radius <= 0: return 0.0, ()

        pulling = []        #(index, G M, vector to it, distance)
        for index, record in zip(top, attractors):
            if index < 0: continue
            offset = get_exact_position(record) - position
            distance = float(np.hypot(*offset))
            if distance <= radius: continue
            gm = GRAVITATIONAL_CONSTANT * record[F_MASS]
            if gm / (distance - radius) ** 2 < TIDE_THRESHOLD: continue
            pulling.append((index, gm, offset, distance))

        if not pulling: return 0.0, ()

        #tides grow with mass / distance³
        strongest = pulling[0]
        for entry in pulling:
            if entry[1] / entry[3] ** 3 > strongest[1] / strongest[3] ** 3: strongest = entry
        up = strongest[2] / strongest[3]
        point = up * radius

        #the tide of a body = its pull at that point of the surface minus its pull at the center
        tide = np.zeros(2)
        for index, gm, offset, distance in pulling:
            towards = offset - point
            tide += gm * (towards / np.hypot(*towards) ** 3 - offset / distance ** 3)

        own = GRAVITATIONAL_CONSTANT * self.body.mass / radius ** 2
        felt = float(np.hypot(*(-own * up + tide)))

        names = []
        for index, gm, offset, distance in pulling:
            member = CircularSolarObject.by_id.get(self.engine.GetObjectIdAtIndex(index, PLANET_BUFFER))
            if member is not None: names.append(member.name)
        return felt - own, tuple(names)

    ### LAYOUT ###

    def calculate_layout(self):
        # Places everything from top to bottom, the height of the window follows from what the body has
        x0 = self.rect.x + 14
        right = self.rect.right - 14
        y = self.rect.y + 12

        #head: color dot, name, delete and close
        self.boxes = {}
        self.swatch_center = (x0 + 11, y + 15)
        self.boxes["name"] = pygame.Rect(x0 + 30, y + 1, right - x0 - 30 - 62, 28)
        self.boxes["delete"] = pygame.Rect(right - 56, y + 3, 24, 24)
        self.boxes["close"] = pygame.Rect(right - 24, y + 3, 24, 24)
        y += 36

        self.subheader_y = y
        y += 24

        if self.body is not None: body_type = self.body.type
        else: body_type = TYPE_PLANET
        self.rows = []
        for key, label, kind, types in self.FIELDS:
            if types is None or body_type in types: self.rows.append((key, label, kind))

        if self.is_star: self.toggles = self.STAR_TOGGLES
        else: self.toggles = self.TOGGLES

        self.row_labels = {}
        for key, label, kind in self.rows:
            self.row_labels[key] = (x0, y + 2)
            if key in self.UNIT_TOGGLES:
                self.boxes[key] = pygame.Rect(x0 + 90, y, right - x0 - 90 - 44, 22)
                self.boxes["unit:" + key] = pygame.Rect(right - 40, y + 1, 40, 20)
            else:
                #"Kreisförmigkeit" is longer than the 90 px the other labels get
                column = max(90, self.font.size(label)[0] + 2)
                self.boxes[key] = pygame.Rect(x0 + column, y, right - x0 - column, 22)
            y += 26
        y += 6

        #the two graphs (title y, area)
        self.plots = []
        if self.show_plots:
            for key in ("distance", "speed"):
                self.plots.append((key, y, pygame.Rect(x0, y + 18, right - x0, 54)))
                y += 18 + 54 + 6
            self.plot_times_y = y - 3
            y += 20

        #toggles, three in a row (a last row with less gets wider ones)
        toggle_rows = (len(self.toggles) + 2) // 3
        for i in range(len(self.toggles)):
            key = self.toggles[i][0]
            row, column = i // 3, i % 3
            in_row = min(3, len(self.toggles) - row * 3)
            width = (right - x0 - 6 * (in_row - 1)) // in_row
            self.boxes["toggle:" + key] = pygame.Rect(x0 + column * (width + 6), y + row * 30, width, 24)
        y += toggle_rows * 30

        width = (right - x0 - 12) // 3
        x = x0
        for key in ("orbit_pick", "aim", "zoom"):
            self.boxes[key] = pygame.Rect(x, y, width, 26)
            x += width + 6
        y += 26

        self.rect.height = y - self.rect.y + 14

    ### INPUT ###

    def handle_event(self, event):
        # Returns True if the event was meant for the window
        if self.body is None: return False

        if event.type == pygame.MOUSEMOTION:
            if self.edit is not None and self.edit.dragging:
                if event.buttons[0]: self.edit.mouse_drag(event.pos[0], self.boxes[self.focus])
                else: self.edit.mouse_up()
                self.dirty = True
            if self.color_drag:
                if event.buttons[0]: self.slide_color(event.pos[0])
                else: self.color_drag = False

            hover = self.get_element_at(event.pos)
            if hover != self.hover:
                self.hover = hover
                self.dirty = True
            self.update_cursor()
            return False

        if event.type == pygame.MOUSEWHEEL:
            return self.rect.collidepoint(pygame.mouse.get_pos())

        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.rect.collidepoint(event.pos):
                self.end_edit()
                return False

            if event.button == 1:
                element = self.get_element_at(event.pos)
                if element is not None and element == self.focus:
                    #a click into the field that is being typed in moves the text cursor
                    shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    self.edit.mouse_down(event.pos[0], self.boxes[element], extend=shift)
                    self.dirty = True
                elif element == "color":
                    #the knob jumps to the click and follows the mouse until the button is let go
                    if self.focus is None or self.apply_edit():
                        self.color_drag = True
                        self.slide_color(event.pos[0])
                else:
                    self.click(element)
            return True

        if event.type == pygame.MOUSEBUTTONUP:
            if self.edit is not None and event.button == 1: self.edit.mouse_up()
            if self.color_drag and event.button == 1:
                self.color_drag = False
                self.dirty = True
                return True
            return self.rect.collidepoint(event.pos)

        if self.focus is not None:
            if event.type == pygame.TEXTINPUT:
                self.edit.insert(event.text)
                self.dirty = True
                return True
            if event.type == pygame.KEYDOWN:
                self.handle_edit_key(event)
                return True
            if event.type == pygame.KEYUP:
                return True

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            #tab starts typing in the first field (shift + tab in the last one)
            self.edit_next_field(None, backwards=bool(event.mod & pygame.KMOD_SHIFT))
            return True

        return False

    def get_element_at(self, position):
        # Returns the key of what is under the mouse ("mass", "unit:radius", "toggle:trail", "close" ...) or None
        # things that cant be used right now count as nothing
        if self.body is None or not self.rect.collidepoint(position): return None
        for key, rect in self.boxes.items():
            if not rect.collidepoint(position): continue
            if (key == "name" or key in self.FIELD_KINDS) and not self.can_edit(key): return None
            if key in ("zoom", "unit:distance") and not self.has_orbit: return None
            if key.startswith("toggle:") and not self.can_toggle(key[7:]): return None
            return key
        return None

    def can_toggle(self, key):
        # A barycenter needs something to orbit, hill sphere and lagrange points need something heavier to orbit
        if key == "barycenter": return self.primary is not None
        if key in ("hill", "lagrange"): return self.primary is not None and not self.companion
        return True

    def update_cursor(self):
        if self.hover is None: cursor = None
        elif self.hover == "name" or self.hover in self.FIELD_KINDS: cursor = pygame.SYSTEM_CURSOR_IBEAM
        else: cursor = pygame.SYSTEM_CURSOR_HAND
        set_wanted_cursor("inspector", cursor)

    def click(self, element):
        if element == "name" or element in self.FIELD_KINDS:
            self.start_edit(element)
            return

        #a click anywhere else in the window first tries to take over what was typed
        if self.focus is not None and not self.apply_edit(): return

        if element == "delete":
            self.body.delete()
        elif element == "close":
            self.on_select(None)
        elif element == "primary" and self.primary is not None:
            self.on_select(self.primary)
        elif element == "partner" and self.partner is not None:
            self.on_select(self.partner)
        elif element == "zoom":
            self.follow = False
            self.on_zoom_orbit(self.primary, self.state["apoapsis"])
        elif element in ("orbit_pick", "aim") and self.tools is not None:
            if element == "orbit_pick": mode = "orbit"
            else: mode = "velocity"
            #a second click on the lit button switches the tool off again
            if self.active_tool == mode: self.tools.cancel()
            elif mode == "orbit": self.tools.start_orbit(self.body)
            else: self.tools.start_velocity(self.body)
        elif element is not None and element.startswith("toggle:"):
            self.toggle(element[7:])
        elif element is not None and element.startswith("unit:"):
            key = element[5:]
            if key == "distance": self.chosen_distance_unit = self.get_next_unit(key)
            else: self.relative_units[key] = not self.relative_units[key]

        self.dirty = True

    def can_edit(self, key):
        if self.body is None: return False
        if key in ("name", "mass"): return True
        if key == "density": return self.body.has_own_density
        if key == "speed": return self.state is not None
        if key == "distance": return self.state is not None and self.state["distance"] is not None
        return False

    def get_editable_keys(self):
        keys = []
        if self.can_edit("name"): keys.append("name")
        for key, label, kind in self.rows:
            if self.can_edit(key): keys.append(key)
        return keys

    ### COLOR SLIDER (planets) ###

    def get_color_track(self):
        box = self.boxes["color"]
        return pygame.Rect(box.x + 8, box.centery - 5, box.width - 16, 10)

    def slide_color(self, x):
        track = self.get_color_track()
        position = min(1.0, max(0.0, (x - track.x) / track.width))
        color = slider_color(position)
        self.body.set_color(color)
        self.color_at = (color, position)
        self.dirty = True

    def draw_color_slider(self, surface):
        track = self.get_color_track()

        #the colored strip only gets painted once
        if self.color_strip is None or self.color_strip.get_size() != track.size:
            strip = pygame.Surface(track.size, pygame.SRCALPHA)
            for x in range(track.width):
                pygame.draw.line(strip, slider_color(x / max(track.width - 1, 1)), (x, 0), (x, track.height - 1))
            #round ends: everything outside of a rounded rect gets cut away
            ends = pygame.Surface(track.size, pygame.SRCALPHA)
            pygame.draw.rect(ends, (255, 255, 255, 255), ends.get_rect(), border_radius=track.height // 2)
            strip.blit(ends, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.color_strip = strip
        surface.blit(self.color_strip, track)

        #if the color was not set with the slider (scene colors) the knob goes to the closest color
        color = tuple(self.body.color)
        if self.color_at[0] != color: self.color_at = (color, slider_position(color))
        center = (track.x + self.color_at[1] * track.width, track.centery)

        if self.hover == "color" or self.color_drag:
            pygame.draw.circle(surface, RIM, center, 9)
            pygame.draw.circle(surface, color, center, 7)
            pygame.draw.circle(surface, (255, 255, 255), center, 8, 2)
        else:
            pygame.draw.circle(surface, RIM, center, 8)
            pygame.draw.circle(surface, color, center, 6)
            pygame.draw.circle(surface, TEXT, center, 7, 2)

    ### TYPING IN A FIELD ###

    def edit_next_field(self, current=None, backwards=False):
        keys = self.get_editable_keys()
        if not keys: return
        if current in keys:
            if backwards: step = -1
            else: step = 1
            self.start_edit(keys[(keys.index(current) + step) % len(keys)])
        elif backwards:
            self.start_edit(keys[-1])
        else:
            self.start_edit(keys[0])

    def start_edit(self, key):
        if key == self.focus or not self.can_edit(key): return
        if self.focus is not None and not self.apply_edit(): return

        #in the text field the unit is written out (Erdmassen instead of M⊕)
        text, unit = self.get_field_text(key)
        if key != "name": text = f"{text} {PLAIN_UNITS.get(unit, unit)}".strip()

        self.focus = key
        if key == "name": self.edit = TextEdit(text, self.font_title)
        else: self.edit = TextEdit(text, self.font)
        self.edit_original = text
        self.edit_unit = unit
        self.error_key = None
        self.set_message("")
        self.dirty = True

    def handle_edit_key(self, event):
        self.dirty = True

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.apply_edit()
        elif event.key == pygame.K_ESCAPE:
            self.cancel_edit()
        elif event.key == pygame.K_TAB:
            current = self.focus
            if self.apply_edit(): self.edit_next_field(current, backwards=bool(event.mod & pygame.KMOD_SHIFT))
        else:
            self.edit.handle_key(event)

    def close_edit(self):
        if self.edit is not None: self.edit.close()
        self.edit = None
        self.focus = None
        self.dirty = True

    def end_edit(self):
        # After a click outside of the window: a good value counts, a wrong one gets thrown away
        if self.focus is not None and not self.apply_edit(): self.cancel_edit()

    def cancel_edit(self):
        self.close_edit()
        self.error_key = None
        self.set_message("")

    def apply_edit(self):
        # Takes over what was typed, returns False if that was not possible (the field stays open and turns red)
        key = self.focus
        if key is None: return True

        kind = self.FIELD_KINDS.get(key)
        text = self.edit.text
        try:
            if text == self.edit_original:
                pass
            elif key == "name":
                if text.strip(): self.body.set_name(text.strip())
            else:
                self.set_body_value(key, parse_value_with_unit(text, kind, self.edit_unit))
        except ValueError as error:
            self.error_key = key
            if isinstance(error, Refused): self.set_message(str(error))
            else: self.set_message(f"Nicht lesbar. Beispiel: {EXAMPLES[kind]}")
            self.dirty = True
            return False

        self.close_edit()
        self.error_key = None
        self.set_message("")
        return True

    def set_body_value(self, key, value):
        body = self.body

        if key == "mass":
            if value <= 0: raise ValueError(value)
            if value > MAX_MASS: raise Refused("Zu schwer: höchstens 100 Millionen Sonnenmassen")
            body.set_mass(value)
        elif key == "density":
            if value <= 0 or not body.set_density(value): raise ValueError(value)
        elif key == "speed":
            if value < 0: raise ValueError(value)
            self.set_speed(value)
        elif key == "distance":
            if value <= 0: raise ValueError(value)
            self.set_distance(value)

        self.forget_measurements()

    #speed and distance of a body are relative to what it orbits
    #to change them the bodies get read back from the gpu first (the copies in python are old)

    def get_reference(self):
        # Returns position and velocity of what the body orbits
        primary, partner = self.primary, self.partner
        primary.sync_from_gpu()
        if partner is not None:
            #two heavy bodies: their common center of mass
            partner.sync_from_gpu()
            total = primary.mass + partner.mass
            return ((primary.position_vector_in_meters * primary.mass + partner.position_vector_in_meters * partner.mass) / total,
                    (primary.speed_vector * primary.mass + partner.speed_vector * partner.mass) / total)

        #else the primary plus the offset of the center of mass the last sample found
        offset, velocity = self.state.get("reference_offset", ((0.0, 0.0), (0.0, 0.0)))
        return (primary.position_vector_in_meters + pygame.math.Vector2(*offset),
                primary.speed_vector + pygame.math.Vector2(*velocity))

    def set_speed(self, speed):
        # Makes the body faster or slower, it keeps flying in the same direction
        body = self.body
        body.sync_from_gpu()

        if self.primary is None:
            if body.speed_vector.length() > 0: direction = body.speed_vector.normalize()
            else: direction = pygame.math.Vector2(1, 0)
            body.set_velocity(direction * speed)
            return

        center, center_velocity = self.get_reference()
        relative = body.speed_vector - center_velocity
        if relative.length() > 0:
            direction = relative.normalize()
        else:
            #a body at rest has no direction, it gets sent around its center
            offset = body.position_vector_in_meters - center
            if offset.length() > 0: direction = pygame.math.Vector2(-offset.y, offset.x).normalize()
            else: direction = pygame.math.Vector2(0, 1)

        body.set_velocity(center_velocity + direction * speed)

    def set_distance(self, distance):
        # Moves the body towards or away from what it orbits
        body = self.body
        if self.primary is None: raise ValueError(distance)

        body.sync_from_gpu()
        center, center_velocity = self.get_reference()

        offset = body.position_vector_in_meters - center
        old_distance = offset.length()
        if old_distance > 0: direction = offset / old_distance
        else: direction = pygame.math.Vector2(1, 0)
        relative = body.speed_vector - center_velocity

        #the orbit speed goes with 1 / sqrt(distance), with that a circular orbit stays circular at the new distance
        if old_distance > 0: speed_factor = math.sqrt(old_distance / distance)
        else: speed_factor = 1.0

        body.set_position(center + direction * distance)
        body.set_velocity(center_velocity + relative * speed_factor)

    ### TOGGLES ###

    def is_toggle_on(self, key):
        if key == "follow": return self.follow
        if key == "trail": return self.body.trail is not None
        return bool(Overlays.get_elements_of(self.body, self.OVERLAYS[key]))

    def toggle(self, key):
        body = self.body
        on = self.is_toggle_on(key)

        if key == "follow":
            self.follow = not on
            self.on_follow(self.follow)
        elif key == "trail":
            if on: body.trail.delete()
            else: TrailParticles(body)
        elif on:
            for element in Overlays.get_elements_of(body, self.OVERLAYS[key]): element.delete()
        else:
            self.OVERLAYS[key](body)

    ### DRAWING ###

    def render(self, surface, now):
        # Draws the window if something changed, returns the areas of the surface that have to go to the gpu
        if self.body is None:
            #closed: what was drawn before has to be wiped once
            if self.drawn is None: return []
            surface.fill((0, 0, 0, 0), self.drawn)
            area = self.drawn
            self.drawn = None
            return [area]

        if not self.dirty and now - self.last_draw < self.REFRESH: return []

        self.dirty = False
        self.last_draw = now

        #the window can have got smaller, so the old area gets cleared as well
        if self.drawn is None: area = self.rect
        else: area = self.rect.union(self.drawn)
        surface.fill((0, 0, 0, 0), area)
        self.draw_window(surface)
        self.drawn = self.rect.copy()
        return [area]

    def get_surface_gravity(self):
        # Returns g on the surface with the tides of other bodies added (None for a body without size)
        body = self.body
        if body.radius_in_meters <= 0: return None
        own = GRAVITATIONAL_CONSTANT * body.mass / body.radius_in_meters ** 2
        if self.state is not None: return own + self.state["tides"][0]
        return own

    def get_field_text(self, key):
        # Returns what a field shows as (text, unit)
        body, state = self.body, self.state
        if key == "name": return body.name, ""
        if key == "mass": return format_mass(body.mass, self.relative_units["mass"], self.solar_units)
        if key == "density": return format_number(body.density, ".5g"), "kg/m³"
        if key == "radius": return format_radius(body.radius_in_meters, self.relative_units["radius"], self.solar_units)
        if key == "temperature": return format_grouped(body.temperature), "K"
        if key == "class": return get_star_class(body.temperature), ""
        if key == "gravity":
            gravity = self.get_surface_gravity()
            if gravity is None: return "—", ""
            return f"g = {format_acceleration(gravity)}", "m/s²"

        #the rest needs a sample from the gpu
        if state is None: return "…", ""
        if key == "speed": return format_speed(state["speed"])
        if key == "distance":
            if state["distance"] is None: return "—", ""
            return format_distance(state["distance"], self.distance_unit)

        if key in ("orbit", "circularity"):
            if state["distance"] is None: return "—", ""

            #measured over a whole orbit if there was one already
            if state.get("measured_period"):
                if key == "orbit": return format_duration(state["measured_period"]), ""
                return format_number(get_circularity(state["measured_eccentricity"]), ".1f"), "%"

            #else calculated from position and speed, marked with ≈
            if not state["bound"]:
                if key == "orbit": return "entweicht", ""
                return "0", "%"
            if key == "orbit": return f"≈ {format_duration(state['period'])}", ""
            return f"≈ {format_number(get_circularity(state['eccentricity']), '.1f')}", "%"
        return "", ""

    def draw_window(self, surface):
        body = self.body
        draw_panel(surface, self.rect)

        pygame.draw.circle(surface, body.color, self.swatch_center, 11)
        pygame.draw.circle(surface, (205, 214, 228), self.swatch_center, 11, width=1)
        self.draw_field(surface, "name", self.font_title)

        if self.hover == "delete": draw_icon(surface, ICON_DELETE, self.boxes["delete"].center, ERROR)
        else: draw_icon(surface, ICON_DELETE, self.boxes["delete"].center, DIM)
        if self.hover == "close": draw_icon(surface, ICON_CLOSE, self.boxes["close"].center, TEXT)
        else: draw_icon(surface, ICON_CLOSE, self.boxes["close"].center, DIM)

        self.draw_relation_line(surface)

        for key, label, kind in self.rows:
            if key == "speed" and self.primary is not None: label = "Geschw. (rel.)"
            self.draw_text(surface, label, self.row_labels[key], DIM)
            if kind == "color": self.draw_color_slider(surface)
            else: self.draw_field(surface, key, self.font)

        #every unit button shows the unit a click switches to
        for key in self.UNIT_TOGGLES:
            rect = self.boxes.get("unit:" + key)
            if rect is None or (key == "distance" and not self.has_orbit): continue
            if self.hover == "unit:" + key:
                pygame.draw.rect(surface, BOX_HOVER, rect, border_radius=10)
                self.draw_text(surface, self.get_next_unit(key), rect.center, TEXT, self.font_small, anchor="center")
            else:
                pygame.draw.rect(surface, TOGGLE_OFF, rect, border_radius=10)
                self.draw_text(surface, self.get_next_unit(key), rect.center, DIM, self.font_small, anchor="center")

        self.draw_plots(surface)

        for key, label in self.toggles:
            rect = self.boxes["toggle:" + key]
            usable = self.can_toggle(key)
            on = usable and self.is_toggle_on(key)
            draw_button(surface, rect, on, self.hover == "toggle:" + key)
            if on: color = TEXT
            elif usable: color = DIM
            else: color = UNUSABLE
            self.draw_text(surface, label, rect.center, color, self.font_small, anchor="center")

        rect = self.boxes["orbit_pick"]
        draw_button(surface, rect, self.active_tool == "orbit", self.hover == "orbit_pick", radius=6)
        self.draw_text(surface, "Umkreisen…", rect.center, TEXT, self.font_small, anchor="center")

        rect = self.boxes["aim"]
        draw_button(surface, rect, self.active_tool == "velocity", self.hover == "aim", radius=6)
        self.draw_text(surface, "Geschwindigkeit…", rect.center, TEXT, self.font_small, anchor="center")

        rect = self.boxes["zoom"]
        draw_button(surface, rect, False, self.hover == "zoom", radius=6)
        if self.has_orbit: self.draw_text(surface, "Bahn zeigen", rect.center, TEXT, self.font_small, anchor="center")
        else: self.draw_text(surface, "Bahn zeigen", rect.center, DIM, self.font_small, anchor="center")

    def draw_relation_line(self, surface):
        # The line under the name, like "Planet · umkreist Erde"
        # the names in it are links: a click selects that body (their boxes get made here, where their place is known)
        x0 = self.rect.x + 14
        if self.primary is None: relation = " · kein Zentralkörper"
        elif self.companion: relation = " · Doppelsystem mit "
        else: relation = " · umkreist "

        surface.set_clip(pygame.Rect(x0, self.subheader_y, self.rect.right - 14 - x0, 22))
        rect = self.draw_text(surface, self.body.kind_name + relation, (x0, self.subheader_y), DIM)
        self.boxes.pop("primary", None)
        self.boxes.pop("partner", None)

        if self.primary is not None:
            for key, member in (("primary", self.primary), ("partner", self.partner)):
                if member is None: continue
                if key == "partner":
                    rect = self.draw_text(surface, " + ", (rect.right, self.subheader_y), DIM)
                link = rect = self.draw_text(surface, member.name, (rect.right, self.subheader_y), ACCENT)
                if self.hover == key:
                    pygame.draw.line(surface, ACCENT, (link.left, link.bottom - 2), (link.right, link.bottom - 2))
                self.boxes[key] = link.clip(surface.get_clip())
            if self.partner is None and len(self.system) > 1:
                self.draw_text(surface, " und weitere", (rect.right, self.subheader_y), DIM)
        surface.set_clip(None)

    def get_next_unit(self, key):
        # Returns the unit a click on the unit button of a field switches to
        if key == "distance":
            return DISTANCE_UNITS[(DISTANCE_UNITS.index(self.distance_unit) + 1) % len(DISTANCE_UNITS)]

        if self.relative_units[key]:
            if key == "mass": return "kg"
            return "km"
        if key == "mass": letter = "M"
        else: letter = "R"
        if self.solar_units: return letter + "☉"
        return letter + "⊕"

    def draw_field(self, surface, key, font):
        # Draws the value of a field, with a box around it if it can be typed over
        rect = self.boxes[key]
        text, unit = self.get_field_text(key)

        if not self.can_edit(key):
            surface.set_clip(rect)
            rendered = self.draw_text(surface, text, (rect.x + 6, rect.centery), TEXT, font, anchor="midleft")
            if unit: self.draw_text(surface, unit, (rendered.right + 5, rect.centery), DIM, font, anchor="midleft")
            surface.set_clip(None)
            return

        if key == self.error_key: border = ERROR
        elif key == self.focus: border = ACCENT
        elif key == self.hover: border = BOX_HOVER
        else: border = BOX_BORDER
        pygame.draw.rect(surface, BOX, rect, border_radius=5)
        pygame.draw.rect(surface, border, rect, width=1, border_radius=5)

        if key == self.focus:
            self.edit.draw(surface, rect)
            return

        surface.set_clip(rect.inflate(-6, 0))
        rendered = self.draw_text(surface, text, (rect.x + 6, rect.centery), TEXT, font, anchor="midleft")
        if unit: self.draw_text(surface, unit, (rendered.right + 5, rect.centery), DIM, font, anchor="midleft")
        surface.set_clip(None)

    def draw_plots(self, surface):
        # The two graphs: distance and speed over time
        # left of the "now" line what was measured during the last orbit (fading out)
        # right of it what Kepler's laws predict for the next half orbit (dashed)
        if not self.plots: return
        state = self.state
        ready = state is not None and state["distance"] is not None and state["window"] > 0

        for key, title_y, rect in self.plots:
            if key == "distance": color = DISTANCE_COLOR
            else: color = SPEED_COLOR
            pygame.draw.rect(surface, PLOT_BG, rect, border_radius=4)
            pygame.draw.line(surface, color, (rect.x, title_y + 8), (rect.x + 9, title_y + 8), 2)

            if ready:
                self.draw_plot_title(surface, key, title_y, rect)
                self.draw_plot_lines(surface, key, rect, color)
            else:
                if key == "distance": self.draw_text(surface, "Abstand", (rect.x + 14, title_y), DIM, self.font_small)
                else: self.draw_text(surface, "Geschwindigkeit", (rect.x + 14, title_y), DIM, self.font_small)
                self.draw_text(surface, "…", rect.center, DIM, self.font_small, anchor="center")

        #the times below the lower graph
        if ready:
            window = state["window"]
            ahead = self.AHEAD * window
            rect = self.plots[-1][2]
            y = self.plot_times_y
            now_x = rect.x + window / (window + ahead) * (rect.width - 1)
            self.draw_text(surface, f"vor {format_duration(window)}", (rect.x, y), DIM, self.font_small)
            self.draw_text(surface, "jetzt", (now_x, y), DIM, self.font_small, anchor="midtop")
            self.draw_text(surface, f"in {format_duration(ahead)}", (rect.right, y), DIM, self.font_small,
                           anchor="topright")

    def draw_plot_title(self, surface, key, title_y, rect):
        # What the graph shows, and at the right end the value right now
        if key == "distance":
            value_now = self.state["distance"]
            title = f"Abstand zu {self.reference_name}"
        else:
            value_now = self.state["speed"]
            title = f"Geschw. rel. zu {self.reference_name}"

        value, value_unit = self.format_plot_value(key, value_now)
        current = self.draw_text(surface, f"{value} {value_unit}", (rect.right, title_y), TEXT, self.font_small,
                                 anchor="topright")
        #a long name must not run into the value
        surface.set_clip(pygame.Rect(rect.x, title_y, current.left - 8 - rect.x, 18))
        self.draw_text(surface, title, (rect.x + 14, title_y), DIM, self.font_small)
        surface.set_clip(None)

    def get_plot_point(self, rect, t, value, low, high):
        # Where a value at the time t (in s from now) is inside a graph whose scale goes from low to high
        window = self.state["window"]
        span = window + self.AHEAD * window
        return (rect.x + (t + window) / span * (rect.width - 1),
                rect.bottom - 1 - (value - low) / (high - low) * (rect.height - 2))

    def draw_plot_lines(self, surface, key, rect, color):
        state = self.state
        window = state["window"]
        ahead = self.AHEAD * window
        span = window + ahead
        if key == "distance": column, value_now = 1, state["distance"]
        else: column, value_now = 2, state["speed"]

        #points as (time from now, value): the measured ones and the forecast
        measured = []
        for sample in self.history:
            if sample[0] >= self.now - window: measured.append((sample[0] - self.now, sample[column]))

        times = np.linspace(0.0, ahead, max(2, int(rect.width * ahead / span / 3)))
        radius, speed = calculate_orbit_forecast(state["r_vec"], state["v_vec"], state["mu"], times)
        if key == "distance": forecast = list(zip(times, radius))
        else: forecast = list(zip(times, speed))

        #the scale goes from the lowest to the highest value with a bit of room
        values = [v for t, v in measured] + [v for t, v in forecast if math.isfinite(v)]
        low, high = min(values), max(values)
        padding = max((high - low) * 0.14, abs(high) * 0.005, 1e-9)
        low, high = low - padding, high + padding

        now_x = self.get_plot_point(rect, 0.0, low, low, high)[0]
        pygame.draw.line(surface, PLOT_LINE, (now_x, rect.y + 1), (now_x, rect.bottom - 2))

        #the forecast: only every second piece gets drawn (dashed), in a darker color
        dim = tuple(int(0.55 * c + 0.45 * b) for c, b in zip(color, PLOT_BG))
        points = [self.get_plot_point(rect, t, v, low, high) for t, v in forecast if math.isfinite(v)]
        for i in range(0, len(points) - 1, 2):
            pygame.draw.line(surface, dim, points[i], points[i + 1], 2)

        #the measured line gets fainter the older it is
        if len(measured) > 1:
            points = [self.get_plot_point(rect, t, v, low, high) for t, v in measured]
            for i in range(len(points) - 1):
                age = -measured[i + 1][0] / window
                strength = 0.25 + 0.75 * (1.0 - min(1.0, max(0.0, age))) ** 1.5
                shade = tuple(int(strength * c + (1.0 - strength) * b) for c, b in zip(color, PLOT_BG))
                pygame.draw.line(surface, shade, points[i], points[i + 1], 2)

        #a dot where the body is right now
        here = self.get_plot_point(rect, 0.0, value_now, low, high)
        pygame.draw.circle(surface, PLOT_BG, here, 6)
        pygame.draw.circle(surface, color, here, 4)

        #highest and lowest value of the scale
        top, top_unit = self.format_plot_value(key, high - padding)
        bottom, bottom_unit = self.format_plot_value(key, low + padding)
        for label, anchor, at in ((f"{top} {top_unit}", "topleft", (rect.x + 4, rect.y + 2)),
                                  (f"{bottom} {bottom_unit}", "bottomleft", (rect.x + 4, rect.bottom - 2))):
            rendered = self.fonts.render(label, DIM, self.font_small)
            place = rendered.get_rect()
            setattr(place, anchor, at)
            pygame.draw.rect(surface, PLOT_BG, place.inflate(6, 0), border_radius=3)
            surface.blit(rendered, place)

    def format_plot_value(self, key, value):
        if key == "distance": return format_distance(value, self.distance_unit)
        return format_speed(value)
