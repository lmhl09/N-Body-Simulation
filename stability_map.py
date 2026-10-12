import math
import time

import numpy as np
import pygame
import moderngl

from engine import Readback
from gpu_cso import (
    CircularSolarObject, TrailParticles, GRAVITATIONAL_CONSTANT, SUN_MASS, AU_IN_METERS, LIGHT_YEAR, PLANET_BUFFER,
    TYPE_PLANET, FLAG_BACKGROUND, F_VEL_X, F_VEL_Y, F_MASS, F_RADIUS, F_TYPE, F_FLAGS,
    to_numpy, get_orbit_direction, get_tangent, calculate_orbit_shape, calculate_orbit_period, create_new_body,
)
from shaders import (
    STABILITY_BODIES, STABILITY_MAP, PREVIEW_VERTEX_SHADER, PREVIEW_FRAGMENT_SHADER, MAP_BODIES, MAP_ROWS, MAP_STEPS,
)
from interface import (
    get_fonts, Panel, set_wanted_cursor, draw_panel, draw_button, format_number, format_grouped,
    format_distance, format_speed, format_duration, get_distance_unit,
    TEXT, DIM, BOX_BORDER, PLOT_BG, RIM,
)

#the stability map answers the question: if I put a new body next to this one, does it stay?
#every cell of the map is one possible start (a distance and a speed), its color says what happens to a body started there
#S always means the body the map belongs to

#buffer bindings of the two map shaders
SYSTEM_BINDING = 19
PATH_BINDING = 14
CELL_BINDING = 21
PARTICLE_BINDING = 23
LEFT_BINDING = 24

COLUMNS = 120               #start distances, spread logarithmically
ROWS = MAP_ROWS             #start speeds
SPEED_MAX = 1.5             #the top edge of the map, in circular speeds (a body escapes from sqrt 2 on)
ORBITS = 30                 #a start gets followed for this many orbits
STEPS_PER_ORBIT = 48
INNER = 0.3                 #a body this far inside a start distance (as ratio) gets counted to the center
LIGHT = 2e-3                #bodies with less than this share of the mass get left out while they are far outside
PAIR = 0.25                 #two bodies closer together than this (compared to their distance from S) count as one
EXTRA_STEPS = 6000          #steps a start may use on top for close encounters
PART = 30                   #so many columns get calculated at once (their body paths already need 22 MB of vram)
SPAWN_MASS = 1e-6           #a body placed from the map gets this share of the mass of S

REFRESH = 2.0               #s between two surveys while the sim runs

#green = round orbit, yellow = eccentric, orange = nearly straight down and up, red = crash, dark = escapes
STABLE = np.array([84, 192, 120], dtype=np.float64)
ECCENTRIC = np.array([230, 204, 82], dtype=np.float64)
RADIAL = np.array([232, 128, 58], dtype=np.float64)
IMPACT = np.array([150, 46, 58], dtype=np.float64)
ESCAPE = np.array([32, 38, 56], dtype=np.float64)
GUIDE = (236, 240, 248)

#what can happen to a start (first value of a cell)
ONGOING, IMPACTS, ESCAPES, HITS = 0, 1, 2, 3
UNFINISHED = -2             #in the last value of a cell: the steps ran out before the time was over

#marks on the speed axis
SPEED_TICKS = ((0.0, "0"), (0.5, "0,5"), (1.0, "1"), (math.sqrt(2.0), "1,41"))


def get_cell_colors(outcome, eccentricity):
    # Returns the rgb colors of cells (the color of an orbit goes from green over yellow to orange with its eccentricity)
    stability = 1.0 - np.clip(eccentricity, 0.0, 1.0)
    upper = np.clip((stability - 0.5) / 0.5, 0.0, 1.0)[..., None]
    lower = np.clip(stability / 0.5, 0.0, 1.0)[..., None]
    orbit = np.where(stability[..., None] >= 0.5, ECCENTRIC + (STABLE - ECCENTRIC) * upper,
                     RADIAL + (ECCENTRIC - RADIAL) * lower)
    rgb = np.where((outcome == ESCAPES)[..., None], ESCAPE, orbit)
    rgb = np.where(((outcome == IMPACTS) | (outcome == HITS))[..., None], IMPACT, rgb)
    return rgb


def merge_close_pairs(bodies):
    # Two bodies that circle each other closely and far away from S (a planet and its moon) get put together into one
    # followed one by one they would need far shorter steps, and from far away they pull like one body anyway
    # bodies = list of (values, name), values = x, y, vx, vy, mass, radius
    bodies = [(np.array(row, dtype=np.float64), name) for row, name in bodies]
    while True:
        pair = None
        for a in range(len(bodies)):
            for b in range(a + 1, len(bodies)):
                first, second = bodies[a][0], bodies[b][0]
                apart = math.hypot(*(first[0:2] - second[0:2]))
                if apart >= PAIR * min(math.hypot(*first[0:2]), math.hypot(*second[0:2])): continue
                #they also have to be bound to each other
                relative = first[2:4] - second[2:4]
                if apart > 0 and 0.5 * float(relative @ relative) >= (first[4] + second[4]) / apart: continue
                pair = (a, b)
        if pair is None: return bodies

        a, b = pair
        first, first_name = bodies[a]
        second, second_name = bodies[b]
        mass = first[4] + second[4]

        #the merged body sits in their center of mass and keeps name and radius of the heavier one
        if first[4] >= second[4]: heavier, name = first, first_name
        else: heavier, name = second, second_name
        merged = np.concatenate(((first[0:4] * first[4] + second[0:4] * second[4]) / mass, [mass, heavier[5]]))

        bodies = [body for k, body in enumerate(bodies) if k not in pair]
        bodies.append((merged, name))


def build_map_bodies(bodies):
    # Brings the bodies into the form the map shaders want (struct MapBody in shaders.py)
    # sorted from near to far, each one relative to the center of mass of S and everything nearer than itself
    # returns the records and the order the bodies were sorted into
    bodies = np.asarray(bodies, dtype=np.float64).reshape(-1, 6)
    order = np.argsort(np.hypot(bodies[:, 0], bodies[:, 1]))
    records = np.zeros((MAP_BODIES + 1, 16), dtype=np.float64)

    centre = np.zeros(4)            #center of mass of S and the bodies so far (position and velocity relative to S)
    inside = 1.0                    #their mass (S has mass 1)
    for k, j in enumerate(order):
        state, mass, radius = bodies[j, 0:4], bodies[j, 4], bodies[j, 5]
        own = state - centre
        mu = inside + mass
        distance = math.hypot(own[0], own[1])

        eccentricity, semi_major = calculate_orbit_shape(own[0:2], own[2:4], mu)
        if semi_major is not None:
            farthest = semi_major * (1.0 + eccentricity)
            nearest = semi_major * (1.0 - eccentricity)
            period = calculate_orbit_period(semi_major, mu)
        else:
            farthest = nearest = distance
            period = calculate_orbit_period(distance, mu)
        share = mass / mu

        #limit = starts further out than this count the body to the center
        #near = if a particle comes closer than this the body counts on its own again
        #low = further inside its orbit than this a light body gets left out
        limit = farthest / INNER * (1.0 + 4.0 * share)
        near = min(limit, farthest * max(1.5, math.sqrt(share / 1e-3)))
        if share <= LIGHT: low = nearest / 1.5
        else: low = 0.0

        records[k, 0:4] = own
        records[k, 4:8] = centre
        records[k, 8:12] = (mass, radius, limit, inside)
        records[k, 12:16] = (period, share, near, low)
        centre = centre + share * own
        inside = mu

    #one more record with the center and mass of everything together
    records[len(order), 4:8] = centre
    records[len(order), 11] = inside
    return records, order


class StabilityMap(Panel):
    # The window below the inspect window
    # x axis = how far from S a new body starts (logarithmic), y axis = how fast (1 = the speed of a circular orbit)
    # "Zweikörper-Schätzung" calculates the map from S alone, with formulas
    # "Gesamte Simulation" really starts a test particle for every cell on the gpu, with all bodies that matter
    # a click on the map places a small body at that start, the rings are the bodies that already orbit S

    GAP = 8                 #between the inspect window and this one
    PAD = 14
    HEADER = 34
    GUTTER = 30             #room left of the map for the speed marks
    MAP_HEIGHT = 160
    MIN_MAP_HEIGHT = 80
    BELOW = 42              #room below the map for the distance marks

    MODES = (("two_body", "Zweikörper-Schätzung"), ("system", "Gesamte Simulation"))

    def __init__(self, engine, inspector, toolbar, get_free_view):
        self.engine = engine
        self.inspector = inspector
        self.toolbar = toolbar
        self.get_free_view = get_free_view      #the part of the window no panel covers

        self.fonts = get_fonts()
        self.font = self.fonts.small

        engine.SetupNewComputeShader("STABILITY_BODIES", STABILITY_BODIES)
        engine.SetupNewComputeShader("STABILITY_MAP", STABILITY_MAP)
        context = engine.ctx
        self.system_buffer = context.buffer(reserve=(MAP_BODIES + 1) * 64)
        self.path_buffer = context.buffer(reserve=PART * (MAP_STEPS + 1) * MAP_BODIES * 16)
        self.cell_buffer = context.buffer(reserve=COLUMNS * ROWS * 16)
        self.particle_buffer = context.buffer(reserve=PART * ROWS * 48)
        #the map shader counts in here how many of its particles are not done yet
        self.left = Readback(engine, 4, LEFT_BINDING, slots=Readback.DELAY + 1)

        #draws the start under the mouse into the view
        self.preview = engine.SetupShaderProgram(PREVIEW_VERTEX_SHADER, PREVIEW_FRAGMENT_SHADER)
        self.vao_preview = context.vertex_array(self.preview, [])

        self.expanded = False
        self.mode = "two_body"
        self.body = None                #S
        self.auto_reach = None          #the outer edge of the map in m, picked to fit the system of S ...
        self.user_reach = None          #... or set with the mouse wheel
        self.survey = None              #what the last look at the system found (see survey_system)
        self.cells = {"two_body": None, "system": None}     #for both modes: (cells, the survey they belong to)
        self.images = {"two_body": None, "system": None}
        self.job = None                 #a map calculation that runs on the gpu right now (see start_job)
        self.step = 0.0                 #length of the last sub step of the simulation (s)
        self.last_survey = -1e9
        self.needs_survey = True

        self.hover = None
        self.pointer = (-1, -1)         #where the mouse was last
        self.dragging = None            #a ring that is held with the right button
        self.dirty = True
        self.last_draw = -1.0
        self.drawn = None
        self.layout = None

    ### PLACE ON SCREEN ###

    @property
    def visible(self):
        return self.inspector.is_open

    @property
    def rect(self):
        #the window hangs below the inspect window, closed only its header shows
        inspector = self.inspector.rect
        top = inspector.bottom + self.GAP
        if not self.expanded: return pygame.Rect(inspector.x, top, inspector.width, self.HEADER)
        return pygame.Rect(inspector.x, top, inspector.width, self.get_layout()["height"])

    def get_layout(self):
        # Returns where everything inside the open window is
        # the inspect window above changes its height, so this gets calculated new whenever the place changed
        inspector = self.inspector.rect
        top = inspector.bottom + self.GAP

        #if there is not enough room above the toolbar the map gets flatter
        room = self.toolbar.hint_rect.y - 6 - top
        fixed = self.HEADER + 32 + 18 + self.BELOW
        map_height = int(max(self.MIN_MAP_HEIGHT, min(self.MAP_HEIGHT, room - fixed)))

        key = (inspector.x, top, inspector.width, map_height)
        if self.layout is not None and self.layout["key"] == key: return self.layout

        x0 = inspector.x + self.PAD
        width = inspector.width - 2 * self.PAD
        y = top + self.HEADER
        half = (width - 6) // 2

        layout = {"key": key, "left": x0, "right": x0 + width}
        layout["modes"] = {"two_body": pygame.Rect(x0, y, half, 24), "system": pygame.Rect(x0 + half + 6, y, half, 24)}
        y += 32
        layout["speed_title_y"] = y
        y += 18
        layout["map"] = pygame.Rect(x0 + self.GUTTER, y, width - self.GUTTER, map_height)
        y += map_height
        layout["ticks_y"] = y + 4
        layout["distance_title_y"] = y + 20
        layout["height"] = y + self.BELOW - top
        layout["header"] = pygame.Rect(inspector.x, top, inspector.width, self.HEADER)

        self.layout = layout
        return layout

    ### CALCULATION ON THE GPU ###

    def frame(self, now, paused, step):
        # Once per frame: looks at the system again when needed and lets a running calculation go on
        if step > 0 and step != self.step:
            self.step = step
            #the "zu schnell simuliert" warning in the message line depends on the step
            if isinstance(self.hover, tuple): self.point_at(self.pointer)

        #another body got selected: everything starts from zero
        body = self.inspector.body
        if body is not self.body:
            if self.dragging is not None: self.drop_ring()
            self.body = body
            self.auto_reach = None
            self.user_reach = None
            self.survey = None
            self.cells = {"two_body": None, "system": None}
            self.images = {"two_body": None, "system": None}
            self.job = None
            self.needs_survey = True
            self.hover = None
            self.dirty = True

        if body is None or not self.expanded:
            self.job = None
            return

        if self.job is not None: self.continue_job()

        #while a ring is held the map must not change under the mouse
        if self.dragging is not None:
            if not self.dragging["body"].alive: self.drop_ring()
            return
        if self.inspector.state is None or self.job is not None: return

        if self.needs_survey or (not paused and now - self.last_survey >= REFRESH):
            survey = self.survey_system()
            self.last_survey = now
            self.needs_survey = False
            if survey is None: return

            self.survey = survey
            self.cells["two_body"] = (self.calculate_two_body_map(survey), survey)
            self.images["two_body"] = None
            if self.mode == "system": self.start_job(survey)
            self.dirty = True
            self.point_at(self.pointer)

    def get_body_at_index(self, index):
        return CircularSolarObject.by_id.get(self.engine.GetObjectIdAtIndex(int(index), PLANET_BUFFER))

    def get_name_at_index(self, index):
        body = self.get_body_at_index(index)
        if body is None: return "?"
        return body.name

    def survey_system(self):
        # Reads all bodies back from the gpu once and works out everything the map needs:
        # how far the map reaches, which bodies orbit S (the rings) and which bodies the calculation has to include
        # returns None if S does not exist anymore
        index = self.body.index
        if index is None: return None

        records, position = CircularSolarObject.read_all_records()
        if index >= len(records) or records[index, F_MASS] <= 0: return None
        velocity = records[:, [F_VEL_X, F_VEL_Y]]
        mass = records[:, F_MASS]
        #how close something can come to a body before it hits (a black hole swallows everything within 3 radii)
        reach_of = np.where(records[:, F_TYPE] > 1.5, 3.0 * records[:, F_RADIUS], records[:, F_RADIUS])

        #everything from here on is relative to S
        M = mass[index]
        mu = GRAVITATIONAL_CONSTANT * M
        surface = reach_of[index]
        offset = position - position[index]
        motion = velocity - velocity[index]
        distance = np.hypot(offset[:, 0], offset[:, 1])
        others = (np.arange(len(records)) != index) & (mass > 0) & (distance > 0)

        #the hill sphere of S and where S is on its own orbit come from the inspect window
        #new bodies start on the far side of S (seen from what S orbits) and go around the same way S does
        state = self.inspector.state
        hill = 0.0
        direction = np.array([1.0, 0.0])
        orbit_direction = 1.0
        if self.inspector.primary is not None and state.get("distance"):
            hill = state["distance"] * (mu / (3.0 * max(state["mu"] - mu, 1e-300))) ** (1.0 / 3.0)
            r_vec, v_vec = np.asarray(state["r_vec"]), np.asarray(state["v_vec"])
            if np.hypot(*r_vec) > 0: direction = r_vec / np.hypot(*r_vec)
            orbit_direction = get_orbit_direction(r_vec, v_vec)

        #satellites of S: lighter bodies that are bound to it (negative energy) and inside its hill sphere
        safe = np.where(distance > 0, distance, 1.0)
        kinetic = 0.5 * (motion[:, 0] ** 2 + motion[:, 1] ** 2)
        bound = kinetic - GRAVITATIONAL_CONSTANT * (M + mass) / safe < 0
        background = (records[:, F_FLAGS].astype(int) & FLAG_BACKGROUND) != 0
        satellites = others & ~background & (mass < M) & (distance > surface) & bound
        if hill > 0: satellites &= distance < hill

        #moons of a satellite dont count: whatever is inside the hill sphere of a heavier satellite drops out
        members = np.flatnonzero(satellites)
        hosts = members[np.argsort(-mass[members])][:64]
        if len(members) and len(hosts):
            hill_of_host = distance[hosts] * (mass[hosts] / (3.0 * M)) ** (1.0 / 3.0)
            difference = offset[members][:, None, :] - offset[hosts][None, :, :]
            gap = np.hypot(difference[:, :, 0], difference[:, :, 1])
            is_moon = ((mass[hosts][None, :] > mass[members][:, None]) & (gap < hill_of_host[None, :])).any(axis=1)
            satellites[members[is_moon]] = False

        #outer edge of the map: four times the farthest satellite or 100 radii of S, but hardly more than its hill sphere
        if self.user_reach is not None:
            reach = self.user_reach
        elif self.auto_reach is not None:
            reach = self.auto_reach
        else:
            if satellites.any(): farthest = distance[satellites].max()
            else: farthest = 0.0
            reach = max(4.0 * farthest, 100.0 * surface)
            if hill > 0 and reach > 0.7 * hill: reach = 1.15 * hill
            self.auto_reach = reach
        reach = max(reach, 3.2 * surface)
        inner = max(1.05 * surface, reach / 3000.0)
        column_width = math.log(reach / inner) / COLUMNS

        #the rings: satellites inside the map that go around the same way, the 16 heaviest
        speed = np.hypot(motion[:, 0], motion[:, 1]) / np.sqrt(mu / safe)       #in circular speeds
        spin = (offset[:, 0] * motion[:, 1] - offset[:, 1] * motion[:, 0]) * orbit_direction
        shown = np.flatnonzero(satellites & (distance >= inner) & (distance <= reach) & (spin > 0)
                               & (speed <= SPEED_MAX))
        markers = []
        for j in shown[np.argsort(-mass[shown])][:16]:
            markers.append((self.get_name_at_index(j), distance[j], speed[j], self.get_body_at_index(j)))

        #which bodies does the calculation need? every body gets a score, the 8 best above 1 are taken
        #outside of the map a body counts by its tides, inside by how wide it disturbs the orbits next to it (Wisdom 1980)
        ratio = mass / M
        outside = distance > 1.5 * reach
        tide = 2.0 * ratio * (reach / safe) ** 3 / 1e-3
        zone = np.where(ratio < 1.0, 1.3 * np.power(np.maximum(ratio, 0.0), 2.0 / 7.0), 1e9) / column_width
        score = np.where(others, np.where(outside, tide, zone), 0.0)
        chosen = []
        for j in np.argsort(-score)[:MAP_BODIES]:
            if score[j] > 1.0: chosen.append(int(j))

        #the shaders calculate in map units: the outer edge of the map is 1 and G M of S is 1
        length = reach
        duration = math.sqrt(length ** 3 / mu)
        bodies = []
        for j in chosen:
            values = np.concatenate((offset[j] / length, motion[j] * duration / length, [ratio[j], reach_of[j] / length]))
            bodies.append((values, self.get_name_at_index(j)))
        bodies = merge_close_pairs(bodies)
        system, order = build_map_bodies([entry[0] for entry in bodies])

        #a start further out than the hill sphere does not orbit S anymore (without a hill sphere: ten map widths)
        if hill > 0: escape = 1.2 * hill
        else: escape = 10.0 * reach

        return {
            "body": self.body, "mass": M, "mu": mu, "surface": surface, "reach": reach, "inner": inner,
            "hill": hill, "direction": direction, "orbit_direction": orbit_direction, "markers": markers,
            "system": system, "names": [bodies[j][1] for j in order], "escape": escape,
        }

    def calculate_two_body_map(self, survey):
        # The map if only S existed: a start with f times the circular speed gives an orbit with eccentricity |1 - f²|
        # from f² = 2 on the body escapes, and it crashes if the lowest point of its orbit is inside S
        r = self.get_column_distances(survey)[:, None]
        f2 = (SPEED_MAX * (np.arange(ROWS) + 0.5) / ROWS)[None, :] ** 2
        escape = f2 >= 2.0
        periapsis = np.where(f2 < 1.0, r * f2 / np.maximum(2.0 - f2, 1e-9), r)

        #a cell: outcome, when (share of the time), eccentricity, body that was hit
        cells = np.zeros((COLUMNS, ROWS, 4))
        cells[..., 0] = np.where(escape, ESCAPES, np.where(periapsis < survey["surface"], IMPACTS, ONGOING))
        cells[..., 1] = 1.0
        cells[..., 2] = np.abs(1.0 - f2)
        cells[..., 3] = -1.0
        return cells

    def get_column_distances(self, survey):
        # Returns the start distance of every column (in m)
        inner, reach = survey["inner"], survey["reach"]
        return inner * (reach / inner) ** ((np.arange(COLUMNS) + 0.5) / COLUMNS)

    def start_job(self, survey):
        # Starts the calculation of the "Gesamte Simulation" map
        # it is far too much for one frame, so continue_job does a part of it every frame
        self.system_buffer.write(survey["system"].astype(np.float32).tobytes())
        self.system_buffer.bind_to_storage_buffer(SYSTEM_BINDING)
        self.path_buffer.bind_to_storage_buffer(PATH_BINDING)
        self.cell_buffer.bind_to_storage_buffer(CELL_BINDING)
        self.particle_buffer.bind_to_storage_buffer(PARTICLE_BINDING)

        reach = survey["reach"]
        count = len(survey["names"])
        uniforms = {
            "u_body_count": count,
            "u_columns": COLUMNS,
            "u_r_min": survey["inner"] / reach,
            "u_orbits": float(ORBITS),
            "u_steps_per_orbit": float(STEPS_PER_ORBIT),
            "u_speed_max": SPEED_MAX,
            "u_direction": (float(survey["direction"][0]), float(survey["direction"][1])),
            "u_sense": survey["orbit_direction"],
            "u_impact": survey["surface"] / reach,
            "u_escape": survey["escape"] / reach,
            "u_extra_steps": EXTRA_STEPS,
            #steps per particle and frame, fewer with many bodies so the frame doesnt get too long
            "u_turns": max(80, 640 // (count + 1)),
        }
        self.engine.SetComputeUniforms("STABILITY_BODIES", uniforms)
        self.engine.SetComputeUniforms("STABILITY_MAP", uniforms)

        #first_column = where the part in work starts, paths_ready = the bodies of that part are calculated,
        #resume = 1 if the particles go on where they stopped, finished = frame in which the last part got done
        self.left.forget_waiting()
        self.job = {"survey": survey, "first_column": 0, "paths_ready": False, "resume": 0, "finished": None}

    def continue_job(self):
        job = self.job

        #all parts are done: the cells can be read once the gpu surely finished them
        if job["finished"] is not None:
            if self.engine.frame - job["finished"] >= Readback.DELAY: self.read_finished_map()
            return

        #no particle of this part is left: on to the next columns
        counter = self.left.get_finished()
        if counter is not None and np.frombuffer(counter[0].read(), dtype=np.uint32)[0] == 0:
            self.left.forget_waiting()
            job["first_column"] += PART
            job["paths_ready"] = False
            if job["first_column"] >= COLUMNS:
                job["finished"] = self.engine.frame
                return

        #a part starts with calculating where the bodies are at every step, after that the particles get moved along
        groups = min(PART, COLUMNS - job["first_column"])
        if not job["paths_ready"]:
            self.engine.RunComputeShader("STABILITY_BODIES", 0, {"u_first_column": job["first_column"]}, groups=groups)
            job["paths_ready"] = True
            job["resume"] = 0
            return

        slot = self.left.next_buffer(bytes(4))
        self.engine.RunComputeShader("STABILITY_MAP", 0, {"u_first_column": job["first_column"], "u_resume": job["resume"]},
                                     groups=groups)
        job["resume"] = 1
        self.left.mark_written(slot)

    def read_finished_map(self):
        survey = self.job["survey"]
        self.job = None
        #the selection changed in the meantime: the map is for a body nobody looks at anymore
        if survey["body"] is not self.body: return

        cells = np.frombuffer(self.cell_buffer.read(), dtype=np.float32).reshape(COLUMNS, ROWS, 4)
        self.cells["system"] = (cells.astype(np.float64), survey)
        self.images["system"] = None
        self.dirty = True
        self.point_at(self.pointer)

    def get_map_image(self, mode, size):
        # Returns the cells as a picture in the size of the map area (kept until there are new cells)
        entry = self.cells[mode]
        if entry is None: return None
        image = self.images[mode]
        if image is not None and image.get_size() == size: return image

        cells = entry[0]
        rgb = get_cell_colors(cells[..., 0], cells[..., 2])
        #row 0 is the slowest start and belongs to the bottom, in a picture y runs downwards
        small = pygame.surfarray.make_surface(np.ascontiguousarray(rgb[:, ::-1, :]).astype(np.uint8))
        self.images[mode] = pygame.transform.smoothscale(small, size)
        return self.images[mode]

    ### INPUT ###

    def get_element_at(self, position):
        # Returns what is under the mouse: "header", "mode:system", "window",
        # ("marker", number of the ring) or ("cell", start distance, start speed)
        if not self.visible or not self.rect.collidepoint(position): return None
        if not self.expanded: return "header"

        layout = self.get_layout()
        if layout["header"].collidepoint(position): return "header"
        for mode, rect in layout["modes"].items():
            if rect.collidepoint(position): return "mode:" + mode

        area = layout["map"]
        if area.collidepoint(position) and self.survey is not None:
            for i, marker in enumerate(self.survey["markers"]):
                x, y = self.get_map_point(marker[1], marker[2])
                if math.hypot(position[0] - x, position[1] - y) <= 6: return ("marker", i)
            return ("cell",) + self.get_start_at(position)
        return "window"

    def get_map_point(self, distance, speed):
        # Where a start (distance in m, speed in circular speeds) is on the map
        area = self.get_layout()["map"]
        inner, reach = self.survey["inner"], self.survey["reach"]
        x = area.x + area.width * math.log(distance / inner) / math.log(reach / inner)
        y = area.bottom - area.height * speed / SPEED_MAX
        return x, y

    def get_start_at(self, position):
        # The other way around: returns (distance, speed) of the start at a point of the map
        area = self.get_layout()["map"]
        inner, reach = self.survey["inner"], self.survey["reach"]
        u = min(1.0, max(0.0, (position[0] - area.x) / area.width))
        v = min(1.0, max(0.0, (area.bottom - position[1]) / area.height))
        return inner * (reach / inner) ** u, SPEED_MAX * v

    def handle_event(self, event):
        # Returns True if the event was meant for the window
        if not self.visible: return False

        if event.type == pygame.MOUSEMOTION:
            self.pointer = event.pos
            if self.dragging is not None:
                if event.buttons[2]: self.drag_ring_to(event.pos)
                else: self.drop_ring()
                return True
            self.point_at(event.pos)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.rect.collidepoint(event.pos): return False
            element = self.get_element_at(event.pos)
            if event.button == 3:
                #the right button grabs a ring
                if isinstance(element, tuple) and element[0] == "marker": self.grab_ring(element[1])
            elif event.button == 1:
                self.click(element)
                self.dirty = True
                self.point_at(event.pos)
            return True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3 and self.dragging is not None:
                self.drop_ring()
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEWHEEL:
            position = self.pointer
            if self.dragging is not None: return True
            if not self.rect.collidepoint(position): return False
            if self.expanded and self.survey is not None and self.get_layout()["map"].collidepoint(position):
                self.zoom(event.y)
            return True

        return False

    def click(self, element):
        if element == "header":
            self.expanded = not self.expanded
            self.needs_survey = True
        elif element in ("mode:two_body", "mode:system"):
            if self.mode != element[5:]:
                self.mode = element[5:]
                self.needs_survey = True
        elif isinstance(element, tuple) and element[0] == "cell":
            self.place_body(element[1], element[2])
        elif isinstance(element, tuple) and element[0] == "marker":
            #a click on a ring selects its body
            satellite = self.survey["markers"][element[1]][3]
            if satellite is not None and satellite.alive: self.inspector.on_select(satellite)

    def point_at(self, position):
        # Updates what the mouse points at: the highlight, the cursor and the text in the message line
        hover = self.get_element_at(position)
        if hover != self.hover:
            self.hover = hover
            self.dirty = True

        on_map = isinstance(hover, tuple)
        if self.dragging is not None: cursor = pygame.SYSTEM_CURSOR_SIZEALL
        elif on_map and hover[0] == "cell": cursor = pygame.SYSTEM_CURSOR_CROSSHAIR
        elif on_map or hover in ("header", "mode:two_body", "mode:system"): cursor = pygame.SYSTEM_CURSOR_HAND
        else: cursor = None
        set_wanted_cursor("stability_map", cursor)

        if on_map: self.toolbar.set_readout("map", self.get_readout(hover))
        else: self.toolbar.set_readout("map", "")

    def get_readout(self, hover):
        # The text for the message line: the start under the mouse and what happens to it, or the body of a ring
        survey = self.survey
        if hover[0] == "marker":
            marker = survey["markers"][hover[1]]
            name, distance, speed = marker[0], marker[1], marker[2]
        else:
            distance, speed = hover[1], hover[2]

        value, unit = format_distance(distance)
        circular_speed = math.sqrt(survey["mu"] / distance)
        speed_value, speed_unit = format_speed(speed * circular_speed)

        if hover[0] == "marker":
            text = f"{name}: {value} {unit} von {self.body.name} · {speed_value} {speed_unit}"
            #while its ring is dragged the text also says what the new orbit would be
            if self.dragging is None: return text
            return f"{text} · {self.get_verdict(distance, speed, alone=True)}{self.get_speed_warning(distance, speed)}"

        return (f"Start {value} {unit} von {self.body.name} mit {speed_value} {speed_unit} · "
                f"{self.get_verdict(distance, speed)}{self.get_speed_warning(distance, speed)}")

    ### MOVING A BODY BY ITS RING (right button) ###

    def grab_ring(self, index):
        satellite = self.survey["markers"][index][3]
        if satellite is None or not satellite.alive: return

        #the trail starts new where the body gets dropped
        had_trail = satellite.trail is not None
        if had_trail: satellite.trail.delete()
        self.dragging = {"index": index, "body": satellite, "trail": had_trail}
        self.dirty = True
        self.point_at(self.pointer)

    def drag_ring_to(self, position):
        # Puts the held body onto the start under the mouse: new distance from S and new speed, its direction from S stays
        survey, center, satellite = self.survey, self.body, self.dragging["body"]
        if center is None or not satellite.alive:
            self.drop_ring()
            return
        distance, speed = self.get_start_at(position)

        #the sim can be running, so both bodies get read back first
        center.sync_from_gpu()
        satellite.sync_from_gpu()
        origin = to_numpy(center.position_vector_in_meters)
        offset = to_numpy(satellite.position_vector_in_meters) - origin
        if np.hypot(*offset) > 0: bearing = offset / np.hypot(*offset)
        else: bearing = survey["direction"]
        velocity = to_numpy(center.speed_vector) \
            + get_tangent(bearing, survey["orbit_direction"]) * speed * math.sqrt(survey["mu"] / distance)

        start = origin + bearing * distance
        satellite.set_position((float(start[0]), float(start[1])))
        satellite.set_velocity((float(velocity[0]), float(velocity[1])))

        index = self.dragging["index"]
        survey["markers"][index] = (satellite.name, distance, speed, satellite)
        self.hover = ("marker", index)
        self.toolbar.set_readout("map", self.get_readout(self.hover))
        self.dirty = True

    def drop_ring(self):
        held = self.dragging
        self.dragging = None
        if held["trail"] and held["body"].alive: TrailParticles(held["body"])
        self.needs_survey = True
        self.point_at(self.pointer)
        self.dirty = True

    def zoom(self, steps):
        # The mouse wheel moves the outer edge of the map
        survey = self.survey
        surface = survey["surface"]
        reach = survey["reach"] * 1.35 ** (-steps)
        self.user_reach = min(max(reach, 3.2 * surface), 1e9 * surface)
        self.needs_survey = True

    def place_body(self, distance, speed):
        # A click on a cell creates a small body with exactly that start
        body = self.body
        if body is None or self.survey is None: return
        body.sync_from_gpu()
        position = to_numpy(body.position_vector_in_meters)
        velocity = to_numpy(body.speed_vector)

        #S moved on since the survey, so the start direction gets taken from where S is now
        direction, orbit_direction = self.survey["direction"], self.survey["orbit_direction"]
        if self.inspector.primary is not None and self.inspector.state is not None:
            center = self.inspector.get_reference()[0]
            outward = position - to_numpy(center)
            if np.hypot(*outward) > 0: direction = outward / np.hypot(*outward)

        start = position + direction * distance
        start_velocity = velocity + get_tangent(direction, orbit_direction) * speed \
            * math.sqrt(GRAVITATIONAL_CONSTANT * body.mass / distance)

        #heavy enough for fusion (0.08 sun masses) it becomes a star, around a planet a moon, else a planet
        mass = max(SPAWN_MASS * body.mass, 1.0)
        if mass >= 0.08 * SUN_MASS:
            new = create_new_body("star", tuple(start), tuple(start_velocity), "Neuer Stern", mass=mass)
        elif body.type == TYPE_PLANET:
            new = create_new_body("planet", tuple(start), tuple(start_velocity), "Neuer Mond", mass=mass, density=2500)
        else:
            new = create_new_body("planet", tuple(start), tuple(start_velocity), "Neuer Planet", mass=mass, density=2500)
        TrailParticles(new)
        self.needs_survey = True
        self.toolbar.notify(f"{new.name} platziert · {self.get_verdict(distance, speed)}", time.perf_counter())

    ### WHAT HAPPENS TO A START (as text) ###

    def get_speed_warning(self, distance, speed):
        # An orbit shorter than 30 sub steps of the simulation would not come out right at the current sim speed
        f2 = speed * speed
        if f2 >= 2.0 or self.step <= 0: return ""
        period = calculate_orbit_period(distance / (2.0 - f2), self.survey["mu"])
        if period < 30.0 * self.step: return " · zu schnell simuliert"
        return ""

    def get_cell_of(self, survey, distance, speed):
        # Returns column and row of the cell a start falls into
        column = int(math.log(distance / survey["inner"]) / math.log(survey["reach"] / survey["inner"]) * COLUMNS)
        column = min(COLUMNS - 1, max(0, column))
        row = min(ROWS - 1, max(0, int(speed / SPEED_MAX * ROWS)))
        return column, row

    def get_followed_time(self, survey, column, row):
        # Returns how long the map shader follows a cell, as (orbits, seconds)
        # this repeats what the shader does at the start of a particle (column_of and main of STABILITY_MAP)
        system = survey["system"]
        r0 = self.get_column_distances(survey)[column] / survey["reach"]
        inner = 0
        while inner < len(survey["names"]) and system[inner, 10] < r0: inner += 1
        mu, centre = system[inner, 11], system[inner, 4:8]

        speed = SPEED_MAX * (row + 0.5) / ROWS
        direction = survey["direction"]
        x = direction * r0 - centre[0:2]
        v = get_tangent(direction, survey["orbit_direction"]) * speed / math.sqrt(r0) - centre[2:4]
        #semi-major axis of the orbit divided by the start distance (a small orbit is over sooner)
        size = 1.0 / max(2.0 - math.hypot(*x) * float(v @ v) / mu, 1e-6)

        circle = calculate_orbit_period(r0, mu) * math.sqrt(survey["reach"] ** 3 / survey["mu"])
        return ORBITS * min(1.0, size ** -1.5), ORBITS * circle * min(1.0, size ** 1.5)

    def get_verdict(self, distance, speed, alone=False):
        # alone = only S and the new body count, even if the simulated map is shown
        survey = self.survey
        name = self.body.name

        if alone or self.mode == "two_body" or self.cells["system"] is None:
            f2 = speed * speed
            if f2 >= 2.0: return "entweicht"
            if f2 < 1.0: periapsis = distance * f2 / (2.0 - f2)
            else: periapsis = distance
            if periapsis < survey["surface"]: return f"stürzt auf {name}"
            period = calculate_orbit_period(distance / (2.0 - f2), survey["mu"])
            return f"Bahn mit e = {format_number(abs(1.0 - f2), '.2f')} · Umlaufzeit {format_duration(period)}"

        #the simulated map belongs to the survey it was started with, which can be older than the newest one
        cells, survey = self.cells["system"]
        column, row = self.get_cell_of(survey, distance, speed)
        outcome, when, eccentricity, hit = cells[column, row]
        orbits, seconds = self.get_followed_time(survey, column, row)

        if outcome == ONGOING:
            if hit == UNFINISHED: orbits *= when
            shape = f"e = {format_number(eccentricity, '.2f')}"
            if orbits < 1.0: return f"noch auf dem ersten Umlauf · {shape}"
            if round(orbits) == 1: return f"stabil über {orbits:.0f} Umlauf · {shape}"
            return f"stabil über {orbits:.0f} Umläufe · {shape}"

        after = format_duration(when * seconds)
        if outcome == IMPACTS: return f"stürzt nach {after} auf {name}"
        if outcome == ESCAPES:
            #it was outside from the start, or it was still on its way out when the time was over
            if distance >= survey["escape"]: return "entweicht: außerhalb der Hill-Sphäre"
            if when >= 1.0: return "entweicht"
            return f"entweicht nach {after}"

        if 0 <= int(hit) < len(survey["names"]): other = survey["names"][int(hit)]
        else: other = "einen anderen Körper"
        return f"prallt nach {after} auf {other}"

    def get_cell_at(self, distance, speed):
        # Returns outcome and eccentricity of the cell a start falls into (two-body map while the other one is not ready)
        mode = self.mode
        if self.cells[mode] is None: mode = "two_body"
        entry = self.cells[mode]
        if entry is None: return ONGOING, 0.0

        cells, survey = entry
        column, row = self.get_cell_of(survey, distance, speed)
        return cells[column, row, 0], cells[column, row, 2]

    ### DRAWING ###

    def render(self, surface, now):
        # Draws the window if something changed, returns the areas of the surface that have to go to the gpu
        if not self.visible:
            set_wanted_cursor("stability_map", None)
            self.toolbar.set_readout("map", "")
            if self.drawn is None: return []
            surface.fill((0, 0, 0, 0), self.drawn)
            area = self.drawn
            self.drawn = None
            return [area]

        #while the gpu calculates, "Wird berechnet …" has to show up even if nothing else changed
        waiting = self.expanded and self.mode == "system" and self.cells["system"] is None
        if not (self.dirty or (waiting and now - self.last_draw >= 0.25) or self.drawn != self.rect): return []

        self.dirty = False
        self.last_draw = now
        if self.drawn is None: area = self.rect
        else: area = self.rect.union(self.drawn)
        surface.fill((0, 0, 0, 0), area)
        self.draw_window(surface)
        self.drawn = self.rect.copy()
        return [area]

    def render_preview(self):
        # While the mouse is on a cell the view shows that start: the circle of the start distance around S,
        # the start point on it and a line for the direction (drawn by a shader, it knows where S is on screen)
        hover = self.hover
        if not (isinstance(hover, tuple) and hover[0] == "cell") or self.survey is None: return
        if self.body is None: return
        index = self.body.index
        if index is None: return

        distance, speed = hover[1], hover[2]
        survey = self.survey
        outcome, eccentricity = self.get_cell_at(distance, speed)
        color = tuple(get_cell_colors(np.array(outcome), np.array(eccentricity)) / 255.0)
        if outcome == ESCAPES: color = (0.59, 0.63, 0.71)       #the dark blue of the map cant be seen on black

        program = self.preview
        program["u_body"].value = index
        program["u_distance"].value = distance
        program["u_direction"].value = tuple(survey["direction"])
        program["u_tangent"].value = tuple(get_tangent(survey["direction"], survey["orbit_direction"]))
        program["u_length"].value = 6 + 26 * min(speed, SPEED_MAX) / SPEED_MAX
        program["u_color"].value = color

        #only inside the free part of the view (opengl counts y from the bottom)
        view = self.get_free_view()
        context = self.engine.ctx
        context.scissor = (view.x, self.engine.height - view.bottom, view.width, view.height)
        self.vao_preview.render(moderngl.TRIANGLE_STRIP, vertices=4)
        context.scissor = None

    def draw_window(self, surface):
        rect = self.rect
        draw_panel(surface, rect)

        #header with a small triangle: pointing down = open, pointing right = closed
        x0 = rect.x + self.PAD
        if self.hover == "header": color = TEXT
        else: color = DIM
        self.draw_text(surface, "Stabilitätskarte", (x0, rect.y + self.HEADER // 2), color, self.fonts.body, "midleft")
        cx, cy = rect.right - self.PAD - 6, rect.y + self.HEADER // 2
        if self.expanded: points = [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 3)]
        else: points = [(cx - 3, cy - 5), (cx - 3, cy + 5), (cx + 3, cy)]
        pygame.draw.polygon(surface, color, points)
        if not self.expanded: return

        layout = self.get_layout()
        for mode, label in self.MODES:
            button = layout["modes"][mode]
            draw_button(surface, button, self.mode == mode, self.hover == "mode:" + mode)
            if self.mode == mode: self.draw_text(surface, label, button.center, TEXT, anchor="center")
            else: self.draw_text(surface, label, button.center, DIM, anchor="center")

        area = layout["map"]
        pygame.draw.rect(surface, PLOT_BG, area)
        image = self.get_map_image(self.mode, area.size)
        if image is not None: surface.blit(image, area)
        elif self.survey is not None: self.draw_text(surface, "Wird berechnet …", area.center, DIM, anchor="center")

        if self.survey is not None: self.draw_guides(surface, area)
        pygame.draw.rect(surface, BOX_BORDER, area, width=1)
        self.draw_axes(surface, layout)

    def draw_guides(self, surface, area):
        # Everything that lies on top of the colors: the lines of circular speed and escape speed,
        # the hill sphere, the rings of the bodies and a frame around the cell under the mouse
        survey = self.survey
        surface.set_clip(area)

        #dashed lines across the map with their names at the left end
        for speed, label, anchor, nudge in ((1.0, "Kreisbahn", "bottomleft", -2),
                                            (math.sqrt(2.0), "Flucht", "topleft", 3)):
            y = area.bottom - area.height * speed / SPEED_MAX
            for x in range(area.x, area.right, 8):
                pygame.draw.line(surface, GUIDE, (x, y), (min(x + 4, area.right), y))
            self.draw_label(surface, label, (area.x + 5, y + nudge), anchor)

        hill = survey["hill"]
        if survey["inner"] < hill < survey["reach"]:
            x = self.get_map_point(hill, 0.0)[0]
            for y in range(area.y, area.bottom, 8):
                pygame.draw.line(surface, GUIDE, (x, y), (x, min(y + 4, area.bottom)))
            self.draw_label(surface, "Hill-Sphäre", (x - 3, area.bottom - 3), "bottomright")

        for i, marker in enumerate(survey["markers"]):
            point = self.get_map_point(marker[1], marker[2])
            if self.hover == ("marker", i):
                pygame.draw.circle(surface, RIM, point, 6)
                pygame.draw.circle(surface, (255, 255, 255), point, 5, 2)
            else:
                pygame.draw.circle(surface, RIM, point, 5)
                pygame.draw.circle(surface, (232, 236, 245), point, 4, 2)

        hover = self.hover
        if isinstance(hover, tuple) and hover[0] == "cell":
            x, y = self.get_map_point(hover[1], hover[2])
            cell_w, cell_h = area.width / COLUMNS, area.height / ROWS
            pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(x - cell_w, y - cell_h, 2 * cell_w + 1, 2 * cell_h + 1), 1)
        surface.set_clip(None)

    def draw_label(self, surface, text, position, anchor):
        # A text on a dark see-through plate, so it can be read on every color of the map
        rendered = self.fonts.render(text, GUIDE, self.font)
        box = rendered.get_rect()
        setattr(box, anchor, position)
        plate = pygame.Surface(box.inflate(6, 2).size, pygame.SRCALPHA)
        plate.fill(PLOT_BG + (190,))
        surface.blit(plate, box.inflate(6, 2))
        surface.blit(rendered, box)

    def draw_axes(self, surface, layout):
        area = layout["map"]

        #speed axis on the left
        self.draw_text(surface, "Starttempo (1 = Tempo der Kreisbahn)", (layout["left"], layout["speed_title_y"]), DIM)
        for speed, label in SPEED_TICKS:
            y = area.bottom - area.height * speed / SPEED_MAX
            pygame.draw.line(surface, BOX_BORDER, (area.x - 3, y), (area.x, y))
            self.draw_text(surface, label, (area.x - 6, y), DIM, anchor="midright")

        survey = self.survey
        if survey is None: return

        #distance axis at the bottom, in km around planets and in AE (or Lj) around stars
        reach = survey["reach"]
        if reach < 0.12 * AU_IN_METERS: unit = "km"
        else: unit = get_distance_unit(reach)
        scale = {"km": 1e3, "AE": AU_IN_METERS, "Lj": LIGHT_YEAR}[unit]
        low, high = survey["inner"] / scale, survey["reach"] / scale
        span = math.log(high / low)

        #marks at 1, 2 and 5 times a power of ten: the powers of ten get their place first,
        #the others only where they stay at least 64 px away from the marks that are already there
        round_values, other_values = [], []
        for exponent in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1):
            for mantissa in (1, 2, 5):
                value = mantissa * 10.0 ** exponent
                if low <= value <= high:
                    if mantissa == 1: round_values.append(value)
                    else: other_values.append(value)

        mark_x = {}
        for value in sorted(round_values) + sorted(other_values):
            x = area.x + area.width * math.log(value / low) / span
            fits = True
            for other_x in mark_x.values():
                if abs(x - other_x) < 64: fits = False
            if fits: mark_x[value] = x

        y = layout["ticks_y"]
        for value, x in mark_x.items():
            pygame.draw.line(surface, BOX_BORDER, (x, area.bottom), (x, area.bottom + 3))
            if value >= 1e6: text = format_number(value / 1e6, "g") + " Mio."
            elif value >= 1000: text = format_grouped(value)
            else: text = format_number(value, "g")
            label = self.fonts.render(text, DIM, self.font)
            box = label.get_rect(midtop=(x, y))
            #the outermost labels would stick out over the edge of the map
            box.clamp_ip(pygame.Rect(area.x, y, area.width, box.height))
            surface.blit(label, box)

        surface.set_clip(pygame.Rect(layout["left"], layout["distance_title_y"], layout["right"] - layout["left"], 18))
        self.draw_text(surface, f"Startabstand von {self.body.name} ({unit})", (area.centerx, layout["distance_title_y"]),
                       DIM, anchor="midtop")
        surface.set_clip(None)
