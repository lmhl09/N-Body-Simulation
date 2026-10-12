import math

import numpy as np
import pygame

from shaders import PLANET_STRIDE

#physical constants and units
GRAVITATIONAL_CONSTANT = 6.67430e-11
LIGHTSPEED = 299792458
ST_BO_CONSTANT = 5.67 * 10**(-8)

SUN_MASS = (1.989*10**30)
SUN_RADIUS = 6.957e8
SUN_LUMINOSITY = 3.828e26
EARTH_MASS = 5.972e24
EARTH_RADIUS = 6.371e6
JUPITER_MASS = 1.898e27

AU_IN_METERS = 149597870700
LIGHT_YEAR = 9.4607e15
KILOPARSEC = 3.0856775814913673e19

MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
WEEK = 7 * DAY
YEAR = 365.25 * DAY
MONTH = YEAR / 12

#heavier bodies are not possible, their mass would not fit into a float on the gpu anymore
MAX_MASS = 1e8 * SUN_MASS

PLANET_BUFFER = "planet_data"
TRAIL_BUFFER = "trail_data"
TRAIL_META_BUFFER = "trail_meta"
TRAIL_RESET_BUFFER = "trail_resets"
PARTICLE_BUFFER = "particle_data"
PARTICLE_GROUP_BUFFER = "particle_groups"

#where the values of a body are inside its record (same order as struct Body in shaders.py)
F_POS_X, F_POS_Y, F_MASS, F_RADIUS = 0, 1, 2, 3
F_VEL_X, F_VEL_Y, F_ACC_X, F_ACC_Y = 4, 5, 6, 7
F_COLOR_R, F_COLOR_G, F_COLOR_B, F_DENSITY = 8, 9, 10, 11
F_TYPE, F_FLAGS, F_TRAIL, F_SOFTENING = 12, 13, 14, 15
F_POS_LO_X, F_POS_LO_Y, F_UID = 16, 17, 18

FLAG_LOCKED = 1             #the body is selected
FLAG_BACKGROUND = 2         #the body belongs to the background (like the stars of the galaxy)
UID_LIMIT = 1 << 24         #a float cant count further than this without gaps

TYPE_PLANET = 0.0
TYPE_STAR = 1.0
TYPE_BLACKHOLE = 2.0


def split_position(value: float):
    # A float32 only has about 7 digits, far too few for a moon 384 000 km next to a planet 150 000 000 km out
    # so every coordinate is saved as a float and the rest the float cant hold
    high = float(np.float32(value))
    return high, float(value) - high


#biggest coordinate anything was placed at (in m)
extent = 0.0


def update_extent(x, y):
    global extent
    for coordinate in (abs(float(x)), abs(float(y))):
        if math.isfinite(coordinate): extent = max(extent, coordinate)


def get_length_scale():
    # The force shaders square distances, which overflows a float in a scene as big as a galaxy
    # they multiply all lengths with this factor first (1 for everything up to the size of a solar system)
    if extent <= 0.0: return 1.0
    return 2.0 ** min(0, 44 - math.ceil(math.log2(extent)))


### COLORS ###

def get_color_from_string(color: str) -> tuple:
    ### Converts string-color into rgb
    COLORS: dict = {
        "WHITE": (255, 255, 255),
        "RED": (255, 0, 0),
        "BLUE": (0, 0, 250),
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


def get_rgb_color(color) -> tuple:
    # A color can be given as rgb tuple or with its name
    if isinstance(color, tuple): return color
    if isinstance(color, str): return get_color_from_string(color)
    print(f"Invalid instance: {color} -> {type(color)}; Expected String or Tuple")
    return (255, 255, 255)


def color_for_shader(color, opacity=None):
    # The shaders want colors from 0 to 1 instead of 0 to 255
    rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    if opacity is None: return rgb
    return rgb + (opacity,)


### SIZES AND STARS ###

def calculate_sphere_radius(mass, density):
    return float(np.cbrt(3 * mass / density / (4 * np.pi)))


def calculate_star_radius(mass):
    # Mass-radius relation of main sequence stars
    mass_ratio = mass / SUN_MASS
    if mass_ratio < 0.43: radius_ratio = mass_ratio ** 0.95
    elif mass_ratio < 1.0: radius_ratio = mass_ratio ** 0.8
    elif mass_ratio < 5.0: radius_ratio = mass_ratio ** 0.57
    else: radius_ratio = mass_ratio ** 0.72
    return radius_ratio * SUN_RADIUS


def calculate_luminosity(object):
    mass_ratio = object.mass / SUN_MASS

    if mass_ratio < 0.43: luminosity = 0.19 * (mass_ratio ** 2.3)
    elif mass_ratio < 2.0: luminosity = mass_ratio ** 4.0
    elif mass_ratio < 20.0: luminosity = 1.4 * (mass_ratio ** 3.5)
    else: luminosity = 14500 * mass_ratio

    return luminosity


def calculate_stars_temperature(object):
    luminosity = object.luminosity
    radius = object.radius_in_meters
    temperature = (luminosity / (4 * np.pi * radius**(2) * ST_BO_CONSTANT))**(1/4)
    return temperature


def get_spectral_class_color(temperature):
    if temperature >= 30000: return (77, 68, 208)
    elif 30000 > temperature >= 10000: return (162, 185, 255)
    elif 10000 > temperature >= 7500: return (233, 239, 255)
    elif 7500 > temperature >= 6000: return (249, 255, 211)
    elif 6000 > temperature >= 5200: return (255, 255, 0)
    elif 5200 > temperature >= 3700: return (172, 94, 15)
    elif 3700 > temperature: return (118, 26, 7)


### ORBITS ###

def get_object_distance(primary_object, secondary_object):
    relative_vector = primary_object.position_vector_in_meters - secondary_object.position_vector_in_meters
    return relative_vector.length()


def to_numpy(vector):
    return np.array([vector.x, vector.y], dtype=np.float64)


def get_exact_position(records):
    # Adds the two parts of the position back together (works for one record and for many)
    records = np.asarray(records, dtype=np.float64)
    return records[..., [F_POS_X, F_POS_Y]] + records[..., [F_POS_LO_X, F_POS_LO_Y]]


def get_orbit_direction(offset, velocity):
    # 1 if the body goes around its center anticlockwise, -1 if clockwise
    if offset[0] * velocity[1] - offset[1] * velocity[0] < 0: return -1.0
    return 1.0


def get_tangent(direction, orbit_direction=1.0):
    # The direction a body on a circular orbit moves in (direction = where the body is, seen from the center)
    return orbit_direction * np.array([-direction[1], direction[0]])


def calculate_circular_velocity(position, attractor_position, attractor_velocity, mu, orbit_direction=1.0):
    # Velocity for a circular orbit: v = sqrt(G M / r), at a right angle to the line to the attractor (mu = G M)
    offset = np.asarray(position, dtype=np.float64) - np.asarray(attractor_position, dtype=np.float64)
    distance = float(np.hypot(*offset))
    if distance <= 0 or mu <= 0: return np.asarray(attractor_velocity, dtype=np.float64)

    tangent = get_tangent(offset / distance, orbit_direction)
    return np.asarray(attractor_velocity, dtype=np.float64) + tangent * math.sqrt(mu / distance)


def calculate_orbit_period(semi_major, mu):
    # Kepler's third law
    return 2.0 * math.pi * math.sqrt(semi_major ** 3 / mu)


def calculate_orbit_shape(r_vec, v_vec, mu):
    # Returns eccentricity and semi-major axis from position and velocity relative to the center
    # a body with positive energy escapes, it has no semi-major axis (None)
    distance = math.hypot(r_vec[0], r_vec[1])
    if distance <= 0 or mu <= 0: return 0.0, None

    energy = 0.5 * (v_vec[0] ** 2 + v_vec[1] ** 2) - mu / distance
    spin = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]
    eccentricity = math.sqrt(max(0.0, 1.0 + 2.0 * energy * spin * spin / (mu * mu)))

    if energy < 0: return eccentricity, -mu / (2.0 * energy)
    return eccentricity, None


def calculate_orbit_velocity(object, center_object):
    velocity = calculate_circular_velocity(to_numpy(object.position_vector_in_meters),
                                           to_numpy(center_object.position_vector_in_meters),
                                           to_numpy(center_object.speed_vector),
                                           GRAVITATIONAL_CONSTANT * center_object.mass)
    return pygame.math.Vector2(*velocity)


def calculate_eliptic_velocity(object, center_object, perihel_radius):
    # The body starts at its farthest point (aphel), there it is slower than on a circular orbit (vis-viva equation)
    aphel_radius = get_object_distance(object, center_object)
    if aphel_radius <= 0: return pygame.math.Vector2(center_object.speed_vector)

    circular = calculate_orbit_velocity(object, center_object) - center_object.speed_vector
    return center_object.speed_vector + circular * math.sqrt(2 * perihel_radius / (aphel_radius + perihel_radius))


class CircularSolarObject():

    objects = []
    by_id = {}
    by_uid = {}             #the gpu only knows the uid of a body, this finds the object for it
    black_holes = []
    softened = 0            #how many bodies have their own softening (ring bodies)
    engine = None

    #functions that want to know when a body gets deleted
    delete_hooks = []

    type = TYPE_PLANET
    kind_name = "Planet"
    has_trail = True
    created_count = 0
    next_uid = 1

    def __init__(self, density: float, mass: float, starting_position: tuple, color="WHITE",
                 speed_vector: tuple = (0, 0), orbit_object=None, name: str = None):

        self.density: float = float(density)
        self.mass: float = min(float(mass), MAX_MASS)

        self.uid = CircularSolarObject.next_uid
        CircularSolarObject.next_uid = CircularSolarObject.next_uid % (UID_LIMIT - 1) + 1

        #without a name the body is called "Planet 12", "Stern 3" ...
        CircularSolarObject.created_count += 1
        if name: self.name: str = name
        else: self.name: str = f"{self.kind_name} {CircularSolarObject.created_count}"

        #position and speed here are only a copy, the real values are on the gpu
        self.position_vector_in_meters = pygame.math.Vector2(starting_position[0], starting_position[1])
        self.speed_vector = pygame.math.Vector2(speed_vector[0], speed_vector[1])
        update_extent(starting_position[0], starting_position[1])

        self.color: tuple = get_rgb_color(color)
        self.flags = 0
        self.softening = 0.0
        self.trail = None

        if self.mass > 0 and self.density > 0:
            self.radius_in_meters: float = calculate_sphere_radius(self.mass, self.density)
        else:
            print(f"Invalid value mass - {self.mass} or density - {self.density}; Expected > 0")
            self.radius_in_meters = 0.0

        self.calculate_properties()

        #orbit_object = a body to circle around, or (body, perihel radius) for an elliptic orbit
        if isinstance(orbit_object, CircularSolarObject):
            orbit_object.sync_from_gpu()
            self.speed_vector = calculate_orbit_velocity(self, orbit_object)
        elif isinstance(orbit_object, tuple):
            orbit_object[0].sync_from_gpu()
            self.speed_vector = calculate_eliptic_velocity(self, orbit_object[0], orbit_object[1])

        self.engine.AddObjectToBuffer(object_id=id(self), buffer_name=PLANET_BUFFER, new_data=self.build_record())
        CircularSolarObject.objects.append(self)
        CircularSolarObject.by_id[id(self)] = self
        CircularSolarObject.by_uid[self.uid] = self
        if self.type == TYPE_BLACKHOLE: CircularSolarObject.black_holes.append(self)

    def calculate_properties(self):
        # Stars and black holes get their radius and color from their mass here
        pass

    def build_record(self):
        # Packs the body into the 20 floats the gpu gets
        record = np.zeros(PLANET_STRIDE, dtype=np.float32)

        record[F_POS_X], record[F_POS_LO_X] = split_position(self.position_vector_in_meters.x)
        record[F_POS_Y], record[F_POS_LO_Y] = split_position(self.position_vector_in_meters.y)
        record[F_MASS] = self.mass
        record[F_RADIUS] = self.radius_in_meters

        record[F_VEL_X] = self.speed_vector.x
        record[F_VEL_Y] = self.speed_vector.y

        record[F_COLOR_R:F_COLOR_B + 1] = color_for_shader(self.color)
        record[F_DENSITY] = self.density

        record[F_TYPE] = self.type
        record[F_FLAGS] = self.flags
        if self.trail is None or self.trail.slot is None: record[F_TRAIL] = -1.0
        else: record[F_TRAIL] = float(self.trail.slot)
        record[F_SOFTENING] = self.softening
        record[F_UID] = self.uid

        return record

    @property
    def index(self):
        return self.engine.GetObjectIndex(id(self), PLANET_BUFFER)

    @property
    def alive(self):
        return self.index is not None

    @classmethod
    def get_body_at_screen(cls, position):
        return cls.by_id.get(cls.engine.PickBodyAtScreen(position))

    @classmethod
    def read_all_records(cls):
        # Reads every body back from the gpu, returns the records and the exact positions
        records = cls.engine.ReadBufferRecords(PLANET_BUFFER).astype(np.float64)
        return records, get_exact_position(records)

    def sync_from_gpu(self):
        # Brings position and speed of the python object up to date
        record = self.engine.ReadObjectData(id(self), PLANET_BUFFER)
        if record is None: return

        self.position_vector_in_meters.update(*get_exact_position(record))
        self.speed_vector.update(float(record[F_VEL_X]), float(record[F_VEL_Y]))

    ### CHANGING A BODY (every change also goes to the gpu) ###

    def set_locked(self, value: bool):
        # Selected bodies get a ring around them in the body shader
        if value: self.flags = self.flags | FLAG_LOCKED
        else: self.flags = self.flags & ~FLAG_LOCKED
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_FLAGS, [float(self.flags)])

    def make_background(self, softening: float = 0.0):
        # Background bodies have no trail, no force arrows point at them and nothing counts as orbiting them
        # softening spreads their pull over that width (in m), so the bodies of a ring dont fling each other out
        self.flags |= FLAG_BACKGROUND
        self.has_trail = False

        if self.softening > 0: CircularSolarObject.softened -= 1
        if softening > 0: CircularSolarObject.softened += 1
        self.softening = float(softening)

        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_FLAGS, [float(self.flags)])
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_SOFTENING, [self.softening])

    def set_color(self, color):
        self.color = get_rgb_color(color)
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_COLOR_R, color_for_shader(self.color))
        if self.trail is not None: self.trail.set_color(self.color)

    def set_position(self, position):
        self.position_vector_in_meters.update(position[0], position[1])
        update_extent(position[0], position[1])
        x, x_lo = split_position(position[0])
        y, y_lo = split_position(position[1])
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_POS_X, [x, y])
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_POS_LO_X, [x_lo, y_lo])

    def set_velocity(self, velocity):
        self.speed_vector.update(velocity[0], velocity[1])
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_VEL_X, [velocity[0], velocity[1]])

    def set_mass(self, mass: float):
        self.mass = min(float(mass), MAX_MASS)
        self.calculate_properties()
        if self.has_own_density: self.radius_in_meters = calculate_sphere_radius(self.mass, self.density)

        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_MASS, [self.mass, self.radius_in_meters])
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_COLOR_R, color_for_shader(self.color) + (self.density,))

    @property
    def has_own_density(self):
        # A planet can have any density, the size of a star or black hole follows from its mass
        return self.type == TYPE_PLANET

    def set_density(self, density: float):
        if not self.has_own_density or density <= 0: return False

        self.density = float(density)
        self.radius_in_meters = calculate_sphere_radius(self.mass, self.density)

        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_RADIUS, [self.radius_in_meters])
        self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_DENSITY, [self.density])
        return True

    def set_name(self, name: str):
        self.name = str(name)

    def update_after_merge(self, mass: float, radius: float, density: float):
        # The gpu merged another body into this one, the python object takes over the new values
        self.mass = float(mass)
        if self.type == TYPE_STAR:
            #a heavier star is hotter, so its color changes
            self.calculate_properties()
            self.engine.WriteObjectFields(id(self), PLANET_BUFFER, F_COLOR_R, color_for_shader(self.color))
        elif self.type == TYPE_BLACKHOLE:
            self.calculate_properties()
        else:
            self.density = float(density)
        self.radius_in_meters = float(radius)

    def delete(self):
        if not self.alive: return

        for hook in list(CircularSolarObject.delete_hooks): hook(self)

        if self.engine.anchor_object_id == id(self): self.engine.ReleaseAnchor()

        if self.trail is not None:
            self.trail.delete()

        self.engine.RemoveObjectFromBuffer(object_id=id(self), buffer_name=PLANET_BUFFER)

        CircularSolarObject.by_id.pop(id(self), None)
        if self.softening > 0: CircularSolarObject.softened -= 1
        if CircularSolarObject.by_uid.get(self.uid) is self: del CircularSolarObject.by_uid[self.uid]
        if self in CircularSolarObject.objects:
            CircularSolarObject.objects.remove(self)
        if self in CircularSolarObject.black_holes:
            CircularSolarObject.black_holes.remove(self)


class Star(CircularSolarObject):

    type = TYPE_STAR
    kind_name = "Stern"

    def calculate_properties(self):
        self.radius_in_meters = calculate_star_radius(self.mass)
        self.density = self.mass / ((4 / 3) * np.pi * self.radius_in_meters ** 3)

        self.luminosity = calculate_luminosity(self) * SUN_LUMINOSITY
        self.temperature = calculate_stars_temperature(self)
        self.color = get_spectral_class_color(self.temperature)


class BlackHole(CircularSolarObject):

    type = TYPE_BLACKHOLE
    kind_name = "Schwarzes Loch"

    def calculate_properties(self):
        #Schwarzschild radius
        self.radius_in_meters = (2 * GRAVITATIONAL_CONSTANT * self.mass) / LIGHTSPEED**2
        self.color = (0, 0, 0)


#what the placing tools of the toolbar create: an earth, a sun and a black hole of 10 sun masses
NEW_BODIES = {
    "planet": {"class": CircularSolarObject, "density": 5515, "mass": EARTH_MASS, "color": "LIGHT BLUE"},
    "star": {"class": Star, "density": 1408, "mass": SUN_MASS, "color": "WHITE"},
    "blackhole": {"class": BlackHole, "density": 1, "mass": 10 * SUN_MASS, "color": "WHITE"},
}


def create_new_body(kind, position, velocity=(0, 0), name=None, mass=None, density=None):
    values = NEW_BODIES[kind]
    if mass is None: mass = values["mass"]
    if density is None: density = values["density"]

    return values["class"](density=density, mass=mass, starting_position=position, color=values["color"],
                           speed_vector=velocity, name=name)


def calculate_ring_orbits(amount, min_radius, max_radius, mu):
    # Random places inside a ring with the velocity of a circular orbit (mu = G M of the center, 0 = no movement)
    # returns x, y, x velocity and y velocity as arrays
    angles = np.random.uniform(0, 2 * np.pi, amount)
    #the root makes the ring evenly filled, without it the inner part would be denser
    radii = np.sqrt(np.random.uniform(min_radius ** 2, max_radius ** 2, amount))

    if mu > 0: speed = np.sqrt(mu / np.maximum(radii, 1e-300))
    else: speed = np.zeros(amount)

    return radii * np.cos(angles), radii * np.sin(angles), -np.sin(angles) * speed, np.cos(angles) * speed


def create_body_ring(kind, amount, min_radius, max_radius, host=None, center=(0.0, 0.0)):
    # A ring of real bodies that pull on each other (around the host, or around center and at rest without one)
    amount = int(amount)
    velocity = (0.0, 0.0)
    mu = 0.0
    if host is not None:
        host.sync_from_gpu()
        center = tuple(host.position_vector_in_meters)
        velocity = tuple(host.speed_vector)
        mu = GRAVITATIONAL_CONSTANT * host.mass

    #mean distance between two neighbours, used as their softening
    spacing = math.sqrt(math.pi * (max_radius ** 2 - min_radius ** 2) / max(amount, 1))

    x, y, x_velocity, y_velocity = calculate_ring_orbits(amount, min_radius, max_radius, mu)
    for i in range(amount):
        position = (center[0] + float(x[i]), center[1] + float(y[i]))
        body = create_new_body(kind, position, (velocity[0] + float(x_velocity[i]), velocity[1] + float(y_velocity[i])))
        body.make_background(softening=spacing)


class TrailParticles():
    # The trail of a body, the points themselves are in a ring buffer on the gpu
    # all trails share one head: every sample_interval simulated seconds it moves on and every body writes its position there

    engine = None
    length = 1500
    sample_interval = 2000.0

    head = 0
    total_recorded = 0
    time_since_sample = 0.0

    free_slots = []
    used_slots = set()
    highest_slot = 0            #trails up to this slot get drawn
    meta = None                 #color and state of every slot (struct TrailMeta in shaders.py)
    changed_slots = None        #first and last + 1 slot that have to be uploaded
    new_trails = []
    warned = False

    META_DTYPE = np.dtype([("color", "<f4", (4,)), ("info", "<i4", (4,))])

    @classmethod
    def setup(cls, engine, max_slots: int, length: int, sample_interval: float = 2000.0):
        cls.engine = engine
        cls.length = length
        cls.sample_interval = float(sample_interval)
        cls.head = 0
        cls.total_recorded = 0
        cls.time_since_sample = 0.0
        cls.free_slots = list(range(max_slots))
        cls.used_slots = set()
        cls.highest_slot = 0
        cls.meta = np.zeros(max_slots, dtype=cls.META_DTYPE)
        cls.changed_slots = None
        cls.new_trails = []

    def __init__(self, tracking_object):
        if tracking_object.trail is not None: tracking_object.trail.delete()

        self.tracking_object = tracking_object
        self.color = tracking_object.color

        if len(TrailParticles.free_slots) == 0:
            if not TrailParticles.warned:
                print(f"No free trail slots left ({len(TrailParticles.meta)} in use); further trails are skipped")
                TrailParticles.warned = True
            self.slot = None
            return

        #always the lowest free slot, so the used slots stay close together
        self.slot = min(TrailParticles.free_slots)
        TrailParticles.free_slots.remove(self.slot)
        TrailParticles.used_slots.add(self.slot)
        TrailParticles.highest_slot = max(TrailParticles.highest_slot, self.slot + 1)

        #the slot stays switched off until its points are set (upload_trail_changes)
        TrailParticles.meta["info"][self.slot] = (0, 0, 0, 0)
        self.set_color(self.color)
        TrailParticles.new_trails.append(self)

        tracking_object.trail = self
        TrailParticles.engine.WriteObjectFields(id(tracking_object), PLANET_BUFFER, F_TRAIL, [float(self.slot)])

    @classmethod
    def mark_changed(cls, slot):
        if cls.changed_slots is None: cls.changed_slots = (slot, slot + 1)
        else: cls.changed_slots = (min(cls.changed_slots[0], slot), max(cls.changed_slots[1], slot + 1))

    def set_color(self, color):
        self.color = get_rgb_color(color)
        if self.slot is None: return
        TrailParticles.meta["color"][self.slot] = color_for_shader(self.color, 1.0)
        TrailParticles.mark_changed(self.slot)

    def delete(self):
        if self.slot is None: return

        TrailParticles.meta["info"][self.slot] = (0, 0, 0, 0)
        TrailParticles.mark_changed(self.slot)
        TrailParticles.engine.WriteObjectFields(id(self.tracking_object), PLANET_BUFFER, F_TRAIL, [-1.0])

        TrailParticles.free_slots.append(self.slot)
        TrailParticles.used_slots.discard(self.slot)
        if TrailParticles.used_slots: TrailParticles.highest_slot = max(TrailParticles.used_slots) + 1
        else: TrailParticles.highest_slot = 0

        if self.tracking_object.trail is self: self.tracking_object.trail = None

        self.slot = None

    @classmethod
    def move_head(cls, frame_sim_time: float):
        # Called every frame, moves the head on by one point at most
        cls.time_since_sample += frame_sim_time
        if cls.time_since_sample >= cls.sample_interval:
            cls.time_since_sample = min(cls.time_since_sample - cls.sample_interval, cls.sample_interval)
            cls.head = (cls.head + 1) % cls.length
            cls.total_recorded += 1

    @classmethod
    def upload_trail_changes(cls):
        engine = cls.engine

        #a new trail starts with all its points on the body, else it would show old points of the slot
        if cls.new_trails:
            rows = []
            for trail in cls.new_trails:
                if trail.slot is None: continue
                body_index = engine.GetObjectIndex(id(trail.tracking_object), PLANET_BUFFER)
                if body_index is None: continue

                cls.meta["info"][trail.slot] = (1, cls.total_recorded, 0, 0)
                cls.mark_changed(trail.slot)
                rows.append((trail.slot, body_index))

            cls.new_trails = []

            if rows:
                engine.WriteRawToBuffer(TRAIL_RESET_BUFFER, np.asarray(rows, dtype=np.int32).tobytes())
                engine.RunComputeShader("TRAIL_RESET", len(rows) * cls.length, {
                    "u_reset_count": len(rows),
                    "u_trail_length": cls.length,
                })

        if cls.changed_slots is not None:
            first, end = cls.changed_slots
            engine.WriteRawToBuffer(TRAIL_META_BUFFER, cls.meta[first:end].tobytes(),
                                    byte_offset=first * cls.meta.itemsize)
            cls.changed_slots = None

    @classmethod
    def get_draw_range(cls):
        # Returns (vertices, first vertex, instances) for drawing all trails
        vertices = min(cls.total_recorded + 1, cls.length)
        return vertices, cls.length - vertices, cls.highest_slot


class ParticleCluster():
    # Particles have no mass: all bodies pull on them but they pull on nothing
    # a cluster with a host is saved relative to that body and moves along with it

    objects = []
    engine = None

    group_count = 0             #clusters with a host, as the gpu knows them
    groups_changed = False
    layout_version = -1

    def __init__(self, x, y, x_velocity, y_velocity, color, opacity: float = 1.0, host=None):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.amount = len(x)
        self.color = get_rgb_color(color)
        self.opacity = float(opacity)
        self.host = host

        block = np.stack((x, y, x_velocity, y_velocity), axis=-1).astype(np.float32)
        self.first_index = ParticleCluster.engine.AppendBlockToBuffer(PARTICLE_BUFFER, block)
        ParticleCluster.objects.append(self)
        if host is not None: ParticleCluster.groups_changed = True

        if self.amount:
            if host is not None: origin = host.position_vector_in_meters
            else: origin = (0.0, 0.0)
            update_extent(abs(origin[0]) + np.abs(x).max(), abs(origin[1]) + np.abs(y).max())

    @classmethod
    def can_attach(cls):
        # The gpu only has room for a limited amount of clusters with a host
        attached = 0
        for cluster in cls.objects:
            if cluster.host is not None: attached += 1
        return attached < cls.engine.GetBufferCapacity(PARTICLE_GROUP_BUFFER)

    @property
    def rgba(self):
        return color_for_shader(self.color, self.opacity)

    def get_host_index(self):
        if self.host is None: return -1
        index = ParticleCluster.engine.GetObjectIndex(id(self.host), PLANET_BUFFER)
        if index is None: return -1
        return index

    @classmethod
    def upload_groups(cls):
        # Tells the gpu which particles belong to which body (only when a cluster or a body index changed)
        engine = cls.engine
        if not cls.groups_changed and engine.layout_version == cls.layout_version: return

        cls.groups_changed = False
        cls.layout_version = engine.layout_version

        rows = []
        for cluster in cls.objects:
            host_index = cluster.get_host_index()
            if host_index >= 0: rows.append((cluster.first_index, cluster.amount, host_index, 0))

        capacity = engine.GetBufferCapacity(PARTICLE_GROUP_BUFFER)
        if len(rows) > capacity:
            print(f"More than {capacity} attached particle clusters; the rest move as free particles")
            rows = rows[:capacity]

        cls.group_count = len(rows)
        if rows: engine.WriteRawToBuffer(PARTICLE_GROUP_BUFFER, np.asarray(rows, dtype=np.int32).tobytes())

    @classmethod
    def detach_from_body(cls, body):
        # The host gets deleted: its particles get position and speed of the host added and fly on freely
        engine = cls.engine

        for cluster in cls.objects:
            if cluster.host is not body: continue

            index = cluster.get_host_index()
            if index >= 0:
                engine.FlushUpdates(PLANET_BUFFER)
                engine.RunComputeShader("PARTICLE_DETACH", cluster.amount, {
                    "u_first": cluster.first_index,
                    "u_count": cluster.amount,
                    "u_host": index,
                })

            cluster.host = None
            cls.groups_changed = True


class ParticleRing(ParticleCluster):
    # A flat ring of particles on circular orbits, like the rings of saturn or an asteroid belt

    def __init__(self, amount: int, host, min_radius, max_radius, color=(185, 190, 200), opacity: float = 1.0):
        x, y, x_velocity, y_velocity = calculate_ring_orbits(int(amount), min_radius, max_radius,
                                                             GRAVITATIONAL_CONSTANT * host.mass)

        if not ParticleCluster.can_attach():
            #all groups are taken: the ring is made of free particles at the place the host is right now
            host.sync_from_gpu()
            x, y = x + host.position_vector_in_meters.x, y + host.position_vector_in_meters.y
            x_velocity, y_velocity = x_velocity + host.speed_vector.x, y_velocity + host.speed_vector.y
            host = None

        super().__init__(x, y, x_velocity, y_velocity, color, opacity, host=host)


def set_engine(engine):
    CircularSolarObject.engine = engine
    ParticleCluster.engine = engine

    if ParticleCluster.detach_from_body not in CircularSolarObject.delete_hooks:
        CircularSolarObject.delete_hooks.append(ParticleCluster.detach_from_body)


def delete_all_objects(engine):
    # Empties everything for the next scene
    global extent
    extent = 0.0
    CircularSolarObject.objects.clear()
    CircularSolarObject.by_id.clear()
    CircularSolarObject.by_uid.clear()
    CircularSolarObject.black_holes.clear()
    CircularSolarObject.softened = 0
    ParticleCluster.objects.clear()
    ParticleCluster.group_count = 0
    ParticleCluster.groups_changed = True

    engine.ResetBuffer(PLANET_BUFFER)
    engine.ResetBuffer(PARTICLE_BUFFER)

    #switches all trail slots off again
    TrailParticles.setup(engine, len(TrailParticles.meta), TrailParticles.length, TrailParticles.sample_interval)
    engine.WriteRawToBuffer(TRAIL_META_BUFFER, TrailParticles.meta.tobytes())


def send_changes_to_gpu(engine):
    # Once per frame, before the shaders run
    engine.FlushUpdates(PLANET_BUFFER)
    TrailParticles.upload_trail_changes()
