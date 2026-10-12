import numpy as np

from gpu_cso import (
    CircularSolarObject, Star, BlackHole, ParticleRing, create_body_ring, calculate_circular_velocity,
    AU_IN_METERS, KILOPARSEC, SUN_MASS, EARTH_MASS, EARTH_RADIUS, GRAVITATIONAL_CONSTANT,
)
from gpu_overlays import HillSphere, HabitableZone, Barycenter, LagrangePoints

CSO = CircularSolarObject

DEFAULT_ZOOM = 2.437658971708419e-09
GALAXY_ZOOM = 1.0e-18                   #around 35 kpc fit into the window

#values of the settings window when the scene got loaded (the galaxy with mass needs the body count)
scene_settings = {}


### HELPERS ###

def calculate_density(mass, radius):
    # Exoplanets are listed with mass and radius, a CSO wants the density
    return mass / ((4 / 3) * np.pi * radius ** 3)


def get_circular_orbit(radius, central_mass, angle=0.0):
    # Position and velocity on a circular orbit around a mass in the origin (the angle says where on the circle)
    position = (radius * np.cos(angle), radius * np.sin(angle))
    return position, tuple(calculate_circular_velocity(position, (0, 0), (0, 0), GRAVITATIONAL_CONSTANT * central_mass))


def stop_system_drift(bodies):
    # If the heavy body starts at rest and the others orbit it, the whole system slowly drifts out of the window
    # taking away the common velocity of all bodies stops that
    total_mass = sum(body.mass for body in bodies)
    drift_x = sum(body.mass * body.speed_vector.x for body in bodies) / total_mass
    drift_y = sum(body.mass * body.speed_vector.y for body in bodies) / total_mass

    for body in bodies:
        body.set_velocity((body.speed_vector.x - drift_x, body.speed_vector.y - drift_y))


### SCENES ###

def empty():
    # Nothing, for building something with the placing tools
    pass


def stress():
    # 10 000 earths on a grid, 1 AU apart (to see how many bodies the pc can handle)
    for i in range(100):
        for j in range(100):
            CSO(density=5515, mass=EARTH_MASS, starting_position=(AU_IN_METERS * i, AU_IN_METERS * j), color="BLUE")


def solar():
    Sagittarius = BlackHole(name="Sagittarius", density=1067000, mass=(SUN_MASS * (4.154*10**6)), starting_position=((1.64427*10**9) * AU_IN_METERS, 0))

    Sonne = Star(name="Sonne", density=1408, mass=SUN_MASS, starting_position=(0, 0), orbit_object=Sagittarius)
    Merkur = CSO(name="Merkur", density=5430, mass=(3.285*10**23), starting_position=(0.39 * AU_IN_METERS, 0), color="LIGHT GRAY", orbit_object=Sonne)
    Venus = CSO(name="Venus", density=5243, mass=(4.867*10**24), starting_position=(0.72 * AU_IN_METERS, 0), color="BEIGE", orbit_object=Sonne)
    Erde = CSO(name="Erde", density=5515, mass=EARTH_MASS, starting_position=(AU_IN_METERS, 0), color="BLUE", orbit_object=Sonne)
    Mond = CSO(name="Mond", density=3340, mass=(7.342*10**22), starting_position=(AU_IN_METERS + 384400000, 0), color="GRAY", speed_vector=(0, Erde.speed_vector.y + 1023))
    Mars = CSO(name="Mars", density=3933, mass=(6.39*10**23), starting_position=(2.279e11, 0), color="RED", orbit_object=Sonne)
    Jupiter = CSO(name="Jupiter", density=1330, mass=(1.898*10**27), starting_position=(5.2 * AU_IN_METERS, 0), color="ORANGE", orbit_object=Sonne)
    Saturn = CSO(name="Saturn", density=687, mass=(5.683*10**26), starting_position=(9.5 * AU_IN_METERS, 0), color="LIGHT BEIGE", orbit_object=Sonne)
    Uranus = CSO(name="Uranus", density=1270, mass=(8.681*10**25), starting_position=(19.2 * AU_IN_METERS, 0), color="LIGHT BLUE", orbit_object=Sonne)
    Neptun = CSO(name="Neptun", density=1760, mass=(1.024*10**26), starting_position=(30.05 * AU_IN_METERS, 0), color="TURQUOISE", orbit_object=Sonne)

    #pluto starts at its farthest point and comes as close as 30 AU
    Pluto = CSO(name="Pluto", density=1860, mass=1.3e22, starting_position=(49.3 * AU_IN_METERS, 0), color="ORANGE", orbit_object=(Sonne, 30 * AU_IN_METERS))

    #moons of jupiter
    Io = CSO(name="Io", density=3528, mass=8.94e22, starting_position=((5.2 * AU_IN_METERS) + 421600000, 0), color="GRAY", orbit_object=Jupiter)
    Europa = CSO(name="Europa", density=3010, mass=4.8e22, starting_position=((5.2 * AU_IN_METERS) + 671000000, 0), color="LIGHT BLUE", orbit_object=Jupiter)
    Ganymed = CSO(name="Ganymed", density=1940, mass=1.48e23, starting_position=((5.2 * AU_IN_METERS) + 1070000000, 0), color="LIGHT GRAY", orbit_object=Jupiter)
    Kallisto = CSO(name="Kallisto", density=1834, mass=1.0759e23, starting_position=(779802700000, 0), color="WHITE", orbit_object=Jupiter)

    #moons of saturn
    Mimas = CSO(name="Mimas", density=1149, mass=3.75e19, starting_position=((9.5 * AU_IN_METERS) + 185539000, 0), color="LIGHT GRAY", orbit_object=Saturn)
    Enceladus = CSO(name="Enceladus", density=1609, mass=1.08e20, starting_position=((9.5 * AU_IN_METERS) + 237948000, 0), color="WHITE", orbit_object=Saturn)
    Tethys = CSO(name="Tethys", density=984, mass=6.17e20, starting_position=((9.5 * AU_IN_METERS) + 294619000, 0), color="LIGHT GRAY", orbit_object=Saturn)
    Dione = CSO(name="Dione", density=1478, mass=1.095e21, starting_position=((9.5 * AU_IN_METERS) + 377396000, 0), color="LIGHT GRAY", orbit_object=Saturn)
    Rhea = CSO(name="Rhea", density=1236, mass=2.307e21, starting_position=((9.5 * AU_IN_METERS) + 527108000, 0), color="LIGHT GRAY", orbit_object=Saturn)
    Titan = CSO(name="Titan", density=1880, mass=1.345e23, starting_position=((9.5 * AU_IN_METERS) + 1221870000, 0), color="ORANGE", orbit_object=Saturn)
    Iapetus = CSO(name="Iapetus", density=1088, mass=1.806e21, starting_position=((9.5 * AU_IN_METERS) + 3560820000, 0), color="GRAY", orbit_object=Saturn)

    #rings of saturn (radii from the center of saturn)
    C_Ring = ParticleRing(amount=10000, host=Saturn, min_radius=74658e3, max_radius=92000e3, color=(120, 110, 95), opacity=0.5)
    B_Ring = ParticleRing(amount=32000, host=Saturn, min_radius=92000e3, max_radius=117580e3, color=(235, 220, 185), opacity=0.85)
    A_Ring = ParticleRing(amount=14000, host=Saturn, min_radius=122170e3, max_radius=133423e3, color=(205, 190, 160), opacity=0.75)
    A_Ring_Outer = ParticleRing(amount=4000, host=Saturn, min_radius=133745e3, max_radius=136775e3, color=(205, 190, 160), opacity=0.75)
    F_Ring = ParticleRing(amount=2000, host=Saturn, min_radius=140000e3, max_radius=140360e3, color=(200, 200, 200), opacity=0.6)

    Main_Belt = ParticleRing(amount=50000, host=Sonne, min_radius=2.2*AU_IN_METERS, max_radius=3.2*AU_IN_METERS, color="GRAY")
    Kuiper_Belt = ParticleRing(amount=100000, host=Sonne, min_radius=30*AU_IN_METERS, max_radius=50*AU_IN_METERS, color="GRAY")

    #overlays
    Earth_Hill_Sphere = HillSphere(Erde)
    Earth_Moon_Barycenter = Barycenter(Mond)
    Earth_Lagrange_Points = LagrangePoints(Erde)
    Sun_Habitable_Zone = HabitableZone(Sonne)


def earth_moon():
    Erde = CSO(name="Erde", density=5515, mass=EARTH_MASS, starting_position=(0, 0), color="BLUE")
    Mond = CSO(name="Mond", density=3340, mass=(7.342*10**22), starting_position=(384400000, 0), color="GRAY", orbit_object=Erde)
    stop_system_drift([Erde, Mond])

    Moon_Hill_Sphere = HillSphere(Mond)
    Earth_Moon_Barycenter = Barycenter(Mond)
    Moon_Lagrange_Points = LagrangePoints(Mond)


def hollow_moon():
    # The "hollow moon": as big as the real one but with only about 6 % of its mass
    Erde = CSO(name="Erde", density=5515, mass=EARTH_MASS, starting_position=(0, 0), color="BLUE")
    Hohler_Mond = CSO(name="Hohler Mond", density=202, mass=(4.45*10**21), starting_position=(384400000, 0), color="GRAY", orbit_object=Erde)
    stop_system_drift([Erde, Hohler_Mond])

    Moon_Hill_Sphere = HillSphere(Hohler_Mond)
    Earth_Moon_Barycenter = Barycenter(Hohler_Mond)
    Moon_Lagrange_Points = LagrangePoints(Hohler_Mond)


def sun_earth_moon():
    # With the force arrows switched on the moon shows that the sun pulls harder on it than the earth does
    Sonne = Star(name="Sonne", density=1408, mass=SUN_MASS, starting_position=(0, 0))
    Erde = CSO(name="Erde", density=5515, mass=EARTH_MASS, starting_position=(AU_IN_METERS, 0), color="BLUE", orbit_object=Sonne)
    Mond = CSO(name="Mond", density=3340, mass=(7.342*10**22), starting_position=(AU_IN_METERS + 384400000, 0), color="GRAY", speed_vector=(0, Erde.speed_vector.y + 1023))

    #L1 and L2 of sun and earth are about 1.5 million km away from the earth
    Sun_Habitable_Zone = HabitableZone(Sonne)
    Earth_Hill_Sphere = HillSphere(Erde)
    Earth_Moon_Barycenter = Barycenter(Mond)
    Earth_Lagrange_Points = LagrangePoints(Erde)


def kepler_47():
    # Two suns orbiting each other with three planets around both (values from Orosz et al. 2019, all orbits taken as circles)
    star_a_mass = 0.957 * SUN_MASS
    star_b_mass = 0.342 * SUN_MASS
    binary_mass = star_a_mass + star_b_mass
    separation = 0.0836 * AU_IN_METERS
    relative_speed = np.sqrt(GRAVITATIONAL_CONSTANT * binary_mass / separation)

    #both suns circle their common center of mass, the lighter one further out and faster
    Kepler47_A = Star(name="Kepler47 A", density=1408, mass=star_a_mass, starting_position=(-separation * star_b_mass / binary_mass, 0), speed_vector=(0, -relative_speed * star_b_mass / binary_mass))
    Kepler47_B = Star(name="Kepler47 B", density=1408, mass=star_b_mass, starting_position=(separation * star_a_mass / binary_mass, 0), speed_vector=(0, relative_speed * star_a_mass / binary_mass))

    b_position, b_velocity = get_circular_orbit(0.2877 * AU_IN_METERS, binary_mass, angle=0.0)
    d_position, d_velocity = get_circular_orbit(0.6992 * AU_IN_METERS, binary_mass, angle=2.1)
    c_position, c_velocity = get_circular_orbit(0.9638 * AU_IN_METERS, binary_mass, angle=4.2)

    Kepler47_b = CSO(name="Kepler47 b", density=calculate_density(2.07 * EARTH_MASS, 3.05 * EARTH_RADIUS), mass=(2.07 * EARTH_MASS), starting_position=b_position, color="LIGHT BLUE", speed_vector=b_velocity)
    Kepler47_d = CSO(name="Kepler47 d", density=calculate_density(19.02 * EARTH_MASS, 7.04 * EARTH_RADIUS), mass=(19.02 * EARTH_MASS), starting_position=d_position, color="TURQUOISE", speed_vector=d_velocity)
    Kepler47_c = CSO(name="Kepler47 c", density=calculate_density(3.17 * EARTH_MASS, 4.65 * EARTH_RADIUS), mass=(3.17 * EARTH_MASS), starting_position=c_position, color="BEIGE", speed_vector=c_velocity)

    Binary_Barycenter = Barycenter(Kepler47_A)

    stop_system_drift([Kepler47_A, Kepler47_B, Kepler47_b, Kepler47_d, Kepler47_c])


def proxima_centauri():
    # Our nearest star with its planets d and b (values from Faria et al. 2022)
    Proxima = Star(name="Proxima", density=1408, mass=(0.1221 * SUN_MASS), starting_position=(0, 0))
    Proxima_d = CSO(name="Proxima d", density=5515, mass=(0.26 * EARTH_MASS), starting_position=(-0.02885 * AU_IN_METERS, 0), color="BEIGE", orbit_object=Proxima)
    Proxima_b = CSO(name="Proxima b", density=5515, mass=(1.07 * EARTH_MASS), starting_position=(0.04857 * AU_IN_METERS, 0), color="TURQUOISE", orbit_object=Proxima)

    #planet c is left out: it is not confirmed and would be far outside the window

    Proxima_Habitable_Zone = HabitableZone(Proxima)


def tau_ceti():
    # Four planets (Feng et al. 2017) and the debris disc as particles (Lawler et al. 2014)
    Tau_Ceti = Star(name="Tau Ceti", density=1408, mass=(0.783 * SUN_MASS), starting_position=(0, 0))
    Tau_Ceti_g = CSO(name="Tau Ceti g", density=5515, mass=(1.75 * EARTH_MASS), starting_position=(0.133 * AU_IN_METERS, 0), color="LIGHT GRAY", orbit_object=Tau_Ceti)
    Tau_Ceti_h = CSO(name="Tau Ceti h", density=5515, mass=(1.83 * EARTH_MASS), starting_position=(0, 0.243 * AU_IN_METERS), color="BEIGE", orbit_object=Tau_Ceti)
    Tau_Ceti_e = CSO(name="Tau Ceti e", density=5515, mass=(3.93 * EARTH_MASS), starting_position=(-0.538 * AU_IN_METERS, 0), color="ORANGE", orbit_object=Tau_Ceti)
    Tau_Ceti_f = CSO(name="Tau Ceti f", density=5515, mass=(3.93 * EARTH_MASS), starting_position=(0, -1.334 * AU_IN_METERS), color="LIGHT BLUE", orbit_object=Tau_Ceti)

    Debris_Disc = ParticleRing(amount=30000, host=Tau_Ceti, min_radius=6.2*AU_IN_METERS, max_radius=52*AU_IN_METERS, color="GRAY")

    Tau_Ceti_Habitable_Zone = HabitableZone(Tau_Ceti)


#a very simple milky way: the stars as a flat ring around Sagittarius A*
#(values from GRAVITY Collaboration 2022 and Goodwin et al. 1998)
#all the mass besides the black hole is missing, so this galaxy turns far slower than the real one
#(thats why the two scenes run at about a billion years per second)
SGR_A_MASS = 4.297e6 * SUN_MASS
SUN_DISTANCE = 8.277 * KILOPARSEC
GALAXY_INNER = 3.0 * KILOPARSEC
GALAXY_OUTER = 13.4 * KILOPARSEC
GALAXY_PARTICLES = 250000
GALAXY_SPEED = 3e16


def milky_way():
    # The stars are particles here: the black hole pulls on them but they dont pull on each other
    Sagittarius_A = BlackHole(density=1, mass=SGR_A_MASS, starting_position=(0, 0), name="Sagittarius A*")
    Sonne = Star(name="Sonne", density=1408, mass=SUN_MASS, starting_position=(0, SUN_DISTANCE), orbit_object=Sagittarius_A)

    Disc = ParticleRing(amount=GALAXY_PARTICLES, host=Sagittarius_A, min_radius=GALAXY_INNER, max_radius=GALAXY_OUTER, color=(205, 212, 228), opacity=0.6)


def milky_way_massive():
    # The same ring made of real stars that do pull on each other (how many is set in the settings window)
    Sagittarius_A = BlackHole(density=1, mass=SGR_A_MASS, starting_position=(0, 0), name="Sagittarius A*")
    Sonne = Star(name="Sonne", density=1408, mass=SUN_MASS, starting_position=(0, SUN_DISTANCE), orbit_object=Sagittarius_A)

    create_body_ring("star", scene_settings["ring_bodies"], GALAXY_INNER, GALAXY_OUTER, host=Sagittarius_A)


#every scene: its function, the zoom and speed it starts with (None = normal speed) and its name in the scene list
SCENES = {
    "empty": (empty, DEFAULT_ZOOM, None, "Leerer Raum"),
    "stress": (stress, DEFAULT_ZOOM, None, "Belastungstest"),
    "solar": (solar, DEFAULT_ZOOM, None, "Sonnensystem"),
    "earth_moon": (earth_moon, 3e-7, None, "Erde und Mond"),
    "hollow_moon": (hollow_moon, 3e-7, None, "Erde und hohler Mond"),
    "sun_earth_moon": (sun_earth_moon, DEFAULT_ZOOM, None, "Sonne, Erde und Mond"),
    "kepler_47": (kepler_47, DEFAULT_ZOOM, None, "Kepler-47 (zwei Sonnen)"),
    "proxima_centauri": (proxima_centauri, 3.4e-8, None, "Proxima Centauri"),
    "tau_ceti": (tau_ceti, 1.6e-9, None, "Tau Ceti"),
    "milky_way": (milky_way, GALAXY_ZOOM, GALAXY_SPEED, "Milchstraße"),
    "milky_way_massive": (milky_way_massive, GALAXY_ZOOM, GALAXY_SPEED, "Milchstraße (mit Masse)"),
}

TITLES = {}
for scene_name in SCENES:
    TITLES[scene_name] = SCENES[scene_name][3]


def load_scene(name, settings):
    # Creates everything of the scene and returns its zoom and speed
    global scene_settings
    scene_settings = dict(settings)

    build_scene, zoom, speed, title = SCENES[name]
    build_scene()
    return zoom, speed
