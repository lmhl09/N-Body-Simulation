Sun = CSO(density=1408, mass=(1.989*10**30), starting_position=(0, 0), color="YELLOW", acceleration_vector=(0, 0), speed_vector=(0, 0))
Mercury = CSO(density=5430, mass=(3.285*10**23), starting_position=(0.39 * AU_IN_METERS, 0), color="LIGHT GRAY", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (0.39 * AU_IN_METERS))))
Venus = CSO(density=5243, mass=(4.867*10**24), starting_position=(0.72 * AU_IN_METERS, 0), color="BEIGE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / (0.72 * AU_IN_METERS))))
Earth = CSO(density=5515, mass=(5.972*10**24), starting_position=(AU_IN_METERS, 0), color="BLUE", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 1.989e30) / 1.496e11)))
Moon = CSO(density=3340, mass=(7.342*10**22), starting_position=(149982270700, 0), color="GRAY", acceleration_vector=(0, 0), speed_vector=(0, (numpy.sqrt((6.67430e-11 * 1.989e30) / 1.496e11)) + (numpy.sqrt((6.67430e-11 * 5.972*10**24) / 384400000))))
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

Sagittarius = CSO(density=1067000, mass=((1.989*10**30) * (4.154*10**6)), starting_position=((1.64427*10**9) * AU_IN_METERS, 0), color="RED", acceleration_vector=(0, 0), speed_vector=(0, 0))

Earth = CSO(density=5515, mass=(5.972*10**24), starting_position=(0, 0), color="BLUE", acceleration_vector=(0, 0), speed_vector=(0, 0))
Moon = CSO(density=3340, mass=(7.342*10**22), starting_position=(384400000, 0), color="GRAY", acceleration_vector=(0, 0), speed_vector=(0, numpy.sqrt((6.67430e-11 * 5.972*10**24) / 384400000)))
