import math

import numpy as np

from engine import Readback
from gpu_cso import (
    CircularSolarObject, GRAVITATIONAL_CONSTANT, SUN_MASS, TYPE_BLACKHOLE, PLANET_BUFFER,
    F_MASS, F_RADIUS, F_DENSITY,
)
from shaders import EVENT_CAPACITY, STEP_LIMIT, NBODY_LOCAL_SIZE
from interface import with_comma

#bodies that touch get merged by a shader on the gpu, without python knowing about it
#for every merge the shader leaves an event, they get read here so the python objects can follow

EVENT_BINDING = 17
AUX_BINDING = 18
BLACK_HOLE_BINDING = 20
STEP_BINDING = 22

MAX_BLACK_HOLES = 256

#same layout as struct Event and struct Aux in shaders.py
EVENT_DTYPE = np.dtype([("info", "<i4", (4,)), ("data", "<f4", (4,))])
AUX_DTYPE = np.dtype([("partner", "<i4"), ("claim", "<i4"), ("tide_source", "<i4"), ("pad0", "<i4"),
                      ("tidal", "<f4"), ("tidal_r2", "<f4"), ("pad1", "<f4"), ("pad2", "<f4")])
NO_CLAIM = 2147483647
EVENT_MERGE = 1

#a sub step may turn the fastest orbit by this angle (rad) at most
STEP_TURN = 0.005


class Impacts():

    INTERVAL = 0.1          #seconds between two looks at the events

    def __init__(self, engine, max_bodies, toolbar, inspector, select_body, get_selected_body):
        self.engine = engine
        self.toolbar = toolbar
        self.inspector = inspector
        self.select_body = select_body
        self.get_selected_body = get_selected_body

        #the collision shader needs some extra values for every body (who it touches, the strongest tide)
        aux = np.zeros(max_bodies, dtype=AUX_DTYPE)
        aux["partner"] = -1
        aux["claim"] = NO_CLAIM
        aux["tide_source"] = -1
        self.empty_aux = aux.tobytes()
        self.aux = engine.ctx.buffer(self.empty_aux)
        self.aux.bind_to_storage_buffer(AUX_BINDING)

        #the first 16 bytes of an event buffer are the event counter
        #the gpu always writes into one buffer while an older one gets read
        self.events = Readback(engine, 16 + EVENT_CAPACITY * EVENT_DTYPE.itemsize, EVENT_BINDING)
        self.active = self.events.next_buffer(bytes(16))
        self.last_switch = 0.0

        #black holes are tiny and fast, the collision shader checks them along their whole way
        #for that it needs the list of their indices
        self.black_hole_buffer = engine.ctx.buffer(reserve=MAX_BLACK_HOLES * 4)
        self.black_hole_buffer.bind_to_storage_buffer(BLACK_HOLE_BINDING)
        self.black_hole_indices = []

    def upload_black_holes(self):
        # Returns how many black holes there are (the list only gets uploaded when it changed)
        indices = []
        for body in CircularSolarObject.black_holes:
            if body.index is not None: indices.append(body.index)
        indices = indices[:MAX_BLACK_HOLES]

        if indices != self.black_hole_indices:
            self.black_hole_indices = indices
            if indices: self.black_hole_buffer.write(np.asarray(indices, dtype=np.int32).tobytes())
        return len(indices)

    def scene_changed(self):
        self.aux.write(self.empty_aux)
        self.events.forget_waiting()
        self.events.buffers[self.active].write(bytes(16))

    ### READING EVENTS ###

    def tick(self, now):
        # Once per frame
        finished = self.events.get_finished()
        if finished is not None: self.read_events(finished[0], now)

        #the gpu continues in the next buffer, the one it used until now waits to be read
        if now - self.last_switch >= self.INTERVAL and self.events.has_free_buffer(kept=1):
            self.last_switch = now
            self.events.mark_written(self.active)
            self.active = self.events.next_buffer(bytes(16))

    def read_events(self, buffer, now):
        count = int(np.frombuffer(buffer.read(size=4), np.uint32)[0])
        if count == 0: return

        saved = min(count, EVENT_CAPACITY)
        raw = buffer.read(size=saved * EVENT_DTYPE.itemsize, offset=16)
        events = np.frombuffer(raw, dtype=EVENT_DTYPE)
        for i in range(saved):
            info, data = events["info"][i], events["data"][i]
            if int(info[0]) == EVENT_MERGE: self.handle_merge(int(info[1]), int(info[2]), data, now)

        #more merges happened than events fit into the buffer: the rest has to be found by comparing all bodies
        if count > EVENT_CAPACITY: self.compare_all_bodies()

    def compare_all_bodies(self):
        # Reads every body back and fixes the python objects (mass 0 = the body got absorbed)
        records = self.engine.ReadBufferRecords(PLANET_BUFFER)
        for body in list(CircularSolarObject.objects):
            index = body.index
            if index is None or index >= len(records): continue
            record = records[index]
            if record[F_MASS] <= 0: body.delete()
            elif record[F_MASS] != np.float32(body.mass):
                body.update_after_merge(float(record[F_MASS]), float(record[F_RADIUS]), float(record[F_DENSITY]))

    def handle_merge(self, survivor_uid, absorbed_uid, data, now):
        # One body absorbed another one: the absorbed object gets deleted, the survivor gets its new values
        survivor = CircularSolarObject.by_uid.get(survivor_uid)
        absorbed = CircularSolarObject.by_uid.get(absorbed_uid)
        selected = self.get_selected_body()
        follow = self.inspector.follow

        if absorbed is not None: absorbed.delete()
        if survivor is None: return

        mass, radius, density, radiated = float(data[0]), float(data[1]), float(data[2]), float(data[3])
        survivor.update_after_merge(mass, radius, density)

        #if the selected body is gone the selection moves on to the body that swallowed it
        if absorbed is not None and selected is absorbed:
            self.select_body(survivor, follow=follow)

        if survivor is selected or absorbed is selected:
            self.inspector.body_changed()
            if absorbed: text = f"{absorbed.name} wurde von {survivor.name} absorbiert"
            else: text = f"Ein Körper wurde von {survivor.name} absorbiert"
            #a black hole radiates a part of what falls into it away
            if radiated > 0 and survivor.type == TYPE_BLACKHOLE:
                text += f" · {with_comma(f'{radiated / SUN_MASS:.3g}')} M☉ abgestrahlt"
            self.toolbar.notify(text, now)


class StepLimit():
    # Finds out how long a sub step of the simulation may be
    # a shader looks for the two bodies with the fastest possible orbit (biggest G (m1 + m2) / r³),
    # with steps that are too long for that orbit a moon would fly off its planet

    INTERVAL = 0.1          #seconds between two measurements
    PAIRS_PER_SECOND = 2e9  #with many bodies it gets measured less often (the shader checks every pair)

    def __init__(self, engine):
        self.engine = engine
        engine.SetupNewComputeShader("STEP_LIMIT", STEP_LIMIT)
        self.readback = Readback(engine, 4, STEP_BINDING, slots=3)
        self.last_run = -1e9
        self.count = -1             #body count at the last measurement (-1 = never measured)
        self.max_step = math.inf    #in s

    def start_measurement(self, count, softening, length_scale):
        slot = self.readback.next_buffer(bytes(4))
        self.engine.RunComputeShader("STEP_LIMIT", count, {
            "u_count": count,
            "u_G": GRAVITATIONAL_CONSTANT,
            "u_softening": softening,
            "u_length_scale": length_scale,
        }, local_size=NBODY_LOCAL_SIZE)
        return slot

    def read_result(self, buffer):
        # fastest = the square of the angular speed of the fastest orbit (in 1/s²)
        fastest = float(np.frombuffer(buffer.read(), dtype=np.float32)[0])
        if fastest > 0.0 and math.isfinite(fastest): self.max_step = STEP_TURN / math.sqrt(fastest)
        else: self.max_step = math.inf

    def reset(self):
        self.readback.forget_waiting()
        self.count = -1

    def update(self, now, count, softening, length_scale):
        # Once per frame
        #the first measurement of a scene gets read at once, the scene must not start with steps that are too long
        if self.count < 0:
            self.count = count
            self.last_run = now
            slot = self.start_measurement(count, softening, length_scale)
            self.read_result(self.readback.buffers[slot])
            return

        finished = self.readback.get_finished()
        if finished is not None: self.read_result(finished[0])

        #a new body can be on a very fast orbit, so a changed body count gets measured right away
        interval = max(self.INTERVAL, count * count / self.PAIRS_PER_SECOND)
        if (count != self.count or now - self.last_run >= interval) and self.readback.has_free_buffer():
            self.count = count
            self.last_run = now
            self.readback.mark_written(self.start_measurement(count, softening, length_scale))
