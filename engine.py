import os
import struct

import numpy as np
import pygame
import moderngl

from shaders import OVERLAY_VERTEX_SHADER, OVERLAY_FRAGMENT_SHADER, update_shaders

#binding points the engine uses itself (all others are listed at the top of shaders.py)
CAMERA_BINDING = 0
PICK_BINDING = 4
UPDATE_BINDING = 7


class Readback():
    # A few small buffers a compute shader writes its result into
    # reading a result in the same frame makes the gpu wait, so every buffer gets read some frames later

    DELAY = 3       #frames between writing and reading

    def __init__(self, engine, size, binding, slots=4):
        self.engine = engine
        self.binding = binding
        self.buffers = []
        for i in range(slots):
            self.buffers.append(engine.ctx.buffer(reserve=size))

        self.waiting = []       #buffers that got written but not read yet: [slot, frame, note]
        self.slot = -1

    def has_free_buffer(self, kept=0):
        return len(self.waiting) + kept < len(self.buffers)

    def next_buffer(self, initial=None):
        # Binds the next buffer for a shader and returns its slot
        self.slot = (self.slot + 1) % len(self.buffers)
        buffer = self.buffers[self.slot]
        if initial is not None: buffer.write(initial)
        buffer.bind_to_storage_buffer(self.binding)
        return self.slot

    def mark_written(self, slot, note=None):
        self.waiting.append([slot, self.engine.frame, note])

    def get_finished(self):
        # Returns (buffer, note) of the oldest written buffer, None if it is still too new
        if len(self.waiting) == 0: return None

        slot, frame, note = self.waiting[0]
        if self.engine.frame - frame < self.DELAY: return None

        self.waiting.pop(0)
        return self.buffers[slot], note

    def forget_waiting(self):
        self.waiting = []


class SimEngine():

    def __init__(self):
        self.gpu_memory_buffers = {}
        self.compute_shaders = {}

        self.frame = 0      #gets counted up in the main loop

        #camera
        self.zoom = 1.0
        self.dragging = False
        self.ZOOM_SPEED = 1.1
        self.min_pixel_radius = 2.0     #bodies never get drawn smaller than this
        self.pick_radius = 6.0

        #the body the camera follows and the buffer it is in
        self.anchor_object_id = None
        self.body_buffer_name = None
        self.body_position_lo_fields = None

        #goes up whenever a record moves to another index, everything that saved indices checks it
        self.layout_version = 0

        self.vbo_quad = None
        self.readback_buffer = None
        self.ui_rects = []

    ### GPU BUFFERS ###

    def ReserveNewBuffer(self, buffer_name: str, capacity: int, length_of_set: int, binding_point: int,
                         update_capacity: int = 0, clear: bool = False):
        # Reserves a new gpu buffer and binds it to the given binding point

        buffer = self.ctx.buffer(reserve=capacity * length_of_set * 4)
        buffer.bind_to_storage_buffer(binding_point)
        if clear: buffer.clear()

        info = {
            "buffer": buffer,
            "count": 0,
            "capacity": capacity,
            "binding_point": binding_point,
            "length_of_set": length_of_set,
            "obj_id_to_idx": {},
            "idx_to_obj_id": {},
            "pending": {},      #changes that wait for the next flush (index -> change)
            "updater": None,
        }
        self.gpu_memory_buffers[buffer_name] = info

        if update_capacity > 0: self.CreateUpdater(info, update_capacity)

        return buffer

    def CreateUpdater(self, info, capacity):
        # A buffer with an updater can be changed record by record:
        # the changes get collected and two small shaders write them into the buffer (see FlushUpdates)
        stride = info["length_of_set"]
        gather_source, scatter_source = update_shaders(stride, info["binding_point"], UPDATE_BINDING)

        #one change as the shaders read it (struct Update in shaders.py)
        command = np.dtype([("header", "<i4", (4,)), ("payload", "<f4", (stride,))])

        info["updater"] = {
            "gather": self.ctx.compute_shader(gather_source),
            "scatter": self.ctx.compute_shader(scatter_source),
            "buffer": self.ctx.buffer(reserve=capacity * command.itemsize),
            "capacity": capacity,
            "dtype": command,
            "all_fields": (1 << stride) - 1,
        }

    def AddNewDataToBuffer(self, buffer_name: str, new_data):
        # Adds a new record to the end of the buffer and returns its index
        info = self.gpu_memory_buffers[buffer_name]
        index = info["count"]

        if index >= info["capacity"]:
            raise MemoryError(f"{buffer_name} is full ({info['capacity']} records)")

        record = np.asarray(new_data, dtype=np.float32).ravel()
        info["pending"][index] = {"copy_from": -1, "fields": info["updater"]["all_fields"], "values": record.copy()}
        info["count"] += 1

        return index

    def WriteRawToBuffer(self, buffer_name: str, data: bytes, byte_offset: int = 0):
        self.gpu_memory_buffers[buffer_name]["buffer"].write(data, offset=byte_offset)

    def RemoveDataFromBuffer(self, buffer_name: str, index: int):
        # Removes a record, the last record of the buffer moves into the gap
        info = self.gpu_memory_buffers[buffer_name]
        pending = info["pending"]
        last_index = info["count"] - 1

        moved = pending.pop(last_index, None)
        pending.pop(index, None)

        if index != last_index:
            if moved is None:
                #the last record has no open change: it gets copied the way it is on the gpu
                moved = {"copy_from": last_index, "fields": 0,
                         "values": np.zeros(info["length_of_set"], dtype=np.float32)}
            elif moved["copy_from"] < 0 and moved["fields"] != info["updater"]["all_fields"]:
                #only some of its fields were changed: the rest still has to come from the gpu
                moved["copy_from"] = last_index
            pending[index] = moved

        info["count"] -= 1
        self.layout_version += 1

    def AddObjectToBuffer(self, object_id: int, buffer_name: str, new_data):
        # Adds an object to the buffer and remembers its index

        index = self.AddNewDataToBuffer(buffer_name, new_data)

        self.gpu_memory_buffers[buffer_name]["obj_id_to_idx"][object_id] = index
        self.gpu_memory_buffers[buffer_name]["idx_to_obj_id"][index] = object_id

        return index

    def RemoveObjectFromBuffer(self, object_id: int, buffer_name: str):
        # Removes an object from the buffer and fixes the index tables

        info = self.gpu_memory_buffers[buffer_name]

        obj_id_to_idx = info["obj_id_to_idx"]
        idx_to_obj_id = info["idx_to_obj_id"]

        index = obj_id_to_idx[object_id]
        last_index = info["count"] - 1

        self.RemoveDataFromBuffer(buffer_name, index)

        #the object that was last sits at the freed index now
        if index != last_index:
            last_object_id = idx_to_obj_id[last_index]
            obj_id_to_idx[last_object_id] = index
            idx_to_obj_id[index] = last_object_id

        del obj_id_to_idx[object_id]
        del idx_to_obj_id[last_index]

    def WriteObjectFields(self, object_id: int, buffer_name: str, first_field: int, values):
        # Changes some fields of an object (the other fields keep what the gpu calculated)
        index = self.GetObjectIndex(object_id, buffer_name)
        if index is None: return

        info = self.gpu_memory_buffers[buffer_name]
        values = np.asarray(values, dtype=np.float32).ravel()

        if index not in info["pending"]:
            info["pending"][index] = {"copy_from": -1, "fields": 0,
                                      "values": np.zeros(info["length_of_set"], dtype=np.float32)}
        change = info["pending"][index]

        #every field has its own bit, only fields with a set bit get written
        change["values"][first_field:first_field + len(values)] = values
        change["fields"] |= ((1 << len(values)) - 1) << first_field

    def FlushUpdates(self, buffer_name: str):
        # Sends all collected changes of a buffer to the gpu
        info = self.gpu_memory_buffers[buffer_name]
        updater = info["updater"]
        pending = info["pending"]

        if updater is None or len(pending) == 0: return

        count = len(pending)
        if count > updater["capacity"]:
            updater["capacity"] = max(count, updater["capacity"] * 2)
            updater["buffer"].orphan(updater["capacity"] * updater["dtype"].itemsize)

        destinations, sources, fields, values = [], [], [], []
        for index, change in pending.items():
            destinations.append(index)
            sources.append(change["copy_from"])
            fields.append(change["fields"])
            values.append(change["values"])
        pending.clear()

        commands = np.zeros(count, dtype=updater["dtype"])
        commands["header"][:, 0] = destinations
        commands["header"][:, 1] = sources
        commands["header"][:, 2] = fields
        commands["payload"] = np.array(values)

        updater["buffer"].write(commands.tobytes())
        updater["buffer"].bind_to_storage_buffer(UPDATE_BINDING)

        groups = (count + 63) // 64

        #records that move have to be read before anything gets overwritten
        if max(sources) >= 0:
            updater["gather"]["u_update_count"].value = count
            updater["gather"].run(group_x=groups)
            self.ctx.memory_barrier()

        updater["scatter"]["u_update_count"].value = count
        updater["scatter"].run(group_x=groups)
        self.ctx.memory_barrier()

    def ResetBuffer(self, buffer_name: str):
        # Empties a buffer (a new scene gets loaded)
        info = self.gpu_memory_buffers[buffer_name]
        info["count"] = 0
        info["obj_id_to_idx"].clear()
        info["idx_to_obj_id"].clear()
        info["pending"].clear()
        if buffer_name == self.body_buffer_name: self.anchor_object_id = None
        self.layout_version += 1

    def GetCurrentBufferCount(self, buffer_name: str):
        return self.gpu_memory_buffers[buffer_name]["count"]

    def GetBufferCapacity(self, buffer_name: str):
        return self.gpu_memory_buffers[buffer_name]["capacity"]

    def GetObjectIndex(self, object_id: int, buffer_name: str):
        return self.gpu_memory_buffers[buffer_name]["obj_id_to_idx"].get(object_id)

    def GetObjectIdAtIndex(self, index: int, buffer_name: str):
        return self.gpu_memory_buffers[buffer_name]["idx_to_obj_id"].get(index)

    def ReadBufferBytes(self, buffer, size: int, offset: int = 0):
        # Reads a part of a buffer back from the gpu
        # (it gets copied into a small buffer first, reading the body buffer directly made every frame after it slow)
        if self.readback_buffer is None or self.readback_buffer.size < size:
            self.readback_buffer = self.ctx.buffer(reserve=max(size, 4096))
        self.ctx.copy_buffer(self.readback_buffer, buffer, size, read_offset=offset)
        return self.readback_buffer.read(size=size)

    #the three functions below are slow (the gpu has to finish all its work first), so they are only used after user input

    def ReadIndexData(self, buffer_name: str, index: int):
        info = self.gpu_memory_buffers[buffer_name]
        length_of_set = info["length_of_set"]

        self.FlushUpdates(buffer_name)

        raw = self.ReadBufferBytes(info["buffer"], length_of_set * 4, index * length_of_set * 4)
        return np.frombuffer(raw, dtype=np.float32)

    def ReadBufferRecords(self, buffer_name: str):
        # Returns all records of a buffer as one array (one row per record)
        info = self.gpu_memory_buffers[buffer_name]
        length_of_set = info["length_of_set"]
        count = info["count"]
        if count == 0: return np.zeros((0, length_of_set), dtype=np.float32)

        self.FlushUpdates(buffer_name)
        raw = self.ReadBufferBytes(info["buffer"], count * length_of_set * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(count, length_of_set)

    def ReadObjectData(self, object_id: int, buffer_name: str):
        index = self.GetObjectIndex(object_id, buffer_name)
        if index is None: return None
        return self.ReadIndexData(buffer_name, index)

    def AppendBlockToBuffer(self, buffer_name: str, block):
        # Writes many records at once to the end of a buffer (particles) and returns the index of the first one
        info = self.gpu_memory_buffers[buffer_name]
        length_of_set = info["length_of_set"]

        block_array = np.asarray(block, dtype=np.float32).reshape(-1, length_of_set)
        first_index = info["count"]
        last_index = first_index + len(block_array)

        if last_index > info["capacity"]:
            raise MemoryError(f"{buffer_name} is full ({info['capacity']} records)")

        if len(block_array): info["buffer"].write(block_array.tobytes(), offset=first_index * length_of_set * 4)
        info["count"] = last_index

        return first_index

    ### WINDOW, PROGRAMS, OVERLAY ###

    def WindowSetup(self, window_dimensions: tuple, title: str = "SimEngine"):
        # Initiates pygame and returns the opengl context

        self.width, self.height = window_dimensions[0], window_dimensions[1]

        pygame.init()   #initialieses pygame

        #if the window is as big as the screen it gets no border and starts in the top left corner
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        desktop = pygame.display.get_desktop_sizes()[0]
        if self.width >= desktop[0] and self.height >= desktop[1]:
            os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
            flags |= pygame.NOFRAME

        pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption(title)

        self.screen_center = pygame.math.Vector2(self.width / 2, self.height / 2)
        self.camera_center = pygame.math.Vector2(0, 0)
        self.mouse_position = pygame.math.Vector2(pygame.mouse.get_pos())

        self.ctx = moderngl.create_context()     #creates the moderngl / opengl context
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        #camera data for all shaders (block Camera in shaders.py)
        self.camera_buffer = self.ctx.buffer(reserve=64)
        self.camera_buffer.bind_to_uniform_block(CAMERA_BINDING)

        #the pick shader writes the clicked body in here
        self.pick_buffer = self.ctx.buffer(reserve=4)
        self.pick_buffer.bind_to_storage_buffer(PICK_BINDING)

        #pixels the view is moved by, so a followed body sits in the middle of the space the panels leave free
        self.view_offset = pygame.math.Vector2(0, 0)

        return self.ctx

    def SetupShaderProgram(self, vert_shader, frag_shader):
        # Compiles and returns the given vert- and frag_shaders as a shader program
        return self.ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)

    def SetupNewComputeShader(self, compute_shader_name: str, compute_shader):
        # Compiles the given compute shader and saves it under its name
        self.compute_shaders[compute_shader_name] = self.ctx.compute_shader(compute_shader)
        return self.compute_shaders[compute_shader_name]

    def CreateVBOSquareReference(self):
        # Creates and Returns the vbo which references the standard square (full screen)

        if self.vbo_quad is not None:
            return self.vbo_quad

        quad_vertices = np.array([
            -1.0, -1.0,  # Unten Links
             1.0, -1.0,  # Unten Rechts
            -1.0,  1.0,  # Oben Links
             1.0,  1.0   # Oben Rechts
        ], dtype='f4')

        self.vbo_quad = self.ctx.buffer(quad_vertices.tobytes())

        return self.vbo_quad

    def SetupOverlay(self):
        # Creates the pygame surface the ui gets drawn on and the texture it gets shown with
        self.overlay_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.overlay_texture = self.ctx.texture((self.width, self.height), 4)
        self.overlay_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.overlay_texture.swizzle = "BGRA"       #pygame saves its pixels as bgra
        self.overlay_texture.write(self.overlay_surface.get_view("1"))

        self.overlay_program = self.SetupShaderProgram(OVERLAY_VERTEX_SHADER, OVERLAY_FRAGMENT_SHADER)
        self.overlay_program["u_overlay"] = 0

        self.overlay_vao = self.ctx.vertex_array(
            self.overlay_program,
            [(self.CreateVBOSquareReference(), "2f", "in_quad_vertex")]
        )

        return self.overlay_surface

    def BeginUIFrame(self):
        self.ui_rects = []

    def PresentOverlay(self, regions=()):
        # Draws the ui over the frame (only the areas that changed get uploaded to the gpu before)
        for region in regions: self.UploadOverlayRegion(region)

        self.overlay_texture.use(0)
        self.overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def UploadOverlayRegion(self, region):
        rect = pygame.Rect(region).clip(self.overlay_surface.get_rect())
        if rect.width <= 0 or rect.height <= 0: return

        #pixels2d is sorted by column, the texture wants rows
        pixels = pygame.surfarray.pixels2d(self.overlay_surface)
        block = np.ascontiguousarray(pixels[rect.x:rect.right, rect.y:rect.bottom].T)
        del pixels

        self.overlay_texture.write(block.tobytes(), viewport=(rect.x, rect.y, rect.width, rect.height))

    ### INPUT AND CAMERA ###

    def RegisterUIRect(self, rect):
        # Clicks and the mouse wheel inside a registered rect dont reach the view
        self.ui_rects.append(pygame.Rect(rect))

    def PointerOverUI(self, position=None):
        if position is None: position = pygame.mouse.get_pos()

        for rect in self.ui_rects:
            if rect.collidepoint(position): return True
        return False

    def ScreenToWorld(self, screen_position, center=None):
        if center is None: center = self.camera_center
        return (pygame.math.Vector2(*screen_position) - self.screen_center - self.view_offset) / self.zoom \
            + pygame.math.Vector2(*center)

    def WorldToScreen(self, world_position, center=None):
        if center is None: center = self.camera_center
        return (pygame.math.Vector2(*world_position) - pygame.math.Vector2(*center)) * self.zoom \
            + self.screen_center + self.view_offset

    def SetViewOffset(self, target, real_delta_time: float):
        # Moves the view offset a bit closer to the target every frame
        target = pygame.math.Vector2(target)
        step = (target - self.view_offset) * min(1.0, real_delta_time * 12.0)
        if (target - self.view_offset).length() < 0.5: step = target - self.view_offset
        if step.length() == 0: return

        self.view_offset += step
        #a free camera moves along, so the picture itself stays where it is
        if self.anchor_object_id is None: self.camera_center += step / self.zoom

    def SetBodyBuffer(self, buffer_name: str, position_lo_fields: tuple = None):
        # Tells the engine which buffer holds the bodies (the camera can only follow something in there)
        self.body_buffer_name = buffer_name
        self.body_position_lo_fields = position_lo_fields

    def GetAnchorPosition(self):
        # Reads the position of the followed body back from the gpu
        record = self.ReadObjectData(self.anchor_object_id, self.body_buffer_name)
        if record is None: return None

        x, y = float(record[0]), float(record[1])
        if self.body_position_lo_fields is not None:
            x += float(record[self.body_position_lo_fields[0]])
            y += float(record[self.body_position_lo_fields[1]])

        return pygame.math.Vector2(x, y)

    def SetAnchor(self, object_id):
        self.anchor_object_id = object_id

    def ReleaseAnchor(self):
        # Stops following, the camera stays where the body was
        if self.anchor_object_id is None: return

        position = self.GetAnchorPosition()
        if position is not None: self.camera_center = position

        self.anchor_object_id = None

    def CurrentCameraCenter(self):
        if self.anchor_object_id is not None:
            position = self.GetAnchorPosition()
            if position is not None: return position

        return pygame.math.Vector2(self.camera_center)

    def UpdateCamera(self, event, over_ui: bool = False):
        # Handels the camera per event

        self.mouse_position = pygame.math.Vector2(pygame.mouse.get_pos())
        anchored = self.anchor_object_id is not None

        if event.type == pygame.MOUSEWHEEL and not over_ui:
            world_mouse_before = self.ScreenToWorld(self.mouse_position)

            if event.y > 0 : self.zoom *= self.ZOOM_SPEED
            else: self.zoom /= self.ZOOM_SPEED

            #a free camera zooms towards the mouse, a following one towards its body
            if not anchored:
                self.camera_center = world_mouse_before - (self.mouse_position - self.screen_center - self.view_offset) / self.zoom

        #checks if the left mouse button is down and tracks it through dragging
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and not over_ui:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        if event.type == pygame.MOUSEMOTION:
            if self.dragging and not anchored:
                delta_mouse_position = pygame.math.Vector2(event.rel)
                self.camera_center -= delta_mouse_position / self.zoom

    def UploadCamera(self):
        # Writes the camera data into the block every shader reads
        anchor = -1
        count = 0

        if self.body_buffer_name is not None:
            count = self.GetCurrentBufferCount(self.body_buffer_name)
            if self.anchor_object_id is not None:
                index = self.GetObjectIndex(self.anchor_object_id, self.body_buffer_name)
                if index is not None: anchor = index

        #the center gets split into a float and what the float cant hold (like the body positions)
        center_x = float(np.float32(self.camera_center.x))
        center_y = float(np.float32(self.camera_center.y))

        self.camera_buffer.write(struct.pack(
            "<6f2i4f",
            self.width, self.height,
            center_x, center_y,
            self.zoom, self.min_pixel_radius,
            anchor, count,
            self.view_offset.x, self.view_offset.y,
            self.camera_center.x - center_x, self.camera_center.y - center_y,
        ))

    ### COMPUTE SHADERS ###

    def SetComputeUniforms(self, compute_shader_name: str, uniforms: dict):
        # Sets the uniforms of a compute shader (the ones it doesnt have get skipped)
        compute_shader = self.compute_shaders[compute_shader_name]

        for name, value in uniforms.items():
            uniform = compute_shader.get(name, None)
            if uniform is not None: uniform.value = value

    def DispatchCompute(self, compute_shader_name: str, count: int = 0, groups: int = None, local_size: int = 256):
        # Runs a compute shader for count elements (or a fixed amount of work groups)
        if groups is None: groups = (count + local_size - 1) // local_size
        if groups <= 0: return

        self.compute_shaders[compute_shader_name].run(group_x=groups)

        #without the barrier the next shader could still see the old values
        self.ctx.memory_barrier()

    def RunComputeShader(self, compute_shader_name: str, count: int, uniforms: dict = None,
                         groups: int = None, local_size: int = 256):
        if uniforms: self.SetComputeUniforms(compute_shader_name, uniforms)
        self.DispatchCompute(compute_shader_name, count, groups, local_size)

    def PickBodyAtScreen(self, mouse_position):
        # Returns the id of the object under the mouse (or None), calculated on the gpu
        buffer_name = self.body_buffer_name
        self.FlushUpdates(buffer_name)

        count = self.GetCurrentBufferCount(buffer_name)
        if count <= 0: return None

        #the shader keeps the smallest value, so it starts with the biggest one
        self.pick_buffer.write(np.array([0xFFFFFFFF], dtype=np.uint32).tobytes())

        self.UploadCamera()
        self.RunComputeShader("PICK_CALCULATION", count, {
            "u_mouse": tuple(pygame.math.Vector2(mouse_position)),
            "u_pick_radius": self.pick_radius,
        })

        key = int(np.frombuffer(self.pick_buffer.read(), dtype=np.uint32)[0])
        if key == 0xFFFFFFFF: return None

        #the lower 20 bits are the index of the body
        return self.GetObjectIdAtIndex(key & 0xFFFFF, buffer_name)
