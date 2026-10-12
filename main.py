import sys
import math
import time

import pygame
import moderngl

from engine import SimEngine
import shaders
import gpu_cso
from gpu_cso import (
    CircularSolarObject, TrailParticles, ParticleCluster,
    GRAVITATIONAL_CONSTANT, LIGHTSPEED,
    PLANET_BUFFER, TRAIL_BUFFER, TRAIL_META_BUFFER, TRAIL_RESET_BUFFER,
    PARTICLE_BUFFER, PARTICLE_GROUP_BUFFER,
)
from gpu_overlays import Overlays
from inspector import Inspector
from toolbar import Toolbar
from tools import Tools
from impacts import Impacts, StepLimit
from stability_map import StabilityMap
from scenes import load_scene, TITLES
from interface import Button

CSO = CircularSolarObject

WINDOW_DIMENSIONS = (1920, 1080)

#how much the gpu buffers can hold
MAX_PLANETS = 500000
MAX_PARTICLES = 2000000
MAX_PARTICLE_GROUPS = 64        #particle clusters that can hang on a body (rings)
MAX_TRAILS = 1024
UPDATE_CAPACITY = 4096

TRAIL_LENGTH = 1500             #points per trail
TRAIL_SAMPLE_INTERVAL = 2000.0  #simulated seconds between trail points

SIM_SPEED = 100000          #simulated seconds per real second a scene starts at
MAX_SUB_STEPS = 100
INTERACTION_BUDGET = 1.5e9  #force calculations (body x body) the gpu gets per frame, big scenes get less sub steps because of it
MAX_FRAME_TIME = 0.05       #a frame never counts longer than this, else the sim would jump after a lag
SOFTENING = 1000.0          #to prevent division by 0 for csos
PARTICLE_SOFTENING = 10000.0    #to prevent division by 0 for particles

SCENE = "solar"     #the scene the program starts with

Engine = SimEngine()
opengl_context = Engine.WindowSetup(window_dimensions=WINDOW_DIMENSIONS, title="N-Körper-Simulation")

#one program each for bodies, trails and particles
shader_program = Engine.SetupShaderProgram(vert_shader=shaders.VERTEX_SHADER, frag_shader=shaders.FRAGMENT_SHADER)
trail_shader_program = Engine.SetupShaderProgram(shaders.TRAIL_VERTEX_SHADER, shaders.TRAIL_FRAGMENT_SHADER)
particle_program = Engine.SetupShaderProgram(shaders.PARTICLE_VERTEX_SHADER, shaders.PARTICLE_FRAGMENT_SHADER)

#the compute shaders do the physics, they are called like their source in shaders.py
for kernel in ("NBODY_FORCES", "NBODY_FORCES_SOFT", "NBODY_FORCES_CONTACT", "NBODY_FORCES_CONTACT_SOFT", "NBODY_MERGE",
               "NBODY_INTEGRATE", "PARTICLE_CALCULATION", "PARTICLE_DETACH", "PICK_CALCULATION", "TRAIL_RESET"):
    Engine.SetupNewComputeShader(compute_shader_name=kernel, compute_shader=getattr(shaders, kernel))

#reserves gpu memory buffers and binds them for the shaders
Engine.ReserveNewBuffer(
    buffer_name=PLANET_BUFFER, capacity=MAX_PLANETS, length_of_set=shaders.PLANET_STRIDE, binding_point=0,
    update_capacity=UPDATE_CAPACITY)
Engine.SetBodyBuffer(PLANET_BUFFER, position_lo_fields=(gpu_cso.F_POS_LO_X, gpu_cso.F_POS_LO_Y))   #tells the engine which buffer holds the bodies (needed to follow and click on bodies)
Engine.ReserveNewBuffer(
    buffer_name=TRAIL_BUFFER, capacity=MAX_TRAILS * TRAIL_LENGTH, length_of_set=2, binding_point=1)
Engine.ReserveNewBuffer(
    buffer_name=PARTICLE_BUFFER, capacity=MAX_PARTICLES, length_of_set=4, binding_point=2)
Engine.ReserveNewBuffer(
    buffer_name=PARTICLE_GROUP_BUFFER, capacity=MAX_PARTICLE_GROUPS, length_of_set=4, binding_point=12)
Engine.ReserveNewBuffer(
    buffer_name=TRAIL_META_BUFFER, capacity=MAX_TRAILS, length_of_set=8, binding_point=3, clear=True)
Engine.ReserveNewBuffer(
    buffer_name=TRAIL_RESET_BUFFER, capacity=MAX_TRAILS, length_of_set=2, binding_point=10)

#the vaos are empty, the shaders read everything out of the buffers themselves
vertex_array_object = opengl_context.vertex_array(shader_program, [])
vao_trails = opengl_context.vertex_array(trail_shader_program, [])
vao_particles = opengl_context.vertex_array(particle_program, [])

#only the last sub step of a frame writes a trail point, this uniform switches that on
integrate_write_trail = Engine.compute_shaders["NBODY_INTEGRATE"]["u_write_trail"]

overlay_surface = Engine.SetupOverlay()

gpu_cso.set_engine(Engine)
TrailParticles.setup(Engine, max_slots=MAX_TRAILS, length=TRAIL_LENGTH, sample_interval=TRAIL_SAMPLE_INTERVAL)
Overlays.setup(Engine)

step_limit = StepLimit(Engine)

### UI ###

Exit_Button = Button((45, 45), image_id=0)

Engine.ZOOM_SPEED = 2

selected_object = None


def select_body(obj, follow: bool = True):
    # Handels the selection of objects

    global selected_object
    if obj is selected_object: return

    if selected_object is not None: selected_object.set_locked(False)
    selected_object = obj
    Overlays.set_selected_object(obj)

    if obj is None:
        inspector.close()
        Engine.ReleaseAnchor()
    else:
        obj.set_locked(True)
        inspector.open(obj, follow)
        follow_selected_body(follow)


def follow_selected_body(follow: bool):
    # Handels following the selected body
    if follow and selected_object is not None: Engine.SetAnchor(id(selected_object))
    else: Engine.ReleaseAnchor()


def get_selected_body():
    return selected_object


def free_view_width():
    # Returns view width to handel the offset when opening the inspect window
    if not inspector.is_open: return Engine.width
    return Engine.width - inspector.rect.width - 2 * inspector.MARGIN


def free_view_height():
    # Returns the height of the view - the toolbar height / margin
    return toolbar.rect.y - Toolbar.MARGIN


def free_view_rect():
    # Returns the part of the window no panel covers.
    return pygame.Rect(0, 0, free_view_width(), toolbar.hint_rect.y)


def zoom_to_orbit(primary, reach: float):
    # Centres the camera on the primary with the selected body's whole orbit
    Engine.SetAnchor(id(primary))
    Engine.zoom = 0.45 * min(free_view_width(), free_view_height()) / max(reach, 1.0)


def release_deleted_body(body):
    # Removes selection tag when deleting an object
    if body is selected_object: select_body(None)


def show_input_error(text):
    # A wrong input in the inspect window gets explained in the message line of the toolbar
    toolbar.set_error("inspector", text)


CSO.delete_hooks.append(release_deleted_body)   #runs release_deleted_body every time a body gets deleted

toolbar = Toolbar(Engine, TITLES, SCENE, SIM_SPEED)

inspector = Inspector(Engine, on_select=select_body, on_follow=follow_selected_body, on_zoom_orbit=zoom_to_orbit,
                      on_error=show_input_error)

#the tools and the inspect window need each other (the orbit and velocity buttons start a tool)
tools = Tools(Engine, toolbar, inspector, select_body, free_view_rect)
inspector.tools = tools

stability_map = StabilityMap(Engine, inspector, toolbar, free_view_rect)

impacts = Impacts(Engine, MAX_PLANETS, toolbar, inspector, select_body, get_selected_body)


### SCENES ###

def rebuild_universe(name: str):
    # Switches the scenes (who would have thought)

    #throws away everything of the old scene
    tools.cancel()
    select_body(None)
    gpu_cso.delete_all_objects(Engine)
    impacts.scene_changed()
    Overlays.reset()
    step_limit.reset()

    zoom, speed = load_scene(name, toolbar.settings.values)

    #gives all possible objects a trail
    for obj in CSO.objects:
        if obj.has_trail: TrailParticles(obj)

    #sets default values
    Engine.ReleaseAnchor()
    inspector.stop_following()
    Engine.zoom = zoom
    Engine.camera_center = pygame.math.Vector2(0, 0)
    toolbar.set_speed(speed or SIM_SPEED)
    toolbar.scene_loaded(name)


def handle_shortcut(key):
    # Keys that nothing else used
    if key == pygame.K_DELETE and selected_object is not None:
        selected_object.delete()
    elif key == pygame.K_ESCAPE:
        select_body(None)
    elif key == pygame.K_SPACE:
        toolbar.toggle_pause()
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        toolbar.change_speed(2.0)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        toolbar.change_speed(0.5)
    elif key == pygame.K_t:
        toolbar.show_trails = not toolbar.show_trails
        toolbar.force_redraw()


rebuild_universe(SCENE)

last_stats_time = 0.0 #time of the last toolbar stats update (fps, body count)

clock = pygame.time.Clock()
previous_time = time.perf_counter()
running: bool = True

while running and __name__ == "__main__":

    Engine.frame += 1

    settings = toolbar.settings.values
    Overlays.set_selection_arrows(settings["selection_arrows"])
    clock.tick(settings["fps_cap"])     #0 = no cap

    now = time.perf_counter()
    real_delta_time = min(now - previous_time, MAX_FRAME_TIME) #dt
    previous_time = now

    #for events without a position (keys, mouse wheel) the place of the mouse counts
    pointer_over_ui = Engine.PointerOverUI()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        position = getattr(event, "pos", None)
        if position is not None: over_ui = Engine.PointerOverUI(position)
        else: over_ui = pointer_over_ui

        #the windows get the event one after the other until one of them used it
        consumed = inspector.handle_event(event)
        if not consumed: consumed = stability_map.handle_event(event)
        if consumed and event.type == pygame.MOUSEBUTTONDOWN: toolbar.end_edit()
        if not consumed: consumed = toolbar.handle_event(event)
        if not consumed: consumed = tools.handle_event(event, over_ui, now)

        if not consumed and event.type == pygame.KEYDOWN:
            handle_shortcut(event.key)

        #a left click into the view selects the body under the mouse (or deselects it if it was selected already)
        if not consumed and event.type == pygame.MOUSEBUTTONDOWN and not over_ui and event.button == 1:
            picked = CSO.get_body_at_screen(event.pos)
            if picked is selected_object and picked is not None: select_body(None)
            elif picked is not None: select_body(picked)

        # moves and handels the "camera"
        Engine.UpdateCamera(event=event, over_ui=over_ui or consumed)

    if toolbar.requested_scene is not None:
        rebuild_universe(toolbar.requested_scene)
        toolbar.requested_scene = None
    tools.update()


    ### GPU SIMULATION ###


    if toolbar.paused: frame_sim_time = 0.0
    else: frame_sim_time = real_delta_time * toolbar.speed
    collisions = settings["collisions"]

    #bodies only merge while the sim runs
    merging = collisions and frame_sim_time > 0.0

    #takes over the merges that happened on the gpu
    impacts.tick(now)

    #everything that was changed in python since the last frame goes to the gpu
    gpu_cso.send_changes_to_gpu(Engine)
    Overlays.upload_overlay_list()
    ParticleCluster.upload_groups()

    count = Engine.GetCurrentBufferCount(PLANET_BUFFER)
    particle_count = Engine.GetCurrentBufferCount(PARTICLE_BUFFER)

    TrailParticles.move_head(frame_sim_time)
    toolbar.advance(frame_sim_time)

    #length_scale shrinks very large distances so the gpu floats dont overflow
    #step_limit checks how long one sim step can be without close orbits breaking
    length_scale = gpu_cso.get_length_scale()
    step_limit.update(now, count, SOFTENING, length_scale)

    #the time of the frame gets cut into sub steps: enough of them that no step is longer than the limit,
    #but not more than the gpu can calculate in one frame
    interactions_per_step = count * (count + particle_count)
    affordable_steps = int(INTERACTION_BUDGET // max(interactions_per_step, 1))
    sub_steps = max(1, min(MAX_SUB_STEPS, affordable_steps, math.ceil(frame_sim_time / step_limit.max_step)))
    dt_per_step = frame_sim_time / sub_steps

    if count > 0 or particle_count > 0:

        #which of the four force shaders: with or without collisions, with or without ring bodies (softened)
        if collisions: forces = "NBODY_FORCES_CONTACT"
        else: forces = "NBODY_FORCES"
        if CSO.softened > 0: forces += "_SOFT"

        black_hole_count = 0
        if collisions: black_hole_count = impacts.upload_black_holes()

        Engine.SetComputeUniforms(forces, {
            "u_count": count,
            "u_G": GRAVITATIONAL_CONSTANT,
            "u_softening": SOFTENING,
            "u_length_scale": length_scale,
            "u_dt": dt_per_step,
            "u_black_hole_count": black_hole_count,
            "u_merge": int(merging),
            "u_c": float(LIGHTSPEED),
        })
        Engine.SetComputeUniforms("NBODY_MERGE", {
            "u_count": count,
            "u_G": GRAVITATIONAL_CONSTANT,
            "u_c": float(LIGHTSPEED),
        })
        Engine.SetComputeUniforms("NBODY_INTEGRATE", {
            "u_count": count,
            "u_dt": dt_per_step,
            "u_trail_length": TRAIL_LENGTH,
            "u_trail_head": TrailParticles.head,
        })
        Engine.SetComputeUniforms("PARTICLE_CALCULATION", {
            "u_planet_count": count,
            "u_count": particle_count,
            "u_group_count": ParticleCluster.group_count,
            "u_dt": dt_per_step,
            "u_G": GRAVITATIONAL_CONSTANT,
            "u_softening": PARTICLE_SOFTENING,
            "u_length_scale": length_scale,
        })

        for step in range(sub_steps):
            #forces on all bodies, then merges, then the particles, and only at the end the bodies move
            #(everything before that still needs the positions from the start of the step)
            Engine.DispatchCompute(forces, count, local_size=shaders.NBODY_LOCAL_SIZE)
            if merging: Engine.DispatchCompute("NBODY_MERGE", count)
            Engine.DispatchCompute("PARTICLE_CALCULATION", particle_count)

            if step == sub_steps - 1: integrate_write_trail.value = 1
            else: integrate_write_trail.value = 0
            Engine.DispatchCompute("NBODY_INTEGRATE", count)

    # centers camera view on width and height
    Engine.SetViewOffset(((free_view_width() - Engine.width) / 2, (free_view_height() - Engine.height) / 2),
                         real_delta_time)

    Engine.UploadCamera()
    Overlays.run_shaders()
    inspector.frame(frame_sim_time, now)
    stability_map.frame(now, toolbar.paused, dt_per_step)


    ### RENDERING ###


    opengl_context.clear(0.0, 0.0, 0.0)

    #from back to front: particles, trails, zones, bodies, markers and arrows
    if particle_count > 0:
        opengl_context.point_size = 1.0
        for cluster in ParticleCluster.objects:
            particle_program["u_first"].value = cluster.first_index
            particle_program["u_host"].value = cluster.get_host_index()
            particle_program["u_color"].value = cluster.rgba
            vao_particles.render(moderngl.POINTS, vertices=cluster.amount)

    trail_vertices, trail_first, trail_instances = TrailParticles.get_draw_range()
    if count > 0 and trail_instances > 0 and toolbar.show_trails:
        trail_shader_program["u_trail_length"].value = TRAIL_LENGTH
        trail_shader_program["u_trail_head"].value = TrailParticles.head
        trail_shader_program["u_total_recorded"].value = TrailParticles.total_recorded

        vao_trails.render(moderngl.LINE_STRIP, vertices=trail_vertices, first=trail_first, instances=trail_instances)

    Overlays.render_zones()
    if collisions: Overlays.render_roche_rings(count)

    if count > 0:
        #all bodies in one draw call, every body is one square the fragment shader paints a disc into
        vertex_array_object.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=count)

    Overlays.render_markers()
    Overlays.render_arrows()
    stability_map.render_preview()


    ### OVERLAY ###

    Engine.BeginUIFrame()

    Exit_Button.update()
    if Exit_Button.pressed: running = False

    #fps and body count for the toolbar (a few times per second are enough)
    if now - last_stats_time >= Toolbar.REFRESH:
        last_stats_time = now
        toolbar.update_stats(clock.get_fps(), count, particle_count, dt_per_step > step_limit.max_step * 1.001)

    #the settings window stays in the middle of the free space
    toolbar.settings.place(free_view_width())

    #draws the tool previews and redraws toolbar / exit button if a preview covered them
    view_regions = tools.render(overlay_surface)
    for region in view_regions:
        for rect in toolbar.area[1:]:
            if region.colliderect(rect): toolbar.force_redraw()
        if region.colliderect(Exit_Button.rectangle): Exit_Button.dirty = True

    #draws all ui parts and collects the areas that changed
    ui_regions = view_regions + inspector.render(overlay_surface, now) + stability_map.render(overlay_surface, now) \
        + toolbar.render(overlay_surface, now) + Exit_Button.render(overlay_surface)

    # handels ui click detection and avoiding unintended camera movement
    Engine.RegisterUIRect(Exit_Button.rectangle)
    if inspector.is_open: Engine.RegisterUIRect(inspector.rect)
    if stability_map.visible: Engine.RegisterUIRect(stability_map.rect)
    for rect in toolbar.area: Engine.RegisterUIRect(rect)

    Engine.PresentOverlay(ui_regions)

    pygame.display.flip()

pygame.quit()
sys.exit()
