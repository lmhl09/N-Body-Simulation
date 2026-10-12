import math

import pygame

from interface import (
    get_fonts, Panel, TextEdit, set_wanted_cursor, draw_icon, draw_panel, draw_button, draw_slider, format_grouped,
    format_time, parse_duration, get_time_unit,
    PANEL_BG, PANEL_BORDER, TEXT, DIM, ACCENT, WARNING, ERROR, BOX, BOX_BORDER, BOX_HOVER, TOGGLE_ON, TOGGLE_OFF,
    ICON_PLANET, ICON_STAR, ICON_BLACK_HOLE, ICON_PARTICLES, ICON_PLAY, ICON_PAUSE, ICON_RELOAD, ICON_SETTINGS,
    ICON_CLOSE,
)

#the speed is simulated seconds per real second
#the slider reaches from real time up to about 3 billion years per second, typed speeds may go a bit further
SPEED_MIN, SPEED_MAX = 1.0, 1e17
TYPED_MIN, TYPED_MAX = 0.01, 1e18

FPS_SHOWN_MAX = 500


def get_first_text(texts):
    # Returns the first text of a dict that is not empty
    for text in texts.values():
        if text: return text
    return ""


class Toolbar(Panel):
    # The bar at the bottom: pause, speed, clock, scene list, trails, placing tools, body count, fps, settings
    # the line above the bar (hints, notices, input errors) belongs to it too

    HEIGHT = 44
    MARGIN = 12
    REFRESH = 0.25              #the numbers in the bar get drawn new this often (s)
    LIST_ROW = 26
    NOTICE_TIME = 3.0           #s

    #name = how a placed body is called, placed and ring = how the tool is called in the message line
    PLACING_TOOLS = {
        "planet": {"icon": ICON_PLANET, "name": "Neuer Planet", "placed": "neuer Planet", "ring": "Planeten"},
        "star": {"icon": ICON_STAR, "name": "Neuer Stern", "placed": "neuer Stern", "ring": "Sternen"},
        "blackhole": {"icon": ICON_BLACK_HOLE, "name": "Neues Schwarzes Loch", "placed": "neues Schwarzes Loch",
                      "ring": "Schwarzen Löchern"},
        "particles": {"icon": ICON_PARTICLES, "name": None, "placed": "Partikelwolke", "ring": "Partikeln"},
    }

    def __init__(self, engine, scenes, scene_name, speed):
        self.engine = engine
        self.fonts = get_fonts()
        self.font = self.fonts.bar
        self.font_small = self.fonts.small

        self.scenes = dict(scenes)      #name -> title
        self.scene_name = scene_name

        self.speed = float(speed)
        self.paused = False
        self.sim_time = 0.0
        self.show_trails = True
        self.too_fast = False           #the sub steps are longer than the fastest orbit allows (main.py tells)

        self.fps = 0.0
        self.body_count = 0
        self.particle_count = 0

        self.calculate_layout()
        self.settings = Settings(self)

        #a scene picked in the list, the main loop loads it
        self.requested_scene = None

        self.armed = None               #placing tool that is picked right now
        self.tool_hint = ""
        self.notice = ""
        self.notice_until = 0.0
        self.errors = {}                #input errors of other windows (window -> text)
        self.readouts = {}              #what the mouse points at in another window (window -> text)

        self.list_open = False
        self.edit = None                #TextEdit while the speed gets typed
        self.message = ""               #error of the typed speed
        self.slider_dragging = False
        self.hover = None
        self.dirty = True
        self.last_draw = -1.0
        self.drawn = []

    ### LAYOUT ###

    def calculate_layout(self):
        width, height = self.engine.width, self.engine.height
        self.rect = pygame.Rect(self.MARGIN, height - self.HEIGHT, width - 2 * self.MARGIN, self.HEIGHT)
        y = self.rect.centery

        #from left to right: (key, width, height), None = a divider line
        left_side = [("pause", 32, 30), ("slider", 130, 30), ("speed", 128, 28), ("clock", 122, 30), None,
                     ("scene", 165, 30), ("reload", 30, 30), ("trails", 58, 26), None]
        for key in self.PLACING_TOOLS: left_side.append(("create:" + key, 30, 30))

        self.boxes = {}
        self.dividers = []
        x = self.rect.x + 8
        for element in left_side:
            if element is None:
                self.dividers.append(x)
                x += 9
                continue
            key, box_width, box_height = element
            self.boxes[key] = pygame.Rect(x, y - box_height // 2, box_width, box_height)
            x += box_width + 8

        #body count, fps and settings sit at the right end
        x = self.rect.right - 8 - (118 + 8 + 80 + 8 + 30)
        self.dividers.append(x - 5)
        for key, box_width in (("bodies", 118), ("fps", 80), ("settings", 30)):
            self.boxes[key] = pygame.Rect(x, y - 15, box_width, 30)
            x += box_width + 8

        self.hint_rect = pygame.Rect(self.rect.x + 2, self.rect.y - 26, 720, 22)

        #the scene list opens upwards from its button
        scene_box = self.boxes["scene"]
        rows = len(self.scenes)
        self.list_rect = pygame.Rect(scene_box.x, self.rect.y - 8 - rows * self.LIST_ROW - 12,
                                     max(scene_box.width, 220), rows * self.LIST_ROW + 12)
        self.list_rows = {}
        row_y = self.list_rect.y + 6
        for name in self.scenes:
            self.list_rows[name] = pygame.Rect(self.list_rect.x + 6, row_y, self.list_rect.width - 12, self.LIST_ROW - 2)
            row_y += self.LIST_ROW

    @property
    def area(self):
        # Everything the bar covers right now (the bar itself first)
        rects = [self.rect]
        if self.list_open: rects.append(self.list_rect)
        if self.settings.open: rects.append(self.settings.rect)
        return rects

    def is_inside(self, position):
        for rect in self.area:
            if rect.collidepoint(position): return True
        return False

    ### USED BY THE MAIN LOOP AND THE OTHER WINDOWS ###

    def toggle_pause(self):
        self.paused = not self.paused
        self.dirty = True

    def change_speed(self, factor):
        self.set_speed(self.speed * factor)

    def set_speed(self, speed, low=SPEED_MIN, high=SPEED_MAX):
        self.speed = min(high, max(low, float(speed)))
        self.dirty = True

    def advance(self, sim_seconds):
        self.sim_time += sim_seconds

    def scene_loaded(self, name):
        self.scene_name = name
        self.sim_time = 0.0
        self.dirty = True

    def update_stats(self, fps, body_count, particle_count, too_fast):
        if too_fast != self.too_fast: self.dirty = True
        self.fps = fps
        self.body_count = body_count
        self.particle_count = particle_count
        self.too_fast = too_fast

    def force_redraw(self):
        self.dirty = True

    def arm(self, kind):
        # Picks a placing tool, a second click on it drops it again
        if kind == self.armed: self.armed = None
        else: self.armed = kind
        #bodies get placed onto a standing picture
        if self.armed is not None: self.paused = True
        self.dirty = True

    def notify(self, text, now):
        # Shows a notice for a few seconds
        self.notice = text
        self.notice_until = now + self.NOTICE_TIME
        self.dirty = True

    def set_tool_hint(self, text):
        if text != self.tool_hint:
            self.tool_hint = text
            self.dirty = True

    def set_readout(self, owner, text):
        if self.readouts.get(owner, "") != text:
            self.readouts[owner] = text
            self.dirty = True

    def set_error(self, owner, text):
        if self.errors.get(owner, "") != text:
            self.errors[owner] = text
            self.dirty = True

    ### SPEED SLIDER ###
    #the slider is logarithmic: every tick is ten times faster than the one before

    def get_slider_ends(self):
        box = self.boxes["slider"]
        return box.x + 8, box.right - 8

    def get_knob_x(self):
        left, right = self.get_slider_ends()
        fraction = (math.log10(min(SPEED_MAX, max(SPEED_MIN, self.speed))) - math.log10(SPEED_MIN)) \
            / (math.log10(SPEED_MAX) - math.log10(SPEED_MIN))
        return left + fraction * (right - left)

    def set_speed_from_slider(self, x):
        left, right = self.get_slider_ends()
        fraction = min(1.0, max(0.0, (x - left) / (right - left)))
        exponent = math.log10(SPEED_MIN) + fraction * (math.log10(SPEED_MAX) - math.log10(SPEED_MIN))

        #rounded to two digits, else the slider gives speeds like 3,47 Tage / s
        speed = 10 ** exponent
        magnitude = 10 ** math.floor(math.log10(speed) - 1)
        self.set_speed(round(speed / magnitude) * magnitude)

    ### INPUT ###

    def handle_event(self, event):
        # Returns True if the event was meant for the bar (then nothing else gets it)
        if self.settings.open and self.settings.handle_event(event): return True

        if event.type == pygame.MOUSEMOTION:
            if self.slider_dragging:
                if event.buttons[0]: self.set_speed_from_slider(event.pos[0])
                else: self.slider_dragging = False
            if self.edit is not None and self.edit.dragging:
                if event.buttons[0]: self.edit.mouse_drag(event.pos[0], self.boxes["speed"])
                else: self.edit.mouse_up()
                self.dirty = True

            hover = self.get_element_at(event.pos)
            if hover != self.hover:
                self.hover = hover
                self.dirty = True
            self.update_cursor()
            return False

        if event.type == pygame.MOUSEWHEEL:
            position = pygame.mouse.get_pos()
            if self.boxes["slider"].collidepoint(position) or self.boxes["speed"].collidepoint(position):
                if event.y > 0: self.change_speed(2.0)
                else: self.change_speed(0.5)
                return True
            return self.is_inside(position)

        if event.type == pygame.MOUSEBUTTONDOWN:
            inside = self.is_inside(event.pos)
            element = self.get_element_at(event.pos)

            #a click anywhere else ends typing and closes the scene list
            if self.edit is not None and element != "speed": self.end_edit()
            if self.list_open and not (element or "").startswith("scene"):
                self.list_open = False
                self.dirty = True

            if not inside or event.button != 1: return inside
            self.click(element, event.pos)
            return True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.slider_dragging = False
                if self.edit is not None: self.edit.mouse_up()
            return self.is_inside(event.pos)

        #while the speed gets typed all keys go into the text
        if self.edit is not None:
            if event.type == pygame.TEXTINPUT:
                self.edit.insert(event.text)
                self.dirty = True
                return True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER): self.apply_typed_speed()
                elif event.key == pygame.K_ESCAPE: self.close_edit()
                else: self.edit.handle_key(event)
                self.dirty = True
                return True
            if event.type == pygame.KEYUP:
                return True

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.list_open:
            self.list_open = False
            self.dirty = True
            return True

        return False

    def get_element_at(self, position):
        # Returns the key of the element under the mouse ("pause", "scene:solar", "create:star" ...) or None
        if self.list_open and self.list_rect.collidepoint(position):
            for name, rect in self.list_rows.items():
                if rect.collidepoint(position): return "scene:" + name
            return "scene-list"

        if not self.rect.collidepoint(position): return None
        for key, rect in self.boxes.items():
            if rect.collidepoint(position): return key
        return None

    def update_cursor(self):
        if self.hover is None or self.hover in ("clock", "bodies", "fps", "scene-list"): cursor = None
        elif self.hover == "speed": cursor = pygame.SYSTEM_CURSOR_IBEAM
        else: cursor = pygame.SYSTEM_CURSOR_HAND
        set_wanted_cursor("toolbar", cursor)

    def click(self, element, position):
        if element == "pause":
            self.toggle_pause()
        elif element == "slider":
            self.slider_dragging = True
            self.set_speed_from_slider(position[0])
        elif element == "speed":
            if self.edit is not None:
                shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                self.edit.mouse_down(position[0], self.boxes["speed"], extend=shift)
            else:
                self.start_edit()
        elif element == "scene":
            self.list_open = not self.list_open
        elif element is not None and element.startswith("scene:"):
            self.list_open = False
            self.requested_scene = element[6:]
        elif element == "reload":
            self.requested_scene = self.scene_name
        elif element is not None and element.startswith("create:"):
            self.arm(element[7:])
        elif element == "trails":
            self.show_trails = not self.show_trails
        elif element == "settings":
            self.settings.toggle()
        self.dirty = True

    ### TYPING THE SPEED ###

    def start_edit(self):
        #a number without a unit counts in the unit the speed is shown in right now
        self.edit_unit = get_time_unit(self.speed)[0]
        self.edit = TextEdit(format_time(self.speed), self.font)
        self.message = ""

    def apply_typed_speed(self):
        # Returns False if the text is no speed (the box stays open then)
        if self.edit is None: return True
        try:
            speed = parse_duration(self.edit.text, self.edit_unit)
        except ValueError:
            self.message = "Tempo nicht lesbar. Beispiel: 2 Tage, 1 Jahr, 3600 s, 5 Mio. Jahre"
            self.dirty = True
            return False
        self.set_speed(speed, TYPED_MIN, TYPED_MAX)
        self.close_edit()
        return True

    def close_edit(self):
        if self.edit is not None: self.edit.close()
        self.edit = None
        self.message = ""
        self.dirty = True

    def end_edit(self):
        # After a click somewhere else: a readable speed counts, everything else gets thrown away
        if self.edit is not None and not self.apply_typed_speed(): self.close_edit()

    ### DRAWING ###

    def render(self, surface, now):
        # Draws the bar if something changed, returns the areas of the surface that have to go to the gpu
        if self.notice and now >= self.notice_until:
            self.notice = ""
            self.dirty = True

        #while typing or dragging the bar has to follow at once, else a few times per second are enough
        if self.edit is not None or self.slider_dragging: wait = 1.0 / 30.0
        else: wait = self.REFRESH
        if not self.dirty and now - self.last_draw < wait: return []

        self.dirty = False
        self.last_draw = now

        areas = [self.rect, self.hint_rect]
        if self.list_open: areas.append(self.list_rect)
        if self.settings.open: areas.append(self.settings.rect)

        #what was drawn last time has to be cleared too (a closed list would stay visible else)
        regions = self.drawn + areas
        for region in regions: surface.fill((0, 0, 0, 0), region)

        self.draw_message_line(surface)
        self.draw_bar(surface)
        if self.list_open: self.draw_scene_list(surface)
        if self.settings.open: self.settings.draw(surface)

        self.drawn = [area.copy() for area in areas]
        return regions

    def draw_box_button(self, surface, key, on=False, radius=8):
        rect = self.boxes[key]
        draw_button(surface, rect, on, self.hover == key, radius)
        return rect

    def draw_message_line(self, surface):
        # One line above the bar, by importance: input error, hint of the active tool, notice, info from the stability map
        error = self.message or get_first_text(self.errors)
        readout = get_first_text(self.readouts)

        #the open scene list covers the line, only errors still get shown
        if self.list_open and not error: return

        if error:
            text, color = error, ERROR
        elif self.tool_hint:
            text, color = self.tool_hint, ACCENT
        elif self.notice or readout:
            text, color = self.notice or readout, TEXT
        else:
            return

        rendered = self.fonts.render(text, color, self.font_small)
        plate = rendered.get_rect(bottomleft=(self.hint_rect.x + 8, self.hint_rect.bottom - 2)).inflate(12, 4)
        plate.width = min(plate.width, self.hint_rect.width)
        pygame.draw.rect(surface, (14, 18, 30, 200), plate, border_radius=6)
        surface.set_clip(plate.inflate(-6, 0))
        surface.blit(rendered, rendered.get_rect(midleft=(plate.x + 6, plate.centery)))
        surface.set_clip(None)

    def draw_bar(self, surface):
        #the bar is only round at the top, its lower edge is outside the window
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_top_left_radius=12, border_top_right_radius=12)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect.inflate(0, 4).move(0, 2), width=1,
                         border_top_left_radius=12, border_top_right_radius=12)
        for x in self.dividers:
            pygame.draw.line(surface, BOX_BORDER, (x, self.rect.y + 10), (x, self.rect.bottom - 10))

        self.draw_time_controls(surface)
        self.draw_scene_controls(surface)
        self.draw_counters(surface)

    def draw_time_controls(self, surface):
        #pause button, orange while paused
        rect = self.boxes["pause"]
        if self.paused:
            draw_button(surface, rect, hovered=self.hover == "pause", radius=8, fill=(150, 105, 40))
            draw_icon(surface, ICON_PLAY, rect.center, TEXT)
        else:
            draw_button(surface, rect, hovered=self.hover == "pause", radius=8, fill=None)
            draw_icon(surface, ICON_PAUSE, rect.center, TEXT)

        self.draw_speed_slider(surface)

        rect = self.boxes["speed"]
        if self.edit is not None:
            pygame.draw.rect(surface, BOX, rect, border_radius=6)
            if self.message: pygame.draw.rect(surface, ERROR, rect, width=1, border_radius=6)
            else: pygame.draw.rect(surface, ACCENT, rect, width=1, border_radius=6)
            self.edit.draw(surface, rect)
        else:
            if self.hover == "speed": border = BOX_HOVER
            else: border = BOX_BORDER
            pygame.draw.rect(surface, BOX, rect, border_radius=6)
            pygame.draw.rect(surface, border, rect, width=1, border_radius=6)
            if self.paused:
                self.draw_text(surface, "Pausiert", (rect.x + 8, rect.centery), WARNING, anchor="midleft")
            else:
                #orange = the simulation cant keep up with this speed anymore (close orbits get wrong)
                if self.too_fast: color = WARNING
                else: color = TEXT
                label = self.draw_text(surface, format_time(self.speed), (rect.x + 8, rect.centery), color, anchor="midleft")
                self.draw_text(surface, "/ s", (label.right + 4, rect.centery), DIM, anchor="midleft")

        #simulated time since the scene started
        rect = self.boxes["clock"]
        self.draw_text(surface, "T+", (rect.x, rect.centery), DIM, anchor="midleft")
        if self.sim_time > 0: clock_text = format_time(self.sim_time)
        else: clock_text = "0 s"
        self.draw_text(surface, clock_text, (rect.x + 22, rect.centery), TEXT, anchor="midleft")

    def draw_scene_controls(self, surface):
        #scene button with a small triangle
        rect = self.draw_box_button(surface, "scene", on=self.list_open)
        surface.set_clip(pygame.Rect(rect.x + 8, rect.y, rect.width - 30, rect.height))
        self.draw_text(surface, self.scenes.get(self.scene_name, self.scene_name), (rect.x + 10, rect.centery),
                       TEXT, anchor="midleft")
        surface.set_clip(None)
        tip = (rect.right - 12, rect.centery)
        pygame.draw.polygon(surface, DIM, [(tip[0] - 4, tip[1] + 2), (tip[0] + 4, tip[1] + 2), (tip[0], tip[1] - 3)])

        rect = self.draw_box_button(surface, "reload")
        draw_icon(surface, ICON_RELOAD, rect.center, TEXT)

        rect = self.draw_box_button(surface, "trails", on=self.show_trails, radius=12)
        if self.show_trails: self.draw_text(surface, "Spuren", rect.center, TEXT, self.font_small, anchor="center")
        else: self.draw_text(surface, "Spuren", rect.center, DIM, self.font_small, anchor="center")

        for key, tool in self.PLACING_TOOLS.items():
            rect = self.draw_box_button(surface, "create:" + key, on=self.armed == key)
            draw_icon(surface, tool["icon"], rect.center)

    def draw_counters(self, surface):
        #body count, with the mouse on it the particle count
        rect = self.boxes["bodies"]
        if self.hover == "bodies": count, word = self.particle_count, "Partikel"
        else: count, word = self.body_count, "Körper"
        word_rect = self.draw_text(surface, word, (rect.right, rect.centery), DIM, anchor="midright")
        self.draw_text(surface, format_grouped(count), (word_rect.left - 5, rect.centery), TEXT, anchor="midright")

        #fps as number and as a small bar (green from 120, orange from 30, red below)
        rect = self.boxes["fps"]
        fps = min(self.fps, FPS_SHOWN_MAX)
        if self.fps >= FPS_SHOWN_MAX: text = f"{FPS_SHOWN_MAX}+"
        else: text = f"{fps:.0f}"
        if fps >= 120: color = (120, 220, 140)
        elif fps >= 30: color = WARNING
        else: color = ERROR
        label = self.draw_text(surface, text, (rect.x + 4, rect.centery - 5), TEXT, anchor="midleft")
        self.draw_text(surface, "FPS", (label.right + 4, rect.centery - 5), DIM, self.font_small, anchor="midleft")
        meter = pygame.Rect(rect.x + 4, rect.centery + 7, rect.width - 8, 4)
        pygame.draw.rect(surface, TOGGLE_OFF, meter, border_radius=2)
        pygame.draw.rect(surface, color, (meter.x, meter.y, max(2, meter.width * fps / FPS_SHOWN_MAX), meter.height),
                         border_radius=2)

        rect = self.draw_box_button(surface, "settings", on=self.settings.open)
        draw_icon(surface, ICON_SETTINGS, rect.center, TEXT)

    def draw_speed_slider(self, surface):
        left, right = self.get_slider_ends()
        low, high = math.log10(SPEED_MIN), math.log10(SPEED_MAX)

        #one tick per power of ten
        ticks = []
        for exponent in range(int(low), int(high) + 1):
            ticks.append(left + (exponent - low) / (high - low) * (right - left))

        if self.paused: color = DIM
        else: color = ACCENT
        draw_slider(surface, left, right, self.boxes["slider"].centery, self.get_knob_x(), ticks,
                    active=self.hover == "slider" or self.slider_dragging, color=color)

    def draw_scene_list(self, surface):
        draw_panel(surface, self.list_rect, 10)
        for name, rect in self.list_rows.items():
            if self.hover == "scene:" + name: pygame.draw.rect(surface, BOX_HOVER, rect, border_radius=6)
            elif name == self.scene_name: pygame.draw.rect(surface, TOGGLE_ON, rect, border_radius=6)
            self.draw_text(surface, self.scenes[name], (rect.x + 10, rect.centery), TEXT, anchor="midleft")


#the fps caps the slider offers (0 = no cap) and the sizes a ring of bodies can have
FPS_CAPS = (30, 60, 90, 120, 144, 165, 240, 360, 0)
RING_BODY_COUNTS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000)

SETTINGS_DEFAULTS = {"fps_cap": 0, "selection_arrows": 3, "collisions": True, "ring_bodies": 20000}


class Settings():
    # The settings window, every setting is a slider with fixed stops or a button that switches through its values

    WIDTH = 320
    ROW = 32            #height of a row with a button
    SLIDER_ROW = 52     #height of a row with a slider
    FOOTER = 10

    #key, name, "slider" or "pill" (the button), the values it can have
    OPTIONS = (
        ("fps_cap", "Bildratenbegrenzung", "slider", FPS_CAPS),
        ("selection_arrows", "Kraftpfeile bei Auswahl", "slider", tuple(range(11))),
        ("collisions", "Kollisionen", "pill", (True, False)),
        ("ring_bodies", "Körper pro Ring", "slider", RING_BODY_COUNTS),
    )

    def __init__(self, toolbar):
        self.toolbar = toolbar
        self.values = dict(SETTINGS_DEFAULTS)
        self.open = False
        self.hover = None
        self.dragging = None            #key of the slider that is held
        self.free_width = toolbar.engine.width
        self.calculate_layout()

    def calculate_layout(self):
        height = 48 + self.FOOTER
        for key, title, kind, choices in self.OPTIONS:
            if kind == "pill": height += self.ROW
            else: height += self.SLIDER_ROW

        #in the middle of the space the inspect window leaves free
        x = (self.free_width - self.WIDTH) // 2
        y = (self.toolbar.rect.y - height) // 2
        self.rect = pygame.Rect(x, y, self.WIDTH, height)
        self.close_rect = pygame.Rect(self.rect.right - 34, self.rect.y + 10, 24, 24)

        self.rows = {}          #the button or the slider line of every setting
        self.hits = {}          #where a click counts for it
        self.titles = {}        #y of its name
        row_y = self.rect.y + 48
        for key, title, kind, choices in self.OPTIONS:
            if kind == "pill":
                self.rows[key] = pygame.Rect(self.rect.right - 16 - 96, row_y, 96, 24)
                self.hits[key] = self.rows[key]
                self.titles[key] = row_y + 12
                row_y += self.ROW
            else:
                self.titles[key] = row_y + 10
                self.rows[key] = pygame.Rect(self.rect.x + 22, row_y + 24, self.WIDTH - 44, 20)
                self.hits[key] = self.rows[key].inflate(16, 0)
                row_y += self.SLIDER_ROW

    def place(self, free_width):
        if free_width != self.free_width:
            self.free_width = free_width
            self.calculate_layout()
            self.toolbar.dirty = True

    def toggle(self):
        # Opens or closes the window
        self.open = not self.open
        self.hover = None
        self.dragging = None
        set_wanted_cursor("settings", None)
        self.toolbar.dirty = True

    def label(self, key):
        value = self.values[key]
        if key == "fps_cap":
            if value == 0: return "Aus"
            return f"{value} FPS"
        if key == "selection_arrows":
            if value == 0: return "Aus"
            return str(value)
        if key == "ring_bodies": return format_grouped(value)
        if isinstance(value, bool):
            if value: return "An"
            return "Aus"
        return str(value)

    def get_choices(self, key):
        for option, title, kind, choices in self.OPTIONS:
            if option == key: return choices

    def get_stop(self, key):
        # Returns at which stop of its slider the setting is
        choices = self.get_choices(key)
        value = self.values[key]
        if value in choices: return choices.index(value)

        #a value between the stops counts as the closest one (0 means "off" and is the highest stop)
        value = value or math.inf
        best = 0
        best_distance = None
        for stop, choice in enumerate(choices):
            distance = abs((choice or math.inf) - value)
            if best_distance is None or distance < best_distance:
                best = stop
                best_distance = distance
        return best

    def set_stop(self, key, stop):
        choices = self.get_choices(key)
        value = choices[max(0, min(len(choices) - 1, stop))]
        if value != self.values[key]:
            self.values[key] = value
            self.toolbar.dirty = True

    def slide_to(self, key, x):
        rect = self.rows[key]
        last_stop = len(self.get_choices(key)) - 1
        self.set_stop(key, round((x - rect.x) / rect.width * last_stop))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.dragging is not None:
                if event.buttons[0]: self.slide_to(self.dragging, event.pos[0])
                else: self.dragging = None
                return True

            hover = None
            if self.close_rect.collidepoint(event.pos): hover = "close"
            for key, rect in self.hits.items():
                if rect.collidepoint(event.pos): hover = key
            if hover != self.hover:
                self.hover = hover
                self.toolbar.dirty = True
            if hover: set_wanted_cursor("settings", pygame.SYSTEM_CURSOR_HAND)
            else: set_wanted_cursor("settings", None)
            return False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.toggle()
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if event.button == 1: self.click(event.pos)
            return True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.dragging is not None:
                self.dragging = None
                self.toolbar.dirty = True
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEWHEEL:
            #the wheel moves a slider by one stop
            position = pygame.mouse.get_pos()
            for key, title, kind, choices in self.OPTIONS:
                if kind == "slider" and self.rows[key].inflate(0, 16).collidepoint(position):
                    if event.y > 0: self.set_stop(key, self.get_stop(key) + 1)
                    else: self.set_stop(key, self.get_stop(key) - 1)
                    return True
            return self.rect.collidepoint(position)

        return False

    def click(self, position):
        if self.close_rect.collidepoint(position):
            self.toggle()

        for key, title, kind, choices in self.OPTIONS:
            if not self.hits[key].collidepoint(position): continue
            if kind == "pill":
                #a button switches to its next value
                if self.values[key] in choices: index = choices.index(self.values[key])
                else: index = -1
                self.values[key] = choices[(index + 1) % len(choices)]
                self.toolbar.dirty = True
            else:
                self.dragging = key
                self.slide_to(key, position[0])

    def draw(self, surface):
        fonts = self.toolbar.fonts
        draw_panel(surface, self.rect)
        fonts.text(surface, "Einstellungen", (self.rect.x + 16, self.rect.y + 12), TEXT, fonts.title)

        if self.hover == "close": draw_icon(surface, ICON_CLOSE, self.close_rect.center, TEXT)
        else: draw_icon(surface, ICON_CLOSE, self.close_rect.center, DIM)

        for key, title, kind, choices in self.OPTIONS:
            rect = self.rows[key]
            fonts.text(surface, title, (self.rect.x + 16, self.titles[key]), DIM, fonts.body, anchor="midleft")
            if kind == "pill":
                if self.hover == key: pygame.draw.rect(surface, BOX_HOVER, rect, border_radius=12)
                else: pygame.draw.rect(surface, TOGGLE_OFF, rect, border_radius=12)
                fonts.text(surface, self.label(key), rect.center, TEXT, fonts.small, anchor="center")
            else:
                fonts.text(surface, self.label(key), (self.rect.right - 16, self.titles[key]), TEXT, fonts.body,
                           anchor="midright")
                self.draw_option_slider(surface, key)

    def draw_option_slider(self, surface, key):
        rect = self.rows[key]
        last_stop = len(self.get_choices(key)) - 1

        ticks = []
        for stop in range(last_stop + 1):
            ticks.append(rect.x + rect.width * stop / last_stop)

        draw_slider(surface, rect.x, rect.right, rect.centery, rect.x + rect.width * self.get_stop(key) / last_stop,
                    ticks, active=self.hover == key or self.dragging == key)
