import math
import re
import time

import pygame

from gpu_cso import (
    AU_IN_METERS, LIGHT_YEAR, SUN_MASS, SUN_RADIUS, EARTH_MASS, EARTH_RADIUS, JUPITER_MASS,
    MINUTE, HOUR, DAY, WEEK, MONTH, YEAR,
)

#everything the ui windows have in common: colors, number formats, units, fonts, icons, text input

PANEL_BG = (20, 25, 40, 238)
PANEL_BORDER = (58, 70, 96)
TEXT = (226, 232, 240)
DIM = (140, 152, 176)
ACCENT = (120, 200, 255)
WARNING = (255, 190, 90)
ERROR = (255, 115, 115)
BOX = (13, 17, 29)
BOX_BORDER = (58, 70, 95)
BOX_HOVER = (96, 112, 145)
SELECTION = (46, 90, 140)
TOGGLE_ON = (40, 102, 150)
TOGGLE_OFF = (30, 36, 54)
PLOT_BG = (12, 16, 27)          #graphs and stability map
KNOB = (200, 208, 222)
RIM = (10, 12, 20)              #dark edge around knobs and markers

#the signs for earth and sun are missing in most fonts
SYMBOLS = "⊕☉"
PLAIN_UNITS = {"M⊕": "Erdmassen", "M☉": "Sonnenmassen", "R⊕": "Erdradien", "R☉": "Sonnenradien"}


### NUMBERS (german: decimal comma, space between thousands) ###

def with_comma(text):
    return text.replace(".", ",")


def format_number(value, spec):
    return with_comma(format(value, spec))


def format_grouped(value):
    # 384400 -> "384 400"
    return f"{value:,.0f}".replace(",", " ")


### UNITS ###

#what can be typed behind a number, german and english (factor = value in SI units)
UNITS = {
    "mass": {"kg": 1.0, "t": 1e3,
             "erdmasse": EARTH_MASS, "erdmassen": EARTH_MASS, "merde": EARTH_MASS, "m⊕": EARTH_MASS,
             "me": EARTH_MASS, "mearth": EARTH_MASS, "earth": EARTH_MASS,
             "jupitermasse": JUPITER_MASS, "jupitermassen": JUPITER_MASS, "mj": JUPITER_MASS,
             "mjup": JUPITER_MASS, "jupiter": JUPITER_MASS,
             "sonnenmasse": SUN_MASS, "sonnenmassen": SUN_MASS, "msonne": SUN_MASS, "m☉": SUN_MASS,
             "msun": SUN_MASS, "sun": SUN_MASS},
    "density": {"kg/m3": 1.0, "g/cm3": 1e3},
    "distance": {"m": 1.0, "km": 1e3, "ae": AU_IN_METERS, "au": AU_IN_METERS,
                 "lj": LIGHT_YEAR, "lichtjahr": LIGHT_YEAR, "lichtjahre": LIGHT_YEAR, "ly": LIGHT_YEAR},
    "speed": {"m/s": 1.0, "km/s": 1e3, "km/h": 1.0 / 3.6},
}

#shown when an input could not be read
EXAMPLES = {
    "mass": "5,97e24 kg, 1 Sonnenmasse",
    "density": "5515 kg/m3, 5,5 g/cm3",
    "distance": "384400 km, 1,5 AE",
    "speed": "1,02 km/s, 950 m/s",
}

#1.5 or .5 or 5e24
NUMBER_PATTERN = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?"

#a number, maybe a power of ten behind it (*10^24 or x10**24), the rest is the unit
VALUE_PATTERN = re.compile(r"^(" + NUMBER_PATTERN + r")(?:(?:\*|x)10(?:\^|\*\*)([-+]?\d+))?(.*)$")


def clean_typed_text(text):
    # "5,97e24 kg" -> "5.97e24kg"
    compact = "".join(text.split()).lower()
    #a comma is the decimal sign, unless there is a point too (then the commas only group the thousands)
    if "," in compact and "." not in compact: return compact.replace(",", ".")
    return compact.replace(",", "")


def clean_unit(unit):
    # "kg/m³" -> "kg/m3"
    return unit.lower().replace(" ", "").replace("^3", "3").replace("³", "3").replace("_", "")


def parse_value_with_unit(text, kind, default_unit=""):
    # Turns typed text into a value in SI units, kind = "mass", "density", "distance" or "speed"
    # without a unit the one the field showed before counts, ValueError if the text makes no sense
    match = VALUE_PATTERN.match(clean_typed_text(text))
    if match is None: raise ValueError(text)

    value = float(match.group(1))
    if match.group(2): value *= 10.0 ** int(match.group(2))

    unit = clean_unit(match.group(3)) or clean_unit(default_unit)
    factors = UNITS[kind]
    if unit not in factors: raise ValueError(text)

    value *= factors[unit]
    if not math.isfinite(value): raise ValueError(text)
    return value


#km for moons, AE for planets, Lj for everything further out
DISTANCE_UNITS = ("km", "AE", "Lj")
DISTANCE_RANGES = {"km": (0.0, 0.02 * AU_IN_METERS),
                   "AE": (0.02 * AU_IN_METERS, 1e4 * AU_IN_METERS),
                   "Lj": (1e4 * AU_IN_METERS, math.inf)}


def get_distance_unit(meters, current=None):
    # Picks the unit for a distance
    # the unit that is shown right now gets kept a bit beyond its range, else it would flicker at the border
    meters = abs(meters)
    if current in DISTANCE_RANGES:
        low, high = DISTANCE_RANGES[current]
        if 0.8 * low <= meters < 1.25 * high: return current

    for unit in DISTANCE_UNITS:
        if meters < DISTANCE_RANGES[unit][1]: return unit
    return "Lj"


#the format functions return the number and its unit separately (the unit gets drawn in another color)

def format_distance(meters, unit=None):
    if unit is None: unit = get_distance_unit(meters)
    if unit == "km": return format_grouped(meters / 1e3), "km"
    if unit == "Lj": return format_number(meters / LIGHT_YEAR, ".5g"), "Lj"
    return format_number(meters / AU_IN_METERS, ".5g"), "AE"


def format_acceleration(value):
    if abs(value) < 1e6: return format_number(value, ".5g")
    return format_number(value, ".4g")


def format_scientific(value):
    # Two digits, "4,9e9" instead of pythons "4,9e+09"
    text = format_number(value, ".2g")
    if "e" not in text: return text
    mantissa, exponent = text.split("e")
    return f"{mantissa}e{int(exponent)}"


def format_speed(speed):
    if abs(speed) < 1e3: return format_number(speed, ".1f"), "m/s"
    return format_number(speed / 1e3, ".3f"), "km/s"


def format_mass(kilograms, relative=True, solar=False):
    # Stars and black holes in sun masses, planets in earth masses
    # kg if relative is switched off or the body is far too light for these units
    if solar: unit_mass = SUN_MASS
    else: unit_mass = EARTH_MASS

    if not relative or kilograms < 1e-6 * unit_mass: return format_number(kilograms, ".5g"), "kg"
    if solar: return format_number(kilograms / SUN_MASS, ".5g"), "M☉"
    return format_number(kilograms / EARTH_MASS, ".5g"), "M⊕"


def format_radius(meters, relative=True, solar=False):
    if not relative: return format_distance(meters, "km")
    if solar: return format_number(meters / SUN_RADIUS, ".4g"), "R☉"
    return format_number(meters / EARTH_RADIUS, ".4g"), "R⊕"


def format_duration(seconds):
    # Short form for the inspect window: h, d and a (annus = year)
    if seconds < 2 * DAY: return f"{format_number(seconds / 3600, '.3g')} h"
    if seconds < 2 * YEAR: return f"{format_number(seconds / DAY, '.4g')} d"
    if seconds < 1e4 * YEAR: return f"{format_number(seconds / YEAR, '.4g')} a"
    if seconds < 1e6 * YEAR: return f"{format_scientific(seconds / YEAR)} a"
    if seconds < 1e9 * YEAR: return f"{format_number(seconds / (1e6 * YEAR), '.3g')} Mio. a"
    return f"{format_number(seconds / (1e9 * YEAR), '.3g')} Mrd. a"


### TIMES OF THE TOOLBAR (written out: "2,5 Tage") ###

#seconds, singular, plural (biggest first)
TIME_UNITS = (
    (1e9 * YEAR, "Mrd. Jahre", "Mrd. Jahre"),
    (1e6 * YEAR, "Mio. Jahre", "Mio. Jahre"),
    (1000 * YEAR, "Jahrtausend", "Jahrtausende"),
    (100 * YEAR, "Jahrhundert", "Jahrhunderte"),
    (YEAR, "Jahr", "Jahre"),
    (MONTH, "Monat", "Monate"),
    (WEEK, "Woche", "Wochen"),
    (DAY, "Tag", "Tage"),
    (HOUR, "Stunde", "Stunden"),
    (MINUTE, "min", "min"),
    (1.0, "s", "s"),
)

#what can be typed into the speed box
TIME_WORDS = {
    "s": 1.0, "sek": 1.0, "sekunde": 1.0, "sekunden": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "min": MINUTE, "minute": MINUTE, "minuten": MINUTE, "minutes": MINUTE, "m": MINUTE,
    "h": HOUR, "std": HOUR, "stunde": HOUR, "stunden": HOUR, "hr": HOUR, "hour": HOUR, "hours": HOUR,
    "d": DAY, "tag": DAY, "tage": DAY, "day": DAY, "days": DAY,
    "w": WEEK, "woche": WEEK, "wochen": WEEK, "wk": WEEK, "week": WEEK, "weeks": WEEK,
    "mo": MONTH, "monat": MONTH, "monate": MONTH, "month": MONTH, "months": MONTH,
    "a": YEAR, "j": YEAR, "jahr": YEAR, "jahre": YEAR, "y": YEAR, "yr": YEAR, "year": YEAR, "years": YEAR,
    "jahrzehnt": 10 * YEAR, "jahrzehnte": 10 * YEAR, "decade": 10 * YEAR, "decades": 10 * YEAR,
    "jh": 100 * YEAR, "jahrhundert": 100 * YEAR, "jahrhunderte": 100 * YEAR,
    "century": 100 * YEAR, "centuries": 100 * YEAR,
    "jahrtausend": 1000 * YEAR, "jahrtausende": 1000 * YEAR,
    "millennium": 1000 * YEAR, "millennia": 1000 * YEAR, "kyr": 1000 * YEAR,
    "mio": 1e6 * YEAR, "miojahre": 1e6 * YEAR, "millionenjahre": 1e6 * YEAR, "myr": 1e6 * YEAR, "ma": 1e6 * YEAR,
    "mrd": 1e9 * YEAR, "mrdjahre": 1e9 * YEAR, "milliardenjahre": 1e9 * YEAR, "gyr": 1e9 * YEAR, "ga": 1e9 * YEAR,
}

#a number, a unit word and maybe "/s" behind it
DURATION_PATTERN = re.compile(r"^(" + NUMBER_PATTERN + r")([a-z]*)(?:/s|/sek|/sec|prosekunde|persecond)?$")

#the dot of "Mio." and "Mrd." (it would be read as a decimal point)
UNIT_DOT_PATTERN = re.compile(r"(?<=[a-z])\.")


def get_time_unit(seconds):
    # Returns the biggest unit the time is at least one of
    for size, singular, plural in TIME_UNITS:
        if abs(seconds) >= size: return size, singular, plural
    return TIME_UNITS[-1]


def format_time(seconds):
    size, singular, plural = get_time_unit(seconds)
    text = format_number(seconds / size, ".3g")
    if text == "1": return f"{text} {singular}"
    return f"{text} {plural}"


def parse_duration(text, default_unit):
    # "2 Tage", "10 min/s", "5 Mio. Jahre" -> seconds (ValueError if it cant be read)
    # default_unit = seconds of the unit that counts if only a number was typed
    match = DURATION_PATTERN.match(clean_typed_text(UNIT_DOT_PATTERN.sub("", text.lower())))
    if match is None: raise ValueError(text)

    unit = match.group(2)
    if unit and unit not in TIME_WORDS: raise ValueError(text)

    if unit: value = float(match.group(1)) * TIME_WORDS[unit]
    else: value = float(match.group(1)) * default_unit
    if not math.isfinite(value) or value <= 0: raise ValueError(text)
    return value


### FONTS ###

def font_has_characters(font, characters):
    # A font draws the same little box for every character it doesnt have, so each one gets compared with that box
    missing = pygame.image.tobytes(font.render("￿", True, (255, 255, 255)), "RGBA")
    for character in characters:
        if pygame.image.tobytes(font.render(character, True, (255, 255, 255)), "RGBA") == missing: return False
    return True


class Fonts():

    def __init__(self):
        if not pygame.font.get_init(): pygame.font.init()
        self.body = pygame.font.SysFont("segoeui,arial", 15)
        self.title = pygame.font.SysFont("segoeui,arial", 18, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 12)
        self.bar = pygame.font.SysFont("segoeui,arial", 13)

        #how each font draws the signs for earth and sun:
        #None = it has them itself, else a symbol font in the same size, or "plain" if no font has them
        path = pygame.font.match_font("segoeuisymbol,symbola,notosanssymbols")
        self.symbol_fonts = {}
        for font, size in ((self.body, 15), (self.title, 18), (self.small, 12), (self.bar, 13)):
            if font_has_characters(font, SYMBOLS):
                self.symbol_fonts[font] = None
                continue

            self.symbol_fonts[font] = "plain"
            if path:
                symbol_font = pygame.font.Font(path, size)
                if font_has_characters(symbol_font, SYMBOLS): self.symbol_fonts[font] = symbol_font

    def render(self, text, color, font):
        symbol_font = self.symbol_fonts.get(font)

        has_symbol = False
        for character in text:
            if character in SYMBOLS: has_symbol = True
        if symbol_font is None or not has_symbol:
            return font.render(text, True, color)

        if symbol_font == "plain":
            for symbol_unit, plain_unit in PLAIN_UNITS.items(): text = text.replace(symbol_unit, plain_unit)
            return font.render(text, True, color)

        #the text gets cut into parts with and without symbols, every part is rendered with its font
        parts = []      #[text, is symbol]
        for character in text:
            is_symbol = character in SYMBOLS
            if parts and parts[-1][1] == is_symbol: parts[-1][0] += character
            else: parts.append([character, is_symbol])

        pieces = []     #(surface, font)
        for part, is_symbol in parts:
            if is_symbol: pieces.append((symbol_font.render(part, True, color), symbol_font))
            else: pieces.append((font.render(part, True, color), font))

        #all parts have to stand on the same baseline
        ascent = max(piece_font.get_ascent() for piece, piece_font in pieces)
        height = max(ascent - piece_font.get_ascent() + piece.get_height() for piece, piece_font in pieces)
        width = sum(piece.get_width() for piece, piece_font in pieces)

        combined = pygame.Surface((width, height), pygame.SRCALPHA)
        x = 0
        for piece, piece_font in pieces:
            combined.blit(piece, (x, ascent - piece_font.get_ascent()))
            x += piece.get_width()
        return combined

    def text(self, surface, text, position, color=TEXT, font=None, anchor="topleft"):
        # Draws a text and returns its rect, anchor = the point of the text that sits on position ("midleft", "center" ...)
        rendered = self.render(text, color, font or self.body)
        rect = rendered.get_rect()
        setattr(rect, anchor, position)
        surface.blit(rendered, rect)
        return rect


fonts = None


def get_fonts():
    # All windows use the same fonts, they get loaded when the first window asks for them
    global fonts
    if fonts is None: fonts = Fonts()
    return fonts


class Panel():
    # Base class of the windows, they all draw their texts the same way

    def draw_text(self, surface, text, position, color=TEXT, font=None, anchor="topleft"):
        return self.fonts.text(surface, text, position, color, font or self.font, anchor)


def draw_panel(surface, rect, radius=12):
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=radius)
    pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=radius)


def draw_button(surface, rect, on=False, hovered=False, radius=12, fill=None):
    # Background of a button: blue while switched on, a bit lighter under the mouse
    if fill is None:
        if on: fill = TOGGLE_ON
        else: fill = TOGGLE_OFF
    if hovered: fill = (min(255, fill[0] + 22), min(255, fill[1] + 22), min(255, fill[2] + 22))
    pygame.draw.rect(surface, fill, rect, border_radius=radius)


def draw_slider(surface, left, right, y, knob, ticks, active=False, color=ACCENT):
    # A slider from left to right, knob = x of the knob, ticks = x of the small marks below
    pygame.draw.line(surface, TOGGLE_OFF, (left, y), (right, y), 4)
    pygame.draw.line(surface, color, (left, y), (knob, y), 4)
    for x in ticks: pygame.draw.line(surface, BOX_BORDER, (x, y + 5), (x, y + 8))
    if active: pygame.draw.circle(surface, TEXT, (knob, y), 7)
    else: pygame.draw.circle(surface, KNOB, (knob, y), 6)


### IMAGES (all of them are in assets.png) ###

def get_image_parameters_from_id(image_id):
    ### [0] = offset ; [1] = dimensions
    IMAGES: dict = {
        0 : ((0, 0),(50, 50)), ### exit button
        4 : ((0, 150), (24, 24)), ### planet (toolbar)
        5 : ((25, 150), (24, 24)), ### star
        6 : ((50, 150), (24, 24)), ### black hole
        7 : ((75, 150), (24, 24)), ### particles
        8 : ((0, 175), (24, 24)), ### play
        9 : ((25, 175), (24, 24)), ### pause
        10 : ((50, 175), (24, 24)), ### reload
        11 : ((75, 175), (24, 24)), ### settings
        12 : ((0, 200), (24, 24)), ### delete (inspect window)
        13 : ((25, 200), (24, 24)) ### close
    }
    return IMAGES.get(image_id)

sprite_sheet = None

def get_image(parameters : tuple):
    global sprite_sheet
    offset : tuple = parameters[0]
    dimensions : tuple = parameters[1]
    if sprite_sheet is None: sprite_sheet = pygame.image.load("assets.png")
    cropping_rect = pygame.Rect(offset[0], offset[1], dimensions[0], dimensions[1])
    image = sprite_sheet.subsurface(cropping_rect).convert_alpha()
    return image


class Button():
    # The exit button

    def __init__(self, position: tuple, image_id: int):
        self.image = get_image(get_image_parameters_from_id(image_id))
        self.rectangle = self.image.get_rect(center=position)

        self.hover = False
        self.pressed = False
        self.dirty = True       #has to be drawn again

    def update(self):
        hover = self.rectangle.collidepoint(pygame.mouse.get_pos())
        if hover != self.hover:
            self.hover = hover
            self.dirty = True
        self.pressed = hover and pygame.mouse.get_just_pressed()[0]

    def render(self, surface):
        # Draws the button if it changed, returns the area that has to go to the gpu
        if not self.dirty: return []
        self.dirty = False

        image = self.image
        if self.hover:
            #lighter under the mouse
            image = image.copy()
            image.fill((50, 50, 50, 50), special_flags=pygame.BLEND_RGBA_ADD)
        surface.fill((0, 0, 0, 0), self.rectangle)
        surface.blit(image, self.rectangle)
        return [self.rectangle]


ICON_PLANET, ICON_STAR, ICON_BLACK_HOLE, ICON_PARTICLES = 4, 5, 6, 7
ICON_PLAY, ICON_PAUSE, ICON_RELOAD, ICON_SETTINGS, ICON_DELETE, ICON_CLOSE = 8, 9, 10, 11, 12, 13

colored_icons = {}


def draw_icon(surface, image_id, center, color=None):
    # The icons in assets.png are white, with a color they get tinted (once, then the tinted icon is kept)
    key = (image_id, color)
    if key not in colored_icons:
        image = get_image(get_image_parameters_from_id(image_id))
        if color is not None: image.fill((color[0], color[1], color[2], 255), special_flags=pygame.BLEND_RGBA_MULT)
        colored_icons[key] = image
    image = colored_icons[key]
    surface.blit(image, image.get_rect(center=center))


### MOUSE CURSOR ###

wanted_cursors = {}     #window -> the cursor it wants right now (None = it doesnt care)
shown_cursor = None


def set_wanted_cursor(owner, cursor=None):
    # Several windows can want a cursor at once, the first one that wants one gets it
    global shown_cursor
    wanted_cursors[owner] = cursor

    cursor_to_show = pygame.SYSTEM_CURSOR_ARROW
    for wanted in wanted_cursors.values():
        if wanted is not None:
            cursor_to_show = wanted
            break

    if cursor_to_show == shown_cursor: return
    shown_cursor = cursor_to_show
    try:
        pygame.mouse.set_cursor(cursor_to_show)
    except pygame.error:
        pass


### TEXT INPUT ###

def get_clipboard_text():
    try:
        return pygame.scrap.get_text() or ""
    except (pygame.error, AttributeError):
        return ""


def set_clipboard_text(text):
    try:
        pygame.scrap.put_text(text)
    except (pygame.error, AttributeError):
        pass


class TextEdit():
    # The text of a field while it is being typed in, with everything a normal text field can do:
    # text cursor, selection (shift, mouse, double click), copy and paste, word jumps with ctrl
    # at the start the whole text is selected, so typing replaces the old value right away

    REPEAT_DELAY = 380          #ms a key has to be held until it repeats
    REPEAT_INTERVAL = 32
    DOUBLE_CLICK = 0.4          #s
    PADDING = 6                 #px between the edge of the box and the text

    open_edits = set()

    def __init__(self, text, font):
        self.text = text
        self.font = font
        #the selection goes from anchor to caret (caret = where the text cursor is)
        self.anchor = 0
        self.caret = len(text)
        self.scroll = 0
        self.dragging = False
        self.last_click = -1.0
        self.last_change = time.perf_counter()
        TextEdit.open_edits.add(self)
        TextEdit.update_key_repeat()

    def close(self):
        TextEdit.open_edits.discard(self)
        TextEdit.update_key_repeat()

    @classmethod
    def update_key_repeat(cls):
        # Held keys only repeat while something gets typed
        if cls.open_edits: pygame.key.set_repeat(cls.REPEAT_DELAY, cls.REPEAT_INTERVAL)
        else: pygame.key.set_repeat(0)

    ### SELECTION ###

    def get_selection(self):
        return min(self.anchor, self.caret), max(self.anchor, self.caret)

    def has_selection(self):
        return self.anchor != self.caret

    def select_all(self):
        self.anchor = 0
        self.caret = len(self.text)
        self.last_change = time.perf_counter()

    def move_caret(self, position, keep_selection):
        self.caret = max(0, min(len(self.text), position))
        if not keep_selection: self.anchor = self.caret
        self.last_change = time.perf_counter()

    def replace_selection(self, text):
        start, end = self.get_selection()
        self.text = self.text[:start] + text + self.text[end:]
        self.anchor = self.caret = start + len(text)
        self.last_change = time.perf_counter()

    def find_word_start(self, position):
        while position > 0 and self.text[position - 1] == " ": position -= 1
        while position > 0 and self.text[position - 1] != " ": position -= 1
        return position

    def find_word_end(self, position):
        end = len(self.text)
        while position < end and self.text[position] != " ": position += 1
        while position < end and self.text[position] == " ": position += 1
        return position

    ### KEYBOARD ###

    def insert(self, text):
        self.replace_selection(text.replace("\n", " ").replace("\r", ""))

    def handle_key(self, event):
        # Enter, Escape and Tab are not handled here, the window that owns the field does that
        key = event.key
        shift = bool(event.mod & pygame.KMOD_SHIFT)
        ctrl = bool(event.mod & pygame.KMOD_CTRL)

        if key == pygame.K_LEFT:
            if self.has_selection() and not shift: self.move_caret(self.get_selection()[0], False)
            elif ctrl: self.move_caret(self.find_word_start(self.caret), shift)
            else: self.move_caret(self.caret - 1, shift)
        elif key == pygame.K_RIGHT:
            if self.has_selection() and not shift: self.move_caret(self.get_selection()[1], False)
            elif ctrl: self.move_caret(self.find_word_end(self.caret), shift)
            else: self.move_caret(self.caret + 1, shift)
        elif key in (pygame.K_HOME, pygame.K_UP):
            self.move_caret(0, shift)
        elif key in (pygame.K_END, pygame.K_DOWN):
            self.move_caret(len(self.text), shift)
        elif key == pygame.K_BACKSPACE:
            #without a selection the character (or with ctrl the word) left of the cursor gets selected first
            if not self.has_selection():
                if ctrl: self.anchor = self.find_word_start(self.caret)
                else: self.anchor = max(0, self.caret - 1)
            self.replace_selection("")
        elif key == pygame.K_DELETE:
            if not self.has_selection():
                if ctrl: self.anchor = self.find_word_end(self.caret)
                else: self.anchor = min(len(self.text), self.caret + 1)
            self.replace_selection("")
        elif ctrl and key == pygame.K_a:
            self.select_all()
        elif ctrl and key in (pygame.K_c, pygame.K_x):
            start, end = self.get_selection()
            if end > start:
                set_clipboard_text(self.text[start:end])
                if key == pygame.K_x: self.replace_selection("")
        elif ctrl and key == pygame.K_v:
            self.insert(get_clipboard_text())
        else:
            return False
        return True

    ### MOUSE ###

    def index_at(self, x, rect):
        # Returns the position in the text that is closest to the x of the mouse
        offset = x - (rect.x + self.PADDING - self.scroll)
        best = 0
        best_distance = abs(offset)
        for i in range(1, len(self.text) + 1):
            distance = abs(self.font.size(self.text[:i])[0] - offset)
            if distance < best_distance:
                best = i
                best_distance = distance
        return best

    def mouse_down(self, x, rect, extend=False):
        now = time.perf_counter()
        if now - self.last_click < self.DOUBLE_CLICK and not extend:
            self.select_all()
            self.last_click = -1.0
            return
        self.last_click = now
        self.move_caret(self.index_at(x, rect), extend)
        self.dragging = True

    def mouse_drag(self, x, rect):
        if self.dragging: self.move_caret(self.index_at(x, rect), True)

    def mouse_up(self):
        self.dragging = False

    ### DRAWING ###

    def draw(self, surface, rect, color=TEXT):
        inner = rect.inflate(-2 * self.PADDING, 0)
        width = self.font.size(self.text)[0]
        caret_x = self.font.size(self.text[:self.caret])[0]

        #a text longer than the box scrolls so the cursor stays visible
        if caret_x - self.scroll > inner.width - 1: self.scroll = caret_x - inner.width + 1
        if caret_x - self.scroll < 0: self.scroll = caret_x
        self.scroll = max(0, min(self.scroll, max(0, width - inner.width + 1)))

        text_x = inner.x - self.scroll
        surface.set_clip(inner.inflate(2, 0))

        start, end = self.get_selection()
        if end > start:
            x0 = text_x + self.font.size(self.text[:start])[0]
            x1 = text_x + self.font.size(self.text[:end])[0]
            pygame.draw.rect(surface, SELECTION, pygame.Rect(x0, rect.y + 3, x1 - x0 + 1, rect.height - 6))

        rendered = self.font.render(self.text, True, color)
        surface.blit(rendered, rendered.get_rect(midleft=(text_x, rect.centery)))

        #the cursor blinks, but stays on for a moment after every change
        if (time.perf_counter() - self.last_change) % 1.0 < 0.6:
            x = text_x + caret_x
            pygame.draw.line(surface, TEXT, (x, rect.y + 4), (x, rect.bottom - 5))

        surface.set_clip(None)
