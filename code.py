import board
import busio
import displayio
from fourwire import FourWire
from adafruit_gc9a01a import GC9A01A
import terminalio
from adafruit_display_text.bitmap_label import Label
import time
import math

from adafruit_seesaw.seesaw import Seesaw
from adafruit_seesaw.rotaryio import IncrementalEncoder
from adafruit_seesaw.digitalio import DigitalIO
from adafruit_debouncer import Button

displayio.release_displays()

spi = board.SPI()
display_bus = FourWire(spi, command=board.RX, chip_select=board.TX)
display = GC9A01A(display_bus, width=240, height=240, rotation=90)

main_group = displayio.Group()
display.root_group = main_group

# --- Optimized Gradient background setup ---
GRADIENT_STEPS = 24 

def lerp_color(color1, color2, t):
    r1, g1, b1 = (color1 >> 16) & 0xFF, (color1 >> 8) & 0xFF, color1 & 0xFF
    r2, g2, b2 = (color2 >> 16) & 0xFF, (color2 >> 8) & 0xFF, color2 & 0xFF
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return (r << 16) | (g << 8) | b

def make_radial_gradient_bitmap(width, height, color_center, color_edge):
    bmp = displayio.Bitmap(width, height, GRADIENT_STEPS)
    palette = displayio.Palette(GRADIENT_STEPS)
    cx, cy = width // 2, height // 2
    max_dist = math.sqrt(cx**2 + cy**2)
    for i in range(GRADIENT_STEPS):
        t = i / (GRADIENT_STEPS - 1)
        palette[i] = lerp_color(color_center, color_edge, t)
    for y in range(height):
        for x in range(width):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            idx = min(int((dist / max_dist) * (GRADIENT_STEPS - 1)), GRADIENT_STEPS - 1)
            bmp[x, y] = idx
    return bmp, palette

# Create the gradient TileGrid
color_center_base = 0xe3f3ff  # light inner
color_edge_base = 0x9ad3fc    # dark outer
gradient_bitmap, gradient_palette = make_radial_gradient_bitmap(240, 240, color_center_base, color_edge_base)
gradient_bg = displayio.TileGrid(gradient_bitmap, pixel_shader=gradient_palette)

# Solid color background (default)
color_bitmap = displayio.Bitmap(240, 240, 1)
color_palette = displayio.Palette(1)
color_palette[0] = color_center_base
bg = displayio.TileGrid(color_bitmap, pixel_shader=color_palette)
main_group.append(bg)

# Circle Indicators
dial_indicators = []
for i in range(10):
    angle = math.pi/2 - (i / 9) * (math.pi/2)
    x = 120 + int(110 * math.cos(angle))
    y = 120 - int(110 * math.sin(angle))
    circle_bitmap = displayio.Bitmap(16, 16, 2)
    circle_palette = displayio.Palette(2)
    circle_palette[0] = color_edge_base
    circle_palette[1] = color_center_base
    circle_palette.make_transparent(0)
    circle = displayio.TileGrid(circle_bitmap, pixel_shader=circle_palette, x=x-8, y=y-8)
    radius = 7
    for cx in range(16):
        for cy in range(16):
            dist = math.sqrt((cx-8)**2 + (cy-8)**2)
            if dist <= radius:
                circle_bitmap[cx, cy] = 1
            else:
                circle_bitmap[cx, cy] = 0
    dial_indicators.append((circle, circle_palette))
    main_group.append(circle)

# Dial value 
value_label = Label(terminalio.FONT, text="", color=0xf6bc14, anchor_point=(0.5, 0.0), anchored_position=(120, 130), scale=3)
main_group.append(value_label)

# Header label (changes based on mode)
header_label = Label(terminalio.FONT, text="", color=0x000000, anchor_point=(0.5, 0.0), anchored_position=(120, 60), scale=1)
main_group.append(header_label)

# Book label - centered (shows call numbers or description)
book_label = Label(terminalio.FONT, text="", color=0xFFFFFF, anchor_point=(0.5, 0.0), anchored_position=(120, 90), scale=2)
main_group.append(book_label)

# Description label 
desc_label = Label(terminalio.FONT, text="", color=0xFFFFFF, anchor_point=(0.5, 0.0), anchored_position=(120, 160), scale=1)
main_group.append(desc_label)

# Book navigation indicator
nav_label = Label(terminalio.FONT, text="", color=0xf6bc14, x=10, y=200, scale=1)
main_group.append(nav_label)

# Status Indicator
status_label = Label(terminalio.FONT, text="Ready", color=0x000000, x=110, y=220, scale=1)
main_group.append(status_label)

print("Device Ready")

# Initialize encoder
i2c = busio.I2C(board.SCL1, board.SDA1)
seesaw = Seesaw(i2c, addr=0x36)
seesaw.pin_mode(24, seesaw.INPUT_PULLUP)
ss_pin = DigitalIO(seesaw, 24)
button = Button(ss_pin, long_duration_ms=600)
encoder = IncrementalEncoder(seesaw)

dial_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
current_index = 4

# State management
MODE_WAITING = -1 # Waiting to recieve books 
MODE_LANDING = 0  # Landing page with all call numbers
MODE_BOOK_VIEW = 1  # Individual book view
MODE_TUNING = 2  # Tuning mode for recommendations

current_mode = MODE_WAITING
current_book_index = 0

last_press_time = 0
press_count = 0
DOUBLE_PRESS_INTERVAL = 0.5  

def wrap_text_to_fit(text, max_chars_per_line=28):
    book_lines = text.split("\n")
    all_wrapped_lines = []
    for book_title in book_lines:
        words = book_title.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    all_wrapped_lines.append(current_line.center(max_chars_per_line))
                current_line = word
        if current_line:
            all_wrapped_lines.append(current_line.center(max_chars_per_line))
        all_wrapped_lines.append("")
    return "\n".join(all_wrapped_lines[:-1][:10])

def center_text(label, text):
    text_width = len(text) * 6
    x_pos = (240 - text_width) // 2
    label.x = x_pos
    label.text = text

def display_waiting_page():
    if gradient_bg in main_group:
        main_group.remove(gradient_bg)
    header_label.text = "CATALOGNET"
    header_label.color = 0x000000
    header_label.anchor_point = (0.5, 0.0)
    header_label.anchored_position = (120, 110) 
    header_label.scale = 3
    book_label.text = ""
    desc_label.text = ""
    nav_label.text = ""
    value_label.text = ""
    center_text(status_label, "Ready")
    status_label.color = 0x000000
    for indicator, palette in dial_indicators:
        indicator.hidden = True

def display_landing_page(books, highlight_index):
    if gradient_bg not in main_group:
        main_group.insert(1, gradient_bg)
    header_label.text = "Ready for your Reading Journey?"
    header_label.color = 0x000000
    header_label.anchor_point = (0.5, 0.0)
    header_label.anchored_position = (120, 60)
    for indicator, palette in dial_indicators:
        indicator.hidden = True
    start_idx = max(0, highlight_index - 2)
    end_idx = min(len(books), start_idx + 5)
    start_idx = max(0, end_idx - 5)
    books_to_show = books[start_idx:end_idx]
    display_lines = []
    for i, book in enumerate(books_to_show):
        actual_idx = start_idx + i
        if actual_idx == highlight_index:
            display_lines.append(f"> {book} <")
        else:
            display_lines.append(f"  {book}")
    book_label.text = "\n".join(display_lines)
    desc_label.text = ""  # Clear description
    center_text(status_label, f"{len(books)} books")
    status_label.color = 0x000000
    nav_label.text = ""
    value_label.text = ""

def display_book_view(call_number):
    if gradient_bg in main_group:
        main_group.remove(gradient_bg)
    header_label.text = "" 
    desc_label.color = 0x000000  
    book_label.text = ""  
    try:
        with open("/selectedBook.txt", "r", encoding="utf-8") as f:
            file_content = f.read().strip()
        raw_books = [book.strip() for book in file_content.split("\n") if book.strip()]
        if 0 <= current_book_index < len(raw_books):
            parts = raw_books[current_book_index].split("|")
            if len(parts) >= 3:
                desc_text = parts[2].strip()
            else:
                desc_text = ""
        else:
            desc_text = ""
    except Exception as e:
        desc_text = ""

    combined_text = "Here is a description of the book, I hope you like it\n\n" + desc_text
    desc_label.text = wrap_text_to_fit(combined_text, max_chars_per_line=28)
    desc_label.anchored_position = (120, 60)  
    center_text(status_label, "Press for Recs")
    status_label.color = 0x000000
    nav_label.text = ""
    value_label.text = ""
    for indicator, palette in dial_indicators:
        indicator.hidden = True

def display_tuning_mode(call_number, dial_value):
    if gradient_bg in main_group:
        main_group.remove(gradient_bg)
    header_label.text = "I want something..."
    header_label.color = 0x000000
    header_label.anchor_point = (0.5, 0.0)
    header_label.anchored_position = (120, 80) 
    book_label.text = ""
    desc_label.text = ""
    mode_text = get_search_mode(dial_value)
    value_label.text = mode_text
    value_label.anchor_point = (0.5, 0.0)
    value_label.anchored_position = (120, 110) 
    value_label.color = 0x000000
    center_text(status_label, "Tuning Mode")
    status_label.color = 0x000000
    nav_label.text = ""
  
    active_count = int((dial_value / 1.0) * 10)
    for i, (indicator, palette) in enumerate(dial_indicators):
        indicator.hidden = False
        if i < active_count:
            palette[1] = color_edge_base
        else:
            palette[1] = color_center_base

def get_search_mode(dial_value):
    if dial_value <= 0.3:
        return "surprising"
    elif dial_value <= 0.6:
        return "broad"
    else:
        return "specific"

def alphabetical_call_number(book_line):
    # Expect: Title | Author | Description
    parts = book_line.split("|")
    if len(parts) >= 2:
        author = parts[1].strip()
    else:
        author = book_line.split()[0] if book_line.split() else book_line
    words = author.split()
    last_name = words[-1] if words else author
    first_char = last_name[0].upper() if last_name else 'A'
    main_num = int((ord(first_char) - ord('A')) / 26 * 1000)
    sub_num = sum(ord(c.upper()) for c in last_name[:3]) % 1000
    return f"{main_num:03d}.{sub_num:03d}"

def clear_dial_indicators():
    for i, (indicator, palette) in enumerate(dial_indicators):
        palette[1] = 0x450000
        indicator.hidden = True

last_book = ""
last_file_content = ""
displayed_books = []
all_books = []

for indicator, palette in dial_indicators:
    indicator.hidden = True

pulse_phase = 0  # For gradient pulsing
frame_count = 0

while True:
    # --- Animate the gradient if on landing page ---
    if current_mode == MODE_LANDING and gradient_bg in main_group:
        frame_count += 1
        if frame_count % 3 == 0:  # Only update every 3 frames
            pulse_phase += 0.07  # Fast pulse
            pulse = (math.sin(pulse_phase) + 1) / 2  # 0..1
            def adjust_brightness(color, factor):
                r = int(((color >> 16) & 0xFF) * factor)
                g = int(((color >> 8) & 0xFF) * factor)
                b = int((color & 0xFF) * factor)
                return (r << 16) | (g << 8) | b
            center = adjust_brightness(color_center_base, 0.8 + 0.2 * pulse)
            edge = adjust_brightness(color_edge_base, 0.8 + 0.2 * pulse)
            for i in range(GRADIENT_STEPS):
                t = i / (GRADIENT_STEPS - 1)
                gradient_palette[i] = lerp_color(center, edge, t)

    button.update()
    if button.pressed:
        now = time.monotonic()
        if now - last_press_time < DOUBLE_PRESS_INTERVAL:
            press_count += 1
        else:
            press_count = 1
        last_press_time = now

        if not all_books:
            current_mode = MODE_WAITING
            display_waiting_page()
        elif current_mode == MODE_LANDING:
            current_mode = MODE_BOOK_VIEW
            display_book_view(all_books[current_book_index])
        elif current_mode == MODE_BOOK_VIEW:
            current_mode = MODE_TUNING
            display_tuning_mode(all_books[current_book_index], dial_values[current_index])
            print(dial_values[current_index])
            print(f"Book: {all_books[current_book_index]} | Mode: {get_search_mode(dial_values[current_index])} ({dial_values[current_index]})")
        elif current_mode == MODE_TUNING:
            if press_count == 2:
                current_mode = MODE_LANDING
                clear_dial_indicators()
                value_label.text = ""
                display_landing_page(all_books, current_book_index)
                press_count = 0

    position = encoder.position
    if position != 0:
        if current_mode == MODE_LANDING:
            current_book_index += position
            current_book_index = max(0, min(len(all_books) - 1, current_book_index))
            display_landing_page(all_books, current_book_index)
        elif current_mode == MODE_TUNING:
            current_index += position
            current_index = max(0, min(len(dial_values) - 1, current_index))
            display_tuning_mode(all_books[current_book_index], dial_values[current_index])
            print(dial_values[current_index])
            try:
                with open("/dial_value.txt", "w") as f:
                    f.write(f"{all_books[current_book_index]}\n{dial_values[current_index]}")
                    f.flush()
            except Exception as e:
                print(f"Error writing dial: {e}")
            print(f"Book: {all_books[current_book_index]} | Mode: {get_search_mode(dial_values[current_index])} ({dial_values[current_index]})")
    
        encoder.position = 0
   
    try:
        with open("/selectedBook.txt", "r", encoding="utf-8") as f:
            file_content = f.read().strip()
        raw_books = [book.strip() for book in file_content.split("\n") if book.strip()]
        all_books = [alphabetical_call_number(book) for book in raw_books]
        displayed_books = all_books
        # Only update landing page if in WAITING mode or books changed
        if current_mode == MODE_WAITING and all_books:
            current_mode = MODE_LANDING
            display_landing_page(all_books, current_book_index)
        elif not all_books:
            current_mode = MODE_WAITING
            display_waiting_page()
    except OSError:
        all_books = []
        displayed_books = []
        if current_mode != MODE_WAITING:
            current_mode = MODE_WAITING
            display_waiting_page()
    except Exception as e:
        print(f"Error: {e}")
        center_text(status_label, "Error")
        status_label.color = 0xFF0000

    time.sleep(0.02)  # Small delay for animation smoothness