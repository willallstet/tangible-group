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
display = GC9A01A(display_bus, width=240, height=240)

main_group = displayio.Group()
display.root_group = main_group

# Display Background
color_bitmap = displayio.Bitmap(240, 240, 1)
color_palette = displayio.Palette(1)
color_palette[0] = 0x610000
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
    circle_palette[0] = 0xBD7500
    circle_palette[1] = 0x450000
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
value_label = Label(terminalio.FONT, text="", color=0xf6bc14, anchor_point=(0.5, 0.0), anchored_position=(120, 150), scale=3)
main_group.append(value_label)

# "You Selected:" header
header_label = Label(terminalio.FONT, text="YOUR SELECTIONS...", color=0xf6bc14, x=15, y=80, scale=1)
main_group.append(header_label)

# Book label - centered
book_label = Label(terminalio.FONT, text="", color=0xFFFFFF, anchor_point=(0.5, 0.0), anchored_position=(120, 100), scale=1)
main_group.append(book_label)

# Book navigation indicator
nav_label = Label(terminalio.FONT, text="", color=0xf6bc14, x=10, y=200, scale=1)
main_group.append(nav_label)

# Status Indicator
status_label = Label(terminalio.FONT, text="Ready", color=0xFFFFFF, x=85, y=220, scale=1)
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

# Dial and book browsing state
dial_active = False
browse_mode = False
current_book_index = 0

# Display Text
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
    
    return "\n".join(all_wrapped_lines[:-1][:6])

def center_text(label, text):
    text_width = len(text) * 6
    x_pos = (240 - text_width) // 2
    label.x = x_pos
    label.text = text

def display_single_book(book_title, index, total):
    """Display a single book with navigation info"""
    wrapped = wrap_text_to_fit(book_title, max_chars_per_line=28)
    book_label.text = wrapped
    nav_label.text = f"< {index + 1}/{total} >"
    center_text(nav_label, f"< {index + 1}/{total} >")

def get_search_mode(dial_value):
    """Convert dial value to search mode text"""
    if dial_value <= 0.3:
        return "SURPRISE ME"
    elif dial_value <= 0.6:
        return "BROAD"
    else:
        return "SPECIFIC"
    
last_book = ""
last_file_content = ""
displayed_books = []
all_books = []

# Button Press - Toggle modes
while True:
    button.update()
    
    if button.pressed:
        if not all_books:
            # No books loaded, do nothing
            center_text(status_label, "No books yet")
            status_label.color = 0xFFFF00
        elif not browse_mode:
            # Enter browse mode from list view
            browse_mode = True
            current_book_index = 0
            display_single_book(all_books[current_book_index], current_book_index, len(all_books))
            center_text(status_label, "Browse Books")
            status_label.color = 0x80BFD9
        elif browse_mode and not dial_active:
            # Click into book - activate dial for this book
            dial_active = True
            mode_text = get_search_mode(dial_values[current_index])
            value_label.text = mode_text
            center_text(status_label, "Tuning Mode")
            status_label.color = 0xFFFFFF
            
            # Print initial dial value to serial
            print(dial_values[current_index])
            
            print(f"Book: {all_books[current_book_index]} | Mode: {mode_text} ({dial_values[current_index]})")
            
            active_count = int((dial_values[current_index] / 1.0) * 10)
            for i, (indicator, palette) in enumerate(dial_indicators):
                if i < active_count:
                    palette[1] = 0xf6bc14
                else:
                    palette[1] = 0x450000
        elif dial_active:
            # Exit dial tuning mode back to browse mode
            dial_active = False
            value_label.text = ""
            # Clear indicators
            for i, (indicator, palette) in enumerate(dial_indicators):
                palette[1] = 0x450000
            
            display_single_book(all_books[current_book_index], current_book_index, len(all_books))
            center_text(status_label, "Browse Books")
            status_label.color = 0x80BFD9

    # Handle encoder rotation
    position = encoder.position
    if position != 0:
        if browse_mode and not dial_active:
            # Book browsing mode - scroll through books
            current_book_index += position
            current_book_index = max(0, min(len(all_books) - 1, current_book_index))
            display_single_book(all_books[current_book_index], current_book_index, len(all_books))
        
        elif dial_active:
            # Dial tuning mode - adjust dial value
            current_index += position
            current_index = max(0, min(len(dial_values) - 1, current_index))
            
            mode_text = get_search_mode(dial_values[current_index])
            value_label.text = mode_text 
            
            active_count = int((dial_values[current_index] / 1.0) * 10)
            for i, (indicator, palette) in enumerate(dial_indicators):
                if i < active_count:
                    palette[1] = 0xf6bc14
                else:
                    palette[1] = 0x450000
            
            # Print the dial value to serial for DialReader
            print(dial_values[current_index])
            
            # Write dial value to file
            try:
                with open("/dial_value.txt", "w") as f:
                    f.write(f"{all_books[current_book_index]}\n{dial_values[current_index]}")
                    f.flush()
            except Exception as e:
                print(f"Error writing dial: {e}")
            
            print(f"Book: {all_books[current_book_index]} | Mode: {mode_text} ({dial_values[current_index]})")
        
        encoder.position = 0
    
    # Check for new book from file
    try:
        with open("/selected_book.txt", "r") as f:
            file_content = f.read().strip()
            
            if file_content:
                all_books = [book.strip() for book in file_content.split("\n") if book.strip()]
                
                if all_books != displayed_books:
                    displayed_books = all_books
                    
                    # Show list view when not browsing
                    if not browse_mode and not dial_active:
                        books_to_show = all_books[-5:] if len(all_books) > 5 else all_books
                        display_text = "\n".join(books_to_show)
                        
                        wrapped_text = wrap_text_to_fit(display_text, max_chars_per_line=28)
                        book_label.text = wrapped_text
                        center_text(status_label, f"{len(all_books)} books")
                        status_label.color = 0xFFFFFF
                        print(f"DISPLAY: Showing {len(books_to_show)} of {len(all_books)} books")
            else:
                all_books = []
                displayed_books = []
                browse_mode = False
                dial_active = False
                nav_label.text = ""
                value_label.text = ""
                    
    except OSError:
        all_books = []
        displayed_books = []
        browse_mode = False
        dial_active = False
        nav_label.text = ""
        value_label.text = ""
        if status_label.text != "No file":
            center_text(status_label, "No file")
            status_label.color = 0xFFFF00
    except Exception as e:
        print(f"Error: {e}")
        center_text(status_label, "Error")
        status_label.color = 0xFF0000
    
    time.sleep(0.2)