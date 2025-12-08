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

# Header label (changes based on mode)
header_label = Label(terminalio.FONT, text="An Adventure Awaits", color=0xf6bc14, anchor_point=(0.5, 0.0), anchored_position=(120, 60), scale=1)
main_group.append(header_label)

# Book label - centered (shows call numbers or description)
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

# State management
MODE_LANDING = 0  # Landing page with all call numbers
MODE_BOOK_VIEW = 1  # Individual book view
MODE_TUNING = 2  # Tuning mode for recommendations

current_mode = MODE_LANDING
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

def display_landing_page(books, highlight_index):
    """Display landing page with scrollable call numbers"""
    header_label.text = "An Adventure Awaits"
    
    # Show 5 books at a time, centered around highlighted book
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
    center_text(status_label, f"{len(books)} books")
    status_label.color = 0xFFFFFF
    nav_label.text = ""
    value_label.text = ""

def display_book_view(call_number):
    """Display individual book view with description"""
    header_label.text = call_number
    book_label.text = wrap_text_to_fit("Here is a description of the book, I hope you like it", max_chars_per_line=28)
    center_text(status_label, "Click for more")
    status_label.color = 0x80BFD9
    nav_label.text = ""
    value_label.text = ""

def display_tuning_mode(call_number, dial_value):
    """Display tuning mode for recommendations"""
    header_label.text = "Other Book Suggestions"
    book_label.text = ""
    
    mode_text = get_search_mode(dial_value)
    value_label.text = mode_text
    
    center_text(status_label, "Tuning Mode")
    status_label.color = 0xFFFFFF
    nav_label.text = ""
    
    # Update indicators
    active_count = int((dial_value / 1.0) * 10)
    for i, (indicator, palette) in enumerate(dial_indicators):
        if i < active_count:
            palette[1] = 0xf6bc14
        else:
            palette[1] = 0x450000

def get_search_mode(dial_value):
    """Convert dial value to search mode text"""
    if dial_value <= 0.3:
        return "SURPRISE ME"
    elif dial_value <= 0.6:
        return "BROAD"
    else:
        return "SPECIFIC"

def alphabetical_call_number(book_title):
    """Generate call number based on author's last name (like library cataloging)"""
    if " by " in book_title.lower():
        parts = book_title.split(" by ")
        author = parts[1].strip() if len(parts) > 1 else book_title
    else:
        author = book_title.split()[0] if book_title.split() else book_title
    
    words = author.split()
    last_name = words[-1] if words else author
    
    first_char = last_name[0].upper() if last_name else 'A'
    main_num = int((ord(first_char) - ord('A')) / 26 * 1000)
    sub_num = sum(ord(c.upper()) for c in last_name[:3]) % 1000

    return f"{main_num:03d}.{sub_num:03d}"

def clear_dial_indicators():
    """Clear all dial indicators"""
    for i, (indicator, palette) in enumerate(dial_indicators):
        palette[1] = 0x450000
    
last_book = ""
last_file_content = ""
displayed_books = []
all_books = []

# Main loop
while True:
    button.update()
    
    if button.pressed:
        if not all_books:
            center_text(status_label, "No books yet")
            status_label.color = 0xFFFF00
        elif current_mode == MODE_LANDING:
            # Enter book view
            current_mode = MODE_BOOK_VIEW
            display_book_view(all_books[current_book_index])
            
        elif current_mode == MODE_BOOK_VIEW:
            # Enter tuning mode
            current_mode = MODE_TUNING
            display_tuning_mode(all_books[current_book_index], dial_values[current_index])
            
            # Print initial dial value to serial
            print(dial_values[current_index])
            print(f"Book: {all_books[current_book_index]} | Mode: {get_search_mode(dial_values[current_index])} ({dial_values[current_index]})")
            
        elif current_mode == MODE_TUNING:
            # Exit back to landing page
            current_mode = MODE_LANDING
            clear_dial_indicators()
            value_label.text = ""
            display_landing_page(all_books, current_book_index)

    # Handle encoder rotation
    position = encoder.position
    if position != 0:
        if current_mode == MODE_LANDING:
            # Scroll through call numbers on landing page
            current_book_index += position
            current_book_index = max(0, min(len(all_books) - 1, current_book_index))
            display_landing_page(all_books, current_book_index)
        
        elif current_mode == MODE_TUNING:
            # Adjust dial value in tuning mode
            current_index += position
            current_index = max(0, min(len(dial_values) - 1, current_index))
            
            display_tuning_mode(all_books[current_book_index], dial_values[current_index])
            
            # Print the dial value to serial
            print(dial_values[current_index])
            
            # Write dial value to file
            try:
                with open("/dial_value.txt", "w") as f:
                    f.write(f"{all_books[current_book_index]}\n{dial_values[current_index]}")
                    f.flush()
            except Exception as e:
                print(f"Error writing dial: {e}")
            
            print(f"Book: {all_books[current_book_index]} | Mode: {get_search_mode(dial_values[current_index])} ({dial_values[current_index]})")
        
        encoder.position = 0
    
    # Check for new book from file
    try:
        with open("/selected_book.txt", "r") as f:
            file_content = f.read().strip()
        
        if file_content:
            raw_books = []
            for line in file_content.split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split("|", 1)]
                title = parts[0] if parts else ""
                if title:
                    raw_books.append(title)
            all_books = [alphabetical_call_number(book) for book in raw_books]

            if all_books != displayed_books:
                displayed_books = all_books
                
                # Update display if in landing mode
                if current_mode == MODE_LANDING:
                    display_landing_page(all_books, current_book_index)
                    print(f"DISPLAY: Showing {len(all_books)} books")
        else:
            all_books = []
            displayed_books = []
            current_mode = MODE_LANDING
            header_label.text = "An Adventure Awaits"
            book_label.text = ""
            nav_label.text = ""
            value_label.text = ""
              
    except OSError:
        all_books = []
        displayed_books = []
        current_mode = MODE_LANDING
        clear_dial_indicators()
        header_label.text = "An Adventure Awaits"
        book_label.text = ""
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