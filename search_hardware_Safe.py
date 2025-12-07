import json
import glob
import threading
import time
import numpy as np
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QScrollArea,
    QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QFont,
    QPalette,
    QPainter,
    QPainterPath,
    QColor,
    QPen,
    QPixmap
)
import datetime

try:
    import serial  # type: ignore
except ImportError:
    serial = None

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None


# Configuration
DATA_FILE = "books.json"
GEMINI_EMBED_MODEL = "models/text-embedding-004"
DIAL_BAUDRATE = 115200
LED_BAUDRATE = 115200
WINDOW_SIZE = 0.25  # similarity window width
GEMINI_CHAT_MODEL = "models/gemini-2.5-flash-lite"
CHAT_SYSTEM_PROMPT = (
    "You are a library assistant. You help people discover their next great read. "
    "Ask light follow-up questions when helpful, then propose recommendations. "
    "After each reply include a line exactly like 'SEARCH_QUERY: <concise keywords>' "
    "describing what you will search for next. Use 'SEARCH_QUERY: NONE' only when no search should be run."
)


def load_env_file(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
elif genai is None:
    print("google-generativeai not installed; run `pip install google-generativeai`.")
else:
    print("GEMINI_API_KEY missing; set it in your environment or .env file.")


def detect_serial_port():
    """Return the first USB serial port that looks like the QT Py."""
    ports = glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/tty.usbserial*")
    return ports[0] if ports else None


class DialReader(threading.Thread):
    """Background thread that reads a normalized value (0-1) from the dial."""

    def __init__(self, port=None, baudrate=DIAL_BAUDRATE):
        super().__init__(daemon=True)
        env_port = os.environ.get("DIAL_SERIAL_PORT")
        self.port = port or env_port or detect_serial_port()
        self.baudrate = baudrate
        self.value = 0.5
        self.ser = None  # ADD THIS LINE

    def run(self):
        if serial is None or not self.port:
            return
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)  # Store reference
        except Exception as exc:
            print(f"DialReader could not open {self.port}: {exc}")
            return

        # REMOVE the 'with' statement - keep the connection open
        while True:
            try:
                line = self.ser.readline().strip()
                if not line:
                    continue
                try:
                    raw = float(line)
                except ValueError:
                    continue
                if raw > 1:
                    raw = raw / 1023.0  # handle 0-1023 readings
                raw = max(0.0, min(1.0, raw))
                self.value = raw
            except Exception as e:
                print(f"DialReader error: {e}")
                break

    def get_value(self):
        return self.value


class ChatManager:
    """Thin wrapper around Gemini chat for conversational queries."""

    def __init__(self, model_name=GEMINI_CHAT_MODEL):
        self.chat = None
        self.model = None
        if genai is None:
            print("ChatManager: google-generativeai not installed.")
            return
        if not GEMINI_API_KEY:
            print("ChatManager: GEMINI_API_KEY not set.")
            return
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=CHAT_SYSTEM_PROMPT,
            )
            self.chat = self.model.start_chat(history=[])
        except Exception as exc:
            print(f"ChatManager init error: {exc}")
            self.chat = None

    def is_available(self):
        return self.chat is not None

    def send_message(self, user_message):
        if not self.chat:
            return "I'm offline right now.", ""
        try:
            response = self.chat.send_message(user_message)
            raw_text = response.text or ""
        except Exception as exc:
            print(f"ChatManager send error: {exc}")
            return "I ran into an issue reaching Gemini.", ""
        return self._split_response(raw_text)

    @staticmethod
    def _split_response(raw_text):
        query = ""
        clean_lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("SEARCH_QUERY:"):
                query_value = stripped.split(":", 1)[1].strip()
                if query_value.upper() == "NONE":
                    query = ""
                else:
                    query = query_value
            else:
                clean_lines.append(line)
        clean_text = "\n".join(clean_lines).strip()
        if not clean_text:
            clean_text = raw_text.strip()
        return clean_text, query


class LEDController:
    """Sends highlighted book positions to an Arduino over serial."""

    def __init__(self, port=None, baudrate=LED_BAUDRATE):
        self.port = port or os.environ.get("LED_SERIAL_PORT")
        self.baudrate = baudrate
        self.device = None

        if serial is None:
            print("LEDController: pyserial not installed; LEDs disabled.")
            return

        if not self.port:
            print("LEDController: set LED_SERIAL_PORT to your Arduino device path.")
            return

        # Try the specified port, then try alternative paths if it fails
        ports_to_try = [self.port]
        if "/dev/cu." in self.port:
            ports_to_try.append(self.port.replace("/dev/cu.", "/dev/tty."))
        elif "/dev/tty." in self.port:
            ports_to_try.append(self.port.replace("/dev/tty.", "/dev/cu."))

        for attempt_port in ports_to_try:
            for attempt in range(3):
                try:
                    self.device = serial.Serial(attempt_port, self.baudrate, timeout=1)
                    if attempt_port != self.port:
                        print(f"LEDController: opened {attempt_port} (tried {self.port} first)")
                    else:
                        print(f"LEDController: opened {attempt_port}")
                    return
                except serial.SerialException as exc:
                    if attempt < 2:
                        time.sleep(0.5)
                    else:
                        if attempt_port == ports_to_try[-1]:
                            print(f"LEDController could not open {attempt_port} after retries: {exc}")
                except Exception as exc:
                    if attempt_port == ports_to_try[-1]:
                        print(f"LEDController could not open {attempt_port}: {exc}")
                    break

    def send_positions(self, positions):
        if not self.device:
            return
        payload = ",".join(str(pos) for pos in positions)
        try:
            self.device.write((payload + "\n").encode("utf-8"))
        except Exception as exc:
            print(f"LEDController write error: {exc}")


def load_data(filename):
    import codecs
    with codecs.open(filename, "r", encoding="utf-8", errors='replace') as f:
        return json.load(f)


def get_embedding(text):
    if genai is None:
        print("Embedding error: google-generativeai not installed.")
        return []
    if not GEMINI_API_KEY:
        print("Embedding error: GEMINI_API_KEY not set.")
        return []
    try:
        response = genai.embed_content(model=GEMINI_EMBED_MODEL, content=text)
        return response.get("embedding", [])
    except Exception as exc:
        print(f"Embedding error: {exc}")
        return []


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def generate_embeddings_for_corpus(books):
    embeddings = []
    for book in books:
        text_content = (
            f"Title: {book['Title']}. Description: {book['Description']}."
            f" Year Published: {book['Year Published']}. Author: {book['Author']}."
        )
        emb = get_embedding(text_content)
        embeddings.append(emb)
    return embeddings


def get_top_similar_books(target_idx, books, book_embeddings, top_k=2, window=None):
    target_vec = book_embeddings[target_idx]
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        if i == target_idx:
            continue
        score = cosine_similarity(target_vec, book_vec)
        if window:
            low, high = window
            if score < low or score > high:
                continue
        scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [idx for score, idx in scores[:top_k]]


def get_top_books_by_query(query, books, book_embeddings, top_k=3, window=None):
    query_vec = get_embedding(query)
    if not query_vec:
        return []
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        score = cosine_similarity(query_vec, book_vec)
        if window:
            low, high = window
            if score < low or score > high:
                continue
        scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [idx for score, idx in scores[:top_k]]


class FolderFrame(QWidget):
    """Custom container that draws a folder-like outline with diagonal tab edges."""

    def __init__(self, bg_color, border_color, accent_color, parent=None):
        super().__init__(parent)
        self.bg_color = QColor(bg_color)
        self.border_color = QColor(border_color)
        self.accent_color = QColor(accent_color)
        # We want transparency outside the folder shape
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()

        width = rect.width()
        height = rect.height()

        # Parameters to mimic file-folder tab with diagonal edges
        margin = 8
        tab_height = 36  # Shorter tab
        tab_width = 200
        diagonal = 10   # Slightly less pronounced diagonal

        # Single continuous path for the folder outline
        path = QPainterPath()
        # Start at top-left of the main body (below the tab)
        path.moveTo(margin, tab_height)
        
        # Tab shape
        # Diagonal up to top-left of tab
        path.lineTo(margin + diagonal, margin)
        # Horizontal across top of tab
        path.lineTo(margin + tab_width - diagonal, margin)
        # Diagonal down to bottom-right of tab
        path.lineTo(margin + tab_width, tab_height)
        
        # Top edge of the main body (right of tab)
        path.lineTo(width - margin, tab_height)
        
        # Right edge
        path.lineTo(width - margin, height - margin)
        
        # Bottom edge
        path.lineTo(margin, height - margin)
        
        # Left edge (back to start)
        path.lineTo(margin, tab_height)
        
        path.closeSubpath()

        # Fill the folder shape with the background color
        painter.fillPath(path, self.bg_color)

        # Draw the border
        painter.setPen(QPen(self.border_color, 2))
        painter.drawPath(path)


class BookGUI(QMainWindow):
    def __init__(self, books, book_embeddings, dial_reader=None, chat_manager=None, led_controller=None):
        super().__init__()
        self.books = books
        self.book_embeddings = book_embeddings
        self.dial_reader = dial_reader
        self.dial_value = dial_reader.get_value() if dial_reader else 0.5
        self.chat_manager = chat_manager
        self.led_controller = led_controller
        self.previously_glowing = []
        self.image_labels = []
        self.chat_display = None
        self.chat_input = None
        self.window_label = None
        self.send_button = None
        self.timer = None
        self.last_query = ""

        # Color scheme: dark brown/black background, yellow/gold accents
        self.bg_color = "#1e1304"  # Dark brown/black (outer background)
        self.folder_bg_color = "#2e1a08"  # Slightly lighter brown (folder interior)
        self.accent_color = "#f6bc14"  # Gold/yellow
        self.text_color = "#f6bc14"  # Gold/yellow text
        self.border_color = "#f6bc14"  # Gold/yellow borders

        self.setGeometry(100, 100, 1200, 800)
        
        # Set monospace font
        self.monospace_font = QFont("Courier", 12, QFont.Normal)
        self.monospace_font.setStyleHint(QFont.Monospace)

        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {self.bg_color};")
        self.setCentralWidget(central_widget)

        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(0)

        folder_frame = FolderFrame(self.folder_bg_color, self.border_color, self.accent_color)
        folder_layout = QVBoxLayout(folder_frame)
        # Leave extra top padding so the tab/connector lines drawn by FolderFrame stay visible
        folder_layout.setContentsMargins(26, 48, 26, 26)
        folder_layout.setSpacing(14)

        # Date label in tab area (positioned absolutely)
        date_label = QLabel(datetime.datetime.now().strftime("%b %d %Y").upper(), folder_frame)
        date_font = QFont(self.monospace_font)
        date_font.setPointSize(13)
        date_font.setBold(True)
        date_label.setFont(date_font)
        date_label.setStyleSheet(f"color: {self.text_color}; background: transparent;")
        date_label.setAlignment(Qt.AlignCenter)
        # Position centered within the tab area (tab_width=200)
        # Manually using margin=8, tab width ~200
        date_label.setGeometry(8, 8, 200, 30)
        # No longer adding date_label to the layout since it's positioned manually

        # Header with tab; wrap in a container with insets to match chatbox width and add vertical spacing
        header_wrapper = QWidget()
        header_wrapper.setStyleSheet("background: transparent;")
        header_wrapper_layout = QHBoxLayout(header_wrapper)
        # Left/Right margins (16) match chat_container, Top/Bottom (20/10) adds spacing
        header_wrapper_layout.setContentsMargins(16, 25, 16, 5) 
        header_wrapper_layout.setSpacing(0)

        header_label = QLabel("CATALOGNET VECTOR SEARCH")
        header_label.setFont(self.monospace_font)
        header_label.setStyleSheet(f"""
            background-color: {self.accent_color};
            color: {self.bg_color};
            padding: 8px 16px;
            font-weight: bold;
            border: 2px solid {self.border_color};
        """)
        header_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_wrapper_layout.addWidget(header_label)
        folder_layout.addWidget(header_wrapper)

        # Conversation area with border, inset to create even gap around the box
        chat_container = QWidget()
        chat_container.setStyleSheet("background: transparent;")
        chat_container_layout = QVBoxLayout(chat_container)
        chat_container_layout.setContentsMargins(16, 12, 16, 12)
        chat_container_layout.setSpacing(0)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(self.monospace_font)
        self.chat_display.setStyleSheet(f"""
            padding: 15px;
            font-size: 13px;
            background-color: {self.bg_color};
            border: 2px solid {self.border_color};
            color: {self.text_color};
        """)
        # Style the scrollbar to match the aesthetic
        self.chat_display.verticalScrollBar().setStyleSheet(f"""
            QScrollBar:vertical {{
                background-color: {self.bg_color};
                width: 12px;
                border: 1px solid {self.border_color};
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.accent_color};
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #ffed4e;
            }}
        """)
        # Set text color in palette
        palette = self.chat_display.palette()
        palette.setColor(QPalette.Text, QColor(self.text_color))
        self.chat_display.setPalette(palette)
        chat_container_layout.addWidget(self.chat_display)
        folder_layout.addWidget(chat_container, stretch=1)

        # Input area with input field and send button
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(16, 0, 16, 0)  # Match chat container inset
        input_layout.setSpacing(10)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("ASK ABOUT ANYTHING")
        self.chat_input.setFont(self.monospace_font)
        self.chat_input.setStyleSheet(f"""
            QLineEdit {{
                padding: 10px;
                font-size: 13px;
                border: 2px solid {self.border_color};
                background-color: {self.bg_color};
                color: {self.text_color};
            }}
            QLineEdit::placeholder {{
                color: {self.text_color};
                opacity: 0.6;
            }}
            QLineEdit:focus {{
                border: 2px solid {self.accent_color};
            }}
        """)
        # Set placeholder and text colors
        input_palette = self.chat_input.palette()
        input_palette.setColor(QPalette.Text, QColor(self.text_color))
        input_palette.setColor(QPalette.PlaceholderText, QColor(self.text_color))
        self.chat_input.setPalette(input_palette)
        self.chat_input.returnPressed.connect(self.on_chat_submit)
        input_layout.addWidget(self.chat_input, stretch=1)

        self.send_button = QPushButton("SEND")
        self.send_button.setFont(self.monospace_font)
        self.send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: {self.bg_color};
                padding: 10px 20px;
                font-weight: bold;
                border: 2px solid {self.border_color};
            }}
            QPushButton:hover {{
                background-color: #ffed4e;
            }}
            QPushButton:pressed {{
                background-color: #ccaa00;
            }}
        """)
        self.send_button.clicked.connect(self.on_chat_submit)
        input_layout.addWidget(self.send_button)
        
        folder_layout.addLayout(input_layout)

        # [NEW] Add Clear Button
        self.clear_button = QPushButton("CLEAR")
        self.clear_button.setFont(self.monospace_font)
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.bg_color};
                color: {self.text_color};
                padding: 10px 20px;
                font-weight: bold;
                border: 2px solid {self.border_color};
            }}
            QPushButton:hover {{
                background-color: #3e2a08;
                border: 2px solid {self.accent_color};
            }}
            QPushButton:pressed {{
                background-color: #4e3a18;
            }}
        """)
        self.clear_button.clicked.connect(self.on_clear_search)
        input_layout.addWidget(self.clear_button)

# [NEW] Add Book Grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {self.folder_bg_color};
                border: 2px solid {self.border_color};
                margin: 16px;
            }}
            QScrollBar:vertical {{
                background-color: {self.bg_color};
                width: 12px;
                border: 1px solid {self.border_color};
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.accent_color};
                min-height: 20px;
            }}
        """)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {self.folder_bg_color};")
        grid_layout = QGridLayout(scroll_content)
        grid_layout.setSpacing(15)
        grid_layout.setContentsMargins(20, 20, 20, 20)
        
        cols = 4
        image_size = 120
        
        for i, book in enumerate(self.books):
            row = i // cols
            col = i % cols
            
            container = QWidget()
            container.setVisible(False)  # ADD THIS - hide by default
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)
            container_layout.setSpacing(5)
            
            
            image_label = QLabel()
            image_label.setFixedSize(image_size, image_size)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setCursor(Qt.PointingHandCursor)
            image_label.setScaledContents(False)
            image_label.setStyleSheet(f"""
                background-color: {self.bg_color};
                border: 2px solid {self.border_color};
                padding: 5px;
            """)
            
            # Load image
            image_path = book.get('Image', '')
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(
                    image_size - 10, 
                    image_size - 10, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                # Show title if no image
                title_text = book.get('Title', 'Unknown')[:15]
                image_label.setText(title_text)
                image_label.setWordWrap(True)
                image_label.setStyleSheet(f"""
                    background-color: {self.bg_color};
                    border: 2px solid {self.border_color};
                    color: {self.text_color};
                    padding: 10px;
                    font-size: 10px;
                """)
            
            # Click handler
            def make_click_handler(idx):
                return lambda event: self.on_book_click(idx)
            image_label.mousePressEvent = make_click_handler(i)
            
            container_layout.addWidget(image_label)
            
            # Book title below image
            title_label = QLabel(book.get('Title', 'Unknown')[:20])
            title_label.setFont(self.monospace_font)
            title_label.setStyleSheet(f"""
                color: {self.text_color};
                font-size: 9px;
                background: transparent;
            """)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setWordWrap(True)
            container_layout.addWidget(title_label)
            
            grid_layout.addWidget(container, row, col, Qt.AlignCenter)
            self.image_labels.append((image_label, container))  # CHANGE: store both label and container
        
        scroll_area.setWidget(scroll_content)
        folder_layout.addWidget(scroll_area, stretch=1)

        # Window label (hidden by default, shown only if dial is active)
        self.window_label = QLabel()
        self.window_label.setAlignment(Qt.AlignRight)
        self.window_label.setFont(self.monospace_font)
        self.window_label.setStyleSheet(f"padding: 6px 12px; font-size: 11px; color: {self.text_color};")
        self.window_label.setVisible(False)
        self.update_window_label()
        folder_layout.addWidget(self.window_label)

        outer_layout.addWidget(folder_frame)

        # Initial greeting
        if self.chat_manager and self.chat_manager.is_available():
            self.append_chat_line("Hello! I'm your CatalogNet library assistant. How can I help you discover your next great read today?")
        else:
            self.append_chat_line("Hello! I'm your CatalogNet library assistant. How can I help you discover your next great read today?")
            self.append_chat_line("(Note: Gemini chat isn't available. Check your API key to enable conversations.)")

        if self.dial_reader:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.sync_dial_value)
            self.timer.start(150)
            self.window_label.setVisible(True)

    def clear_glow_effects(self):
        """Clear all highlights and hide all books"""
        self.previously_glowing = []

        for image_label, container in self.image_labels:
            container.setVisible(False)
            image_label.setStyleSheet(f"""
                background-color: {self.bg_color};
                border: 2px solid {self.border_color};
                padding: 5px;
            """)
        self.push_led_update([])

    def add_glow_to_books(self, indices):
        """Show and highlight only the matching books"""

        for image_label, container in self.image_labels:
            container.setVisible(False)
            image_label.setStyleSheet(f"""
                background-color: {self.bg_color};
                border: 2px solid {self.border_color};
                padding: 5px;
            """)
        
        # Show and highlight only matching books
        if not indices:
            self.push_led_update([])
            return
            
        self.previously_glowing = indices
        matching_books = []
        
        for idx in indices:
            if 0 <= idx < len(self.books):
                book = self.books[idx]
                matching_books.append(f"{book.get('Title', 'Unknown')} by {book.get('Author', 'Unknown')}")
                
                if idx < len(self.image_labels):
                    image_label, container = self.image_labels[idx]
                    container.setVisible(True)
                    image_label.setStyleSheet(f"""
                        background-color: {self.accent_color};
                        border: 3px solid {self.accent_color};
                        padding: 5px;
                    """)
        
        if matching_books:
            print(f"Matching books: {', '.join(matching_books)}")
        
        self.push_led_update(indices)

    def push_led_update(self, indices=None):
        if not self.led_controller:
            return
        indices = indices if indices is not None else self.previously_glowing
        positions = []
        seen = set()
        for idx in indices:
            if 0 <= idx < len(self.books):
                position = self.books[idx].get("Position")
                if isinstance(position, int) and position not in seen:
                    positions.append(position)
                    seen.add(position)
        self.led_controller.send_positions(positions)

    def append_chat_line(self, text):
        if not self.chat_display or not text:
            return
        self.chat_display.append("CatalogNet: " + text)
        scrollbar = self.chat_display.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def on_chat_submit(self):
        if not self.chat_input:
            return
        user_message = self.chat_input.text().strip()
        if not user_message:
            return
        self.chat_input.clear()
        # Display user message in the chat
        self.append_chat_line("You: " + user_message)

        if not self.chat_manager or not self.chat_manager.is_available():
            self.append_chat_line("Gemini chat isn't configured yet.")
            return

        reply_text, search_query = self.chat_manager.send_message(user_message)
        if reply_text:
            self.append_chat_line(reply_text)
        if search_query:
            self.show_query_results(search_query)

    def show_query_results(self, query):
        if not query:
            return
        self.clear_glow_effects()
        self.last_query = query
        indices = get_top_books_by_query(
            query,
            self.books,
            self.book_embeddings,
            top_k=3,
            window=self.current_window_bounds(),
        )
        self.add_glow_to_books(indices)

    def current_window_bounds(self):
        half = WINDOW_SIZE / 2
        min_center = half
        max_center = 1 - half
        if max_center <= min_center:
            return 0.0, 1.0
        center = min_center + (max_center - min_center) * self.dial_value
        low = max(0.0, center - half)
        high = min(1.0, center + half)
        return low, high

    def update_window_label(self):
        low, high = self.current_window_bounds()
        self.window_label.setText(f"Similarity window: {low:.2f} – {high:.2f}")
        if self.dial_reader:
            self.window_label.setVisible(True)

    def sync_dial_value(self):
        if not self.dial_reader:
            return
        new_value = self.dial_reader.get_value()
        if abs(new_value - self.dial_value) < 0.01:
            return
        self.dial_value = new_value
        self.update_window_label()
        self.refresh_highlights()
 
 # [NEW] Click Book
    def on_book_click(self, clicked_idx):
        """Handle clicking on a book cover"""
        if clicked_idx < 0 or clicked_idx >= len(self.books):
            return
        book = self.books[clicked_idx]
        title = book.get('Title', 'Unknown')
        author = book.get('Author', 'Unknown')
        
        self.append_chat_line(f"You selected: {title} by {author}")
        
        similar_indices = get_top_similar_books(
            clicked_idx,
            self.books,
            self.book_embeddings,
            top_k=3,
            window=self.current_window_bounds()
        )
        
        self.add_glow_to_books([clicked_idx])
        
        if similar_indices:
            similar_titles = [self.books[i].get('Title', 'Unknown') for i in similar_indices]
            self.append_chat_line(f"Similar books: {', '.join(similar_titles)}")
        
        try:
            with open("D:/selected_book.txt", "a", encoding="utf-8") as f:
                f.write(title + "\n")  
                f.flush()
            print(f"Added to file: {title}")
        except Exception as e:
            print(f"Error: {e}")


    def refresh_highlights(self):
        """Refresh which books are shown based on current dial value"""
        if not self.last_query:
            return
        window = self.current_window_bounds()
        self.clear_glow_effects()
        indices = get_top_books_by_query(
            self.last_query,
            self.books,
            self.book_embeddings,
            top_k=3,
            window=window,
        )
        self.add_glow_to_books(indices)
        print(f"Refreshed highlights for window {window[0]:.2f}-{window[1]:.2f}")

    # [NEW] Book Similarity Indices
    def get_books_in_window(self):
        """Get indices of books currently in the similarity window"""
        if not self.last_query:
            return []
        
        window = self.current_window_bounds()
        indices = get_top_books_by_query(
            self.last_query,
            self.books,
            self.book_embeddings,
            top_k=3,
            window=window,
        )
        return indices
    
    # [NEW] Clear Search
    def on_clear_search(self):
        """Clear the search and hide all book covers"""
        if self.chat_input:
            self.chat_input.clear()
        try:
            with open("D:/selected_book.txt", "w", encoding="utf-8") as f:
                f.write("")
            print("Cleared selected_book.txt")
        except Exception as e:
            print(f"Error clearing file: {e}")
        
        self.last_query = ""
        self.clear_glow_effects()
        self.append_chat_line("Search cleared. Ask me about books to see recommendations!")
        
        print("Search cleared") 
    
    def refresh_highlights(self):
        if not self.last_query:
            return
        window = self.current_window_bounds()
        self.clear_glow_effects()
        indices = get_top_books_by_query(
            self.last_query,
            self.books,
            self.book_embeddings,
            top_k=3,
            window=window,
        )
        self.add_glow_to_books(indices)


def main():
    books = load_data(DATA_FILE)
    if not books:
        print(f"Error: No books found in {DATA_FILE}")
        return

    book_embeddings = generate_embeddings_for_corpus(books)
    if not book_embeddings or len(book_embeddings) != len(books):
        print("Error: Failed to generate embeddings for all books.")
        return

    dial_reader = None
    led_controller = None
    if serial is None:
        print("pyserial not installed; dial and LED features disabled.")
    else:
        reader = DialReader()
        if reader.port:
            reader.start()
            dial_reader = reader
        else:
            print("DialReader could not find a serial device; dial disabled.")
        led_controller = LEDController()

    chat_manager = ChatManager()

    app = QApplication([])
    window = BookGUI(
        books,
        book_embeddings,
        dial_reader=dial_reader,
        chat_manager=chat_manager,
        led_controller=led_controller,
    )
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
