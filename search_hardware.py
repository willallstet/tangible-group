import json
import glob
import threading
import numpy as np
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QLabel,
    QScrollArea,
    QLineEdit,
    QVBoxLayout,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

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
WINDOW_SIZE = 0.25  # similarity window width
GEMINI_CHAT_MODEL = "models/gemini-2.5-flash-lite"
CHAT_SYSTEM_PROMPT = (
    "You are a friendly book guide. Ask light follow-up questions when helpful, "
    "then propose recommendations. After each reply include a line exactly like "
    "'SEARCH_QUERY: <concise keywords>' describing what you will search for next. "
    "Use 'SEARCH_QUERY: NONE' only when no search should be run."
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

    def run(self):
        if serial is None or not self.port:
            return
        try:
            device = serial.Serial(self.port, self.baudrate, timeout=1)
        except Exception as exc:
            print(f"DialReader could not open {self.port}: {exc}")
            return

        with device:
            while True:
                line = device.readline().strip()
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


def load_data(filename):
    with open(filename, "r") as f:
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


class BookGUI(QMainWindow):
    def __init__(self, books, book_embeddings, dial_reader=None, chat_manager=None):
        super().__init__()
        self.books = books
        self.book_embeddings = book_embeddings
        self.dial_reader = dial_reader
        self.dial_value = dial_reader.get_value() if dial_reader else 0.5
        self.chat_manager = chat_manager
        self.image_labels = []
        self.previously_glowing = []
        self.chat_display = None
        self.chat_input = None
        self.window_label = None
        self.timer = None
        self.last_clicked_idx = None
        self.last_query = ""

        self.setWindowTitle("Book Similarity Browser (Dial)")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: white;")

        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: white;")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "padding: 10px; font-size: 13px; background-color: #fafafa; border: none; color: #222;"
        )
        main_layout.addWidget(self.chat_display)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Tell me what kind of book you want...")
        self.chat_input.setStyleSheet(
            """
            QLineEdit {
                padding: 10px;
                font-size: 14px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                color: black;
            }
            QLineEdit:focus {
                border: 1px solid #0066ff;
            }
        """
        )
        self.chat_input.returnPressed.connect(self.on_chat_submit)
        main_layout.addWidget(self.chat_input)
        if self.chat_manager and self.chat_manager.is_available():
            self.append_chat_line("Bot: Hi! Tell me what you're in the mood to read.")
        else:
            self.append_chat_line(
                "Bot: Gemini chat isn't available. Check your API key to enable conversations."
            )

        self.window_label = QLabel()
        self.window_label.setAlignment(Qt.AlignRight)
        self.window_label.setStyleSheet("padding: 6px 12px; font-size: 12px; color: #555;")
        self.update_window_label()
        main_layout.addWidget(self.window_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("background-color: white; border: none;")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: white;")
        grid_layout = QGridLayout(scroll_content)
        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(20, 20, 20, 20)

        cols = 4
        image_size = 200
        for i, book in enumerate(books):
            row = i // cols
            col = i % cols
            image_label = QLabel()
            image_label.setFixedSize(image_size, image_size)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setCursor(Qt.PointingHandCursor)
            image_label.setScaledContents(False)
            image_label.setStyleSheet("background-color: transparent;")

            image_path = book.get("Image", "")
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(
                    image_size, image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("No Image")
                image_label.setStyleSheet(
                    "border: 1px solid #ccc; color: #999; background-color: transparent;"
                )

            def make_click_handler(idx):
                return lambda event: self.on_image_click(idx)

            image_label.mousePressEvent = make_click_handler(i)

            grid_layout.addWidget(image_label, row, col, Qt.AlignCenter)
            self.image_labels.append(image_label)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        if self.dial_reader:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.sync_dial_value)
            self.timer.start(150)

    def clear_glow_effects(self):
        for idx in self.previously_glowing:
            if 0 <= idx < len(self.image_labels):
                self.image_labels[idx].setStyleSheet("background-color: transparent;")
        self.previously_glowing = []

    def add_glow_to_books(self, indices):
        for idx in indices:
            if 0 <= idx < len(self.image_labels):
                self.image_labels[idx].setStyleSheet("background-color: yellow;")
                self.previously_glowing.append(idx)

    def append_chat_line(self, text):
        if not self.chat_display or not text:
            return
        self.chat_display.append(text)
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
        self.append_chat_line(f"You: {user_message}")

        if not self.chat_manager or not self.chat_manager.is_available():
            self.append_chat_line("Bot: Gemini chat isn't configured yet.")
            return

        reply_text, search_query = self.chat_manager.send_message(user_message)
        if reply_text:
            self.append_chat_line(f"Bot: {reply_text}")
        if search_query:
            self.append_chat_line(f"(Searching for: {search_query})")
            self.show_query_results(search_query)

    def show_query_results(self, query):
        if not query:
            return
        self.clear_glow_effects()
        self.last_query = query
        self.last_clicked_idx = None
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

    def sync_dial_value(self):
        if not self.dial_reader:
            return
        new_value = self.dial_reader.get_value()
        if abs(new_value - self.dial_value) < 0.01:
            return
        self.dial_value = new_value
        self.update_window_label()
        self.refresh_highlights()

    def refresh_highlights(self):
        window = self.current_window_bounds()
        if self.last_clicked_idx is not None:
            self.clear_glow_effects()
            indices = get_top_similar_books(
                self.last_clicked_idx,
                self.books,
                self.book_embeddings,
                top_k=2,
                window=window,
            )
            self.add_glow_to_books(indices)
        elif self.last_query:
            self.clear_glow_effects()
            indices = get_top_books_by_query(
                self.last_query,
                self.books,
                self.book_embeddings,
                top_k=3,
                window=window,
            )
            self.add_glow_to_books(indices)

    def on_image_click(self, clicked_idx):
        self.clear_glow_effects()
        self.last_clicked_idx = clicked_idx
        self.last_query = ""
        indices = get_top_similar_books(
            clicked_idx,
            self.books,
            self.book_embeddings,
            top_k=2,
            window=self.current_window_bounds(),
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
    if serial is None:
        print("pyserial not installed; dial disabled.")
    else:
        reader = DialReader()
        if reader.port:
            reader.start()
            dial_reader = reader
        else:
            print("DialReader could not find a serial device; dial disabled.")

    chat_manager = ChatManager()

    app = QApplication([])
    window = BookGUI(books, book_embeddings, dial_reader=dial_reader, chat_manager=chat_manager)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

