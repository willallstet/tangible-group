import json
import numpy as np
import ollama
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QLabel, QScrollArea, QLineEdit, QVBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# Configuration
DATA_FILE = 'books.json'
MODEL_NAME = 'nomic-embed-text'

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_embedding(text):
    try:
        response = ollama.embeddings(model=MODEL_NAME, prompt=text)
        return response['embedding']
    except Exception as e:
        print(e)
        return []

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def generate_embeddings_for_corpus(books):
    embeddings = []
    for i, book in enumerate(books):
        text_content = f"Title: {book['Title']}. Description: {book['Description']}. Year Published: {book['Year Published']}. Author: {book['Author']}."
        
        emb = get_embedding(text_content)
        embeddings.append(emb)
                
    return embeddings

def search_by_query(query, books, book_embeddings, top_k=3):
    query_vec = get_embedding(query)
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        score = cosine_similarity(query_vec, book_vec)
        scores.append((score, books[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    
    for score, book in scores[:top_k]:
        print(f"[Score: {score:.4f}] {book['Title']} by {book['Author']}")

def search_by_book_similarity(target_idx, books, book_embeddings, top_k=3):
    target_book = books[target_idx]
    target_vec = book_embeddings[target_idx]
        
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        if i == target_idx:
            continue
        score = cosine_similarity(target_vec, book_vec)
        scores.append((score, books[i]))

    scores.sort(key=lambda x: x[0], reverse=True)


def get_top_similar_books(target_idx, books, book_embeddings, top_k=2):
    target_vec = book_embeddings[target_idx]
    SIMILARITY_THRESHOLD = 0.55
    
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        if i == target_idx:
            continue
            
        score = cosine_similarity(target_vec, book_vec)
        if score >= SIMILARITY_THRESHOLD:
            scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    
    return [idx for score, idx in scores[:top_k]]

def get_top_books_by_query(query, books, book_embeddings, top_k=3):
    query_vec = get_embedding(query)
    if not query_vec:
        return []
    
    SIMILARITY_THRESHOLD = 0.55

    scores = []
    for i, book_vec in enumerate(book_embeddings):
        score = cosine_similarity(query_vec, book_vec)
        if score >= SIMILARITY_THRESHOLD:
            scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [idx for score, idx in scores[:top_k]]

class BookGUI(QMainWindow):
    def __init__(self, books, book_embeddings):
        super().__init__()
        self.books = books
        self.book_embeddings = book_embeddings
        self.image_labels = []
        self.previously_glowing = []
        self.glow_effects = []  
        self.search_box = None  
        
        self.setWindowTitle("Book Similarity Browser")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: white;")
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: white;")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search for books...")
        self.search_box.setStyleSheet("""
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
        """)
        self.search_box.returnPressed.connect(self.on_search)
        main_layout.addWidget(self.search_box)
        
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
            
            image_path = book.get('Image', '')
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(image_size, image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("No Image")
                image_label.setStyleSheet("border: 1px solid #ccc; color: #999; background-color: transparent;")
            
            def make_click_handler(idx):
                return lambda event: self.on_image_click(idx)
            image_label.mousePressEvent = make_click_handler(i)
            
            grid_layout.addWidget(image_label, row, col, Qt.AlignCenter)
            self.image_labels.append(image_label)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
    
    def clear_glow_effects(self):
        for idx in self.previously_glowing:
            if 0 <= idx < len(self.image_labels):
                self.image_labels[idx].setStyleSheet("background-color: transparent;")
        self.glow_effects = []
        self.previously_glowing = []
    
    def add_glow_to_books(self, indices):
        for idx in indices:
            if 0 <= idx < len(self.image_labels):
                self.image_labels[idx].setStyleSheet("background-color: yellow;")
                self.previously_glowing.append(idx)
    
    def on_image_click(self, clicked_idx):
        self.clear_glow_effects()
        
        similar_indices = get_top_similar_books(clicked_idx, self.books, self.book_embeddings, top_k=2)
        
        self.add_glow_to_books(similar_indices)
    
    def on_search(self):
        query = self.search_box.text().strip()
        
        if not query:
            self.clear_glow_effects()
            return
        
        self.clear_glow_effects()
        
        matching_indices = get_top_books_by_query(query, self.books, self.book_embeddings, top_k=3)
        
        self.add_glow_to_books(matching_indices)

def main():
    books = load_data(DATA_FILE)
    if not books:
        print(f"Error: No books found in {DATA_FILE}")
        return

    book_embeddings = generate_embeddings_for_corpus(books)
    
    if not book_embeddings or len(book_embeddings) != len(books):
        print("Error: Failed to generate embeddings for all books.")
        return
    app = QApplication([])
    window = BookGUI(books, book_embeddings)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()