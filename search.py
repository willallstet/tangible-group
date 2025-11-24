import json
import numpy as np
import ollama
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QScrollArea, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Configuration
DATA_FILE = 'books.json'
MODEL_NAME = 'nomic-embed-text'

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return []
    with open(filename, 'r') as f:
        return json.load(f)

def get_embedding(text):
    """Generates a vector embedding using the local Ollama model."""
    try:
        response = ollama.embeddings(model=MODEL_NAME, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def generate_embeddings_for_corpus(books):
    """
    Generates embeddings for all books.
    We combine Title + Description for a richer semantic representation.
    """
    print(f"Generating embeddings for {len(books)} books using {MODEL_NAME}...")
    embeddings = []
    for i, book in enumerate(books):
        # Construct a rich text representation for the embedding
        text_content = f"Title: {book['Title']}. Description: {book['Description']}"
        
        emb = get_embedding(text_content)
        embeddings.append(emb)
        
        # Simple progress indicator
        print(f"[{i+1}/{len(books)}] Processed '{book['Title']}'")
        
    return embeddings

def search_by_query(query, books, book_embeddings, top_k=3):
    """Mode 1: User inputs text -> Find similar books."""
    query_vec = get_embedding(query)
    if not query_vec:
        return

    scores = []
    for i, book_vec in enumerate(book_embeddings):
        score = cosine_similarity(query_vec, book_vec)
        scores.append((score, books[i]))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n--- Results for query: '{query}' ---")
    for score, book in scores[:top_k]:
        print(f"[Score: {score:.4f}] {book['Title']} by {book['Author']}")

def search_by_book_similarity(target_idx, books, book_embeddings, top_k=3):
    """Mode 2: Select a book -> Find other books similar to it."""
    target_book = books[target_idx]
    target_vec = book_embeddings[target_idx]
    
    print(f"\nFinding books similar to: '{target_book['Title']}'...")
    
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        # Skip the book itself
        if i == target_idx:
            continue
            
        score = cosine_similarity(target_vec, book_vec)
        scores.append((score, books[i]))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    print(f"--- Recommendations based on '{target_book['Title']}' ---")
    for score, book in scores[:top_k]:
        print(f"[Score: {score:.4f}] {book['Title']} by {book['Author']}")

def get_top_similar_books(target_idx, books, book_embeddings, top_k=2):
    """Returns the indices of the top k most similar books (excluding the target)."""
    target_vec = book_embeddings[target_idx]
    
    scores = []
    for i, book_vec in enumerate(book_embeddings):
        # Skip the book itself
        if i == target_idx:
            continue
            
        score = cosine_similarity(target_vec, book_vec)
        scores.append((score, i))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the indices of the top k books
    return [idx for score, idx in scores[:top_k]]

class BookGUI(QMainWindow):
    def __init__(self, books, book_embeddings):
        super().__init__()
        self.books = books
        self.book_embeddings = book_embeddings
        self.title_labels = []
        self.previously_bolded = []
        
        self.setWindowTitle("Book Similarity Browser")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title label
        title_label = QLabel("Click a book title to see the top 2 most related books:")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create scrollable content widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)
        
        # Create clickable labels for each book title
        for i, book in enumerate(books):
            # Create a frame for each book title to make it more clickable
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("QFrame { border: 1px solid transparent; }"
                              "QFrame:hover { border: 1px solid #ccc; background-color: #f0f0f0; }")
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(10, 5, 10, 5)
            
            label = QLabel(book['Title'])
            label.setFont(QFont("Arial", 11))
            label.setStyleSheet("color: #0066cc;")
            label.setCursor(Qt.PointingHandCursor)
            # Use a closure to properly capture the index
            def make_click_handler(idx):
                return lambda event: self.on_title_click(idx)
            label.mousePressEvent = make_click_handler(i)
            label.setWordWrap(True)
            
            frame_layout.addWidget(label)
            scroll_layout.addWidget(frame)
            self.title_labels.append(label)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Status label
        self.status_label = QLabel("Ready")
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_label.setFont(status_font)
        main_layout.addWidget(self.status_label)
    
    def on_title_click(self, clicked_idx):
        """Handle click on a book title - bold the top 2 most related titles."""
        # Reset previously bolded titles
        for idx in self.previously_bolded:
            if 0 <= idx < len(self.title_labels):
                font = QFont("Arial", 11)
                self.title_labels[idx].setFont(font)
        
        self.previously_bolded = []
        
        # Get top 2 similar books
        similar_indices = get_top_similar_books(clicked_idx, self.books, self.book_embeddings, top_k=2)
        
        # Bold the similar books
        for idx in similar_indices:
            if 0 <= idx < len(self.title_labels):
                font = QFont("Arial", 11)
                font.setBold(True)
                self.title_labels[idx].setFont(font)
                self.previously_bolded.append(idx)
        
        # Update status
        clicked_title = self.books[clicked_idx]['Title']
        similar_titles = [self.books[idx]['Title'] for idx in similar_indices]
        self.status_label.setText(
            f"Selected: {clicked_title} | Most related: {', '.join(similar_titles)}"
        )

def main():
    # 1. Load Data
    books = load_data(DATA_FILE)
    if not books:
        print(f"Error: No books found in {DATA_FILE}")
        return

    print(f"Loaded {len(books)} books from {DATA_FILE}")

    # 2. Generate Vectors (In a real app, you would save/cache these to disk)
    book_embeddings = generate_embeddings_for_corpus(books)
    
    if not book_embeddings or len(book_embeddings) != len(books):
        print("Error: Failed to generate embeddings for all books.")
        return

    # 3. Launch PyQt GUI
    print("\nLaunching GUI...")
    app = QApplication([])
    window = BookGUI(books, book_embeddings)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()