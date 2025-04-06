#!/usr/bin/env python

import os
import json
import glob
import numpy as np
import pandas as pd
import faiss
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

class PhishVectorDBNew:
    def __init__(self, vector_size=128):
        """Initialize the vector database with the specified vector size"""
        self.vector_size = vector_size
        self.vectorizer = TfidfVectorizer(max_features=vector_size)
        self.index = None
        self.shows_data = []
        self.shows_df = None
    
    def load_shows_from_directory(self, directory_path):
        """Load all JSON files from the directory containing show data"""
        print(f"Loading shows from {directory_path}...")
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        
        all_shows = []
        for file_path in tqdm(json_files, desc="Loading JSON files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # Handle both individual show files and year collection files
                if isinstance(file_data, list):
                    all_shows.extend(file_data)
                else:
                    all_shows.append(file_data)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # Filter out any potential empty shows
        self.shows_data = [show for show in all_shows if show and isinstance(show, dict)]
        print(f"Loaded {len(self.shows_data)} shows")
        
        # Convert to DataFrame for easier processing
        self.shows_df = pd.DataFrame(self.shows_data)
        return self.shows_data
    
    def create_text_representations(self):
        """Create text representations for each show that capture its essence"""
        if self.shows_df is None or len(self.shows_df) == 0:
            raise ValueError("No shows data loaded. Call load_shows_from_directory first.")
        
        print("Creating text representations for vector embedding...")
        text_representations = []
        
        for _, show in tqdm(self.shows_df.iterrows(), total=len(self.shows_df), desc="Processing shows"):
            # Initialize with basic show info
            text = f"Date: {show.get('showdate', '')} "
            text += f"Venue: {show.get('venue', '')} "
            text += f"Location: {show.get('city', '')} {show.get('state', '')} {show.get('country', '')} "
            
            if pd.notna(show.get('tour_name')) and show.get('tour_name'):
                text += f"Tour: {show.get('tour_name')} "
            
            # Add setlist information
            sets = show.get('sets', {})
            if isinstance(sets, dict):
                for set_name, songs in sets.items():
                    set_display = set_name.upper() if set_name.lower() in ['e', 'e2'] else f"Set {set_name}"
                    text += f"{set_display}: "
                    
                    # Add songs for this set
                    if isinstance(songs, list):
                        song_texts = []
                        for song in songs:
                            if isinstance(song, dict) and song.get('song'):
                                song_text = song.get('song', '')
                                # Add jam indicator
                                if song.get('jam'):
                                    song_text += " (jam)"
                                song_texts.append(song_text)
                        
                        text += ", ".join(song_texts) + " "
            
            # Add notes if available
            if pd.notna(show.get('setlist_notes')) and show.get('setlist_notes'):
                # Strip HTML tags from notes for better text representation
                notes = show.get('setlist_notes', '')
                notes = notes.replace('<p>', ' ').replace('</p>', ' ').replace('<br>', ' ')
                notes = notes.replace('\r\n', ' ').replace('\n', ' ')
                text += f"Notes: {notes}"
            
            text_representations.append(text)
        
        return text_representations
    
    def build_index(self):
        """Build the FAISS index from show data"""
        if self.shows_df is None or len(self.shows_df) == 0:
            raise ValueError("No shows data loaded. Call load_shows_from_directory first.")
        
        # Create text representations
        text_representations = self.create_text_representations()
        
        # Create embeddings using TF-IDF
        print("Creating embeddings...")
        self.vectorizer.fit(text_representations)
        embeddings = self.vectorizer.transform(text_representations).toarray()
        
        # Normalize the vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Ensure we have the correct dimensionality
        actual_dim = embeddings.shape[1]
        if actual_dim != self.vector_size:
            print(f"Note: Actual vector dimension is {actual_dim} (requested was {self.vector_size})")
            self.vector_size = actual_dim
        
        # Create and populate the index
        print(f"Building FAISS index with dimension {self.vector_size}...")
        self.index = faiss.IndexFlatL2(self.vector_size)
        self.index.add(np.array(embeddings).astype(np.float32))
        
        print(f"Vector database created with {self.index.ntotal} shows")
        self.text_representations = text_representations  # Save for later use
        return self.index
    
    def search(self, query, k=5):
        """Search for shows similar to the query"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Create embedding for the query
        query_vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        
        # Normalize the query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Search the index
        distances, indices = self.index.search(query_vector, k=k)
        
        # Return the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.shows_data):
                show = self.shows_data[idx]
                results.append({
                    "show": show,
                    "distance": float(distances[0][i]),
                    "similarity_score": 1.0 / (1.0 + float(distances[0][i]))
                })
        
        return results
    
    def save(self, output_dir="vector_db_new"):
        """Save the vector database to disk"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, os.path.join(output_dir, "phish_shows.index"))
        
        # Save the shows data
        with open(os.path.join(output_dir, "phish_shows_data.pkl"), 'wb') as f:
            pickle.dump(self.shows_data, f)
            
        # Save the vectorizer
        with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Vector database saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir="vector_db_new"):
        """Load a previously saved vector database"""
        db = cls()
        
        # Load the index
        db.index = faiss.read_index(os.path.join(input_dir, "phish_shows.index"))
        db.vector_size = db.index.d  # Set the correct dimension from the loaded index
        
        # Load the shows data
        with open(os.path.join(input_dir, "phish_shows_data.pkl"), 'rb') as f:
            db.shows_data = pickle.load(f)
        
        # Load the vectorizer
        with open(os.path.join(input_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
            db.vectorizer = pickle.load(f)
        
        # Create DataFrame
        db.shows_df = pd.DataFrame(db.shows_data)
        
        print(f"Loaded vector database with {len(db.shows_data)} shows")
        return db

def format_show_info(show):
    """Format a show for display"""
    output = []
    output.append("Phish Show Information:")
    output.append("-" * 50)
    
    # Basic show information
    output.append(f"Date: {show.get('showdate', 'N/A')}")
    output.append(f"Venue: {show.get('venue', 'N/A')}")
    output.append(f"Location: {show.get('city', 'N/A')}, {show.get('state', 'N/A')}, {show.get('country', 'N/A')}")
    if show.get('tour_name'):
        output.append(f"Tour: {show.get('tour_name')}")
    
    # Setlist information
    if show.get('sets'):
        output.append("\nSetlist:")
        for set_name in sorted(show.get('sets', {}).keys()):
            set_songs = show.get('sets', {}).get(set_name, [])
            if set_name.lower() == 'e':
                output.append("\nEncore:")
            elif set_name.lower() == 'e2':
                output.append("\nEncore 2:")
            else:
                output.append(f"\nSet {set_name}:")
            
            # Format songs with transitions
            songs_formatted = []
            for song in set_songs:
                song_str = song.get('song', '')
                if song.get('jam'):
                    song_str += '*'
                if song.get('footnote'):
                    song_str += f" [{song.get('footnote')}]"
                songs_formatted.append(song_str)
                
            output.append(" > ".join(songs_formatted) if any(s.get('transition') for s in set_songs) else ", ".join(songs_formatted))
    
    if show.get('setlist_notes'):
        # Strip HTML tags for cleaner display
        notes = show.get('setlist_notes', '')
        notes = notes.replace('<p>', '').replace('</p>', '\n').replace('<br>', '\n')
        notes = notes.replace('\r\n', '\n')
        output.append(f"\nNotes: {notes}")
    
    return "\n".join(output)

def main():
    """Main function to build and test the vector database"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and search a vector database of Phish shows")
    parser.add_argument('--data_dir', default='shows', help='Directory containing show JSON files')
    parser.add_argument('--output_dir', default='vector_db_new', help='Directory to save the vector database')
    parser.add_argument('--search', action='store_true', help='Search the database after creating it')
    parser.add_argument('--test', action='store_true', help='Run test queries after creating the database')
    parser.add_argument('--vector_size', type=int, default=128, help='Maximum size of the vector embeddings')
    args = parser.parse_args()
    
    # Create new vector database
    print(f"Creating new vector database with data from {args.data_dir}...")
    db = PhishVectorDBNew(vector_size=args.vector_size)
    db.load_shows_from_directory(args.data_dir)
    db.build_index()
    db.save(args.output_dir)
    
    # Show summary of years
    years = {}
    for show in db.shows_data:
        show_date = show.get('showdate', '')
        if show_date and len(show_date) >= 4:
            year = show_date[:4]
            years[year] = years.get(year, 0) + 1
    
    print("\nVector database summary:")
    print(f"Total shows: {len(db.shows_data)}")
    print("Shows by year:")
    for year in sorted(years.keys()):
        print(f"  {year}: {years[year]} shows")
    
    # Run test queries if requested
    if args.test:
        test_queries = [
            "Shows with Tweezer",
            "Shows in New York",
            "Summer tour",
            "Shows with jams",
        ]
        
        print("\nTesting vector database with sample queries:")
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = db.search(query, k=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    show = result['show']
                    score = result['similarity_score']
                    date = show.get('showdate', 'Unknown date')
                    venue = show.get('venue', 'Unknown venue')
                    print(f"Result #{i} (Score: {score:.2f}): {date} - {venue}")
            else:
                print("No results found.")
    
    # Interactive search if requested
    if args.search:
        print("\nSearch mode:")
        while True:
            query = input("\nEnter a search query (or 'quit' to exit): ")
            if query.lower() in ('quit', 'exit', 'q'):
                break
            
            print(f"Searching for: {query}")
            results = db.search(query, k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    show = result['show']
                    score = result['similarity_score']
                    print(f"\nResult #{i} (Similarity: {score:.2f}):")
                    print(format_show_info(show))
                    print("-" * 70)
            else:
                print("No results found.")
    
    print(f"\nVector database created successfully in: {args.output_dir}")
    print("You can use this database for searching Phish shows with the command:")
    print(f"python create_new_vector_db.py --output_dir {args.output_dir} --search")

if __name__ == "__main__":
    main()