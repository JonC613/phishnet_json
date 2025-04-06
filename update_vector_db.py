#!/usr/bin/env python

import os
import json
import sys
from create_phish_vector_db import PhishVectorDB, format_show_info
import pandas as pd

def update_vector_db():
    """Load the existing vector database and add new shows from the shows directory"""
    vector_db_path = 'vector_db'
    
    # Check if vector database exists
    if not os.path.exists(os.path.join(vector_db_path, "phish_shows.index")):
        print("Vector database not found. Creating a new one...")
        # Use the main function from create_phish_vector_db.py
        from create_phish_vector_db import main as create_db_main
        import sys
        sys.argv = ['create_phish_vector_db.py']  # Reset argv to avoid issues
        create_db_main()
        return
    
    # Load existing database
    print("Loading existing vector database...")
    db = PhishVectorDB.load(vector_db_path)
    original_count = len(db.shows_data)
    print(f"Loaded {original_count} existing shows")
    
    # Create a set of existing showdates for quick lookup
    existing_dates = set(show.get('showdate') for show in db.shows_data if show.get('showdate'))
    print(f"Existing show dates: {', '.join(sorted(existing_dates))}")
    
    # Load all JSON files from shows directory
    shows_dir = 'shows'
    new_shows_added = 0
    
    for filename in os.listdir(shows_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(shows_dir, filename)
            print(f"Processing {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # Handle both individual show files and year collection files
            shows_to_add = file_data if isinstance(file_data, list) else [file_data]
            
            # Add only shows that don't already exist
            for show in shows_to_add:
                show_date = show.get('showdate')
                if show_date and show_date not in existing_dates:
                    db.shows_data.append(show)
                    existing_dates.add(show_date)
                    new_shows_added += 1
                    print(f"Added new show: {show_date}")
    
    if new_shows_added == 0:
        print("No new shows to add. Database already contains all shows from the shows directory.")
        return
    
    # Update the DataFrame
    db.shows_df = pd.DataFrame(db.shows_data)
    print(f"Added {new_shows_added} new shows. Database now contains {len(db.shows_data)} shows")
    
    # Rebuild the index and save
    print("Rebuilding vector index...")
    db.build_index()
    db.save(vector_db_path)
    print(f"Vector database updated and saved with {len(db.shows_data)} total shows")
    
    # Print summary of shows by year
    years = {}
    for show in db.shows_data:
        show_date = show.get('showdate', '')
        if show_date and len(show_date) >= 4:
            year = show_date[:4]
            years[year] = years.get(year, 0) + 1
    
    print("\nShows by year:")
    for year in sorted(years.keys()):
        print(f"  {year}: {years[year]} shows")

if __name__ == "__main__":
    update_vector_db()