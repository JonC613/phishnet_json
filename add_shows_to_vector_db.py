#!/usr/bin/env python

import os
import sys
from create_phish_vector_db import PhishVectorDB
import json

def add_shows_to_db(show_files=None, rebuild=False):
    """
    Add shows from specific JSON files to the vector database
    If no files specified, it will add all shows from the shows directory
    """
    # Determine if vector database already exists
    vector_db_path = 'vector_db'
    vector_db_exists = os.path.exists(os.path.join(vector_db_path, "phish_shows.index"))
    
    # If rebuilding or db doesn't exist, create a new one
    if rebuild or not vector_db_exists:
        print("Creating new vector database...")
        db = PhishVectorDB()
        shows_dir = 'shows'
        db.load_shows_from_directory(shows_dir)
        print(f"Loaded {len(db.shows_data)} shows")
        
    else:
        # Load existing database
        print(f"Loading existing vector database from {vector_db_path}")
        db = PhishVectorDB.load(vector_db_path)
        print(f"Loaded {len(db.shows_data)} existing shows")
        
        # If show files are specified, add them to the database
        if show_files:
            existing_dates = set(show.get('showdate') for show in db.shows_data if show.get('showdate'))
            shows_added = 0
            
            for file_path in show_files:
                print(f"Processing {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # Handle both individual show files and year collection files
                shows_to_add = file_data if isinstance(file_data, list) else [file_data]
                
                # Add only shows that don't already exist
                new_shows = []
                for show in shows_to_add:
                    if show.get('showdate') and show.get('showdate') not in existing_dates:
                        new_shows.append(show)
                        existing_dates.add(show.get('showdate'))
                
                if new_shows:
                    db.shows_data.extend(new_shows)
                    shows_added += len(new_shows)
                    print(f"Added {len(new_shows)} new shows from {file_path}")
                else:
                    print(f"No new shows to add from {file_path} (shows may already exist in database)")
            
            if shows_added > 0:
                print(f"Total of {shows_added} new shows added to the database")
            else:
                print("No new shows were added to the database")
                
            # Update the DataFrame with the new shows
            import pandas as pd
            db.shows_df = pd.DataFrame(db.shows_data)
    
    # Build or rebuild the index and save
    if rebuild or not vector_db_exists or (show_files and len(show_files) > 0):
        print("Building vector index...")
        db.build_index()
        db.save(vector_db_path)
        print(f"Vector database saved with {len(db.shows_data)} total shows")
    
    return db

def main():
    """Main function to add shows to the vector database"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add shows to the Phish vector database")
    parser.add_argument('--files', nargs='+', help='Specific JSON files to add to the database')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild the entire index')
    parser.add_argument('--year', type=int, help='Add all shows from a specific year')
    args = parser.parse_args()
    
    # If year is specified, find all files for that year
    if args.year:
        year_files = []
        for file in os.listdir('shows'):
            if file.endswith('.json') and str(args.year) in file:
                year_files.append(os.path.join('shows', file))
        
        if year_files:
            print(f"Found {len(year_files)} files for year {args.year}")
            files_to_add = year_files
        else:
            print(f"No show files found for year {args.year}")
            return
    else:
        files_to_add = args.files if args.files else None
    
    # Add shows to the database
    db = add_shows_to_db(show_files=files_to_add, rebuild=args.rebuild)
    
    # Display some statistics
    years = set()
    for show in db.shows_data:
        if show.get('showdate') and len(show.get('showdate', '')) >= 4:
            years.add(show.get('showdate')[:4])
    
    print(f"\nVector database now contains {len(db.shows_data)} shows from {len(years)} years")
    print(f"Years included: {', '.join(sorted(years))}")

if __name__ == "__main__":
    main()