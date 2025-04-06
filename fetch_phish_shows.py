import requests
import json
import sys
import os
import argparse
from datetime import datetime
import time
from dotenv import load_dotenv

class PhishNetAPI:
    def __init__(self, rate_limit_delay=1.0):
        load_dotenv()  # Load environment variables from .env file
        self.base_url = "https://api.phish.net/v5"
        self.api_key = os.getenv("PHISHNET_API_KEY")
        if not self.api_key:
            raise ValueError("Please set PHISHNET_API_KEY in your .env file")
        self.api_key = self.api_key.strip()
        self.rate_limit_delay = rate_limit_delay

    def get_shows_by_year(self, year, limit=None):
        params = {
            'apikey': self.api_key
        }

        url = f"{self.base_url}/shows/showyear/{year}.json"
        print(f"Fetching shows for {year}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch shows: {str(e)}")
        
        response_json = response.json()
        shows_data = response_json.get('data', [])
        
        # Filter for Phish shows only (artistid = 1)
        shows_data = [show for show in shows_data if int(show.get('artistid', 0)) == 1]
        
        # Apply limit if specified
        if limit and limit > 0:
            shows_data = shows_data[:limit]
            print(f"Processing {len(shows_data)} Phish shows for {year} (limited to {limit})")
        else:
            print(f"Processing {len(shows_data)} Phish shows for {year}")

        formatted_shows = []
        total_shows = len(shows_data)
        
        for i, show in enumerate(shows_data, 1):
            # Extract the showdate instead of showid
            show_date = show.get('showdate')
            if not show_date:
                print(f"Warning: Show at index {i} has no date, skipping")
                continue
                
            print(f"Fetching details for show {i}/{total_shows} (Date: {show_date})")
            show_details = self._get_show_details(show_date)
            if show_details and show_details.get('showdate'):  # Only include shows with valid data
                formatted_shows.append(show_details)
            
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)

        return formatted_shows

    def _format_setlist(self, setlist_data):
        if not setlist_data:
            return {}
        
        sets = {}
        for song in setlist_data:
            set_name = song.get('set', '1')  # Default to set 1 if not specified
            if set_name not in sets:
                sets[set_name] = []
                
            song_entry = {
                "song": song.get('song', ''),
                "transition": bool(song.get('transition')),
            }
            
            if song.get('footnote'):
                song_entry["footnote"] = song.get('footnote')
                
            sets[set_name].append(song_entry)
            
        return sets

    def _get_show_details(self, show_date):
        """Get show details using the showdate endpoint instead of showid"""
        try:
            # Use the same approach as fetch_phish_show_single.py
            params = {'apikey': self.api_key}
            
            # Get setlist directly using showdate endpoint
            setlist_url = f"{self.base_url}/setlists/showdate/{show_date}.json"
            
            try:
                response = requests.get(setlist_url, params=params)
                response.raise_for_status()
                
                setlist_data = response.json().get('data', [])
                if not setlist_data:
                    print(f"No setlist data found for date: {show_date}")
                    return None
                
                # Initialize show data from first entry
                show_data = {
                    'showdate': setlist_data[0].get('showdate'),
                    'venue': setlist_data[0].get('venue'),
                    'city': setlist_data[0].get('city'),
                    'state': setlist_data[0].get('state'),
                    'country': setlist_data[0].get('country'),
                    'setlist_notes': setlist_data[0].get('setlistnotes'),
                    'tour_name': setlist_data[0].get('tourname'),
                    'sets': {}
                }
                
                # Process each song and organize by set
                for song in setlist_data:
                    set_name = song.get('set', '')
                    if set_name:
                        if set_name not in show_data['sets']:
                            show_data['sets'][set_name] = []
                        
                        song_entry = {
                            'song': song.get('song', ''),
                            'transition': song.get('transition') == 1
                        }
                        
                        # Add additional song details if present
                        if song.get('isjam') == 1:
                            song_entry['jam'] = True
                        if song.get('footnote'):
                            song_entry['footnote'] = song.get('footnote')
                            
                        show_data['sets'][set_name].append(song_entry)
                
                # Ensure we have at least basic show data
                if show_data['showdate'] and show_data['venue']:
                    return show_data
                return None
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching show data for date {show_date}: {str(e)}")
                return None

        except Exception as e:
            print(f"Error processing show {show_date}: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Fetch Phish shows data for a specific year')
    parser.add_argument('year', type=int, help='Year to fetch shows from (1983-present)')
    parser.add_argument('--limit', type=int, help='Limit the number of shows to fetch (for testing)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API requests in seconds (default: 1.0)')
    args = parser.parse_args()

    year = args.year
    if year < 1983 or year > datetime.now().year:
        print(f"Error: Please provide a valid year (1983-{datetime.now().year})")
        sys.exit(1)

    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Warning: .env file not found. You'll need to set PHISHNET_API_KEY environment variable.")

    try:
        api = PhishNetAPI(rate_limit_delay=args.delay)
        shows = api.get_shows_by_year(year, limit=args.limit)

        if not shows:
            print(f"No shows found for {year}")
            sys.exit(0)

        # Create output directory if it doesn't exist
        os.makedirs('shows', exist_ok=True)

        # Write to JSON file
        output_file = f"shows/phish_shows_{year}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(shows, f, indent=2, ensure_ascii=False)

        print(f"\nSuccessfully downloaded {len(shows)} shows from {year}")
        print(f"Data saved to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()