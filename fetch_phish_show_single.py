import requests
import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

class PhishNetAPI:
    def __init__(self):
        load_dotenv()
        self.base_url = "https://api.phish.net/v5"
        self.api_key = os.getenv("PHISHNET_API_KEY")
        if not self.api_key:
            raise ValueError("Please set PHISHNET_API_KEY in your .env file")
        self.api_key = self.api_key.strip()

    def get_show_by_date(self, date):
        """Get show information for a specific date in YYYY-MM-DD format"""
        return self._get_show_details(date)

    def _get_show_details(self, show_date):
        show_data = {}
        params = {'apikey': self.api_key}
        
        # Get setlist directly using showdate endpoint
        setlist_url = f"{self.base_url}/setlists/showdate/{show_date}.json"
        print(f"\nFetching setlist from: {setlist_url}")
        
        try:
            response = requests.get(setlist_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            setlist_data = response.json().get('data', [])
            if setlist_data:
                # Initialize show data from first entry
                show_data.update({
                    'showdate': setlist_data[0].get('showdate'),
                    'venue': setlist_data[0].get('venue'),
                    'city': setlist_data[0].get('city'),
                    'state': setlist_data[0].get('state'),
                    'country': setlist_data[0].get('country'),
                    'setlist_notes': setlist_data[0].get('setlistnotes'),
                    'tour_name': setlist_data[0].get('tourname'),
                    'sets': {}
                })
                
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
        except requests.exceptions.RequestException as e:
            print(f"Error fetching show data for date {show_date}: {str(e)}")
            return {}
        
        if not show_data:
            print(f"No show data found for date: {show_date}")
            
        return show_data

def format_show_info(show_data):
    if not show_data:
        return "No show data available."
        
    output = []
    output.append("Phish Show Information:")
    output.append("-" * 50)
    
    # Basic show information
    output.append(f"Date: {show_data.get('showdate', 'N/A')}")
    output.append(f"Venue: {show_data.get('venue', 'N/A')}")
    output.append(f"Location: {show_data.get('city', 'N/A')}, {show_data.get('state', 'N/A')}, {show_data.get('country', 'N/A')}")
    if show_data.get('tour_name'):
        output.append(f"Tour: {show_data['tour_name']}")
    
    # Setlist information
    if show_data.get('sets'):
        output.append("\nSetlist:")
        for set_name in sorted(show_data['sets'].keys()):
            set_songs = show_data['sets'][set_name]
            if set_name == 'E':
                output.append("\nEncore:")
            elif set_name == 'E2':
                output.append("\nEncore 2:")
            else:
                output.append(f"\nSet {set_name}:")
            
            # Format songs with transitions
            songs_formatted = []
            for song in set_songs:
                song_str = song['song']
                if song.get('jam'):
                    song_str += '*'
                if song.get('footnote'):
                    song_str += f" [{song['footnote']}]"
                songs_formatted.append(song_str)
                
            output.append(" > ".join(songs_formatted) if any(s.get('transition') for s in set_songs) else ", ".join(songs_formatted))
    
    if show_data.get('setlist_notes'):
        output.append(f"\nNotes: {show_data['setlist_notes']}")
    
    return "\n".join(output)

def validate_date_format(date_str):
    """Validate date string is in YYYY-MM-DD format and is a valid date"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def main():
    parser = argparse.ArgumentParser(description='Fetch Phish show data for a specific date')
    parser.add_argument('date', help='Date in YYYY-MM-DD format (e.g., 1999-07-24)')
    parser.add_argument('--latest', action='store_true', help='Fetch the latest show instead of a specific date')
    parser.add_argument('--no-save', action='store_true', help='Display show info but don\'t save to file')
    args = parser.parse_args()

    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Warning: .env file not found. You'll need to set PHISHNET_API_KEY environment variable.")

    try:
        api = PhishNetAPI()
        
        if args.latest:
            # For the latest show, we'd ideally query the API for the most recent show
            # For now, using current date as a simple approximation
            show_date = datetime.now().strftime('%Y-%m-%d')
            print(f"Fetching the latest show (using today's date: {show_date})")
        else:
            show_date = args.date
            if not validate_date_format(show_date):
                print("Error: Invalid date format. Please use YYYY-MM-DD format.")
                sys.exit(1)
        
        show_data = api.get_show_by_date(show_date)
        
        if not show_data:
            print(f"No show found for date: {show_date}")
            sys.exit(1)
            
        print(format_show_info(show_data))
        
        # Save to file if requested
        if not args.no_save:
            os.makedirs('shows', exist_ok=True)
            output_file = f"shows/show_{show_date}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(show_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nData saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()