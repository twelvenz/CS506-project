import os
import sys
import time
import re
import pandas as pd
import requests
import urllib.parse

# CHANGE THIS if you are running it on the newly aggregated dataset!
file_name = 'top_artists_dataset.csv' 

if not os.path.exists(file_name):
    print(f"ERROR: Could not find '{file_name}'!")
    sys.exit()

df_top = pd.read_csv(file_name, low_memory=False)

# Find remaining unknown dates
unknown_mask = df_top['release_date'].astype(str).str.lower() == 'unknown'
tracks_to_process = df_top[unknown_mask]

print(f"Found {len(tracks_to_process)} tracks still marked 'Unknown'.")
if len(tracks_to_process) == 0:
    print("All done! No unknowns left.")
    sys.exit()

print("Beginning Round 3: MusicBrainz Database (Paced at 1.2s per track)...")

def clean_name(text):
    """Clean up track names for better search results"""
    text = str(text)
    text = re.sub(r'\(.*?\)', '', text)  
    text = re.sub(r'\[.*?\]', '', text)  
    text = text.split(' - ')[0]          
    return text.strip()

updated_count = 0

# MusicBrainz requires a "User-Agent" so their servers know who is pinging them
headers = {
    "User-Agent": "CS506-SuperBowlProject/1.0 ( DataScience )"
}

for index, row in tracks_to_process.iterrows():
    track_id = row['track_id'] if 'track_id' in row else index
    clean_track = clean_name(row['track_name'])
    clean_artist = clean_name(row['artist_name'])
    
    if not clean_track:
        continue
        
    # Format the query specifically for MusicBrainz's search engine
    query = f'recording:"{clean_track}" AND artist:"{clean_artist}"'
    url = f"https://musicbrainz.org/ws/2/recording?query={urllib.parse.quote(query)}&fmt=json"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'recordings' in data and len(data['recordings']) > 0:
                # Look for the first-release-date field
                rec = data['recordings'][0]
                if 'first-release-date' in rec:
                    rdate = rec['first-release-date'][:10]
                    # Update the dataframe based on track_id if it exists, otherwise by index
                    if 'track_id' in df_top.columns:
                        df_top.loc[df_top['track_id'] == track_id, 'release_date'] = rdate
                    else:
                        df_top.loc[index, 'release_date'] = rdate
                    updated_count += 1
                    
    except Exception as e:
        pass # Ignore minor connection blips
        
    # CRUCIAL: Wait 1.2 seconds to perfectly obey MusicBrainz API rules
    time.sleep(1.2)
    
    # Print progress every 10 tracks
    current_progress = tracks_to_process.index.get_loc(index) + 1
    if current_progress % 10 == 0:
        print(f"   -> Searched {current_progress} tracks... (Found {updated_count} dates via MusicBrainz)")

print(f"\nRound 3 complete! Successfully retrieved {updated_count} additional dates.")

# Save the final progress
df_top.to_csv(file_name, index=False)
remaining = len(df_top[df_top['release_date'].astype(str).str.lower() == 'unknown'])
print(f"File saved! 'Unknown' dates remaining: {remaining}")