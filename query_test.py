from query import find_closest_tracks


track_id = str(210).zfill(6).encode().decode("utf-8") + ".mp3"
print(find_closest_tracks(track_id))
