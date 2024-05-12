import os
import json
import logging
from confluent_kafka import Producer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Local file system base directory
base_directory = "/home/mohammad/Documents/GitHub/project-spotify/static/fma_small"

# Kafka details
bootstrap_servers = "localhost:9092"
topic = "streamed_music_local"

# Create a Kafka producer
producer = Producer({
    'bootstrap.servers': bootstrap_servers,
    'message.max.bytes': 11210584,
    'queue.buffering.max.messages': 100000
})

# Counter for messages
message_counter = 0

def get_songs_from_directory(directory):
    """Generator to read song files from a given directory"""
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.mp3'):  # Ensure we are reading only song files if needed
                filepath = os.path.join(directory, filename)
                with open(filepath, 'rb') as file:
                    yield file.read(), filename
    except FileNotFoundError:
        logging.error(f"Directory not found: {directory}")
        return
    except Exception as e:
        logging.error(f"Failed to read songs from directory {directory}: {e}")
        return

# Iterate over the virtual directories (000 to 155)
for i in range(156):
    directory = os.path.join(base_directory, f"{i:03d}")  # Adjust the path to the local directory structure
    # Send the song files in the directory to the Kafka topic
    for song_data, filename in get_songs_from_directory(directory):
        if song_data:
            track_id = int(filename.split('.')[0])  # Convert to integer
            track_info = {'track_id': track_id}
            producer.produce(topic, song_data, key=json.dumps(track_info))
            message_counter += 1
            logging.info(f"Sent song: {track_id}")
            if message_counter % 100 == 0:
                producer.flush()
        else:
            logging.warning(f"Empty or missing song in directory: {directory}")
    producer.flush()

logging.info(f"Total {message_counter} messages sent.")
