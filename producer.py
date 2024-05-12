import os
import logging
from confluent_kafka import Producer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Local file system base directory
base_directory = "/home/mohammad/Desktop/manal/fma_small"

# Kafka details
bootstrap_servers = "localhost:9092"
topic = "streamed_music_local"

# Create a Kafka producer
producer = Producer(
    {
        "bootstrap.servers": bootstrap_servers,
        "message.max.bytes": 11210584,
        "queue.buffering.max.messages": 100000,
    }
)

# Counter for messages
message_counter = 0


def get_songs_from_directory(directory):
    """Generator to read song files from a given directory"""
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".mp3"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "rb") as file:
                    yield file.read(), filename
    except FileNotFoundError:
        logging.error(f"Directory not found: {directory}")
        return
    except Exception as e:
        logging.error(f"Failed to read songs from directory {directory}: {e}")
        return


# Text file to write to for tracking present files in the selected dataset.
with open("files_in_dataset.txt", "w") as file:
    # Iterate over the directories in the dataset (000 to 155)
    for i in range(156):
        directory = os.path.join(base_directory, f"{i:03d}")
        # Send the blobs (songs) in the directory to the Kafka topic
        for song, filename in get_songs_from_directory(directory):
            if song is not None and len(song) > 0:
                producer.produce(topic, key=filename.encode(), value=song)
                message_counter += 1
                print(f"Sent song: {filename}")
                file.write(f"{filename}\n")
                if message_counter % 50 == 0:
                    producer.flush()
            else:
                logging.warning(f"Empty or missing song: {filename}")
        producer.flush()

    logging.info(f"Total {message_counter} messages sent.")